//! Active nematic on a deforming sphere (Milestone 3 demo).
//!
//! Full coupled system:
//! 1. Active stress drives both tangential flow (Stokes) and normal force (shape)
//! 2. Shape evolves under Helfrich + active normal stress
//! 3. Flow advects the Q-tensor
//! 4. Q evolves with molecular field + advection + v_n correction
//!
//! Starts from a slightly oblate ellipsoid to break spherical symmetry
//! (active normal stress vanishes on a perfect sphere).
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_active_deforming

use std::path::Path;
use std::time::Instant;

use cartan_core::fiber::{Section, U1Spin2, VecSection};
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use volterra_dec::connection_laplacian::ConnectionLaplacian;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::write_snapshot;
use volterra_dec::stokes_dec::StokesSolverDec;
use volterra_dec::QFieldDec;
use volterra_dec::EvolvingDomain;

fn main() {
    let refinement = 3; // 642 vertices
    let n_steps = 5000;
    let snap_every = 100;
    let dt = 0.0005;

    let out_dir = "output/active_deforming";
    let out = Path::new(out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("=== Active Nematic on Deforming Sphere ===");

    // Build icosphere, convert to Euclidean<3>, then perturb to oblate ellipsoid.
    let sphere_mesh = icosphere(refinement);
    let nv = sphere_mesh.n_vertices();
    let oblate_factor = 0.9; // compress z-axis by 10%

    let verts_euc: Vec<nalgebra::SVector<f64, 3>> = sphere_mesh.vertices.iter()
        .map(|v| nalgebra::SVector::<f64, 3>::new(v[0], v[1], v[2] * oblate_factor))
        .collect();
    let euc_mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(
        &Euclidean::<3>, verts_euc, sphere_mesh.simplices.clone(),
    );

    let mut ed = EvolvingDomain::new(euc_mesh, Euclidean::<3>).unwrap();
    ed.recompute_curvatures();
    println!("  vertices: {nv}");
    println!("  oblate factor: {oblate_factor}");

    // Parameters.
    let kb = 0.05;       // bending rigidity
    let tension = 0.0;   // no surface tension
    let eta_s = 50.0;    // surface viscosity
    let zeta = 0.5;      // activity (extensile)
    let k_frank = 0.01;  // Frank elastic constant
    let a_eff = -1.0;    // Landau parameter
    let c_landau = 1.0;  // Landau quartic
    let gamma_r = 1.0;   // rotational viscosity
    let eta_fluid = 1.0; // fluid viscosity (for Stokes)
    let h0 = vec![0.0; nv];

    // Initial Q: random.
    let mut q = QFieldDec::random_perturbation(nv, 0.5, 42);

    // Write initial mesh and metadata.
    let write_mesh = |ed: &EvolvingDomain<Euclidean<3>>, path: &Path| {
        let mesh_json = serde_json::json!({
            "vertices": ed.domain.mesh.vertices.iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>(),
            "triangles": ed.domain.mesh.simplices,
        });
        std::fs::write(path, serde_json::to_string(&mesh_json).unwrap()).ok();
    };
    write_mesh(&ed, &out.join("mesh.json"));

    let meta = serde_json::json!({
        "geometry": "oblate_sphere",
        "mode": "active_coupled",
        "refinement": refinement,
        "n_vertices": nv,
        "n_steps": n_steps,
        "dt": dt,
        "kb": kb,
        "zeta": zeta,
        "eta_surface": eta_s,
        "k_frank": k_frank,
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta).ok();

    println!("Running full active coupled evolution (Pe={:.0})...", zeta / k_frank);
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            write_snapshot(&q, &out.join(format!("q_{step:06}.npy"))).ok();
            write_mesh(&ed, &out.join(format!("mesh_{step:06}.json")));
        }

        if step % (snap_every * 5) == 0 {
            let s = q.mean_order_param();
            let mean_r: f64 = ed.domain.mesh.vertices.iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .sum::<f64>() / nv as f64;
            let mean_z: f64 = ed.domain.mesh.vertices.iter()
                .map(|v| v[2].abs()).sum::<f64>() / nv as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "  step {step:>5}/{n_steps}  <S>={s:.4}  <R>={mean_r:.4}  <|z|>={mean_z:.4}  wall={elapsed:.1}s"
            );
        }

        if step < n_steps {
            // 1. Shape evolution: Helfrich + active normal stress -> v_n.
            let v_n = ed.shape_velocity_active(
                kb, &h0, tension, eta_s, zeta, &q.q1, &q.q2,
            );

            // Move mesh.
            let normals: Vec<[f64; 3]> = ed.domain.vertex_normals.clone();
            let new_positions: Vec<nalgebra::SVector<f64, 3>> = (0..nv)
                .map(|i| {
                    let p = ed.domain.mesh.vertices[i];
                    nalgebra::SVector::<f64, 3>::new(
                        p[0] + v_n[i] * normals[i][0] * dt,
                        p[1] + v_n[i] * normals[i][1] * dt,
                        p[2] + v_n[i] * normals[i][2] * dt,
                    )
                })
                .collect();
            ed.deform(&new_positions).unwrap();
            ed.recompute_curvatures();

            // 2. Stokes solve on the deformed mesh (rebuild each step).
            let stokes = match StokesSolverDec::new(&ed.domain.ops, &ed.domain.mesh) {
                Ok(s) => s,
                Err(_) => continue, // skip if factorisation fails
            };

            // Set up Stokes-compatible params.
            let mut stokes_params = volterra_core::ActiveNematicParams::default_test();
            stokes_params.zeta_eff = zeta;
            stokes_params.eta = eta_fluid;
            let vel = stokes.solve(&q, &stokes_params, &ed.domain.ops, &ed.domain.mesh);

            // 3. Q-tensor evolution: molecular field + advection + v_n correction.
            let section = VecSection::<U1Spin2>::from_vec(
                q.q1.iter().zip(&q.q2).map(|(&a, &b)| [a, b]).collect()
            );
            let lap = ed.cov_lap.apply::<U1Spin2, 2, _>(&section, &ed.transport);

            let edge_phases: Vec<f64> = (0..ed.transport.transports.len())
                .map(|e| {
                    let t = &ed.transport.transports[e];
                    t[2].atan2(t[0]) // sin(alpha).atan2(cos(alpha)) = alpha
                })
                .collect();

            let adv = volterra_dec::stokes_dec::advect_q_covariant(
                &q, &vel,
                &ed.domain.mesh.boundaries,
                &ed.domain.mesh.vertex_boundaries,
                &ed.domain.mesh.vertices.iter()
                    .map(|v| [v[0], v[1], v[2]]).collect::<Vec<_>>(),
                &edge_phases,
            );

            let (vn_dq1, vn_dq2) = ed.vn_correction(&v_n, &q.q1, &q.q2);
            let tr_q2 = q.trace_q_squared();
            let bulk_linear = -a_eff;

            for i in 0..nv {
                let tr = tr_q2[i];
                let bulk = bulk_linear - 2.0 * c_landau * tr;
                let h_q1 = -k_frank * lap.at(i)[0] + bulk * q.q1[i];
                let h_q2 = -k_frank * lap.at(i)[1] + bulk * q.q2[i];

                q.q1[i] += dt * (gamma_r * h_q1 - adv.q1[i] + vn_dq1[i]);
                q.q2[i] += dt * (gamma_r * h_q2 - adv.q2[i] + vn_dq2[i]);
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nDone: {} snapshots in {elapsed:.1}s", n_steps / snap_every + 1);
    println!("Output: {out_dir}");
}
