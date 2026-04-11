//! Passive nematic on a deforming sphere (Milestone 2 demo).
//!
//! Demonstrates the coupled nematic + shape evolution:
//! 1. Shape evolves under Helfrich bending + surface tension
//! 2. Q-tensor evolves via molecular field on the moving mesh
//! 3. v_n correction couples normal velocity to Q evolution
//!
//! The sphere starts at radius 1.0 and deforms under bending energy.
//! Nematic order relaxes simultaneously, with connection and operators
//! rebuilt after each shape step.
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_deforming_sphere

use std::path::Path;
use std::time::Instant;

use cartan_core::fiber::{Section, U1Spin2, VecSection};
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_manifolds::sphere::Sphere;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::write_snapshot;
use volterra_dec::QFieldDec;
use volterra_dec::EvolvingDomain;

fn main() {
    let refinement = 3; // 642 vertices (fast for demo)
    let n_steps = 2000;
    let snap_every = 50;
    let dt = 0.001;

    let out_dir = "output/deforming_sphere";
    let out = Path::new(out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("=== Passive Nematic on Deforming Sphere ===");
    println!("  refinement = {refinement}");
    println!("  dt = {dt}");
    println!("  n_steps = {n_steps}");

    // Build icosphere on Sphere<3>, then convert to Euclidean<3> for evolving surface.
    // The Sphere manifold's metric (geodesic distances) is only correct on the unit sphere;
    // once the mesh deforms, we need the flat R^3 metric.
    let sphere_mesh = icosphere(refinement);
    let nv = sphere_mesh.n_vertices();
    println!("  vertices: {nv}");

    let verts_euc: Vec<nalgebra::SVector<f64, 3>> = sphere_mesh.vertices.iter()
        .map(|v| nalgebra::SVector::<f64, 3>::new(v[0], v[1], v[2]))
        .collect();
    let euc_mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(
        &Euclidean::<3>, verts_euc, sphere_mesh.simplices.clone(),
    );
    let mut ed = EvolvingDomain::new(euc_mesh, Euclidean::<3>).unwrap();
    ed.recompute_curvatures();

    // Write mesh.
    let mesh_json = serde_json::json!({
        "vertices": ed.domain.mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>(),
        "triangles": ed.domain.mesh.simplices,
    });
    std::fs::write(out.join("mesh.json"), serde_json::to_string(&mesh_json).unwrap())
        .expect("failed to write mesh.json");

    // Shape parameters (slow shape evolution relative to nematic relaxation).
    let kb = 0.1;      // bending rigidity
    let tension = 0.01; // gentle surface tension
    let eta_s = 100.0;  // high surface viscosity -> slow shape evolution
    let h0 = vec![0.0; nv]; // zero spontaneous curvature

    // Nematic parameters.
    let k_frank = 0.01;
    let a_eff = -1.0;
    let c_landau = 1.0;
    let gamma_r = 1.0;

    // Initial Q: random perturbation.
    let mut q = QFieldDec::random_perturbation(nv, 0.5, 42);

    let meta = serde_json::json!({
        "geometry": "deforming_sphere",
        "mode": "passive_coupled",
        "refinement": refinement,
        "n_vertices": nv,
        "n_steps": n_steps,
        "dt": dt,
        "kb": kb,
        "tension": tension,
        "eta_surface": eta_s,
        "k_frank": k_frank,
        "a_eff": a_eff,
        "c_landau": c_landau,
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("failed to write meta.json");

    println!("Running coupled shape + nematic evolution...");
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            write_snapshot(&q, &out.join(format!("q_{step:06}.npy")))
                .expect("failed to write snapshot");

            // Also write current mesh (it evolves).
            let mesh_json = serde_json::json!({
                "vertices": ed.domain.mesh.vertices.iter()
                    .map(|v| [v[0], v[1], v[2]])
                    .collect::<Vec<_>>(),
                "triangles": ed.domain.mesh.simplices,
            });
            std::fs::write(
                out.join(format!("mesh_{step:06}.json")),
                serde_json::to_string(&mesh_json).unwrap(),
            ).ok();
        }

        if step % (snap_every * 5) == 0 {
            let s = q.mean_order_param();
            let mean_r: f64 = ed.domain.mesh.vertices.iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .sum::<f64>() / nv as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "  step {step:>5}/{n_steps}  <S>={s:.4}  <R>={mean_r:.4}  wall={elapsed:.1}s"
            );
        }

        if step < n_steps {
            // 1. Shape evolution: compute normal velocity and displace.
            let v_n = ed.shape_velocity(kb, &h0, tension, eta_s);

            // Get current normals for displacement.
            let normals: Vec<[f64; 3]> = ed.domain.vertex_normals.clone();

            // Apply normal displacement and rebuild operators.
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

            // 2. Q-tensor evolution: molecular field + v_n correction.
            // Molecular field using CovLaplacian from the updated connection.
            let lap = ed.cov_lap.apply::<U1Spin2, 2, _>(
                &VecSection::<U1Spin2>::from_vec(
                    q.q1.iter().zip(&q.q2).map(|(&a, &b)| [a, b]).collect()
                ),
                &ed.transport,
            );

            let tr_q2 = q.trace_q_squared();
            let bulk_linear = -a_eff;

            // v_n correction: dQ += v_n * 2H * Q.
            let (vn_dq1, vn_dq2) = ed.vn_correction(&v_n, &q.q1, &q.q2);

            for i in 0..nv {
                let tr = tr_q2[i];
                let bulk = bulk_linear - 2.0 * c_landau * tr;
                // H = -K * lap + bulk * Q (note: -K because DEC lap is positive-semidef)
                let h_q1 = -k_frank * lap.at(i)[0] + bulk * q.q1[i];
                let h_q2 = -k_frank * lap.at(i)[1] + bulk * q.q2[i];

                // dQ/dt = gamma_r * H + v_n correction
                q.q1[i] += dt * (gamma_r * h_q1 + vn_dq1[i]);
                q.q2[i] += dt * (gamma_r * h_q2 + vn_dq2[i]);
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    println!("\nDone: {} snapshots in {elapsed:.1}s", n_steps / snap_every + 1);
    println!("Output: {out_dir}");
}
