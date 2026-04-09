//! Wet active nematic on S^2 (unit sphere).
//!
//! Runs the full Beris-Edwards + Stokes DEC solver on an icosahedral mesh
//! with Weitzenboeck curvature correction K = 1 (unit sphere).
//! Writes .npy snapshots and mesh.json for the visualisation pipeline.
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_sphere

use std::path::Path;
use std::time::Instant;

use cartan_manifolds::sphere::Sphere;
use volterra_core::ActiveNematicParams;
use volterra_dec::curvature_correction::constant_curvature_2d;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::write_snapshot;
use volterra_dec::stokes_dec::StokesSolverDec;
use volterra_dec::QFieldDec;
use volterra_dec::DecDomain;

fn main() {
    let refinement = 4; // 2562 vertices, 5120 faces
    let n_steps = 20000;
    let snap_every = 100;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let out_dir = format!("{home}/.volterra-bench/viz/sphere");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("Building icosphere (refinement = {refinement})...");
    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  vertices: {nv}, faces: {nf}, chi: {}", mesh.euler_characteristic());

    println!("Assembling DEC operators...");
    let domain = DecDomain::new(mesh, Sphere::<3>).expect("DecDomain assembly failed");

    // Write mesh JSON for visualisation.
    let mesh_json = serde_json::json!({
        "vertices": domain.mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>(),
        "triangles": domain.mesh.simplices,
    });
    std::fs::write(out.join("mesh.json"), serde_json::to_string(&mesh_json).unwrap())
        .expect("failed to write mesh.json");

    // Wet active nematic parameters.
    // Activity drives flow, flow creates distortions, distortions nucleate defects.
    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.0001;     // small dt for CFL stability with flow
    params.zeta_eff = 0.3;  // extensile activity
    params.k_r = 0.04;      // Frank elastic constant
    params.gamma_r = 1.0;   // rotational viscosity
    params.eta = 5.0;       // high viscosity damps velocity, improves stability
    params.a_landau = -0.2;
    params.c_landau = 2.0;
    params.lambda = 0.7;    // flow-alignment parameter

    // Curvature correction for the unit sphere: K = 1.
    let curv_cb = constant_curvature_2d(1.0);

    // Pre-factorise the Stokes solver.
    println!("Factorising Stokes solver...");
    let stokes = StokesSolverDec::new(&domain.ops)
        .expect("Stokes solver factorisation failed");

    let mut q = QFieldDec::random_perturbation(nv, 0.05, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "sphere",
        "mode": "wet",
        "refinement": refinement,
        "n_vertices": nv,
        "n_faces": nf,
        "n_steps": n_steps,
        "snap_every": snap_every,
        "dt": params.dt,
        "zeta_eff": params.zeta_eff,
        "k_r": params.k_r,
        "eta": params.eta,
        "gaussian_curvature": 1.0,
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("failed to write meta.json");

    println!("Running WET active nematic on S^2: {n_steps} steps...");
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            write_snapshot(&q, &out.join(format!("q_{step:06}.npy")))
                .expect("failed to write snapshot");

            if step % (snap_every * 5) == 0 {
                let s = q.mean_order_param();
                let elapsed = t0.elapsed().as_secs_f64();
                println!("  step {step:>6}/{n_steps}  <S> = {s:.4}  t = {elapsed:.1}s");
            }
        }
        if step < n_steps {
            // 1. Stokes solve: get velocity from active stress.
            let vel = stokes.solve(&q, &params, &domain.ops, &domain.mesh);

            // 2. RK4 step on Q with molecular field + advection from velocity.
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = volterra_dec::molecular_field_dec(
                    qq, &params, &domain.ops, Some(&curv_cb),
                );
                // Molecular field + advection contribution.
                // Advection is handled approximately via the velocity magnitude
                // (full edge-based advection is in runner_dec_wet.rs).
                let mut dq = h.scale(params.gamma_r);

                // Advection: -u dot grad(Q) approximated per edge.
                let ne = domain.mesh.n_boundaries();
                for e in 0..ne {
                    let [v0, v1] = domain.mesh.boundaries[e];
                    let dq1 = qq.q1[v1] - qq.q1[v0];
                    let dq2 = qq.q2[v1] - qq.q2[v0];
                    let ux = 0.5 * (vel.vx[v0] + vel.vx[v1]);
                    let uy = 0.5 * (vel.vy[v0] + vel.vy[v1]);
                    let vmag = (ux * ux + uy * uy).sqrt();
                    let adv1 = 0.5 * vmag * dq1;
                    let adv2 = 0.5 * vmag * dq2;
                    let n_e0 = domain.mesh.vertex_boundaries[v0].len() as f64;
                    let n_e1 = domain.mesh.vertex_boundaries[v1].len() as f64;
                    if n_e0 > 0.0 {
                        dq.q1[v0] -= adv1 / n_e0;
                        dq.q2[v0] -= adv2 / n_e0;
                    }
                    if n_e1 > 0.0 {
                        dq.q1[v1] -= adv1 / n_e1;
                        dq.q2[v1] -= adv2 / n_e1;
                    }
                }
                dq
            };

            let k1 = rhs(&q);
            let q2 = q.add(&k1.scale(0.5 * params.dt));
            let k2 = rhs(&q2);
            let q3 = q.add(&k2.scale(0.5 * params.dt));
            let k3 = rhs(&q3);
            let q4 = q.add(&k3.scale(params.dt));
            let k4 = rhs(&q4);
            let update = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
            q = q.add(&update.scale(params.dt / 6.0));
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let n_snaps = (n_steps / snap_every) + 1;
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {out_dir}");
}
