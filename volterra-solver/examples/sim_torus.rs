//! Wet active nematic on a torus (T^2 embedded in R^3).
//!
//! Full Beris-Edwards + Stokes on a torus mesh with variable Gaussian
//! curvature. K > 0 on the outer equator, K < 0 on the inner equator.

use std::path::Path;
use std::time::Instant;

use volterra_core::ActiveNematicParams;
use volterra_dec::curvature_correction::variable_curvature_2d;
use volterra_dec::mesh_gen::{torus_mesh, torus_gaussian_curvature};
use volterra_dec::snapshot::write_snapshot;
use volterra_dec::stokes_dec::StokesSolverDec;
use volterra_dec::QFieldDec;
use volterra_dec::DecDomain;

fn main() {
    let major_r = 3.0;
    let minor_r = 1.0;
    let n_major = 80;
    let n_minor = 40;
    let n_steps = 5000;
    let snap_every = 25;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let out_dir = format!("{home}/.volterra-bench/viz/torus");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("Building torus mesh (R={major_r}, r={minor_r}, {n_major}x{n_minor})...");
    let mesh = torus_mesh(major_r, minor_r, n_major, n_minor);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  vertices: {nv}, faces: {nf}, chi: {}", mesh.euler_characteristic());

    println!("Assembling DEC operators...");
    let domain = DecDomain::new(mesh, cartan_manifolds::euclidean::Euclidean::<3>)
        .expect("DecDomain assembly failed");

    // Write mesh JSON.
    let mesh_json = serde_json::json!({
        "vertices": domain.mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>(),
        "triangles": domain.mesh.simplices,
    });
    std::fs::write(out.join("mesh.json"), serde_json::to_string(&mesh_json).unwrap())
        .expect("failed to write mesh.json");

    // Curvature correction: variable K across the torus.
    let curvatures = torus_gaussian_curvature(major_r, minor_r, n_major, n_minor);
    let k_max = curvatures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let k_min = curvatures.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  K range: [{k_min:.4}, {k_max:.4}]");
    let curv_cb = variable_curvature_2d(curvatures.clone());

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.001;
    params.zeta_eff = 0.5;
    params.k_r = 0.04;
    params.gamma_r = 1.0;
    params.eta = 1.0;
    params.a_landau = -0.2;
    params.c_landau = 2.0;
    params.lambda = 0.7;

    // Pre-factorise Stokes solver.
    println!("Factorising Stokes solver...");
    let stokes = StokesSolverDec::new(&domain.ops)
        .expect("Stokes solver factorisation failed");

    let mut q = QFieldDec::random_perturbation(nv, 0.3, 42);

    let meta = serde_json::json!({
        "geometry": "torus",
        "mode": "wet",
        "major_radius": major_r,
        "minor_radius": minor_r,
        "n_major": n_major,
        "n_minor": n_minor,
        "n_vertices": nv,
        "n_faces": nf,
        "n_steps": n_steps,
        "snap_every": snap_every,
        "dt": params.dt,
        "zeta_eff": params.zeta_eff,
        "k_r": params.k_r,
        "K_range": [k_min, k_max],
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("failed to write meta.json");

    println!("Running WET active nematic on torus: {n_steps} steps...");
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
            // 1. Stokes solve.
            let vel = stokes.solve(&q, &params, &domain.ops, &domain.mesh);

            // 2. RK4 with molecular field + advection.
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = volterra_dec::molecular_field_dec(
                    qq, &params, &domain.ops, Some(&curv_cb),
                );
                let mut dq = h.scale(params.gamma_r);

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
    println!("Done in {elapsed:.1}s");
    println!("Output: {out_dir}");
}
