//! Dry active nematic on a torus (T^2 embedded in R^3).
//!
//! Runs the DEC solver on a torus mesh with variable Gaussian curvature.
//! K > 0 on the outer equator, K < 0 on the inner equator.
//! Defects are expected to migrate toward regions of maximum positive K.

use std::path::Path;
use std::time::Instant;

use volterra_core::ActiveNematicParams;
use volterra_dec::curvature_correction::variable_curvature_2d;
use volterra_dec::mesh_gen::{torus_mesh, torus_gaussian_curvature};
use volterra_dec::snapshot::write_snapshot;
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
    params.dt = 0.0005;
    params.zeta_eff = 0.3;
    params.k_r = 0.1;
    params.gamma_r = 1.0;
    params.a_landau = -0.5;
    params.c_landau = 4.5;

    let q0 = QFieldDec::random_perturbation(nv, 0.01, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "torus",
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

    println!("Running dry active nematic on torus: {n_steps} steps...");
    let t0 = Instant::now();

    let mut q = q0;
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
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = volterra_dec::molecular_field_dec(
                    qq, &params, &domain.ops, Some(&curv_cb),
                );
                h.scale(params.gamma_r)
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
