//! Dry active nematic on S^2 (unit sphere).
//!
//! Runs the DEC solver on an icosahedral mesh with the Weitzenboeck
//! curvature correction K = 1 (unit sphere). Writes .npy snapshots
//! and a mesh.json for the visualisation pipeline.
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
use volterra_dec::QFieldDec;
use volterra_dec::DecDomain;
// run_dry_active_nematic_dec available for flat meshes; here we run the
// RK4 loop inline with the curvature correction callback.

fn main() {
    let refinement = 4; // 2562 vertices, 5120 faces
    let n_steps = 5000;
    let snap_every = 25;

    let out_dir = std::env::var("OUT_DIR")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{home}/.volterra-bench/viz/sphere")
        });
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("Building icosphere (refinement = {refinement})...");
    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  vertices: {nv}, faces: {nf}, chi: {}", mesh.euler_characteristic());

    println!("Assembling DEC operators...");
    let domain = DecDomain::new(mesh, Sphere::<3>).expect("DecDomain assembly failed");

    // Write mesh as JSON for the visualisation pipeline.
    let mesh_json = serde_json::json!({
        "vertices": domain.mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>(),
        "triangles": domain.mesh.simplices,
    });
    let mesh_path = out.join("mesh.json");
    std::fs::write(&mesh_path, serde_json::to_string(&mesh_json).unwrap())
        .expect("failed to write mesh.json");
    println!("  mesh written to {}", mesh_path.display());

    // Simulation parameters: active turbulent phase on S^2.
    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.0005; // small dt for stability on curved mesh
    params.zeta_eff = 0.5; // moderate activity
    params.k_r = 0.1; // Frank elastic constant
    params.gamma_r = 1.0;
    params.a_landau = -0.5;
    params.c_landau = 4.5;

    // Curvature correction for the unit sphere: K = 1.
    let curv_cb = constant_curvature_2d(1.0);

    // Random initial Q field.
    let q0 = QFieldDec::random_perturbation(nv, 0.01, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "sphere",
        "refinement": refinement,
        "n_vertices": nv,
        "n_faces": nf,
        "n_steps": n_steps,
        "snap_every": snap_every,
        "dt": params.dt,
        "zeta_eff": params.zeta_eff,
        "k_r": params.k_r,
        "gaussian_curvature": 1.0,
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("failed to write meta.json");

    println!("Running dry active nematic on S^2: {n_steps} steps, snap every {snap_every}...");
    let t0 = Instant::now();

    // Run with snapshot writing: we call the runner and also write snapshots manually.
    let mut q = q0.clone();
    let mut stats = Vec::new();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            let snap_path = out.join(format!("q_{step:06}.npy"));
            write_snapshot(&q, &snap_path).expect("failed to write snapshot");

            let s = q.mean_order_param();
            stats.push((step, s));
            if step % (snap_every * 5) == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                println!("  step {step:>6}/{n_steps}  <S> = {s:.4}  t = {elapsed:.1}s");
            }
        }
        if step < n_steps {
            // Single RK4 step (inline, matching the runner logic).
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
    let n_snaps = stats.len();
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {}", out.display());
}
