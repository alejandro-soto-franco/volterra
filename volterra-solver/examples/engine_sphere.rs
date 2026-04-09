//! Active nematic on S^2 using the NematicEngine.
//!
//! Pe = 1: expect 4 +1/2 defects oscillating at tetrahedral positions.

use std::path::Path;
use std::time::Instant;

use cartan_manifolds::sphere::Sphere;
use volterra_core::NematicParams;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::write_snapshot;
use volterra_solver::NematicEngine;

fn main() {
    let refinement = 4; // 2562 vertices
    let n_steps = 50000;
    let snap_every = 250;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let out_dir = format!("{home}/.volterra-bench/viz/engine_sphere");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("create output dir");

    println!("Building icosphere (refinement {refinement})...");
    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  {nv} vertices, {nf} faces, chi = {}", mesh.euler_characteristic());

    // Write mesh JSON before consuming the mesh.
    let mesh_json = serde_json::json!({
        "vertices": mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect::<Vec<_>>(),
        "triangles": mesh.simplices,
    });
    std::fs::write(out.join("mesh.json"), serde_json::to_string(&mesh_json).unwrap())
        .expect("write mesh.json");

    // Unit sphere: K = 1 everywhere.
    let gaussian_k = vec![1.0; nv];

    // Pe = 1: gentle activity, 4 tetrahedral defects expected.
    let params = NematicParams::new(
        1.0,   // Pe
        1.0,   // Er
        1.0,   // La
        1.0,   // Lc
        0.7,   // lambda
    );
    println!("  Pe = {}, Er = {}, La = {}, Lc = {}", params.pe, params.er, params.la, params.lc);
    println!("  S_eq = {:.3}", params.s_eq());

    println!("Building engine...");
    let engine = NematicEngine::new(mesh, Sphere::<3>, params, gaussian_k)
        .expect("engine construction failed");
    println!("  dt = {:.6} (auto CFL)", engine.dt());

    let mut q = volterra_dec::QFieldDec::random_perturbation(nv, 0.3, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "sphere",
        "engine": "NematicEngine",
        "refinement": refinement,
        "n_vertices": nv,
        "n_faces": nf,
        "pe": params.pe,
        "er": params.er,
        "la": params.la,
        "lc": params.lc,
        "s_eq": params.s_eq(),
        "dt": engine.dt(),
        "n_steps": n_steps,
        "snap_every": snap_every,
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("write meta");

    println!("Running NematicEngine: {n_steps} steps, Pe = {}...", params.pe);
    let t0 = Instant::now();

    engine.run(&mut q, n_steps, snap_every, |step, q_snap, _vel, stats| {
        write_snapshot(q_snap, &out.join(format!("q_{step:06}.npy")))
            .expect("write snapshot");

        if step % (snap_every * 4) == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "  step {step:>6}/{n_steps}  <S> = {:.4}  v_rms = {:.4}  t = {elapsed:.1}s",
                stats.mean_s, stats.velocity_rms
            );
        }
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let n_snaps = n_steps / snap_every + 1;
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {out_dir}");
}
