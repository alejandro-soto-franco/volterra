//! Ensemble topological entropy of a sphere run.
//!
//! A band is wrapped around an ensemble of tracers and carried by the run's own
//! velocity field. Its exponential growth rate bounds the flow's topological
//! entropy below. Unlike a braid word read off the four defects, this uses
//! every tracer, needs no projection chart, and returns a rate rather than the
//! entropy of a word over a window.
//!
//! The reading sits between two others that bracket it. The defect braid uses
//! four trajectories and must come in at or below this, and the stretching rate
//! from tracer pairs counts shear that no topology can see and must come in at
//! or above it.
//!
//!     sphere_etec <run-dir> [--tracers N] [--from FRAC]
//!
//! Writes `etec.json`.

use std::path::Path;

use volterra_braid::etec::{Band, Sphere, delaunay_sphere};
use volterra_dec::tracers::{Buckets, advect, read_npy};

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut run = String::new();
    let mut n_tracers = 200usize;
    let mut from = 0.2_f64;
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--tracers" => { i += 1; n_tracers = raw[i].parse().unwrap(); }
            "--from" => { i += 1; from = raw[i].parse().unwrap(); }
            other => run = other.to_string(),
        }
        i += 1;
    }
    if run.is_empty() {
        eprintln!("sphere_etec <run-dir> [--tracers N] [--from FRAC]");
        std::process::exit(2);
    }
    let run = Path::new(&run);

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("meta.json")).unwrap()).unwrap();
    let mesh: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("mesh.json")).unwrap()).unwrap();
    let verts: Vec<[f64; 3]> = mesh["vertices"].as_array().unwrap().iter()
        .map(|v| {
            let a = v.as_array().unwrap();
            [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
        })
        .collect();
    let tris_mesh: Vec<[usize; 3]> = mesh["triangles"].as_array().unwrap().iter()
        .map(|v| {
            let a = v.as_array().unwrap();
            [
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
                a[2].as_u64().unwrap() as usize,
            ]
        })
        .collect();
    let nv = verts.len();
    let dt_step = meta["dt"].as_f64().unwrap();
    let pe = meta["pe"].as_f64().unwrap();

    let mut vert_faces = vec![Vec::new(); nv];
    for (f, t) in tris_mesh.iter().enumerate() {
        for &v in t {
            vert_faces[v].push(f);
        }
    }
    let buckets = Buckets::new(&verts);

    let mut steps: Vec<usize> = std::fs::read_dir(run).unwrap()
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().to_string();
            n.strip_prefix("vel_")?.strip_suffix(".npy")?.parse::<usize>().ok()
        })
        .collect();
    steps.sort_unstable();
    let cut = (steps.len() as f64 * from) as usize;
    let steps = &steps[cut..];
    let dt_snap = (steps[1] - steps[0]) as f64 * dt_step;
    let t0 = steps[0] as f64 * dt_step;
    println!(
        "run {} at Pe = {pe}: {} snapshots from t = {t0:.0}",
        run.display(),
        steps.len()
    );

    // Tracers spread evenly over the sphere.
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let seeds: Vec<[f64; 3]> = (0..n_tracers)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n_tracers as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * k as f64;
            [r * th.cos(), r * th.sin(), z]
        })
        .collect();

    println!("  triangulating {n_tracers} tracers...");
    let tri0 = delaunay_sphere(&seeds);
    assert_eq!(
        tri0.len(),
        2 * n_tracers - 4,
        "the tracer triangulation did not close"
    );
    let mut band = Band::new(Sphere, seeds, tri0);
    // A band separating the two hemispheres, which is an essential curve and so
    // has somewhere to grow.
    let inside: Vec<bool> = band.points.iter().map(|p| p[2] > 0.0).collect();
    band.encircle(&inside);
    println!("  {} edges, {} triangles", band.n_edges(), band.n_tris());

    let sub = 4usize;
    let h = dt_snap / sub as f64;
    let mut curve: Vec<(f64, f64)> = Vec::new();
    let mut u0 = read_npy(&run.join(format!("vel_{:06}.npy", steps[0])), nv, 3).unwrap();
    let mut tangled = 0usize;

    for w in 1..steps.len() {
        let u1 = read_npy(&run.join(format!("vel_{:06}.npy", steps[w])), nv, 3).unwrap();
        for s in 0..sub {
            let f0 = s as f64 / sub as f64;
            let f1 = (s + 1) as f64 / sub as f64;
            let blend = |f: f64| -> Vec<f64> {
                (0..nv * 3).map(|i| (1.0 - f) * u0[i] + f * u1[i]).collect()
            };
            let ua = blend(f0);
            let ub = blend(f1);
            for p in band.points.iter_mut() {
                *p = advect(*p, h, &verts, &tris_mesh, &vert_faces, &buckets, &ua, &ub);
            }
            band.repair(12);
            band.accumulate();
        }
        let inv = band.inverted();
        if inv > 0 {
            tangled += 1;
        }
        let elapsed = steps[w] as f64 * dt_step - t0;
        if elapsed > 0.0 {
            curve.push((elapsed, band.log_growth / elapsed));
        }
        u0 = u1;
        if w % 200 == 0 {
            println!(
                "    t = {:.0}: rate {:.4e}, {} flips",
                elapsed,
                band.log_growth / elapsed,
                band.flips
            );
        }
    }

    let elapsed = curve.last().map(|c| c.0).unwrap_or(1.0);
    let rate = band.log_growth / elapsed;
    // A rate that has converged is flat across the second half of the record.
    let n = curve.len();
    let q3: f64 = curve[n / 2..3 * n / 4].iter().map(|c| c.1).sum::<f64>()
        / (3 * n / 4 - n / 2).max(1) as f64;
    let q4: f64 = curve[3 * n / 4..].iter().map(|c| c.1).sum::<f64>()
        / (n - 3 * n / 4).max(1) as f64;
    println!("  entropy rate {rate:.5e} per unit time over {elapsed:.0}");
    println!(
        "  third quarter {q3:.4e}, fourth {q4:.4e}, drift {:.1} per cent",
        100.0 * (q4 - q3).abs() / q4.abs().max(1e-300)
    );
    println!("  {} flips, {} steps with a turned-over face", band.flips, tangled);

    let out = serde_json::json!({
        "run": run.file_name().unwrap().to_string_lossy(),
        "pe": pe,
        "tracers": n_tracers,
        "elapsed": elapsed,
        "rate": rate,
        "drift": (q4 - q3).abs() / q4.abs().max(1e-300),
        "flips": band.flips,
        "tangled_steps": tangled,
        "convergence": curve,
    });
    std::fs::write(run.join("etec.json"), serde_json::to_string(&out).unwrap()).unwrap();
    println!("  wrote {}", run.join("etec.json").display());
}
