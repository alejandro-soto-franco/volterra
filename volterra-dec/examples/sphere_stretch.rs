//! Stretching rate of the flow on a sphere, by tracer pairs.
//!
//! Topological entropy of a defect braid is a proxy for how much a flow
//! stretches material. On four strands over one period it is close to binary,
//! and it depends on a projection axis and a window that the flow knows nothing
//! about. The stretching rate is the quantity itself: how fast two nearby
//! parcels of fluid separate, in inverse time.
//!
//! # Method
//!
//! The Benettin algorithm. Each seed is a pair of tracers a distance `delta`
//! apart, advected by the run's own velocity field. Separation grows
//! exponentially and would leave the linear regime within a few hundred time
//! units, so at every renormalisation the growth factor is accumulated and the
//! companion is pulled back along its geodesic to `delta` again. The rate is
//! the accumulated log growth over the elapsed time.
//!
//! Pulling back along the geodesic keeps the direction the flow has selected,
//! which is what makes the accumulated rate the leading Lyapunov exponent
//! rather than an average over directions.
//!
//! # What is reported
//!
//! The mean rate over seeds spread evenly across the sphere, its spread, and
//! the rate as a function of elapsed time so its convergence can be seen. A
//! Lyapunov exponent that has converged is flat in the elapsed time; one that
//! is still climbing has not been watched long enough.
//!
//!     sphere_stretch <run-dir> [--seeds N] [--delta D] [--from FRAC]
//!
//! Writes `stretch.json`.

use std::path::Path;

use volterra_dec::tracers::{advect, geodesic, norm3, cross3, dot3, read_npy, Buckets, MeshRef};

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut run = String::new();
    let mut seeds = 512usize;
    let mut delta = 1e-3_f64;
    let mut from = 0.2_f64;
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--seeds" => { i += 1; seeds = raw[i].parse().unwrap(); }
            "--delta" => { i += 1; delta = raw[i].parse().unwrap(); }
            "--from" => { i += 1; from = raw[i].parse().unwrap(); }
            other => run = other.to_string(),
        }
        i += 1;
    }
    if run.is_empty() {
        eprintln!("sphere_stretch <run-dir> [--seeds N] [--delta D] [--from FRAC]");
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
    let tris: Vec<[usize; 3]> = mesh["triangles"].as_array().unwrap().iter()
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
    for (f, t) in tris.iter().enumerate() {
        for &v in t {
            vert_faces[v].push(f);
        }
    }
    let buckets = Buckets::new(&verts);
    let mesh = MeshRef { verts: &verts, tris: &tris, vert_faces: &vert_faces, buckets: &buckets };

    // The velocity snapshots, in order, skipping the developing part of the run.
    let mut steps: Vec<usize> = std::fs::read_dir(run).unwrap()
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().to_string();
            n.strip_prefix("vel_")?.strip_suffix(".npy")?.parse::<usize>().ok()
        })
        .collect();
    steps.sort_unstable();
    let cut = (steps.len() as f64 * from) as usize;
    let steps = &steps[cut..];
    if steps.len() < 8 {
        eprintln!("only {} velocity snapshots after the cut", steps.len());
        std::process::exit(1);
    }
    let dt_snap = (steps[1] - steps[0]) as f64 * dt_step;
    println!(
        "run {} at Pe = {pe}: {} snapshots from t = {:.0}, spacing {dt_snap}",
        run.display(),
        steps.len(),
        steps[0] as f64 * dt_step
    );

    // Seeds spread evenly, each a pair a distance `delta` apart.
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut a: Vec<[f64; 3]> = Vec::with_capacity(seeds);
    let mut b: Vec<[f64; 3]> = Vec::with_capacity(seeds);
    for k in 0..seeds {
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / seeds as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let th = ga * k as f64;
        let p = [r * th.cos(), r * th.sin(), z];
        // A companion offset along an arbitrary tangent direction.
        let t = norm3(cross3(p, [0.0, 0.0, 1.0]));
        let t = if dot3(t, t) < 0.5 { norm3(cross3(p, [1.0, 0.0, 0.0])) } else { t };
        a.push(p);
        b.push(norm3([
            p[0] + delta * t[0],
            p[1] + delta * t[1],
            p[2] + delta * t[2],
        ]));
    }
    let mut acc = vec![0.0_f64; seeds];

    // Substeps per snapshot interval, so a step moves a tracer well inside one
    // mesh cell at the speeds these runs reach.
    let sub = 4usize;
    let h = dt_snap / sub as f64;
    let mut curve: Vec<(f64, f64)> = Vec::new();
    let t0 = steps[0] as f64 * dt_step;

    let u0 = read_npy(&run.join(format!("vel_{:06}.npy", steps[0])), nv, 3).unwrap();
    for w in 1..steps.len() {
        let u1 = read_npy(&run.join(format!("vel_{:06}.npy", steps[w])), nv, 3).unwrap();
        for s in 0..sub {
            let f0 = s as f64 / sub as f64;
            let f1 = (s + 1) as f64 / sub as f64;
            // The two ends of this substep, blended from the bracketing snapshots.
            let blend = |f: f64| -> Vec<f64> {
                (0..nv * 3).map(|i| (1.0 - f) * u0[i] + f * u1[i]).collect()
            };
            let ua = blend(f0);
            let ub = blend(f1);
            for k in 0..seeds {
                a[k] = advect(a[k], h, &mesh, &ua, &ub);
                b[k] = advect(b[k], h, &mesh, &ua, &ub);
            }
        }
        // Renormalise: accumulate the growth and pull the companion back to
        // `delta` along the geodesic the flow has selected.
        for k in 0..seeds {
            let d = geodesic(a[k], b[k]);
            if d > 1e-15 {
                acc[k] += (d / delta).ln();
                let axis = norm3(cross3(a[k], b[k]));
                let perp = cross3(axis, a[k]);
                b[k] = norm3([
                    a[k][0] * delta.cos() + perp[0] * delta.sin(),
                    a[k][1] * delta.cos() + perp[1] * delta.sin(),
                    a[k][2] * delta.cos() + perp[2] * delta.sin(),
                ]);
            }
        }
        let elapsed = steps[w] as f64 * dt_step - t0;
        if elapsed > 0.0 {
            let mean = acc.iter().sum::<f64>() / seeds as f64 / elapsed;
            curve.push((elapsed, mean));
        }
    }

    let elapsed = *curve.last().map(|c| &c.0).unwrap_or(&1.0);
    let rates: Vec<f64> = acc.iter().map(|v| v / elapsed).collect();
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let sd = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    let mut sorted = rates.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("  stretching rate {mean:.5e} +/- {sd:.2e} per unit time over {seeds} seeds");
    println!(
        "  quartiles {:.3e} / {:.3e} / {:.3e}, elapsed {elapsed:.0}",
        sorted[seeds / 4],
        sorted[seeds / 2],
        sorted[3 * seeds / 4]
    );
    // The e-folding time is the rate read as a time, which is the number a
    // reader can hold beside the orbit period.
    if mean > 0.0 {
        println!("  e-folding time {:.1}", 1.0 / mean);
    }

    let out = serde_json::json!({
        "run": run.file_name().unwrap().to_string_lossy(),
        "pe": pe,
        "seeds": seeds,
        "delta": delta,
        "elapsed": elapsed,
        "rate_mean": mean,
        "rate_sd": sd,
        "rate_median": sorted[seeds / 2],
        "rates": rates,
        "seed_points": a,
        "convergence": curve,
    });
    std::fs::write(run.join("stretch.json"), serde_json::to_string(&out).unwrap()).unwrap();
    println!("  wrote {}", run.join("stretch.json").display());
}
