//! Wet active nematic on a 2D periodic box (Cartesian FFT Stokes).
//!
//! Uses the existing run_active_nematic_hydro which has the full spectral
//! Stokes solver. Writes .npy snapshots of the 2D Q-tensor field for the
//! visualisation pipeline.

use std::path::Path;
use std::time::Instant;

use volterra_core::ActiveNematicParams;
use volterra_fields::QField2D;
use volterra_solver::run_active_nematic_hydro;

fn main() {
    let nx = 128;
    let ny = 128;
    let n_steps = 20000;
    let snap_every = 100;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let out_dir = format!("{home}/.volterra-bench/viz/wet2d");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output dir");

    let mut params = ActiveNematicParams::default_test();
    params.nx = nx;
    params.ny = ny;
    params.dx = 1.0;
    params.dt = 0.005;
    params.zeta_eff = 3.0;   // strongly active
    params.k_r = 1.0;
    params.gamma_r = 1.0;
    params.eta = 1.0;
    params.a_landau = -0.5;
    params.c_landau = 4.5;
    params.lambda = 0.7;
    params.noise_amp = 0.02;

    let q0 = QField2D::random_perturbation(nx, ny, params.dx, 0.01, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "periodic_2d",
        "mode": "wet",
        "nx": nx, "ny": ny,
        "dx": params.dx, "dt": params.dt,
        "zeta_eff": params.zeta_eff,
        "k_r": params.k_r,
        "n_steps": n_steps,
        "snap_every": snap_every,
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap())
        .expect("failed to write meta.json");

    println!("Running WET active nematic 2D: {nx}x{ny}, {n_steps} steps, zeta={}", params.zeta_eff);
    let t0 = Instant::now();

    let (q_fin, stats) = run_active_nematic_hydro(&q0, &params, n_steps, snap_every);

    let elapsed = t0.elapsed().as_secs_f64();
    println!("Simulation done in {elapsed:.1}s, {} snapshots", stats.len());

    // Write snapshots: extract from the stats (the runner only returns final Q + stats).
    // We need to re-run with snapshot writing. Instead, write the final Q and
    // run a second pass where we save snapshots.
    // Actually, the existing runner doesn't write to disk. Let me run it step-by-step.
    println!("Re-running with snapshot export...");

    let mut q = q0.clone();
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            // Write Q as (nx*ny, 2) npy.
            let n = nx * ny;
            let mut data = vec![0.0f64; n * 2];
            for k in 0..n {
                data[k * 2] = q.q[k][0];
                data[k * 2 + 1] = q.q[k][1];
            }

            let snap_path = out.join(format!("q_{step:06}.npy"));
            write_npy_2d(&snap_path, &data, nx, ny).expect("failed to write snapshot");

            if step % (snap_every * 10) == 0 {
                let s = q.mean_order_param();
                let elapsed = t0.elapsed().as_secs_f64();
                println!("  step {step:>6}/{n_steps}  <S> = {s:.4}  t = {elapsed:.1}s");
            }
        }
        if step < n_steps {
            // Euler step with Stokes flow (matching run_active_nematic_hydro internals).
            let v = volterra_solver::stokes_solve(&q, &params);
            let dq = volterra_solver::beris_edwards_rhs(&q, Some(&v), &params);
            q = q.add(&dq.scale(params.dt));

            // Langevin noise.
            if params.noise_amp > 0.0 {
                use rand::rngs::SmallRng;
                use rand::{Rng, SeedableRng};
                let noise_scale = params.noise_amp * params.dt.sqrt();
                let seed = (params.nx as u64).wrapping_mul(6364136223846793005)
                    ^ (params.ny as u64).wrapping_mul(1442695040888963407)
                    ^ step as u64;
                let mut rng = SmallRng::seed_from_u64(seed);
                for [q1, q2] in q.q.iter_mut() {
                    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let u2: f64 = rng.random::<f64>();
                    let mag = noise_scale * (-2.0 * u1.ln()).sqrt();
                    let angle = std::f64::consts::TAU * u2;
                    *q1 += mag * angle.cos();
                    *q2 += mag * angle.sin();
                }
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let n_snaps = n_steps / snap_every + 1;
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {out_dir}");
    println!("\nRender with:");
    println!("  python tools/viz/render_2d.py {out_dir} --nx {nx} --ny {ny} --video output/videos/wet2d.mp4 --fps 30");
}

fn write_npy_2d(path: &Path, data: &[f64], nx: usize, ny: usize) -> std::io::Result<()> {
    use std::io::Write;
    let n = nx * ny;
    let shape_str = format!("({}, 2)", n);
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    let header_content_len = dict.len() + 1;
    let total_prefix = 10 + header_content_len;
    let padding = (64 - (total_prefix % 64)) % 64;
    let header_len = (header_content_len + padding) as u16;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    file.write_all(&[1, 0])?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(dict.as_bytes())?;
    for _ in 0..padding { file.write_all(b" ")?; }
    file.write_all(b"\n")?;
    for &v in data {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
