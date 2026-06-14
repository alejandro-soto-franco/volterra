//! Wet active nematic on a 2D periodic box (Cartesian FFT Stokes).
//!
//! Living-documentation demo of the shared runner core: it builds the
//! `Cartesian2DWet` `PhysicsStep` (the same physics the library's
//! `run_active_nematic_hydro` drives) and runs it through
//! `SimulationRunner`, with an `Observer` that streams `.npy` snapshots of
//! the 2D Q-tensor field for the visualisation pipeline.
//!
//! This example previously hand-rolled its own time-stepping loop with a
//! forked, per-step-seeded Box-Muller noise and a local `.npy` writer. Both
//! are now gone: noise comes from `LangevinNoise` (canonical per-run seed,
//! the same stream `Cartesian2DWet` uses) and snapshots from
//! `volterra_core::sim::snapshot::write_npy`.
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_wet_2d

use std::path::Path;
use std::time::Instant;

use volterra_core::sim::noise::LangevinNoise;
use volterra_core::sim::snapshot::write_npy;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::{Observer, RunConfig, SimulationRunner};
use volterra_core::ActiveNematicParams;
use volterra_fields::QField2D;
use volterra_solver::sim_impls::cartesian2d::Cartesian2DWet;

/// Observer that streams Q-tensor snapshots to `.npy` on the snapshot cadence.
struct SnapshotSink<'a> {
    out: &'a Path,
    nx: usize,
    ny: usize,
    snap_every: usize,
    n_steps: usize,
    t0: Instant,
}

impl Observer<QField2D> for SnapshotSink<'_> {
    fn observe(&mut self, step: usize, _t: f64, q: &QField2D, _stats: &StepStats) {
        // Write Q as a (nx, ny, 1, 2) buffer: two components per cell, C-order.
        let n = self.nx * self.ny;
        let mut data = vec![0.0f64; n * 2];
        for k in 0..n {
            data[k * 2] = q.q[k][0];
            data[k * 2 + 1] = q.q[k][1];
        }
        let snap_path = self.out.join(format!("q_{step:06}.npy"));
        write_npy(&snap_path, &data, self.nx, self.ny, 1, 2)
            .expect("failed to write snapshot");

        if step % (self.snap_every * 10) == 0 {
            let s = q.mean_order_param();
            let elapsed = self.t0.elapsed().as_secs_f64();
            let n_steps = self.n_steps;
            println!("  step {step:>6}/{n_steps}  <S> = {s:.4}  t = {elapsed:.1}s");
        }
    }
}

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

    // Shared runner core: Cartesian2DWet physics + canonical per-run Langevin
    // noise (the same stream the library's run_active_nematic_hydro uses),
    // driven by SimulationRunner with a snapshot-writing Observer.
    let use_noise = params.noise_amp > 0.0;
    let mut physics = Cartesian2DWet {
        params: params.clone(),
        noise: use_noise.then(|| LangevinNoise::per_run_seed(nx, ny, n_steps)),
    };
    let mut q = q0.clone();
    let mut sink = SnapshotSink {
        out,
        nx,
        ny,
        snap_every,
        n_steps,
        t0,
    };
    let runner = SimulationRunner {
        config: RunConfig { steps: n_steps, snap_every, dt: params.dt, seed: 0, snap_final: false },
    };
    runner.run(&mut q, &mut physics, &mut sink);

    let elapsed = t0.elapsed().as_secs_f64();
    let n_snaps = n_steps / snap_every + 1;
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {out_dir}");
    println!("\nRender with:");
    println!("  python tools/viz/render_2d.py {out_dir} --nx {nx} --ny {ny} --video output/videos/wet2d.mp4 --fps 30");
}
