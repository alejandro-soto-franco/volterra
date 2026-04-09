//! Thread scaling benchmark: wall-clock vs RAYON_NUM_THREADS at N=100.
//!
//! Run with: RAYON_NUM_THREADS=K cargo run --release -p volterra-solver --example bench_scaling

use std::time::Instant;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::run_dry_active_nematic_3d;

fn main() {
    let n = 100;
    let n_steps = 100;

    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.001;

    let q0 = QField3D::random_perturbation(n, n, n, p.dx, 0.01, 42);
    let sites = n * n * n;
    let out = std::env::temp_dir().join("volterra_bench_scaling");
    std::fs::create_dir_all(&out).ok();

    let threads = rayon::current_num_threads();

    let t0 = Instant::now();
    let (_q_fin, _stats) = run_dry_active_nematic_3d(
        &q0,
        &p,
        n_steps,
        n_steps + 1,
        &out,
        false,
    );
    let elapsed = t0.elapsed().as_secs_f64();
    let usps = elapsed * 1e6 / (sites as f64 * n_steps as f64);

    println!(
        "threads={:<4} N={:<4} sites={:<10} steps={:<6} wall={:.3}s  us/site/step={:.4}",
        threads, n, sites, n_steps, elapsed, usps
    );
}
