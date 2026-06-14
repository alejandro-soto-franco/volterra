//! Release-mode performance regression gate for the 2D wet active nematic solver.
//!
//! ## What this guards
//! It runs a fixed workload (64x64 grid, 100 steps, snapshots disabled), excludes
//! a warmup run, measures wall-clock, and computes microseconds per lattice site
//! per step. It then asserts that figure stays under a GENEROUS ceiling. The goal
//! is to catch CATASTROPHIC regressions (lost parallelism, an accidental O(n^2)
//! kernel, an accidental debug build in a release pipeline), not 2-3x noise.
//!
//! ## Why release-only
//! Debug timings are meaningless (no inlining, overflow checks, no SIMD), so a
//! hard wall-clock threshold can only be trusted in release. When compiled
//! WITHOUT optimisation (`cfg!(debug_assertions)`), the test prints the measured
//! value and returns early WITHOUT asserting, so `cargo test` in a debug CI lane
//! never flakes. The assertion fires only in release builds:
//!   `cargo test --release -p volterra-solver --test perf_floor`
//!
//! ## The ceiling and how to update it
//! `CEILING_US_PER_SITE_STEP` is set to roughly 8-10x the measured release median
//! on the development machine, giving headroom for slower CI runners and noise
//! while still tripping on a true blow-up. If the solver legitimately gets faster
//! you may lower it; if CI hardware legitimately changes, re-measure and re-set:
//!   1. `cargo test --release -p volterra-solver --test perf_floor -- --nocapture`
//!   2. read the printed `us_per_site_step`,
//!   3. set the constant to ~8-10x that median.
//!
//! Measured release median on the dev box (2026-06-14): ~0.044 us/site/step.
//! Ceiling set to 0.45 (~10x) to absorb slower CI hardware without flaking.

use std::time::Instant;

use volterra_core::ActiveNematicParams;
use volterra_fields::QField2D;
use volterra_solver::run_active_nematic_hydro;

/// Generous release-mode ceiling, microseconds per site per step.
/// See the module comment for the rationale and the update recipe.
const CEILING_US_PER_SITE_STEP: f64 = 0.45;

#[test]
fn wet_2d_throughput_above_floor() {
    let nx = 64usize;
    let ny = 64usize;
    let n_steps = 100usize;
    let no_snap = 10_000_000usize;

    let mut params = ActiveNematicParams::default_test();
    params.nx = nx;
    params.ny = ny;
    params.dx = 1.0;
    params.dt = 0.005;
    params.zeta_eff = 3.0;

    let q0 = QField2D::random_perturbation(nx, ny, params.dx, 0.01, 42);

    // Warmup, excluded from timing (cache warmup, allocation).
    let _ = run_active_nematic_hydro(&q0, &params, 20, no_snap);

    let t0 = Instant::now();
    let _ = run_active_nematic_hydro(&q0, &params, n_steps, no_snap);
    let wall = t0.elapsed().as_secs_f64();

    let sites = (nx * ny) as f64;
    let us_per_site_step = 1e6 * wall / (n_steps as f64 * sites);

    println!(
        "perf_floor wet_2d grid={nx}x{ny} steps={n_steps} wall={wall:.4}s \
         us_per_site_step={us_per_site_step:.5} ceiling={CEILING_US_PER_SITE_STEP}"
    );

    // Debug timings are meaningless; never assert outside release builds.
    if cfg!(debug_assertions) {
        println!("perf_floor: debug build, skipping assertion (release-only gate)");
        return;
    }

    assert!(
        us_per_site_step < CEILING_US_PER_SITE_STEP,
        "wet-2d throughput regressed: {us_per_site_step:.5} us/site/step exceeds \
         ceiling {CEILING_US_PER_SITE_STEP} (see top-of-file note to re-measure/update)"
    );
}
