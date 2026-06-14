//! Criterion micro-benchmarks for the hot solver paths.
//!
//! Two benchmarks, both built to keep per-iteration work small enough that the
//! whole suite finishes in a few seconds:
//!   * `wet_2d_step`: `run_active_nematic_hydro` on a 64x64 grid, ~50 steps per
//!     iteration, active-turbulent params (zeta_eff=3.0, dt=0.005), snapshots
//!     disabled.
//!   * `dry_3d_step`: `run_dry_active_nematic_3d` on N=24, ~50 steps, passive
//!     (zeta_eff=0), snapshots disabled, frames written to a temp dir that is
//!     never populated (snap_every > n_steps).
//!
//! Initial fields are built ONCE outside the timed closures; `black_box` guards
//! the inputs so the optimiser cannot fold the work away.
//!
//! Run: `cargo bench -p volterra-solver --bench solver_bench`

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};

use volterra_core::{ActiveNematicParams, ActiveNematicParams3D};
use volterra_fields::{QField2D, QField3D};
use volterra_solver::{run_active_nematic_hydro, run_dry_active_nematic_3d};

// Snapshot cadence set far above the step count: no per-snapshot defect scan or
// frame write happens inside the timed region.
const NO_SNAP: usize = 10_000_000;

fn wet_2d_step(c: &mut Criterion) {
    let nx = 64usize;
    let ny = 64usize;
    let n_steps = 50usize;

    let mut params = ActiveNematicParams::default_test();
    params.nx = nx;
    params.ny = ny;
    params.dx = 1.0;
    params.dt = 0.005;
    params.zeta_eff = 3.0; // active-turbulent, matches bench_wet_2d_clean.rs

    let q0 = QField2D::random_perturbation(nx, ny, params.dx, 0.01, 42);

    c.bench_function("wet_2d_step", |b| {
        b.iter(|| {
            let _ = run_active_nematic_hydro(
                black_box(&q0),
                black_box(&params),
                black_box(n_steps),
                NO_SNAP,
            );
        });
    });
}

fn dry_3d_step(c: &mut Criterion) {
    let n = 24usize;
    let n_steps = 50usize;

    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0; // passive
    p.noise_amp = 0.0;
    p.dt = 0.001;

    let q0 = QField3D::random_perturbation(n, n, n, p.dx, 0.01, 42);

    // Temp output dir; with snap_every > n_steps no frame is ever written here.
    let out_dir: PathBuf = std::env::temp_dir().join("volterra_bench_dry_3d");
    std::fs::create_dir_all(&out_dir).expect("create temp bench dir");

    c.bench_function("dry_3d_step", |b| {
        b.iter(|| {
            let _ = run_dry_active_nematic_3d(
                black_box(&q0),
                black_box(&p),
                black_box(n_steps),
                NO_SNAP,
                &out_dir,
                false,
            );
        });
    });
}

criterion_group!(benches, wet_2d_step, dry_3d_step);
criterion_main!(benches);
