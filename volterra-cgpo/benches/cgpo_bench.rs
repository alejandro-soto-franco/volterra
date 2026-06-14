//! Criterion micro-benchmark for the nephroid-confined CGPO finite-difference
//! solver.
//!
//! One benchmark, `cgpo_step`, advances the solver a few steps on a small grid
//! (lx=32) through the same public surface `cgpo_fd` uses: a `CgpoStep`
//! `PhysicsStep` driven by `SimulationRunner` against a no-op observer, with
//! snapshots disabled (`snap_every` > steps, `snap_final = false`).
//!
//! The boundary and params are built ONCE outside the timed closure. `State` is
//! not `Clone`, so each iteration rebuilds its tiny (32x32) state from the same
//! seeded initial condition; that allocation is negligible against the step work
//! and keeps every measured run starting from an identical point. `black_box`
//! guards the state handed to the runner.
//!
//! Run: `cargo bench -p volterra-cgpo --bench cgpo_bench`

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use volterra_cgpo::{
    boundary::nephroid_boundary,
    index::vi,
    sim_step::CgpoStep,
    step::State,
    Params,
};
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::{Observer, RunConfig, SimulationRunner};

/// No-op observer: never invoked because snapshots are disabled, but the runner
/// requires one.
struct NullObserver;
impl Observer<State> for NullObserver {
    fn observe(&mut self, _step: usize, _t: f64, _state: &State, _stats: &StepStats) {}
}

/// Random theta IC at interior cells, matching `cgpo_fd::random_theta_ic`.
fn random_theta_ic(q: &mut [f64], s0: f64, lx: usize, ly: usize, inside: &[bool], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            let (qxx, qxy) = if inside[idx] {
                let theta: f64 = PI * rng.random::<f64>();
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                (s0 * (cos_t * cos_t - 0.5), s0 * (cos_t * sin_t))
            } else {
                (0.0, 0.0)
            };
            q[vi(x, y, ly, 0)] = qxx;
            q[vi(x, y, ly, 1)] = qxy;
        }
    }
}

fn cgpo_step(c: &mut Criterion) {
    let lx = 32usize;
    let ly = lx;
    let n_steps = 5usize;

    // Same parameter map cgpo_fd uses (defaults als=2.8, ncl=4.8, lambda=1).
    let params = Params::new(lx, 2.8, 4.8, 1.0, 1e-4, -1);
    let boundary = nephroid_boundary(lx, ly);
    let target_rel_change = 1e-4_f64;

    let cfg = RunConfig {
        steps: n_steps,
        snap_every: n_steps + 1, // > steps: no snapshot work
        dt: params.dt,
        seed: 0,
        snap_final: false,
    };

    // `State` is not `Clone`, so each iteration rebuilds its tiny (32x32) state
    // from the same seeded IC. This allocation is negligible against the step
    // work and keeps every measured run starting from an identical point.
    c.bench_function("cgpo_step", |b| {
        b.iter(|| {
            let mut physics = CgpoStep {
                params: params.clone(),
                boundary: boundary.clone(),
                target_rel_change,
            };
            let mut state = State::new(lx, ly);
            random_theta_ic(&mut state.q, params.s0, lx, ly, &boundary.inside, 0);
            let runner = SimulationRunner { config: cfg.clone() };
            let mut obs = NullObserver;
            runner.run(black_box(&mut state), &mut physics, &mut obs);
        });
    });
}

criterion_group!(benches, cgpo_step);
criterion_main!(benches);
