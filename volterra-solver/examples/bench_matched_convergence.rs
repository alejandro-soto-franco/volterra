//! Matched-metric convergence benchmark: volterra's Euler integrator and its
//! FIRE minimiser, both scored against open-Qmin's own residual quantity
//! (`force_max_metric`, `sqrt(sum_i |f_i|^2) / N`), from the same kind of
//! initial condition open-Qmin uses by default: a fully randomised director
//! field at the analytic bulk equilibrium magnitude S0, not a small
//! perturbation around Q=0.
//!
//! This replaces the informal per-element `max|dQ/dt|` metric used in the
//! earlier `bench_convergence.rs`, which is not the quantity open-Qmin
//! reports (see `volterra_solver::fire` for why that mismatch mattered: a
//! small-amplitude IC scored by `sqrt(sum|f|^2)/N` trivially reads as
//! "already converged" for large N, regardless of physical progress).
//!
//! Run: `cargo run --release --example bench_matched_convergence -p volterra-solver`

use std::time::Instant;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::{
    beris_edwards_rhs_3d_par_dry, euler_step_par, fire_step_3d_par, force_max_metric, FireParams,
    FireState,
};

fn analytic_s0(p: &ActiveNematicParams3D) -> f64 {
    (-3.0 * p.a_eff() / (4.0 * p.c_landau)).sqrt()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let target: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let max_steps: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;

    let s0 = analytic_s0(&p);
    let sites = n * n * n;
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);

    eprintln!("N={n} sites={sites} S0={s0:.6} target={target:e} max_steps={max_steps}");

    // FIRE
    {
        let fire_params = FireParams::open_qmin_defaults(p.dt, 1e-12, max_steps);
        let mut q = q0.clone();
        let mut f = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);
        let mut state = FireState::new(&q, &fire_params);
        let t0 = Instant::now();
        let mut force_max = f64::INFINITY;
        while state.iterations < max_steps && force_max > target {
            force_max = fire_step_3d_par(&mut q, &mut f, &mut state, &fire_params, &p, 0.0);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let usps = elapsed * 1e6 / (sites as f64 * state.iterations.max(1) as f64);
        println!(
            "FIRE   N={n} steps={} force_max={:.6e} wall={:.4}s us/site/step={:.4} reached={}",
            state.iterations,
            force_max,
            elapsed,
            usps,
            force_max <= target
        );
    }

    // Euler
    {
        let mut q = q0.clone();
        let t0 = Instant::now();
        let mut force_max = f64::INFINITY;
        let mut step = 0usize;
        while step < max_steps && force_max > target {
            let rhs = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);
            force_max = force_max_metric(&rhs);
            if force_max <= target {
                break;
            }
            q = euler_step_par(&q, p.dt, &rhs);
            step += 1;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let usps = elapsed * 1e6 / (sites as f64 * step.max(1) as f64);
        println!(
            "Euler  N={n} steps={step} force_max={:.6e} wall={:.4}s us/site/step={:.4} reached={}",
            force_max,
            elapsed,
            usps,
            force_max <= target
        );
    }
}
