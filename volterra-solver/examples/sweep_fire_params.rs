//! Quick sweep over FIRE's tuning constants (delta_t_inc, alpha_dec, n_min),
//! ported unchanged from open-Qmin's own defaults so far. open-Qmin tuned
//! these for its own energy landscape; volterra's bulk/elastic constants
//! differ (see BENCHMARKS.md's "Scale mismatch" caveat), so there is no
//! reason the same constants are optimal here. Cheap to check: a step saved
//! is worth as much as a faster kernel, and this needs no GPU.
//!
//! Run: `cargo run --release --example sweep_fire_params -p volterra-solver`

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_solver::{fire_minimize_3d_par, FireParams};

fn analytic_s0(p: &ActiveNematicParams3D) -> f64 {
    (-3.0 * p.a_eff() / (4.0 * p.c_landau)).sqrt()
}

fn main() {
    let n = 100usize;
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;

    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let target = 2.09e-5; // the scale-matched target from BENCHMARKS.md

    println!("baseline (open-Qmin's own defaults): delta_t_inc=1.1 alpha_dec=0.9 n_min=4");
    let base = FireParams::open_qmin_defaults(p.dt, target, 20000);
    let r = fire_minimize_3d_par(&q0, &p, &base, 0.0);
    println!(
        "  steps={} converged={} force_max={:.3e}",
        r.iterations, r.converged, r.force_max
    );

    let delta_t_incs = [1.05, 1.1, 1.15, 1.2, 1.3];
    let alpha_decs = [0.8, 0.9, 0.95, 0.99];
    let n_mins = [1, 2, 4, 8];

    println!("\ndelta_t_inc sweep (alpha_dec=0.9, n_min=4):");
    for &inc in &delta_t_incs {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.delta_t_inc = inc;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!(
            "  delta_t_inc={inc:<5} steps={:<5} converged={} force_max={:.3e}",
            r.iterations, r.converged, r.force_max
        );
    }

    println!("\nalpha_dec sweep (delta_t_inc=1.1, n_min=4):");
    for &dec in &alpha_decs {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.alpha_dec = dec;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!(
            "  alpha_dec={dec:<5} steps={:<5} converged={} force_max={:.3e}",
            r.iterations, r.converged, r.force_max
        );
    }

    println!("\nn_min sweep (delta_t_inc=1.1, alpha_dec=0.9):");
    for &nm in &n_mins {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.n_min = nm;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!(
            "  n_min={nm:<5} steps={:<5} converged={} force_max={:.3e}",
            r.iterations, r.converged, r.force_max
        );
    }

    // The chosen constants (delta_t_inc=1.6, alpha_dec=0.7, n_min=0, wired
    // into FireParams::volterra_tuned), checked across four seeds so the
    // 43->19 step reduction is not an artefact of one initial condition.
    println!("\nvolterra_tuned (delta_t_inc=1.6, alpha_dec=0.7, n_min=0), four seeds:");
    for seed in [42u64, 7, 100, 999] {
        let q0_seed = QField3D::random_director_field(n, n, n, p.dx, s0, seed);
        let tuned = FireParams::volterra_tuned(p.dt, target, 20000);
        let r = fire_minimize_3d_par(&q0_seed, &p, &tuned, 0.0);
        println!(
            "  seed={seed:<5} steps={:<5} converged={} force_max={:.3e}",
            r.iterations, r.converged, r.force_max
        );
    }
}
