//! Same sweep as `sweep_fire_params.rs`, but on the matched-physics
//! landscape (`volterra-cuda/src/main.rs`'s `setup_matched`: open-Qmin's own
//! default `(a,b,c,L1)` under the `a_eff=2a, b_landau=b, c_landau=2c,
//! k_r=L1` mapping), at the literal `1e-3` target directly. `volterra_tuned`
//! (delta_t_inc=1.6, alpha_dec=0.7, n_min=0) was tuned on volterra's own,
//! different, pre-cubic-term landscape; this checks whether it is still
//! near-optimal here, or whether the matched landscape wants something else.
//!
//! Run: `cargo run --release --example sweep_fire_params_matched -p volterra-solver`

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_fd::{fire_minimize_3d_par, FireParams};

const A_OQ: f64 = -0.172;
const B_OQ: f64 = -2.12;
const C_OQ: f64 = 1.73;
const L1_OQ: f64 = 4.64;

fn setup_matched() -> ActiveNematicParams3D {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = 100;
    p.ny = 100;
    p.nz = 100;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.0005;
    p.a_landau = 2.0 * A_OQ;
    p.b_landau = B_OQ;
    p.c_landau = 2.0 * C_OQ;
    p.k_r = L1_OQ;
    p
}

fn analytic_s0_matched(p: &ActiveNematicParams3D) -> f64 {
    let a_eff = p.a_eff();
    let disc = 9.0 * p.b_landau * p.b_landau - 48.0 * a_eff * p.c_landau;
    (-3.0 * p.b_landau + disc.sqrt()) / (8.0 * p.c_landau)
}

fn main() {
    let p = setup_matched();
    let s0 = analytic_s0_matched(&p);
    let q0 = QField3D::random_director_field(p.nx, p.ny, p.nz, p.dx, s0, 42);
    let target = 1e-3;

    println!("baseline (open-Qmin's own defaults): delta_t_inc=1.1 alpha_dec=0.9 n_min=4");
    let base = FireParams::open_qmin_defaults(p.dt, target, 20000);
    let r = fire_minimize_3d_par(&q0, &p, &base, 0.0);
    println!("  steps={} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);

    println!("\nvolterra_tuned (delta_t_inc=1.6, alpha_dec=0.7, n_min=0) -- the OLD tuning:");
    let old_tuned = FireParams::volterra_tuned(p.dt, target, 20000);
    let r = fire_minimize_3d_par(&q0, &p, &old_tuned, 0.0);
    println!("  steps={} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);

    let delta_t_incs = [1.05, 1.1, 1.15, 1.2, 1.3];
    let alpha_decs = [0.8, 0.9, 0.95, 0.99];
    let n_mins = [1, 2, 4, 8];

    println!("\ndelta_t_inc sweep (alpha_dec=0.9, n_min=4):");
    for &inc in &delta_t_incs {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.delta_t_inc = inc;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!("  delta_t_inc={inc:<5} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\nalpha_dec sweep (delta_t_inc=1.1, n_min=4):");
    for &dec in &alpha_decs {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.alpha_dec = dec;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!("  alpha_dec={dec:<5} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\nn_min sweep (delta_t_inc=1.1, alpha_dec=0.9):");
    for &nm in &n_mins {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.n_min = nm;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!("  n_min={nm:<5} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\ncombined trials:");
    for (label, inc, dec, nm) in [
        ("individually-best (1.3, 0.99, 1)", 1.3, 0.99, 1),
        ("old volterra_tuned (1.6, 0.7, 0)", 1.6, 0.7, 0),
        ("push further (1.8, 0.99, 0)", 1.8, 0.99, 0),
        ("push further (2.0, 0.99, 0)", 2.0, 0.99, 0),
        ("push further (2.2, 0.99, 0)", 2.2, 0.99, 0),
        ("push further (2.5, 0.99, 0)", 2.5, 0.99, 0),
        ("push further (3.0, 0.99, 0)", 3.0, 0.99, 0),
        ("push further (3.5, 0.99, 0)", 3.5, 0.99, 0),
        ("push further (3.0, 0.999, 0)", 3.0, 0.999, 0),
    ] {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.delta_t_inc = inc;
        params.alpha_dec = dec;
        params.n_min = nm;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!("  {label:<40} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\nalpha_dec=0.7 held fixed (volterra_tuned's own, proven stable at N=8 on the\ndevice), delta_t_inc pushed, n_min=0:");
    for inc in [1.6, 1.8, 2.0, 2.2, 2.5, 3.0] {
        let mut params = FireParams::open_qmin_defaults(p.dt, target, 20000);
        params.delta_t_inc = inc;
        params.alpha_dec = 0.7;
        params.n_min = 0;
        let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        println!("  delta_t_inc={inc:<5} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\nmatched_tuned (delta_t_inc=2.5, alpha_dec=0.7, n_min=0) -- the value that passes\nthe N=8 GPU-vs-CPU tight-tolerance gate reproducibly (six repeated runs on the\ndevice, see FireParams::matched_tuned's doc comment), four seeds:");
    for seed in [42u64, 7, 100, 999] {
        let q0_seed = QField3D::random_director_field(p.nx, p.ny, p.nz, p.dx, s0, seed);
        let mut tuned = FireParams::open_qmin_defaults(p.dt, target, 20000);
        tuned.delta_t_inc = 2.5;
        tuned.alpha_dec = 0.7;
        tuned.n_min = 0;
        let r = fire_minimize_3d_par(&q0_seed, &p, &tuned, 0.0);
        println!("  seed={seed:<5} steps={:<5} converged={} force_max={:.3e}", r.iterations, r.converged, r.force_max);
    }

    println!("\nCPU wall-clock, matched physics, N=100, three repeats each:");
    for (label, params_fn) in [
        ("baseline", FireParams::open_qmin_defaults as fn(f64, f64, usize) -> FireParams),
        ("matched_tuned", FireParams::matched_tuned as fn(f64, f64, usize) -> FireParams),
    ] {
        for rep in 0..3 {
            let params = params_fn(p.dt, target, 20000);
            let t0 = std::time::Instant::now();
            let r = fire_minimize_3d_par(&q0, &p, &params, 0.0);
            let elapsed = t0.elapsed().as_secs_f64();
            println!("  {label:<15} rep={rep} steps={:<5} wall={elapsed:.4}s", r.iterations);
        }
    }
}

