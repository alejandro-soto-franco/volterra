//! Convergence benchmark: time to reach a target max |dQ/dt| residual.
//!
//! Compares volterra's Euler integrator convergence rate against
//! open-Qmin's FIRE minimiser on equivalent passive LdG problems.
//!
//! Prints the max |dQ/dt| (proxy for max force) at each snapshot,
//! and the wall-clock time when the target is reached.

use std::time::Instant;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::{beris_edwards_rhs_3d_par_dry, euler_step_par};

fn max_rhs_norm(rhs: &QField3D) -> f64 {
    rhs.q
        .iter()
        .map(|c| c.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
        .fold(0.0_f64, f64::max)
}

fn main() {
    let sizes: Vec<usize> = vec![50, 100];
    let targets = [1e-3, 1e-4];
    let max_steps = 20_000;
    let check_every = 10;

    println!("volterra convergence benchmark (passive LdG, Euler)");
    println!(
        "{:<6} {:<10} {:<10} {:<14} {:<12} {:<14}",
        "N", "target", "steps", "max|dQ/dt|", "wall (s)", "us/site/step"
    );

    for &n in &sizes {
        let mut p = ActiveNematicParams3D::default_test();
        p.nx = n;
        p.ny = n;
        p.nz = n;
        p.zeta_eff = 0.0;
        p.noise_amp = 0.0;
        p.dt = 0.005; // larger dt for faster convergence (CFL safe at N<=100)

        let mut q = QField3D::random_perturbation(n, n, n, p.dx, 0.01, 42);
        let sites = n * n * n;

        let t0 = Instant::now();
        let mut step = 0;
        let mut current_target_idx = 0;

        while step < max_steps && current_target_idx < targets.len() {
            let rhs = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);

            if step % check_every == 0 {
                let max_f = max_rhs_norm(&rhs);
                let elapsed = t0.elapsed().as_secs_f64();

                // Check all remaining targets
                while current_target_idx < targets.len() && max_f < targets[current_target_idx] {
                    let usps = if step > 0 {
                        elapsed * 1e6 / (sites as f64 * step as f64)
                    } else {
                        0.0
                    };
                    println!(
                        "{:<6} {:<10.0e} {:<10} {:<14.6e} {:<12.3} {:<14.4}",
                        n, targets[current_target_idx], step, max_f, elapsed, usps
                    );
                    current_target_idx += 1;
                }
            }

            q = euler_step_par(&q, p.dt, &rhs);
            step += 1;
        }

        // Report final state if targets not all met
        if current_target_idx < targets.len() {
            let rhs = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);
            let max_f = max_rhs_norm(&rhs);
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "{:<6} {:<10} {:<10} {:<14.6e} {:<12.3} (did not converge)",
                n, "---", step, max_f, elapsed
            );
        }
    }
}
