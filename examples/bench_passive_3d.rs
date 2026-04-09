//! Quick benchmark: passive LdG relaxation on a 3D periodic grid.
//! Equivalent to open-Qmin with zero activity.
//!
//! Usage: cargo run --release --example bench_passive_3d

use std::time::Instant;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::run_dry_active_nematic_3d;

fn main() {
    let sizes = [20, 30, 50];
    let n_steps = 500;

    println!("volterra passive LdG relaxation (3D periodic, RK4)");
    println!("{:<8} {:<12} {:<12} {:<15} {:<12}", "N", "sites", "steps", "wall_clock_s", "us/site/step");

    for &n in &sizes {
        let mut p = ActiveNematicParams3D::default_test();
        p.nx = n;
        p.ny = n;
        p.nz = n;
        p.zeta_eff = 0.0;  // passive
        p.noise_amp = 0.0;
        p.dt = 0.001;

        let q0 = QField3D::random_perturbation(n, n, n, p.dx, 0.01, 42);
        let sites = n * n * n;

        let t0 = Instant::now();
        let (_q_fin, _stats) = run_dry_active_nematic_3d(
            &q0, &p, n_steps, n_steps,
            std::path::Path::new("/dev/null"), false,
        );
        let elapsed = t0.elapsed().as_secs_f64();

        let us_per_site_step = elapsed * 1e6 / (sites as f64 * n_steps as f64);
        println!("{:<8} {:<12} {:<12} {:<15.3} {:<12.4}", n, sites, n_steps, elapsed, us_per_site_step);
    }
}
