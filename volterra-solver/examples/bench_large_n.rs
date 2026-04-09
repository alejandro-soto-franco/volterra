//! Large-N scaling benchmark: N=50, 100, 150, 200.
//! Tests cache behaviour at large problem sizes.

use std::time::Instant;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::run_dry_active_nematic_3d;

fn main() {
    let configs: Vec<(usize, usize)> = vec![
        (50, 100),
        (100, 50),
        (150, 20),
        (200, 10),
    ];

    println!("volterra large-N scaling (passive LdG, rayon auto)");
    println!(
        "{:<6} {:<12} {:<8} {:<12} {:<14} {:<10}",
        "N", "sites", "steps", "wall (s)", "us/site/step", "MB RSS"
    );

    for (n, n_steps) in configs {
        let mut p = ActiveNematicParams3D::default_test();
        p.nx = n;
        p.ny = n;
        p.nz = n;
        p.zeta_eff = 0.0;
        p.noise_amp = 0.0;
        p.dt = 0.001;

        let q0 = QField3D::random_perturbation(n, n, n, p.dx, 0.01, 42);
        let sites = n * n * n;

        let out = std::env::temp_dir().join("volterra_bench_large");
        std::fs::create_dir_all(&out).ok();

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

        // Estimate RSS from Q-tensor size (2 copies: q + rhs)
        let q_bytes = sites * 5 * 8; // 5 components * 8 bytes
        let est_mb = (q_bytes * 2) as f64 / 1e6; // q + rhs

        println!(
            "{:<6} {:<12} {:<8} {:<12.3} {:<14.4} {:<10.0}",
            n, sites, n_steps, elapsed, usps, est_mb
        );
    }
}
