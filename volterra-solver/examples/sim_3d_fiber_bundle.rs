//! 3D Beris-Edwards relaxation using the fiber bundle framework (Milestone 4).
//!
//! Demonstrates that the generic CovLaplacian<NematicFiber3D> with SO(3)
//! identity transport on a Cartesian grid reproduces the standard 3D
//! Landau-de Gennes relaxation.
//!
//! This validates the fiber bundle stack end-to-end:
//!   cartan-core: NematicFiber3D + CovLaplacian
//!   cartan-dec: cartesian_3d_connection
//!   volterra: 5-component Q-tensor evolution
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_3d_fiber_bundle

use std::time::Instant;

use cartan_core::bundle::EdgeTransport3D;
use cartan_core::fiber::{NematicFiber3D, Section, VecSection, FiberOps};
use cartan_dec::cartesian_connection::cartesian_3d_connection;

fn main() {
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 1.0;
    let dt = 0.01;
    let n_steps = 2000;
    let n = nx * ny * nz;

    // Physics parameters.
    let k_frank = 1.0;
    let a_eff = -1.5; // ordered phase
    let c_landau = 4.5;
    let gamma_r = 1.0;

    println!("=== 3D Beris-Edwards via Fiber Bundle Framework ===");
    println!("  grid: {nx}x{ny}x{nz} = {n} vertices");
    println!("  K = {k_frank}, a_eff = {a_eff}, c = {c_landau}");
    println!("  dt = {dt}, steps = {n_steps}");

    // Build Cartesian connection + CovLaplacian.
    println!("Building Cartesian connection...");
    let (transport, cov_lap) = cartesian_3d_connection(nx, ny, nz, dx);
    println!("  {} edges", transport.edges.len());

    // Initial Q: random perturbation.
    let mut rng_seed = 42_u64;
    let mut data: Vec<[f64; 5]> = (0..n)
        .map(|i| {
            // Simple PRNG for reproducibility without rand dependency.
            let mut v = [0.0_f64; 5];
            for c in 0..5 {
                rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                v[c] = 0.01 * ((rng_seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5);
            }
            v
        })
        .collect();

    let mean_s = |d: &[[f64; 5]]| -> f64 {
        let sum: f64 = d.iter().map(|q| {
            let q33 = -q[0] - q[3];
            let tr2 = q[0]*q[0] + q[3]*q[3] + q33*q33
                + 2.0*(q[1]*q[1] + q[2]*q[2] + q[4]*q[4]);
            // S = sqrt(3/2 * tr(Q^2)) for 3D (largest eigenvalue scaling).
            (1.5 * tr2).sqrt()
        }).sum::<f64>();
        sum / d.len() as f64
    };

    println!("Running LdG relaxation...");
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % 200 == 0 {
            let s = mean_s(&data);
            let elapsed = t0.elapsed().as_secs_f64();
            println!("  step {step:>5}/{n_steps}  <S>={s:.4}  wall={elapsed:.1}s");
        }

        if step < n_steps {
            // Apply CovLaplacian to get the connection Laplacian of Q.
            let section = VecSection::<NematicFiber3D>::from_vec(data.clone());
            let lap = cov_lap.apply::<NematicFiber3D, 3, _>(&section, &transport);

            // Euler step: dQ/dt = gamma_r * (-K*lap + bulk*Q)
            let bulk_linear = -a_eff; // positive when a_eff < 0
            for i in 0..n {
                let q = &data[i];
                let q33 = -q[0] - q[3];
                let tr_q2 = q[0]*q[0] + q[3]*q[3] + q33*q33
                    + 2.0*(q[1]*q[1] + q[2]*q[2] + q[4]*q[4]);
                let bulk = bulk_linear - 2.0 * c_landau * tr_q2;

                let lap_v = lap.at(i);
                for c in 0..5 {
                    // H = -K*lap + bulk*Q (DEC lap is positive-semidef, so -K*lap smooths)
                    let h_c = -k_frank * lap_v[c] + bulk * data[i][c];
                    data[i][c] += dt * gamma_r * h_c;
                }
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let final_s = mean_s(&data);
    println!("\nDone in {elapsed:.1}s");
    println!("Final <S> = {final_s:.4}");

    // Expected equilibrium: S_eq where bulk = 0.
    // bulk_linear - 2*c*tr = 0 => tr = bulk_linear/(2*c) = 1.5/9 = 0.1667
    // S = sqrt(1.5 * tr) = sqrt(0.25) = 0.5
    let s_eq = (1.5 * (-a_eff) / (2.0 * c_landau)).sqrt();
    println!("Expected S_eq = {s_eq:.4}");
    println!(
        "Relative error: {:.2}%",
        ((final_s - s_eq) / s_eq * 100.0).abs()
    );
}
