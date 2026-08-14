//! Validation + timing harness for the CUDA FIRE minimiser.
//!
//! Two phases, in order, matching the task's non-negotiable rule that
//! correctness gates speed:
//!
//! 1. **Validate**: a small grid (N=8), GPU FIRE vs CPU FIRE
//!    (`volterra_solver::fire_minimize_3d_par`) from the *same* initial
//!    condition, checked to a stated tolerance. Refuses to time anything if
//!    this fails.
//! 2. **Time**: N=100, the benchmark size, wall-clock to reach the target
//!    residual (open-Qmin's own `force_max = sqrt(sum|f|^2)/N` quantity),
//!    reported the same way `bench_matched_convergence` reports the CPU
//!    numbers.
//!
//! Run: `cargo oxide run volterra-cuda` (or `cargo oxide run volterra-cuda --release -- <n> <target>`).

use std::time::Instant;

use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::{fire_minimize_3d_par, FireParams as CpuFireParams};

use volterra_cuda::{Device, FireParams as GpuFireParams, LdgParams};

fn flatten(q: &QField3D) -> Vec<f64> {
    let mut out = Vec::with_capacity(q.len() * 5);
    for site in &q.q {
        out.extend_from_slice(site);
    }
    out
}

fn analytic_s0(p: &ActiveNematicParams3D) -> f64 {
    (-3.0 * p.a_eff() / (4.0 * p.c_landau)).sqrt()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn run_validation(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = 8usize;
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;

    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 7);
    let q0_flat = flatten(&q0);

    // --- CPU FIRE (the already-validated reference) ---
    let cpu_params = CpuFireParams::open_qmin_defaults(p.dt, 1e-9, 5000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "CPU FIRE reference did not converge: force_max={}",
            cpu_result.force_max
        )
        .into());
    }
    let cpu_q = flatten(&cpu_result.q);

    // --- GPU FIRE ---
    let ldg = LdgParams {
        nx: n as u32,
        ny: n as u32,
        nz: n as u32,
        dx: p.dx,
        a_eff: p.a_eff(),
        c_landau: p.c_landau,
        k_r: p.k_r,
        gamma_r: p.gamma_r,
    };
    let gpu_params = GpuFireParams::open_qmin_defaults(p.dt, 1e-9, 5000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "GPU FIRE did not converge: force_max={}",
            gpu_result.force_max
        )
        .into());
    }

    let tol = 1e-9;
    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    println!(
        "validation (N={n}): CPU {} iters force_max={:.3e}; GPU {} iters force_max={:.3e}; max|Q_cpu - Q_gpu|={:.3e}",
        cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff
    );
    if diff > tol {
        return Err(format!(
            "GPU FIRE result disagrees with CPU FIRE by {diff:e}, exceeding tolerance {tol:e}"
        )
        .into());
    }
    println!("validation PASSED: GPU agrees with CPU to {diff:e} (tolerance {tol:e})");
    Ok(())
}

fn run_timed(dev: &Device, n: usize, target: f64, max_iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
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
    let q0_flat = flatten(&q0);

    let ldg = LdgParams {
        nx: n as u32,
        ny: n as u32,
        nz: n as u32,
        dx: p.dx,
        a_eff: p.a_eff(),
        c_landau: p.c_landau,
        k_r: p.k_r,
        gamma_r: p.gamma_r,
    };
    let gpu_params = GpuFireParams::open_qmin_defaults(p.dt, target, max_iterations);

    // One untimed warm-up run: pays for context/module lazy init so the
    // timed run measures the algorithm, not first-touch driver overhead.
    let _ = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;

    let t0 = Instant::now();
    let result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    let elapsed = t0.elapsed().as_secs_f64();
    let usps = elapsed * 1e6 / (sites as f64 * result.iterations.max(1) as f64);

    println!(
        "GPU FIRE  N={n} steps={} force_max={:.6e} wall={:.4}s us/site/step={:.4} reached={}",
        result.iterations, result.force_max, elapsed, usps, result.converged
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let target: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let max_iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let dev = Device::new(0)?;

    run_validation(&dev)?;
    run_timed(&dev, n, target, max_iterations)?;

    Ok(())
}
