//! Validation + timing harness for the CUDA FIRE minimiser.
//!
//! Two phases per FIRE preset, in order, matching the task's non-negotiable
//! rule that correctness gates speed:
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
//! Two presets are run: `open_qmin_defaults` (the faithful port, for parity
//! with open-Qmin's own tuning) and `volterra_tuned` (retuned for volterra's
//! own bulk/elastic constants; see `FireParams::volterra_tuned`'s doc
//! comment). Both are validated before either is timed.
//!
//! Also reports the roofline (measured achieved bandwidth and kernel launch
//! overhead, not read off a spec sheet) and a phase breakdown (CUDA context
//! and module load, validation, warm-up vs timed run, total process
//! wall-clock), so the fairness accounting in `BENCHMARKS.md` is measured
//! rather than assumed.
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

/// A named pair of CPU/GPU FIRE parameter builders, so the same driver code
/// below validates and times both presets identically.
struct Preset {
    name: &'static str,
    cpu: fn(f64, f64, usize) -> CpuFireParams,
    gpu: fn(f64, f64, usize) -> GpuFireParams,
}

const PRESETS: [Preset; 2] = [
    Preset {
        name: "open_qmin_defaults (ported)",
        cpu: CpuFireParams::open_qmin_defaults,
        gpu: GpuFireParams::open_qmin_defaults,
    },
    Preset {
        name: "volterra_tuned",
        cpu: CpuFireParams::volterra_tuned,
        gpu: GpuFireParams::volterra_tuned,
    },
];

fn run_validation(dev: &Device, preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
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
    let cpu_params = (preset.cpu)(p.dt, 1e-9, 5000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "[{}] CPU FIRE reference did not converge: force_max={}",
            preset.name, cpu_result.force_max
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
    let gpu_params = (preset.gpu)(p.dt, 1e-9, 5000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "[{}] GPU FIRE did not converge: force_max={}",
            preset.name, gpu_result.force_max
        )
        .into());
    }

    let tol = 1e-9;
    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    println!(
        "[{}] validation (N={n}): CPU {} iters force_max={:.3e}; GPU {} iters force_max={:.3e}; max|Q_cpu - Q_gpu|={:.3e}",
        preset.name, cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff
    );
    if diff > tol {
        return Err(format!(
            "[{}] GPU FIRE result disagrees with CPU FIRE by {diff:e}, exceeding tolerance {tol:e}",
            preset.name
        )
        .into());
    }
    println!("[{}] validation PASSED: GPU agrees with CPU to {diff:e} (tolerance {tol:e})", preset.name);
    Ok(())
}

fn run_timed(
    dev: &Device,
    preset: &Preset,
    n: usize,
    target: f64,
    max_iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let gpu_params = (preset.gpu)(p.dt, target, max_iterations);

    // One untimed warm-up run: pays for context/module lazy init so the
    // timed run measures the algorithm, not first-touch driver overhead.
    // Timed separately (not just excluded) so the fairness accounting in
    // BENCHMARKS.md is measured, not assumed: if this is markedly slower
    // than the timed run below, that gap is what a "both codes include
    // startup" comparison would need to add back to volterra's number.
    let tw0 = Instant::now();
    let warmup = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    let warmup_elapsed = tw0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    let elapsed = t0.elapsed().as_secs_f64();
    let usps = elapsed * 1e6 / (sites as f64 * result.iterations.max(1) as f64);

    println!(
        "[{}] GPU FIRE warm-up  N={n} steps={} wall={:.4}s (untimed run, first touch of this n/target)",
        preset.name, warmup.iterations, warmup_elapsed
    );
    println!(
        "[{}] GPU FIRE  N={n} steps={} force_max={:.6e} wall={:.4}s us/site/step={:.4} reached={}",
        preset.name, result.iterations, result.force_max, elapsed, usps, result.converged
    );
    println!(
        "[{}] warm-up vs timed delta: {:.4}s ({:.1}% of timed run)",
        preset.name,
        warmup_elapsed - elapsed,
        100.0 * (warmup_elapsed - elapsed) / elapsed
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let process_start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let target: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let max_iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    // Phase accounting for the fairness correction in BENCHMARKS.md: how
    // much of a from-nothing run is one-time CUDA context/module setup vs
    // the algorithm itself. `Device::new` opens the context and loads the
    // compiled kernel module (cuModuleLoadData under the hood); everything
    // timed after it has a live context and a resident module.
    let t_ctx0 = Instant::now();
    let dev = Device::new(0)?;
    let t_ctx = t_ctx0.elapsed().as_secs_f64();
    println!("phase: CUDA context + module load  {t_ctx:.4}s");

    // Roofline: measured, not read off a spec sheet. 256M f64 elements
    // (2 GiB) is comfortably past the size where a copy kernel is
    // bandwidth-bound rather than latency-bound, and comfortably inside
    // this card's 8 GiB with room for the FIRE buffers used elsewhere.
    let bw_elements = 256usize * 1024 * 1024;
    let bandwidth_gbps = dev.measure_bandwidth(bw_elements, 20)?;
    println!(
        "roofline: measured achieved bandwidth (pure copy, {} MiB, 20 reps) = {bandwidth_gbps:.1} GB/s",
        bw_elements * 8 / (1024 * 1024)
    );
    let launch_overhead_us = dev.measure_launch_overhead(2000)? * 1e6;
    println!("roofline: mean kernel launch overhead (1-element launches, 2000 reps) = {launch_overhead_us:.2} us");

    for preset in &PRESETS {
        let t_val0 = Instant::now();
        run_validation(&dev, preset)?;
        let t_val = t_val0.elapsed().as_secs_f64();
        println!("[{}] phase: validation (N=8, CPU+GPU cross-check)  {t_val:.4}s", preset.name);

        run_timed(&dev, preset, n, target, max_iterations)?;
    }

    let t_total = process_start.elapsed().as_secs_f64();
    println!("phase: total process wall-clock (args parse through exit)  {t_total:.4}s");
    println!(
        "one-time setup (context+module load, ahead of every timed run) = {t_ctx:.4}s, {:.1}% of total process wall-clock",
        100.0 * t_ctx / t_total
    );

    Ok(())
}
