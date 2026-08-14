//! Validation + timing harness for the CUDA FIRE minimiser.
//!
//! Correctness gates speed throughout: nothing in the `time-*`/`kernels`
//! phases below runs until the matching `validate` phase has passed.
//!
//! Phases (first CLI argument selects one; default `all`):
//!
//! - `roofline`: measured achieved bandwidth, kernel launch overhead and
//!   host-to-device upload time (not read off a spec sheet), plus the
//!   context+module load phase and the derived per-iteration bandwidth
//!   floor at N=100.
//! - `validate`: GPU FIRE vs CPU FIRE, both presets, at N=8 (to full
//!   convergence, the existing tight-tolerance check) and at N=100 (the
//!   benchmark size, at both the literal and scale-matched targets).
//!   Refuses (non-zero exit) if any check fails.
//! - `time-tuned`: `volterra_tuned` only, N=100, three repeats at the
//!   literal `1e-3` target and three at the scale-matched target, reporting
//!   the spread.
//! - `kernels`: `force_fused_aos` and `force_soa` timed back-to-back against
//!   the split `trq2`+`force` pipeline `fire_minimize` actually uses, same
//!   conditions, N=100.
//! - `all`: every phase above, in order.
//!
//! Run: `./target/release/volterra-cuda <phase>` once built with `cargo
//! oxide build volterra-cuda`.

use std::time::Instant;

use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::{fire_minimize_3d_par, FireParams as CpuFireParams};

use volterra_cuda::{Device, FireParams as GpuFireParams, LdgParams};

const LITERAL_TARGET: f64 = 1e-3;
const SCALE_MATCHED_TARGET: f64 = 2.09e-5;

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

fn setup(n: usize) -> (ActiveNematicParams3D, LdgParams) {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;
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
    (p, ldg)
}

/// N=8, run to convergence (`force_cutoff=1e-9`), GPU checked against CPU to
/// a `1e-9` tolerance. Refuses to let anything downstream time if this fails.
fn run_validation_n8(dev: &Device, preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let n = 8usize;
    let (p, ldg) = setup(n);
    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 7);
    let q0_flat = flatten(&q0);

    let cpu_params = (preset.cpu)(p.dt, 1e-9, 5000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "[{}] N=8 CPU FIRE reference did not converge: force_max={}",
            preset.name, cpu_result.force_max
        )
        .into());
    }
    let cpu_q = flatten(&cpu_result.q);

    let gpu_params = (preset.gpu)(p.dt, 1e-9, 5000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "[{}] N=8 GPU FIRE did not converge: force_max={}",
            preset.name, gpu_result.force_max
        )
        .into());
    }

    let tol = 1e-9;
    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    println!(
        "[{}] N=8 validation: CPU {} iters force_max={:.3e}; GPU {} iters force_max={:.3e}; max|Q_cpu-Q_gpu|={:.3e}",
        preset.name, cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff
    );
    if diff > tol {
        return Err(format!(
            "[{}] N=8: GPU disagrees with CPU by {diff:e}, exceeding tolerance {tol:e}",
            preset.name
        )
        .into());
    }
    println!("[{}] N=8 validation PASSED (max diff {diff:e} <= {tol:e})", preset.name);
    Ok(())
}

/// N=100 (the benchmark size), at a given target: CPU and GPU both run to
/// that target (not full 1e-9 convergence -- this checks agreement at the
/// exact operating point that gets timed), and must agree on iteration
/// count, `force_max`, and the converged field itself.
fn run_validation_n100(
    dev: &Device,
    preset: &Preset,
    target: f64,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = 100usize;
    let (p, ldg) = setup(n);
    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let q0_flat = flatten(&q0);

    let cpu_params = (preset.cpu)(p.dt, target, 2000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "[{}] N=100 ({label}) CPU FIRE did not reach target {target:e}: force_max={}",
            preset.name, cpu_result.force_max
        )
        .into());
    }
    let cpu_q = flatten(&cpu_result.q);

    let gpu_params = (preset.gpu)(p.dt, target, 2000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "[{}] N=100 ({label}) GPU FIRE did not reach target {target:e}: force_max={}",
            preset.name, gpu_result.force_max
        )
        .into());
    }

    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    let rel_force_diff =
        (cpu_result.force_max - gpu_result.force_max).abs() / cpu_result.force_max.max(1e-300);
    println!(
        "[{}] N=100 ({label}, target={target:e}): CPU {} iters force_max={:.6e}; GPU {} iters force_max={:.6e}; max|Q_cpu-Q_gpu|={:.3e}; rel force_max diff={:.3e}",
        preset.name, cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff, rel_force_diff
    );
    if cpu_result.iterations != gpu_result.iterations {
        return Err(format!(
            "[{}] N=100 ({label}): step count disagreement, CPU={} GPU={}",
            preset.name, cpu_result.iterations, gpu_result.iterations
        )
        .into());
    }
    let tol = 1e-6;
    if diff > tol {
        return Err(format!(
            "[{}] N=100 ({label}): GPU disagrees with CPU by {diff:e}, exceeding tolerance {tol:e}",
            preset.name
        )
        .into());
    }
    println!("[{}] N=100 ({label}) validation PASSED (max diff {diff:e} <= {tol:e}, steps match)", preset.name);
    Ok(())
}

fn phase_validate(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    for preset in &PRESETS {
        run_validation_n8(dev, preset)?;
    }
    for preset in &PRESETS {
        run_validation_n100(dev, preset, LITERAL_TARGET, "literal")?;
        run_validation_n100(dev, preset, SCALE_MATCHED_TARGET, "scale-matched")?;
    }
    println!("ALL VALIDATION PASSED");
    Ok(())
}

fn phase_roofline(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let bw_elements = 256usize * 1024 * 1024;
    let bandwidth_gbps = dev.measure_bandwidth(bw_elements, 20)?;
    println!(
        "roofline: measured achieved bandwidth (pure copy, {} MiB, 20 reps) = {bandwidth_gbps:.1} GB/s",
        bw_elements * 8 / (1024 * 1024)
    );
    let launch_overhead_us = dev.measure_launch_overhead(2000)? * 1e6;
    println!("roofline: mean kernel launch overhead (1-element launches, 2000 reps) = {launch_overhead_us:.2} us");

    let n = 100usize;
    let n_sites = n * n * n;
    let h2d_s = dev.measure_h2d_upload(n_sites * 5, 20)? * 1e6;
    println!(
        "roofline: mean host-to-device upload, N={n} field ({} MB, 20 reps) = {h2d_s:.1} us",
        n_sites * 5 * 8 / 1_000_000
    );

    // Bytes moved per FIRE iteration at N=100 (derived from the kernel
    // bodies in kernels.rs, not estimated): position_update 160n, two
    // half-kicks 240n total, trq2 48n, force 360n, reduce_fire 80n, fire_mix
    // 120n = 1008n bytes on a non-reset iteration (n = n_sites).
    let bytes_no_reset = 1008.0 * n_sites as f64;
    let floor_no_reset_ms = bytes_no_reset / (bandwidth_gbps * 1e9) * 1e3;
    let bytes_reset = 1088.0 * n_sites as f64;
    let floor_reset_ms = bytes_reset / (bandwidth_gbps * 1e9) * 1e3;
    println!(
        "roofline: derived floor at N={n}, 1008n bytes/iteration (no reset) = {floor_no_reset_ms:.3} ms; 1088n (reset) = {floor_reset_ms:.3} ms"
    );
    Ok(())
}

fn timed_run(
    dev: &Device,
    preset: &Preset,
    n: usize,
    target: f64,
    max_iterations: usize,
    q0_flat: &[f64],
    ldg: &LdgParams,
) -> Result<(f64, usize, f64, bool), Box<dyn std::error::Error>> {
    let gpu_params = (preset.gpu)(0.005, target, max_iterations);
    let t0 = Instant::now();
    let result = dev.fire_minimize(q0_flat, ldg, &gpu_params)?;
    let elapsed = t0.elapsed().as_secs_f64();
    let _ = n;
    Ok((elapsed, result.iterations, result.force_max, result.converged))
}

fn phase_time_tuned(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let preset = &PRESETS[1]; // volterra_tuned
    let n = 100usize;
    let (p, ldg) = setup(n);
    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let q0_flat = flatten(&q0);

    for (label, target) in [("literal (1e-3)", LITERAL_TARGET), ("scale-matched (2.09e-5)", SCALE_MATCHED_TARGET)] {
        // Untimed warm-up (pays first-touch cost once), then 3 timed repeats.
        let (_warm_t, warm_iters, _warm_fm, warm_ok) =
            timed_run(dev, preset, n, target, 2000, &q0_flat, &ldg)?;
        if !warm_ok {
            return Err(format!("[time-tuned] {label}: warm-up run did not reach target").into());
        }

        let mut times = Vec::with_capacity(3);
        let mut iters_seen = Vec::with_capacity(3);
        for rep in 0..3 {
            let (t, iters, force_max, ok) = timed_run(dev, preset, n, target, 2000, &q0_flat, &ldg)?;
            if !ok {
                return Err(format!("[time-tuned] {label} rep {rep}: did not reach target").into());
            }
            println!(
                "[volterra_tuned] GPU FIRE {label} rep={rep} N={n} steps={iters} force_max={force_max:.6e} wall={t:.4}s"
            );
            times.push(t);
            iters_seen.push(iters);
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[volterra_tuned] GPU FIRE {label} N={n} steps={:?} (warm-up steps={warm_iters}) wall min={min:.4}s mean={mean:.4}s max={max:.4}s spread={:.4}s",
            iters_seen, max - min
        );
    }
    Ok(())
}

fn phase_kernels(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = 100usize;
    let (p, ldg) = setup(n);
    let s0 = analytic_s0(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let q0_flat = flatten(&q0);
    let reps = 50usize;

    let t_split = dev.time_split_force(&q0_flat, &ldg, reps)? * 1e3;
    let t_fused_aos = dev.time_fused_aos_force(&q0_flat, &ldg, reps)? * 1e3;
    let t_fused_soa = dev.time_fused_soa_force(&q0_flat, &ldg, reps)? * 1e3;

    println!("[kernels] N={n}, {reps} reps each, mean ms/launch (no host round trip mid-loop):");
    println!("[kernels] split (trq2+force, current)      = {t_split:.4} ms");
    println!("[kernels] force_fused_aos (fused, AoS)      = {t_fused_aos:.4} ms  ({:+.1}% vs split)",
        100.0 * (t_fused_aos - t_split) / t_split);
    println!("[kernels] force_fused_soa (fused, SoA)      = {t_fused_soa:.4} ms  ({:+.1}% vs split)",
        100.0 * (t_fused_soa - t_split) / t_split);
    let _ = p;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let process_start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let phase = args.get(1).map(String::as_str).unwrap_or("all");

    let t_ctx0 = Instant::now();
    let dev = Device::new(0)?;
    let t_ctx = t_ctx0.elapsed().as_secs_f64();
    println!("phase: CUDA context + module load  {t_ctx:.4}s");

    match phase {
        "roofline" => phase_roofline(&dev)?,
        "validate" => phase_validate(&dev)?,
        "time-tuned" => {
            // Correctness gates speed: refuse to time if validation fails.
            phase_validate(&dev)?;
            phase_time_tuned(&dev)?;
        }
        "kernels" => phase_kernels(&dev)?,
        "all" => {
            phase_roofline(&dev)?;
            phase_validate(&dev)?;
            phase_time_tuned(&dev)?;
            phase_kernels(&dev)?;
        }
        other => {
            return Err(format!("unknown phase '{other}'; expected roofline|validate|time-tuned|kernels|all").into());
        }
    }

    let t_total = process_start.elapsed().as_secs_f64();
    println!("phase: total process wall-clock (args parse through exit)  {t_total:.4}s");
    Ok(())
}
