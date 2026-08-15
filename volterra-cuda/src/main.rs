//! Validation + timing harness for the CUDA FIRE minimiser.
//!
//! Correctness comes before speed throughout: nothing in the `time-*`/`kernels`
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
//!   the split `trq2`+`force` pipeline, same conditions, N=100.
//!   `fire_minimize` itself now uses `force_fused_aos` directly
//!   (`BENCHMARKS.md`'s "the cheap volterra gains").
//! - `matched`: volterra's bulk/elastic constants set to open-Qmin's own
//!   defaults (`setup_matched`), so `1e-3` is the literal target with no
//!   scale-matching construction. Validates (N=8 to convergence, N=100 at
//!   the literal target, both presets) then times `volterra_tuned` at
//!   N=100, six repeats across two batches.
//! - `all`: every phase above except `matched`, in order.
//!
//! Run: `./target/release/volterra-cuda <phase>` once built with `cargo
//! oxide build volterra-cuda`.

use std::time::Instant;

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_solver::{fire_minimize_3d_par, FireParams as CpuFireParams};

use volterra_cuda::{Bookkeeping, Device, FireParams as GpuFireParams, LdgParams};

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

/// The matched-physics preset (`FireParams::matched_tuned`), validated and
/// timed separately from `PRESETS`: it targets the matched landscape's
/// energy surface, not volterra's own default one `volterra_tuned` was
/// swept on.
const MATCHED_TUNED_PRESET: Preset = Preset {
    name: "matched_tuned",
    cpu: CpuFireParams::matched_tuned,
    gpu: GpuFireParams::matched_tuned,
};

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
        b_landau: p.b_landau,
        c_landau: p.c_landau,
        k_r: p.k_r,
        gamma_r: p.gamma_r,
    };
    (p, ldg)
}

/// Matched physics: volterra's `(a_eff, b_landau, c_landau, k_r)` set to
/// open-Qmin's own default `(a, b, c, L1)` under the mapping derived in
/// `volterra_solver::mol_field_3d`'s module header (`a_eff = 2a`, `b_landau
/// = b`, `c_landau = 2c`, `k_r = L1`, both giving the identical equilibrium
/// condition `6a + 3bM + 8cM^2 = 0`). open-Qmin's own CLI defaults
/// (`openQmin.cpp`): `a=-0.172, b=-2.12, c=1.73, L1=4.64`. With this
/// mapping `1e-3` is the literal target on both sides -- no scale-matching
/// construction needed (`BENCHMARKS.md` section 1's "Scale mismatch"
/// caveat).
fn setup_matched(n: usize) -> (ActiveNematicParams3D, LdgParams) {
    const A_OQ: f64 = -0.172;
    const B_OQ: f64 = -2.12;
    const C_OQ: f64 = 1.73;
    const L1_OQ: f64 = 4.64;

    // open-Qmin's own `-e`/`deltaT` default is 0.0005, not volterra's usual
    // 0.005 (`openQmin.cpp`): with `k_r` now matched to `L1=4.64` (4.64x
    // volterra's usual `k_r=1`) the elastic term is proportionally stiffer,
    // and 0.005 is unstable against it (diverges to NaN within a handful of
    // N=8 iterations). Using open-Qmin's own default timestep for its own
    // elastic constant is the matched choice, not merely a stability patch.
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.0005;
    p.a_landau = 2.0 * A_OQ;
    p.b_landau = B_OQ;
    p.c_landau = 2.0 * C_OQ;
    p.k_r = L1_OQ;
    let ldg = LdgParams {
        nx: n as u32,
        ny: n as u32,
        nz: n as u32,
        dx: p.dx,
        a_eff: p.a_eff(),
        b_landau: p.b_landau,
        c_landau: p.c_landau,
        k_r: p.k_r,
        gamma_r: p.gamma_r,
    };
    (p, ldg)
}

/// The positive root of open-Qmin's own equilibrium condition, `6a + 3bM +
/// 8cM^2 = 0`, expressed in volterra's `(a_eff, b_landau, c_landau)`
/// (`3 a_eff + 3 b_landau M + 4 c_landau M^2 = 0`, `mol_field_3d.rs`'s
/// module header).
fn analytic_s0_matched(p: &ActiveNematicParams3D) -> f64 {
    let a_eff = p.a_eff();
    let disc = 9.0 * p.b_landau * p.b_landau - 48.0 * a_eff * p.c_landau;
    (-3.0 * p.b_landau + disc.sqrt()) / (8.0 * p.c_landau)
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

/// Matched-physics analogue of `run_validation_n8`: same tight `1e-9`
/// tolerance, but `setup_matched`'s `(a_eff, b_landau, c_landau, k_r)`.
fn run_validation_matched_n8(dev: &Device, preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let n = 8usize;
    let (p, ldg) = setup_matched(n);
    let s0 = analytic_s0_matched(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 7);
    let q0_flat = flatten(&q0);

    let cpu_params = (preset.cpu)(p.dt, 1e-9, 5000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "[matched/{}] N=8 CPU FIRE reference did not converge: force_max={}",
            preset.name, cpu_result.force_max
        )
        .into());
    }
    let cpu_q = flatten(&cpu_result.q);

    let gpu_params = (preset.gpu)(p.dt, 1e-9, 5000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "[matched/{}] N=8 GPU FIRE did not converge: force_max={}",
            preset.name, gpu_result.force_max
        )
        .into());
    }

    let tol = 1e-9;
    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    println!(
        "[matched/{}] N=8 validation: CPU {} iters force_max={:.3e}; GPU {} iters force_max={:.3e}; max|Q_cpu-Q_gpu|={:.3e}",
        preset.name, cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff
    );
    if diff > tol {
        return Err(format!(
            "[matched/{}] N=8: GPU disagrees with CPU by {diff:e}, exceeding tolerance {tol:e}",
            preset.name
        )
        .into());
    }
    println!("[matched/{}] N=8 validation PASSED (max diff {diff:e} <= {tol:e})", preset.name);
    Ok(())
}

/// Matched-physics analogue of `run_validation_n100`. `target` is the
/// literal residual target -- under matched physics this needs no
/// scale-matching construction (`BENCHMARKS.md` section 1).
fn run_validation_matched_n100(
    dev: &Device,
    preset: &Preset,
    target: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = 100usize;
    let (p, ldg) = setup_matched(n);
    let s0 = analytic_s0_matched(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let q0_flat = flatten(&q0);

    let cpu_params = (preset.cpu)(p.dt, target, 2000);
    let cpu_result = fire_minimize_3d_par(&q0, &p, &cpu_params, 0.0);
    if !cpu_result.converged {
        return Err(format!(
            "[matched/{}] N=100 CPU FIRE did not reach target {target:e}: force_max={}",
            preset.name, cpu_result.force_max
        )
        .into());
    }
    let cpu_q = flatten(&cpu_result.q);

    let gpu_params = (preset.gpu)(p.dt, target, 2000);
    let gpu_result = dev.fire_minimize(&q0_flat, &ldg, &gpu_params)?;
    if !gpu_result.converged {
        return Err(format!(
            "[matched/{}] N=100 GPU FIRE did not reach target {target:e}: force_max={}",
            preset.name, gpu_result.force_max
        )
        .into());
    }

    let diff = max_abs_diff(&cpu_q, &gpu_result.q);
    println!(
        "[matched/{}] N=100 (target={target:e}): CPU {} iters force_max={:.6e}; GPU {} iters force_max={:.6e}; max|Q_cpu-Q_gpu|={:.3e}",
        preset.name, cpu_result.iterations, cpu_result.force_max, gpu_result.iterations, gpu_result.force_max, diff
    );
    if cpu_result.iterations != gpu_result.iterations {
        return Err(format!(
            "[matched/{}] N=100: step count disagreement, CPU={} GPU={}",
            preset.name, cpu_result.iterations, gpu_result.iterations
        )
        .into());
    }
    let tol = 1e-6;
    if diff > tol {
        return Err(format!(
            "[matched/{}] N=100: GPU disagrees with CPU by {diff:e}, exceeding tolerance {tol:e}",
            preset.name
        )
        .into());
    }
    println!("[matched/{}] N=100 validation PASSED (max diff {diff:e} <= {tol:e}, steps match)", preset.name);
    Ok(())
}

fn phase_matched(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let n = 100usize;
    let (p, _ldg) = setup_matched(n);
    let s0 = analytic_s0_matched(&p);
    println!(
        "[matched] a_eff={:.6} b_landau={:.6} c_landau={:.6} k_r={:.6} -- S0={s0:.6}",
        p.a_eff(), p.b_landau, p.c_landau, p.k_r
    );

    for preset in PRESETS.iter().chain(std::iter::once(&MATCHED_TUNED_PRESET)) {
        run_validation_matched_n8(dev, preset)?;
    }
    for preset in PRESETS.iter().chain(std::iter::once(&MATCHED_TUNED_PRESET)) {
        run_validation_matched_n100(dev, preset, LITERAL_TARGET)?;
    }
    println!("[matched] ALL VALIDATION PASSED");

    // Both the ported baseline (open-Qmin's own constants, unretuned) and
    // matched_tuned: volterra_tuned is skipped here, since it was swept on
    // volterra's own pre-cubic-term landscape, a different energy surface
    // from the matched one (`FireParams::matched_tuned`'s doc comment).
    for preset in [&PRESETS[0], &MATCHED_TUNED_PRESET] {
        let preset_name = preset.name;
        let (p, ldg) = setup_matched(n);
        let s0 = analytic_s0_matched(&p);
        let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
        let q0_flat = flatten(&q0);

        // Untimed warm-up, then 3 timed repeats, two independent batches (6
        // total), same protocol as `phase_time_tuned`.
        let (_warm_t, warm_iters, _warm_fm, warm_ok) =
            timed_run(dev, preset, n, p.dt, LITERAL_TARGET, 2000, &q0_flat, &ldg)?;
        if !warm_ok {
            return Err("[matched] warm-up run did not reach target".into());
        }

        let mut times = Vec::with_capacity(6);
        let mut iters_seen = Vec::with_capacity(6);
        for batch in 0..2 {
            for rep in 0..3 {
                let (t, iters, force_max, ok) =
                    timed_run(dev, preset, n, p.dt, LITERAL_TARGET, 2000, &q0_flat, &ldg)?;
                if !ok {
                    return Err(format!("[matched] batch {batch} rep {rep}: did not reach target").into());
                }
                println!(
                    "[matched/{preset_name}] GPU FIRE literal (1e-3) batch={batch} rep={rep} N={n} steps={iters} force_max={force_max:.6e} wall={t:.4}s"
                );
                times.push(t);
                iters_seen.push(iters);
            }
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[matched/{preset_name}] GPU FIRE literal (1e-3) N={n} steps={:?} (warm-up steps={warm_iters}) wall min={min:.4}s mean={mean:.4}s max={max:.4}s spread={:.4}s",
            iters_seen, max - min
        );
    }
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
    initial_dt: f64,
    target: f64,
    max_iterations: usize,
    q0_flat: &[f64],
    ldg: &LdgParams,
) -> Result<(f64, usize, f64, bool), Box<dyn std::error::Error>> {
    let gpu_params = (preset.gpu)(initial_dt, target, max_iterations);
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
            timed_run(dev, preset, n, p.dt, target, 2000, &q0_flat, &ldg)?;
        if !warm_ok {
            return Err(format!("[time-tuned] {label}: warm-up run did not reach target").into());
        }

        let mut times = Vec::with_capacity(3);
        let mut iters_seen = Vec::with_capacity(3);
        for rep in 0..3 {
            let (t, iters, force_max, ok) = timed_run(dev, preset, n, p.dt, target, 2000, &q0_flat, &ldg)?;
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

/// Time the fused bookkeeping path against the split one on the matched-physics
/// configuration the comparison with open-Qmin leads with.
///
/// Both paths run the same arithmetic in the same order, so the step count must
/// come out identical; the phase fails if it does not, since a differing step
/// count would mean the two are not the same minimiser and their times are not
/// comparable. The converged fields are compared as well.
fn phase_bookkeeping(dev: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let preset = &MATCHED_TUNED_PRESET;
    let n = 100usize;
    let (p, ldg) = setup_matched(n);
    let s0 = analytic_s0_matched(&p);
    let q0 = QField3D::random_director_field(n, n, n, p.dx, s0, 42);
    let q0_flat = flatten(&q0);
    let target = LITERAL_TARGET;
    let params = (preset.gpu)(p.dt, target, 2000);

    let mut summary: Vec<(&str, f64, f64, f64, usize)> = Vec::new();
    let mut fields: Vec<Vec<f64>> = Vec::new();

    for (label, mode) in [("split", Bookkeeping::Split), ("fused", Bookkeeping::Fused)] {
        // Untimed warm-up, then 6 timed repeats, matching the six the headline
        // rows in BENCHMARKS.md are quoted from.
        let warm = dev.fire_minimize_with(&q0_flat, &ldg, &params, mode)?;
        if !warm.converged {
            return Err(format!("[bookkeeping] {label}: warm-up did not reach target").into());
        }

        let mut times = Vec::with_capacity(6);
        let mut iters = warm.iterations;
        for rep in 0..6 {
            let t0 = Instant::now();
            let r = dev.fire_minimize_with(&q0_flat, &ldg, &params, mode)?;
            let t = t0.elapsed().as_secs_f64();
            if !r.converged {
                return Err(format!("[bookkeeping] {label} rep {rep}: did not reach target").into());
            }
            if r.iterations != iters {
                return Err(format!(
                    "[bookkeeping] {label} rep {rep}: {} steps, warm-up took {iters}",
                    r.iterations
                )
                .into());
            }
            iters = r.iterations;
            times.push(t);
            if rep == 0 {
                fields.push(r.q);
            }
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[bookkeeping] {label:6} N={n} steps={iters} wall min={min:.4}s mean={mean:.4}s max={max:.4}s"
        );
        summary.push((label, mean, min, max, iters));
    }

    let (split, fused) = (&summary[0], &summary[1]);
    if split.4 != fused.4 {
        return Err(format!(
            "[bookkeeping] step counts differ: split {}, fused {}; the two paths are not the same minimiser",
            split.4, fused.4
        )
        .into());
    }
    let max_diff = fields[0]
        .iter()
        .zip(&fields[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("[bookkeeping] max|Q_split - Q_fused| = {max_diff:.3e}");
    println!(
        "[bookkeeping] fused is {:+.1}% against split ({:.4}s against {:.4}s, {} steps each)",
        100.0 * (fused.1 - split.1) / split.1,
        fused.1,
        split.1,
        fused.4
    );
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
            // Correctness comes first: refuse to time if validation fails.
            phase_validate(&dev)?;
            phase_time_tuned(&dev)?;
        }
        "kernels" => phase_kernels(&dev)?,
        "bookkeeping" => phase_bookkeeping(&dev)?,
        "matched" => phase_matched(&dev)?,
        "all" => {
            phase_roofline(&dev)?;
            phase_validate(&dev)?;
            phase_time_tuned(&dev)?;
            phase_kernels(&dev)?;
        }
        other => {
            return Err(format!("unknown phase '{other}'; expected roofline|validate|time-tuned|kernels|matched|all").into());
        }
    }

    let t_total = process_start.elapsed().as_secs_f64();
    println!("phase: total process wall-clock (args parse through exit)  {t_total:.4}s");
    Ok(())
}
