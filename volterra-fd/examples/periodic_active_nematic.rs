//! Active nematic on a flat torus, at the parameters of Mitchell et al.
//!
//! Drives two reproductions off one solver:
//!
//! - Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing
//!   active nematics", Phys. Rev. E 109, 014606 (2024). A square with periodic
//!   boundaries at `L = 100`, `lambda = 1`, `Re = 0.01`, `gamma_tilde = 50`,
//!   `C_tilde = 9`, active length swept over `1 <= ell_a <= 4.5`. The claims are
//!   that `ell_a = 3` settles into a periodic four-defect orbit while `ell_a = 1`
//!   stays chaotic, and that the periodic orbit mixes harder: dimensionless
//!   topological entropy `h_tilde = 1.66e-3` against `1.25e-3`.
//!
//! - Mitchell, Sabbir, Klein and Beller, "Modelling active nematics via the
//!   nematic locking principle", Soft Matter (2025), arXiv:2506.20996. The same
//!   groups on a `200 x 200` torus, `S_eq = 1`, comparing the standard model
//!   against the enhanced-locking one through the two director rotation rates.
//!   Its reported statistics are RMS `omega_A = 0.263`, `omega_F = 0.158` and
//!   medians `0.1490`, `0.0687` for standard Beris-Edwards, against RMS `0.249`,
//!   `0.605` and medians `0.168`, `7.14e-7` with enhanced locking.
//!
//! # Configuration
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `VP_CONVENTION` | `mitchell` | `mitchell` (`S_eq = sqrt 2`) or `locking` (`S_eq = 1`) |
//! | `VP_LX` | `100` | Square side, in lattice units |
//! | `VP_ELL_A` | `3.0` | Active length, in lattice units |
//! | `VP_DT` | `5e-5` | Timestep |
//! | `VP_STEPS` | `200000` | Steps to run |
//! | `VP_SAVE_EVERY` | `2000` | Steps between recorded observations |
//! | `VP_FRAME_EVERY` | `0` | Steps between `.npy` field frames; `0` writes none |
//! | `VP_SEED` | `1` | Random director seed |
//! | `VP_IC` | `random` | `random` directors, `uniform` for a nearly uniform field, `symrandom` for a random field invariant under the half-diagonal shift, `seeded` four-defect, or `fig2a` at the paper's own placement |
//! | `VP_UNIFORM_AMP` | `0.05` | Fluctuation of a `uniform` field, in units of pi |
//! | `VP_THETA0` | `0` | Far-field director angle of a seeded field, in units of pi |
//! | `VP_Q_INIT` | unset | Path to a `q_*.npy` frame to start from, for a continuation |
//! | `VP_STRESS` | `full` | `full` Beris-Edwards, or `giomi` for `-lambda H + [Q, H]` alone |
//! | `VP_RELAX_STEPS` | `0` | Passive steps (`zeta = 0`) before activity is switched on |
//! | `VP_LOCKING` | `0` | `1` switches on enhanced nematic locking |
//! | `VP_SIGMA` | `0.2` | Switch width, in units of `S_eq` |
//! | `VP_MAX_P_ITERS` | `50` | Pressure-Poisson iteration cap |
//! | `VP_LINES` | `0` | Material lines advected for the entropy measurement |
//! | `VP_LINE_FROM` | `0` | Step at which the lines are seeded |
//! | `VP_LINE_LEN` | `4.0` | Initial length of each seeded line |
//! | `VP_LINE_MAX_POINTS` | `100000` | Vertices at which a line stops refining and freezes |
//! | `VP_LINE_SEG` | `0.25` | Refinement tolerance, in lattice units |
//! | `VP_ROTATION` | `0` | `1` records the `omega_A` / `omega_F` statistics |
//! | `VP_TRACERS` | `0` | Passive tracers per side of a square lattice; `0` advects none |
//! | `VP_OUT` | `runs/periodic` | Output directory |
//!
//! # Output
//!
//! `config.json`, `stats.csv` (one row per observation) and `defects.csv` (one
//! row per defect per observation), plus `q_<step>.npy` frames when
//! `VP_FRAME_EVERY` is set.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rand::{RngExt, SeedableRng, rngs::StdRng};
use volterra_braid::{Defect, detect_defects};
use volterra_fd::{
    Dimensionless, Locking, MaterialLine, Params, StressModel, boundary::periodic_boundary,
    ic::{mitchell_figure_2a, mitchell_four_defect, seeded_q},
    locking::{rms_and_median, rotation_rates},
    ops::div_vector,
    step::{State, update_step_inner},
    stokes::subtract_p_avg,
    stretching::sample,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Random director everywhere, `theta` uniform on `[0, pi)`, and `u = 0`.
///
/// The initial condition both papers use.
fn random_director(q: &mut [f64], s0: f64, lx: usize, ly: usize, rng: &mut StdRng) {
    use std::f64::consts::PI;
    for x in 0..lx {
        for y in 0..ly {
            let theta: f64 = PI * rng.random::<f64>();
            let (s, c) = theta.sin_cos();
            q[(x * ly + y) * 2] = s0 * (c * c - 0.5);
            q[(x * ly + y) * 2 + 1] = s0 * (c * s);
        }
    }
}

/// A nearly uniform director field: one angle everywhere plus a small
/// fluctuation.
///
/// The second initial condition of Mitchell et al.'s Fig. 5, whose red curve
/// "is the entropy resulting from a nearly uniform initial director field" and
/// finds none of the periodic orbit the black curve does. The amplitude is in
/// units of pi, so `1.0` is the plain random field and the paper's case is a
/// few per cent.
fn uniform_director(q: &mut [f64], s0: f64, lx: usize, ly: usize, amp: f64,
                    rng: &mut StdRng) {
    use std::f64::consts::PI;
    let base = PI * rng.random::<f64>();
    for x in 0..lx {
        for y in 0..ly {
            let theta = base + amp * PI * (rng.random::<f64>() - 0.5);
            let (sn, cs) = theta.sin_cos();
            q[(x * ly + y) * 2] = s0 * (cs * cs - 0.5);
            q[(x * ly + y) * 2 + 1] = s0 * (cs * sn);
        }
    }
}

/// A random director field invariant under the half-diagonal translation
/// `(x, y) -> (x + lx/2, y + ly/2)`.
///
/// Mitchell et al.'s Fig. 2(a) state has exactly this symmetry: its two `-1/2`
/// defects sit at `(0, 0)` and `(L/2, L/2)` and its two `+1/2` defects at
/// `(0, L/2)` and `(L/2, 0)`, so each species is one orbit of the translation.
/// Every operator in the scheme is translation-equivariant on the periodic
/// lattice and the shift is a whole number of cells, so a field that starts
/// with the symmetry keeps it exactly, and the run is confined to the symmetry
/// class the target orbit lives in. A plain random field is not in that class
/// and reaches the defect-free state instead.
///
/// The construction takes the half `x < lx/2` at random and defines the other
/// half as its image, which is invariant because `y + ly/2` and `y - ly/2`
/// agree modulo `ly`.
fn symmetric_random_director(q: &mut [f64], s0: f64, lx: usize, ly: usize, rng: &mut StdRng) {
    use std::f64::consts::PI;
    assert!(lx % 2 == 0 && ly % 2 == 0, "the half-diagonal shift needs an even side");
    let (hx, hy) = (lx / 2, ly / 2);
    for x in 0..hx {
        for y in 0..ly {
            let theta: f64 = PI * rng.random::<f64>();
            let (s, c) = theta.sin_cos();
            let (qxx, qxy) = (s0 * (c * c - 0.5), s0 * (c * s));
            q[(x * ly + y) * 2] = qxx;
            q[(x * ly + y) * 2 + 1] = qxy;
            let yy = (y + hy) % ly;
            q[((x + hx) * ly + yy) * 2] = qxx;
            q[((x + hx) * ly + yy) * 2 + 1] = qxy;
        }
    }
}

/// One RK4 tracer step on the torus, the same integrator [`MaterialLine`] uses.
///
/// A tracer is a fluid element and nothing else: it never refines, never
/// saturates, and is four samples a step for the life of a run, which is what
/// lets the mixing panel of a film stay live where a material line has long
/// since stopped being resolved.
fn advect_tracers(pts: &mut [[f64; 2]], u: &[f64], lx: usize, ly: usize, dt: f64) {
    let (fx, fy) = (lx as f64, ly as f64);
    let wrap = |p: [f64; 2]| [p[0].rem_euclid(fx), p[1].rem_euclid(fy)];
    for p in pts.iter_mut() {
        let k1 = sample(u, *p, lx, ly);
        let k2 = sample(u, wrap([p[0] + 0.5 * dt * k1[0], p[1] + 0.5 * dt * k1[1]]), lx, ly);
        let k3 = sample(u, wrap([p[0] + 0.5 * dt * k2[0], p[1] + 0.5 * dt * k2[1]]), lx, ly);
        let k4 = sample(u, wrap([p[0] + dt * k3[0], p[1] + dt * k3[1]]), lx, ly);
        let vx = (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
        let vy = (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
        *p = wrap([p[0] + dt * vx, p[1] + dt * vy]);
    }
}

/// RMS speed over the whole domain.
fn rms_speed(u: &[f64], n: usize) -> f64 {
    let s: f64 = (0..n).map(|i| u[i * 2] * u[i * 2] + u[i * 2 + 1] * u[i * 2 + 1]).sum();
    (s / n as f64).sqrt()
}

/// Mean scalar order parameter, `S = 2 sqrt(Qxx^2 + Qxy^2)`.
fn mean_s(q: &[f64], n: usize) -> f64 {
    let s: f64 = (0..n)
        .map(|i| 2.0 * (q[i * 2] * q[i * 2] + q[i * 2 + 1] * q[i * 2 + 1]).sqrt())
        .sum();
    s / n as f64
}

fn main() -> std::io::Result<()> {
    let convention = std::env::var("VP_CONVENTION").unwrap_or_else(|_| "mitchell".into());
    let lx: usize = env_or("VP_LX", 100);
    let ell_a: f64 = env_or("VP_ELL_A", 3.0);
    let dt: f64 = env_or("VP_DT", 5e-5);
    let steps: usize = env_or("VP_STEPS", 200_000);
    let save_every: usize = env_or("VP_SAVE_EVERY", 2_000);
    let frame_every: usize = env_or("VP_FRAME_EVERY", 0);
    let seed: u64 = env_or("VP_SEED", 1);
    let use_locking: usize = env_or("VP_LOCKING", 0);
    let sigma: f64 = env_or("VP_SIGMA", 0.2);
    let max_p_iters: i64 = env_or("VP_MAX_P_ITERS", 50);
    let n_lines: usize = env_or("VP_LINES", 0);
    let line_from: usize = env_or("VP_LINE_FROM", 0);
    let line_len: f64 = env_or("VP_LINE_LEN", 4.0);
    let line_max_points: usize = env_or("VP_LINE_MAX_POINTS", 100_000);
    let line_seg: f64 = env_or("VP_LINE_SEG", 0.25);
    let want_rotation: usize = env_or("VP_ROTATION", 0);
    let n_tracers: usize = env_or("VP_TRACERS", 0);
    let ic = std::env::var("VP_IC").unwrap_or_else(|_| "random".into());
    // Mitchell et al. map the black curve of their Fig. 5 by continuation:
    // "The black curve uses an initial Q-field taken from the periodic state
    // at ell_a = 3". Loading a saved frame is that protocol.
    let q_init = std::env::var("VP_Q_INIT").ok();
    let relax_steps: usize = env_or("VP_RELAX_STEPS", 0);
    // The one seeded-field parameter neither paper states. A `+1/2` defect
    // self-propels along its own axis, so the far-field angle sets which way
    // each of the four starts moving and therefore which state the run reaches.
    let theta0: f64 = env_or::<f64>("VP_THETA0", 0.0) * std::f64::consts::PI;
    let uniform_amp: f64 = env_or("VP_UNIFORM_AMP", 0.05);
    let out = PathBuf::from(std::env::var("VP_OUT").unwrap_or_else(|_| "runs/periodic".into()));

    let ly = lx;
    let dims = match convention.as_str() {
        "locking" => Dimensionless::nematic_locking(ell_a),
        _ => Dimensionless::mitchell(ell_a),
    };
    let mut params = Params::from_dimensionless(
        lx, ly, dims, Dimensionless::MITCHELL_K, dt, max_p_iters,
    );
    if use_locking != 0 {
        params = params.with_locking(Locking { sigma });
    }
    let stress_name = std::env::var("VP_STRESS").unwrap_or_else(|_| "full".into());
    params = params.with_stress(match stress_name.as_str() {
        "giomi" => StressModel::Giomi,
        _ => StressModel::Full,
    });
    let bnd = periodic_boundary(lx, ly);
    let n = lx * ly;

    fs::create_dir_all(&out)?;
    let cfg = serde_json::json!({
        "convention": convention,
        "params": params,
        "dimensionless": dims,
        "ell_n": dims.ell_n(),
        "active_time": params.active_time(),
        "steps": steps,
        "save_every": save_every,
        "seed": seed,
        "ic": if q_init.is_some() { "continuation".to_string() } else { ic.clone() },
        "q_init": q_init,
        "relax_steps": relax_steps,
        "theta0": theta0,
        "lines": n_lines,
        "tracers": n_tracers,
    });
    fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap())?;

    println!(
        "periodic {lx}x{ly}  ell_a={:.4}  ell_n={:.4}  t_a={:.6}\n\
         K={:.1} eta={:.1} gamma={:.1} zeta={:.4} A={:.1} C={:.1} S_eq={:.6} lambda={}\n\
         locking={} sigma={sigma}  stress={stress_name}  dt={dt:e}  steps={steps}",
        params.active_length(),
        params.coherence_length(),
        params.active_time(),
        params.k_elastic, params.eta, params.gamma, params.zeta,
        params.a_landau, params.c_landau, params.s0, params.lambda,
        if use_locking != 0 { "on" } else { "off" },
    );

    let mut state = State::new(lx, ly);
    let mut rng = StdRng::seed_from_u64(seed);
    if let Some(path) = &q_init {
        let q = read_npy_2c(std::path::Path::new(path), lx, ly)?;
        state.q.copy_from_slice(&q);
        println!("  continued from {path}");
    } else {
    match ic.as_str() {
        // Mitchell et al. (2024) map the periodic orbit by continuation from a
        // state that already has two `+1/2` defects. A random field at the same
        // parameters competes with a defect-free stationary state, so the orbit
        // needs a field that starts with the right defects.
        "seeded" | "fig2a" => {
            let defects = if ic == "fig2a" {
                mitchell_figure_2a(lx, ly)
            } else {
                mitchell_four_defect(lx, ly)
            };
            state.q = seeded_q(&defects, lx, ly, params.s0, theta0)
                .expect("the four-defect arrangement has zero total charge");
            println!("  seeded {} defects, theta_0 = {:.4} rad", defects.len(), theta0);
        }
        "uniform" => {
            uniform_director(&mut state.q, params.s0, lx, ly, uniform_amp, &mut rng);
            println!("  nearly uniform directors, fluctuation {uniform_amp} pi");
        }
        "symrandom" => {
            symmetric_random_director(&mut state.q, params.s0, lx, ly, &mut rng);
            println!("  random directors, symmetric under the half-diagonal shift");
        }
        _ => random_director(&mut state.q, params.s0, lx, ly, &mut rng),
    }
    }

    // Relax the seeded cores against the free energy alone before the activity
    // is switched on: the protocol Head et al. (2026) and others use, and what
    // gives an analytically placed disclination its physical core profile.
    if relax_steps > 0 {
        let mut passive = params.clone();
        passive.zeta = 0.0;
        for _ in 0..relax_steps {
            update_step_inner(&mut state, &passive, &bnd, 1e-6);
        }
        state.u.iter_mut().for_each(|v| *v = 0.0);
        state.p.iter_mut().for_each(|v| *v = 0.0);
        println!("  relaxed {relax_steps} passive steps");
    }

    // Material lines, seeded as short segments at random positions and angles.
    let mut lines: Vec<MaterialLine> = Vec::new();
    let mut line_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9);

    // Passive tracers on a square lattice, one colour per initial column, which
    // is the picture a mixing rate is a number for.
    let mut tracers: Vec<[f64; 2]> = Vec::new();
    let mut tracer_col: Vec<usize> = Vec::new();
    for i in 0..n_tracers {
        for j in 0..n_tracers {
            let f = |k: usize, l: usize| (k as f64 + 0.5) * l as f64 / n_tracers as f64;
            tracers.push([f(i, lx), f(j, ly)]);
            tracer_col.push(i);
        }
    }

    let mut stats = BufWriter::new(File::create(out.join("stats.csv"))?);
    writeln!(
        stats,
        "step,t,rms_u,max_div_u,n_plus,n_minus,mean_S,\
         omega_a_rms,omega_a_median,omega_f_rms,omega_f_median,line_len_mean,line_points"
    )?;
    let mut defects_out = BufWriter::new(File::create(out.join("defects.csv"))?);
    writeln!(defects_out, "step,t,x,y,charge")?;

    let mut div_scratch = vec![0.0; n];
    let t_start = Instant::now();
    let mut last_report = Instant::now();

    for step in 0..=steps {
        if step > 0 {
            update_step_inner(&mut state, &params, &bnd, 1e-6);
            // Only grad p enters the velocity update, so the gauge on a torus is
            // free; pinning the mean stops it drifting and keeps the
            // relative-change convergence test in `relax_pressure` meaningful.
            subtract_p_avg(&mut state.p, &bnd);

            if step >= line_from {
                if lines.is_empty() && n_lines > 0 {
                    for _ in 0..n_lines {
                        let cx: f64 = line_rng.random::<f64>() * lx as f64;
                        let cy: f64 = line_rng.random::<f64>() * ly as f64;
                        let th: f64 = line_rng.random::<f64>() * std::f64::consts::PI;
                        let (s, c) = th.sin_cos();
                        let h = 0.5 * line_len;
                        lines.push(
                            MaterialLine::segment(
                                [cx - h * c, cy - h * s],
                                [cx + h * c, cy + h * s],
                                lx, ly, 16,
                            )
                            .with_limits(line_seg, line_max_points),
                        );
                    }
                }
                let t = step as f64 * dt;
                for line in lines.iter_mut() {
                    line.advect(&state.u, ly, dt, step, t);
                }
            }
            if !tracers.is_empty() {
                advect_tracers(&mut tracers, &state.u, lx, ly, dt);
            }
        }

        if step % save_every != 0 {
            continue;
        }
        let t = step as f64 * dt;

        if !state.q.iter().all(|v| v.is_finite()) {
            eprintln!("run diverged at step {step}; stopping");
            break;
        }

        div_vector(&state.u, &mut div_scratch, &bnd);
        let max_div = div_scratch.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

        let qxx: Vec<f64> = (0..n).map(|i| state.q[i * 2]).collect();
        let qxy: Vec<f64> = (0..n).map(|i| state.q[i * 2 + 1]).collect();
        // The reference tracker's threshold scales with the Q amplitude; the
        // Jacobian is quartic in Q, so a fixed fraction of `S_eq^4` transfers
        // between the two `S` conventions unchanged.
        let thresh = 0.05 * params.s0.powi(4);
        let ds: Vec<Defect> = detect_defects(&qxx, &qxy, lx, ly, thresh, &bnd.inside);
        let n_plus = ds.iter().filter(|d| d.charge > 0).count();
        let n_minus = ds.len() - n_plus;
        for d in &ds {
            writeln!(defects_out, "{step},{t:.6},{:.4},{:.4},{}", d.pos[0], d.pos[1], d.charge)?;
        }

        let (wa_rms, wa_med, wf_rms, wf_med) = if want_rotation != 0 {
            let r = rotation_rates(
                &state.u, &state.q, &state.h, params.gamma, params.s0,
                params.locking, &bnd,
            );
            let (ar, am) = rms_and_median(&r.omega_a, &bnd);
            let (fr, fm) = rms_and_median(&r.omega_f, &bnd);
            (ar, am, fr, fm)
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };

        let (lmean, lpts) = if lines.is_empty() {
            (f64::NAN, 0usize)
        } else {
            (
                lines.iter().map(|l| l.length()).sum::<f64>() / lines.len() as f64,
                lines.iter().map(|l| l.len()).sum::<usize>() / lines.len(),
            )
        };

        writeln!(
            stats,
            "{step},{t:.6},{:.8e},{:.4e},{n_plus},{n_minus},{:.6},\
             {wa_rms:.6e},{wa_med:.6e},{wf_rms:.6e},{wf_med:.6e},{lmean:.6e},{lpts}",
            rms_speed(&state.u, n),
            max_div,
            mean_s(&state.q, n),
        )?;
        stats.flush()?;
        defects_out.flush()?;
        write_entropy(&out, &lines, dt, save_every, params.active_time(), false)?;

        // Frames land on the observation cadence, so `VP_FRAME_EVERY` is only
        // ever honoured at multiples of `VP_SAVE_EVERY`.
        if frame_every > 0 && step % frame_every == 0 {
            write_npy_2c(&out.join(format!("q_{step:08}.npy")), &state.q, lx, ly)?;
            write_npy_2c(&out.join(format!("u_{step:08}.npy")), &state.u, lx, ly)?;
            write_npy_1c(&out.join(format!("p_{step:08}.npy")), &state.p, lx, ly)?;
            // The two rotation rates of arXiv:2506.20996, as fields. `state.h`
            // is the molecular field the step just used, one Q update behind the
            // saved `q`, which at this cadence is a difference below the line
            // width. The same lag is in the `stats.csv` statistics.
            let r = rotation_rates(
                &state.u, &state.q, &state.h, params.gamma, params.s0,
                params.locking, &bnd,
            );
            write_npy_1c(&out.join(format!("wa_{step:08}.npy")), &r.omega_a, lx, ly)?;
            write_npy_1c(&out.join(format!("wf_{step:08}.npy")), &r.omega_f, lx, ly)?;
            if !tracers.is_empty() {
                let mut f = BufWriter::new(File::create(
                    out.join(format!("tracer_{step:08}.csv")),
                )?);
                writeln!(f, "col,x,y")?;
                for (c, pt) in tracer_col.iter().zip(tracers.iter()) {
                    writeln!(f, "{c},{:.4},{:.4}", pt[0], pt[1])?;
                }
            }
            if !lines.is_empty() {
                let mut f = BufWriter::new(File::create(
                    out.join(format!("line_{step:08}.csv")),
                )?);
                writeln!(f, "line,x,y")?;
                for (i, line) in lines.iter().enumerate() {
                    for pt in &line.points {
                        writeln!(f, "{i},{:.4},{:.4}", pt[0], pt[1])?;
                    }
                }
            }
        }

        if last_report.elapsed().as_secs_f64() >= 20.0 || step == steps {
            let el = t_start.elapsed().as_secs_f64();
            println!(
                "  step {step}/{steps}  t={t:.4}  +{n_plus}/-{n_minus}  \
                 rms_u={:.4e}  {:.0} steps/s  {el:.0}s",
                rms_speed(&state.u, n),
                step as f64 / el.max(1e-9),
            );
            last_report = Instant::now();
        }
    }

    // The final write. Every observation already wrote one, so a run that is
    // stopped part way still leaves both files on disk.
    write_entropy(&out, &lines, dt, save_every, params.active_time(), true)?;

    println!("done in {:.1}s -> {}", t_start.elapsed().as_secs_f64(), out.display());
    Ok(())
}

/// Read back a `(lx, ly, 2)` float64 C-order `.npy` this example wrote.
///
/// Deliberately narrow: it accepts the header this file writes and refuses
/// anything else, rather than parsing the format in general and silently
/// reinterpreting a `f32` or a Fortran-ordered array as the state.
fn read_npy_2c(path: &std::path::Path, lx: usize, ly: usize) -> std::io::Result<Vec<f64>> {
    use std::io::{Error, ErrorKind, Read};
    let mut f = std::io::BufReader::new(File::open(path)?);
    let mut magic = [0u8; 10];
    f.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(Error::new(ErrorKind::InvalidData, "not a .npy file"));
    }
    let hlen = u16::from_le_bytes([magic[8], magic[9]]) as usize;
    let mut header = vec![0u8; hlen];
    f.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header).to_string();
    let want = format!("'descr': '<f8', 'fortran_order': False, 'shape': ({lx}, {ly}, 2)");
    if !header.contains(&want) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("header is {}, wanted {want}", header.trim()),
        ));
    }
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let n = lx * ly * 2;
    if buf.len() != n * 8 {
        return Err(Error::new(ErrorKind::InvalidData, "wrong payload length"));
    }
    Ok((0..n)
        .map(|i| f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect())
}


/// Write `line_lengths.csv` and `entropy.json` for the lines as they stand.
///
/// Called at every observation, not only at the end. A material-line entropy is
/// a fit to a history that is complete the moment the line saturates, so there
/// is nothing to wait for, and a run that is stopped part way is exactly as
/// informative about mixing as one that reaches its last step. Writing only on
/// the way out threw that away.
///
/// Both files are rewritten in full each time, which is a few thousand rows.
fn write_entropy(
    out: &std::path::Path,
    lines: &[MaterialLine],
    dt: f64,
    save_every: usize,
    t_a: f64,
    verbose: bool,
) -> std::io::Result<()> {
    let Some(first) = lines.first() else { return Ok(()) };
    if first.history.is_empty() {
        return Ok(());
    }
    let t_lo = first.history.first().map(|p| p.0).unwrap_or(0.0);
    let t_hi = first
        .saturated_at
        .map(|s| s as f64 * dt)
        .unwrap_or_else(|| first.history.last().map(|p| p.0).unwrap_or(0.0));
    // Discard the first fifth, where the line is still adjusting to the flow.
    let t0 = t_lo + 0.2 * (t_hi - t_lo);
    let mut hs = Vec::new();
    let mut lengths = BufWriter::new(File::create(out.join("line_lengths.csv"))?);
    writeln!(lengths, "line,t,length")?;
    for (i, line) in lines.iter().enumerate() {
        for &(t, l) in line.history.iter().step_by(save_every.max(1)) {
            writeln!(lengths, "{i},{t:.6},{l:.6e}")?;
        }
        if let Some(f) = line.fit(t0, t_hi) {
            if verbose {
                println!(
                    "  line {i}: h = {:.5} +- {:.5}  (r2 = {:.5}, n = {}, window ends t = {:.4})",
                    f.h, f.stderr, f.r2, f.n, t_hi
                );
            }
            hs.push(f.h);
        }
    }
    lengths.flush()?;
    if hs.is_empty() {
        return Ok(());
    }
    let mean = hs.iter().sum::<f64>() / hs.len() as f64;
    let sem = if hs.len() > 1 {
        (hs.iter().map(|h| (h - mean).powi(2)).sum::<f64>()
            / ((hs.len() - 1) * hs.len()) as f64)
            .sqrt()
    } else {
        f64::NAN
    };
    if verbose {
        println!(
            "\n  h = {mean:.4} +- {sem:.4} per unit integration time\n  \
             h_tilde = h t_a = {:.4e} +- {:.1e}   (t_a = {t_a:.6})",
            mean * t_a,
            sem * t_a
        );
    }
    fs::write(
        out.join("entropy.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "h": mean, "h_sem": sem, "t_a": t_a,
            "h_tilde": mean * t_a, "h_tilde_sem": sem * t_a,
            "fit_window": [t0, t_hi], "per_line": hs,
            "saturated": first.saturated_at.is_some(),
        }))
        .unwrap(),
    )
}

/// Minimal `.npy` writer for a `(lx, ly)` float64 C-order array.
fn write_npy_1c(path: &std::path::Path, data: &[f64], lx: usize, ly: usize) -> std::io::Result<()> {
    write_npy(path, data, &format!("({lx}, {ly})"))
}

/// Minimal `.npy` writer for a `(lx, ly, 2)` float64 C-order array.
fn write_npy_2c(path: &std::path::Path, data: &[f64], lx: usize, ly: usize) -> std::io::Result<()> {
    write_npy(path, data, &format!("({lx}, {ly}, 2)"))
}

fn write_npy(path: &std::path::Path, data: &[f64], shape: &str) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}");
    let mut h = header.into_bytes();
    // Total header length (magic 6 + version 2 + len 2 + header) padded to 64.
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(b' ');
    }
    h.push(b'\n');
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(h.len() as u16).to_le_bytes())?;
    f.write_all(&h)?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
