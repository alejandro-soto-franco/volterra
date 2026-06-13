//! `cgpo_fd` — finite-difference driver for the nephroid-confined CGPO solver.
//!
//! All configuration is via environment variables (defaults in parentheses):
//!
//! | Variable          | Default                                      | Description                        |
//! |-------------------|----------------------------------------------|------------------------------------|
//! | `CGPO_LX`         | 100                                          | Grid side length (square grid)     |
//! | `CGPO_ALS`        | 1                                            | Active length scale (pixels)       |
//! | `CGPO_NCL`        | 9                                            | Nematic correlation length (px)    |
//! | `CGPO_DT`         | 1e-4                                         | Time step                          |
//! | `CGPO_MAX_P_ITERS`| 50                                           | Max pressure Jacobi iters per step |
//! | `CGPO_MAX_STEPS`  | 400000                                       | Total simulation steps             |
//! | `CGPO_SAVE_EVERY` | 1000                                         | Steps between frame saves          |
//! | `CGPO_OUT`        | /mnt/ASF-EX1/volterra-cgpo-nephroid          | Root output directory              |
//! | `CGPO_SEED`       | 0                                            | RNG seed (u64) for IC              |
//! | `CGPO_THETA_IC`   | (unset)                                      | Path to flat θ grid (optional)     |

use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;

use volterra_cgpo::{
    boundary::nephroid_boundary,
    step::{update_step, State},
    Params,
};

// ---------------------------------------------------------------------------
// env helpers
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

// ---------------------------------------------------------------------------
// IC helpers
// ---------------------------------------------------------------------------

/// Flat index: scalar (x*ly + y).
#[inline(always)]
fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}

/// Flat index: 2-component vector ((x*ly + y)*2 + c).
#[inline(always)]
fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}

/// Initialise Q from a theta grid (x*ly+y order, both inside and outside cells).
///
/// `initialize_Q_from_θ` in the Python:
/// ```python
/// Q[:,:,0] = S0 * (cos(θ)^2 - 0.5)
/// Q[:,:,1] = S0 * (cos(θ) * sin(θ))
/// ```
fn init_q_from_theta(q: &mut [f64], theta: &[f64], s0: f64, lx: usize, ly: usize) {
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let t = theta[idx];
            let cos_t = t.cos();
            let sin_t = t.sin();
            q[vi(x, y, ly, 0)] = s0 * (cos_t * cos_t - 0.5);
            q[vi(x, y, ly, 1)] = s0 * (cos_t * sin_t);
        }
    }
}

/// Generate random theta IC, matching Python:
/// ```python
/// theta_initial = 1.0 * π * np.random.random((Lx, Ly))
/// ```
/// (fully random uniform angles in [0, π); then Q from theta, applied only at
/// interior cells — outside cells are zeroed since theta_mask zeros them).
///
/// We replicate the mask: theta is non-zero only for inside cells.
fn random_theta_ic(
    q: &mut [f64],
    s0: f64,
    lx: usize,
    ly: usize,
    inside: &[bool],
    rng: &mut StdRng,
) {
    use std::f64::consts::PI;
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let (qxx, qxy) = if inside[idx] {
                let theta: f64 = PI * rng.random::<f64>();
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                (s0 * (cos_t * cos_t - 0.5), s0 * (cos_t * sin_t))
            } else {
                (0.0, 0.0)
            };
            q[vi(x, y, ly, 0)] = qxx;
            q[vi(x, y, ly, 1)] = qxy;
        }
    }
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Write a 2-column text file in numpy savetxt default format (%.18e).
///
/// Column 0: arr[..][0] components (e.g. Q_xx or u_x) in x*ly+y order.
/// Column 1: arr[..][1] components (e.g. Q_xy or u_y) in x*ly+y order.
///
/// `arr` is a 2-component field: index `(x*ly + y)*2 + c`.
fn write_2col_txt(path: &Path, arr: &[f64], n: usize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    for i in 0..n {
        writeln!(f, "{:.18e}  {:.18e}", arr[i * 2], arr[i * 2 + 1])?;
    }
    Ok(())
}

/// Format step count with zero-padding to 10 digits (matching Python:
/// `f'_{stepcount:10d}'.replace(' ','0')`).
fn step_name(step: usize) -> String {
    format!("{:010}", step)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // --- read config ---
    let lx = env_usize("CGPO_LX", 100);
    let als = env_usize("CGPO_ALS", 1);
    let ncl = env_usize("CGPO_NCL", 9);
    let dt = env_f64("CGPO_DT", 1e-4);
    let max_p_iters = env_usize("CGPO_MAX_P_ITERS", 50);
    let max_steps = env_usize("CGPO_MAX_STEPS", 400_000);
    let save_every = env_usize("CGPO_SAVE_EVERY", 1_000);
    let out_root = env_string("CGPO_OUT")
        .unwrap_or_else(|| "/mnt/ASF-EX1/volterra-cgpo-nephroid".to_string());
    let seed = env_u64("CGPO_SEED", 0);
    let theta_ic_path = env_string("CGPO_THETA_IC");

    // --- derived ---
    let ly = lx; // square grid
    let n = lx * ly;

    println!("cgpo_fd: lx={lx} als={als} ncl={ncl} dt={dt} max_p_iters={max_p_iters}");
    println!("  max_steps={max_steps} save_every={save_every}");
    println!("  out_root={out_root}  seed={seed}");

    // --- build params ---
    let params = Params::new(lx, als, ncl, dt, max_p_iters);
    println!(
        "  k_elastic={:.4e} zeta={:.4e} c_landau={:.4e} s0={:.6} eta={:.4e}",
        params.k_elastic, params.zeta, params.c_landau, params.s0, params.eta
    );

    // --- build boundary ---
    println!("Building nephroid boundary (lx={lx})…");
    let t_bnd = Instant::now();
    let boundary = nephroid_boundary(lx, ly);
    let n_interior = boundary.interior_count();
    println!("  boundary built in {:.2}s — {n_interior} interior cells", t_bnd.elapsed().as_secs_f64());

    // --- output directories ---
    let run_label = format!("als_{als}_ncl_{ncl}");
    let run_dir = Path::new(&out_root).join(&run_label);
    let q_dir = run_dir.join("Q");
    let u_dir = run_dir.join("u");
    fs::create_dir_all(&q_dir).expect("could not create Q output dir");
    fs::create_dir_all(&u_dir).expect("could not create u output dir");

    // --- allocate state ---
    let mut state = State::new(lx, ly);

    // --- initial condition ---
    if let Some(ref path) = theta_ic_path {
        // Load flat theta grid from text file (one value per line, x*ly+y order)
        println!("Loading theta IC from {path}");
        let contents = fs::read_to_string(path).expect("could not read CGPO_THETA_IC file");
        let theta: Vec<f64> = contents
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("non-float in theta IC file"))
            .collect();
        assert_eq!(
            theta.len(),
            n,
            "theta IC file has {} values, expected {}",
            theta.len(),
            n
        );
        init_q_from_theta(&mut state.q, &theta, params.s0, lx, ly);
        println!("  theta IC loaded and Q initialised");
    } else {
        // Random IC matching Python: theta ~ Uniform([0, π)) per interior cell
        let mut rng = StdRng::seed_from_u64(seed);
        random_theta_ic(&mut state.q, params.s0, lx, ly, &boundary.inside, &mut rng);
        println!("  random IC generated (seed={seed})");
    }

    // --- save frame 0 ---
    let save_frame = |step: usize, state: &State| {
        let sn = step_name(step);
        let qp = q_dir.join(format!("Q_{sn}.txt"));
        let up = u_dir.join(format!("u_{sn}.txt"));
        write_2col_txt(&qp, &state.q, n).expect("failed to write Q frame");
        write_2col_txt(&up, &state.u, n).expect("failed to write u frame");
    };

    save_frame(0, &state);
    println!("  step 0 saved");

    // --- pressure relaxation target (matching Python p_target_rel_change = 1e-4) ---
    // With max_p_iters set, the loop stops at cap; target is permissive fallback.
    let target_rel_change = 1e-4_f64;

    // --- run loop ---
    println!("Starting run loop…");
    let t_start = Instant::now();
    let mut total_steps: usize = 0;
    let mut last_report = Instant::now();

    while total_steps < max_steps {
        let steps_this_chunk = save_every.min(max_steps - total_steps);

        let (steps_done, _dt_elapsed) =
            update_step(&mut state, &params, &boundary, steps_this_chunk, target_rel_change);

        total_steps += steps_done;

        // save frame
        save_frame(total_steps, &state);

        // progress report every 10 seconds or every save
        let elapsed = t_start.elapsed().as_secs_f64();
        let sps = total_steps as f64 / elapsed;
        let since_last = last_report.elapsed().as_secs_f64();
        if since_last >= 10.0 || total_steps == max_steps {
            println!(
                "  step {total_steps}/{max_steps}  elapsed={elapsed:.1}s  {sps:.1} steps/sec"
            );
            last_report = Instant::now();
        }
    }

    let total_secs = t_start.elapsed().as_secs_f64();
    let steps_per_sec = total_steps as f64 / total_secs;

    println!();
    println!("Run complete.");
    println!("  total steps : {total_steps}");
    println!("  wall time   : {total_secs:.3}s");
    println!("  steps/sec   : {steps_per_sec:.2}");
    println!(
        "  per-{save_every}-step rate : {:.3}s",
        save_every as f64 / steps_per_sec
    );

    // --- write meta.json ---
    let meta = json!({
        "lx": lx,
        "ly": ly,
        "als": als,
        "ncl": ncl,
        "dt": dt,
        "max_p_iters": max_p_iters,
        "n_steps": total_steps,
        "save_every": save_every,
        "zeta": params.zeta,
        "total_seconds": total_secs,
        "steps_per_sec": steps_per_sec,
        "seed": seed,
        "theta_ic_path": theta_ic_path,
    });
    let meta_path = run_dir.join("meta.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .expect("failed to write meta.json");
    println!("  meta.json   : {}", meta_path.display());
    println!("  output dir  : {}", run_dir.display());
}
