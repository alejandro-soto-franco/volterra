//! `fd` -- driver for the confined 2D finite-difference solver.
//!
//! All configuration is via environment variables (defaults in parentheses):
//!
//! | Variable          | Default                                      | Description                        |
//! |-------------------|----------------------------------------------|------------------------------------|
//! | `FD_LX`         | 100                                          | Grid side length (square grid)     |
//! | `FD_ALS`        | 1                                            | Active length scale (pixels)       |
//! | `FD_NCL`        | 9                                            | Nematic correlation length (px)    |
//! | `FD_DT`         | 1e-4                                         | Time step                          |
//! | `FD_MAX_P_ITERS`| 50                                           | Max pressure Jacobi iters per step |
//! | `FD_MAX_STEPS`  | 400000                                       | Total simulation steps             |
//! | `FD_SAVE_EVERY` | 1000                                         | Steps between frame saves          |
//! | `FD_OUT`        | ./output/fd                                | Root output directory              |
//! | `FD_SEED`       | 0                                            | RNG seed (u64) for IC              |
//! | `FD_THETA_IC`   | (unset)                                      | Path to flat theta grid (optional) |
//! | `FD_BOUNDARY`   | nephroid                                     | `circular`, `cardioid`, `nephroid`, `trefoiloid` |
//! | `FD_NET_CHARGE` | 1.0                                          | Boundary winding charge q          |
//! | `FD_ELL_A`      | (unset)                                      | Dimensionless active length; overrides `FD_ALS`  |
//! | `FD_ELL_C`      | (unset)                                      | Dimensionless coherence length; overrides `FD_NCL` |
//!
//! `FD_ELL_A` and `FD_ELL_C` are the quantities arXiv:2503.10880 reports:
//! lengths divided by `sqrt(A_sys)`, the square root of the confined area in
//! lattice sites (p. 3). Setting them converts to the pixel values `FD_ALS` and
//! `FD_NCL` want using the built boundary's own interior count, which is the
//! only way to get the conversion right for a domain whose area depends on the
//! geometry rather than on the grid.

use std::fs;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;

use volterra_fd::{
    boundary::{cardioid_boundary, circular_boundary, nephroid_boundary, trefoiloid_boundary},
    index::{si, vi},
    output::write_state_frame,
    sim_step::FdStep,
    step::State,
    FdError, FdResult, Params,
};
use volterra_core::sim::{Observer, RunConfig, SimulationRunner, stats::StepStats};

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

fn env_i64(key: &str, default: i64) -> i64 {
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
// I/O error helper
// ---------------------------------------------------------------------------

/// Map an `io::Result` to a `FdResult`, attaching the failing path as context.
fn io_ctx<T>(path: &Path, r: std::io::Result<T>) -> FdResult<T> {
    r.map_err(|source| FdError::Io {
        path: path.to_path_buf(),
        source,
    })
}

// ---------------------------------------------------------------------------
// IC helpers
// ---------------------------------------------------------------------------

/// Initialise Q from a theta grid (x*ly+y order, both inside and outside cells).
///
/// `initialize_Q_from_theta` in the Python:
/// ```python
/// Q[:,:,0] = S0 * (cos(theta)^2 - 0.5)
/// Q[:,:,1] = S0 * (cos(theta) * sin(theta))
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
/// theta_initial = 1.0 * pi * np.random.random((Lx, Ly))
/// ```
/// (fully random uniform angles in [0, pi); then Q from theta, applied only at
/// interior cells -- outside cells are zeroed since theta_mask zeros them).
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
// Observer: writes frames at each snapshot point and tracks timing/progress
// ---------------------------------------------------------------------------

struct FdObserver<'a> {
    run_dir: &'a Path,
    boundary: &'a volterra_fd::boundary::Boundary,
    t_start: Instant,
    last_report: Instant,
    max_steps: usize,
    guard_err: Option<FdError>,
}

impl<'a> Observer<State> for FdObserver<'a> {
    fn observe(&mut self, step: usize, _t: f64, state: &State, _stats: &StepStats) {
        if self.guard_err.is_some() {
            return;
        }
        if let Err(e) = write_state_frame(self.run_dir, step, state, self.boundary) {
            self.guard_err = Some(e);
            return;
        }
        println!("  step {step} saved");

        // Progress report every 10 seconds or at the final step.
        let elapsed = self.t_start.elapsed().as_secs_f64();
        let sps = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
        let since_last = self.last_report.elapsed().as_secs_f64();
        if since_last >= 10.0 || step == self.max_steps {
            println!(
                "  step {step}/{}  elapsed={elapsed:.1}s  {sps:.1} steps/sec",
                self.max_steps
            );
            self.last_report = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> FdResult<()> {
    // read config
    let lx = env_usize("FD_LX", 100);
    let als = env_f64("FD_ALS", 2.8);
    let ncl = env_f64("FD_NCL", 4.8);
    let lambda = env_f64("FD_LAMBDA", 1.0); // code-truth flow-alignment (flow-solver.py lambda=1)
    let dt = env_f64("FD_DT", 1e-4);
    let max_p_iters = env_i64("FD_MAX_P_ITERS", -1); // code-truth: uncapped (relax to convergence)
    let max_steps = env_usize("FD_MAX_STEPS", 400_000);
    let save_every = env_usize("FD_SAVE_EVERY", 1_000);
    let out_root = env_string("FD_OUT").unwrap_or_else(|| "./output/fd".to_string());
    let seed = env_u64("FD_SEED", 0);
    let theta_ic_path = env_string("FD_THETA_IC");
    let boundary_kind = std::env::var("FD_BOUNDARY").unwrap_or_else(|_| "nephroid".to_string());
    let net_charge = env_f64("FD_NET_CHARGE", 1.0);
    let ell_a = env_string("FD_ELL_A").and_then(|v| v.parse::<f64>().ok());
    let ell_c = env_string("FD_ELL_C").and_then(|v| v.parse::<f64>().ok());

    // derived
    let ly = lx; // square grid
    let n = lx * ly;

    println!("fd: lx={lx} lambda={lambda} dt={dt} max_p_iters={max_p_iters}");
    println!("  max_steps={max_steps} save_every={save_every}");
    println!("  out_root={out_root}  seed={seed}");
    println!("  boundary={boundary_kind} net_charge={net_charge}");

    // build boundary first: the dimensionless-length conversion needs its area
    println!("Building {boundary_kind} boundary (lx={lx})...");
    let t_bnd = Instant::now();
    let boundary = match boundary_kind.as_str() {
        "circular" => circular_boundary(lx, ly),
        "cardioid" => cardioid_boundary(lx, ly),
        "nephroid" => nephroid_boundary(lx, ly),
        "trefoiloid" => trefoiloid_boundary(lx, ly),
        other => {
            return Err(FdError::Config(format!(
                "unknown FD_BOUNDARY={other}, expected 'circular', 'cardioid', \
                 'nephroid' or 'trefoiloid'"
            )));
        }
    };
    let n_interior = boundary.interior_count();
    let sqrt_area = boundary.sqrt_area();
    println!(
        "  boundary built in {:.2}s -- {n_interior} interior cells, sqrt(A_sys)={sqrt_area:.3}",
        t_bnd.elapsed().as_secs_f64()
    );

    // Dimensionless lengths, if given, set the pixel values.
    let als = match ell_a {
        Some(e) => {
            let px = e * sqrt_area;
            println!("  FD_ELL_A={e} -> als={px:.4} px");
            px
        }
        None => als,
    };
    let ncl = match ell_c {
        Some(e) => {
            let px = e * sqrt_area;
            println!("  FD_ELL_C={e} -> ncl={px:.4} px");
            px
        }
        None => ncl,
    };
    println!("  als={als} ncl={ncl}");

    // build params
    let params = Params::new(lx, als, ncl, lambda, dt, max_p_iters).with_net_charge(net_charge);
    println!(
        "  k_elastic={:.4e} zeta={:.4e} c_landau={:.4e} s0={:.6} eta={:.4e}",
        params.k_elastic, params.zeta, params.c_landau, params.s0, params.eta
    );

    // output directories
    let run_label = format!("als_{als:.4}_ncl_{ncl:.4}");
    let run_dir = Path::new(&out_root).join(&run_label);
    for sub in &["Q", "u", "p"] {
        io_ctx(&run_dir.join(sub), fs::create_dir_all(run_dir.join(sub)))?;
    }

    // The interior mask, one 0/1 per line in x*ly+y order. Defect detection
    // needs it to know which cells are in the domain, and for a geometry whose
    // interior test is a root solve per cell, recomputing it downstream would
    // mean reimplementing that solve. Writing it once is the alternative.
    let mask_path = run_dir.join("mask.txt");
    let mask: String = boundary
        .inside
        .iter()
        .map(|&b| if b { "1\n" } else { "0\n" })
        .collect();
    io_ctx(&mask_path, fs::write(&mask_path, mask))?;

    // allocate state
    let mut state = State::new(lx, ly);

    // initial condition
    if let Some(ref path) = theta_ic_path {
        // Load flat theta grid from text file (one value per line, x*ly+y order)
        println!("Loading theta IC from {path}");
        let ic_path = Path::new(path);
        let contents = io_ctx(ic_path, fs::read_to_string(ic_path))?;
        let theta: Vec<f64> = contents
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| FdError::Config(format!("non-float in theta IC file: {s}")))
            })
            .collect::<FdResult<Vec<f64>>>()?;
        if theta.len() != n {
            return Err(FdError::Config(format!(
                "theta IC file has {} values, expected {}",
                theta.len(),
                n
            )));
        }
        init_q_from_theta(&mut state.q, &theta, params.s0, lx, ly);
        println!("  theta IC loaded and Q initialised");
    } else {
        // Random IC matching Python: theta ~ Uniform([0, pi)) per interior cell
        let mut rng = StdRng::seed_from_u64(seed);
        random_theta_ic(&mut state.q, params.s0, lx, ly, &boundary.inside, &mut rng);
        println!("  random IC generated (seed={seed})");
    }

    // pressure relaxation target (matching Python p_target_rel_change = 1e-4)
    let target_rel_change = 1e-4_f64;

    // physics step + runner
    let mut physics = FdStep {
        params: params.clone(),
        boundary: boundary.clone(),
        target_rel_change,
    };

    let cfg = RunConfig {
        steps: max_steps,
        snap_every: save_every,
        dt: params.dt,
        seed,
        // Output-facing: always write the final frame even when
        // max_steps is not a multiple of save_every.
        snap_final: true,
    };
    let runner = SimulationRunner { config: cfg };

    // run
    println!("Starting run loop...");
    let t_start = Instant::now();

    let mut obs = FdObserver {
        run_dir: &run_dir,
        boundary: &boundary,
        t_start,
        last_report: t_start,
        max_steps,
        guard_err: None,
    };

    runner.run(&mut state, &mut physics, &mut obs);

    if let Some(e) = obs.guard_err {
        return Err(e);
    }

    let total_secs = t_start.elapsed().as_secs_f64();
    let steps_per_sec = max_steps as f64 / total_secs;

    println!();
    println!("Run complete.");
    println!("  total steps : {max_steps}");
    println!("  wall time   : {total_secs:.3}s");
    println!("  steps/sec   : {steps_per_sec:.2}");
    println!(
        "  per-{save_every}-step rate : {:.3}s",
        save_every as f64 / steps_per_sec
    );

    // write meta.json
    let meta = json!({
        "lx": lx,
        "ly": ly,
        "als": als,
        "ncl": ncl,
        "dt": dt,
        "max_p_iters": max_p_iters,
        "n_steps": max_steps,
        "save_every": save_every,
        "zeta": params.zeta,
        "total_seconds": total_secs,
        "steps_per_sec": steps_per_sec,
        "seed": seed,
        "theta_ic_path": theta_ic_path,
    });
    let meta_path = run_dir.join("meta.json");
    // serde_json::to_string_pretty on a purely in-memory json! value cannot fail
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    io_ctx(&meta_path, fs::write(&meta_path, meta_json))?;
    println!("  meta.json   : {}", meta_path.display());
    println!("  output dir  : {}", run_dir.display());

    Ok(())
}
