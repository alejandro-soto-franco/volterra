//! `volterra` command-line interface definitions (clap derive).
//!
//! The top-level binary parses these types and dispatches to the appropriate
//! simulation runner via `dispatch`. Subcommand bodies are filled in later tasks;
//! the skeleton here gives the full flag surface and a no-op stub so the binary
//! compiles and the parse tests pass.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Active nematics simulations and CGPO on the command line.
#[derive(Debug, Parser)]
#[command(name = "volterra", about = "Active nematics simulations and CGPO")]
pub struct Cli {
    /// Top-level command.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a simulation to completion and write output.
    #[command(subcommand)]
    Run(RunTarget),
}

/// Flags shared by every `run` subcommand.
///
/// These are flattened into each geometry-specific arg struct so they appear
/// on every subcommand without repetition.
#[derive(Debug, Args)]
pub struct CommonArgs {
    /// Number of physics steps to advance.
    #[arg(long, default_value_t = 100)]
    pub steps: usize,

    /// Write a snapshot every this many steps (and always at step 0).
    #[arg(long, default_value_t = 10)]
    pub snap_every: usize,

    /// Base RNG seed for noise and initial conditions.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Output directory override. If absent, a per-subcommand default under
    /// `./output/<subcommand>` is used; see `out_or_default`.
    #[arg(long = "out")]
    pub out_raw: Option<PathBuf>,

    /// Optional TOML config file; CLI flags take precedence over file values.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

impl CommonArgs {
    /// Resolve the output directory, falling back to `./output/<sub>`.
    ///
    /// `sub` is the subcommand name string, e.g. `"cgpo"` or `"cartesian2d"`.
    pub fn out_or_default(&self, sub: &str) -> PathBuf {
        self.out_raw
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("./output/{sub}")))
    }
}

/// Which geometry/solver to run.
#[derive(Debug, Subcommand)]
pub enum RunTarget {
    /// Flat 2D Cartesian active nematic (dry, wet, or BECH).
    Cartesian2d(Cartesian2dArgs),
    /// 3D Cartesian active nematic (dry or BECH).
    Cartesian3d(Cartesian3dArgs),
    /// DEC manifold run on a sphere or torus.
    Dec(DecArgs),
    /// Nephroid-confined CGPO finite-difference solver.
    Cgpo(CgpoArgs),
}

/// Arguments for a flat 2D Cartesian run.
#[derive(Debug, Args)]
pub struct Cartesian2dArgs {
    /// Solver mode: `dry`, `wet`, or `bech`.
    #[arg(long, default_value = "dry")]
    pub mode: String,

    /// Grid width (number of cells in x).
    #[arg(long, default_value_t = 128)]
    pub nx: usize,

    /// Grid height (number of cells in y).
    #[arg(long, default_value_t = 128)]
    pub ny: usize,

    /// Common loop flags (steps, snap-every, seed, out, config).
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Arguments for a 3D Cartesian run.
#[derive(Debug, Args)]
pub struct Cartesian3dArgs {
    /// Solver mode: `dry` or `bech`.
    #[arg(long, default_value = "dry")]
    pub mode: String,

    /// Grid width (x).
    #[arg(long, default_value_t = 32)]
    pub nx: usize,

    /// Grid depth (y).
    #[arg(long, default_value_t = 32)]
    pub ny: usize,

    /// Grid height (z).
    #[arg(long, default_value_t = 32)]
    pub nz: usize,

    /// Common loop flags.
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Arguments for a DEC manifold run.
#[derive(Debug, Args)]
pub struct DecArgs {
    /// Mesh topology: `sphere` or `torus`.
    #[arg(long, default_value = "sphere")]
    pub mesh: String,

    /// Solver mode: `dry` or `wet`.
    #[arg(long, default_value = "dry")]
    pub mode: String,

    /// Common loop flags.
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Arguments for the nephroid-confined CGPO finite-difference solver.
///
/// These mirror the env-var knobs of `cgpo_fd` but are expressed as CLI flags.
/// All physics parameters default to the `Params::new` defaults when absent.
#[derive(Debug, Args)]
pub struct CgpoArgs {
    /// Grid width (and height) in lattice units.
    #[arg(long, default_value_t = 64)]
    pub lx: usize,

    /// Aspect ratio of the semi-axes of the nephroid boundary.
    #[arg(long, default_value_t = 1.5)]
    pub als: f64,

    /// Number of cortical layers (boundary thickness in cells).
    #[arg(long, default_value_t = 1)]
    pub ncl: usize,

    /// Frank elastic constant \ell (overrides Params default when set).
    #[arg(long)]
    pub lambda: Option<f64>,

    /// Time step dt (overrides Params default when set).
    #[arg(long)]
    pub dt: Option<f64>,

    /// Maximum Poisson-pressure relaxation iterations per step.
    #[arg(long)]
    pub max_p_iters: Option<i64>,

    /// Path to a flat-text theta initial-condition file.
    #[arg(long)]
    pub theta_ic: Option<PathBuf>,

    /// Run the finiteness guard every step, not just at snapshots.
    ///
    /// When set, any NaN or Inf causes the run to abort with a non-zero exit.
    #[arg(long, default_value_t = false)]
    pub strict: bool,

    /// Common loop flags.
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Boxed error alias for dispatch bodies.
type DynErr = Box<dyn std::error::Error>;

/// Execute the parsed CLI command.
///
/// Each `run` subcommand builds the appropriate initial field and parameters,
/// drives the matching runner from `volterra-solver`, writes its output under
/// the resolved output directory, and prints a one-line summary.
pub fn dispatch(cli: Cli) -> Result<(), DynErr> {
    match cli.command {
        Command::Run(target) => match target {
            RunTarget::Cartesian2d(args) => run_cartesian2d(args),
            RunTarget::Cartesian3d(args) => run_cartesian3d(args),
            RunTarget::Dec(args) => run_dec(args),
            RunTarget::Cgpo(args) => run_cgpo(args),
        },
    }
}

/// Create a directory, mapping any IO error into the boxed dispatch error.
fn make_out_dir(dir: &std::path::Path) -> Result<(), DynErr> {
    std::fs::create_dir_all(dir)
        .map_err(|e| -> DynErr { format!("create {}: {e}", dir.display()).into() })
}

/// Run the flat 2D Cartesian active nematic.
fn run_cartesian2d(args: Cartesian2dArgs) -> Result<(), DynErr> {
    use volterra_core::ActiveNematicParams;
    use volterra_fields::{QField2D, ScalarField2D};
    use volterra_solver::{run_active_nematic_hydro, run_bech, run_dry_active_nematic};

    // Start from the deterministic test defaults, optionally merge a TOML config
    // (ActiveNematicParams derives Deserialize), then let CLI flags win for nx/ny.
    let mut params = ActiveNematicParams::default_test();
    if let Some(cfg) = &args.common.config {
        let text = std::fs::read_to_string(cfg)
            .map_err(|e| -> DynErr { format!("read {}: {e}", cfg.display()).into() })?;
        params = toml::from_str(&text)
            .map_err(|e| -> DynErr { format!("parse {}: {e}", cfg.display()).into() })?;
    }
    params.nx = args.nx;
    params.ny = args.ny;

    let q0 = QField2D::random_perturbation(params.nx, params.ny, params.dx, 0.001, args.common.seed);

    let out = args.common.out_or_default("cartesian2d");
    make_out_dir(&out)?;

    // Each branch produces a per-snapshot (time, mean_s) summary; the stats
    // structs themselves are not Serialize, so we hand-build the JSON.
    let mode = args.mode.as_str();
    let summary: Vec<(f64, f64)> = match mode {
        "dry" => {
            let (_qf, stats) =
                run_dry_active_nematic(&q0, &params, args.common.steps, args.common.snap_every);
            stats.iter().map(|s| (s.time, s.mean_s)).collect()
        }
        "wet" => {
            let (_qf, stats) =
                run_active_nematic_hydro(&q0, &params, args.common.steps, args.common.snap_every);
            stats.iter().map(|s| (s.time, s.mean_s)).collect()
        }
        "bech" => {
            // phi near equilibrium with a small sinusoidal perturbation, matching
            // the bech golden fixture construction.
            let n = params.nx * params.ny;
            let phi_vals: Vec<f64> = (0..n)
                .map(|k| {
                    let frac = k as f64 / n as f64;
                    0.5 + 0.05 * (frac * 7.3).sin()
                })
                .collect();
            let phi0 = ScalarField2D { phi: phi_vals, nx: params.nx, ny: params.ny, dx: params.dx };
            let (_qf, _phif, stats) =
                run_bech(&q0, &phi0, &params, args.common.steps, args.common.snap_every);
            stats.iter().map(|s| (s.time, s.mean_s)).collect()
        }
        other => return Err(format!("unknown cartesian2d mode '{other}' (expected dry|wet|bech)").into()),
    };

    write_summary_json(&out, &summary)?;
    println!(
        "cartesian2d {mode}: {}x{} grid, {} steps, {} snapshots -> {}",
        params.nx,
        params.ny,
        args.common.steps,
        summary.len(),
        out.display()
    );
    Ok(())
}

/// Run the 3D Cartesian active nematic. The runner writes its own npy frames
/// and `stats.json` into the output directory.
fn run_cartesian3d(args: Cartesian3dArgs) -> Result<(), DynErr> {
    use volterra_core::ActiveNematicParams3D;
    use volterra_fields::{QField3D, ScalarField3D};
    use volterra_solver::runner_3d::{run_bech_3d, run_dry_active_nematic_3d};

    let mut params = ActiveNematicParams3D::default_test();
    if let Some(cfg) = &args.common.config {
        let text = std::fs::read_to_string(cfg)
            .map_err(|e| -> DynErr { format!("read {}: {e}", cfg.display()).into() })?;
        params = toml::from_str(&text)
            .map_err(|e| -> DynErr { format!("parse {}: {e}", cfg.display()).into() })?;
    }
    params.nx = args.nx;
    params.ny = args.ny;
    params.nz = args.nz;

    let q0 = QField3D::random_perturbation(
        params.nx,
        params.ny,
        params.nz,
        params.dx,
        0.001,
        args.common.seed,
    );

    let out = args.common.out_or_default("cartesian3d");
    make_out_dir(&out)?;

    let mode = args.mode.as_str();
    let n_snaps = match mode {
        "dry" => {
            let (_qf, stats) = run_dry_active_nematic_3d(
                &q0,
                &params,
                args.common.steps,
                args.common.snap_every,
                &out,
                false,
            );
            stats.len()
        }
        "bech" => {
            let phi0 = ScalarField3D::zeros(params.nx, params.ny, params.nz, params.dx);
            let (_qf, _phif, stats) = run_bech_3d(
                &q0,
                &phi0,
                &params,
                args.common.steps,
                args.common.snap_every,
                &out,
                false,
            );
            stats.len()
        }
        other => return Err(format!("unknown cartesian3d mode '{other}' (expected dry|bech)").into()),
    };

    println!(
        "cartesian3d {mode}: {}x{}x{} grid, {} steps, {} snapshots -> {}",
        params.nx,
        params.ny,
        params.nz,
        args.common.steps,
        n_snaps,
        out.display()
    );
    Ok(())
}

/// Run a DEC manifold simulation on a sphere or torus.
///
/// Meshes are built at low refinement so a smoke run completes in a few seconds:
/// an icosphere (`Sphere<3>`) for `--mesh sphere` and a small torus
/// (`Euclidean<3>`) for `--mesh torus`. Both go through the proven
/// `DecDomain` operator assembly used by the example binaries.
fn run_dec(args: DecArgs) -> Result<(), DynErr> {
    use cartan_core::Manifold;
    use cartan_dec::{Mesh, Operators};
    use cartan_manifolds::{euclidean::Euclidean, sphere::Sphere};
    use volterra_core::ActiveNematicParams;
    use volterra_dec::mesh_gen::{icosphere, torus_mesh};
    use volterra_dec::{DecDomain, QFieldDec};
    use volterra_solver::{run_dry_active_nematic_dec, run_wet_active_nematic_dec};

    let mut params = ActiveNematicParams::default_test();
    if let Some(cfg) = &args.common.config {
        let text = std::fs::read_to_string(cfg)
            .map_err(|e| -> DynErr { format!("read {}: {e}", cfg.display()).into() })?;
        params = toml::from_str(&text)
            .map_err(|e| -> DynErr { format!("parse {}: {e}", cfg.display()).into() })?;
    }
    params.dt = 0.005;

    let out = args.common.out_or_default("dec");
    make_out_dir(&out)?;

    // Generic driver shared by both mesh topologies: dry uses only the operators,
    // wet additionally needs the mesh and returns a Result.
    fn drive<M: Manifold>(
        mesh: Mesh<M, 3, 2>,
        manifold: M,
        params: &ActiveNematicParams,
        mode: &str,
        seed: u64,
        steps: usize,
        snap_every: usize,
    ) -> Result<Vec<(f64, f64)>, DynErr> {
        let domain = DecDomain::new(mesh, manifold)
            .map_err(|e| -> DynErr { format!("DEC domain assembly: {e:?}").into() })?;
        let nv = domain.mesh.n_vertices();
        let q0 = QFieldDec::random_perturbation(nv, 0.01, seed);
        let ops: &Operators<M, 3, 2> = &domain.ops;
        match mode {
            "dry" => {
                let (_qf, stats) =
                    run_dry_active_nematic_dec(&q0, params, ops, None, steps, snap_every);
                Ok(stats.iter().map(|s| (s.time, s.mean_s)).collect())
            }
            "wet" => {
                let (_qf, stats) = run_wet_active_nematic_dec(
                    &q0, params, ops, &domain.mesh, None, steps, snap_every,
                )
                .map_err(|e| -> DynErr { format!("wet dec runner: {e}").into() })?;
                Ok(stats.iter().map(|s| (s.time, s.mean_s)).collect())
            }
            other => Err(format!("unknown dec mode '{other}' (expected dry|wet)").into()),
        }
    }

    let mode = args.mode.as_str();
    let summary = match args.mesh.as_str() {
        "sphere" => {
            // refinement 2 -> 162 vertices: small and fast, real curvature.
            let mesh = icosphere(2);
            drive(mesh, Sphere::<3>, &params, mode, args.common.seed, args.common.steps, args.common.snap_every)?
        }
        "torus" => {
            let mesh = torus_mesh(2.0, 1.0, 12, 8);
            drive(mesh, Euclidean::<3>, &params, mode, args.common.seed, args.common.steps, args.common.snap_every)?
        }
        other => return Err(format!("unknown dec mesh '{other}' (expected sphere|torus)").into()),
    };

    write_summary_json(&out, &summary)?;
    println!(
        "dec {} {mode}: {} steps, {} snapshots -> {}",
        args.mesh,
        args.common.steps,
        summary.len(),
        out.display()
    );
    Ok(())
}

/// Run the nephroid-confined CGPO solver.
///
/// Builds `Params` from `CgpoArgs` (optionally merging a TOML config), constructs
/// the nephroid boundary and initial condition, then drives the solver through
/// `CgpoStep` and `SimulationRunner`. At each snapshot the observer writes Q/u/p
/// frames via the shared `volterra_cgpo::output::write_state_frame` and checks
/// finiteness. With `--strict`, finiteness is also checked after every step.
fn run_cgpo(args: CgpoArgs) -> Result<(), DynErr> {
    use volterra_cgpo::{
        guard::check_finite,
        nephroid_boundary,
        output::write_state_frame,
        sim_step::CgpoStep,
        step::State,
        CgpoError, Params,
    };
    use volterra_core::sim::{Observer, RunConfig, SimulationRunner, stats::StepStats};

    // --- build Params ---
    // Start from TOML config if provided, then apply CLI overrides.
    let mut params = if let Some(cfg) = &args.common.config {
        let text = std::fs::read_to_string(cfg)
            .map_err(|e| -> DynErr { format!("read {}: {e}", cfg.display()).into() })?;
        toml::from_str::<Params>(&text)
            .map_err(|e| -> DynErr { format!("parse {}: {e}", cfg.display()).into() })?
    } else {
        // Default: als=1.5, ncl=1.0 (sensible mid-range), lambda=1.0, dt=1e-4, max_p_iters=-1
        Params::new(args.lx, args.als, args.ncl as f64, 1.0, 1e-4, -1)
    };
    // CLI overrides (lx always wins; optional flags override matching field)
    params.lx = args.lx;
    params.ly = args.lx; // square grid
    if let Some(lambda) = args.lambda {
        params.lambda = lambda;
    }
    if let Some(dt) = args.dt {
        params.dt = dt;
    }
    if let Some(mpi) = args.max_p_iters {
        params.max_p_iters = mpi;
    }

    let lx = params.lx;
    let ly = params.ly;

    // --- boundary + initial condition ---
    let boundary = nephroid_boundary(lx, ly);

    let mut state = State::new(lx, ly);
    if let Some(ref theta_path) = args.theta_ic {
        // Load theta IC from file (same format as cgpo_fd).
        let n = lx * ly;
        let contents = std::fs::read_to_string(theta_path)
            .map_err(|e| -> DynErr { format!("read theta IC {}: {e}", theta_path.display()).into() })?;
        let theta: Vec<f64> = contents
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(|_| -> DynErr { format!("non-float in theta IC: {s}").into() }))
            .collect::<Result<Vec<f64>, DynErr>>()?;
        if theta.len() != n {
            return Err(format!("theta IC has {} values, expected {n}", theta.len()).into());
        }
        // Q_xx = s0*(cos^2(theta) - 0.5), Q_xy = s0*cos(theta)*sin(theta)
        for x in 0..lx {
            for y in 0..ly {
                let idx = x * ly + y;
                let t = theta[idx];
                let cos_t = t.cos();
                let sin_t = t.sin();
                state.q[idx * 2]     = params.s0 * (cos_t * cos_t - 0.5);
                state.q[idx * 2 + 1] = params.s0 * (cos_t * sin_t);
            }
        }
    } else {
        // Deterministic IC: small uniform Q on interior cells (same as test_smoke / test_cgpo_step).
        let amplitude = 0.1 * params.s0;
        for x in 0..lx {
            for y in 0..ly {
                let idx = x * ly + y;
                if boundary.inside[idx] {
                    state.q[idx * 2]     =  amplitude;
                    state.q[idx * 2 + 1] = -amplitude * 0.5;
                }
            }
        }
    }

    // --- output directory ---
    let out = args.common.out_or_default("cgpo");
    // Per-run sub-dir matching cgpo_fd layout: <out>/als_<als>_ncl_<ncl>/
    let run_dir = out.join(format!("als_{}_ncl_{}", args.als, args.ncl));
    for sub in &["Q", "u", "p"] {
        make_out_dir(&run_dir.join(sub))?;
    }

    // --- physics + runner ---
    let target_rel_change = 1e-4_f64;
    let mut physics = CgpoStep {
        params: params.clone(),
        boundary: boundary.clone(),
        target_rel_change,
    };

    let cfg = RunConfig {
        steps: args.common.steps,
        snap_every: args.common.snap_every,
        dt: params.dt,
        seed: args.common.seed,
        // Output-facing: never silently drop the final state if steps is not a
        // multiple of snap_every. Inherited by strict_runner via config.clone().
        snap_final: true,
    };
    let runner = SimulationRunner { config: cfg };

    // --- observer: writes frames + guards at snapshot cadence ---
    // On strict mode, also guard after every step (implemented via a wrapper observer).
    let strict = args.strict;
    let boundary_ref = &boundary;
    let run_dir_ref = &run_dir;

    // Track the first guard error across snapshots.
    let mut guard_err: Option<CgpoError> = None;

    struct CgpoObserver<'a> {
        run_dir: &'a std::path::Path,
        boundary: &'a volterra_cgpo::boundary::Boundary,
        guard_err: &'a mut Option<CgpoError>,
    }

    impl<'a> Observer<State> for CgpoObserver<'a> {
        fn observe(&mut self, step: usize, _t: f64, state: &State, _stats: &StepStats) {
            if self.guard_err.is_some() {
                return;
            }
            // Finiteness guard
            let r = check_finite(&state.q, "Q", step)
                .and_then(|_| check_finite(&state.u, "u", step))
                .and_then(|_| check_finite(&state.p, "p", step));
            if let Err(e) = r {
                *self.guard_err = Some(e);
                return;
            }
            // Write frame
            if let Err(e) = write_state_frame(self.run_dir, step, state, self.boundary) {
                *self.guard_err = Some(e);
            }
        }
    }

    // For --strict we need to observe every step.  The simplest approach:
    // set snap_every=1 in the runner if strict, so we guard every step.
    // But we still want to write frames at the original cadence.
    // Instead, use a two-observer approach via a wrapper that calls the
    // real observer at snap cadence and guards-only at every step.
    //
    // Since SimulationRunner calls observe() per snap_every, the guard is
    // already at snapshot cadence by default. For --strict, we run with
    // snap_every=1 but only write frames at the original cadence.

    let orig_snap_every = args.common.snap_every;

    if strict {
        // Strict: observe every step; write frames only at original cadence.
        struct StrictObserver<'a> {
            run_dir: &'a std::path::Path,
            boundary: &'a volterra_cgpo::boundary::Boundary,
            orig_snap_every: usize,
            guard_err: &'a mut Option<CgpoError>,
        }
        impl<'a> Observer<State> for StrictObserver<'a> {
            fn observe(&mut self, step: usize, _t: f64, state: &State, _stats: &StepStats) {
                if self.guard_err.is_some() {
                    return;
                }
                let r = check_finite(&state.q, "Q", step)
                    .and_then(|_| check_finite(&state.u, "u", step))
                    .and_then(|_| check_finite(&state.p, "p", step));
                if let Err(e) = r {
                    *self.guard_err = Some(e);
                    return;
                }
                // Write frame only at original cadence.
                if self.orig_snap_every == 0 || step % self.orig_snap_every == 0 {
                    if let Err(e) = write_state_frame(self.run_dir, step, state, self.boundary) {
                        *self.guard_err = Some(e);
                    }
                }
            }
        }

        let strict_runner = SimulationRunner {
            config: RunConfig {
                snap_every: 1,
                ..runner.config.clone()
            },
        };
        let mut obs = StrictObserver {
            run_dir: run_dir_ref,
            boundary: boundary_ref,
            orig_snap_every,
            guard_err: &mut guard_err,
        };
        strict_runner.run(&mut state, &mut physics, &mut obs);
    } else {
        let mut obs = CgpoObserver {
            run_dir: run_dir_ref,
            boundary: boundary_ref,
            guard_err: &mut guard_err,
        };
        runner.run(&mut state, &mut physics, &mut obs);
    }

    if let Some(e) = guard_err {
        return Err(Box::new(e));
    }

    println!(
        "cgpo: {}x{} grid, {} steps, snap_every={}, strict={} -> {}",
        lx,
        ly,
        args.common.steps,
        orig_snap_every,
        strict,
        run_dir.display()
    );
    Ok(())
}

/// Write a `stats.json` summary of `(time, mean_s)` snapshots to `out/stats.json`.
///
/// The runner stats structs do not derive `Serialize`, so we serialise a small
/// hand-built array of objects instead.
fn write_summary_json(out: &std::path::Path, summary: &[(f64, f64)]) -> Result<(), DynErr> {
    let arr: Vec<serde_json::Value> = summary
        .iter()
        .map(|(t, s)| serde_json::json!({ "time": t, "mean_s": s }))
        .collect();
    let json = serde_json::to_string_pretty(&arr)
        .map_err(|e| -> DynErr { format!("serialise stats: {e}").into() })?;
    let path = out.join("stats.json");
    std::fs::write(&path, json)
        .map_err(|e| -> DynErr { format!("write {}: {e}", path.display()).into() })?;
    Ok(())
}
