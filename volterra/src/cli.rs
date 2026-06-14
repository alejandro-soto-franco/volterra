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
    #[arg(long)]
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

    /// Common loop flags (steps, snap-every, seed, out-raw, config).
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
            RunTarget::Cartesian3d(_) => {
                Err("cartesian3d subcommand is not implemented yet (Task 3.4)".into())
            }
            RunTarget::Dec(_) => {
                Err("dec subcommand is not implemented yet (Task 3.5)".into())
            }
            RunTarget::Cgpo(_) => {
                Err("cgpo subcommand is not implemented yet (Task 3.6)".into())
            }
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
