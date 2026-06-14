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

/// Execute the parsed CLI command.
///
/// Subcommand bodies are filled in Tasks 3.3-3.6; this stub returns `Ok(())`
/// so the binary compiles and the skeleton tests pass.
pub fn dispatch(_cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
