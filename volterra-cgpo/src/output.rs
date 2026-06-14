//! Shared output helpers for CGPO frame writing.
//!
//! Both the `volterra` CLI and the `cgpo_fd` binary write frames through
//! these functions so the on-disk layout is identical regardless of which
//! entry point drives the solver.
//!
//! Layout (matching the Python `flow-solver.py` numpy savetxt output):
//!
//! ```text
//! <run_dir>/Q/Q_<0000000000>.txt  -- 2-col: Q_xx  Q_xy  (%.18e)
//! <run_dir>/u/u_<0000000000>.txt  -- 2-col: u_x   u_y   (%.18e)
//! <run_dir>/p/p_<0000000000>.txt  -- 1-col: p gauge-fixed (interior-mean=0)
//! ```

use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;

use crate::{boundary::Boundary, error::{CgpoError, CgpoResult}, step::State};

// ---------------------------------------------------------------------------
// Low-level text writers
// ---------------------------------------------------------------------------

/// Write a 2-column text file in numpy savetxt format (`%.18e`).
///
/// `arr` is a 2-component field packed as `[(x*ly+y)*2 + c]`.
/// Writes `n` rows, one per cell.
pub fn write_2col_txt(path: &Path, arr: &[f64], n: usize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    for i in 0..n {
        writeln!(f, "{:.18e}  {:.18e}", arr[i * 2], arr[i * 2 + 1])?;
    }
    Ok(())
}

/// Write a 1-column scalar field, gauge-fixed so the interior mean is zero.
///
/// The CGPO pressure Poisson solve uses pure Neumann BCs, so the absolute
/// pressure is defined only up to an additive constant. Subtracting the
/// interior mean here changes nothing physical (only gradients enter the
/// dynamics) and prevents the field from drifting off-screen in renderings.
pub fn write_1col_gauge_fixed(
    path: &Path,
    field: &[f64],
    inside: &[bool],
    n: usize,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for i in 0..n {
        if inside[i] {
            sum += field[i];
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    let mut f = fs::File::create(path)?;
    for i in 0..n {
        let v = if inside[i] { field[i] - mean } else { 0.0 };
        writeln!(f, "{:.18e}", v)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level frame writer
// ---------------------------------------------------------------------------

/// Format a step index with 10-digit zero-padding (matches Python:
/// `f'_{stepcount:10d}'.replace(' ','0')`).
pub fn step_name(step: usize) -> String {
    format!("{:010}", step)
}

/// Write one CGPO frame (Q, u, p) into `run_dir` under `Q/`, `u/`, `p/`.
///
/// Creates subdirectories as needed. Returns `CgpoError::Io` on any failure.
///
/// `run_dir` should be the per-run directory (e.g.
/// `<out_root>/als_<als>_ncl_<ncl>/`).
pub fn write_state_frame(
    run_dir: &Path,
    step: usize,
    state: &State,
    boundary: &Boundary,
) -> CgpoResult<()> {
    let n = boundary.lx * boundary.ly;
    let sn = step_name(step);

    let q_path = run_dir.join("Q").join(format!("Q_{sn}.txt"));
    let u_path = run_dir.join("u").join(format!("u_{sn}.txt"));
    let p_path = run_dir.join("p").join(format!("p_{sn}.txt"));

    write_2col_txt(&q_path, &state.q, n)
        .map_err(|source| CgpoError::Io { path: q_path.clone(), source })?;
    write_2col_txt(&u_path, &state.u, n)
        .map_err(|source| CgpoError::Io { path: u_path.clone(), source })?;
    write_1col_gauge_fixed(&p_path, &state.p, &boundary.inside, n)
        .map_err(|source| CgpoError::Io { path: p_path.clone(), source })?;

    Ok(())
}
