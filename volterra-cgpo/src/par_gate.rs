//! Parallelism gate: size-based threshold with env-var overrides.
//!
//! # Threshold
//!
//! Kernels run the rayon parallel path only when the grid has at least
//! `PAR_THRESHOLD` cells.  At 100×100 = 10 000 cells the per-row spawn/join
//! overhead dominates (rayon overhead ~µs per task, stencil ~ns per cell), so
//! we stay on the serial path.  At 512×512 = 262 144 cells parallelism wins.
//!
//! # Chunk sizing
//!
//! Instead of one row per rayon task (the previous approach that regressed at
//! small grids), we split into `~threads * 4` tasks so each task handles
//! `rows_per_chunk` consecutive rows.  This amortizes spawn overhead while
//! still allowing work-stealing between cores.
//!
//! ```text
//! rows_per_chunk = max(1, lx / (rayon::current_num_threads() * 4))
//! ```
//!
//! # Env overrides (read once at first use)
//!
//! - `CGPO_FORCE_PARALLEL=1`  — always use rayon regardless of grid size
//! - `CGPO_FORCE_SERIAL=1`    — always use serial regardless of grid size

use std::sync::OnceLock;

/// Minimum grid cell count to engage the rayon parallel paths.
pub const PAR_THRESHOLD: usize = 250_000;

// ---------------------------------------------------------------------------
// Env-var overrides (read once)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Override {
    None,
    ForceParallel,
    ForceSerial,
}

static OVERRIDE: OnceLock<Override> = OnceLock::new();

fn get_override() -> Override {
    *OVERRIDE.get_or_init(|| {
        if std::env::var("CGPO_FORCE_PARALLEL").as_deref() == Ok("1") {
            Override::ForceParallel
        } else if std::env::var("CGPO_FORCE_SERIAL").as_deref() == Ok("1") {
            Override::ForceSerial
        } else {
            Override::None
        }
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns `true` if the kernel should use the rayon parallel path.
#[inline]
pub fn use_parallel(lx: usize, ly: usize) -> bool {
    match get_override() {
        Override::ForceParallel => true,
        Override::ForceSerial => false,
        Override::None => lx * ly >= PAR_THRESHOLD,
    }
}

/// Number of consecutive rows per rayon chunk.
///
/// Targets `~threads * 4` tasks so work-stealing has room but spawn overhead
/// is amortized.  Always at least 1.
#[inline]
pub fn rows_per_chunk(lx: usize) -> usize {
    let threads = rayon::current_num_threads();
    (lx / (threads * 4)).max(1)
}
