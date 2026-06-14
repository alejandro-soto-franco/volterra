//! Error type for the CGPO solver harness.

use std::path::PathBuf;

/// Errors raised by the CGPO harness (I/O, configuration, and numerical guards).
///
/// Kernel code stays panic-free in release; the harness propagates these with
/// `?` so a failure mid-run cannot silently lose output.
#[derive(Debug, thiserror::Error)]
pub enum CgpoError {
    /// An I/O operation failed; the offending path is carried for context.
    #[error("CGPO I/O error at {path}: {source}")]
    Io {
        /// Path that failed to read or write.
        path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A configuration value (env var, IC file, flag) was malformed.
    #[error("CGPO configuration error: {0}")]
    Config(String),

    /// The CFL condition was violated at `step` (advective dt too large).
    #[error("CFL violation at step {step}: dt={dt} exceeds safe {safe} (|u|max={umax})")]
    Cfl {
        /// Step index at which the violation was detected.
        step: usize,
        /// Configured time step.
        dt: f64,
        /// Largest CFL-safe time step at this state.
        safe: f64,
        /// Maximum velocity magnitude driving the bound.
        umax: f64,
    },

    /// A field contained a NaN or infinity at `step`.
    #[error("non-finite value in field '{field}' at step {step}")]
    NonFinite {
        /// Step index at which the non-finite value was found.
        step: usize,
        /// Name of the field (e.g. "u", "Q", "p").
        field: &'static str,
    },
}

/// Convenience result alias for harness code.
pub type CgpoResult<T> = Result<T, CgpoError>;
