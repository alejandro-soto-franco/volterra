//! Numerical safety guards for the CGPO harness.
//!
//! Both guards are off the kernel hot path: the runner invokes them at snapshot
//! cadence by default, and every step under a `--strict` flag.

use crate::error::{CgpoError, CgpoResult};

/// Verify the advective CFL condition `dt <= safety * min(dx, dy) / max|u|`.
///
/// `u` is the packed 2-vector velocity field (`[ux, uy, ux, uy, ...]`). When the
/// field is at rest (`max|u| == 0`) the condition is vacuously satisfied.
pub fn check_cfl(
    u: &[f64],
    dt: f64,
    dx: f64,
    dy: f64,
    safety: f64,
    step: usize,
) -> CgpoResult<()> {
    let mut umax_sq = 0.0_f64;
    let mut i = 0;
    while i + 1 < u.len() {
        let s = u[i] * u[i] + u[i + 1] * u[i + 1];
        if s > umax_sq {
            umax_sq = s;
        }
        i += 2;
    }
    let umax = umax_sq.sqrt();
    if umax == 0.0 {
        return Ok(());
    }
    let safe = safety * dx.min(dy) / umax;
    if dt > safe {
        return Err(CgpoError::Cfl { step, dt, safe, umax });
    }
    Ok(())
}

/// Verify every entry of `field` is finite; error names the first offending step/field.
pub fn check_finite(field: &[f64], name: &'static str, step: usize) -> CgpoResult<()> {
    if field.iter().any(|v| !v.is_finite()) {
        return Err(CgpoError::NonFinite { step, field: name });
    }
    Ok(())
}
