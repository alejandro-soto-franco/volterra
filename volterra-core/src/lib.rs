// ~/volterra/volterra-core/src/lib.rs

//! # volterra-core
//!
//! Trait definitions, error types, and shared parameters for the volterra framework.
//!
//! ## Traits
//!
//! - [`Integrator`] -- a time-integration scheme (Euler, RK4, ...)
//!
//! ## Parameters
//!
//! - [`MarsParams`] -- all physical and numerical parameters for the MARS + lipid system
//!
//! ## Error type
//!
//! - [`VError`] -- unified error type for all volterra crates

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Unified error type for all volterra crates.
#[derive(Debug, Error)]
pub enum VError {
    /// A field dimension does not match what was expected.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A numerical solver failed to converge.
    #[error("convergence failure after {iters} iterations (residual = {residual:.3e})")]
    ConvergenceFailure { iters: usize, residual: f64 },

    /// Invalid physical or numerical parameter.
    #[error("invalid parameter: {0}")]
    InvalidParams(String),

    /// I/O error (writing field snapshots, etc.).
    #[error("I/O error: {0}")]
    Io(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical parameters
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the single-phase MARS simulation
/// and the coupled MARS + lyotropic lipid (LNP) system.
///
/// ## Dimensionless groups
///
/// Given these parameters the key dimensionless numbers are:
///
/// ```text
/// ℓ_d  = sqrt(K_r / zeta_eff)        defect length scale
/// Da   = Gamma_r * eta / zeta_eff     Damköhler number
/// Sp   = K_r / (Gamma_l * eta * K_l) existence condition (< 1 for coherent transfer)
/// ```
///
/// The coherent transfer window is zeta_one < zeta_eff < zeta_star.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsParams {
    // ── Grid ──────────────────────────────────────────────────────────────
    /// Number of grid vertices in x.
    pub nx: usize,
    /// Number of grid vertices in y.
    pub ny: usize,
    /// Grid spacing (same in x and y; periodic boundary conditions assumed).
    pub dx: f64,
    /// Time step.
    pub dt: f64,

    // ── Rotor phase ────────────────────────────────────────────────────────
    /// Rotor Frank elastic constant K_r (one-constant approximation).
    pub k_r: f64,
    /// Rotor rotational viscosity Γ_r (collective relaxation rate = 1/Γ_r).
    pub gamma_r: f64,
    /// Effective activity ζ_eff = ζ₀ B₀² ω_B τ_r / (1 + (ω_B τ_r)²).
    /// Controls defect density: ρ_d ~ ζ_eff / K_r.
    pub zeta_eff: f64,
    /// Fluid viscosity η.
    pub eta: f64,
    /// Landau coefficient a (< 0 for the ordered nematic without activity).
    /// Effective driving is a_eff = a - zeta_eff/2.
    pub a_landau: f64,
    /// Landau coefficient c > 0 (stabilizes large |Q|).
    pub c_landau: f64,
    /// Flow alignment parameter λ (tumbling vs. flow-aligning nematics).
    /// λ = 1.0 for flow-aligning; |λ| < 1 for tumbling.
    pub lambda: f64,

    // ── Langevin noise ────────────────────────────────────────────────────
    /// RMS amplitude of the Langevin noise term added at each time step.
    ///
    /// The noise is applied as `Q += noise_amp * sqrt(dt) * W` where `W` is
    /// an i.i.d. standard Gaussian per component per grid vertex.  This models
    /// the stochastic reorientation fluctuations of the MARS rod ensemble.
    /// Set to 0.0 (default) to disable noise.
    pub noise_amp: f64,

    // ── Lipid phase (Component 2: one-way coupling) ───────────────────────
    /// Lipid Frank elastic constant K_l.
    pub k_l: f64,
    /// Lipid rotational viscosity Γ_l.
    pub gamma_l: f64,
    /// Lipid coupling length ξ_l (range of the K₀ response kernel).
    /// Determines LNP radius: R_LNP ~ ℓ_d = sqrt(K_r / zeta_eff).
    pub xi_l: f64,
}

impl MarsParams {
    /// Defect length scale ℓ_d = sqrt(K_r / ζ_eff).
    ///
    /// This equals the mean rotor defect spacing and sets the LNP radius.
    pub fn defect_length(&self) -> f64 {
        (self.k_r / self.zeta_eff).sqrt()
    }

    /// Dimensionless existence condition Π = K_r / (Γ_l η K_l).
    ///
    /// Coherent transfer requires Π < 1.
    pub fn pi_number(&self) -> f64 {
        self.k_r / (self.gamma_l * self.eta * self.k_l)
    }

    /// Effective Landau parameter a_eff = a_landau - zeta_eff / 2.
    ///
    /// When a_eff < 0 the system is in the active turbulent (defect-laden) phase.
    pub fn a_eff(&self) -> f64 {
        self.a_landau - self.zeta_eff / 2.0
    }

    /// Validate that parameters are physically reasonable.
    pub fn validate(&self) -> Result<(), VError> {
        if self.nx < 2 {
            return Err(VError::InvalidParams("nx must be >= 2".into()));
        }
        if self.ny < 2 {
            return Err(VError::InvalidParams("ny must be >= 2".into()));
        }
        if self.dx <= 0.0 {
            return Err(VError::InvalidParams("dx must be positive".into()));
        }
        if self.dt <= 0.0 {
            return Err(VError::InvalidParams("dt must be positive".into()));
        }
        if self.k_r <= 0.0 {
            return Err(VError::InvalidParams("k_r must be positive".into()));
        }
        if self.gamma_r <= 0.0 {
            return Err(VError::InvalidParams("gamma_r must be positive".into()));
        }
        if self.zeta_eff < 0.0 {
            return Err(VError::InvalidParams("zeta_eff must be non-negative".into()));
        }
        if self.eta <= 0.0 {
            return Err(VError::InvalidParams("eta must be positive".into()));
        }
        if self.c_landau <= 0.0 {
            return Err(VError::InvalidParams("c_landau must be positive".into()));
        }
        if self.xi_l <= 0.0 {
            return Err(VError::InvalidParams("xi_l must be positive".into()));
        }
        if self.noise_amp < 0.0 {
            return Err(VError::InvalidParams("noise_amp must be non-negative".into()));
        }
        Ok(())
    }

    /// Construct a minimal default parameter set useful for testing.
    ///
    /// Grid: 64x64, dx=1.0, dt=0.01.
    /// Physics: K_r=1, Γ_r=1, ζ_eff=2 (active turbulent), η=1.
    pub fn default_test() -> Self {
        Self {
            nx: 64,
            ny: 64,
            dx: 1.0,
            dt: 0.01,
            k_r: 1.0,
            gamma_r: 1.0,
            zeta_eff: 2.0,
            eta: 1.0,
            a_landau: -0.5,
            c_landau: 4.5,
            lambda: 0.7,
            noise_amp: 0.0,
            k_l: 0.5,
            gamma_l: 1.0,
            xi_l: 5.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integrator trait
// ─────────────────────────────────────────────────────────────────────────────

/// A time integration scheme for the Beris-Edwards equation.
///
/// The generic parameter `S` is the full simulation state (e.g., a struct
/// containing the Q-tensor field and any other evolving quantities).
pub trait Integrator<S> {
    /// Advance `state` by one time step `dt`, given the RHS function `rhs`.
    ///
    /// `rhs(state) -> dstate/dt` should return the time derivative in the
    /// same representation as `state`.
    fn step<F>(&self, state: &S, dt: f64, rhs: F) -> S
    where
        F: Fn(&S) -> S;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_test_params_valid() {
        let p = MarsParams::default_test();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn defect_length_correct() {
        let mut p = MarsParams::default_test();
        p.k_r = 4.0;
        p.zeta_eff = 1.0;
        assert!((p.defect_length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_eff_correct() {
        let mut p = MarsParams::default_test();
        p.a_landau = -1.0;
        p.zeta_eff = 3.0;
        // a_eff = -1.0 - 1.5 = -2.5
        assert!((p.a_eff() - (-2.5)).abs() < 1e-12);
    }

    #[test]
    fn invalid_params_caught() {
        let mut p = MarsParams::default_test();
        p.dx = -1.0;
        assert!(p.validate().is_err());
        p.dx = 1.0;
        p.k_r = 0.0;
        assert!(p.validate().is_err());
    }
}
