//! Dimensionless parameters for active nematohydrodynamics on 2-manifolds.
//!
//! All simulations are controlled by four dimensionless numbers:
//!
//! | Symbol | Definition | Physical meaning |
//! |--------|-----------|-----------------|
//! | Pe | alpha R^2 / K | Activity vs. elastic relaxation |
//! | Er | Pe gamma / eta | Viscous vs. elastic stress |
//! | La | |A| R^2 / K | Bulk ordering strength |
//! | Lc | C R^2 / K | Cubic saturation |
//!
//! The equilibrium scalar order parameter is S_eq = 2 sqrt(La / Lc),
//! independent of Pe, Er, K, or alpha.
//!
//! The nondimensionalised governing equations are:
//!
//! ```text
//! dz/dt + Lie_u(z) = Delta_B z + La z - Lc |z|^2 z
//! (1/Er) Delta_1 u = -dp + Pe div(V(z))
//! ```
//!
//! ## References
//!
//! Zhu, Saintillan & Chern. arXiv:2405.06044.

/// Dimensionless parameters for the active nematic engine.
///
/// The characteristic scales are:
/// - Length: R (manifold characteristic size, e.g., sphere radius)
/// - Time: tau = gamma R^2 / K (elastic relaxation time)
/// - Velocity: K / (gamma R)
#[derive(Debug, Clone, Copy)]
pub struct NematicParams {
    /// Peclet number: activity / elastic relaxation.
    /// Pe = alpha R^2 / K.
    /// Controls defect dynamics: Pe ~ 1 (equilibrium), Pe ~ 10^4 (turbulent).
    pub pe: f64,

    /// Ericksen number: viscous / elastic stress.
    /// Er = Pe gamma / eta = alpha gamma R^2 / (eta K).
    pub er: f64,

    /// Landau ordering coefficient (dimensionless).
    /// La = |A| R^2 / K.
    /// Controls equilibrium S via S_eq = 2 sqrt(La / Lc).
    pub la: f64,

    /// Cubic saturation coefficient (dimensionless).
    /// Lc = C R^2 / K.
    pub lc: f64,

    /// Flow alignment parameter.
    /// lambda = (r^2 - 1)/(r^2 + 1) where r is the rod aspect ratio.
    /// |lambda| < 1 for tumbling, lambda = 1 for flow-aligning.
    pub lambda: f64,
}

impl NematicParams {
    /// Construct from dimensionless numbers directly.
    pub fn new(pe: f64, er: f64, la: f64, lc: f64, lambda: f64) -> Self {
        Self { pe, er, la, lc, lambda }
    }

    /// Construct from physical (dimensional) parameters.
    ///
    /// - `k`: Frank elastic constant [energy/length]
    /// - `alpha`: activity coefficient [stress]
    /// - `gamma`: rotational viscosity [stress * time]
    /// - `eta`: fluid viscosity [stress * time]
    /// - `a`: Landau bulk coefficient (negative for ordered phase) [energy/length^3]
    /// - `c`: cubic saturation coefficient [energy/length^3]
    /// - `r`: characteristic length of the manifold (sphere radius, etc.)
    /// - `lambda`: flow alignment parameter
    #[allow(clippy::too_many_arguments)]
    pub fn from_physical(
        k: f64, alpha: f64, gamma: f64, eta: f64,
        a: f64, c: f64, r: f64, lambda: f64,
    ) -> Self {
        let r2 = r * r;
        Self {
            pe: alpha * r2 / k,
            er: alpha * gamma * r2 / (eta * k),
            la: a.abs() * r2 / k,
            lc: c * r2 / k,
            lambda,
        }
    }

    /// Default parameters for testing: Pe = 1, moderate ordering.
    /// 4 tetrahedral defects on S^2, gentle oscillation.
    pub fn default_low_activity() -> Self {
        Self { pe: 1.0, er: 1.0, la: 1.0, lc: 1.0, lambda: 0.7 }
    }

    /// Parameters for active turbulence: Pe = 10^4.
    /// ~100-200 defects on S^2, chaotic dynamics.
    pub fn default_turbulent() -> Self {
        Self { pe: 1e4, er: 100.0, la: 1.0, lc: 1.0, lambda: 0.7 }
    }

    /// Equilibrium scalar order parameter: S_eq = 2 sqrt(La / Lc).
    pub fn s_eq(&self) -> f64 {
        2.0 * (self.la / self.lc).sqrt()
    }

    /// Diffusive CFL timestep bound: dt < C h^2.
    ///
    /// `h` is the mean edge length of the mesh.
    /// The safety factor C accounts for the stencil amplification.
    pub fn dt_diffusive(&self, h: f64) -> f64 {
        // The diffusion coefficient is 1 (nondimensionalised).
        // CFL: dt < h^2 / (2 * d) where d = 1 (2D diffusion).
        // Safety factor 0.25 for RK-like stability.
        0.25 * h * h
    }

    /// Validate that parameters are physically reasonable.
    pub fn validate(&self) -> Result<(), String> {
        if self.pe < 0.0 {
            return Err("Pe must be non-negative".into());
        }
        if self.er <= 0.0 {
            return Err("Er must be positive".into());
        }
        if self.la <= 0.0 {
            return Err("La must be positive".into());
        }
        if self.lc <= 0.0 {
            return Err("Lc must be positive".into());
        }
        Ok(())
    }
}

impl Default for NematicParams {
    fn default() -> Self {
        Self::default_low_activity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s_eq_la_equals_lc() {
        let p = NematicParams::new(1.0, 1.0, 1.0, 1.0, 0.7);
        assert!((p.s_eq() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn s_eq_varies_with_ratio() {
        let p = NematicParams::new(1.0, 1.0, 4.0, 1.0, 0.7);
        assert!((p.s_eq() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn from_physical_unit_sphere() {
        let p = NematicParams::from_physical(
            0.01,  // K
            0.5,   // alpha
            1.0,   // gamma
            0.1,   // eta
            0.01,  // A (|A| = 0.01)
            0.01,  // C
            1.0,   // R = 1 (unit sphere)
            0.7,   // lambda
        );
        assert!((p.pe - 50.0).abs() < 1e-10);
        assert!((p.er - 500.0).abs() < 1e-10);
        assert!((p.la - 1.0).abs() < 1e-10);
        assert!((p.lc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn default_validates() {
        assert!(NematicParams::default().validate().is_ok());
        assert!(NematicParams::default_turbulent().validate().is_ok());
    }

    #[test]
    fn dt_diffusive_scales_with_h_squared() {
        let p = NematicParams::default();
        let dt1 = p.dt_diffusive(0.1);
        let dt2 = p.dt_diffusive(0.05);
        assert!((dt1 / dt2 - 4.0).abs() < 1e-10);
    }
}
