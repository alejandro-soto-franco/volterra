//! # volterra-mars
//!
//! Domain-specific extensions for the MARS (Microfluidic Active Rod Suspension)
//! coupled with lyotropic nematic phases (LNP assembly).
//!
//! This crate provides:
//!
//! - **Parameter presets** for MARS simulations at experimentally relevant scales.
//! - **Dimensionless groups** specific to the MARS-LNP coherent transfer window:
//!   the existence condition (Pi number), Damkohler number, and coherent transfer
//!   bounds.
//! - **Convenience re-exports** of the generic volterra runner functions with
//!   MARS-specific documentation.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use volterra_mars::MarsPreset;
//! use volterra_solver::run_dry_active_nematic;
//! use volterra_fields::QField2D;
//!
//! let params = MarsPreset::active_turbulent_64x64();
//! let q0 = QField2D::random_perturbation(params.nx, params.ny, params.dx, 0.001, 42);
//! let (_q, stats) = run_dry_active_nematic(&q0, &params, 1000, 100);
//! ```

use volterra_core::ActiveNematicParams;

/// MARS-specific parameter presets.
///
/// Each preset constructs an [`ActiveNematicParams`] with values chosen to match
/// the MARS experimental system: magnetically actuated nematic rotor suspension
/// coupled to a lyotropic lipid phase.
pub struct MarsPreset;

impl MarsPreset {
    /// 64x64 grid in the active turbulent phase (zeta_eff = 2.0, a_eff < 0).
    ///
    /// Suitable for quick tests and demonstrations of defect-mediated dynamics.
    /// The defect length scale is l_d = sqrt(K_r / zeta_eff) = 0.71 grid spacings.
    pub fn active_turbulent_64x64() -> ActiveNematicParams {
        ActiveNematicParams::default_test()
    }

    /// 128x128 grid in the active turbulent phase.
    ///
    /// Large enough for meaningful defect statistics (expect approximately
    /// 20-40 defects at steady state with zeta_eff = 3.0).
    pub fn active_turbulent_128x128() -> ActiveNematicParams {
        let mut p = ActiveNematicParams::default_test();
        p.nx = 128;
        p.ny = 128;
        p.zeta_eff = 3.0;
        p
    }
}

/// MARS-LNP dimensionless groups.
///
/// These quantities characterise the coherent transfer window specific to the
/// MARS-LNP experimental system. They are computed from generic
/// [`ActiveNematicParams`] fields.
pub struct MarsLnpDimensionless;

impl MarsLnpDimensionless {
    /// Existence condition for coherent orientational transfer:
    /// Pi = K_r / (Gamma_l * eta * K_l).
    ///
    /// Coherent transfer from the rotor defect network to the lipid phase
    /// requires Pi < 1. When Pi >= 1, the lipid relaxation is too slow to
    /// follow the rotor defect dynamics.
    pub fn pi_number(p: &ActiveNematicParams) -> f64 {
        p.pi_number()
    }

    /// Defect length scale l_d = sqrt(K_r / zeta_eff).
    ///
    /// Sets the characteristic spacing of MARS rotor defects and the expected
    /// LNP (lipid nanoparticle) radius in the coherent transfer regime.
    pub fn defect_length(p: &ActiveNematicParams) -> f64 {
        p.defect_length()
    }

    /// Effective Landau parameter a_eff = a_landau - zeta_eff / 2.
    ///
    /// When a_eff < 0, the system is in the active turbulent phase with dense
    /// defects. The MARS system operates in this regime.
    pub fn a_eff(p: &ActiveNematicParams) -> f64 {
        p.a_eff()
    }

    /// Cahn-Hilliard coherence length xi_CH = sqrt(kappa_ch / a_ch).
    ///
    /// At physical scale xi_CH is approximately 1-5 nm, far below the defect
    /// length l_d of approximately 50-200 nm. This scale separation justifies
    /// the Uniform Concentration Approximation (UCA) as a leading-order limit
    /// of the full BECH.
    pub fn ch_coherence_length(p: &ActiveNematicParams) -> f64 {
        p.ch_coherence_length()
    }

    /// Equilibrium lipid fraction phi_eq = sqrt(a_ch / b_ch).
    ///
    /// The double-well free energy F^CH = a_ch phi^2/2 + b_ch phi^4/4 has minima
    /// at phi = 0 and phi = phi_eq; the system phase-separates toward phi_eq in
    /// regions of strong Maier-Saupe coupling.
    pub fn phi_eq(p: &ActiveNematicParams) -> f64 {
        p.phi_eq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_64x64_valid() {
        let p = MarsPreset::active_turbulent_64x64();
        assert!(p.validate().is_ok());
        assert_eq!(p.nx, 64);
        assert!(p.a_eff() < 0.0, "must be in active turbulent phase");
    }

    #[test]
    fn preset_128x128_valid() {
        let p = MarsPreset::active_turbulent_128x128();
        assert!(p.validate().is_ok());
        assert_eq!(p.nx, 128);
    }

    #[test]
    fn dimensionless_groups_consistent() {
        let p = MarsPreset::active_turbulent_64x64();
        assert!(MarsLnpDimensionless::pi_number(&p) > 0.0);
        assert!(MarsLnpDimensionless::defect_length(&p) > 0.0);
        assert!(MarsLnpDimensionless::ch_coherence_length(&p) > 0.0);
        assert!(MarsLnpDimensionless::phi_eq(&p) > 0.0);
    }
}
