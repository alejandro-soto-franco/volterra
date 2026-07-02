//! Braid-group analysis of active-nematic defect trajectories.
//!
//! This crate is the analysis layer for the "chaos-generating periodic orbits"
//! work (Klein, Soto Franco et al. 2026, arXiv:2503.10880). It takes a time
//! series of topological-defect positions and produces:
//!
//! 1. **Worldlines** by frame-to-frame tracking ([`track`]).
//! 2. A **braid word** in the Artin generators by projecting defect positions
//!    onto the x-axis and reading off adjacent transpositions ([`extract_braidword`]).
//! 3. The **topological entropy** of that braid ([`topological_entropy`]).
//!
//! It also detects defects from a 2D Q-tensor grid ([`detect_defects`]), matching
//! the scheme used by the reference Python implementation
//! (`Chaos-Generating-Periodic-Orbits/braid_tracker.py`), so that volterra's
//! output and the reference can be compared on identical inputs.
//!
//! The crate has no dependency on the PDE solver: the input is geometry (defect
//! positions or a Q grid), so the braid algebra is testable in milliseconds and
//! independent of the (expensive) confined active-nematic simulation.
//!
//! # Analytic targets
//!
//! The two periodic orbits identified in the paper, used as ground truth:
//!
//! - **golden braid** (cardioid): `{sigma_2^-1 sigma_1}` on 3 strands,
//!   topological entropy [`GOLDEN_H`] `= 2 log phi`.
//! - **silver braid** (nephroid): `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}`
//!   on 4 strands, topological entropy [`SILVER_H`] `= log(3 + 2 sqrt 2)`.

pub mod braidword;
pub mod defect;
pub mod entropy;
pub mod synthetic;
pub mod track;

pub use braidword::{BraidWord, Generator, extract_braidword};
pub use defect::{Defect, detect_defects};
pub use entropy::{
    burau_spectral_radius_minus1, is_exact_regime, paper_burau_matrix,
    paper_burau_spectral_radius, paper_burau_word, topological_entropy,
};
pub use synthetic::{RealizeOpts, golden_orbit, realize_braid, silver_orbit};
pub use track::{Worldline, track};

/// The golden ratio `phi = (1 + sqrt 5) / 2`.
pub const PHI: f64 = 1.618_033_988_749_895;

/// The silver ratio `delta_S = 1 + sqrt 2`.
pub const SILVER_RATIO: f64 = 2.414_213_562_373_095;

/// Topological entropy of the golden braid `{sigma_2^-1 sigma_1}`: `2 log phi`.
///
/// The dilatation is `phi^2 = (3 + sqrt 5) / 2`, so `h = log(phi^2) = 2 log phi`.
pub const GOLDEN_H: f64 = 0.962_423_650_119_205_8;

/// Topological entropy of the silver braid: `log(3 + 2 sqrt 2)`.
///
/// The dilatation is `(1 + sqrt 2)^2 = 3 + 2 sqrt 2`, so `h = log(3 + 2 sqrt 2)`.
pub const SILVER_H: f64 = 1.762_747_174_039_086;

#[cfg(test)]
mod constants_tests {
    use super::*;

    #[test]
    fn golden_constant_matches_two_log_phi() {
        assert!((GOLDEN_H - 2.0 * PHI.ln()).abs() < 1e-12);
    }

    #[test]
    fn silver_constant_matches_log_3_plus_2_sqrt2() {
        assert!((SILVER_H - (3.0 + 2.0 * 2.0_f64.sqrt()).ln()).abs() < 1e-12);
        assert!((SILVER_RATIO - (1.0 + 2.0_f64.sqrt())).abs() < 1e-12);
    }
}
