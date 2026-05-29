//! Topological entropy of a braid.
//!
//! The entropy is computed as the logarithm of the braid's dilatation, obtained
//! from the spectral radius of its (unreduced) Burau matrix evaluated at
//! `t = -1`.
//!
//! At `t = -1` each generator `sigma_i` acts on an `n`-dimensional space as the
//! identity except on the `(i, i+1)` block, where it is `[[2, -1], [1, 0]]`
//! (and `sigma_i^-1` is `[[0, 1], [-1, 2]]`). The all-ones vector is fixed
//! (eigenvalue 1), so the unreduced representation splits as the trivial
//! summand plus the reduced Burau representation; since a pseudo-Anosov
//! dilatation exceeds 1, the spectral radius of the unreduced product equals the
//! dilatation.
//!
//! This is exact for the orientable-invariant-foliation cases, which include the
//! golden (3-strand) and silver (4-strand) braids of interest. For a general
//! braid it is a lower bound on the topological entropy (Fried); the cases this
//! crate exists to test are exact.

use crate::braidword::{BraidWord, Generator};
use nalgebra::DMatrix;

/// Tolerance below which a dilatation is treated as exactly 1 (zero entropy).
///
/// A reducible or finite-order braid has dilatation exactly 1, but its Burau
/// matrix at `t = -1` can have a *defective* eigenvalue at 1 (a non-trivial
/// Jordan block); the eigenvalue solver then returns a spectral radius of
/// `1 + delta` with `delta` up to a few times `1e-4`. The smallest dilatation of
/// any pseudo-Anosov braid is well above this: it exceeds 2 on at most four
/// strands and is conjecturally bounded below by Lehmer's number `~1.17628` in
/// general. So any spectral radius within `1e-3` of 1 is numerical noise around
/// a true value of 1, and clamping there never hides a genuine positive entropy.
const DILATATION_TOL: f64 = 1e-3;

/// Topological entropy of `word`: `log` of the dilatation (Burau at `t = -1`).
///
/// Returns `0.0` for finite-order or reducible braids (dilatation 1 within
/// [`DILATATION_TOL`]); see that constant for why the tolerance is safe.
pub fn topological_entropy(word: &BraidWord) -> f64 {
    let lambda = burau_spectral_radius_minus1(word);
    if lambda <= 1.0 + DILATATION_TOL {
        0.0
    } else {
        lambda.ln()
    }
}

/// The dilatation estimate: spectral radius of the unreduced Burau matrix of
/// `word` at `t = -1`.
pub fn burau_spectral_radius_minus1(word: &BraidWord) -> f64 {
    let n = word.n_strands;
    // Accumulate the homomorphism image left-to-right: rho(g_1) rho(g_2) ... rho(g_k).
    let mut m = DMatrix::<f64>::identity(n, n);
    for g in &word.gens {
        m *= generator_matrix(n, g);
    }
    m.complex_eigenvalues()
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .fold(0.0, f64::max)
}

/// The unreduced Burau matrix of a single generator at `t = -1`.
///
/// Identity except on the `(i, i+1)` block (0-based `i = index - 1`):
/// `sigma_i -> [[2, -1], [1, 0]]`, `sigma_i^-1 -> [[0, 1], [-1, 2]]`.
fn generator_matrix(n: usize, g: &Generator) -> DMatrix<f64> {
    let i = g.index - 1;
    let mut b = DMatrix::<f64>::identity(n, n);
    if g.inverse {
        b[(i, i)] = 0.0;
        b[(i, i + 1)] = 1.0;
        b[(i + 1, i)] = -1.0;
        b[(i + 1, i + 1)] = 2.0;
    } else {
        b[(i, i)] = 2.0;
        b[(i, i + 1)] = -1.0;
        b[(i + 1, i)] = 1.0;
        b[(i + 1, i + 1)] = 0.0;
    }
    b
}

#[cfg(test)]
mod entropy_tests {
    use super::*;
    use crate::{GOLDEN_H, PHI, SILVER_H};

    #[test]
    fn golden_dilatation_is_phi_squared() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        let lambda = burau_spectral_radius_minus1(&w);
        assert!(
            (lambda - PHI * PHI).abs() < 1e-9,
            "golden dilatation {lambda}, expected phi^2 = {}",
            PHI * PHI
        );
    }

    #[test]
    fn golden_entropy_is_two_log_phi() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert!((topological_entropy(&w) - GOLDEN_H).abs() < 1e-9);
    }

    #[test]
    fn silver_entropy_is_log_3_plus_2_sqrt2() {
        let w = BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]);
        let h = topological_entropy(&w);
        assert!(
            (h - SILVER_H).abs() < 1e-9,
            "silver entropy {h}, expected {SILVER_H}"
        );
    }

    #[test]
    fn single_generator_has_zero_entropy() {
        // A single half-twist is periodic (finite order on the foliation): h = 0.
        let w = BraidWord::from_codes(3, &[1]);
        assert!(topological_entropy(&w) < 1e-9);
    }

    #[test]
    fn trivial_word_has_zero_entropy() {
        let w = BraidWord::from_codes(3, &[1, -1]);
        assert!(topological_entropy(&w) < 1e-9);
    }

    #[test]
    fn reducible_word_with_defective_unit_eigenvalue_is_zero() {
        // sigma_1 sigma_2^-1 sigma_3 sigma_1^-1 sigma_2 is reducible (dilatation 1).
        // Its Burau matrix at t=-1 has a defective eigenvalue at 1, so the eigen
        // solver returns ~1 + 1e-4; this must still report zero entropy.
        let w = BraidWord::from_codes(4, &[1, -2, 3, -1, 2]);
        assert_eq!(
            topological_entropy(&w),
            0.0,
            "spectral radius {} should clamp to zero entropy",
            burau_spectral_radius_minus1(&w)
        );
    }
}
