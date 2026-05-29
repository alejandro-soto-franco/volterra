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

/// The paper's reduced Burau matrix for a single generator, exactly as
/// tabulated in arXiv:2503.10880 (SI.11 for B_3, SI.16 for B_4).
///
/// This reproduces the paper's representation verbatim so that the braid
/// products beta_golden (SI.12) and beta_silver (SI.17) match the paper exactly,
/// confirming volterra's topological entropy equals the paper's `h = log|b_max|`.
///
/// Note: the paper's B_3 (SI.11) and B_4 (SI.16) tabulations use mutually
/// inconsistent sigma_1/sigma_2 labelings (SI.11 disagrees with both the general
/// Burau formula and SI.16). We reproduce each as written, so the stated products
/// SI.12 and SI.17 are matched verbatim; the dilatation `|b_max|` -- and hence the
/// entropy -- is unaffected by the labeling. Tabulated for B_3 and B_4 only (the
/// golden and silver braids); other strand counts use the unreduced-at-`t=-1`
/// representation via [`burau_spectral_radius_minus1`].
pub fn paper_burau_matrix(n_strands: usize, g: crate::braidword::Generator) -> DMatrix<f64> {
    let i = g.index;
    match n_strands {
        3 => {
            let m: [f64; 4] = match (i, g.inverse) {
                (1, false) => [1., 1., 0., 1.],
                (1, true) => [1., -1., 0., 1.],
                (2, false) => [1., 0., -1., 1.],
                (2, true) => [1., 0., 1., 1.],
                _ => panic!("B_3 generator index {i} out of range"),
            };
            DMatrix::from_row_slice(2, 2, &m)
        }
        4 => {
            let m: [f64; 9] = match (i, g.inverse) {
                (1, false) => [1., 0., 0., -1., 1., 0., 0., 0., 1.],
                (1, true) => [1., 0., 0., 1., 1., 0., 0., 0., 1.],
                (2, false) => [1., 1., 0., 0., 1., 0., 0., -1., 1.],
                (2, true) => [1., -1., 0., 0., 1., 0., 0., 1., 1.],
                (3, false) => [1., 0., 0., 0., 1., 1., 0., 0., 1.],
                (3, true) => [1., 0., 0., 0., 1., -1., 0., 0., 1.],
                _ => panic!("B_4 generator index {i} out of range"),
            };
            DMatrix::from_row_slice(3, 3, &m)
        }
        _ => panic!("paper_burau_matrix is tabulated for B_3 and B_4 only"),
    }
}

/// The braidword's image under the paper's reduced Burau representation
/// (left-to-right product), as an `(n-1) x (n-1)` matrix.
pub fn paper_burau_word(word: &BraidWord) -> DMatrix<f64> {
    let mut m = DMatrix::<f64>::identity(word.n_strands - 1, word.n_strands - 1);
    for g in &word.gens {
        m *= paper_burau_matrix(word.n_strands, *g);
    }
    m
}

/// Spectral radius `|b_max|` of the paper's reduced Burau matrix of `word`.
pub fn paper_burau_spectral_radius(word: &BraidWord) -> f64 {
    paper_burau_word(word)
        .complex_eigenvalues()
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod entropy_tests {
    use super::*;
    use crate::{GOLDEN_H, PHI, SILVER_H};

    #[test]
    fn paper_golden_matrix_matches_si12() {
        // arXiv:2503.10880 SI.12: beta_golden = sigma_2^-1 sigma_1 = [[1,1],[1,2]].
        let w = BraidWord::from_codes(3, &[-2, 1]);
        let expected = DMatrix::from_row_slice(2, 2, &[1., 1., 1., 2.]);
        assert_eq!(paper_burau_word(&w), expected);
    }

    #[test]
    fn paper_silver_matrix_matches_si17() {
        // arXiv:2503.10880 SI.17: beta_silver = [[2,-2,-1],[-2,3,2],[-1,2,2]].
        let w = BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]);
        let expected = DMatrix::from_row_slice(3, 3, &[2., -2., -1., -2., 3., 2., -1., 2., 2.]);
        assert_eq!(paper_burau_word(&w), expected);
    }

    #[test]
    fn paper_burau_dilatations_match_metallic_ratios() {
        // golden b_max = phi_0^2 = (3+sqrt5)/2; silver b_max = phi_1^2 = 3+2sqrt2.
        let golden = paper_burau_spectral_radius(&BraidWord::from_codes(3, &[-2, 1]));
        assert!((golden - (3.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-9);
        let silver = paper_burau_spectral_radius(&BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]));
        assert!((silver - (3.0 + 2.0 * 2.0_f64.sqrt())).abs() < 1e-9);
    }

    #[test]
    fn paper_burau_entropy_agrees_with_unreduced() {
        // The paper's reduced Burau and volterra's unreduced-at-t=-1 give the same h.
        for (n, codes) in [(3usize, vec![-2, 1]), (4, vec![3, 1, 2, -3, -1, -2])] {
            let w = BraidWord::from_codes(n, &codes);
            let h_paper = paper_burau_spectral_radius(&w).ln();
            assert!((h_paper - topological_entropy(&w)).abs() < 1e-9);
        }
    }

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
