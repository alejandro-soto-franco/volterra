//! Silver-braid analytic gate.
//!
//! The nephroid orbit is the silver braid
//! `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` on 4 strands,
//! with topological entropy `log(3 + 2 sqrt 2)`.

use volterra_braid::{
    BraidWord, RealizeOpts, SILVER_H, extract_braidword, silver_orbit, topological_entropy, track,
};

const SILVER_CODES: [i32; 6] = [3, 1, 2, -3, -1, -2];

fn extracted(periods: usize) -> BraidWord {
    let frames = silver_orbit(&RealizeOpts {
        frames_per_gen: 10,
        periods,
    });
    extract_braidword(&track(&frames))
}

#[test]
fn orbit_extracts_to_canonical_silver_word() {
    let word = extracted(2);
    let period: Vec<i32> = word.fundamental_period().iter().map(|g| g.code()).collect();
    assert_eq!(period, SILVER_CODES.to_vec(), "extracted word: {word}");
    assert_eq!(word.gens.len(), 12);
}

#[test]
fn silver_entropy_matches_log_3_plus_2_sqrt2() {
    let word = BraidWord::from_codes(4, &SILVER_CODES);
    let h = topological_entropy(&word);
    assert!(
        (h - SILVER_H).abs() < 1e-9,
        "entropy {h}, expected {SILVER_H}"
    );
}

#[test]
fn silver_permutation_reverses_strand_order() {
    let word = BraidWord::from_codes(4, &SILVER_CODES);
    assert_eq!(word.permutation(), vec![3, 2, 1, 0]);
}
