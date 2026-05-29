//! Golden-braid analytic gate.
//!
//! The cardioid orbit is the golden braid `{sigma_2^-1 sigma_1}` on 3 strands,
//! with topological entropy `2 log phi`. A synthetic realisation of the orbit
//! must extract back to that word and reproduce the analytic entropy.

use volterra_braid::{
    BraidWord, GOLDEN_H, RealizeOpts, extract_braidword, golden_orbit, topological_entropy, track,
};

fn extracted(periods: usize) -> BraidWord {
    let frames = golden_orbit(&RealizeOpts {
        frames_per_gen: 10,
        periods,
    });
    extract_braidword(&track(&frames))
}

#[test]
fn orbit_extracts_to_canonical_golden_word() {
    let word = extracted(3);
    let period: Vec<i32> = word.fundamental_period().iter().map(|g| g.code()).collect();
    assert_eq!(period, vec![-2, 1], "extracted word: {word}");
    // Three clean periods of a 2-generator word.
    assert_eq!(word.gens.len(), 6);
}

#[test]
fn golden_entropy_matches_two_log_phi() {
    let word = extracted(1);
    let h = topological_entropy(&word);
    assert!(
        (h - GOLDEN_H).abs() < 1e-9,
        "entropy {h}, expected {GOLDEN_H}"
    );
}

#[test]
fn golden_permutation_is_three_cycle() {
    let one_period = BraidWord::from_codes(3, &[-2, 1]);
    assert_eq!(one_period.permutation(), vec![1, 2, 0]);
}
