//! Synthetic defect trajectories that realise a given braid word.
//!
//! Used to test the extraction pipeline: a braid word is turned into a
//! continuous, well-separated point motion sampled into frames, so that feeding
//! the frames back through [`track`](crate::track) and
//! [`extract_braidword`](crate::extract_braidword) recovers exactly that word.

use crate::braidword::BraidWord;
use crate::defect::Defect;

/// Options controlling how a braid word is rendered into frames.
#[derive(Debug, Clone, Copy)]
pub struct RealizeOpts {
    /// Frames generated per generator (the smoothness of each swap).
    pub frames_per_gen: usize,
    /// How many times to repeat the whole word (periods of the orbit).
    pub periods: usize,
}

impl Default for RealizeOpts {
    fn default() -> Self {
        RealizeOpts {
            frames_per_gen: 8,
            periods: 1,
        }
    }
}

/// Render `word` as a defect-position time series whose extracted braid word is
/// exactly `word` (repeated `opts.periods` times).
///
/// Strands occupy integer x-slots `1..=n`. Each generator smoothly swaps the two
/// strands currently at adjacent x-ranks `i, i+1`, with the over/under (the y
/// ordering at the crossing) chosen so the extractor emits the generator with
/// the requested sign.
pub fn realize_braid(word: &BraidWord, opts: &RealizeOpts) -> Vec<Vec<Defect>> {
    let n = word.n_strands;
    let m = opts.frames_per_gen.max(2);
    // Distinct, tiny static baseline y per strand so rest frames are never
    // crossing-degenerate; the swing amplitude h dominates at a crossing so the
    // emitted generator's sign is set by the swap, not the baseline.
    const EPS: f64 = 1e-3;
    const H: f64 = 2.0;
    let baseline = |s: usize| s as f64 * EPS;

    // strand_at_slot[k] = strand currently occupying slot k (0-based); x = k + 1.
    let mut strand_at_slot: Vec<usize> = (0..n).collect();

    // Emit a frame as a strand-indexed Vec<Defect> from explicit x/y per strand.
    let frame_from = |xs: &[f64], ys: &[f64]| -> Vec<Defect> {
        (0..n)
            .map(|s| Defect {
                pos: [xs[s], ys[s]],
                charge: 1,
            })
            .collect()
    };
    let rest = |strand_at_slot: &[usize]| -> Vec<Defect> {
        let mut xs = vec![0.0; n];
        let mut ys = vec![0.0; n];
        for (slot, &s) in strand_at_slot.iter().enumerate() {
            xs[s] = (slot + 1) as f64;
            ys[s] = baseline(s);
        }
        frame_from(&xs, &ys)
    };

    let mut frames = vec![rest(&strand_at_slot)];

    for _ in 0..opts.periods.max(1) {
        for g in &word.gens {
            let left_slot = g.index - 1; // 0-based
            let a = strand_at_slot[left_slot]; // strand at the left of the pair
            let b = strand_at_slot[left_slot + 1]; // strand at the right
            for k in 0..m {
                let tau = (k + 1) as f64 / m as f64;
                let pert = H * (std::f64::consts::PI * tau).sin();
                let mut xs = vec![0.0; n];
                let mut ys = vec![0.0; n];
                for (slot, &s) in strand_at_slot.iter().enumerate() {
                    xs[s] = (slot + 1) as f64;
                    ys[s] = baseline(s);
                }
                // The two swapping strands move within [left_slot+1, left_slot+2].
                xs[a] = (left_slot + 1) as f64 + tau;
                xs[b] = (left_slot + 2) as f64 - tau;
                // sign: sigma_i (not inverse) => b ends on the left passing high.
                if g.inverse {
                    ys[a] = baseline(a) + pert;
                    ys[b] = baseline(b) - pert;
                } else {
                    ys[a] = baseline(a) - pert;
                    ys[b] = baseline(b) + pert;
                }
                frames.push(frame_from(&xs, &ys));
            }
            strand_at_slot.swap(left_slot, left_slot + 1);
        }
    }
    frames
}

/// The golden-braid orbit `{sigma_2^-1 sigma_1}` on 3 strands.
pub fn golden_orbit(opts: &RealizeOpts) -> Vec<Vec<Defect>> {
    realize_braid(&BraidWord::from_codes(3, &[-2, 1]), opts)
}

/// The silver-braid orbit `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}`
/// on 4 strands.
pub fn silver_orbit(opts: &RealizeOpts) -> Vec<Vec<Defect>> {
    realize_braid(&BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]), opts)
}

#[cfg(test)]
mod synthetic_tests {
    use super::*;
    use crate::{extract_braidword, track};

    fn round_trip(word: &BraidWord) -> BraidWord {
        let frames = realize_braid(
            word,
            &RealizeOpts {
                frames_per_gen: 8,
                periods: 1,
            },
        );
        extract_braidword(&track(&frames))
    }

    #[test]
    fn realizes_each_single_generator() {
        for code in [1i32, -1, 2, -2, 3, -3] {
            let n = code.unsigned_abs() as usize + 1;
            let w = BraidWord::from_codes(n, &[code]);
            assert_eq!(round_trip(&w).codes(), vec![code], "failed for code {code}");
        }
    }

    #[test]
    fn realizes_a_two_generator_word() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(round_trip(&w).codes(), vec![-2, 1]);
    }
}
