//! Braid words in the Artin generators and their extraction from worldlines.

use crate::defect::Defect;
use crate::track::{Worldline, track};

/// A signed Artin generator of the braid group `B_n`.
///
/// `index` is **1-based**: `sigma_1` swaps strands at positions 1 and 2, the
/// leftmost adjacent pair. This matches the convention in the reference Python
/// implementation, which writes `sigma_{swap+1}`. `inverse == false` denotes
/// `sigma_i` (the strand at position `i` passing in front); `inverse == true`
/// denotes `sigma_i^-1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Generator {
    /// 1-based generator index `i`, with `1 <= i <= n_strands - 1`.
    pub index: usize,
    /// Whether this is the inverse generator `sigma_i^-1`.
    pub inverse: bool,
}

impl Generator {
    /// Build a generator from a signed 1-based code: `+i` is `sigma_i`,
    /// `-i` is `sigma_i^-1`. `0` is invalid and panics.
    pub fn from_code(code: i32) -> Self {
        assert!(code != 0, "generator code must be non-zero");
        Generator {
            index: code.unsigned_abs() as usize,
            inverse: code < 0,
        }
    }

    /// The signed 1-based code for this generator (`+i` or `-i`).
    pub fn code(self) -> i32 {
        let i = self.index as i32;
        if self.inverse { -i } else { i }
    }
}

impl std::fmt::Display for Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.inverse {
            write!(f, "sigma_{}^-1", self.index)
        } else {
            write!(f, "sigma_{}", self.index)
        }
    }
}

/// A braid on `n_strands`, as an ordered sequence of generators read left to right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BraidWord {
    /// Number of strands `n` (the braid lives in `B_n`).
    pub n_strands: usize,
    /// The generators, in application order.
    pub gens: Vec<Generator>,
}

impl BraidWord {
    /// Build from signed 1-based codes (`+i` / `-i`). `n_strands` must exceed the
    /// largest generator index used.
    pub fn from_codes(n_strands: usize, codes: &[i32]) -> Self {
        let gens: Vec<Generator> = codes.iter().map(|&c| Generator::from_code(c)).collect();
        for g in &gens {
            assert!(
                g.index >= 1 && g.index < n_strands,
                "generator index {} out of range for {} strands",
                g.index,
                n_strands
            );
        }
        BraidWord { n_strands, gens }
    }

    /// The signed 1-based codes of the generators.
    pub fn codes(&self) -> Vec<i32> {
        self.gens.iter().map(|g| g.code()).collect()
    }

    /// The permutation of `{0, .., n_strands-1}` induced by the braid.
    ///
    /// `perm[i]` is the final position of the strand that started at position `i`.
    /// Applying `sigma_k` (1-based) transposes positions `k-1` and `k`.
    pub fn permutation(&self) -> Vec<usize> {
        // strand_at[pos] = strand currently occupying that position.
        let mut strand_at: Vec<usize> = (0..self.n_strands).collect();
        for g in &self.gens {
            strand_at.swap(g.index - 1, g.index);
        }
        let mut perm = vec![0usize; self.n_strands];
        for (pos, &strand) in strand_at.iter().enumerate() {
            perm[strand] = pos;
        }
        perm
    }

    /// The exponent sum (abelianisation): `+1` per `sigma_i`, `-1` per `sigma_i^-1`.
    pub fn exponent_sum(&self) -> i32 {
        self.gens
            .iter()
            .map(|g| if g.inverse { -1 } else { 1 })
            .sum()
    }

    /// The braid word of a defect-position time series: track into worldlines,
    /// then extract. The one-call entry point from raw frames.
    ///
    /// `frames` is a slice of frames, each a list of [`Defect`]s for that time.
    pub fn from_frames(frames: &[Vec<Defect>]) -> Self {
        extract_braidword(&track(frames))
    }

    /// The topological entropy of this braid (see [`crate::topological_entropy`]).
    pub fn topological_entropy(&self) -> f64 {
        crate::entropy::topological_entropy(self)
    }

    /// If the word is an exact repetition of a shorter block, return the shortest
    /// such generating period; otherwise return the whole word.
    pub fn fundamental_period(&self) -> &[Generator] {
        let len = self.gens.len();
        for period in 1..=len {
            if len % period != 0 {
                continue;
            }
            if (0..len).all(|i| self.gens[i] == self.gens[i % period]) {
                return &self.gens[..period];
            }
        }
        &self.gens[..]
    }
}

impl std::fmt::Display for BraidWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.gens.iter().map(|g| g.to_string()).collect();
        write!(f, "{{{}}}", parts.join(" "))
    }
}

#[cfg(test)]
mod braidword_tests {
    use super::*;

    #[test]
    fn from_codes_round_trips() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(w.codes(), vec![-2, 1]);
        assert_eq!(
            w.gens[0],
            Generator {
                index: 2,
                inverse: true
            }
        );
        assert_eq!(
            w.gens[1],
            Generator {
                index: 1,
                inverse: false
            }
        );
    }

    #[test]
    fn display_uses_paper_notation() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(w.to_string(), "{sigma_2^-1 sigma_1}");
    }

    #[test]
    fn golden_permutation_is_three_cycle() {
        // sigma_2^-1 then sigma_1: strand_at_position [0,1,2] -> [0,2,1] -> [2,0,1].
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(w.permutation(), vec![1, 2, 0]);
    }

    #[test]
    fn permutation_ignores_generator_sign() {
        let pos = BraidWord::from_codes(3, &[1]).permutation();
        let neg = BraidWord::from_codes(3, &[-1]).permutation();
        assert_eq!(pos, neg);
        assert_eq!(pos, vec![1, 0, 2]);
    }

    #[test]
    fn exponent_sum_counts_signs() {
        assert_eq!(BraidWord::from_codes(3, &[-2, 1]).exponent_sum(), 0);
        assert_eq!(
            BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]).exponent_sum(),
            0
        );
        assert_eq!(BraidWord::from_codes(3, &[1, 1, 1]).exponent_sum(), 3);
        assert_eq!(BraidWord::from_codes(3, &[1, -2, 1]).exponent_sum(), 1);
    }

    #[test]
    fn from_frames_equals_track_then_extract() {
        let frames = crate::synthetic::golden_orbit(&crate::RealizeOpts {
            frames_per_gen: 10,
            periods: 1,
        });
        let direct = BraidWord::from_frames(&frames);
        let manual = extract_braidword(&crate::track::track(&frames));
        assert_eq!(direct, manual);
        assert_eq!(direct.codes(), vec![-2, 1]);
    }

    #[test]
    fn entropy_method_matches_free_function() {
        let w = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(
            w.topological_entropy(),
            crate::entropy::topological_entropy(&w)
        );
        assert!((w.topological_entropy() - crate::GOLDEN_H).abs() < 1e-9);
    }

    #[test]
    fn fundamental_period_finds_shortest_repeat() {
        let repeated = BraidWord::from_codes(3, &[-2, 1, -2, 1, -2, 1]);
        assert_eq!(repeated.fundamental_period(), &repeated.gens[0..2]);

        let once = BraidWord::from_codes(3, &[-2, 1]);
        assert_eq!(once.fundamental_period(), &once.gens[..]);

        let aab = BraidWord::from_codes(3, &[1, 1, 2]);
        assert_eq!(aab.fundamental_period(), &aab.gens[..]);
    }
}

/// Extract a braid word from defect worldlines.
///
/// At each time step the defects are sorted by x-coordinate; when the sorted
/// order changes by an adjacent transposition at position `i` (0-based gap `i`,
/// i.e. generator `sigma_{i+1}`), a generator is emitted. The sign follows the
/// reference convention: if, in the new x-order, the y-coordinate of the strand
/// now at gap-left exceeds that of the strand at gap-right, emit `sigma_{i+1}`,
/// else `sigma_{i+1}^-1`.
///
/// Frames in which two strands share an exact x or y coordinate are skipped
/// (they are crossing degeneracies; the reference does the same).
pub fn extract_braidword(worldlines: &[Worldline]) -> BraidWord {
    let dim = worldlines.len();
    let mut gens = Vec::new();
    if dim < 2 {
        return BraidWord {
            n_strands: dim.max(1),
            gens,
        };
    }
    let n_frames = worldlines[0].positions.len();
    let mut prev_order: Option<Vec<usize>> = None;

    for t in 0..n_frames {
        let xs: Vec<f64> = worldlines.iter().map(|w| w.positions[t][0]).collect();
        let ys: Vec<f64> = worldlines.iter().map(|w| w.positions[t][1]).collect();
        // Skip crossing degeneracies: two strands sharing an exact x or y.
        if has_exact_duplicate(&xs) || has_exact_duplicate(&ys) {
            continue;
        }
        // Sorted order: strand indices in increasing x.
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());

        let Some(prev) = prev_order.as_ref() else {
            prev_order = Some(order);
            continue;
        };
        if &order == prev {
            continue;
        }
        // Decompose prev -> order into adjacent transpositions (left to right).
        // Emit sigma_{k+1} for a swap at 0-based position k; the sign follows the
        // reference: positive if, in the resulting order, the strand now at the
        // left of the pair has the larger y.
        let mut cur = prev.clone();
        for k in 0..dim {
            while cur[k] != order[k] {
                let p = (k + 1..dim).find(|&p| cur[p] == order[k]).unwrap();
                let j = p - 1;
                cur.swap(j, j + 1);
                let positive = ys[cur[j]] > ys[cur[j + 1]];
                gens.push(Generator {
                    index: j + 1,
                    inverse: !positive,
                });
            }
        }
        prev_order = Some(order);
    }
    BraidWord {
        n_strands: dim,
        gens,
    }
}

/// True if any two entries of `vals` are exactly equal.
fn has_exact_duplicate(vals: &[f64]) -> bool {
    for i in 0..vals.len() {
        for j in (i + 1)..vals.len() {
            if vals[i] == vals[j] {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod extract_tests {
    use super::*;
    use crate::track::Worldline;

    fn wl(positions: &[[f64; 2]]) -> Worldline {
        Worldline {
            positions: positions.to_vec(),
            charge: 1,
        }
    }

    #[test]
    fn no_crossing_gives_empty_word() {
        let wls = vec![wl(&[[0.0, 0.0], [0.0, 0.1]]), wl(&[[1.0, 0.5], [1.0, 0.6]])];
        assert_eq!(extract_braidword(&wls).gens.len(), 0);
    }

    #[test]
    fn right_over_left_crossing_is_sigma_1() {
        // strand 1 (initially right) crosses to the left passing ABOVE: sigma_1.
        let wls = vec![wl(&[[0.0, 0.0], [1.0, 0.0]]), wl(&[[1.0, 0.5], [0.0, 1.0]])];
        assert_eq!(extract_braidword(&wls), BraidWord::from_codes(2, &[1]));
    }

    #[test]
    fn right_over_left_passing_below_is_sigma_1_inverse() {
        let wls = vec![wl(&[[0.0, 0.0], [1.0, 1.0]]), wl(&[[1.0, 0.5], [0.0, 0.0]])];
        assert_eq!(extract_braidword(&wls), BraidWord::from_codes(2, &[-1]));
    }

    #[test]
    fn crossing_of_middle_pair_on_three_strands_is_sigma_2() {
        // strands 1 and 2 (the right pair) cross; strand 0 stays leftmost.
        let wls = vec![
            wl(&[[0.0, 0.3], [0.0, 0.3]]), // strand 0, stays leftmost at x=0
            wl(&[[1.0, 0.0], [2.0, 0.0]]), // strand 1: x 1 -> 2, lower y after
            wl(&[[2.0, 0.6], [1.0, 1.0]]), // strand 2: x 2 -> 1, higher y after
        ];
        // after the cross, position 1 holds strand 2 (y=1.0) > position 2 strand 1 (y=0.0): sigma_2.
        assert_eq!(extract_braidword(&wls), BraidWord::from_codes(3, &[2]));
    }
}
