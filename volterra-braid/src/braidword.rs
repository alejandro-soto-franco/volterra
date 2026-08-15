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

    /// The shortest block the word repeats, allowing the last repeat to be cut
    /// short.
    ///
    /// A sampling window closes wherever the run ended, so an extracted word
    /// generally holds a whole number of periods plus part of one more. Requiring
    /// the length to divide by the period, as an earlier version did, rejects the
    /// period in that case and returns the whole word, whose entropy is then the
    /// accumulated entropy of every period in the window rather than the entropy
    /// of the braid. Two full repeats are required before a period is claimed.
    pub fn fundamental_period(&self) -> &[Generator] {
        let len = self.gens.len();
        for period in 1..=len / 2 {
            if (0..len).all(|i| self.gens[i] == self.gens[i % period]) {
                return &self.gens[..period];
            }
        }
        &self.gens[..]
    }

    /// The word with adjacent commuting generators put in index order.
    ///
    /// `sigma_i` and `sigma_j` commute whenever `|i - j| >= 2`, so a swap of one
    /// pair of strands and a swap of a disjoint pair are the same braid in
    /// either order. Extraction emits them in whichever order the sampling
    /// caught, which is set by how close the two crossings fell in time and not
    /// by the braid, so two samplings of one orbit can differ as strings while
    /// being the same element. Sorting each commuting adjacent pair by index is
    /// a normal form for those relations, and comparing normal forms compares
    /// braids.
    ///
    /// The braid relation `sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i
    /// sigma_{i+1}` is not applied: adjacent generators do not commute, and the
    /// far-commutation relations alone are what an extraction order can vary.
    pub fn commutation_normal_form(&self) -> BraidWord {
        let mut gens = self.gens.clone();
        let mut moved = true;
        while moved {
            moved = false;
            for i in 0..gens.len().saturating_sub(1) {
                let (a, b) = (gens[i], gens[i + 1]);
                if a.index.abs_diff(b.index) >= 2 && a.index > b.index {
                    gens.swap(i, i + 1);
                    moved = true;
                }
            }
        }
        BraidWord {
            n_strands: self.n_strands,
            gens,
        }
    }

    /// The braid of one period, as a word in its own right.
    ///
    /// The period is taken of the [commutation normal
    /// form](Self::commutation_normal_form), so a period whose generators were
    /// emitted in a different order from one repeat to the next is still found.
    pub fn period_word(&self) -> BraidWord {
        let normal = self.commutation_normal_form();
        BraidWord {
            n_strands: self.n_strands,
            gens: normal.fundamental_period().to_vec(),
        }
    }

    /// The topological entropy of one period.
    ///
    /// This is the quantity a braid is quoted by, and the quantity to compare
    /// against a published dilatation. [`topological_entropy`](Self::topological_entropy)
    /// applied to a multi-period window returns the entropy of the whole window,
    /// which grows with the length of the run.
    pub fn entropy_per_period(&self) -> f64 {
        self.period_word().topological_entropy()
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

    #[test]
    fn fundamental_period_survives_a_truncated_last_repeat() {
        // Six full silver periods and half of a seventh: the length is not a
        // multiple of six, which is the shape a window that closes mid-period has.
        let block = [3, 1, 2, -3, -1, -2];
        let mut codes: Vec<i32> = block.iter().cycle().take(39).copied().collect();
        let w = BraidWord::from_codes(4, &codes);
        assert_eq!(w.fundamental_period().len(), 6);
        assert!((w.entropy_per_period() - crate::SILVER_H).abs() < 1e-9);

        // Six whole periods carry six times the entropy, so the window measure
        // depends on how long the run was.
        let six = BraidWord::from_codes(4, &block.repeat(6));
        assert!((six.topological_entropy() - 6.0 * crate::SILVER_H).abs() < 1e-9);

        // Cutting the seventh period short does not shorten the window measure by
        // a known amount: here it collapses the dilatation to 1 outright. A
        // whole-window entropy is not comparable to a published one at all.
        assert!(w.topological_entropy() < 1e-9);

        // A word with no repeat still reports itself.
        codes[20] = 1;
        let broken = BraidWord::from_codes(4, &codes);
        assert_eq!(broken.fundamental_period().len(), codes.len());
    }

    #[test]
    fn commutation_normal_form_preserves_the_braid() {
        // sigma_1 and sigma_3 commute in B_4, so ordering them either way is the
        // same element and must carry the same entropy.
        let a = BraidWord::from_codes(4, &[3, 1, 2, -3, -1, -2]);
        let b = BraidWord::from_codes(4, &[1, 3, 2, -1, -3, -2]);
        assert_eq!(a.commutation_normal_form(), b.commutation_normal_form());
        assert!((a.topological_entropy() - b.topological_entropy()).abs() < 1e-12);

        // Adjacent generators do not commute and must not be reordered.
        let adj = BraidWord::from_codes(3, &[2, 1]);
        assert_eq!(adj.commutation_normal_form().codes(), vec![2, 1]);
    }

    #[test]
    fn period_survives_commuting_generators_emitted_in_either_order() {
        // Six silver periods, one of them with its commuting pair swapped, then
        // cut short: the shape the measured silver word has.
        let block = [1, 3, 2, -1, -3, -2];
        let mut codes: Vec<i32> = block.iter().cycle().take(39).copied().collect();
        codes.swap(6, 7); // sigma_1 sigma_3 -> sigma_3 sigma_1 in the second repeat
        let w = BraidWord::from_codes(4, &codes);

        assert_eq!(w.fundamental_period().len(), codes.len(), "as a string, no period");
        assert_eq!(w.period_word().gens.len(), 6, "as a braid, period six");
        assert!((w.entropy_per_period() - crate::SILVER_H).abs() < 1e-9);
    }

    #[test]
    fn entropy_per_period_is_independent_of_window_length() {
        let block = [-2, 1];
        for repeats in 2..12 {
            let codes: Vec<i32> = block.iter().cycle().take(2 * repeats).copied().collect();
            let w = BraidWord::from_codes(3, &codes);
            assert!(
                (w.entropy_per_period() - crate::GOLDEN_H).abs() < 1e-9,
                "{repeats} repeats gave {}",
                w.entropy_per_period()
            );
        }
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
