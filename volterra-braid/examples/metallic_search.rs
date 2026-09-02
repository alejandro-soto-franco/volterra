//! Do the metallic-mean entropies occur as braid entropies on the strand count
//! the cusp law supplies?
//!
//! The confined nematic's index law fixes the defect counts from the anchoring
//! winding and the cusp count: `n_plus = k + 2q` and `n_minus = k`, so
//! at tangential anchoring a shape with `k` cusps braids `k + 2` strands.
//!
//! The two braids the paper reports sit at twice the log of the first two
//! metallic means, `m_j = (j + sqrt(j^2 + 4)) / 2`:
//!
//! ```text
//! cardioid   k = 1   3 strands   2 log m_1 = 0.9624236  golden
//! nephroid   k = 2   4 strands   2 log m_2 = 1.7627472  silver
//! ```
//!
//! which suggests `h = 2 log m_k` for `k` cusps. Before running anything, the
//! braid theory has to admit the value: this enumerates words on `n` strands and
//! reports the entropies found, so the prediction can be checked against what is
//! achievable rather than assumed.
//!
//!     cargo run --release -p volterra-braid --example metallic_search

use std::collections::BTreeMap;

use volterra_braid::braidword::BraidWord;

fn metallic(j: f64) -> f64 {
    2.0 * ((j + (j * j + 4.0).sqrt()) / 2.0).ln()
}

/// Enumerate reduced words up to a length, returning entropy to word.
fn search(n_strands: usize, max_len: usize) -> BTreeMap<i64, Vec<i32>> {
    let gens: Vec<i32> = (1..n_strands as i32)
        .flat_map(|g| [g, -g])
        .collect();
    let mut best: BTreeMap<i64, Vec<i32>> = BTreeMap::new();
    let mut word: Vec<i32> = Vec::with_capacity(max_len);

    fn rec(
        gens: &[i32],
        n_strands: usize,
        word: &mut Vec<i32>,
        max_len: usize,
        best: &mut BTreeMap<i64, Vec<i32>>,
    ) {
        if word.len() >= 2 {
            // The word has to braid EVERY strand, or it is a braid on fewer
            // strands padded out with spectators. Without this the search
            // "confirms" the prediction with impostors: at six strands it
            // returned (sigma_1 sigma_2^-1)^3, which touches three strands and
            // whose entropy is three times golden, and 2 log m_4 equals 6 log phi
            // exactly because m_4 = phi^3. The match was arithmetic, not braiding.
            let uses_all = (1..n_strands as i32)
                .all(|g| word.iter().any(|&c| c.abs() == g));
            let w = BraidWord::from_codes(n_strands, word);
            // Per period, so a word that is a repetition is not credited with the
            // accumulated entropy of its repeats.
            let h = w.entropy_per_period();
            if uses_all && h.is_finite() && h > 1e-9 {
                // Bucket to 1e-6 so equal entropies collapse to one entry.
                let key = (h * 1e6).round() as i64;
                best.entry(key).or_insert_with(|| word.clone());
            }
        }
        if word.len() == max_len {
            return;
        }
        for &g in gens {
            // Skip an immediate cancellation, which cannot change the braid.
            if word.last() == Some(&-g) {
                continue;
            }
            word.push(g);
            rec(gens, n_strands, word, max_len, best);
            word.pop();
        }
    }

    rec(&gens, n_strands, &mut word, max_len, &mut best);
    best
}

fn main() {
    println!(
        "{:>2} {:>8} {:>14} {:>14} {:>10}  word",
        "k", "strands", "2 log m_k", "nearest found", "gap"
    );
    println!("{}", "-".repeat(78));
    for k in 1..=4usize {
        let n = k + 2;
        let want = metallic(k as f64);
        let max_len = if n <= 4 { 9 } else if n == 5 { 8 } else { 7 };
        let found = search(n, max_len);
        let mut best_key = 0i64;
        let mut best_gap = f64::INFINITY;
        for &key in found.keys() {
            let h = key as f64 / 1e6;
            let gap = (h - want).abs();
            if gap < best_gap {
                best_gap = gap;
                best_key = key;
            }
        }
        let w = found.get(&best_key).cloned().unwrap_or_default();
        println!(
            "{k:>2} {n:>8} {want:>14.7} {:>14.7} {best_gap:>10.2e}  {:?}",
            best_key as f64 / 1e6,
            w
        );
    }
    println!(
        "\nA gap at the rounding level means the entropy the cusp law predicts is\n\
         realised by an actual braid on that many strands, so the prediction is\n\
         admissible and the question is whether the flow selects it."
    );
}
