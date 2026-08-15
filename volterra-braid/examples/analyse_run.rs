//! Read an `fd` run directory and report the braid of its mobile `+1/2` defects.
//!
//! ```text
//! cargo run --release -p volterra-braid --example analyse_run -- <run_dir> [--min-window N]
//! ```
//!
//! `<run_dir>` is what the `fd` driver writes: `Q/Q_*.txt` frames and a
//! `mask.txt` giving the confined interior. Detection is by director winding
//! ([`volterra_braid::detect_defects_winding`]), which carries no threshold, so
//! the same command works for a steady-winding circle whose defect cores are
//! under a lattice spacing across and for an epitrochoid whose cores are ten.
//!
//! Negative defects are reported but excluded from the braid. On an epitrochoid
//! each regularised cusp pins a `-1/2` defect, which is part of the effective
//! boundary rather than a braid strand.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use volterra_braid::{
    BraidWord, Defect, Worldline, detect_defects_winding, extract_braidword, track,
};

fn read_mask(path: &Path) -> Result<Vec<bool>, Box<dyn Error>> {
    Ok(fs::read_to_string(path)?
        .split_whitespace()
        .map(|t| t != "0")
        .collect())
}

fn read_frame(path: &Path) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut qxx = Vec::new();
    let mut qxy = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            continue;
        };
        qxx.push(a.parse::<f64>()?);
        qxy.push(b.parse::<f64>()?);
    }
    Ok((qxx, qxy))
}

/// Maximal runs of a constant count, latest first, as `(start, end, count)`.
fn trailing_runs(counts: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut i = counts.len() as i64 - 1;
    while i >= 0 {
        let mut j = i;
        while j > 0 && counts[(j - 1) as usize] == counts[i as usize] {
            j -= 1;
        }
        runs.push((j as usize, i as usize, counts[i as usize]));
        i = j - 1;
    }
    runs
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(run_dir) = args.get(1).map(PathBuf::from) else {
        eprintln!("usage: analyse_run <run_dir> [--min-window N]");
        std::process::exit(2);
    };
    let min_window: usize = args
        .iter()
        .position(|a| a == "--min-window")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let mask = read_mask(&run_dir.join("mask.txt"))?;
    let n = mask.len();
    let lx = (n as f64).sqrt().round() as usize;
    if lx * lx != n {
        return Err(format!("mask has {n} cells, which is not a square grid").into());
    }
    let ly = lx;

    let mut frames: Vec<PathBuf> = fs::read_dir(run_dir.join("Q"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    frames.sort();
    if frames.is_empty() {
        return Err(format!("no Q frames under {}", run_dir.join("Q").display()).into());
    }

    let mut positive: Vec<Vec<Defect>> = Vec::with_capacity(frames.len());
    let mut negative: Vec<Vec<Defect>> = Vec::with_capacity(frames.len());
    for f in &frames {
        let (qxx, qxy) = read_frame(f)?;
        let found = detect_defects_winding(&qxx, &qxy, lx, ly, &mask);
        positive.push(found.iter().copied().filter(|d| d.charge > 0).collect());
        negative.push(found.into_iter().filter(|d| d.charge < 0).collect());
    }

    let pos_counts: Vec<usize> = positive.iter().map(Vec::len).collect();
    let neg_counts: Vec<usize> = negative.iter().map(Vec::len).collect();
    println!(
        "{} frames on a {lx}x{ly} grid, {} interior cells, sqrt(A_sys)={:.3}",
        frames.len(),
        mask.iter().filter(|&&b| b).count(),
        (mask.iter().filter(|&&b| b).count() as f64).sqrt()
    );

    // Counts as a histogram rather than a per-frame list: at several hundred
    // frames the list is unreadable and what matters is which count dominates.
    let histogram = |counts: &[usize]| {
        let mut h: BTreeMap<usize, usize> = BTreeMap::new();
        for &c in counts {
            *h.entry(c).or_default() += 1;
        }
        h.iter()
            .map(|(c, n)| format!("{c}x{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("  +1/2 per frame (count x frames): {}", histogram(&pos_counts));
    println!("  -1/2 per frame (count x frames): {}", histogram(&neg_counts));

    // Where the negatives sit over the trailing half. A cusp-pinned defect
    // barely moves, so a large spread is evidence against pinning.
    let tail = &negative[negative.len() / 2..];
    let centre = (lx as f64 / 2.0 - 1.0, ly as f64 / 2.0 - 1.0);
    let radii: Vec<f64> = tail
        .iter()
        .flatten()
        .map(|d| ((d.pos[0] - centre.0).powi(2) + (d.pos[1] - centre.1).powi(2)).sqrt())
        .collect();
    if !radii.is_empty() {
        let mean = radii.iter().sum::<f64>() / radii.len() as f64;
        let sd = (radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64).sqrt();
        println!(
            "  -1/2 radial position over the trailing half: mean {mean:.1}, sd {sd:.1} px \
             (domain radius ~{:.0})",
            lx as f64 / 2.0 - 1.0
        );
    }

    let Some(&(start, end, count)) = trailing_runs(&pos_counts)
        .iter()
        .find(|&&(s, e, c)| e - s + 1 >= min_window && c >= 2)
    else {
        println!("no run of >={min_window} frames with a stable +1/2 count of at least 2");
        std::process::exit(3);
    };
    println!(
        "  braiding frames [{start}, {end}] ({} frames), {count} mobile +1/2 defects",
        end - start + 1
    );

    let worldlines: Vec<Worldline> = track(&positive[start..=end]);
    let word: BraidWord = extract_braidword(&worldlines);
    let period = word.period_word();

    println!("n_strands={}", word.n_strands);
    println!(
        "window word ({} generators, {:.2} periods)",
        word.gens.len(),
        if period.gens.is_empty() {
            0.0
        } else {
            word.gens.len() as f64 / period.gens.len() as f64
        }
    );
    if period.gens.is_empty() {
        println!("braid word: {{}} (trivial)");
    } else {
        println!("braid word: {{{}}}", format_word(&period));
    }
    println!("topological entropy: {:.6}", word.entropy_per_period());
    println!(
        "whole-window entropy (not comparable to a published value): {:.6}",
        word.topological_entropy()
    );
    Ok(())
}

fn format_word(w: &BraidWord) -> String {
    w.gens
        .iter()
        .map(|g| {
            if g.inverse {
                format!("sigma_{}^-1", g.index)
            } else {
                format!("sigma_{}", g.index)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
