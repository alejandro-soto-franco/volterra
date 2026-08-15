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
        // Naming the file matters: pointed at a run still in flight, this hits
        // the frame currently being written, and the failure should say so
        // rather than read as a diverged field.
        let parse = |t: &str| -> Result<f64, Box<dyn Error>> {
            t.parse::<f64>()
                .map_err(|e| format!("{}: {e} on {t:?}", path.display()).into())
        };
        qxx.push(parse(a)?);
        qxy.push(parse(b)?);
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
    for (i, f) in frames.iter().enumerate() {
        let (qxx, qxy) = match read_frame(f) {
            Ok(v) => v,
            // Only the newest frame is allowed to be unreadable, and only
            // because pointing this at a run still in flight catches the frame
            // being written. Anywhere earlier, an unreadable frame is a
            // corrupted one and the run should be looked at, not analysed.
            Err(e) if i + 1 == frames.len() => {
                println!("  dropping the newest frame, still being written: {e}");
                frames.pop();
                break;
            }
            Err(e) => return Err(e),
        };
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

    let transient = frames.len() / 4;

    // Two ways to choose the frames the braid is read from.
    //
    // The default is the longest stable-count stretch past the initial
    // transient, rather than the latest one: the last stable stretch can be a
    // short tail that happens to end the run. Ties go to the later stretch.
    //
    // --bridge-interruptions instead keeps every post-transient frame whose
    // count equals the modal count and drops the rest. arXiv:2503.10880 reports
    // most epitrochoid braids as interrupted, each +1/2 defect being
    // pair-annihilated at a cusp and re-emitted at the same place a moment
    // later, and rectifies this by "permitting that the newly created defect
    // carries on the same braid strands as the recently annihilated defect".
    // Dropping the frames where the count dips and letting the tracker resume
    // is that rectification: the nearest previous strand to a defect emitted at
    // a cusp is the one that vanished into it. The count of dropped frames is
    // printed, since the rectification is only defensible while it stays small.
    let (window, start, end, count): (Vec<Vec<Defect>>, usize, usize, usize) =
        if args.iter().any(|a| a == "--bridge-interruptions") {
            let mut modal: BTreeMap<usize, usize> = BTreeMap::new();
            for &c in &pos_counts[transient..] {
                if c >= 2 {
                    *modal.entry(c).or_default() += 1;
                }
            }
            let Some((&c, &held)) = modal.iter().max_by_key(|&(_, n)| *n) else {
                println!("no post-transient frame carries two or more +1/2 defects");
                std::process::exit(3);
            };
            let kept: Vec<usize> = (transient..pos_counts.len())
                .filter(|&i| pos_counts[i] == c)
                .collect();
            let dropped = pos_counts.len() - transient - kept.len();
            println!(
                "  bridging interruptions: {held} frames at the modal count {c}, \
                 {dropped} dropped ({:.0}% of the post-transient run)",
                100.0 * dropped as f64 / (pos_counts.len() - transient) as f64
            );
            (
                kept.iter().map(|&i| positive[i].clone()).collect(),
                kept[0],
                *kept.last().unwrap(),
                c,
            )
        } else {
            let mut candidates: Vec<(usize, usize, usize)> = trailing_runs(&pos_counts)
                .into_iter()
                .filter(|&(s, e, c)| e >= transient && e - s + 1 >= min_window && c >= 2)
                .collect();
            candidates.sort_by_key(|&(s, e, _)| (e - s, e));
            let Some(&(s, e, c)) = candidates.last() else {
                println!("no run of >={min_window} frames with a stable +1/2 count of at least 2");
                std::process::exit(3);
            };
            (positive[s..=e].to_vec(), s, e, c)
        };
    println!(
        "  braiding frames [{start}, {end}] ({} frames), {count} mobile +1/2 defects",
        window.len()
    );

    let window = &window[..];
    report_period(window);

    // The braid word depends on the projection direction; the topological
    // entropy of a genuinely pseudo-Anosov stirring does not, as long as the
    // projection is generic. A degenerate direction, one along which two
    // strands stay nearly collinear, drops crossings and reads low. Scanning
    // the angle separates the two: a real braid holds its entropy over a range
    // of angles, and an artefact of one projection does not.
    if args.iter().any(|a| a == "--axis-scan") {
        println!("  projection scan (angle, period length, entropy per period):");
        for k in 0..12 {
            let theta = std::f64::consts::PI * k as f64 / 12.0;
            let rotated: Vec<Vec<Defect>> = window.iter().map(|f| rotate(f, theta)).collect();
            let w = extract_braidword(&track(&rotated));
            let p = w.period_word();
            println!(
                "    {:>5.1} deg  {:>3} gens  h = {:.6}  {{{}}}",
                theta.to_degrees(),
                p.gens.len(),
                w.entropy_per_period(),
                format_word(&p)
            );
        }
    }

    let worldlines: Vec<Worldline> = track(window);
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

/// Rotate a frame's defect positions by `theta` about the origin.
///
/// `extract_braidword` projects onto the first coordinate, so rotating the
/// positions is how the projection direction is chosen. Tracking is by distance
/// and so is unaffected.
fn rotate(frame: &[Defect], theta: f64) -> Vec<Defect> {
    let (s, c) = theta.sin_cos();
    frame
        .iter()
        .map(|d| Defect {
            pos: [
                c * d.pos[0] + s * d.pos[1],
                -s * d.pos[0] + c * d.pos[1],
            ],
            charge: d.charge,
        })
        .collect()
}

/// Report the dominant period of the defect configuration, in frames.
///
/// Braid extraction is only as good as the sampling: a braid word is read from
/// the order of the strands changing, so a run sampled at four frames per
/// period cannot resolve a period carrying four generators, however long the
/// run is. This measures the period without tracking anything, so it is
/// independent of the extraction it is there to qualify.
///
/// The observable is the mean pairwise distance among the `+1/2` defects, which
/// is invariant under relabelling them and so needs no worldlines. Its
/// autocorrelation peaks at the period.
fn report_period(window: &[Vec<Defect>]) {
    let spread: Vec<f64> = window
        .iter()
        .map(|frame| {
            let mut total = 0.0;
            let mut pairs = 0usize;
            for i in 0..frame.len() {
                for j in i + 1..frame.len() {
                    let dx = frame[i].pos[0] - frame[j].pos[0];
                    let dy = frame[i].pos[1] - frame[j].pos[1];
                    total += (dx * dx + dy * dy).sqrt();
                    pairs += 1;
                }
            }
            if pairs == 0 { 0.0 } else { total / pairs as f64 }
        })
        .collect();
    if spread.len() < 12 {
        return;
    }
    let mean = spread.iter().sum::<f64>() / spread.len() as f64;
    let dev: Vec<f64> = spread.iter().map(|v| v - mean).collect();
    let norm: f64 = dev.iter().map(|v| v * v).sum();
    if norm <= 0.0 {
        println!("  defect configuration is stationary: no period to resolve");
        return;
    }
    let max_lag = (spread.len() / 3).min(200);
    let mut best = (0usize, 0.0f64);
    for lag in 2..max_lag {
        let r: f64 = (0..dev.len() - lag).map(|i| dev[i] * dev[i + lag]).sum::<f64>() / norm;
        if r > best.1 {
            best = (lag, r);
        }
    }
    let rel_spread = (norm / spread.len() as f64).sqrt() / mean;
    if best.1 < 0.2 {
        println!(
            "  no clear period: strongest autocorrelation {:.2} at lag {} frames, \
             configuration varying by {:.0}% of its mean",
            best.1,
            best.0,
            100.0 * rel_spread
        );
    } else {
        println!(
            "  period about {} frames (autocorrelation {:.2}), configuration varying by \
             {:.0}% of its mean",
            best.0,
            best.1,
            100.0 * rel_spread
        );
    }
}
