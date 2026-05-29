//! Native throughput benchmark for the braid-analysis pipeline (no Python/PyO3).
//!
//! Renders Q-tensor grids of a moving 3-defect configuration, then times defect
//! detection, tracking, and braid-word extraction in pure Rust.
//!
//! Run: `cargo run --release --example bench_braid -p volterra-braid`

use std::time::Instant;
use volterra_braid::{detect_defects, extract_braidword, track};

const LX: usize = 100;
const THRESHOLD: f64 = 0.1;

/// Superpose +1/2 winding fields at `defects` -> (qxx, qxy), row-major x*LX+y.
fn render(defects: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut qxx = vec![0.0; LX * LX];
    let mut qxy = vec![0.0; LX * LX];
    for x in 0..LX {
        for y in 0..LX {
            let mut ang = 0.0;
            for &(px, py) in defects {
                ang += (y as f64 - py).atan2(x as f64 - px);
            }
            qxx[x * LX + y] = ang.cos();
            qxy[x * LX + y] = ang.sin();
        }
    }
    (qxx, qxy)
}

/// One-cell border ring zeroed (true = active interior), matching the harness.
fn interior_mask() -> Vec<bool> {
    let r = (LX / 2 - 1) as f64;
    let c = (LX / 2 - 1) as f64;
    let mut m = vec![false; LX * LX];
    for x in 0..LX {
        for y in 0..LX {
            let inside = (x as f64 - c).powi(2) + (y as f64 - c).powi(2) <= r * r;
            let border = x == 0 || y == 0 || x == LX - 1 || y == LX - 1;
            m[x * LX + y] = inside && !border;
        }
    }
    m
}

/// `n_defects` sliding along grid-y (spaced), each with a small grid-x wobble.
/// Exact braid is irrelevant for timing; this just exercises detection +
/// tracking + extraction with the right number of defects.
fn make_grids(n_defects: usize, n_frames: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let cx = (LX / 2 - 1) as f64;
    let span = (LX as f64) * 0.44;
    let mut grids = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let t = f as f64 / n_frames as f64;
        let defects: Vec<(f64, f64)> = (0..n_defects)
            .map(|k| {
                let gy = cx + (k as f64 - (n_defects as f64 - 1.0) / 2.0) * span / n_defects as f64;
                let phase = (k as f64) * 1.7 + t * std::f64::consts::TAU;
                (cx + 6.0 * phase.sin(), gy)
            })
            .collect();
        grids.push(render(&defects));
    }
    grids
}

fn bench(label: &str, n_defects: usize, n_frames: usize, mask: &[bool]) {
    let grids = make_grids(n_defects, n_frames);

    for (qxx, qxy) in grids.iter().take(5) {
        let _ = detect_defects(qxx, qxy, LX, LX, THRESHOLD, mask);
    }

    let t0 = Instant::now();
    let mut frames = Vec::with_capacity(n_frames);
    for (qxx, qxy) in &grids {
        frames.push(detect_defects(qxx, qxy, LX, LX, THRESHOLD, mask));
    }
    let t_det = t0.elapsed();

    let t1 = Instant::now();
    let word = extract_braidword(&track(&frames));
    let t_word = t1.elapsed();

    let sites = (n_frames * LX * LX) as f64;
    let det_s = t_det.as_secs_f64();
    let counts: std::collections::BTreeSet<usize> = frames.iter().map(|f| f.len()).collect();
    println!("\n=== {label} ({n_defects} defects) ===");
    println!(
        "  detection:   {:8.2} ms total, {:7.2} us/frame, {:6.2} ns/site  (defects/frame {:?})",
        det_s * 1e3,
        det_s / n_frames as f64 * 1e6,
        det_s / sites * 1e9,
        counts,
    );
    println!(
        "  track+word:  {:8.3} ms total ({} generators over {} frames)",
        t_word.as_secs_f64() * 1e3,
        word.gens.len(),
        n_frames,
    );
}

fn main() {
    let n_frames = 120usize;
    let mask = interior_mask();
    println!("Native braid pipeline: {n_frames} frames at {LX}x{LX}");
    bench("golden", 3, n_frames, &mask);
    bench("silver", 4, n_frames, &mask);
}
