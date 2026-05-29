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

fn main() {
    let n_frames = 120usize;
    let cx = (LX / 2 - 1) as f64;

    // Three defects sliding along grid-y with a small grid-x wobble (enough to
    // exercise detection + tracking); exact braid is irrelevant for timing.
    let mut grids = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let t = f as f64 / n_frames as f64;
        let defects = [
            (cx + 6.0 * (t * std::f64::consts::TAU).sin(), cx - 22.0),
            (cx - 6.0 * (t * std::f64::consts::TAU).sin(), cx),
            (cx + 6.0 * (t * std::f64::consts::TAU).cos(), cx + 22.0),
        ];
        grids.push(render(&defects));
    }
    let mask = interior_mask();

    // Warm-up.
    for (qxx, qxy) in grids.iter().take(5) {
        let _ = detect_defects(qxx, qxy, LX, LX, THRESHOLD, &mask);
    }

    let t0 = Instant::now();
    let mut frames = Vec::with_capacity(n_frames);
    for (qxx, qxy) in &grids {
        frames.push(detect_defects(qxx, qxy, LX, LX, THRESHOLD, &mask));
    }
    let t_det = t0.elapsed();

    let t1 = Instant::now();
    let word = extract_braidword(&track(&frames));
    let t_word = t1.elapsed();

    let sites = (n_frames * LX * LX) as f64;
    let det_s = t_det.as_secs_f64();
    println!("Native braid pipeline: {n_frames} frames at {LX}x{LX}");
    println!(
        "  detection:   {:8.2} ms total, {:7.2} us/frame, {:6.2} ns/site",
        det_s * 1e3,
        det_s / n_frames as f64 * 1e6,
        det_s / sites * 1e9,
    );
    println!(
        "  track+word:  {:8.3} ms total ({} generators over {} frames)",
        t_word.as_secs_f64() * 1e3,
        word.gens.len(),
        n_frames,
    );
}
