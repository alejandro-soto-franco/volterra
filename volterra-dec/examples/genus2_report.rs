//! Is the defect motion on a genus-2 surface periodic?
//!
//! The sphere's four `+1/2` defects orbit periodically at low activity. A
//! genus-2 surface is forced to carry four `-1/2` instead, and whether THOSE
//! settle into a repeating configuration is a different question with no
//! reason to have the same answer.
//!
//! The measure is the shape period: the autocorrelation of the sorted pairwise
//! separations, which is invariant under relabelling and under any rigid
//! motion, so a configuration that precesses while repeating its shape still
//! registers. Separations are straight-line distances rather than angles at
//! the origin, since the defects do not lie on a sphere.
//!
//!     cargo run --release -p volterra-dec --example genus2_report -- <run-dir> [--from=F]

use std::path::Path;
use volterra_braid::sphere::{track_with, Separation, SphereFrame};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut from = 0.2_f64;
    let mut dir = String::new();
    for a in &args {
        if let Some(v) = a.strip_prefix("--from=") {
            from = v.parse().expect("invalid --from");
        } else {
            dir = a.clone();
        }
    }
    let run = Path::new(&dir);

    // defects.csv: step, t, x, y, z, charge
    let text = std::fs::read_to_string(run.join("defects.csv")).expect("no defects.csv");
    let mut by_time: Vec<(f64, Vec<([f64; 3], i32)>)> = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            continue;
        }
        let t: f64 = f[1].parse().unwrap();
        let p = [
            f[2].parse::<f64>().unwrap(),
            f[3].parse::<f64>().unwrap(),
            f[4].parse::<f64>().unwrap(),
        ];
        let c: i32 = f[5].parse().unwrap();
        match by_time.last_mut() {
            Some((tt, v)) if (*tt - t).abs() < 1e-9 => v.push((p, c)),
            _ => by_time.push((t, vec![(p, c)])),
        }
    }
    let t_end = by_time.last().map(|x| x.0).unwrap_or(0.0);
    let frames: Vec<SphereFrame> = by_time
        .into_iter()
        .filter(|(t, _)| *t >= from * t_end)
        .collect();

    if frames.len() < 16 {
        println!("only {} frames in the window; nothing to measure", frames.len());
        return;
    }
    let counts: Vec<usize> = frames.iter().map(|(_, v)| v.len()).collect();
    let n_min = *counts.iter().min().unwrap();
    let n_max = *counts.iter().max().unwrap();
    let charge: i32 = frames[0].1.iter().map(|d| d.1).sum();
    println!("=== {} ===", run.display());
    println!(
        "  window t in [{:.1}, {:.1}], {} frames, {} to {} defects, total charge {charge}",
        frames[0].0, t_end, frames.len(), n_min, n_max
    );
    if n_min != n_max {
        println!("  the census moves, so the shape is not a fixed configuration");
    }

    // A generous cap: the defects move slowly against the mesh, and a cap that
    // is too tight silently drops the run to a shorter window.
    let w = match track_with(&frames, 0.5, Separation::Chord) {
        Some(w) => w,
        None => {
            println!("  tracking failed");
            return;
        }
    };
    println!("  tracked {} strands over {} frames", w.n_strands(), w.n_frames());

    let (p, q) = w.shape_period_with(Separation::Chord);
    if p.is_finite() {
        println!("  shape period T = {p:.3} (repeat quality {q:.3})");
        if q < 0.5 {
            println!("  the repeat is weak, so this is not a periodic orbit");
        }
    } else {
        println!("  no shape period found");
    }

    let sig = w.shape_signature_with(0, Separation::Chord);
    print!("  separations at the first frame:");
    for v in &sig {
        print!(" {v:.4}");
    }
    println!();
}
