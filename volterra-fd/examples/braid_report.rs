//! The defect braid of a periodic run, on the torus.
//!
//! Reads a run directory written by `periodic_active_nematic` and reports the
//! braid its defects write, after Mitchell, Sabbir, Geumhan, Smith, Klein and
//! Beller, "Maximally mixing active nematics", Phys. Rev. E 109, 014606 (2024).
//!
//! The paper's claim about `ell_a = 3` is that the four defects settle into a
//! periodic orbit whose two `+1/2` defects trace bounded circular paths and
//! "repeatedly encounter and revolve around each other counterclockwise, with
//! four such encounters during each orbit", which is the maximal mixing braid of
//! its Fig. 2a. That braid's entropy per operation is `log(phi + sqrt phi)`, so
//! with the measured period it predicts
//!
//! ```text
//! h_tilde_max = log(phi + sqrt phi) / (T_tilde / 4).
//! ```
//!
//! Every part of that is measured here except the constant, which is quoted.
//!
//! # Window
//!
//! Tracking needs a fixed cast, so the report runs over the longest stretch of
//! frames whose defect count never changes. That stretch is taken from the
//! DEVELOPED state only: the longest one over a whole run lands in the quench,
//! where the census happens to sit still for a while and the defects wind around
//! the torus rather than orbiting, and a braid read there is the transient's,
//! not the attractor's. `--from` sets the fraction of the run to skip, default
//! half. A run whose census never holds for eight frames after that has no braid
//! to read and the report says so.
//!
//!     braid_report [--from=0.5] [--close=0.5] <run-dir> [<run-dir> ...]
//!
//! `--close` is the pass threshold as a fraction of the box side.

use std::fs;
use std::path::{Path, PathBuf};

use volterra_braid::torus::{
    TorusWorldlines, h_tepo_maximal_mixing, track_on_torus,
};

/// One observation: the defects seen at one time.
type Frame = (f64, Vec<([f64; 2], i32)>);

fn read_defects(run: &Path) -> std::io::Result<Vec<Frame>> {
    let text = fs::read_to_string(run.join("defects.csv"))?;
    let mut frames: Vec<Frame> = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 5 {
            continue;
        }
        let (t, x, y, c) = (
            f[1].parse::<f64>().unwrap_or(f64::NAN),
            f[2].parse::<f64>().unwrap_or(f64::NAN),
            f[3].parse::<f64>().unwrap_or(f64::NAN),
            f[4].parse::<i32>().unwrap_or(0),
        );
        if !t.is_finite() || !x.is_finite() || !y.is_finite() {
            continue;
        }
        match frames.last_mut() {
            Some(last) if (last.0 - t).abs() < 1e-12 => last.1.push(([x, y], c)),
            _ => frames.push((t, vec![([x, y], c)])),
        }
    }
    Ok(frames)
}

/// The longest stretch of frames that keeps the run's usual cast of defects.
///
/// The cast is the modal census over the frames offered. A stretch keeps it
/// while every charge is present at least that many times, so a frame with a
/// short-lived extra pair stays in and a frame that has lost a member of the
/// cast ends the stretch. The stretch also starts on a frame whose census is
/// exactly the cast, so the tracker is seeded without a transient in it.
///
/// Demanding an exactly constant census instead splits a developed orbit
/// wherever the detector flickers. On a run of four defects across 35 time
/// units, with 78 per cent of frames at exactly two and two, that rule returned
/// a window of 54 frames spanning 1.3 time units.
fn steady_window(frames: &[Frame]) -> (usize, usize) {
    let census = |f: &Frame| {
        let p = f.1.iter().filter(|d| d.1 > 0).count();
        (p, f.1.len() - p)
    };
    let mut tally: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for f in frames {
        *tally.entry(census(f)).or_default() += 1;
    }
    let Some((&cast, _)) = tally.iter().max_by_key(|&(k, v)| (*v, k.0 + k.1)) else {
        return (0, 0);
    };
    let keeps = |f: &Frame| {
        let c = census(f);
        c.0 >= cast.0 && c.1 >= cast.1
    };
    let (mut bi, mut bj) = (0usize, 0usize);
    let mut i = 0;
    while i < frames.len() {
        if census(&frames[i]) != cast {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < frames.len() && keeps(&frames[j + 1]) {
            j += 1;
        }
        if j - i >= bj - bi {
            bi = i;
            bj = j;
        }
        i = j + 1;
    }
    (bi, bj + 1)
}

/// Period of a series from the first interior maximum of its autocorrelation.
///
/// Returns `(T, peak)`; a peak near one is a periodic signal and a small one is
/// not. The same rule `analyse_periodic.py` reports on.
fn dominant_period(t: &[f64], y: &[f64]) -> (f64, f64) {
    let n = y.len();
    if n < 8 {
        return (f64::NAN, 0.0);
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let a0: f64 = d.iter().map(|v| v * v).sum();
    if a0 <= 0.0 {
        return (f64::NAN, 0.0);
    }
    let ac: Vec<f64> = (0..n)
        .map(|lag| (0..n - lag).map(|i| d[i] * d[i + lag]).sum::<f64>() / a0)
        .collect();
    let dt = t[1] - t[0];
    let mut i = 1;
    while i + 1 < n && ac[i] < ac[i - 1] {
        i += 1;
    }
    for j in i..n - 1 {
        if ac[j] >= ac[j - 1] && ac[j] >= ac[j + 1] {
            return (j as f64 * dt, ac[j]);
        }
    }
    (f64::NAN, ac[i..].iter().cloned().fold(0.0_f64, f64::max))
}

fn read_rms(run: &Path) -> std::io::Result<(Vec<f64>, Vec<f64>)> {
    let text = fs::read_to_string(run.join("stats.csv"))?;
    let mut t = Vec::new();
    let mut u = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 3 {
            continue;
        }
        if let (Ok(a), Ok(b)) = (f[1].trim().parse::<f64>(), f[2].trim().parse::<f64>()) {
            t.push(a);
            u.push(b);
        }
    }
    Ok((t, u))
}

fn report(run: &Path, from: f64, close_frac: f64) -> std::io::Result<()> {
    let cfg: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(run.join("config.json"))?)?;
    let lx = cfg["params"]["lx"].as_f64().unwrap_or(100.0);
    let ly = cfg["params"]["ly"].as_f64().unwrap_or(lx);
    let t_a = cfg["active_time"].as_f64().unwrap_or(f64::NAN);
    let locking = !cfg["params"]["locking"].is_null();

    let all = read_defects(run)?;
    // The developed state only. See the note on the window above.
    let cut = match (all.first(), all.last()) {
        (Some(a), Some(b)) => a.0 + from * (b.0 - a.0),
        _ => 0.0,
    };
    let frames: Vec<Frame> = all.iter().filter(|f| f.0 >= cut).cloned().collect();
    let (i, j) = steady_window(&frames);
    let window = &frames[i..j];
    println!("\n=== {}", run.file_name().unwrap().to_string_lossy());
    if window.len() < 8 {
        println!(
            "    no steady window after t = {cut:.3}: the defect census never \
             stayed put for 8 frames"
        );
        return Ok(());
    }
    let n_plus = window[0].1.iter().filter(|d| d.1 > 0).count();
    let n_minus = window[0].1.len() - n_plus;
    let exact = window
        .iter()
        .filter(|f| {
            let p = f.1.iter().filter(|d| d.1 > 0).count();
            p == n_plus && f.1.len() - p == n_minus
        })
        .count();
    println!(
        "    {lx:.0}x{ly:.0} torus, locking {}, steady window t in [{:.3}, {:.3}] \
         ({} frames) with {n_plus} +1/2 and {n_minus} -1/2",
        if locking { "on" } else { "off" },
        window[0].0,
        window[window.len() - 1].0,
        window.len()
    );
    println!(
        "    census      : {:.1} per cent of the window is exactly that cast, \
         the rest carries a transient pair",
        100.0 * exact as f64 / window.len() as f64
    );

    // A defect may not move further between frames than a defect can: the frame
    // spacing times a generous multiple of the RMS flow speed.
    let dt_frame = window[1].0 - window[0].0;
    let max_disp = (0.25 * lx).max(4.0 * dt_frame * 100.0);
    let Some(w) = track_on_torus(window, lx, ly, max_disp) else {
        println!("    tracking refused: the first frame was empty");
        return Ok(());
    };
    if w.n_frames() < window.len() {
        println!(
            "    tracking ended early at frame {} of {}",
            w.n_frames(),
            window.len()
        );
    }

    // The period, from the RMS velocity over the same window.
    let (ts, us) = read_rms(run)?;
    let sel: Vec<usize> = (0..ts.len())
        .filter(|&k| ts[k] >= window[0].0 && ts[k] <= window[window.len() - 1].0)
        .collect();
    let (period, peak) = if sel.len() >= 8 {
        let tt: Vec<f64> = sel.iter().map(|&k| ts[k]).collect();
        let uu: Vec<f64> = sel.iter().map(|&k| us[k]).collect();
        dominant_period(&tt, &uu)
    } else {
        (f64::NAN, 0.0)
    };

    // A pass is a local minimum of the separation below this. The braid of
    // Fig. 2(a) swaps at `0.293 L`, so the threshold sits above that; a
    // threshold at the box half-width also admits the shallow minima between
    // passes and reads nine an orbit where the braid has four.
    let close = close_frac * lx;
    // The orbit period is the strands' own recurrence, not the dominant period
    // of the RMS velocity: the velocity peaks at every pass, so on the braid of
    // Fig. 2(a) it cycles four times an orbit. Reading it as the orbit period
    // divides `T_tilde` by four and multiplies the braid prediction by four.
    let (orbit_t, orbit_q) = w.orbit_period();
    let t_orbit = if orbit_t.is_finite() && orbit_q > 0.3 { orbit_t } else { period };
    let m = w.is_maximal_mixing(t_orbit, close);
    let enc = w.encounters_apart(close, t_orbit / 8.0);
    let gyr = w.gyration();
    let wind = w.winding();

    println!(
        "    period      : orbit T = {t_orbit:.4} from the worldlines \
         (recurrence {orbit_q:.3}), T_tilde = {:.1}",
        t_orbit / t_a
    );
    println!(
        "                  u_rms cycles at {period:.4} \
         (autocorrelation {peak:.4}), {:.2} a revolution",
        if period > 0.0 { t_orbit / period } else { f64::NAN }
    );
    println!(
        "    encounters  : {} over the window, {:.2} per period, \
         sense {} ({:.0} per cent agreeing)",
        m.encounters,
        m.per_period,
        m.sense,
        100.0 * m.sense_fraction
    );
    for &s in w.positive().iter() {
        println!(
            "      strand {s}: gyration {:.2}, winding [{:.2}, {:.2}]",
            gyr[s], wind[s][0], wind[s][1]
        );
    }
    println!(
        "    verdict     : {}",
        if m.verdict {
            "the maximal mixing braid of Fig. 2a"
        } else if m.n_positive != 2 {
            "not the maximal mixing braid: the cast is not two +1/2 defects"
        } else if !m.bounded {
            "not the maximal mixing braid: an orbit is unbounded or winds"
        } else if !m.one_sense {
            "not the maximal mixing braid: the passes are of mixed sense"
        } else {
            "not the maximal mixing braid: the encounter rate is not four a period"
        }
    );
    let pred = TorusWorldlines::braid_prediction(t_orbit, t_a);
    println!(
        "    prediction  : h_tilde_max = log(phi + sqrt phi) / (T_tilde / 4) = {pred:.4e}"
    );

    // Subsample the worldlines so the file stays a plotting input rather than a
    // copy of the run.
    let stride = (w.n_frames() / 4000).max(1);
    let keep: Vec<usize> = (0..w.n_frames()).step_by(stride).collect();
    let out = serde_json::json!({
        "run": run.file_name().unwrap().to_string_lossy(),
        "lx": lx, "ly": ly, "t_a": t_a, "locking": locking,
        "window": [window[0].0, window[window.len() - 1].0],
        "close": close,
        "from": from,
        "n_plus": n_plus, "n_minus": n_minus,
        "exact_census_fraction": exact as f64 / window.len() as f64,
        "charge": w.charge,
        "times": keep.iter().map(|&k| w.times[k]).collect::<Vec<_>>(),
        "worldlines": (0..w.n_strands())
            .map(|s| keep.iter().map(|&k| w.pts[k][s]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "encounters": enc.iter().map(|e| serde_json::json!({
            "t": e.t, "strands": [e.strands.0, e.strands.1],
            "image": e.image, "distance": e.distance, "sense": e.sense,
        })).collect::<Vec<_>>(),
        "period": t_orbit, "period_peak": peak,
        "orbit_recurrence": orbit_q,
        "u_rms_period": period, "t_tilde": t_orbit / t_a,
        "encounters_per_period": m.per_period,
        "one_sense": m.one_sense, "sense": m.sense,
        "sense_fraction": m.sense_fraction, "bounded": m.bounded,
        "verdict": m.verdict,
        "h_tepo": h_tepo_maximal_mixing(),
        "h_tilde_max": pred,
        "gyration": gyr, "winding": wind,
    });
    fs::write(run.join("braid.json"), serde_json::to_string_pretty(&out).unwrap())?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut from = 0.5;
    let mut close_frac = 0.5;
    let mut args: Vec<PathBuf> = Vec::new();
    for a in raw {
        if let Some(v) = a.strip_prefix("--from=") {
            from = v.parse().unwrap_or(0.5);
        } else if let Some(v) = a.strip_prefix("--close=") {
            close_frac = v.parse().unwrap_or(0.5);
        } else {
            args.push(PathBuf::from(a));
        }
    }
    if args.is_empty() {
        eprintln!("braid_report [--from=0.5] [--close=0.5] <run-dir> [<run-dir> ...]");
        std::process::exit(2);
    }
    for run in &args {
        if let Err(e) = report(run, from, close_frac) {
            println!("\n=== {}\n    {e}", run.display());
        }
    }
    println!();
    Ok(())
}
