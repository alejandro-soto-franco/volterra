//! Braid reading of a defect worldline series, windowed and swept over axes.
//!
//! `analyse_run` reads Q fields from text dumps, which makes the braid analysis
//! depend on the bulkiest and least reusable part of a run. The braid only needs
//! the defect positions, so this takes those directly and the fields can be
//! deleted.
//!
//! Three things move a topological entropy read off worldlines, and all three
//! have to be reported with it:
//!
//!   * the projection axis, since the word depends on the direction the strands
//!     are ordered along, and a real braid holds its value over a range of them;
//!   * the window length, which has to be set in physical time, not frames;
//!   * whether the window holds a whole period, since `entropy_per_period`
//!     returns the whole word's entropy when it cannot factor, and that climbs
//!     with the window while looking stable.
//!
//! So the output is per window: every axis reading, the modal value, and how
//! many axes hold it.
//!
//! Input is `defects.tsv` as written by `run_stats.py`: one line per defect,
//! tab or space separated, `frame x y charge`, ascending in frame.
//!
//!     braid_series <defects.tsv> [--window N] [--stride N] [--axes 12]
//!                  [--charge 1] [--min-strands 3]
//!                  [--core] [--max-step 12] [--max-over 0.01]
//!
//! One JSON object per line on stdout, then a summary object.

use std::collections::BTreeMap;
use std::fs;

use volterra_braid::{
    braidword::BraidWord,
    defect::Defect,
    extract_braidword,
    track::track_core,
};

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn arg_usize(name: &str, default: usize) -> usize {
    arg(name, &default.to_string()).parse().unwrap_or(default)
}

/// Frames of defects of the requested charge, indexed by frame number.
///
/// Frames absent from the file are frames with no defect of that charge; they
/// are kept as empty so a window's length in frames stays a length in time.
fn read_frames(path: &str, charge: i8) -> Vec<Vec<Defect>> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut by_frame: BTreeMap<usize, Vec<Defect>> = BTreeMap::new();
    let mut max_frame = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            panic!("expected `frame x y charge`, got: {line}");
        }
        let frame: usize = f[0].parse().expect("frame");
        max_frame = max_frame.max(frame);
        let q: i32 = f[3].parse::<f64>().expect("charge") as i32;
        if q as i8 != charge {
            continue;
        }
        by_frame.entry(frame).or_default().push(Defect {
            pos: [f[1].parse().expect("x"), f[2].parse().expect("y")],
            charge,
        });
    }
    (0..=max_frame)
        .map(|i| by_frame.remove(&i).unwrap_or_default())
        .collect()
}

fn rotate(frames: &[Vec<Defect>], theta: f64) -> Vec<Vec<Defect>> {
    let (c, s) = (theta.cos(), theta.sin());
    frames
        .iter()
        .map(|f| {
            f.iter()
                .map(|d| Defect {
                    pos: [
                        c * d.pos[0] + s * d.pos[1],
                        -s * d.pos[0] + c * d.pos[1],
                    ],
                    charge: d.charge,
                })
                .collect()
        })
        .collect()
}

/// Longest cycle of a permutation, which is 1 exactly when it is the identity.
fn longest_cycle(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut seen = vec![false; n];
    let mut best = 0usize;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut len = 0usize;
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            i = perm[i];
            len += 1;
        }
        best = best.max(len);
    }
    best
}

/// Round to a fixed number of places so equal readings compare equal.
fn key(x: f64) -> i64 {
    (x * 1e6).round() as i64
}

struct AxisRead {
    theta_deg: f64,
    entropy: f64,
    per_period: f64,
    period: usize,
    gens: usize,
    word: Vec<i32>,
    /// Whether the braid returns every strand to its own starting place.
    ///
    /// This is the discrete question underneath the entropy, and it is the one
    /// that survives a bad window. Defects can travel a long way and still never
    /// exchange: a rigidly rotating or breathing configuration traces a braid
    /// whose permutation is the identity and whose entropy is therefore zero, no
    /// matter how fast it moves. Six seeds at the paper's stable-silver point do
    /// exactly that, with four defects covering 33 to 82 lattice units in a domain
    /// of radius 49 and reading zero entropy at every axis. Reporting the
    /// permutation separates "did not braid" from "the window was wrong".
    identity_permutation: bool,
    /// Length of the longest cycle, so a partial exchange is visible.
    longest_cycle: usize,
}

struct WindowRead {
    reads: Vec<AxisRead>,
    strands: usize,
    worst_step: f64,
    steps_over: usize,
    steps: usize,
}

fn read_window(frames: &[Vec<Defect>], axes: usize, core: Option<f64>) -> Option<WindowRead> {
    let mut out = Vec::with_capacity(axes);
    let mut strands = 0usize;
    let mut worst_step = 0.0f64;
    let mut steps_over = 0usize;
    let mut steps = 0usize;
    for i in 0..axes {
        // Axes over a half turn: ordering along theta and along theta + pi give
        // the same braid read backwards, so a full turn would double every
        // reading.
        let theta = std::f64::consts::PI * i as f64 / axes as f64;
        let rot = rotate(frames, theta);
        // The core is extracted per axis rather than once, because the rotation
        // is applied to the frames and the tracker is run on the rotated copy.
        // A rotation is an isometry, so the nearest-neighbour matching and the
        // step bound are unchanged by it and every axis recovers the same core;
        // running it inside the loop keeps that an outcome rather than an
        // assumption, and the strand count is asserted equal across axes below.
        let word = match core {
            None => BraidWord::from_frames(&rot),
            Some(max_step) => {
                let (wls, worst, over, n) = track_core(&rot, max_step)?;
                if strands != 0 && wls.len() != strands {
                    return None;
                }
                strands = wls.len();
                worst_step = worst_step.max(worst);
                steps_over = steps_over.max(over);
                steps = n;
                extract_braidword(&wls)
            }
        };
        let perm = word.permutation();
        let ident = perm.iter().enumerate().all(|(i, &p)| i == p);
        let longest = longest_cycle(&perm);
        out.push(AxisRead {
            theta_deg: theta.to_degrees(),
            entropy: word.topological_entropy(),
            per_period: word.entropy_per_period(),
            period: word.fundamental_period().len(),
            gens: word.codes().len(),
            word: word.period_word().codes(),
            identity_permutation: ident,
            longest_cycle: longest,
        });
    }
    Some(WindowRead {
        reads: out,
        strands,
        worst_step,
        steps_over,
        steps,
    })
}

/// Read the crate's own golden and silver orbits through the same windowing and
/// axis sweep the real runs go through, and check the known values come back.
///
/// The point is not to test `topological_entropy`, which has its own tests, but
/// to test this program: the rotation, the windowing, the modal vote and the
/// count handling are what stand between a worldline file and a claimed entropy,
/// and a synthetic orbit is the only input whose answer is known beforehand.
fn self_test(axes: usize) -> i32 {
    use volterra_braid::synthetic::{RealizeOpts, golden_orbit, silver_orbit};

    let golden = 2.0 * ((1.0 + 5.0_f64.sqrt()) / 2.0).ln();
    let silver = (3.0 + 2.0 * 2.0_f64.sqrt()).ln();
    let mut bad = 0;

    // Windows are built in whole periods, because the number of periods is what
    // decides whether the reading is trustworthy, and in both of the two shapes
    // a window can take: cut on a period boundary, and cut with a part-period
    // tail. A sliding window over a real run does not know the period, so the
    // tail case is the one that matters, and it is where a short window fails.
    //
    // Four is the measured threshold. The golden braid survives three periods in
    // both shapes and fails at two whole ones; the silver survives three cut on a
    // boundary but reads 5.823037 at three plus a half-period tail, which is
    // 3.3 times the true value and so is a word that never factored rather than
    // an accumulated one. Anything below four periods is therefore untrustworthy
    // for the silver, and the silver is the braid this study is chasing.
    const REQUIRED_PERIODS: usize = 4;
    for (name, orbit, want) in [
        ("golden", golden_orbit as fn(&RealizeOpts) -> Vec<Vec<Defect>>, golden),
        ("silver", silver_orbit as fn(&RealizeOpts) -> Vec<Vec<Defect>>, silver),
    ] {
        for periods in [8usize, 6, 4, 3, 2] {
            for tail in [0usize, 1] {
                let frames =
                    orbit(&RealizeOpts { frames_per_gen: 8, periods: periods + 1 });
                let per = frames.len() / (periods + 1);
                let window = (per * periods + tail * (per / 2)).min(frames.len());
                // Both readers are put through the same orbits. On a clean
                // synthetic series every frame holds the same defects, so the
                // persistent core is the whole configuration and the two must
                // agree; a core reader that changed the answer here would be
                // changing it on the runs too.
                for core in [None, Some(5.0f64)] {
                    let reads = read_window(&frames[..window], axes, core)
                        .expect("a reading")
                        .reads;

                let hit = reads
                    .iter()
                    .filter(|r| (r.per_period - want).abs() < 1e-6)
                    .count();
                let mut tally: BTreeMap<i64, usize> = BTreeMap::new();
                for r in &reads {
                    *tally.entry(key(r.per_period)).or_insert(0) += 1;
                }
                let (&mk, &mc) = tally.iter().max_by_key(|&(_, &c)| c).unwrap();

                // The test is on the modal reading, which is what the tool
                // reports. A majority of axes is a stricter thing and is not
                // required: oblique axes fail to factor an oblique word, and
                // that is a property of the projection rather than a defect in
                // the extraction.
                let ok = (mk as f64 / 1e6 - want).abs() < 1e-6;
                let required = periods >= REQUIRED_PERIODS;
                if !ok && required {
                    bad += 1;
                }
                let vals: Vec<String> =
                    reads.iter().map(|r| format!("{:.4}", r.per_period)).collect();
                println!(
                    "{{\"self_test\":\"{name}\",\"periods\":{periods},\
                     \"tail\":{tail},\"period_frames\":{per},\
                     \"window\":{window},\"want\":{want:.6},\"modal\":{:.6},\
                     \"axes_at_mode\":{mc},\"axes_at_want\":{hit},\
                     \"axes\":{axes},\"required\":{required},\"pass\":{ok},\
                     \"core\":{},\"per_axis\":[{}]}}",
                    mk as f64 / 1e6,
                    core.is_some(),
                    vals.join(",")
                );
                }
            }
        }
    }
    if bad == 0 {
        println!("{{\"self_test\":\"all\",\"pass\":true}}");
        0
    } else {
        println!("{{\"self_test\":\"all\",\"pass\":false,\"failures\":{bad}}}");
        1
    }
}

fn main() {
    if std::env::args().any(|a| a == "--self-test") {
        std::process::exit(self_test(arg_usize("--axes", 12)));
    }
    let path = std::env::args()
        .nth(1)
        .expect("usage: braid_series <defects.tsv> [--window N] ...");
    let charge: i8 = arg("--charge", "1").parse().expect("charge");
    let axes = arg_usize("--axes", 12);
    let min_strands = arg_usize("--min-strands", 3);
    // Read the persistent core rather than the first frame's configuration. A
    // confined active nematic nucleates and annihilates pairs around a braided
    // core, so the first frame's count is not the number of strands that braid,
    // and `--max-step` bounds how far a core strand may be matched in one frame.
    let core = if std::env::args().any(|a| a == "--core") {
        Some(arg("--max-step", "12").parse::<f64>().expect("max-step"))
    } else {
        None
    };
    let max_over: f64 = arg("--max-over", "0.01").parse().expect("max-over");

    let frames = read_frames(&path, charge);
    let counts: Vec<usize> = frames.iter().map(|f| f.len()).collect();
    let window = arg_usize("--window", (frames.len() / 4).max(2));
    let stride = arg_usize("--stride", (window / 4).max(1));

    let mut modal_tally: BTreeMap<i64, usize> = BTreeMap::new();
    let mut windows = 0usize;
    let mut skipped_count = 0usize;
    let mut skipped_strands = 0usize;
    let mut skipped_step = 0usize;

    let mut start = 0usize;
    while start + window <= frames.len() {
        let w = &frames[start..start + window];
        let n0 = w[0].len();
        let nmin = w.iter().map(|f| f.len()).min().unwrap_or(0);
        let nmax = w.iter().map(|f| f.len()).max().unwrap_or(0);

        // Which count has to be large enough depends on the reader. The default
        // reader fixes the strand number at the first frame and cannot extend a
        // worldline into a frame holding fewer defects, so a window whose count
        // drops is not a braid on a fixed number of strands and is reported and
        // skipped rather than silently truncated. The core reader takes the
        // persistent strands, whose number is the window's minimum count, and a
        // drop is then an annihilation among the transients rather than a reason
        // to discard the window.
        let n_avail = if core.is_some() { nmin } else { n0 };
        if (core.is_none() && nmin < n0) || n_avail < min_strands {
            if n_avail < min_strands {
                skipped_strands += 1;
            } else {
                skipped_count += 1;
            }
            println!(
                "{{\"start\":{start},\"window\":{window},\"skipped\":true,\
                 \"n_first\":{n0},\"n_min\":{nmin},\"n_max\":{nmax}}}"
            );
            start += stride;
            continue;
        }

        // A core the step bound rejects is a window with no core to read, and is
        // counted apart from a window with too few strands, since the two say
        // different things about the run.
        let Some(wr) = read_window(w, axes, core) else {
            skipped_step += 1;
            println!(
                "{{\"start\":{start},\"window\":{window},\"skipped\":true,\
                 \"reason\":\"no core\",\"n_first\":{n0},\"n_min\":{nmin},\
                 \"n_max\":{nmax}}}"
            );
            start += stride;
            continue;
        };
        // A window whose assignments repeatedly jump further than the flow can
        // carry a defect in a frame is not tracking a core, and is separated from
        // one that jumps once where a strand died and was replaced. The default
        // admits up to one per cent of assignments, which is far above the one or
        // two an annihilation costs and far below what a mistracked window shows.
        if core.is_some() && wr.steps > 0 && wr.steps_over as f64 > max_over * wr.steps as f64 {
            skipped_step += 1;
            println!(
                "{{\"start\":{start},\"window\":{window},\"skipped\":true,\
                 \"reason\":\"step\",\"steps_over\":{},\"steps\":{},\
                 \"worst_step\":{:.3},\"n_first\":{n0},\"n_min\":{nmin},\
                 \"n_max\":{nmax}}}",
                wr.steps_over, wr.steps, wr.worst_step
            );
            start += stride;
            continue;
        }
        let WindowRead {
            reads,
            strands: core_strands,
            worst_step,
            steps_over,
            steps: n_steps,
        } = wr;
        let mut tally: BTreeMap<i64, usize> = BTreeMap::new();
        for r in &reads {
            *tally.entry(key(r.per_period)).or_insert(0) += 1;
        }
        let (&mk, &mc) = tally.iter().max_by_key(|&(_, &c)| c).unwrap();
        let modal = mk as f64 / 1e6;
        *modal_tally.entry(mk).or_insert(0) += mc;
        windows += 1;

        let at_mode = reads.iter().find(|r| key(r.per_period) == mk).unwrap();
        let per_axis: Vec<String> = reads
            .iter()
            .map(|r| {
                format!(
                    "{{\"deg\":{:.1},\"h\":{:.6},\"per_period\":{:.6},\
                     \"period\":{},\"gens\":{},\"identity\":{},\
                     \"longest_cycle\":{}}}",
                    r.theta_deg, r.entropy, r.per_period, r.period, r.gens,
                    r.identity_permutation, r.longest_cycle
                )
            })
            .collect();
        let word: Vec<String> = at_mode.word.iter().map(|c| c.to_string()).collect();
        // The period in frames, which is what turns an entropy per period into an
        // entropy per unit time and so makes it comparable with a Lyapunov
        // exponent. The word holds `gens` generators over `window` frames and
        // repeats every `period` of them, so the period lasts
        // window * period / gens frames. Read off the modal axis, since that is
        // the axis whose reading is being quoted.
        let period_frames = if at_mode.gens > 0 {
            window as f64 * at_mode.period as f64 / at_mode.gens as f64
        } else {
            f64::NAN
        };
        // The share of axes on which the strands genuinely exchange. A window
        // whose defects move without exchanging reads this at zero, which is a
        // different statement from a badly chosen window and is worth separating.
        let exchanging = reads.iter().filter(|r| !r.identity_permutation).count();
        let max_cycle = reads.iter().map(|r| r.longest_cycle).max().unwrap_or(0);
        println!(
            "{{\"start\":{start},\"window\":{window},\"skipped\":false,\
             \"exchanging_axes\":{exchanging},\"max_cycle\":{max_cycle},\
             \"strands\":{},\"n_max\":{nmax},\"worst_step\":{worst_step:.3},\
             \"steps_over\":{steps_over},\"steps\":{n_steps},\
             \"modal_per_period\":{modal:.6},\
             \"axes_at_mode\":{mc},\"axes\":{axes},\"period\":{},\
             \"gens\":{},\"period_frames\":{period_frames:.3},\
             \"h_per_frame\":{:.6},\"word\":[{}],\"per_axis\":[{}]}}",
            if core.is_some() { core_strands } else { n0 },
            at_mode.period,
            at_mode.gens,
            modal / period_frames,
            word.join(","),
            per_axis.join(",")
        );
        start += stride;
    }

    let total: usize = modal_tally.values().sum();
    let best = modal_tally.iter().max_by_key(|&(_, &c)| c);
    let summary: Vec<String> = modal_tally
        .iter()
        .map(|(k, c)| format!("{{\"h\":{:.6},\"axis_windows\":{c}}}", *k as f64 / 1e6))
        .collect();
    println!(
        "{{\"summary\":true,\"frames\":{},\"charge\":{charge},\"window\":{window},\
         \"stride\":{stride},\"axes\":{axes},\"windows_read\":{windows},\
         \"windows_skipped_count_drop\":{skipped_count},\
         \"windows_skipped_too_few_strands\":{skipped_strands},\
         \"windows_skipped_step_bound\":{skipped_step},\
         \"axis_windows_total\":{total},\"dominant_h\":{},\
         \"dominant_axis_windows\":{},\"strand_count_min\":{},\
         \"strand_count_max\":{},\"distribution\":[{}]}}",
        frames.len(),
        best.map(|(k, _)| *k as f64 / 1e6).unwrap_or(f64::NAN),
        best.map(|(_, c)| *c).unwrap_or(0),
        counts.iter().min().copied().unwrap_or(0),
        counts.iter().max().copied().unwrap_or(0),
        summary.join(",")
    );
}
