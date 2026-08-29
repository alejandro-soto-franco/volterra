//! Braids of point defects on a flat torus.
//!
//! Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing active
//! nematics", Phys. Rev. E 109, 014606 (2024), report that a square with
//! periodic boundaries settles into a four-defect periodic orbit whose two
//! `+1/2` defects trace bounded circular paths and "repeatedly encounter and
//! revolve around each other counterclockwise, with four such encounters during
//! each orbit". They identify that motion with the *maximal mixing braid* of
//! their Fig. 2a, whose topological entropy per operation is
//!
//! ```text
//! h_TEPO = log(phi + sqrt phi) = 1.0613...,   phi = (1 + sqrt 5) / 2,
//! ```
//!
//! the maximum over that class of surface braids, conjectured by Smith and Dunn
//! and strictly above the Ceilidh dance's `log(1 + sqrt 2) = 0.8814`. With `T`
//! the period in units of the active time `t_a`, the braid then predicts a
//! dimensionless topological entropy
//!
//! ```text
//! h_tilde_max = log(phi + sqrt phi) / (T_tilde / 4),
//! ```
//!
//! the blue curve of their Fig. 5.
//!
//! # What this module computes
//!
//! The braid is read off the defect worldlines, never assumed:
//!
//! - [`track_on_torus`] follows each defect between frames by minimum image and
//!   accumulates the lift, so a worldline is a path in the plane rather than a
//!   sequence that jumps by `L` whenever a defect crosses the seam. A bounded
//!   orbit is then a closed loop and a defect that winds is not.
//! - [`TorusWorldlines::encounters`] finds the swaps. Two `+1/2` defects on a
//!   torus meet as often through the images of one as through the defect
//!   itself: in Fig. 2a each rod's circle cuts four image circles of the other,
//!   which is where the four encounters per orbit come from. The search is
//!   therefore over the image lattice, and each encounter records the sense from
//!   the turning of the separation vector.
//! - [`TorusWorldlines::is_maximal_mixing`] applies the paper's own criterion:
//!   two positive defects, both orbits bounded, four encounters per period, all
//!   of one sense.
//!
//! The entropy per operation is quoted, not derived. The paper quotes it too.

use std::f64::consts::PI;

/// `(1 + sqrt 5) / 2`.
pub const PHI: f64 = 1.618_033_988_749_895;

/// Topological entropy per operation of the maximal mixing braid on the torus,
/// `log(phi + sqrt phi) = 1.0613`.
pub fn h_tepo_maximal_mixing() -> f64 {
    (PHI + PHI.sqrt()).ln()
}

/// Topological entropy per operation of the Ceilidh dance, `log(1 + sqrt 2)`.
///
/// The optimum for defects restricted to a linear arrangement, which the
/// maximal mixing braid beats.
pub fn h_tepo_ceilidh() -> f64 {
    (1.0 + std::f64::consts::SQRT_2).ln()
}

/// One observation: the time and the defects seen at it, as `(position, charge)`.
pub type DefectFrame = (f64, Vec<([f64; 2], i32)>);

/// A defect worldline set, lifted to the universal cover of the torus.
#[derive(Debug, Clone)]
pub struct TorusWorldlines {
    /// Torus periods.
    pub lx: f64,
    pub ly: f64,
    /// Frame times.
    pub times: Vec<f64>,
    /// `pts[k][s]`: strand `s` at frame `k`, lifted, so `pts[k+1][s] - pts[k][s]`
    /// is the minimum image displacement and never a jump of a period.
    pub pts: Vec<Vec<[f64; 2]>>,
    /// Topological charge of each strand, in half units: `1` for `+1/2`.
    pub charge: Vec<i32>,
}

/// One swap of a pair of defects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Encounter {
    /// Frame at which the separation is least.
    pub frame: usize,
    /// Time of that frame.
    pub t: f64,
    /// The two strands.
    pub strands: (usize, usize),
    /// Which image of the second strand was met, in lattice units.
    pub image: [i32; 2],
    /// Least separation, in lattice units.
    pub distance: f64,
    /// `+1` counterclockwise, `-1` clockwise: the sign of the turning of the
    /// separation vector across the encounter.
    pub sense: i32,
}

/// The minimum image of `b - a` on a torus of periods `lx`, `ly`.
///
/// Returns the vector and the lattice translation applied to `b`.
pub fn min_image(a: [f64; 2], b: [f64; 2], lx: f64, ly: f64) -> ([f64; 2], [i32; 2]) {
    let raw = [b[0] - a[0], b[1] - a[1]];
    let nx = (raw[0] / lx).round();
    let ny = (raw[1] / ly).round();
    ([raw[0] - nx * lx, raw[1] - ny * ly], [-(nx as i32), -(ny as i32)])
}

/// Follow defects frame to frame and lift the trajectories.
///
/// `frames[k]` lists the defects seen at frame `k` as `(position, charge)` with
/// positions in `[0, lx) x [0, ly)`. Every frame must show the same defects, in
/// any order: a frame whose count differs from the first, or whose nearest
/// assignment moves a defect further than `max_disp`, ends the tracking and
/// returns what was followed up to there. `None` means the first frame was
/// empty.
///
/// The assignment is greedy over minimum image distance, which is what a
/// four-defect orbit needs and is the same rule `track` uses in the plane.
pub fn track_on_torus(
    frames: &[DefectFrame],
    lx: f64,
    ly: f64,
    max_disp: f64,
) -> Option<TorusWorldlines> {
    let (t0, first) = frames.first()?;
    if first.is_empty() {
        return None;
    }
    let n = first.len();
    let charge: Vec<i32> = first.iter().map(|d| d.1).collect();
    let mut times = vec![*t0];
    let mut pts = vec![first.iter().map(|d| d.0).collect::<Vec<_>>()];

    for (t, frame) in frames.iter().skip(1) {
        if frame.len() != n {
            break;
        }
        let prev = pts.last().unwrap().clone();
        let mut taken = vec![false; n];
        let mut next = vec![[f64::NAN; 2]; n];
        let mut ok = true;
        // Greedy over all pairs, closest first, so a crossing pair is not
        // assigned by scan order.
        let mut cand: Vec<(f64, usize, usize, [f64; 2])> = Vec::with_capacity(n * n);
        for (s, p) in prev.iter().enumerate() {
            for (j, d) in frame.iter().enumerate() {
                if charge[s] != d.1 {
                    continue;
                }
                // The lift: measure from the strand's own lifted position, so
                // the displacement is the minimum image and the lift accumulates.
                let (v, _) = min_image(
                    [p[0].rem_euclid(lx), p[1].rem_euclid(ly)],
                    d.0,
                    lx,
                    ly,
                );
                cand.push((v[0].hypot(v[1]), s, j, [p[0] + v[0], p[1] + v[1]]));
            }
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut used = vec![false; n];
        let mut count = 0;
        for (d, s, j, lifted) in cand {
            if taken[s] || used[j] || d > max_disp {
                continue;
            }
            taken[s] = true;
            used[j] = true;
            next[s] = lifted;
            count += 1;
        }
        if count != n {
            ok = false;
        }
        if !ok {
            break;
        }
        times.push(*t);
        pts.push(next);
    }
    Some(TorusWorldlines { lx, ly, times, pts, charge })
}

impl TorusWorldlines {
    /// Number of strands.
    pub fn n_strands(&self) -> usize {
        self.charge.len()
    }

    /// Number of frames followed.
    pub fn n_frames(&self) -> usize {
        self.pts.len()
    }

    /// Indices of the `+1/2` strands, which are the ones that braid.
    ///
    /// The paper measures the entropy from the positive defects alone and finds
    /// the negative ones "introduce no additional stretching".
    pub fn positive(&self) -> Vec<usize> {
        (0..self.n_strands()).filter(|&s| self.charge[s] > 0).collect()
    }

    /// Net lattice displacement of each strand over the run.
    ///
    /// A bounded orbit returns `[0, 0]`. Anything else is a defect that winds
    /// around the torus, which is not the motion of Fig. 2.
    pub fn winding(&self) -> Vec<[f64; 2]> {
        let (a, b) = (self.pts.first(), self.pts.last());
        match (a, b) {
            (Some(a), Some(b)) => (0..self.n_strands())
                .map(|s| [(b[s][0] - a[s][0]) / self.lx, (b[s][1] - a[s][1]) / self.ly])
                .collect(),
            _ => vec![],
        }
    }

    /// Radius of gyration of each lifted worldline, in lattice units.
    ///
    /// The paper's "bounded, circular shape": a bounded orbit has a radius that
    /// stops growing, a wandering one does not.
    pub fn gyration(&self) -> Vec<f64> {
        (0..self.n_strands())
            .map(|s| {
                let n = self.n_frames() as f64;
                let (mx, my) = self.pts.iter().fold((0.0, 0.0), |(x, y), f| {
                    (x + f[s][0] / n, y + f[s][1] / n)
                });
                (self
                    .pts
                    .iter()
                    .map(|f| (f[s][0] - mx).powi(2) + (f[s][1] - my).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt()
            })
            .collect()
    }

    /// The swaps of each pair of `+1/2` strands, over the image lattice.
    ///
    /// An encounter is a local minimum of the separation, over every periodic
    /// image of the second strand, that falls below `close`. Its sense is the
    /// sign of the turning of the separation vector between the surrounding
    /// frames, which is `+1` for a counterclockwise pass.
    ///
    /// `close` is in lattice units; `0.5 * lx` admits every pass that matters on
    /// a square torus, since two defects further apart than that are closer to
    /// some other image.
    pub fn encounters(&self, close: f64) -> Vec<Encounter> {
        let pos = self.positive();
        let mut out = Vec::new();
        for (ia, &a) in pos.iter().enumerate() {
            for &b in pos.iter().skip(ia + 1) {
                // Separation and its image label at every frame.
                let sep: Vec<([f64; 2], [i32; 2])> = (0..self.n_frames())
                    .map(|k| {
                        let (v, im) = min_image(
                            [
                                self.pts[k][a][0].rem_euclid(self.lx),
                                self.pts[k][a][1].rem_euclid(self.ly),
                            ],
                            [
                                self.pts[k][b][0].rem_euclid(self.lx),
                                self.pts[k][b][1].rem_euclid(self.ly),
                            ],
                            self.lx,
                            self.ly,
                        );
                        (v, im)
                    })
                    .collect();
                let d: Vec<f64> = sep.iter().map(|(v, _)| v[0].hypot(v[1])).collect();
                for k in 1..d.len().saturating_sub(1) {
                    if d[k] <= d[k - 1] && d[k] < d[k + 1] && d[k] < close {
                        // Turning of the separation vector across the minimum.
                        let (u, _) = (sep[k - 1].0, 0);
                        let (w, _) = (sep[k + 1].0, 0);
                        let cross = u[0] * w[1] - u[1] * w[0];
                        out.push(Encounter {
                            frame: k,
                            t: self.times[k],
                            strands: (a, b),
                            image: sep[k].1,
                            distance: d[k],
                            sense: if cross >= 0.0 { 1 } else { -1 },
                        });
                    }
                }
            }
        }
        out.sort_by_key(|e| e.frame);
        out
    }

    /// The paper's criterion for the maximal mixing braid, applied to a window.
    ///
    /// Two `+1/2` defects, both orbits bounded, `4` encounters in one period and
    /// every one of the same sense. `period` is in the same units as `times`.
    pub fn is_maximal_mixing(&self, period: f64, close: f64) -> MaximalMixing {
        let enc = self.encounters(close);
        // The rate comes from the spacing of the encounters, never from the
        // count over the window. A window that opens or closes on an encounter
        // loses it, and over four periods that reads 3.75 an orbit rather than
        // four; the interval between the first and the last is not exposed to
        // that.
        let per_period = if period > 0.0 && enc.len() >= 2 {
            let span = enc.last().unwrap().t - enc.first().unwrap().t;
            if span > 0.0 { (enc.len() - 1) as f64 * period / span } else { f64::NAN }
        } else {
            f64::NAN
        };
        let net: i32 = enc.iter().map(|e| e.sense).sum();
        let one_sense = !enc.is_empty() && net.unsigned_abs() as usize == enc.len();
        let gyr = self.gyration();
        let bounded = self
            .positive()
            .iter()
            .all(|&s| gyr[s] < 0.5 * self.lx.max(self.ly));
        let winds = self.winding();
        let no_winding = self
            .positive()
            .iter()
            .all(|&s| winds[s][0].abs() < 0.5 && winds[s][1].abs() < 0.5);
        MaximalMixing {
            n_positive: self.positive().len(),
            encounters: enc.len(),
            per_period,
            one_sense,
            sense: net.signum(),
            bounded: bounded && no_winding,
            verdict: self.positive().len() == 2
                && bounded
                && no_winding
                && one_sense
                && (per_period - 4.0).abs() < 0.5,
        }
    }

    /// The braid's prediction for the dimensionless topological entropy.
    ///
    /// `log(phi + sqrt phi) / (T_tilde / 4)` with `T_tilde = period / t_a`, the
    /// blue curve of Fig. 5. Stated for any measured period, so a run whose
    /// braid is not the maximal mixing one still reports what that braid would
    /// have predicted, next to the verdict that says it is a different braid.
    pub fn braid_prediction(period: f64, t_a: f64) -> f64 {
        let t_tilde = period / t_a;
        h_tepo_maximal_mixing() / (t_tilde / 4.0)
    }
}

/// The outcome of the maximal-mixing test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaximalMixing {
    /// Number of `+1/2` strands followed.
    pub n_positive: usize,
    /// Encounters over the whole window.
    pub encounters: usize,
    /// Encounters per period.
    pub per_period: f64,
    /// Whether every encounter turned the same way.
    pub one_sense: bool,
    /// `+1` counterclockwise, `-1` clockwise, `0` mixed.
    pub sense: i32,
    /// Whether both positive orbits stayed bounded and unwound.
    pub bounded: bool,
    /// Every condition met.
    pub verdict: bool,
}

/// The ideal motion of Fig. 2a, as trajectories.
///
/// The construction the figure draws. The two `-1/2` defects sit at `(0, 0)` and
/// at the cell centre `(L/2, L/2)`, and each `+1/2` rod runs counterclockwise
/// around one of them on the ellipse of semi-axes `(lx/2, ly/2)`. Those two
/// ellipses meet exactly at `(0, L/2)` and `(L/2, 0)`, which are the rods' own
/// sites, so each rod passes through both sites once a revolution; counting the
/// periodic images of the other ellipse as well, a rod meets the other rod's
/// track at four points a revolution. That is the "four such encounters during
/// each orbit" of the paper, and each of them exchanges the two rods on the
/// torus, since every image of a site is the same point there.
///
/// `phase` offsets the second rod, `periods` is how many revolutions to
/// generate. This exists to test the reader, not to model anything: a run's
/// braid is read off its own worldlines.
pub fn ideal_figure_2a(
    lx: f64,
    ly: f64,
    phase: f64,
    n_frames: usize,
    periods: f64,
) -> Vec<DefectFrame> {
    (0..n_frames)
        .map(|k| {
            let t = periods * k as f64 / (n_frames - 1).max(1) as f64;
            // Rod A starts at (0, ly/2), rod B at (lx/2, 0).
            let a = 0.5 * PI + 2.0 * PI * t;
            let b = -0.5 * PI + 2.0 * PI * t + phase;
            let wrap = |p: [f64; 2]| [p[0].rem_euclid(lx), p[1].rem_euclid(ly)];
            (
                t,
                vec![
                    (wrap([0.5 * lx * a.cos(), 0.5 * ly * a.sin()]), 1),
                    (wrap([0.5 * lx * (1.0 + b.cos()), 0.5 * ly * (1.0 + b.sin())]), 1),
                    (wrap([0.0, 0.0]), -1),
                    (wrap([0.5 * lx, 0.5 * ly]), -1),
                ],
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_quoted_constants() {
        assert!((h_tepo_maximal_mixing() - 1.0613).abs() < 5e-5);
        assert!((h_tepo_ceilidh() - 0.8814).abs() < 5e-5);
        // The maximal mixing braid beats the Ceilidh dance, which is the point
        // the paper makes about it.
        assert!(h_tepo_maximal_mixing() > h_tepo_ceilidh());
    }

    /// A defect crossing the seam must lift, never jump by a period.
    #[test]
    fn a_worldline_lifts_across_the_seam() {
        let (lx, ly) = (100.0, 100.0);
        let frames: Vec<_> = (0..40)
            .map(|k| {
                let x = (95.0 + 0.5 * k as f64).rem_euclid(lx);
                (k as f64, vec![([x, 50.0], 1), ([50.0, 50.0], -1)])
            })
            .collect();
        let w = track_on_torus(&frames, lx, ly, 5.0).unwrap();
        assert_eq!(w.n_frames(), 40);
        // Total lifted travel is 39 * 0.5, with no jump of 100 anywhere.
        let d = w.pts[39][0][0] - w.pts[0][0][0];
        assert!((d - 19.5).abs() < 1e-9, "lifted displacement {d}");
        for k in 1..40 {
            assert!((w.pts[k][0][0] - w.pts[k - 1][0][0]).abs() < 1.0);
        }
        // And it has wound once around the torus, which `winding` reports.
        assert!((w.winding()[0][0] - 0.195).abs() < 1e-9);
    }

    /// The ideal motion of Fig. 2a gives four same-sense encounters an orbit.
    #[test]
    fn the_ideal_braid_has_four_encounters_of_one_sense_per_orbit() {
        let (lx, ly) = (100.0, 100.0);
        let frames = ideal_figure_2a(lx, ly, 0.0, 4001, 4.0);
        let w = track_on_torus(&frames, lx, ly, 5.0).unwrap();
        assert_eq!(w.n_frames(), 4001);
        assert_eq!(w.positive().len(), 2);

        let m = w.is_maximal_mixing(1.0, 0.5 * lx);
        assert!(
            (m.per_period - 4.0).abs() < 0.25,
            "encounters per orbit {} (total {})",
            m.per_period,
            m.encounters
        );
        assert!(m.one_sense, "senses were mixed: {m:?}");
        assert!(m.bounded, "orbits were not bounded: {m:?}");
        assert!(m.verdict, "{m:?}");
    }

    /// A pair that stays apart is not braiding, whatever else it does.
    #[test]
    fn separated_rods_write_no_encounters() {
        let (lx, ly) = (100.0, 100.0);
        // Both rods parked at their sites, `L / sqrt 2` apart for all time.
        let frames: Vec<_> = (0..500)
            .map(|k| {
                (
                    k as f64 * 0.01,
                    vec![
                        ([0.0, 50.0], 1),
                        ([50.0, 0.0], 1),
                        ([0.0, 0.0], -1),
                        ([50.0, 50.0], -1),
                    ],
                )
            })
            .collect();
        let w = track_on_torus(&frames, lx, ly, 5.0).unwrap();
        let m = w.is_maximal_mixing(1.0, 0.25 * lx);
        assert_eq!(m.encounters, 0, "{m:?}");
        assert!(!m.verdict);
    }

    /// The braid prediction is the paper's own formula, at its own numbers.
    ///
    /// Mitchell et al. report `h_tilde = 1.66e-3` at `ell_a = 3`, closely
    /// tracked by the braid curve. A period of `T_tilde = 2560` active times
    /// puts the prediction there.
    #[test]
    fn the_braid_prediction_is_the_papers_formula() {
        let t_a = 3.515_625e-3;
        let t_tilde = 4.0 * h_tepo_maximal_mixing() / 1.66e-3;
        let period = t_tilde * t_a;
        let p = TorusWorldlines::braid_prediction(period, t_a);
        assert!((p - 1.66e-3).abs() < 1e-9, "prediction {p}");
        // Twice the period halves the prediction: the entropy is per unit time,
        // and a slower orbit writes the same braid over longer.
        let half = TorusWorldlines::braid_prediction(2.0 * period, t_a);
        assert!((half - 0.83e-3).abs() < 1e-9, "prediction {half}");
    }

    /// Charges are never confused with one another by the tracker.
    #[test]
    fn strands_keep_their_charge() {
        let (lx, ly) = (100.0, 100.0);
        let frames = ideal_figure_2a(lx, ly, 0.0, 501, 1.0);
        let w = track_on_torus(&frames, lx, ly, 5.0).unwrap();
        assert_eq!(w.charge, vec![1, 1, -1, -1]);
        for f in &w.pts {
            // The two negatives never move in the ideal motion.
            assert!((f[2][0] - 0.0).abs() < 1e-9 && (f[2][1] - 0.0).abs() < 1e-9);
            assert!((f[3][0] - 50.0).abs() < 1e-9 && (f[3][1] - 50.0).abs() < 1e-9);
        }
    }
}
