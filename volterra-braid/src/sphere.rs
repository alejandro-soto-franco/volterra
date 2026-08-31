//! Braiding of defects on the sphere.
//!
//! The flat-torus reader in [`crate::torus`] answers a question about a doubly
//! periodic box: strands live in the plane, separations are minimum images, and
//! a pass is a close approach modulo the lattice. On a sphere none of that
//! applies. Strands are unit vectors, separation is a geodesic angle, and there
//! is no lattice, so this module defines its own geometry and hands the result
//! to the braid-word and entropy machinery the crate already has.
//!
//! # What a braid on a sphere is, and what is read here
//!
//! `n` points moving on `S^2` trace a loop in the configuration space of the
//! sphere, and that loop is an element of the sphere braid group `B_n(S^2)`.
//! The group is a
//! quotient of the disc braid group `B_n(D^2)`: it adds the relation that
//! sweeping one strand all the way round the others is trivial, which is
//! possible on a sphere and not in a disc.
//!
//! A word is read here by deleting one point of the sphere and flattening what
//! remains. [`SphereWorldlines::project`] does that by stereographic
//! projection from a pole, after which [`crate::extract_braidword`] and
//! [`crate::topological_entropy`] apply unchanged. The consequence has to be
//! stated rather than buried: the projected word is a word in `B_n(D^2)`, on
//! the sphere punctured at the pole, and the puncture is a strand the flow does
//! not have. Its entropy therefore **bounds the closed-sphere value from
//! above**, since removing a puncture cannot make a mapping class more complex.
//! The two agree when the pole sits somewhere the flow itself keeps fixed.
//!
//! Choosing the pole is not a free parameter either. Stereographic projection
//! diverges at the pole, so a pole that any strand approaches produces
//! arbitrarily large coordinates and an ordering that flips on rounding.
//! [`SphereWorldlines::far_pole`] picks the direction furthest from every
//! strand over the whole run.
//!
//! # Where the entropy is exact
//!
//! [`crate::topological_entropy`] is the spectral radius of the unreduced Burau
//! representation at `t = -1`, which equals the dilatation while Burau is
//! faithful, that is through four strands. Poincaré-Hopf puts four `+1/2`
//! defects on a sphere when every defect is a half charge, so the standard
//! configuration sits inside the exact regime while a five-defect one sits
//! outside it.
//! [`crate::is_exact_regime`] is the branch to take.

use crate::braidword::{BraidWord, extract_braidword};
use crate::track::Worldline;

/// One observation: the defects seen at one time, as unit vectors with charge.
pub type SphereFrame = (f64, Vec<([f64; 3], i32)>);

/// Tracked worldlines on the unit sphere.
#[derive(Debug, Clone)]
pub struct SphereWorldlines {
    /// Observation times, one per frame.
    pub times: Vec<f64>,
    /// `pts[frame][strand]`, each a unit vector.
    pub pts: Vec<Vec<[f64; 3]>>,
    /// Charge sign of each strand, fixed for its life.
    pub charge: Vec<i32>,
}

/// A pass of one strand by another.
/// A pooled entropy reading over sliding windows and projection axes.
pub struct EntropyScan {
    /// Every reading, one per window and axis.
    pub values: Vec<f64>,
    /// Mean over the pool.
    pub mean: f64,
    /// Standard deviation over the pool.
    pub sd: f64,
    /// Median over the pool.
    pub median: f64,
    /// Fraction of readings that are not zero, so a mixing braid.
    pub mixing: f64,
    /// Number of projection axes pooled.
    pub n_axes: usize,
    /// Number of sliding windows per axis.
    pub n_windows: usize,
    /// The window length in physical time.
    pub window: f64,
}

impl EntropyScan {
    fn empty() -> Self {
        Self {
            values: Vec::new(),
            mean: f64::NAN,
            sd: f64::NAN,
            median: f64::NAN,
            mixing: 0.0,
            n_axes: 0,
            n_windows: 0,
            window: f64::NAN,
        }
    }

    fn from_values(values: Vec<f64>, n_axes: usize, window: f64) -> Self {
        if values.is_empty() || n_axes == 0 {
            return Self::empty();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let mixing = values.iter().filter(|v| **v > 1e-12).count() as f64 / n;
        let n_windows = values.len() / n_axes;
        Self { values, mean, sd, median, mixing, n_axes, n_windows, window }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpherePass {
    /// Frame index within the tracked window.
    pub frame: usize,
    /// Time of the pass.
    pub t: f64,
    /// The two strands.
    pub strands: (usize, usize),
    /// Geodesic separation at closest approach, in radians.
    pub distance: f64,
    /// Sense of the turn of the separation across the pass, `+1` or `-1`.
    pub sense: i32,
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n <= 0.0 { [0.0, 0.0, 1.0] } else { [v[0] / n, v[1] / n, v[2] / n] }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Geodesic distance between two unit vectors, in radians.
///
/// From the cross and dot products rather than from `acos` of the dot alone:
/// `acos` loses all its precision for nearby points, which is exactly where a
/// pass is decided.
pub fn geodesic(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = cross(a, b);
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(dot(a, b))
}

/// Track defects across frames on the sphere.
///
/// The cast is fixed by the first frame. Each later frame is matched to it
/// closest pair first, so a crossing pair is not assigned by scan order, and a
/// frame with MORE defects than the cast is kept with the extras unassigned:
/// a short-lived pair beside a persistent orbit should not end the track. A
/// frame that leaves any strand unmatched does end it.
///
/// `max_disp` is in radians, the furthest a defect may move between frames.
pub fn track_on_sphere(
    frames: &[SphereFrame],
    max_disp: f64,
) -> Option<SphereWorldlines> {
    let (t0, first) = frames.first()?;
    if first.is_empty() {
        return None;
    }
    let n = first.len();
    let charge: Vec<i32> = first.iter().map(|d| d.1).collect();
    let mut times = vec![*t0];
    let mut pts = vec![first.iter().map(|d| norm(d.0)).collect::<Vec<_>>()];

    for (t, frame) in frames.iter().skip(1) {
        let prev = pts.last().unwrap().clone();
        let mut cand: Vec<(f64, usize, usize, [f64; 3])> = Vec::with_capacity(n * frame.len());
        for (s, p) in prev.iter().enumerate() {
            for (j, d) in frame.iter().enumerate() {
                if charge[s] != d.1 {
                    continue;
                }
                let q = norm(d.0);
                cand.push((geodesic(*p, q), s, j, q));
            }
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut taken = vec![false; n];
        let mut used = vec![false; frame.len()];
        let mut next = vec![[f64::NAN; 3]; n];
        let mut count = 0usize;
        for (d, s, j, q) in cand {
            if taken[s] || used[j] || d > max_disp {
                continue;
            }
            taken[s] = true;
            used[j] = true;
            next[s] = q;
            count += 1;
        }
        if count != n {
            break;
        }
        times.push(*t);
        pts.push(next);
    }
    Some(SphereWorldlines { times, pts, charge })
}

impl SphereWorldlines {
    /// Number of tracked frames.
    pub fn n_frames(&self) -> usize {
        self.times.len()
    }

    /// Number of strands.
    pub fn n_strands(&self) -> usize {
        self.charge.len()
    }

    /// Indices of the positive strands.
    pub fn positive(&self) -> Vec<usize> {
        (0..self.n_strands()).filter(|&s| self.charge[s] > 0).collect()
    }

    /// The direction furthest from every strand over the whole run.
    ///
    /// Stereographic projection diverges at the pole, so a pole any strand
    /// approaches gives coordinates that grow without bound and an ordering
    /// that flips on rounding. The search is over a Fibonacci lattice, which
    /// spreads directions evenly without a pole of its own, unlike a
    /// latitude-longitude grid whose samples pile up at its own axis.
    pub fn far_pole(&self) -> [f64; 3] {
        let m = 512usize;
        let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut best = ([0.0, 0.0, 1.0], -1.0);
        for i in 0..m {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / m as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * i as f64;
            let p = [r * th.cos(), r * th.sin(), z];
            let mut worst = f64::INFINITY;
            for f in &self.pts {
                for q in f {
                    worst = worst.min(geodesic(p, *q));
                }
            }
            if worst > best.1 {
                best = (p, worst);
            }
        }
        // The lattice only locates the best cap to within its own spacing,
        // about `sqrt(4 pi / m)` radians, which on a run whose strands leave a
        // wide gap throws away most of the clearance. Walk downhill from there
        // with a step that halves, so the answer is the cap's centre rather
        // than the nearest sample to it.
        let clearance = |p: [f64; 3]| {
            self.pts
                .iter()
                .flat_map(|f| f.iter().map(|q| geodesic(p, *q)))
                .fold(f64::INFINITY, f64::min)
        };
        let (mut cur, mut val) = best;
        let mut step = 2.0 * (std::f64::consts::PI / m as f64).sqrt();
        for _ in 0..40 {
            let mut moved = false;
            for d in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
                let trial = norm([
                    cur[0] + step * d[0],
                    cur[1] + step * d[1],
                    cur[2] + step * d[2],
                ]);
                let v = clearance(trial);
                if v > val {
                    cur = trial;
                    val = v;
                    moved = true;
                }
            }
            if !moved {
                step *= 0.5;
                if step < 1e-9 {
                    break;
                }
            }
        }
        cur
    }

    /// Stereographic projection from `pole`, as plane worldlines.
    ///
    /// The frame is built from the pole itself, so the projection is a
    /// conformal chart of the sphere minus that point and no separate rotation
    /// convention enters. Only the strand ORDER matters downstream, and any
    /// orientation-preserving chart gives the same order.
    pub fn project(&self, pole: [f64; 3]) -> Vec<Worldline> {
        let p = norm(pole);
        // Any two unit vectors completing a right-handed frame with the pole.
        let seed = if p[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let e1 = norm(cross(seed, p));
        let e2 = cross(p, e1);
        (0..self.n_strands())
            .map(|s| Worldline {
                positions: (0..self.n_frames())
                    .map(|k| {
                        let q = self.pts[k][s];
                        // Project from `pole` onto the plane through the centre.
                        let d = 1.0 - dot(q, p);
                        let scale = if d.abs() < 1e-12 { 1e12 } else { 1.0 / d };
                        [dot(q, e1) * scale, dot(q, e2) * scale]
                    })
                    .collect(),
                charge: self.charge[s] as i8,
            })
            .collect()
    }

    /// The braid word of the positive strands, read in the chart at `pole`.
    ///
    /// A word in the DISC braid group on the sphere punctured at the pole. See
    /// the module note: its entropy bounds the closed-sphere value from above.
    pub fn braid_word(&self, pole: [f64; 3]) -> BraidWord {
        let pos = self.positive();
        let all = self.project(pole);
        let sel: Vec<Worldline> = pos.into_iter().map(|s| all[s].clone()).collect();
        extract_braidword(&sel)
    }

    /// Recurrence period of a strand: the lag at which it returns to where it
    /// was, and how deeply.
    ///
    /// Measured from the worldline rather than from a flow diagnostic, because
    /// the two differ by however many passes a revolution makes: a diagnostic
    /// that peaks at every pass cycles a multiple of the orbit frequency, and
    /// reading it as the period divides the period by that multiple.
    ///
    /// The deepest return is taken rather than the first. A closed orbit on a
    /// sphere generally comes part of the way back at some fraction of its
    /// period, which is a shallow dip and not a recurrence.
    pub fn recurrence_period(&self, strand: usize) -> (f64, f64) {
        let n = self.n_frames();
        if n < 16 || strand >= self.n_strands() {
            return (f64::NAN, 0.0);
        }
        let dt = self.times[1] - self.times[0];
        let max_lag = n / 2;
        let mut sep = vec![f64::NAN; max_lag];
        for (lag, out) in sep.iter_mut().enumerate().skip(1) {
            let mut acc = 0.0;
            for k in 0..n - lag {
                acc += geodesic(self.pts[k][strand], self.pts[k + lag][strand]);
            }
            *out = acc / (n - lag) as f64;
        }
        let mean: f64 = sep[1..].iter().sum::<f64>() / (max_lag - 1) as f64;
        let mut i = 1;
        while i + 1 < max_lag && sep[i + 1] > sep[i] {
            i += 1;
        }
        if i + 1 >= max_lag {
            return (f64::NAN, 0.0);
        }
        let deepest = sep[i..].iter().cloned().fold(f64::INFINITY, f64::min);
        let cut = deepest + 0.2 * (mean - deepest).max(0.0);
        for j in i..max_lag - 1 {
            if sep[j] <= cut && sep[j] <= sep[j - 1] && sep[j] <= sep[j + 1] {
                return (j as f64 * dt, 1.0 - sep[j] / mean.max(1e-12));
            }
        }
        (f64::NAN, 0.0)
    }

    /// Recurrence period of the positive strands, averaged, with the weakest
    /// strand's quality.
    pub fn orbit_period(&self) -> (f64, f64) {
        let pos = self.positive();
        let mut ts = Vec::new();
        let mut q = 1.0_f64;
        for &s in &pos {
            let (t, quality) = self.recurrence_period(s);
            if t.is_finite() {
                ts.push(t);
            }
            q = q.min(quality);
        }
        if ts.is_empty() {
            return (f64::NAN, 0.0);
        }
        (ts.iter().sum::<f64>() / ts.len() as f64, q)
    }

    /// The configuration's shape at one frame: the sorted pairwise geodesic
    /// separations of the positive strands.
    ///
    /// A geodesic separation is unchanged by any rotation of the whole sphere,
    /// and sorting removes the labelling, so the signature sees the shape the
    /// defects form and nothing about where that shape sits or which defect is
    /// which.
    ///
    /// This is what makes a precessing orbit legible. A configuration that
    /// turns as it repeats never brings a defect back to its own earlier
    /// position, so a measure built on a strand returning to itself reports no
    /// period however cleanly the motion repeats. The shape does return.
    ///
    /// A reflection preserves every separation, so a configuration and its
    /// mirror image share a signature. Motion that passes through its own
    /// mirror halfway round therefore reads at half its period, which is the
    /// shape's period rather than the motion's.
    pub fn shape_signature(&self, frame: usize) -> Vec<f64> {
        let pos = self.positive();
        let mut out = Vec::with_capacity(pos.len() * (pos.len().saturating_sub(1)) / 2);
        for a in 0..pos.len() {
            for b in a + 1..pos.len() {
                out.push(geodesic(self.pts[frame][pos[a]], self.pts[frame][pos[b]]));
            }
        }
        out.sort_by(|x, y| x.partial_cmp(y).unwrap());
        out
    }

    /// The period of the configuration's shape, and how strongly it repeats.
    ///
    /// Measured as the first peak of the autocorrelation of the shape
    /// signature past its first sign change, summed over the signature's
    /// components with each component's mean removed. The returned quality is
    /// that peak's height, normalised so a perfectly repeating signal gives 1
    /// and an uncorrelated one gives 0.
    ///
    /// The first sign change is skipped because the autocorrelation of any
    /// smooth signal starts at 1 and descends, so its own maximum at zero lag
    /// is not a period.
    pub fn shape_period(&self) -> (f64, f64) {
        let n = self.n_frames();
        if n < 16 {
            return (f64::NAN, 0.0);
        }
        let dt = self.times[1] - self.times[0];
        let sig: Vec<Vec<f64>> = (0..n).map(|k| self.shape_signature(k)).collect();
        let m = sig[0].len();
        if m == 0 {
            return (f64::NAN, 0.0);
        }
        // Mean-removed components, so the autocorrelation measures repetition
        // rather than the average separation.
        let mut comp = vec![vec![0.0_f64; n]; m];
        for (j, c) in comp.iter_mut().enumerate() {
            let mean: f64 = (0..n).map(|k| sig[k][j]).sum::<f64>() / n as f64;
            for k in 0..n {
                c[k] = sig[k][j] - mean;
            }
        }
        let max_lag = n / 2;
        let mut ac = vec![0.0_f64; max_lag];
        for c in &comp {
            for (lag, a) in ac.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..n - lag {
                    acc += c[k] * c[k + lag];
                }
                *a += acc;
            }
        }
        if ac[0].abs() < 1e-300 {
            return (f64::NAN, 0.0);
        }
        let norm = ac[0];
        for a in ac.iter_mut() {
            *a /= norm;
        }
        let Some(zero) = (1..max_lag).find(|&i| ac[i] < 0.0) else {
            return (f64::NAN, 0.0);
        };
        let mut best = (0usize, f64::NEG_INFINITY);
        for i in zero + 1..max_lag - 1 {
            if ac[i] >= ac[i - 1] && ac[i] >= ac[i + 1] && ac[i] > best.1 {
                best = (i, ac[i]);
                break;
            }
        }
        if best.0 == 0 {
            return (f64::NAN, 0.0);
        }
        (best.0 as f64 * dt, best.1)
    }

    /// The topological entropy per unit time, and how well that rate is
    /// defined.
    ///
    /// The entropy of a braid WORD is not a property of the flow: a longer
    /// window writes a longer word and a larger number, so quoting one is
    /// quoting the window. The rate `h(T) / T` is the invariant, and it means
    /// something only when it settles.
    ///
    /// Windows from a quarter of the record to all of it are read from the
    /// same start. The returned triple is the median rate, the spread across
    /// those windows relative to it, and the fraction of windows that wrote a
    /// mixing braid at all.
    ///
    /// The fraction is what separates the two ways a rate can fail to exist.
    /// A fraction near one with a large spread is a braid whose crossings
    /// bunch. A small fraction is a window artefact: one long word happened to
    /// be pseudo-Anosov while its own sub-words were not, and quoting its
    /// entropy would be quoting the window.
    pub fn entropy_rate(&self, pole: [f64; 3]) -> (f64, f64, f64) {
        let n = self.n_frames();
        if n < 32 {
            return (f64::NAN, f64::INFINITY, 0.0);
        }
        let pos = self.positive();
        let all = self.project(pole);
        let sel: Vec<Worldline> = pos.iter().map(|&s| all[s].clone()).collect();
        let mut rates = Vec::new();
        for step in 4..=16 {
            let end = n * step / 16;
            if end < 16 {
                continue;
            }
            let cut: Vec<Worldline> = sel
                .iter()
                .map(|w| Worldline {
                    positions: w.positions[..end].to_vec(),
                    charge: w.charge,
                })
                .collect();
            let h = crate::topological_entropy(&extract_braidword(&cut));
            let span = self.times[end - 1] - self.times[0];
            if span > 0.0 {
                rates.push(h / span);
            }
        }
        if rates.is_empty() {
            return (f64::NAN, f64::INFINITY, 0.0);
        }
        let positive = rates.iter().filter(|r| **r > 0.0).count() as f64 / rates.len() as f64;
        let mut sorted = rates.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        if median <= 0.0 {
            return (0.0, 0.0, positive);
        }
        let spread = (sorted[sorted.len() - 1] - sorted[0]) / median;
        (median, spread, positive)
    }

    /// A windowed, multi-axis entropy scan.
    ///
    /// A single reading is not a number the run hands you. The word depends on
    /// the direction the strands are ordered along, so one axis can differ
    /// from another by a factor of two on the same data, and it depends on the
    /// window, so a cumulative reading grows with however long it was watched.
    ///
    /// The scan fixes both. A window of `window` in PHYSICAL time slides along
    /// the record in quarter-window steps, every window is read at `n_axes`
    /// projection poles spread over the sphere, and the pooled readings are
    /// returned. The mean over that pool is the entropy of a window, with the
    /// spread saying whether the axes and windows agree.
    ///
    /// A window shorter than the motion's period reads a sub-braid and returns
    /// a value that climbs with the window, so pass the shape period.
    pub fn entropy_scan(&self, window: f64, n_axes: usize) -> EntropyScan {
        let n = self.n_frames();
        if n < 8 || !window.is_finite() || window <= 0.0 {
            return EntropyScan::empty();
        }
        let dt = self.times[1] - self.times[0];
        let w = ((window / dt).round() as usize).max(4);
        if w >= n {
            return EntropyScan::empty();
        }
        let stride = (w / 4).max(1);

        // Poles spread evenly, keeping only those with room from every strand:
        // a chart deleted on top of a strand sends it to infinity and the
        // ordering it induces is meaningless.
        let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let m = (n_axes * 8).max(64);
        let mut poles: Vec<([f64; 3], f64)> = Vec::new();
        for i in 0..m {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / m as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * i as f64;
            let p = [r * th.cos(), r * th.sin(), z];
            let clearance = self
                .pts
                .iter()
                .flat_map(|f| f.iter().map(|q| geodesic(p, *q)))
                .fold(f64::INFINITY, f64::min);
            poles.push((p, clearance));
        }
        poles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        poles.truncate(n_axes);

        let pos = self.positive();
        let mut values = Vec::new();
        for (pole, _) in &poles {
            let all = self.project(*pole);
            let sel: Vec<Worldline> = pos.iter().map(|&s| all[s].clone()).collect();
            let mut start = 0usize;
            while start + w <= n {
                let cut: Vec<Worldline> = sel
                    .iter()
                    .map(|wl| Worldline {
                        positions: wl.positions[start..start + w].to_vec(),
                        charge: wl.charge,
                    })
                    .collect();
                values.push(crate::topological_entropy(&extract_braidword(&cut)));
                start += stride;
            }
        }
        EntropyScan::from_values(values, poles.len(), window)
    }

    /// Accumulated topological entropy against time.
    ///
    /// The braid word is read from the start of the record to each of `points`
    /// times in turn, and the entropy of each prefix returned with the time it
    /// ended at. The slope is the entropy rate, so the shape of this curve is
    /// the whole question of whether a flow mixes steadily: a straight line is
    /// a sustained braid, and a staircase is a sequence of rearrangements with
    /// quiet stretches between that no single number distinguishes from it.
    pub fn entropy_curve(&self, pole: [f64; 3], points: usize) -> Vec<(f64, f64)> {
        let n = self.n_frames();
        if n < 8 || points == 0 {
            return Vec::new();
        }
        let pos = self.positive();
        let all = self.project(pole);
        let sel: Vec<Worldline> = pos.iter().map(|&s| all[s].clone()).collect();
        (1..=points)
            .filter_map(|i| {
                let end = n * i / points;
                if end < 4 {
                    return None;
                }
                let cut: Vec<Worldline> = sel
                    .iter()
                    .map(|w| Worldline {
                        positions: w.positions[..end].to_vec(),
                        charge: w.charge,
                    })
                    .collect();
                let h = crate::topological_entropy(&extract_braidword(&cut));
                Some((self.times[end - 1], h))
            })
            .collect()
    }

    /// Passes between positive strands, no two closer in time than
    /// `refractory`.
    ///
    /// A pass is a local minimum of the geodesic separation below `close`, in
    /// radians. Detected positions are quantised by whatever produced them, so
    /// one approach plateaus and every frame of the plateau reads as a pass
    /// without the refractory window.
    pub fn passes(&self, close: f64, refractory: f64) -> Vec<SpherePass> {
        let pos = self.positive();
        let mut out = Vec::new();
        for (ia, &a) in pos.iter().enumerate() {
            for &b in pos.iter().skip(ia + 1) {
                let d: Vec<f64> = (0..self.n_frames())
                    .map(|k| geodesic(self.pts[k][a], self.pts[k][b]))
                    .collect();
                let mut here: Vec<SpherePass> = Vec::new();
                for k in 1..d.len().saturating_sub(1) {
                    if !(d[k] < d[k - 1] && d[k] <= d[k + 1] && d[k] < close) {
                        continue;
                    }
                    // Sense from the turn of the separation across the pass,
                    // measured in the tangent plane at the midpoint so the
                    // sign is the one an observer outside the sphere sees.
                    let lo = k.saturating_sub(3);
                    let hi = (k + 3).min(d.len() - 1);
                    let mid = norm([
                        self.pts[k][a][0] + self.pts[k][b][0],
                        self.pts[k][a][1] + self.pts[k][b][1],
                        self.pts[k][a][2] + self.pts[k][b][2],
                    ]);
                    let u = tangent(self.pts[lo][a], self.pts[lo][b], mid);
                    let w = tangent(self.pts[hi][a], self.pts[hi][b], mid);
                    let sense = if dot(cross(u, w), mid) >= 0.0 { 1 } else { -1 };
                    let e = SpherePass {
                        frame: k,
                        t: self.times[k],
                        strands: (a, b),
                        distance: d[k],
                        sense,
                    };
                    match here.last_mut() {
                        Some(prev) if e.t - prev.t < refractory => {
                            if e.distance < prev.distance {
                                *prev = e;
                            }
                        }
                        _ => here.push(e),
                    }
                }
                out.extend(here);
            }
        }
        out.sort_by_key(|e| e.frame);
        out
    }
}

/// The separation of two points, projected onto the tangent plane at `at`.
fn tangent(a: [f64; 3], b: [f64; 3], at: [f64; 3]) -> [f64; 3] {
    let v = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let d = dot(v, at);
    [v[0] - d * at[0], v[1] - d * at[1], v[2] - d * at[2]]
}

/// Four points turning as a rigid tetrahedron about an axis.
///
/// The strands stay mutually equidistant, so nothing ever passes anything: a
/// braid read from this is trivial and its entropy is zero. It is the negative
/// control: a tracker that mislabels strands and a projection with the pole in
/// the wrong place both invent crossings, and this is the configuration where
/// there are none to find.
pub fn rigid_tetrahedron(n_frames: usize, turns: f64) -> Vec<SphereFrame> {
    let base = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    (0..n_frames)
        .map(|k| {
            let t = k as f64 / (n_frames - 1).max(1) as f64;
            let a = 2.0 * std::f64::consts::PI * turns * t;
            let (s, c) = a.sin_cos();
            let pts = base
                .iter()
                .map(|p| {
                    let q = norm(*p);
                    (norm([q[0] * c - q[1] * s, q[0] * s + q[1] * c, q[2]]), 1)
                })
                .collect();
            (t, pts)
        })
        .collect()
}

#[cfg(test)]
mod tests {

    /// A tetrahedron that breathes while it turns is periodic, and the shape
    /// measure must say so.
    ///
    /// This is the case the old self-recurrence measure could not see. The
    /// configuration precesses, so no defect ever returns to its own earlier
    /// position and a strand-to-itself measure reports nothing; the shape
    /// repeats exactly once a period.
    #[test]
    fn a_precessing_configuration_reads_as_periodic() {
        let n = 400usize;
        let dt = 0.5_f64;
        let period = 40.0_f64;
        let mut frames: Vec<SphereFrame> = Vec::with_capacity(n);
        let base = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        for k in 0..n {
            let t = k as f64 * dt;
            // A precession that never repeats on its own, and a breathing that
            // repeats every `period`.
            let spin = 0.031 * t;
            // One-signed, so the deformation at `+b` is never the mirror of
            // one at `-b`. A sine would make the shape repeat twice a period,
            // correctly, since a mirrored configuration has the same
            // separations.
            let breathe = 0.2 * (1.0 - (std::f64::consts::TAU * t / period).cos());
            let pts: Vec<([f64; 3], i32)> = base
                .iter()
                .map(|b| {
                    let v = norm(*b);
                    // Open the configuration towards the z axis and back.
                    let w = norm([v[0], v[1], v[2] + breathe]);
                    let (c, s) = (spin.cos(), spin.sin());
                    (norm([c * w[0] - s * w[1], s * w[0] + c * w[1], w[2]]), 1)
                })
                .collect();
            frames.push((t, pts));
        }
        let w = track_on_sphere(&frames, 0.6).expect("tracking");

        let (p_shape, q_shape) = w.shape_period();
        assert!(
            (p_shape - period).abs() < 0.15 * period,
            "the shape period should be {period}, got {p_shape}"
        );
        assert!(q_shape > 0.7, "a clean repeat should score high, got {q_shape}");

        // The control: no defect returns to its own position, which is why the
        // measure had to be built on the shape.
        let (p_self, _) = w.orbit_period();
        assert!(
            !p_self.is_finite() || (p_self - period).abs() > 0.3 * period,
            "the self-recurrence measure should NOT find this period, got {p_self}"
        );
    }

    /// A rigid rotation writes no entropy however long it is watched.
    #[test]
    fn a_rigid_rotation_has_no_entropy_rate() {
        let frames = rigid_tetrahedron(400, 6.0);
        let w = track_on_sphere(&frames, 0.6).expect("tracking");
        let (rate, spread, positive) = w.entropy_rate(w.far_pole());
        assert_eq!(rate, 0.0, "a rigid rotation should write no entropy");
        assert_eq!(spread, 0.0);
        assert_eq!(positive, 0.0, "and no window of it should mix");
    }

    /// A rate that exists is a rate that settles.
    ///
    /// The spread across windows is the whole point of the measure: a word's
    /// entropy alone cannot tell a steady braid from one burst of crossings.
    #[test]
    fn the_entropy_rate_reports_its_own_spread() {
        let frames = rigid_tetrahedron(400, 6.0);
        let w = track_on_sphere(&frames, 0.6).expect("tracking");
        let (_, spread, positive) = w.entropy_rate(w.far_pole());
        assert!(spread.is_finite(), "the spread must be reportable");
        assert!((0.0..=1.0).contains(&positive), "a fraction, got {positive}");
    }
    use super::*;

    #[test]
    fn geodesic_is_the_angle() {
        let a = [1.0, 0.0, 0.0];
        assert!((geodesic(a, [0.0, 1.0, 0.0]) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((geodesic(a, [-1.0, 0.0, 0.0]) - std::f64::consts::PI).abs() < 1e-12);
        assert!(geodesic(a, a) < 1e-12);
        // Near-coincident points, where `acos` of the dot product would have
        // lost most of its significant figures.
        let eps = 1e-8;
        let b = norm([1.0, eps, 0.0]);
        assert!((geodesic(a, b) - eps).abs() < 1e-16 + 1e-6 * eps);
    }

    #[test]
    fn a_rigid_turn_writes_no_crossings() {
        let frames = rigid_tetrahedron(721, 2.0);
        let w = track_on_sphere(&frames, 0.2).expect("first frame is not empty");
        assert_eq!(w.n_frames(), 721);
        assert_eq!(w.n_strands(), 4);

        // The four stay mutually equidistant, at the tetrahedral angle.
        let tetra = (-1.0_f64 / 3.0).acos();
        for k in (0..w.n_frames()).step_by(37) {
            for i in 0..4 {
                for j in i + 1..4 {
                    assert!(
                        (geodesic(w.pts[k][i], w.pts[k][j]) - tetra).abs() < 1e-9,
                        "frame {k}, strands {i} {j}"
                    );
                }
            }
        }
        // So there is nothing to pass, at any threshold short of the whole
        // sphere, and the braid is trivial.
        assert!(w.passes(tetra - 1e-6, 0.0).is_empty());
        let word = w.braid_word(w.far_pole());
        assert_eq!(
            crate::topological_entropy(&word),
            0.0,
            "a rigid turn should write no entropy, word {word:?}"
        );
    }

    #[test]
    fn the_pole_avoids_every_strand() {
        let frames = rigid_tetrahedron(181, 1.0);
        let w = track_on_sphere(&frames, 0.2).unwrap();
        let p = w.far_pole();
        let worst = w
            .pts
            .iter()
            .flat_map(|f| f.iter().map(|q| geodesic(p, *q)))
            .fold(f64::INFINITY, f64::min);
        // A tetrahedron turning about its own axis sweeps two circles of
        // latitude at polar angle `arccos(1/sqrt 3) = 0.9553`, so the widest
        // empty cap is centred on the axis and has exactly that radius. The
        // refined search should land on it rather than on the nearest lattice
        // sample, which is a tenth of a radian short.
        let optimum = (1.0_f64 / 3.0_f64.sqrt()).acos();
        assert!(
            worst > optimum - 1e-3,
            "closest strand to the pole was {worst} rad, optimum {optimum}"
        );
        // Projected coordinates stay finite, which is the point of the choice.
        for wl in w.project(p) {
            for q in wl.positions {
                assert!(q[0].is_finite() && q[1].is_finite());
                assert!(q[0].abs() < 1e3 && q[1].abs() < 1e3, "{q:?}");
            }
        }
    }

    #[test]
    fn tracking_survives_a_transient_pair() {
        let mut frames = rigid_tetrahedron(120, 0.5);
        // A short-lived extra pair beside the cast, as a detector flicker or a
        // real nucleation would give.
        for f in frames[40..50].iter_mut() {
            f.1.push(([0.0, 0.0, 1.0], 1));
            f.1.push(([0.0, 0.0, -1.0], -1));
        }
        let w = track_on_sphere(&frames, 0.2).unwrap();
        assert_eq!(w.n_strands(), 4, "the cast is the first frame's");
        assert_eq!(w.n_frames(), 120, "the extras should not end the track");
    }

    #[test]
    fn charge_is_kept_for_the_life_of_a_strand() {
        let mut frames = rigid_tetrahedron(60, 0.25);
        for f in frames.iter_mut() {
            f.1[2].1 = -1;
            f.1[3].1 = -1;
        }
        let w = track_on_sphere(&frames, 0.2).unwrap();
        assert_eq!(w.charge, vec![1, 1, -1, -1]);
        assert_eq!(w.positive(), vec![0, 1]);
        // The word is read from the positive strands alone.
        assert_eq!(w.braid_word(w.far_pole()).n_strands, 2);
    }
}
