//! Plane curves as boundary geometry for the confined mesher.
//!
//! [`confined_mesh`](crate::confined::confined_mesh) reads a wall through
//! [`PlaneCurve`]: its position, the speed of its parametrisation, its
//! curvature, and which side is the interior. A type implementing the trait is
//! a domain the mesher accepts.
//!
//! Two implementations ship. [`Epitrochoid`](crate::confined::Epitrochoid) is
//! analytic, and the cusp parameters it reports are exact. [`PolyCurve`] takes a
//! closed list of sampled points and interpolates them with a periodic cubic
//! spline, so a wall supplied as a table of points is a domain. The spline is
//! `C^2`, which is the regularity the curvature-graded boundary sampling reads,
//! and it interpolates, so the mesh's boundary vertices sit on the points
//! supplied.
//!
//! ## What the mesher asks for
//!
//! - `point`, `d1` and `d2` at a parameter. Everything geometric is derived:
//!   speed, unit tangent, inward normal, radius of curvature.
//! - `period`, the parameter length of one circuit.
//! - `features`, the parameters of the corners and cusps that set the local
//!   element size. An empty list means the wall is smooth everywhere and the
//!   bulk size is used throughout.
//! - `interior`, any point inside the domain, which orients the normal.
//!
//! ## Anchoring
//!
//! The mesher stores the inward normal at every boundary vertex, and the
//! anchoring is a function of its angle alone
//! ([`anchored_q`](crate::confined_ldg::anchored_q)), so a new geometry uses the
//! boundary-condition code unchanged.

use std::f64::consts::TAU;
use std::sync::Arc;

/// Radius of curvature, as a fraction of the curve's circle-equivalent radius,
/// below which a sample is a feature. A circle sits at exactly 1 everywhere, so
/// any fraction under 1 reports none on it. The value is a quarter because the
/// walls this meshes have features an order of magnitude tighter than that, and
/// a lower cut starts reporting the gentle turns a bulk element already resolves.
pub const AUTO_FEATURE_FRACTION: f64 = 0.25;

/// A closed, oriented plane curve bounding a simply connected domain.
///
/// Implement `point`, `d1` and `d2`. Override the rest where the type has an
/// analytic form for it.
pub trait PlaneCurve: Send + Sync {
    /// Position at parameter `u`.
    fn point(&self, u: f64) -> [f64; 2];

    /// First derivative with respect to the parameter.
    fn d1(&self, u: f64) -> [f64; 2];

    /// Second derivative with respect to the parameter.
    fn d2(&self, u: f64) -> [f64; 2];

    /// Parameter length of one circuit.
    fn period(&self) -> f64 {
        TAU
    }

    /// Parameters of the corners and cusps, in increasing order.
    ///
    /// These points set the local element size, and, when
    /// `MeshOpts::cusp_edge` is set, the mesher samples the branches leaving
    /// each of them down to a single element. An empty list is the smooth case.
    fn features(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Whether the features are equally spaced in the parameter and the arcs
    /// between them congruent.
    ///
    /// True lets the mesher build one arc and generate the rest by reflection
    /// and rotation, which makes the boundary sampling exactly symmetric. False
    /// walks once round instead.
    fn feature_symmetric(&self) -> bool {
        false
    }

    /// A point inside the domain, used to orient the normal.
    fn interior(&self) -> [f64; 2] {
        [0.0, 0.0]
    }

    /// `|r'(u)|`, which collapses at a cusp.
    fn speed(&self, u: f64) -> f64 {
        let p = self.d1(u);
        (p[0] * p[0] + p[1] * p[1]).sqrt()
    }

    /// Unit tangent, taken from a neighbouring parameter where the speed
    /// vanishes.
    fn tangent(&self, u: f64) -> [f64; 2] {
        let p = self.d1(u);
        let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if n > 1e-12 {
            return [p[0] / n, p[1] / n];
        }
        let eps = 1e-6 * self.period() / TAU;
        let p = self.d1(u + eps);
        let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if n > 1e-300 { [p[0] / n, p[1] / n] } else { [1.0, 0.0] }
    }

    /// Inward unit normal, oriented by testing a short step against
    /// [`PlaneCurve::interior`].
    fn inward_normal(&self, u: f64) -> [f64; 2] {
        let t = self.tangent(u);
        let n = [-t[1], t[0]];
        let p = self.point(u);
        let c = self.interior();
        let towards = (c[0] - p[0]) * n[0] + (c[1] - p[1]) * n[1];
        if towards >= 0.0 { n } else { [-n[0], -n[1]] }
    }

    /// Radius of curvature, infinite where the curve is locally straight.
    fn curvature_radius(&self, u: f64) -> f64 {
        let (p1, p2) = (self.d1(u), self.d2(u));
        let cross = (p1[0] * p2[1] - p1[1] * p2[0]).abs();
        let s = self.speed(u);
        if cross <= 1e-300 { f64::INFINITY } else { s * s * s / cross }
    }

    /// Parametric half-width at which the two branches leaving the first
    /// feature are `edge` apart along the curve, so the two boundary edges
    /// meeting at that vertex have that length.
    ///
    /// Zero when the curve has no feature, and for `edge <= 0`.
    fn feature_edge_param(&self, edge: f64) -> f64 {
        let f = self.features();
        // `!(edge > 0.0)` catches NaN as well as non-positive, which is the
        // point: an unset or malformed edge length must fall through here.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if f.is_empty() || !(edge > 0.0) {
            return 0.0;
        }
        let half = if f.len() >= 2 {
            0.5 * (f[1] - f[0])
        } else {
            0.5 * self.period()
        };
        let uc = f[0];
        let pc = self.point(uc);
        let along = |e: f64| {
            let p = self.point(uc + e);
            ((p[0] - pc[0]).powi(2) + (p[1] - pc[1]).powi(2)).sqrt()
        };
        let (mut lo, mut hi) = (0.0_f64, 0.9 * half);
        if along(hi) <= edge {
            return hi;
        }
        for _ in 0..90 {
            let mid = 0.5 * (lo + hi);
            if along(mid) < edge { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }
}

/// A curve behind a shared pointer is the same curve.
///
/// Every method is forwarded, the provided ones included, so an implementation
/// that overrides a default keeps that override through the pointer.
impl<T: PlaneCurve + ?Sized> PlaneCurve for Arc<T> {
    fn point(&self, u: f64) -> [f64; 2] { (**self).point(u) }
    fn d1(&self, u: f64) -> [f64; 2] { (**self).d1(u) }
    fn d2(&self, u: f64) -> [f64; 2] { (**self).d2(u) }
    fn period(&self) -> f64 { (**self).period() }
    fn features(&self) -> Vec<f64> { (**self).features() }
    fn feature_symmetric(&self) -> bool { (**self).feature_symmetric() }
    fn interior(&self) -> [f64; 2] { (**self).interior() }
    fn speed(&self, u: f64) -> f64 { (**self).speed(u) }
    fn tangent(&self, u: f64) -> [f64; 2] { (**self).tangent(u) }
    fn inward_normal(&self, u: f64) -> [f64; 2] { (**self).inward_normal(u) }
    fn curvature_radius(&self, u: f64) -> f64 { (**self).curvature_radius(u) }
    fn feature_edge_param(&self, edge: f64) -> f64 { (**self).feature_edge_param(edge) }
}

/// A closed curve through sampled points, interpolated by a periodic cubic
/// spline.
///
/// The parameter is the sample index: `u` runs over `[0, n)` and the period is
/// `n`, so `point(k as f64)` returns the `k`th point supplied, exactly. Sample
/// density is therefore the caller's parametrisation, and a wall that turns
/// sharply somewhere is described by sampling it more finely there.
///
/// The spline is the standard one: second-derivative moments `M` from
///
/// ```text
/// M_{i-1} + 4 M_i + M_{i+1} = 6 (y_{i-1} - 2 y_i + y_{i+1})
/// ```
///
/// taken round the loop, which is a cyclic tridiagonal system solved exactly by
/// Sherman-Morrison. The spline interpolates, which puts the boundary vertices
/// on the wall the caller supplied.
#[derive(Debug, Clone)]
pub struct PolyCurve {
    xs: Vec<f64>,
    ys: Vec<f64>,
    mx: Vec<f64>,
    my: Vec<f64>,
    features: Vec<f64>,
    interior: [f64; 2],
    ccw: bool,
}

impl PolyCurve {
    /// Build from a closed list of points, in order along the wall.
    ///
    /// The list must not repeat the first point at the end: the curve closes on
    /// its own. `features` are parameters in `[0, n)`, the corners that set the
    /// local element size; pass an empty slice for a smooth wall.
    ///
    /// Fails with `None` for fewer than three points, or for a degenerate
    /// polygon of zero signed area.
    pub fn new(points: &[[f64; 2]], features: &[f64]) -> Option<Self> {
        let n = points.len();
        if n < 3 {
            return None;
        }
        let xs: Vec<f64> = points.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = points.iter().map(|p| p[1]).collect();
        if !xs.iter().chain(ys.iter()).all(|v| v.is_finite()) {
            return None;
        }

        // Shoelace: twice the signed area, and six times the centroid moment.
        let (mut a2, mut cx, mut cy) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = xs[i] * ys[j] - xs[j] * ys[i];
            a2 += cross;
            cx += (xs[i] + xs[j]) * cross;
            cy += (ys[i] + ys[j]) * cross;
        }
        if a2.abs() < 1e-300 {
            return None;
        }
        let interior = [cx / (3.0 * a2), cy / (3.0 * a2)];

        let mut features = features.to_vec();
        features.retain(|f| f.is_finite());
        features.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Some(Self {
            mx: spline_moments(&xs),
            my: spline_moments(&ys),
            xs,
            ys,
            features,
            interior,
            ccw: a2 > 0.0,
        })
    }

    /// Build from a closed list of points, taking the features from the
    /// curvature.
    ///
    /// See [`detect_features`](Self::detect_features) for what is found and
    /// what is missed.
    pub fn new_auto(points: &[[f64; 2]]) -> Option<Self> {
        let probe = Self::new(points, &[])?;
        let features = probe.detect_features(AUTO_FEATURE_FRACTION);
        Some(Self { features, ..probe })
    }

    /// Sample parameters where the wall turns sharply, one per run.
    ///
    /// A sample is tight when its radius of curvature falls below `frac` of the
    /// curve's circle-equivalent radius, the perimeter over `2 pi`. That scale
    /// is the curve's own, so the answer is independent of the units the points
    /// were given in and of the mesh they are later meshed at. Consecutive tight
    /// samples are one feature, reported at the tightest of them.
    ///
    /// A wall whose tight samples exceed half the total is uniformly curved
    /// rather than featured, and returns none: refining everywhere is what
    /// `h_bulk` is for.
    pub fn detect_features(&self, frac: f64) -> Vec<f64> {
        let n = self.xs.len();
        if n < 3 || !frac.is_finite() || frac <= 0.0 {
            return Vec::new();
        }
        let mut perimeter = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += (self.xs[i] - self.xs[j]).hypot(self.ys[i] - self.ys[j]);
        }
        let scale = perimeter / TAU;
        if !scale.is_finite() || scale <= 0.0 {
            return Vec::new();
        }
        let cut = frac * scale;

        let radii: Vec<f64> = (0..n).map(|i| self.curvature_radius(i as f64)).collect();
        let tight: Vec<bool> = radii.iter().map(|r| *r < cut).collect();
        let count = tight.iter().filter(|t| **t).count();
        if count == 0 || count * 2 > n {
            return Vec::new();
        }

        // Start on a slack sample so a run never straddles the wrap.
        let start = match tight.iter().position(|t| !*t) {
            Some(k) => k,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < n {
            let k = (start + i) % n;
            if !tight[k] {
                i += 1;
                continue;
            }
            let (mut best, mut best_r) = (k, radii[k]);
            let mut run = 0usize;
            while i + run < n {
                let m = (start + i + run) % n;
                if !tight[m] {
                    break;
                }
                if radii[m] < best_r {
                    best_r = radii[m];
                    best = m;
                }
                run += 1;
            }
            out.push(best as f64);
            i += run;
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Number of samples, which is also the period of the parametrisation.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Always false: a `PolyCurve` needs three points to exist.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Split `u` into a segment index and a local coordinate in `[0, 1)`.
    fn locate(&self, u: f64) -> (usize, usize, f64) {
        let n = self.xs.len();
        let nf = n as f64;
        let w = u - nf * (u / nf).floor();
        let i = (w.floor() as usize).min(n - 1);
        let t = (w - i as f64).clamp(0.0, 1.0);
        (i, (i + 1) % n, t)
    }
}

impl PlaneCurve for PolyCurve {
    fn point(&self, u: f64) -> [f64; 2] {
        let (i, j, t) = self.locate(u);
        let s = 1.0 - t;
        let a = (s * s * s - s) / 6.0;
        let b = (t * t * t - t) / 6.0;
        [
            s * self.xs[i] + t * self.xs[j] + a * self.mx[i] + b * self.mx[j],
            s * self.ys[i] + t * self.ys[j] + a * self.my[i] + b * self.my[j],
        ]
    }

    fn d1(&self, u: f64) -> [f64; 2] {
        let (i, j, t) = self.locate(u);
        let s = 1.0 - t;
        let a = (1.0 - 3.0 * s * s) / 6.0;
        let b = (3.0 * t * t - 1.0) / 6.0;
        [
            self.xs[j] - self.xs[i] + a * self.mx[i] + b * self.mx[j],
            self.ys[j] - self.ys[i] + a * self.my[i] + b * self.my[j],
        ]
    }

    fn d2(&self, u: f64) -> [f64; 2] {
        let (i, j, t) = self.locate(u);
        let s = 1.0 - t;
        [
            s * self.mx[i] + t * self.mx[j],
            s * self.my[i] + t * self.my[j],
        ]
    }

    fn period(&self) -> f64 {
        self.xs.len() as f64
    }

    fn features(&self) -> Vec<f64> {
        self.features.clone()
    }

    fn interior(&self) -> [f64; 2] {
        self.interior
    }

    /// The polygon's own orientation, which stays correct on a re-entrant
    /// domain, where a step towards the centroid can pick the wrong side.
    fn inward_normal(&self, u: f64) -> [f64; 2] {
        let t = self.tangent(u);
        if self.ccw { [-t[1], t[0]] } else { [t[1], -t[0]] }
    }
}

/// Second-derivative moments of the periodic cubic spline through `y`.
///
/// The system is circulant with 4 on the diagonal and 1 either side, wrapping at
/// both corners, so it is diagonally dominant and Sherman-Morrison on a
/// tridiagonal solve returns it exactly.
fn spline_moments(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let rhs: Vec<f64> = (0..n)
        .map(|i| {
            let p = y[(i + n - 1) % n];
            let q = y[(i + 1) % n];
            6.0 * (p - 2.0 * y[i] + q)
        })
        .collect();

    // Sherman-Morrison: A = T + u v^T with u = (gamma, 0, .., 0, alpha) and
    // v = (1, 0, .., 0, beta / gamma), for alpha = beta = 1 the corner entries.
    let gamma = -4.0_f64;
    let mut diag = vec![4.0_f64; n];
    diag[0] -= gamma;
    diag[n - 1] -= 1.0 / gamma;

    let x = thomas(&diag, &rhs);
    let mut z_rhs = vec![0.0_f64; n];
    z_rhs[0] = gamma;
    z_rhs[n - 1] = 1.0;
    let z = thomas(&diag, &z_rhs);

    let fact = (x[0] + x[n - 1] / gamma) / (1.0 + z[0] + z[n - 1] / gamma);
    (0..n).map(|i| x[i] - fact * z[i]).collect()
}

/// Thomas algorithm for a tridiagonal system with unit off-diagonals.
fn thomas(diag: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    let mut b = diag[0];
    d[0] = rhs[0] / b;
    c[0] = 1.0 / b;
    for i in 1..n {
        b = diag[i] - c[i - 1];
        c[i] = 1.0 / b;
        d[i] = (rhs[i] - d[i - 1]) / b;
    }
    let mut out = vec![0.0_f64; n];
    out[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = d[i] - c[i] * out[i + 1];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sampled(n: usize, f: impl Fn(f64) -> [f64; 2]) -> Vec<[f64; 2]> {
        (0..n).map(|i| f(TAU * i as f64 / n as f64)).collect()
    }

    fn nephroid_points(n: usize, d: f64) -> Vec<[f64; 2]> {
        sampled(n, |u| {
            [
                10.0 * (3.0 * u.cos() + d * (3.0 * u).cos()),
                10.0 * (3.0 * u.sin() + d * (3.0 * u).sin()),
            ]
        })
    }

    #[test]
    fn a_circle_has_no_features() {
        // Every sample sits at exactly the circle-equivalent radius, so no
        // fraction below one reports anything. This is the case that says the
        // scale is right rather than merely consistent.
        let c = PolyCurve::new_auto(&sampled(600, |u| [30.0 * u.cos(), 30.0 * u.sin()])).unwrap();
        assert!(c.features().is_empty(), "found {:?}", c.features());
    }

    #[test]
    fn auto_finds_both_tips_of_a_nephroid() {
        let c = PolyCurve::new_auto(&nephroid_points(900, 0.85)).unwrap();
        let f = c.features();
        assert_eq!(f.len(), 2, "found {f:?}");
        // The tips are at u = pi/2 and 3 pi/2, which are rows 225 and 675.
        assert!((f[0] - 225.0).abs() <= 1.0, "{f:?}");
        assert!((f[1] - 675.0).abs() <= 1.0, "{f:?}");
    }

    #[test]
    fn auto_finds_every_corner_of_a_square() {
        let n = 40;
        let mut pts = Vec::new();
        for side in 0..4 {
            for i in 0..n {
                let t = 2.0 * (i as f64 / n as f64) - 1.0;
                pts.push(match side {
                    0 => [30.0 * t, -30.0],
                    1 => [30.0, 30.0 * t],
                    2 => [-30.0 * t, 30.0],
                    _ => [-30.0, -30.0 * t],
                });
            }
        }
        let c = PolyCurve::new_auto(&pts).unwrap();
        assert_eq!(c.features().len(), 4, "found {:?}", c.features());
    }

    #[test]
    fn auto_agrees_with_declaring_by_hand() {
        let pts = nephroid_points(900, 0.85);
        let auto = PolyCurve::new_auto(&pts).unwrap().features();
        let hand = PolyCurve::new(&pts, &[225.0, 675.0]).unwrap().features();
        assert_eq!(auto, hand);
    }

    #[test]
    fn detection_is_independent_of_the_units() {
        // The cut is a fraction of the curve's own size, so scaling the points
        // scales nothing that matters.
        let small = PolyCurve::new_auto(&nephroid_points(900, 0.85)).unwrap();
        let big: Vec<[f64; 2]> = nephroid_points(900, 0.85)
            .iter()
            .map(|p| [1000.0 * p[0], 1000.0 * p[1]])
            .collect();
        assert_eq!(small.features(), PolyCurve::new_auto(&big).unwrap().features());
    }

    #[test]
    fn every_lobe_of_a_flower_is_a_feature() {
        // Twenty lobes turn sharply at both the crest and the trough, so forty
        // features is the answer rather than a detector running away.
        let pts = sampled(200, |u| {
            let r = 30.0 * (1.0 + 0.35 * (20.0 * u).cos());
            [r * u.cos(), r * u.sin()]
        });
        let c = PolyCurve::new_auto(&pts).unwrap();
        assert_eq!(c.features().len(), 40, "found {:?}", c.features());
    }

    #[test]
    fn a_uniformly_curved_wall_reports_none() {
        // A fraction above one makes every sample of a circle tight, which is
        // the wall that turns everywhere. Refining everywhere is what the bulk
        // size is for, so the answer is none.
        let c = PolyCurve::new(&sampled(600, |u| [30.0 * u.cos(), 30.0 * u.sin()]), &[]).unwrap();
        assert!(c.detect_features(2.0).is_empty());
    }

    #[test]
    fn a_run_of_tight_samples_is_one_feature() {
        // Densely sampled through the tip, so many consecutive samples fall
        // below the cut and a naive scan would report each of them.
        let c = PolyCurve::new_auto(&nephroid_points(4000, 0.9)).unwrap();
        assert_eq!(c.features().len(), 2, "found {:?}", c.features());
    }

    #[test]
    fn a_feature_at_the_wrap_is_found_once() {
        // Rotated so a tip sits on sample zero, where a run straddles the seam.
        let pts = sampled(900, |u| {
            let v = u + PI / 2.0;
            [
                10.0 * (3.0 * v.cos() + 0.85 * (3.0 * v).cos()),
                10.0 * (3.0 * v.sin() + 0.85 * (3.0 * v).sin()),
            ]
        });
        let c = PolyCurve::new_auto(&pts).unwrap();
        assert_eq!(c.features().len(), 2, "found {:?}", c.features());
    }

    #[test]
    fn a_zero_fraction_reports_none() {
        let c = PolyCurve::new(&nephroid_points(900, 0.85), &[]).unwrap();
        assert!(c.detect_features(0.0).is_empty());
    }

    fn circle(n: usize, r: f64) -> PolyCurve {
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        PolyCurve::new(&pts, &[]).unwrap()
    }

    #[test]
    fn spline_interpolates_its_samples() {
        let c = circle(64, 3.0);
        for k in 0..64 {
            let p = c.point(k as f64);
            let a = TAU * k as f64 / 64.0;
            assert!((p[0] - 3.0 * a.cos()).abs() < 1e-12);
            assert!((p[1] - 3.0 * a.sin()).abs() < 1e-12);
        }
    }

    #[test]
    fn spline_is_periodic() {
        let c = circle(48, 1.0);
        for &u in &[0.0, 0.37, 12.9, 47.5] {
            let a = c.point(u);
            let b = c.point(u + 48.0);
            assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        }
    }

    /// The circle has a radius of curvature equal to its radius at every
    /// parameter, and the spline returns it to the order of its own truncation
    /// error.
    #[test]
    fn circle_curvature_radius_is_the_radius() {
        let c = circle(256, 5.0);
        for k in 0..256 {
            let u = k as f64 + 0.5;
            let rc = c.curvature_radius(u);
            assert!(
                (rc - 5.0).abs() < 5.0 * 1e-3,
                "u = {u}: radius of curvature {rc}, expected 5"
            );
        }
    }

    #[test]
    fn normal_points_inward() {
        for n in [16usize, 64] {
            let c = circle(n, 2.0);
            for k in 0..n {
                let u = k as f64 + 0.25;
                let p = c.point(u);
                let nrm = c.inward_normal(u);
                assert!(
                    p[0] * nrm[0] + p[1] * nrm[1] < 0.0,
                    "normal at u = {u} points out of the disc"
                );
            }
        }
    }

    /// Reversing the sample order reverses the tangent and must leave the normal
    /// pointing the same way.
    #[test]
    fn normal_is_orientation_independent() {
        let n = 32;
        let fwd: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                [2.0 * a.cos(), 2.0 * a.sin()]
            })
            .collect();
        let mut rev = fwd.clone();
        rev.reverse();
        let a = PolyCurve::new(&fwd, &[]).unwrap();
        let b = PolyCurve::new(&rev, &[]).unwrap();
        let na = a.inward_normal(0.0);
        let nb = b.inward_normal((n - 1) as f64);
        assert!((na[0] - nb[0]).abs() < 1e-9 && (na[1] - nb[1]).abs() < 1e-9);
    }

    #[test]
    fn centroid_of_an_offset_circle() {
        let n = 128;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                [3.0 + a.cos(), -1.0 + a.sin()]
            })
            .collect();
        let c = PolyCurve::new(&pts, &[]).unwrap();
        let i = c.interior();
        assert!((i[0] - 3.0).abs() < 1e-6 && (i[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn too_few_points_is_rejected() {
        assert!(PolyCurve::new(&[[0.0, 0.0], [1.0, 0.0]], &[]).is_none());
        let flat = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert!(PolyCurve::new(&flat, &[]).is_none());
    }

    /// A square, whose corners are features: the edge parameter is the offset
    /// from the first corner at which the arc reaches the requested length, and
    /// on a unit-spaced sampling of a side that offset is the length itself.
    #[test]
    fn feature_edge_param_measures_arc() {
        let mut pts = Vec::new();
        let m = 20;
        for i in 0..m { pts.push([i as f64 / m as f64, 0.0]); }
        for i in 0..m { pts.push([1.0, i as f64 / m as f64]); }
        for i in 0..m { pts.push([1.0 - i as f64 / m as f64, 1.0]); }
        for i in 0..m { pts.push([0.0, 1.0 - i as f64 / m as f64]); }
        let c = PolyCurve::new(&pts, &[0.0, 20.0, 40.0, 60.0]).unwrap();
        let e = c.feature_edge_param(0.1);
        let p = c.point(e);
        let len = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!((len - 0.1).abs() < 1e-6, "arc {len}, expected 0.1");
        assert_eq!(c.feature_edge_param(0.0), 0.0);
        assert_eq!(c.feature_edge_param(-1.0), 0.0);
    }

    /// An ellipse has a known radius of curvature, `(a^2 sin^2 + b^2 cos^2)^{3/2}
    /// / (a b)`, and the spline should land on it away from the ends of its own
    /// segments.
    #[test]
    fn ellipse_curvature_matches_the_closed_form() {
        let (a, b) = (4.0, 2.0);
        let n = 512;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = TAU * i as f64 / n as f64;
                [a * t.cos(), b * t.sin()]
            })
            .collect();
        let c = PolyCurve::new(&pts, &[]).unwrap();
        for k in [0usize, 37, 128, 300] {
            // The spline is evaluated at the midpoint of a segment, so the
            // closed form has to be read at that parameter and not at the knot.
            let u = k as f64 + 0.5;
            let t = TAU * u / n as f64;
            let exact = (a * a * t.sin().powi(2) + b * b * t.cos().powi(2)).powf(1.5) / (a * b);
            let got = c.curvature_radius(u);
            assert!(
                (got - exact).abs() / exact < 2e-3,
                "k = {k}: {got}, expected {exact}"
            );
        }
    }

    /// `PolyCurve` overrides `inward_normal`, so the forwarding impl is what
    /// decides whether the override survives being put behind a pointer.
    #[test]
    fn arc_forwards_an_override() {
        let c = circle(24, 1.5);
        let a: Arc<dyn PlaneCurve> = Arc::new(circle(24, 1.5));
        for k in 0..24 {
            let u = k as f64 + 0.4;
            assert_eq!(c.inward_normal(u), a.inward_normal(u));
            assert_eq!(c.period(), a.period());
            assert_eq!(c.interior(), a.interior());
            assert_eq!(c.curvature_radius(u), a.curvature_radius(u));
        }
    }

    #[test]
    fn period_is_the_sample_count() {
        let c = circle(37, 1.0);
        assert_eq!(c.period(), 37.0);
        assert_eq!(c.len(), 37);
        assert!(!c.is_empty());
    }

    /// A parametrisation that runs at wildly different speeds is still
    /// interpolated at its own samples, which is what lets a caller refine the
    /// sampling where the wall turns.
    #[test]
    fn nonuniform_sampling_is_interpolated() {
        let n = 200;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let s = i as f64 / n as f64;
                // Crowds samples near s = 0.
                let t = TAU * s * s;
                [t.cos(), t.sin()]
            })
            .collect();
        let c = PolyCurve::new(&pts, &[]).unwrap();
        for k in [1usize, 50, 199] {
            let p = c.point(k as f64);
            assert!((p[0] - pts[k][0]).abs() < 1e-12);
            assert!((p[1] - pts[k][1]).abs() < 1e-12);
        }
    }

    #[test]
    fn tangent_is_a_unit_vector() {
        let c = circle(64, 2.5);
        for k in 0..64 {
            let t = c.tangent(k as f64 + 0.3);
            let n = (t[0] * t[0] + t[1] * t[1]).sqrt();
            assert!((n - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn features_come_back_sorted() {
        let c = PolyCurve::new(
            &(0..12)
                .map(|i| {
                    let a = TAU * i as f64 / 12.0;
                    [a.cos(), a.sin()]
                })
                .collect::<Vec<_>>(),
            &[7.0, 1.0, 4.0],
        )
        .unwrap();
        assert_eq!(c.features(), vec![1.0, 4.0, 7.0]);
    }

    /// The default normal orientation, the one an analytic curve inherits, has
    /// to agree with the polygon rule on a star-shaped domain.
    #[test]
    fn default_orientation_agrees_on_a_disc() {
        struct Circle;
        impl PlaneCurve for Circle {
            fn point(&self, u: f64) -> [f64; 2] { [u.cos(), u.sin()] }
            fn d1(&self, u: f64) -> [f64; 2] { [-u.sin(), u.cos()] }
            fn d2(&self, u: f64) -> [f64; 2] { [-u.cos(), -u.sin()] }
        }
        for k in 0..16 {
            let u = PI * k as f64 / 8.0;
            let n = Circle.inward_normal(u);
            let p = Circle.point(u);
            assert!(p[0] * n[0] + p[1] * n[1] < 0.0);
        }
    }
}
