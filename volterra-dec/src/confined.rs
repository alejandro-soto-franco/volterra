//! Boundary-conforming graded meshes for confined epitrochoid domains.
//!
//! `epitrochoid.rs` samples the true epicycloid and fills the interior with a
//! regular grid, then triangulates the union. That puts the boundary vertices on
//! the curve exactly, which is the point of a mesh, but wherever the grid meets
//! the curve at a shallow incidence the Delaunay triangulation of the union has
//! no choice but to emit a sliver: measured min angle falls from 0.60 degrees at
//! spacing 4 to 0.09 degrees at spacing 0.5, so refinement makes it worse. The DEC
//! cotangent Laplacian carries the weight `(cot a + cot b) / 2` per edge, and
//! `cot 179.7 = -191`, so a sliver costs the discrete maximum principle and lets
//! the order parameter overshoot into a core the physics never put there.
//!
//! This module fixes three things at once.
//!
//! **The regularisation `d` is a parameter.** `epitrochoid.rs` hard-codes the true
//! cusp. The tip's radius of curvature is `3 R (1 - d)^2 / 8`, which vanishes
//! quadratically, and at `d = 1` the domain fails to be Lipschitz: the interior
//! angle at the tip is zero, no mesh of bounded aspect ratio exists there, and the
//! regularity of the solution is delicate in its own right. The defensible study
//! keeps `d < 1`, resolves the tip with several elements, and converges in `d`.
//! That is the study a lattice cannot run at all, since at `d = 0.99` and
//! `L = 100` the tip is 268 times finer than one cell.
//!
//! **The boundary is sampled by curvature.** Parameter-uniform sampling is
//! already close to right, because the speed `|r'(u)| = 3 a (1 - d)` collapses at
//! the tip and samples crowd there unasked, but it is uncontrolled: at 512 samples
//! and `d = 0.99` the edge sits at 2 per cent of the local curvature radius
//! everywhere except the tip, where it reaches 297 per cent. Here the step is
//! chosen so the arc never exceeds a set fraction of the local radius.
//!
//! **The interior is layered, then filled.** Points march inward from each
//! boundary sample along the inward normal at a geometric rate, thinning along the
//! wall as they go so each layer stays near-isotropic, until the local size
//! reaches the bulk size. The remainder is filled by dart throwing at the bulk
//! size. Both steps place points at a controlled spacing from their neighbours, so
//! the Delaunay triangulation of the result has no reason to produce a sliver, and
//! a few passes of area-weighted smoothing clean up what is left.
//!
//! The acceptance test is [`MeshQuality`], and the quantity to watch is the
//! minimum angle: a DEC operator wants it above about 25 degrees.

use std::collections::HashMap;
use std::f64::consts::PI;

use std::sync::Arc;

use cartan_dec::mesh::FlatMesh;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::curve::PlaneCurve;

/// A regularised epitrochoid, the curve the reference's `set_boundary` traces.
///
/// ```text
/// x(u) = a [ (k+1) cos u + d cos((k+1) u) ]
/// y(u) = a [ (k+1) sin u + d sin((k+1) u) ]
/// ```
///
/// with `k = 2(q - 1)` cusps and `a = R / (k + 2)`, so the lobe tip sits at
/// `R (k + 1 + d) / (k + 2)` and the cusp at `R (k + 1 - d) / (k + 2)`. At
/// `d = 1` this is the epicycloid with true cusps; at `d = 0` it is a circle.
#[derive(Debug, Clone, Copy)]
pub struct Epitrochoid {
    /// Half-integer winding of the paper's parametrisation: 3/2 cardioid,
    /// 2 nephroid, 5/2 trefoiloid, 3 quatrefoiloid, 7/2 quintefoiloid.
    /// In general `q = 1 + k/2` for `k` cusps.
    pub q: f64,
    /// Cusp regularisation, in `(0, 1]`.
    pub d: f64,
    /// Outer scale, so the lobe tip is at `r (k + 1 + d) / (k + 2)`.
    pub r: f64,
}

impl Epitrochoid {
    /// Cusp count, `k = 2(q - 1)`.
    pub fn cusps(&self) -> usize {
        (2.0 * (self.q - 1.0)).round() as usize
    }

    fn k(&self) -> f64 {
        2.0 * (self.q - 1.0)
    }

    fn a(&self) -> f64 {
        self.r / (self.k() + 2.0)
    }

    /// Point on the curve.
    pub fn point(&self, u: f64) -> [f64; 2] {
        let (k, a, d) = (self.k(), self.a(), self.d);
        let c = k + 1.0;
        [
            a * (c * u.cos() + d * (c * u).cos()),
            a * (c * u.sin() + d * (c * u).sin()),
        ]
    }

    /// First derivative with respect to the parameter.
    pub fn d1(&self, u: f64) -> [f64; 2] {
        let (k, a, d) = (self.k(), self.a(), self.d);
        let c = k + 1.0;
        [
            a * (-c * u.sin() - d * c * (c * u).sin()),
            a * (c * u.cos() + d * c * (c * u).cos()),
        ]
    }

    /// Second derivative with respect to the parameter.
    pub fn d2(&self, u: f64) -> [f64; 2] {
        let (k, a, d) = (self.k(), self.a(), self.d);
        let c = k + 1.0;
        [
            a * (-c * u.cos() - d * c * c * (c * u).cos()),
            a * (-c * u.sin() - d * c * c * (c * u).sin()),
        ]
    }

    /// `|r'(u)|`, which collapses to `3 a (1 - d)` at a cusp of the nephroid.
    pub fn speed(&self, u: f64) -> f64 {
        let p = self.d1(u);
        (p[0] * p[0] + p[1] * p[1]).sqrt()
    }

    /// Radius of curvature, infinite where the curve is locally straight.
    pub fn curvature_radius(&self, u: f64) -> f64 {
        let (p1, p2) = (self.d1(u), self.d2(u));
        let cross = (p1[0] * p2[1] - p1[1] * p2[0]).abs();
        let s = self.speed(u);
        if cross <= 1e-300 { f64::INFINITY } else { s * s * s / cross }
    }

    /// Radius of curvature at a cusp.
    ///
    /// For the nephroid it is `3 a (1 - d)^2 / |3d - 1|` with `a = r / 4`, which
    /// reduces to `3 a (1 - d)^2 / 2` only in the limit `d -> 1`. Using the limit
    /// away from it is wrong by a factor of four at `d = 0.5`.
    pub fn cusp_radius(&self) -> f64 {
        if self.cusps() == 0 {
            return f64::INFINITY;
        }
        self.curvature_radius(PI / self.k())
    }

    /// Tangent-line winding of the EXACT curve, and so the total defect charge
    /// tangential anchoring imposes on an interior that resolves the wall.
    ///
    /// `1` below `d = 1` at every cusp count, since the tangent traces
    /// `w(u) = exp(i u)(1 + d exp(i k u))` whose second factor has real part at
    /// least `1 - d > 0` and therefore contributes no turning. `(k + 2)/2` at
    /// `d = 1`, where `w = 2 cos(k u/2) exp(i (k+2) u/2)` and the `k` sign
    /// changes of the cosine are the cusps, which a line field cannot see.
    ///
    /// A mesh reads this number only when it resolves the tip. See
    /// [`Epitrochoid::aliasing_deficit`].
    pub fn exact_winding(&self) -> f64 {
        if self.cusps() == 0 {
            1.0
        } else if self.d >= 1.0 {
            1.0 + self.k() / 2.0
        } else {
            1.0
        }
    }

    /// Doubled-angle deficit one boundary step can accumulate at a blunted tip.
    ///
    /// The tangent line swings through `2 arcsin(d)` and back across a tip, over
    /// a parameter width that shrinks with `1 - d`. A boundary step spanning the
    /// swing sees a doubled-angle change of `2 (p + m) - 2(|A(m)| + |A(p)|)` for
    /// `A` the single-valued part of `arg w`, and `imposed_charge` wraps that
    /// into a turn. The most negative it can be is `-Lambda` with
    ///
    /// ```text
    ///     Lambda(k, d) = 4 max_{m > 0} [ |A(m)| - m ],
    ///     A(m) = arg( 1 - d exp(i k m) )
    /// ```
    ///
    /// so the wrap books the wrong branch, and the tip contributes a half turn
    /// it does not have, exactly when `Lambda > pi`. Verified against a
    /// brute-force scan over 16 sample counts and 6 phases at 45 `(k, d)` points
    /// in `cgpo-reproduction/symbolic-review/forms/sympy/index_law.py`.
    ///
    /// Infinite at a true cusp, where the swing is a half turn and the cusped
    /// winding is what every sampling correctly reads.
    pub fn aliasing_deficit(&self) -> f64 {
        let k = self.k();
        if self.cusps() == 0 {
            return 0.0;
        }
        if self.d >= 1.0 {
            return f64::INFINITY;
        }
        let f = |m: f64| {
            let (sn, cs) = (k * m).sin_cos();
            (-self.d * sn).atan2(1.0 - self.d * cs).abs() - m
        };
        // `|A| - m` rises to a single interior maximum and falls, so a coarse
        // scan brackets it and a ternary search finishes it.
        let top = PI / k;
        let n = 4096;
        let (mut arg, mut best) = (0.0, 0.0);
        for i in 0..=n {
            let m = top * i as f64 / n as f64;
            let v = f(m);
            if v > best {
                best = v;
                arg = m;
            }
        }
        let step = top / n as f64;
        let (mut lo, mut hi) = ((arg - step).max(0.0), (arg + step).min(top));
        for _ in 0..80 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if f(a) < f(b) { lo = a } else { hi = b }
        }
        4.0 * f(0.5 * (lo + hi)).max(best)
    }

    /// Whether every boundary sampling reads the same winding, however coarse.
    ///
    /// True below the threshold `d_c(k)` where the deficit reaches a half turn.
    /// Above it a boundary that steps across a tip in one go reads
    /// `(k + 2)/2` instead of `1`, which is what every `d = 0.99` production run
    /// does.
    pub fn winding_is_sampling_independent(&self) -> bool {
        self.aliasing_deficit() <= PI
    }

    /// Largest `d` at which no boundary sampling can misread the winding.
    ///
    /// `1/sqrt(2) <= d_c(k) <= sin(pi (k+2)/(4(k+1)))`, falling with the lobe
    /// count towards the lower bound: 0.896316, 0.847487, 0.818864, 0.800000
    /// and 0.786612 for `k = 1..5`.
    pub fn alias_threshold(cusps: usize) -> f64 {
        if cusps == 0 {
            return 1.0;
        }
        let at = |d: f64| Epitrochoid { q: 1.0 + cusps as f64 / 2.0, d, r: 1.0 }.aliasing_deficit();
        let (mut lo, mut hi) = (0.5, 1.0 - 1e-9);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if at(mid) > PI { hi = mid } else { lo = mid }
        }
        0.5 * (lo + hi)
    }

    /// Parameter values of the cusps, the minima of the radius.
    pub fn cusp_params(&self) -> Vec<f64> {
        let k = self.k();
        if k <= 0.0 {
            return vec![];
        }
        (0..self.cusps())
            .map(|j| PI / k + 2.0 * PI * j as f64 / k)
            .collect()
    }

    /// Unit tangent, taken from a neighbouring parameter where the speed vanishes.
    pub fn tangent(&self, u: f64) -> [f64; 2] {
        let p = self.d1(u);
        let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if n > 1e-12 {
            return [p[0] / n, p[1] / n];
        }
        let eps = 1e-6;
        let p = self.d1(u + eps);
        let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
        [p[0] / n, p[1] / n]
    }

    /// Inward unit normal, oriented by testing a short step against the interior.
    pub fn inward_normal(&self, u: f64) -> [f64; 2] {
        let t = self.tangent(u);
        let n = [-t[1], t[0]];
        let p = self.point(u);
        // The curve is traced anticlockwise, so the left normal points inward;
        // rather than rely on that, take the candidate that moves towards the
        // centroid of the whole curve, which is the origin by symmetry.
        let towards = -(p[0] * n[0] + p[1] * n[1]);
        if towards >= 0.0 { n } else { [-n[0], -n[1]] }
    }

    /// Parametric half-width at which the two branches leaving a cusp are
    /// `edge` apart ALONG THE CURVE, so the two boundary edges meeting at the
    /// cusp vertex have that length.
    ///
    /// This is the sharp treatment, and it is the one the index law needs. A
    /// re-entrant cusp has interior angle `2 pi`, so it contributes `-pi` to the
    /// boundary's total turning and the tangent's turning number is `k/2 + 1`
    /// rather than 1: the nephroid imposes 2, not 1, and each cusp then holds a
    /// `-1/2` surface defect, leaving the interior at `4 x (1/2) - 2 x (1/2) = +1`.
    /// That is `(k+2, k)`. Rounding the cusp, by a chord or a fillet or a `d < 1`
    /// curve, drops the turning number to 1 and takes the `k` negative defects
    /// with it, after which `(2, 0)` satisfies the topology perfectly well and
    /// nothing pins at the wall.
    ///
    /// The obstruction is not the corner, which a polygon vertex represents
    /// exactly at zero cost. It is the EXTERIOR spike between the branches: they
    /// separate as `8 a eps^3` while running `3 a eps^2` along, so one element
    /// from the tip the walls are half an element apart and no mesh of that size
    /// fits between them. Sampling down to a one-element separation and closing
    /// on the exact cusp point leaves the corner intact and no sub-element
    /// feature behind it, which is what the reference lattice's mask does by
    /// simply not resolving the spike below one cell.
    ///
    /// Zero when the curve has no cusp, and for `edge <= 0`.
    pub fn cusp_edge_param(&self, edge: f64) -> f64 {
        // `!(edge > 0.0)` catches NaN as well as non-positive, which is the
        // point: an unset or malformed edge length must fall through here.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if self.cusps() == 0 || !(edge > 0.0) {
            return 0.0;
        }
        let uc = PI / self.k();
        let pc = self.point(uc);
        let along = |e: f64| {
            let p = self.point(uc + e);
            ((p[0] - pc[0]).powi(2) + (p[1] - pc[1]).powi(2)).sqrt()
        };
        let (mut lo, mut hi) = (0.0_f64, 0.9 * PI / self.k());
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

/// The epitrochoid as a boundary geometry.
///
/// Every geometric quantity is the analytic one already written above; the
/// trait adds the cusp parameters as the features to grade towards, and the
/// statement that the arcs between them are congruent, which is what lets the
/// boundary be sampled symmetrically.
impl PlaneCurve for Epitrochoid {
    fn point(&self, u: f64) -> [f64; 2] {
        Epitrochoid::point(self, u)
    }

    fn d1(&self, u: f64) -> [f64; 2] {
        Epitrochoid::d1(self, u)
    }

    fn d2(&self, u: f64) -> [f64; 2] {
        Epitrochoid::d2(self, u)
    }

    fn features(&self) -> Vec<f64> {
        self.cusp_params()
    }

    fn feature_symmetric(&self) -> bool {
        self.cusps() > 0
    }
}

/// How to build the mesh.
#[derive(Debug, Clone, Copy)]
pub struct MeshOpts {
    /// Target element size away from any cusp.
    pub h_bulk: f64,
    /// Smallest element size, used at the tip. Set from the cusp radius.
    pub h_min: f64,
    /// Geometric growth of element size per layer, marching inward.
    pub grade: f64,
    /// Boundary arc as a fraction of the local radius of curvature.
    pub boundary_frac: f64,
    /// Passes of area-weighted smoothing over interior vertices.
    pub smooth_passes: usize,
    /// Seed for the dart throwing, so a mesh is reproducible.
    pub seed: u64,
    /// Length of the two boundary edges meeting at a cusp vertex, or zero to
    /// sample the cusp naively.
    ///
    /// The sharp treatment: both branches are sampled down to this separation
    /// along the curve and closed on the exact cusp point, so the corner is a
    /// polygon vertex at radius zero and nothing finer than an element is left
    /// behind it. See `Epitrochoid::cusp_edge_param`.
    pub cusp_edge: f64,
}

impl Default for MeshOpts {
    fn default() -> Self {
        Self {
            h_bulk: 1.0,
            h_min: 0.05,
            grade: 1.3,
            boundary_frac: 0.25,
            smooth_passes: 8,
            seed: 0,
            cusp_edge: 0.0,
        }
    }
}

impl MeshOpts {
    /// Bulk size from a lattice side, so element count is comparable with a run
    /// at that resolution, and `h_min` from the curve's own cusp radius.
    pub fn matching_lattice(curve: &Epitrochoid, lx: usize, per_cusp: f64) -> Self {
        let h_bulk = 2.0 * curve.r / (curve.k() + 2.0) / (lx as f64 / 2.0);
        let rc = curve.cusp_radius();
        let h_min = if rc.is_finite() {
            (rc / per_cusp).min(h_bulk).max(h_bulk * 1e-4)
        } else {
            h_bulk
        };
        Self { h_bulk, h_min, ..Default::default() }
    }
}

/// Measured quality of a built mesh.
#[derive(Debug, Clone)]
pub struct MeshQuality {
    pub vertices: usize,
    pub triangles: usize,
    pub boundary_vertices: usize,
    /// Smallest angle over all triangles, in degrees. Above 25 is workable.
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    /// Triangles with an angle past a right angle, which give a negative
    /// cotangent weight on the opposite edge.
    pub obtuse: usize,
    /// The most negative cotangent weight in the Laplacian, in units of the
    /// weight's own scale; zero means every weight is non-negative.
    pub worst_cot_weight: f64,
    pub min_edge: f64,
    pub max_edge: f64,
    pub min_area: f64,
}

/// A built confined mesh with its boundary tagged.
#[derive(Clone)]
pub struct ConfinedMesh2 {
    pub mesh: FlatMesh,
    /// Boundary vertices, in order along the curve.
    pub boundary_vertices: Vec<usize>,
    /// Curve parameter at each boundary vertex, in the same order.
    pub boundary_params: Vec<f64>,
    /// Inward unit normal at each boundary vertex.
    pub boundary_normals: Vec<[f64; 2]>,
    /// The wall this mesh conforms to, kept so the anchoring and the diagnostics
    /// can ask it for a tangent at a boundary parameter.
    pub curve: Arc<dyn PlaneCurve>,
    pub quality: MeshQuality,
}

/// Local target size: the bulk size, reduced towards `h_min` near a cusp.
///
/// The reduction is geometric in the distance to the nearest cusp, which is what
/// makes the element count grow with the logarithm of the grading ratio rather
/// than with its square.
fn target_size(p: [f64; 2], cusps: &[[f64; 2]], o: &MeshOpts) -> f64 {
    // A cusp gentler than the bulk size needs no refinement, so the floor is the
    // smaller of the two: clamping to a floor above the ceiling would panic, and a
    // caller setting h_min from the cusp radius hits that whenever the cusp is
    // already resolved.
    let floor = o.h_min.min(o.h_bulk);
    let mut h = o.h_bulk;
    for c in cusps {
        let dist = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
        h = h.min(floor + (o.grade - 1.0) * dist);
    }
    h.clamp(floor, o.h_bulk)
}

/// Sample the boundary so no arc exceeds `boundary_frac` of the local curvature
/// radius, nor the local target size.
/// Parameter offsets from a cusp out to the next tip, sampled once.
///
/// The epitrochoid is symmetric about every cusp AND every tip, so one such arc
/// generates the whole boundary by reflection and rotation. Building it that way
/// rather than walking once round makes the sampling exactly symmetric: walking
/// round leaves the stride wherever the previous step ended, and beside a cusp
/// that gave edges of 0.60 approaching against 1.24 leaving, a factor of two at
/// the one feature whose symmetry holds a defect in place.
fn cusp_to_tip_offsets<C: PlaneCurve + ?Sized>(
    curve: &C,
    o: &MeshOpts,
    trunc: f64,
) -> Vec<f64> {
    let feats = curve.features();
    let half = 0.5 * curve.period() / feats.len() as f64;
    let uc = feats[0];
    let cusps: Vec<[f64; 2]> = feats.iter().map(|&u| curve.point(u)).collect();
    // The cusp itself, then the vertex one edge along it, then the ordinary
    // stride out to the tip.
    let mut offs = vec![0.0_f64];
    let mut t = trunc;
    if t <= 0.0 || t >= half {
        t = 0.0;
    } else {
        offs.push(t);
    }
    let floor = o.h_min.min(o.h_bulk);
    while t < half {
        let u = uc + t;
        let p = curve.point(u);
        let rc = curve.curvature_radius(u);
        let h_geom = if rc.is_finite() { o.boundary_frac * rc } else { o.h_bulk };
        let ds = h_geom.min(target_size(p, &cusps, o)).max(floor);
        let speed = curve.speed(u).max(1e-12);
        let mut dt = (ds / speed).min(curve.period() / 64.0);
        // Share the run to the tip out evenly, so the last edge before it is not
        // a stub. The tip is a symmetry point, so a stub there is mirrored into a
        // pair of stubs.
        let run = half - t;
        if run > 1e-12 && run < 2.5 * dt {
            let n = (run / dt).round().max(1.0);
            dt = dt.min(run / n);
        }
        t += dt;
        if t < half - 1e-12 {
            offs.push(t);
        }
    }
    offs
}

/// Sample the boundary so no arc exceeds `boundary_frac` of the local curvature
/// radius, nor the local target size.
///
/// Two constructions. With `cusp_edge` set the curve has true cusps and the
/// boundary is built from one cusp-to-tip arc, reflected and rotated, which is
/// exactly symmetric. Otherwise it walks once round, which is what every smooth
/// `d < 1` mesh was calibrated against.
fn sample_boundary<C: PlaneCurve + ?Sized>(
    curve: &C,
    o: &MeshOpts,
) -> (Vec<[f64; 2]>, Vec<f64>) {
    let feats = curve.features();
    let sharp = if curve.feature_symmetric() && !feats.is_empty() {
        curve.feature_edge_param(o.cusp_edge)
    } else {
        0.0
    };
    if sharp > 0.0 {
        let half = 0.5 * curve.period() / feats.len() as f64;
        let offs = cusp_to_tip_offsets(curve, o, sharp);
        let mut params = Vec::with_capacity(offs.len() * 2 * feats.len());
        for &uc in &feats {
            // Cusp out to the tip.
            for &t in &offs {
                params.push(uc + t);
            }
            // Tip back down to the next cusp, the same offsets mirrored.
            for &t in offs.iter().skip(1).rev() {
                params.push(uc + 2.0 * half - t);
            }
        }
        let pts = params.iter().map(|&u| curve.point(u)).collect();
        return (pts, params);
    }
    sample_boundary_walk(curve, o)
}

fn sample_boundary_walk<C: PlaneCurve + ?Sized>(
    curve: &C,
    o: &MeshOpts,
) -> (Vec<[f64; 2]>, Vec<f64>) {
    let cusps: Vec<[f64; 2]> = curve.features().iter().map(|&u| curve.point(u)).collect();
    let period = curve.period();
    let mut pts = Vec::new();
    let mut params = Vec::new();
    let mut u = 0.0_f64;
    // A hard cap on the sample count, so a pathological d cannot spin forever.
    let max_pts = 2_000_000usize;
    while u < period && pts.len() < max_pts {
        let p = curve.point(u);
        pts.push(p);
        params.push(u);
        let rc = curve.curvature_radius(u);
        let h_geom = if rc.is_finite() { o.boundary_frac * rc } else { o.h_bulk };
        let ds = h_geom
            .min(target_size(p, &cusps, o))
            .max(o.h_min.min(o.h_bulk) * 0.25);
        let speed = curve.speed(u).max(1e-12);
        // Step in the parameter that realises that arc, capped so a vanishing
        // speed at a cusp cannot produce an unbounded parameter jump.
        u += (ds / speed).min(period / 64.0);
    }
    // Drop a final sample that has wrapped onto the first.
    while pts.len() > 3 {
        let a = pts[pts.len() - 1];
        let b = pts[0];
        if ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() < 0.25 * o.h_min {
            pts.pop();
            params.pop();
        } else {
            break;
        }
    }
    (pts, params)
}

/// A hash grid over cells of a fixed size, for nearest-point rejection.
struct HashGrid {
    cell: f64,
    map: HashMap<(i64, i64), Vec<usize>>,
}

impl HashGrid {
    fn new(cell: f64) -> Self {
        Self { cell: cell.max(1e-12), map: HashMap::new() }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        ((p[0] / self.cell).floor() as i64, (p[1] / self.cell).floor() as i64)
    }

    fn insert(&mut self, p: [f64; 2], i: usize) {
        self.map.entry(self.key(p)).or_default().push(i);
    }

    /// Whether any stored point lies within `r` of `p`.
    fn any_within(&self, pts: &[[f64; 2]], p: [f64; 2], r: f64) -> bool {
        let reach = (r / self.cell).ceil() as i64;
        let (kx, ky) = self.key(p);
        let r2 = r * r;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                if let Some(v) = self.map.get(&(kx + dx, ky + dy)) {
                    for &i in v {
                        let q = pts[i];
                        if (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < r2 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// Whether `p` is inside the boundary polygon, by ray casting.
///
/// The ray is cast along `+x` and a crossing is counted when the edge straddles
/// the ray's height, with the half-open convention on the endpoints so a vertex
/// lying exactly on the ray is counted once rather than twice.
fn inside(bpts: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = bpts.len();
    let mut c = false;
    for i in 0..n {
        let (a, b) = (bpts[i], bpts[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                c = !c;
            }
        }
    }
    c
}

/// Circumcentre and squared circumradius of a triangle, or `None` when it is
/// degenerate.
fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-300 {
        return None;
    }
    let (aa, bb, cc) = (
        a[0] * a[0] + a[1] * a[1],
        b[0] * b[0] + b[1] * b[1],
        c[0] * c[0] + c[1] * c[1],
    );
    let ux = (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d;
    let uy = (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d;
    let r2 = (a[0] - ux).powi(2) + (a[1] - uy).powi(2);
    Some(([ux, uy], r2))
}

/// Delaunay triangulation of `pts` restricted to the domain `bpts` bounds.
///
/// Bowyer-Watson against a super-triangle, then every triangle whose centroid
/// falls outside the boundary polygon is discarded. That is what keeps the mesh
/// off the exterior spike between the branches of a cusp, which is the whole
/// difficulty of a re-entrant corner: the triangulation of the point set covers
/// its convex hull, and only the domain test knows the notch is not domain.
///
/// Points are inserted in Hilbert-ish sorted order, by rows of a coarse grid
/// with alternating direction, so the walk that finds the bad triangles stays
/// local and the cost does not degenerate to quadratic.
fn triangulate(pts: &[[f64; 2]], bpts: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
    let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    // Super-triangle, comfortably enclosing every point.
    let s = 20.0 * span;
    let mut v: Vec<[f64; 2]> = pts.to_vec();
    v.push([mid[0] - s, mid[1] - s]);
    v.push([mid[0] + s, mid[1] - s]);
    v.push([mid[0], mid[1] + s]);
    let (s0, s1, s2) = (n, n + 1, n + 2);

    let mut tris: Vec<[usize; 3]> = vec![[s0, s1, s2]];
    let mut circ: Vec<([f64; 2], f64)> = vec![circumcircle(v[s0], v[s1], v[s2]).unwrap()];

    // Insertion order: banded, alternating direction, so successive points are
    // near each other.
    let band = (span / (n as f64).sqrt().max(1.0) / 1.5).max(1e-9);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let (bi, bj) = (
            ((pts[i][1] - lo[1]) / band) as i64,
            ((pts[j][1] - lo[1]) / band) as i64,
        );
        bi.cmp(&bj).then_with(|| {
            let (xi, xj) = (pts[i][0], pts[j][0]);
            if bi % 2 == 0 {
                xi.partial_cmp(&xj).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                xj.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
    });

    let mut bad: Vec<usize> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for &ip in &order {
        let p = v[ip];
        bad.clear();
        for (t, &(c, r2)) in circ.iter().enumerate() {
            if (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) < r2 {
                bad.push(t);
            }
        }
        if bad.is_empty() {
            continue;
        }
        // Boundary of the cavity: edges held by exactly one bad triangle.
        edges.clear();
        for &t in &bad {
            let tr = tris[t];
            for k in 0..3 {
                let (a, b) = (tr[k], tr[(k + 1) % 3]);
                let e = if a < b { (a, b) } else { (b, a) };
                if let Some(pos) = edges.iter().position(|&x| x == e) {
                    edges.swap_remove(pos);
                } else {
                    edges.push(e);
                }
            }
        }
        // Remove the bad triangles, high index first so the swaps stay valid.
        bad.sort_unstable_by(|a, b| b.cmp(a));
        for &t in &bad {
            tris.swap_remove(t);
            circ.swap_remove(t);
        }
        for &(a, b) in &edges {
            if let Some(cc) = circumcircle(v[a], v[b], p) {
                tris.push([a, b, ip]);
                circ.push(cc);
            }
        }
    }

    // Drop anything touching the super-triangle, then anything outside the
    // domain. The centroid test is what excludes the exterior notch at a cusp.
    tris.retain(|t| t.iter().all(|&i| i < n));
    tris.retain(|t| {
        let c = [
            (pts[t[0]][0] + pts[t[1]][0] + pts[t[2]][0]) / 3.0,
            (pts[t[0]][1] + pts[t[1]][1] + pts[t[2]][1]) / 3.0,
        ];
        inside(bpts, c)
    });
    // Consistent orientation, so the DEC operators see a coherent surface.
    for t in tris.iter_mut() {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        let area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if area2 < 0.0 {
            t.swap(1, 2);
        }
    }
    tris
}

/// Measured quality of a triangulation.
///
/// `worst_cot_weight` is the most negative `cot(theta) / 2` over every corner,
/// which is the contribution a single triangle makes to the cotangent weight of
/// the edge opposite it. Zero means no corner is obtuse and every weight in the
/// Laplacian is non-negative, which is the condition for the discrete maximum
/// principle.
fn quality(pts: &[[f64; 2]], tris: &[[usize; 3]], n_b: usize) -> MeshQuality {
    let mut min_angle = f64::MAX;
    let mut max_angle = f64::MIN;
    let mut obtuse = 0usize;
    let mut worst_cot = 0.0_f64;
    let mut min_edge = f64::MAX;
    let mut max_edge = f64::MIN;
    let mut min_area = f64::MAX;
    for t in tris {
        let p = [pts[t[0]], pts[t[1]], pts[t[2]]];
        let l = [
            ((p[1][0] - p[2][0]).powi(2) + (p[1][1] - p[2][1]).powi(2)).sqrt(),
            ((p[2][0] - p[0][0]).powi(2) + (p[2][1] - p[0][1]).powi(2)).sqrt(),
            ((p[0][0] - p[1][0]).powi(2) + (p[0][1] - p[1][1]).powi(2)).sqrt(),
        ];
        for &e in &l {
            min_edge = min_edge.min(e);
            max_edge = max_edge.max(e);
        }
        let area = 0.5
            * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                .abs();
        min_area = min_area.min(area);
        let mut any_obtuse = false;
        for k in 0..3 {
            let (a, b, c) = (l[k], l[(k + 1) % 3], l[(k + 2) % 3]);
            if b <= 0.0 || c <= 0.0 {
                continue;
            }
            let cos = ((b * b + c * c - a * a) / (2.0 * b * c)).clamp(-1.0, 1.0);
            let ang = cos.acos();
            min_angle = min_angle.min(ang);
            max_angle = max_angle.max(ang);
            if ang > std::f64::consts::FRAC_PI_2 {
                any_obtuse = true;
                let sin = ang.sin().max(1e-300);
                worst_cot = worst_cot.min(0.5 * cos / sin);
            }
        }
        if any_obtuse {
            obtuse += 1;
        }
    }
    if tris.is_empty() {
        min_angle = 0.0;
        max_angle = 0.0;
        min_edge = 0.0;
        max_edge = 0.0;
        min_area = 0.0;
    }
    MeshQuality {
        vertices: pts.len(),
        triangles: tris.len(),
        boundary_vertices: n_b,
        min_angle_deg: min_angle.to_degrees(),
        max_angle_deg: max_angle.to_degrees(),
        obtuse,
        worst_cot_weight: worst_cot,
        min_edge,
        max_edge,
        min_area,
    }
}

/// Build a boundary-conforming graded mesh of the interior of any
/// [`PlaneCurve`].
///
/// Three stages. The boundary is sampled first and its points are the mesh's
/// first `n_b` vertices, in order along the curve, which is what lets the
/// anchoring index them directly. Layers then march inward from the wall, each
/// offset by its own local size so the elements stay near-isotropic rather than
/// stretching as the grading tightens. The remainder is filled by dart throwing
/// against a hash grid, and the whole point set is triangulated, smoothed and
/// re-triangulated.
///
/// The domain test is what handles a re-entrant cusp: a Delaunay triangulation
/// covers the convex hull of the point set, so the exterior spike between the
/// two branches would be filled in unless the triangles there are discarded.
pub fn confined_mesh<C: PlaneCurve + 'static>(curve: C, o: MeshOpts) -> ConfinedMesh2 {
    let curve: Arc<dyn PlaneCurve> = Arc::new(curve);
    let (bpts, bparams) = sample_boundary(curve.as_ref(), &o);
    let n_b = bpts.len();
    let cusps: Vec<[f64; 2]> = curve.features().iter().map(|&u| curve.point(u)).collect();

    let mut pts = bpts.clone();
    let mut grid = HashGrid::new(o.h_bulk);
    for (i, p) in pts.iter().enumerate() {
        grid.insert(*p, i);
    }

    // Layers marching inward. At each layer the along-wall stride grows with the
    // layer's own size, so the elements stay near-isotropic instead of becoming
    // long and thin as the layer shortens.
    let normals: Vec<[f64; 2]> = bparams.iter().map(|&u| curve.inward_normal(u)).collect();
    let mut depth = vec![0.0_f64; n_b];
    for _layer in 0..64 {
        let mut added = 0usize;
        let mut k = 0usize;
        while k < n_b {
            let h = target_size(bpts[k], &cusps, &o);
            depth[k] += h;
            let cand = [
                bpts[k][0] + normals[k][0] * depth[k],
                bpts[k][1] + normals[k][1] * depth[k],
            ];
            let hc = target_size(cand, &cusps, &o);
            if inside(&bpts, cand) && !grid.any_within(&pts, cand, 0.85 * hc) {
                grid.insert(cand, pts.len());
                pts.push(cand);
                added += 1;
            }
            // Stride along the wall in proportion to the local size, so a layer
            // does not oversample where the boundary is finely sampled.
            let stride = (h / (o.h_bulk * 0.35)).ceil() as usize;
            k += stride.max(1);
        }
        if added == 0 {
            break;
        }
    }

    // Dart throwing for whatever the layers did not reach.
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in &bpts {
        for c in 0..2 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let mut rng = StdRng::seed_from_u64(o.seed);
    let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    let target = (area / (0.7 * o.h_bulk * o.h_bulk)) as usize;
    let mut misses = 0usize;
    while misses < 12 * target.max(1) {
        let p = [
            lo[0] + rng.random::<f64>() * (hi[0] - lo[0]),
            lo[1] + rng.random::<f64>() * (hi[1] - lo[1]),
        ];
        let h = target_size(p, &cusps, &o);
        if !inside(&bpts, p) || grid.any_within(&pts, p, 0.85 * h) {
            misses += 1;
            continue;
        }
        grid.insert(p, pts.len());
        pts.push(p);
        misses = 0;
    }

    let mut tris = triangulate(&pts, &bpts);

    // Area-weighted smoothing of the interior vertices. The boundary never
    // moves: its vertices are on the curve exactly and that is the point.
    for _ in 0..o.smooth_passes {
        let n = pts.len();
        let mut sum = vec![[0.0_f64; 2]; n];
        let mut wsum = vec![0.0_f64; n];
        for t in &tris {
            let p = [pts[t[0]], pts[t[1]], pts[t[2]]];
            let w = 0.5
                * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                    .abs();
            let c = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            for &i in t {
                sum[i][0] += w * c[0];
                sum[i][1] += w * c[1];
                wsum[i] += w;
            }
        }
        for i in n_b..n {
            if wsum[i] <= 0.0 {
                continue;
            }
            let target_p = [sum[i][0] / wsum[i], sum[i][1] / wsum[i]];
            // Damped, and rejected if it would leave the domain.
            let cand = [
                pts[i][0] + 0.6 * (target_p[0] - pts[i][0]),
                pts[i][1] + 0.6 * (target_p[1] - pts[i][1]),
            ];
            if inside(&bpts, cand) {
                pts[i] = cand;
            }
        }
        tris = triangulate(&pts, &bpts);
    }

    let q = quality(&pts, &tris, n_b);
    // The sharp cusp needs no special case here: its vertex IS on the curve, and
    // the director is well defined there by continuity even though the tangent
    // vector is not, because the limits from either side differ by a half turn
    // and a director is headless.
    let boundary_normals: Vec<[f64; 2]> =
        bparams.iter().map(|&u| curve.inward_normal(u)).collect();
    let mesh = FlatMesh::from_triangles(pts, tris);

    ConfinedMesh2 {
        mesh,
        boundary_vertices: (0..n_b).collect(),
        boundary_params: bparams,
        boundary_normals,
        curve,
        quality: q,
    }
}

impl ConfinedMesh2 {
    /// Total defect charge the anchoring imposes, read off the mesh boundary.
    ///
    /// The anchoring sets the director to the vector at angle `q_anchor * theta_n`
    /// turned by a quarter turn, so the charge inside is the winding of that
    /// direction as a line field: sum the increments of the doubled angle, wrapped
    /// into a half turn, and divide by `4 pi`. A boundary that resolves the wall
    /// returns `q_anchor` exactly. The lattice does not: at `q_anchor = 2`,
    /// `d = 0.99` and `L = 100` it imposes `+4`.
    ///
    /// A TRUE cusp is not a failure to resolve. Its interior angle is `2 pi`, so
    /// it contributes `-pi` to the boundary's turning and the turning number is
    /// `k/2 + 1`: the nephroid imposes 2 where the same curve rounded imposes 1.
    ///
    /// The interior then holds that whole number. An earlier revision of this
    /// comment had the `k` cusps holding a `-1/2` surface defect each, bringing
    /// the interior back to `+1`; the runs say otherwise. Final-frame census of
    /// `defects.tsv`, in half-charge units, against each run's own recorded
    /// `imposed_charge`:
    ///
    /// ```text
    ///   nephroid   d = 0.72   imposes 1.0   complement (4, 2)   interior +1.0
    ///   nephroid   d = 1.00   imposes 2.0   complement (4, 0)   interior +2.0
    ///   cardioid   d = 1.00   imposes 1.5   complement (3, 0)   interior +1.5
    ///   trefoiloid d = 1.00   imposes 2.5   complement (6, 1)   interior +2.5
    /// ```
    ///
    /// so sharpening the nephroid REMOVES its two `-1/2` cores rather than
    /// binding them to the wall, and the interior total equals the imposed
    /// charge on every run measured.
    pub fn imposed_charge(&self, q_anchor: f64) -> (f64, f64, usize) {
        let n = self.boundary_params.len();
        let mut sum = 0.0;
        let mut worst = 0.0_f64;
        let mut big = 0usize;
        let wrap = |x: f64| x - 2.0 * PI * (x / (2.0 * PI)).round();
        for i in 0..n {
            let a = self.angle_at(i, q_anchor);
            let b = self.angle_at((i + 1) % n, q_anchor);
            let step = wrap(2.0 * (b - a));
            sum += step;
            if step.abs() > worst {
                worst = step.abs();
            }
            if step.abs() > PI / 2.0 {
                big += 1;
            }
        }
        (sum / (4.0 * PI), worst.to_degrees(), big)
    }

    fn angle_at(&self, i: usize, q_anchor: f64) -> f64 {
        let n = self.boundary_normals[i];
        // The reference's own convention: the outward normal's angle, times the
        // anchoring winding. The quarter turn is a constant and drops out of every
        // increment.
        let theta = (-n[1]).atan2(-n[0]);
        q_anchor * theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neph(d: f64) -> Epitrochoid {
        Epitrochoid { q: 2.0, d, r: 98.0 }
    }

    /// Curve for the lattice-matched nephroid: the paper's own epicycloid at the
    /// size that carries its effective system length.
    fn latmatch() -> Epitrochoid {
        Epitrochoid { q: 2.0, d: 1.0, r: 49.778_694_002 }
    }

    /// Curve for the lattice-matched cardioid, the golden braid's geometry.
    ///
    /// Fig 7's divisor is per shape. For the nephroid it is `0.764031 Lx`, and
    /// for the cardioid `0.71962 Lx`, recovered from the published
    /// `(l_a, l_c) = (0.0139, 0.0903)` at `als = 2`, `ncl = 13`, `Lx = 200`.
    /// With `sqrt(A) / r = sqrt(2 pi / 3)` at `d = 1` that puts `r` at 49.725,
    /// so both shapes sit inscribed in a 100 lattice.
    fn cardmatch() -> Epitrochoid {
        Epitrochoid { q: 1.5, d: 1.0, r: 49.725 }
    }

    fn sharp_opts() -> MeshOpts {
        MeshOpts { h_bulk: 1.0, h_min: 1.0, cusp_edge: 1.0, ..Default::default() }
    }

    #[test]
    fn cusp_radius_matches_the_closed_form() {
        // rho = speed^3 / |x' y'' - y' x''| worked out at u = pi/2 for k = 2:
        // speed = 3 a (1 - d), cross = 9 a^2 (1 - d) |3d - 1|, so
        // rho = 3 a (1 - d)^2 / |3d - 1| with a = r / 4.
        for d in [0.2, 0.5, 0.9, 0.99] {
            let c = neph(d);
            let a = c.r / 4.0;
            let want = 3.0 * a * (1.0 - d).powi(2) / (3.0 * d - 1.0).abs();
            let got = c.cusp_radius();
            assert!(
                (got - want).abs() < 1e-8 * want.max(1.0),
                "d = {d}: cusp radius {got}, closed form {want}"
            );
        }
    }

    #[test]
    fn inward_normal_points_into_the_domain() {
        for d in [0.5, 0.9, 1.0] {
            let c = Epitrochoid { q: 2.0, d, r: 98.0 };
            for j in 0..64 {
                let u = 2.0 * PI * j as f64 / 64.0;
                // Skip the immediate neighbourhood of a cusp at d = 1, where the
                // derivative vanishes and there is no tangent to take a normal
                // from. That is the whole reason the sharp treatment exists.
                if d >= 1.0 && c.cusp_params().iter().any(|&uc| (u - uc).abs() < 0.05) {
                    continue;
                }
                let p = c.point(u);
                let n = c.inward_normal(u);
                let step = [p[0] + 1e-3 * n[0], p[1] + 1e-3 * n[1]];
                // Inside means nearer the centre than the wall along that ray,
                // which for these shapes is simply a smaller radius.
                assert!(
                    step[0].hypot(step[1]) < p[0].hypot(p[1]) + 1e-9,
                    "d = {d}, u = {u}: normal points outward"
                );
            }
        }
    }

    #[test]
    fn boundary_vertices_sit_on_the_curve() {
        // The point of a conforming mesh. Every boundary vertex carries the
        // parameter it came from, and has to be that point of the curve.
        let c = neph(0.9);
        let m = confined_mesh(
            c,
            MeshOpts { h_bulk: 4.0, h_min: c.cusp_radius() / 4.0, ..Default::default() },
        );
        for (k, &vi) in m.boundary_vertices.iter().enumerate() {
            let v = m.mesh.vertex(vi);
            let p = c.point(m.boundary_params[k]);
            let d = (v[0] - p[0]).hypot(v[1] - p[1]);
            assert!(d < 1e-9, "boundary vertex {k} is {d} off the curve");
        }
    }

    #[test]
    fn imposed_charge_equals_the_anchoring_winding() {
        // The whole reason for a conforming boundary: the charge the anchoring
        // imposes has to be the number it was asked for, at every d and for both
        // anchoring windings, where the lattice gives +4 instead of +2.
        //
        // Every d here is below 1, so the curve is smooth and its turning number
        // is 1. A TRUE cusp is a different case and is tested separately.
        for d in [0.5, 0.9, 0.99] {
            let c = neph(d);
            let m = confined_mesh(
                c,
                MeshOpts { h_bulk: 4.0, h_min: c.cusp_radius() / 4.0, ..Default::default() },
            );
            for q_anchor in [1.0, 2.0] {
                let (charge, worst, big) = m.imposed_charge(q_anchor);
                assert!(
                    (charge.abs() - q_anchor).abs() < 1e-6,
                    "d = {d}, q = {q_anchor}: imposed {charge}, worst step {worst} deg, {big} over a quarter turn"
                );
            }
        }
    }

    #[test]
    fn a_true_cusp_imposes_the_turning_number_its_corners_give_it() {
        // The correction that cost an evening. A re-entrant cusp has interior
        // angle `2 pi`, so it contributes `-pi` to the boundary's total turning:
        // the smooth part must turn by `2 pi + k pi` and the tangent's turning
        // number is `k/2 + 1`, which is 2 for the nephroid, NOT 1.
        //
        // That is not a defect to be repaired. Each cusp then holds a `-1/2`
        // surface defect, and the interior is left at `4 x (1/2) - 2 x (1/2) = +1`,
        // which is the index law `(k+2, k)`. Rounding the cusp drops the turning
        // number to 1 and takes the k negative defects with it, after which
        // `(2, 0)` satisfies the topology and nothing pins at the wall.
        let c = latmatch();
        let (charge, worst, big) = confined_mesh(c, sharp_opts()).imposed_charge(1.0);
        assert!(
            (charge - 2.0).abs() < 1e-6,
            "sharp d = 1 nephroid should impose 2, got {charge}, worst step {worst} deg"
        );
        assert_eq!(big, 0, "worst step {worst} deg crossed a quarter turn");

        // One cusp, and the law is the same: `k / 2 + 1` is 3/2 for the
        // cardioid. That is three mobile `+1/2` cores against the nephroid's
        // four, which is the golden braid's three strands against the silver's
        // four, and it is why the paper pairs each shape with the braid it does.
        let (charge, worst, big) = confined_mesh(cardmatch(), sharp_opts()).imposed_charge(1.0);
        assert!(
            (charge - 1.5).abs() < 1e-6,
            "sharp d = 1 cardioid should impose 3/2, got {charge}, worst step {worst} deg"
        );
        assert_eq!(big, 0, "worst step {worst} deg crossed a quarter turn");

        // The discriminator: a ROUNDED cusp is smooth, its turning number is 1,
        // and it therefore requires no negative defects at all. If this test
        // passed for both it would be testing nothing.
        let smooth = neph(0.99);
        let m = confined_mesh(
            smooth,
            MeshOpts { h_bulk: 4.0, h_min: smooth.cusp_radius() / 4.0, ..Default::default() },
        );
        let (charge, _, _) = m.imposed_charge(1.0);
        assert!(
            (charge.abs() - 1.0).abs() < 1e-6,
            "a rounded cusp should impose 1, got {charge}"
        );

        // And the cardioid rounds the same way, so neither shape gets its
        // winding from anything except the corner.
        let smooth = Epitrochoid { q: 1.5, d: 0.9, r: 98.0 };
        let m = confined_mesh(
            smooth,
            MeshOpts { h_bulk: 4.0, h_min: smooth.cusp_radius() / 4.0, ..Default::default() },
        );
        let (charge, _, _) = m.imposed_charge(1.0);
        assert!(
            (charge.abs() - 1.0).abs() < 1e-6,
            "a rounded cardioid cusp should impose 1, got {charge}"
        );
    }

    #[test]
    fn the_sharp_cusp_puts_a_vertex_on_the_cusp_and_keeps_its_edges_an_element_long() {
        let c = latmatch();
        let o = sharp_opts();
        let m = confined_mesh(c, o);
        let pts: Vec<[f64; 2]> = m
            .boundary_vertices
            .iter()
            .map(|&i| {
                let v = m.mesh.vertex(i);
                [v[0], v[1]]
            })
            .collect();
        let n = pts.len();
        for &uc in c.cusp_params().iter() {
            let cusp = c.point(uc);
            // A vertex exactly on the cusp, which is what makes the corner exact.
            let k = (0..n)
                .min_by(|&a, &b| {
                    let da = (pts[a][0] - cusp[0]).hypot(pts[a][1] - cusp[1]);
                    let db = (pts[b][0] - cusp[0]).hypot(pts[b][1] - cusp[1]);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            let miss = (pts[k][0] - cusp[0]).hypot(pts[k][1] - cusp[1]);
            assert!(miss < 1e-9, "no vertex on the cusp at {uc}, nearest is {miss} away");
            // Both edges meeting it are the requested length, so nothing finer
            // than an element is left behind the corner for the mesher to choke
            // on. Sampling the cusp naively instead left sub-element structure
            // and took the minimum angle to 5 degrees.
            for j in [(k + n - 1) % n, (k + 1) % n] {
                let e = (pts[j][0] - cusp[0]).hypot(pts[j][1] - cusp[1]);
                assert!(
                    (e - o.cusp_edge).abs() < 1e-6,
                    "edge at the cusp is {e}, asked for {}",
                    o.cusp_edge
                );
            }
        }
    }

    #[test]
    fn the_cusp_treatment_is_symmetric_about_the_cusp() {
        // The reason this test exists. An earlier repair excised a parametric
        // window and jumped to its far edge, cutting the cusp between whatever
        // the stride had reached and the exact far side: x = +1.229 against
        // x = -1.000, a cut 2.25 long tilted by 9 degrees, with the SAME
        // handedness at both cusps. Nothing in the winding or the area caught it.
        // It was visible in a picture.
        for o in [sharp_opts()] {
            let c = latmatch();
            let m = confined_mesh(c, o);
            let pts: Vec<[f64; 2]> = m
                .boundary_vertices
                .iter()
                .map(|&i| {
                    let v = m.mesh.vertex(i);
                    [v[0], v[1]]
                })
                .collect();
            // The cusp at u = PI/k is on the +y axis for the nephroid, so its
            // mirror is x -> -x.
            // The sharp path builds the whole boundary from one cusp-to-tip
            // arc, reflected and rotated, so its symmetry is exact everywhere
            // and the test says so. The fillet path walks once round, and beyond
            // the tangency the stride is not mirrored: edge lengths differ by
            // about fifteen percent, which moves sample points along the curve
            // and never the curve. The centroid check below covers the domain.
            // The sharp path builds the whole boundary from one cusp-to-tip
            // arc, reflected and rotated, so its symmetry is exact everywhere.
            let span = f64::INFINITY;
            let near: Vec<[f64; 2]> =
                pts.iter().copied().filter(|p| p[1] > 0.0 && p[0].abs() <= span).collect();
            assert!(near.len() >= 3, "only {} points at the cusp", near.len());
            for p in &near {
                let best = near
                    .iter()
                    .map(|q| ((q[0] + p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    best < 1e-9,
                    "point ({}, {}) has no mirror partner, nearest miss {best}",
                    p[0],
                    p[1]
                );
            }

            // And the domain itself. A tilted cut moves area from one side of a
            // cusp to the other and walks the centroid off the origin; sampling
            // jitter on a symmetric curve does not.
            let n = pts.len();
            let (mut area, mut cx, mut cy) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let (a, b) = (pts[i], pts[(i + 1) % n]);
                let cross = a[0] * b[1] - b[0] * a[1];
                area += cross;
                cx += (a[0] + b[0]) * cross;
                cy += (a[1] + b[1]) * cross;
            }
            area /= 2.0;
            cx /= 6.0 * area;
            cy /= 6.0 * area;
            let scale = area.abs().sqrt();
            assert!(
                cx.abs() / scale < 1e-6 && cy.abs() / scale < 1e-6,
                "centroid ({cx}, {cy}) is off the origin by {}",
                cx.hypot(cy) / scale
            );
        }
    }

    #[test]
    fn the_sharp_cusp_realises_the_edge_it_is_given() {
        let c = latmatch();
        for edge in [0.5, 1.0, 2.0] {
            let e = c.cusp_edge_param(edge);
            assert!(e > 0.0, "edge {edge} gave no truncation");
            let uc = PI / c.k();
            let p = c.point(uc + e);
            let pc = c.point(uc);
            let got = (p[0] - pc[0]).hypot(p[1] - pc[1]);
            assert!((got - edge).abs() < 1e-9 * edge, "edge {edge}: realised {got}");
        }
        // No cusp to treat, and nothing asked for.
        assert_eq!(Epitrochoid { q: 1.0, d: 1.0, r: 50.0 }.cusp_edge_param(1.0), 0.0);
        assert_eq!(c.cusp_edge_param(0.0), 0.0);
        assert_eq!(c.cusp_edge_param(-1.0), 0.0);
    }

    #[test]
    fn a_cusp_treatment_costs_little_area_and_nothing_at_all_when_off() {
        // Either treatment perturbs the domain, and it has to stay far below the
        // 0.9 percent the reference lattice's own staircased mask loses, or the
        // geometry no longer carries the paper's effective system length.
        let c = latmatch();
        let a = c.r / (c.k() + 2.0);
        let exact = PI * a * a * (c.k() + 1.0) * ((c.k() + 1.0) + c.d * c.d);
        for o in [sharp_opts()] {
            let m = confined_mesh(c, o);
            let p: Vec<[f64; 2]> = m
                .boundary_vertices
                .iter()
                .map(|&i| {
                    let v = m.mesh.vertex(i);
                    [v[0], v[1]]
                })
                .collect();
            let n = p.len();
            let mut area = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                area += p[i][0] * p[j][1] - p[j][0] * p[i][1];
            }
            let rel = ((area / 2.0).abs() - exact) / exact;
            assert!(rel.abs() < 3e-3, "area off by {rel} of the curve's own");
        }
        // Off by default, and off leaves a smooth curve's sampling alone.
        assert_eq!(MeshOpts::default().cusp_edge, 0.0);
        let smooth = neph(0.9);
        let o = MeshOpts { h_bulk: 4.0, h_min: smooth.cusp_radius() / 4.0, ..Default::default() };
        let m1 = confined_mesh(smooth, o);
        let m2 = confined_mesh(smooth, MeshOpts { cusp_edge: 0.0, ..o });
        assert_eq!(m1.boundary_params.len(), m2.boundary_params.len());
        for (x, y) in m1.boundary_params.iter().zip(m2.boundary_params.iter()) {
            assert_eq!(x, y);
        }
    }
}
