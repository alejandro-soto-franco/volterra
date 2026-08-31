//! Ensemble topological entropy, by an advected band on a triangulation.
//!
//! A lower bound on the topological entropy of a two-dimensional flow, taken
//! from an ensemble of trajectories rather than from a handful of distinguished
//! ones. The construction follows E-tec (Roberts, Sindi, Smith and Mitchell,
//! Chaos 29, 013124) and the surface train-track formulation of Smith, and it
//! is written here so a single implementation serves a bounded planar domain
//! and a closed curved surface.
//!
//! # Method
//!
//! The moving points are the vertices of a triangulation. A closed curve, the
//! band, is stored not as a polyline but as its intersection number with every
//! edge, which is a measured train track in normal coordinates. As the points
//! move the triangulation is repaired by flips, and every flip transforms the
//! weights by
//!
//! ```text
//! E' = max(A + C, B + D) - E
//! ```
//!
//! with `A, B, C, D` the quadrilateral's weights in cyclic order and `E` the
//! diagonal being replaced. The band's total measure grows exponentially in a
//! mixing flow and its growth rate bounds the topological entropy below.
//!
//! The point of the representation is that the edge COUNT stays fixed while
//! the weights grow. Advecting the curve itself would need exponentially many
//! points to keep it resolved, which is the cost this avoids.
//!
//! # Coordinate freedom
//!
//! The algorithm needs the domain for exactly two decisions: whether a triangle
//! is positively oriented, and whether a point falls inside another triangle's
//! circumcircle. Both are signs, both are intrinsic to an oriented surface with
//! a notion of circle, and nothing else about the geometry enters. [`Domain`]
//! is that interface, so a bounded region of the plane and a sphere differ only
//! in two predicates.
//!
//! # Dimension
//!
//! Braiding entropy is two-dimensional and not by convention. The fundamental
//! group of the configuration space of `n` points in `R^3` is the symmetric
//! group, so there are no braids to speak of in three dimensions and no
//! entropy to extract from point trajectories there. A three-dimensional flow
//! is treated through two-dimensional sections, or through the growth of
//! material surfaces, which is a different construction from this one.

/// The two predicates a triangulation needs, and nothing more.
///
/// Everything else about the geometry stays with the implementation. A bounded
/// planar region and a sphere give different answers to these two questions and
/// identical answers to every other question the algorithm asks.
pub trait Domain {
    /// A point of the domain.
    type Point: Copy + std::fmt::Debug;

    /// Sign of the oriented area of `(a, b, c)`, positive when the triangle
    /// agrees with the domain's orientation.
    fn orient(&self, a: Self::Point, b: Self::Point, c: Self::Point) -> f64;

    /// Positive when `d` lies inside the circumcircle of the positively
    /// oriented triangle `(a, b, c)`, so the pair should be flipped.
    fn in_circle(
        &self,
        a: Self::Point,
        b: Self::Point,
        c: Self::Point,
        d: Self::Point,
    ) -> f64;
}

/// A bounded region of the plane.
///
/// The boundary enters as points that do not move, which is how a bounded
/// domain is presented to a triangulation: the band cannot escape past them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Plane;

impl Domain for Plane {
    type Point = [f64; 2];

    fn orient(&self, a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    fn in_circle(&self, a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
        // The lifted determinant: the plane's points raised to the paraboloid,
        // where a circle becomes a plane section.
        let ax = a[0] - d[0];
        let ay = a[1] - d[1];
        let bx = b[0] - d[0];
        let by = b[1] - d[1];
        let cx = c[0] - d[0];
        let cy = c[1] - d[1];
        let a2 = ax * ax + ay * ay;
        let b2 = bx * bx + by * by;
        let c2 = cx * cx + cy * cy;
        ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx)
    }
}

/// The unit sphere, with points as unit vectors in `R^3`.
///
/// The Delaunay triangulation of points on a sphere is its convex hull, so the
/// circumcircle test is whether a point lies outside the plane of a face.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sphere;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }

impl Domain for Sphere {
    type Point = [f64; 3];

    fn orient(&self, a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        // The triple product, which is the oriented area form of the sphere
        // read through the ambient space: positive when the outward normal of
        // `(a, b, c)` points away from the centre.
        dot(cross(sub(b, a), sub(c, a)), a)
    }

    fn in_circle(&self, a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
        // `d` outside the face's plane is `d` inside the circumcircle of the
        // face, since every point sits on the sphere.
        dot(cross(sub(b, a), sub(c, a)), sub(d, a))
    }
}

/// A triangulation of moving points, with a band stored as edge weights.
pub struct Band<D: Domain> {
    domain: D,
    /// Point positions, in the domain's own representation.
    pub points: Vec<D::Point>,
    /// Triangles, each positively oriented.
    tris: Vec<[usize; 3]>,
    /// The three edges of each triangle, edge `k` opposite vertex `k`.
    tri_edges: Vec<[usize; 3]>,
    /// Edge endpoints.
    edges: Vec<[usize; 2]>,
    /// The at most two triangles on each edge.
    edge_tris: Vec<[Option<usize>; 2]>,
    /// Intersection number of the band with each edge.
    pub weights: Vec<f64>,
    /// Accumulated log growth of the band's total measure.
    pub log_growth: f64,
    /// Total measure at the last renormalisation.
    last_total: f64,
    /// Flips performed, which is the work the motion has done on the band.
    pub flips: usize,
}

impl<D: Domain> Band<D> {
    /// Build from a triangulation the caller has already formed.
    ///
    /// `tris` must be positively oriented and must tile the domain.
    pub fn new(domain: D, points: Vec<D::Point>, tris: Vec<[usize; 3]>) -> Self {
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut index = std::collections::HashMap::new();
        let mut tri_edges = Vec::with_capacity(tris.len());
        let mut edge_tris: Vec<[Option<usize>; 2]> = Vec::new();
        for (t, tri) in tris.iter().enumerate() {
            let mut te = [0usize; 3];
            for k in 0..3 {
                // Edge `k` is opposite vertex `k`.
                let (u, v) = (tri[(k + 1) % 3], tri[(k + 2) % 3]);
                let key = if u < v { (u, v) } else { (v, u) };
                let e = *index.entry(key).or_insert_with(|| {
                    edges.push([key.0, key.1]);
                    edge_tris.push([None, None]);
                    edges.len() - 1
                });
                te[k] = e;
                if edge_tris[e][0].is_none() {
                    edge_tris[e][0] = Some(t);
                } else {
                    edge_tris[e][1] = Some(t);
                }
            }
            tri_edges.push(te);
        }
        let n_e = edges.len();
        Self {
            domain,
            points,
            tris,
            tri_edges,
            edges,
            edge_tris,
            weights: vec![0.0; n_e],
            log_growth: 0.0,
            last_total: 1.0,
            flips: 0,
        }
    }

    /// Number of edges, which stays fixed however far the band is stretched.
    pub fn n_edges(&self) -> usize { self.edges.len() }
    /// Number of triangles.
    pub fn n_tris(&self) -> usize { self.tris.len() }

    /// Wrap the band around `inside`, as the boundary of that set of vertices.
    ///
    /// A simple closed curve bounding a union of cells crosses exactly those
    /// edges with one endpoint in the set, once each, so the intersection
    /// numbers are one there and zero elsewhere. Every triangle then reads
    /// `(1, 1, 0)` or `(0, 0, 0)`, both of which satisfy the triangle
    /// inequalities and the parity a curve system needs.
    pub fn encircle(&mut self, inside: &[bool]) {
        for (e, w) in self.weights.iter_mut().enumerate() {
            let [u, v] = self.edges[e];
            *w = if inside[u] != inside[v] { 1.0 } else { 0.0 };
        }
        self.last_total = self.total().max(1e-300);
    }

    /// Total measure of the band.
    pub fn total(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// The two triangles on an edge, with the local index of the edge in
    /// each and the apex opposite it.
    fn quad(&self, e: usize) -> Option<(usize, usize, usize, usize, usize, usize)> {
        let (t0, t1) = (self.edge_tris[e][0]?, self.edge_tris[e][1]?);
        let k0 = (0..3).find(|&k| self.tri_edges[t0][k] == e)?;
        let k1 = (0..3).find(|&k| self.tri_edges[t1][k] == e)?;
        let p = self.tris[t0][k0];
        let q = self.tris[t1][k1];
        Some((t0, t1, k0, k1, p, q))
    }

    /// Flip the edge, rewriting both triangles and updating the band's weight.
    ///
    /// The weight rule is `E' = max(A + C, B + D) - E` on the quadrilateral's
    /// four sides in cyclic order, which is involutive: flipping back returns
    /// the original weight, so the band loses nothing to a flip it later undoes.
    fn flip(&mut self, e: usize) -> bool {
        let Some((t0, t1, k0, _k1, p, q)) = self.quad(e) else {
            return false;
        };
        // The shared edge as the first triangle sees it, so the quadrilateral
        // reads `p, u, q, v` in the domain's own cyclic order.
        let u = self.tris[t0][(k0 + 1) % 3];
        let v = self.tris[t0][(k0 + 2) % 3];
        let side = |t: usize, a: usize, b: usize, s: &Self| -> usize {
            let k = (0..3)
                .find(|&k| {
                    let x = s.tris[t][(k + 1) % 3];
                    let y = s.tris[t][(k + 2) % 3];
                    (x == a && y == b) || (x == b && y == a)
                })
                .expect("a triangle knows its own edges");
            s.tri_edges[t][k]
        };
        let e_pu = side(t0, p, u, self);
        let e_uq = side(t1, u, q, self);
        let e_qv = side(t1, q, v, self);
        let e_vp = side(t0, v, p, self);

        let (a, b, c, d) = (
            self.weights[e_pu],
            self.weights[e_uq],
            self.weights[e_qv],
            self.weights[e_vp],
        );
        self.weights[e] = (a + c).max(b + d) - self.weights[e];

        // The diagonal now runs `p` to `q`. Both new triangles keep the cycle's
        // orientation, which is what `flippable` checked before coming here.
        self.edges[e] = [p.min(q), p.max(q)];
        self.tris[t0] = [p, u, q];
        self.tris[t1] = [p, q, v];
        self.tri_edges[t0] = [e_uq, e, e_pu];
        self.tri_edges[t1] = [e_qv, e_vp, e];

        // Only two sides change hands: `u->q` passes from the second triangle
        // to the first, and `v->p` the other way. The diagonal and the other
        // two sides keep the triangles they had.
        let swap = |et: &mut [Option<usize>; 2], from: usize, to: usize| {
            for slot in et.iter_mut() {
                if *slot == Some(from) {
                    *slot = Some(to);
                    return;
                }
            }
        };
        swap(&mut self.edge_tris[e_uq], t1, t0);
        swap(&mut self.edge_tris[e_vp], t0, t1);
        self.flips += 1;
        true
    }

    /// True when the quadrilateral on this edge is convex, so a flip leaves
    /// two positively oriented triangles.
    fn flippable(&self, e: usize) -> bool {
        let Some((t0, k0, _t1, _k1, p, q)) = self
            .quad(e)
            .map(|(t0, t1, k0, k1, p, q)| (t0, k0, t1, k1, p, q))
        else {
            return false;
        };
        let u = self.tris[t0][(k0 + 1) % 3];
        let v = self.tris[t0][(k0 + 2) % 3];
        let (pp, uu, qq, vv) = (
            self.points[p],
            self.points[u],
            self.points[q],
            self.points[v],
        );
        self.domain.orient(pp, uu, qq) > 0.0 && self.domain.orient(pp, qq, vv) > 0.0
    }

    /// Restore the Delaunay condition, flipping and updating weights as it goes.
    ///
    /// Any sequence of flips transports the band correctly, so the triangulation
    /// is free to be repaired by whatever rule keeps its triangles well shaped.
    /// The Delaunay condition is that rule.
    pub fn repair(&mut self, max_passes: usize) -> usize {
        let mut done = 0;
        for _ in 0..max_passes {
            let mut any = false;
            for e in 0..self.edges.len() {
                if self.edge_tris[e][1].is_none() || !self.flippable(e) {
                    continue;
                }
                let (t0, k0, q) = {
                    let (t0, _t1, k0, k1, _p, q) = self.quad(e).unwrap();
                    let _ = k1;
                    (t0, k0, q)
                };
                let a = self.points[self.tris[t0][k0]];
                let b = self.points[self.tris[t0][(k0 + 1) % 3]];
                let c = self.points[self.tris[t0][(k0 + 2) % 3]];
                if self.domain.in_circle(a, b, c, self.points[q]) > 0.0 {
                    self.flip(e);
                    any = true;
                    done += 1;
                }
            }
            if !any {
                break;
            }
        }
        done
    }

    /// Triangles that have turned over, which means the motion outran the
    /// repair and the reading is no longer trustworthy.
    pub fn inverted(&self) -> usize {
        self.tris
            .iter()
            .filter(|t| {
                self.domain.orient(self.points[t[0]], self.points[t[1]], self.points[t[2]])
                    <= 0.0
            })
            .count()
    }

    /// Take the growth since the last call and renormalise the band.
    ///
    /// The measure grows exponentially, so it is divided out each step and its
    /// logarithm accumulated. Without this the weights overflow long before a
    /// rate can be read.
    pub fn accumulate(&mut self) {
        let total = self.total();
        if total > 0.0 && self.last_total > 0.0 {
            self.log_growth += (total / self.last_total).ln();
        }
        if total > 0.0 {
            for w in self.weights.iter_mut() {
                *w /= total;
            }
            self.last_total = 1.0;
        }
    }
}

/// Every Delaunay triangle, by the empty-circumcircle property applied
/// directly.
///
/// Quartic in the point count, so this is the reference the incremental
/// constructions are checked against rather than the one a run uses. See
/// `delaunay_sphere` and `delaunay_plane`, which are quadratic and agree with
/// this one wherever the triangulation is unique.
pub fn delaunay_small<D: Domain>(domain: &D, points: &[D::Point]) -> Vec<[usize; 3]> {
    let n = points.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (mut a, mut b, c) = (i, j, k);
                if domain.orient(points[a], points[b], points[c]) < 0.0 {
                    std::mem::swap(&mut a, &mut b);
                }
                if domain.orient(points[a], points[b], points[c]) <= 0.0 {
                    continue;
                }
                let empty = (0..n).all(|d| {
                    d == a
                        || d == b
                        || d == c
                        || domain.in_circle(points[a], points[b], points[c], points[d]) <= 0.0
                });
                if empty {
                    out.push([a, b, c]);
                }
            }
        }
    }
    out
}

/// The Delaunay triangulation of points on a sphere, built by insertion.
///
/// Points on a sphere are the vertices of their own convex hull, so a point
/// lies inside a face's circumcircle exactly when it lies outside that face's
/// plane. Inserting a point is then removing the faces it sees and joining it
/// to the horizon those faces leave behind, which is the incremental hull.
pub fn delaunay_sphere(points: &[[f64; 3]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 4 {
        return Vec::new();
    }
    // The signed volume of a tetrahedron, positive when `d` lies on the
    // outward side of the face `(a, b, c)` and so sees it.
    let vol = |a: usize, b: usize, c: usize, d: usize| {
        Sphere.in_circle(points[a], points[b], points[c], points[d])
    };

    // Four points in general position, each taken subject to the ones before
    // it: distinct, then off their line, then off their plane.
    let mut seed: Vec<usize> = vec![0];
    for i in 1..n {
        match seed.len() {
            1 => {
                let e = sub(points[i], points[seed[0]]);
                if dot(e, e) > 0.0 {
                    seed.push(i);
                }
            }
            2 => {
                let u = sub(points[seed[1]], points[seed[0]]);
                let w = sub(points[i], points[seed[0]]);
                let c = cross(u, w);
                if dot(c, c) > 0.0 {
                    seed.push(i);
                }
            }
            _ => {
                if vol(seed[0], seed[1], seed[2], i) != 0.0 {
                    seed.push(i);
                    break;
                }
            }
        }
    }
    if seed.len() < 4 {
        return Vec::new();
    }

    // The seed tetrahedron, with every face turned outward. Orienting the
    // first face so the fourth point lies behind it puts the other three the
    // right way round as well.
    let (mut a, mut b) = (seed[0], seed[1]);
    let (c, d) = (seed[2], seed[3]);
    if vol(a, b, c, d) > 0.0 {
        std::mem::swap(&mut a, &mut b);
    }
    let mut faces = vec![[a, b, c], [a, c, d], [a, d, b], [b, d, c]];

    let mut placed = vec![false; n];
    for &v in &seed {
        placed[v] = true;
    }

    let mut visible: Vec<bool> = Vec::new();
    let mut directed: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for p in 0..n {
        if placed[p] {
            continue;
        }
        // The faces the new point lies outside of. On a convex hull this set
        // is connected, so scanning it whole needs no walk to reach it.
        visible.clear();
        visible.extend(
            faces
                .iter()
                .map(|f| vol(f[0], f[1], f[2], p) > 0.0),
        );
        if !visible.iter().any(|&v| v) {
            // Inside the hull already, which on a sphere means a repeated
            // point. It contributes no face.
            continue;
        }

        // The horizon is those directed edges of the visible set whose
        // reverse is absent, which is to say the edges shared with a face
        // that stays.
        directed.clear();
        for (f, _) in faces.iter().zip(&visible).filter(|&(_, &v)| v) {
            directed.insert((f[0], f[1]));
            directed.insert((f[1], f[2]));
            directed.insert((f[2], f[0]));
        }
        let horizon: Vec<(usize, usize)> = directed
            .iter()
            .copied()
            .filter(|&(x, y)| !directed.contains(&(y, x)))
            .collect();

        let mut kept: Vec<[usize; 3]> = faces
            .iter()
            .zip(&visible)
            .filter(|&(_, &v)| !v)
            .map(|(f, _)| *f)
            .collect();
        // The new point takes the horizon edge in the direction the removed
        // face gave it, which leaves the join outward like its neighbour.
        kept.extend(horizon.into_iter().map(|(x, y)| [x, y, p]));
        faces = kept;
        placed[p] = true;
    }
    faces
}

/// The Delaunay triangulation of points of the plane, built by insertion.
///
/// Inverse stereographic projection sends circles of the plane to circles of
/// the sphere and the plane's point at infinity to the pole. The planar
/// triangulation is then the sphere's with the pole's own faces dropped, and
/// those dropped faces are the convex hull's edges.
pub fn delaunay_plane(points: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let mut lifted: Vec<[f64; 3]> = points
        .iter()
        .map(|&[x, y]| {
            let q = x * x + y * y;
            let d = 1.0 + q;
            [2.0 * x / d, 2.0 * y / d, (q - 1.0) / d]
        })
        .collect();
    // The pole is the plane's point at infinity, and the faces that reach it
    // are the hull's edges seen from outside.
    let pole = n;
    lifted.push([0.0, 0.0, 1.0]);

    delaunay_sphere(&lifted)
        .into_iter()
        .filter(|f| !f.contains(&pole))
        .map(|mut f| {
            // Projection reverses orientation on one of the two hemispheres,
            // so the sign is settled in the plane rather than assumed.
            if Plane.orient(points[f[0]], points[f[1]], points[f[2]]) < 0.0 {
                f.swap(0, 1);
            }
            f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points of a disc, spread without the co-circularity a grid would give.
    fn disc(n: usize) -> Vec<[f64; 2]> {
        // A reproducible sequence, since a triangulation is only unique when
        // no four points share a circle.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let r = 5.0 * next().sqrt();
                let th = std::f64::consts::TAU * next();
                [r * th.cos(), r * th.sin()]
            })
            .collect()
    }

    /// The lifted construction agrees with the brute force in the plane.
    #[test]
    fn the_planar_insertion_agrees_with_the_brute_force() {
        for n in [8usize, 30, 90] {
            let pts = disc(n);
            let want = face_set(&delaunay_small(&Plane, &pts));
            let got = face_set(&delaunay_plane(&pts));
            assert_eq!(got, want, "the two constructions differ at {n} points");
        }
    }

    /// The planar triangulation covers the hull and turns the right way.
    ///
    /// Euler gives `2n - 2 - h` faces for `h` points on the convex hull, so a
    /// face count that matches is a triangulation with no piece missing.
    #[test]
    fn the_planar_insertion_covers_the_hull() {
        for n in [8usize, 30, 90, 400] {
            let pts = disc(n);
            let tris = delaunay_plane(&pts);
            for t in &tris {
                assert!(
                    Plane.orient(pts[t[0]], pts[t[1]], pts[t[2]]) > 0.0,
                    "face {t:?} is not positively oriented at {n} points"
                );
            }
            let hull = (0..n)
                .filter(|&i| {
                    (0..n).any(|j| {
                        j != i
                            && (0..n).all(|k| {
                                k == i || k == j || Plane.orient(pts[i], pts[j], pts[k]) >= 0.0
                            })
                    })
                })
                .count();
            assert_eq!(
                tris.len(),
                2 * n - 2 - hull,
                "{n} points with {hull} on the hull gave {} faces",
                tris.len()
            );
        }
    }

    /// What the incremental construction is for.
    ///
    /// The brute force is quartic in the point count and the insertion is
    /// quadratic, so the two part company well before the ensemble sizes a
    /// mixing measurement wants. Ignored by default because it measures a
    /// duration rather than an answer.
    #[test]
    #[ignore]
    fn the_incremental_hull_outruns_the_brute_force() {
        for n in [200usize, 400, 800, 2000, 5000] {
            let pts = fib_sphere(n);
            let t0 = std::time::Instant::now();
            let fast = delaunay_sphere(&pts);
            let t_fast = t0.elapsed();
            assert_eq!(fast.len(), 2 * n - 4);
            if n <= 400 {
                let t1 = std::time::Instant::now();
                let slow = delaunay_small(&Sphere, &pts);
                let t_slow = t1.elapsed();
                assert_eq!(face_set(&fast), face_set(&slow));
                println!(
                    "n = {n}: insertion {:?}, brute force {:?}, ratio {:.0}",
                    t_fast,
                    t_slow,
                    t_slow.as_secs_f64() / t_fast.as_secs_f64()
                );
            } else {
                println!("n = {n}: insertion {t_fast:?}");
            }
        }
    }

    /// Faces as sorted index triples, so two triangulations compare as sets.
    fn face_set(tris: &[[usize; 3]]) -> std::collections::BTreeSet<[usize; 3]> {
        tris.iter()
            .map(|t| {
                let mut f = *t;
                f.sort_unstable();
                f
            })
            .collect()
    }

    /// No point lies inside the circumcircle of any face.
    ///
    /// This is the definition of a Delaunay triangulation, so it tests the
    /// construction against the property it claims rather than against another
    /// implementation of itself.
    fn every_face_is_empty(pts: &[[f64; 3]], tris: &[[usize; 3]]) -> Option<String> {
        for t in tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            if Sphere.orient(pts[a], pts[b], pts[c]) <= 0.0 {
                return Some(format!("face {t:?} is not positively oriented"));
            }
            for (d, p) in pts.iter().enumerate() {
                if d == a || d == b || d == c {
                    continue;
                }
                let s = Sphere.in_circle(pts[a], pts[b], pts[c], *p);
                if s > 1e-12 {
                    return Some(format!("point {d} sits inside face {t:?} by {s:e}"));
                }
            }
        }
        None
    }

    /// The incremental hull gives a Delaunay triangulation.
    #[test]
    fn the_incremental_sphere_hull_is_delaunay() {
        for n in [4usize, 12, 40, 120, 501] {
            let pts = fib_sphere(n);
            let tris = delaunay_sphere(&pts);
            assert_eq!(tris.len(), 2 * n - 4, "{n} points gave {} faces", tris.len());
            if let Some(why) = every_face_is_empty(&pts, &tris) {
                panic!("{n} points: {why}");
            }
        }
    }

    /// The incremental hull agrees with the brute force.
    #[test]
    fn the_incremental_sphere_hull_agrees_with_the_brute_force() {
        for n in [12usize, 40, 120] {
            let pts = fib_sphere(n);
            let want = face_set(&delaunay_small(&Sphere, &pts));
            let got = face_set(&delaunay_sphere(&pts));
            assert_eq!(got, want, "the two constructions differ at {n} points");
        }
    }

    /// The flip rule undoes itself.
    ///
    /// `E' = max(A + C, B + D) - E` applied twice returns `E`, so a band loses
    /// nothing to a flip that is later reversed. Without this the measure would
    /// drift with the triangulation's history rather than with the motion.
    #[test]
    fn a_flip_and_its_reverse_restore_the_weight() {
        let cases: [(f64, f64, f64, f64, f64); 4] = [
            (1.0, 2.0, 3.0, 4.0, 2.5),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (7.0, 1.0, 1.0, 7.0, 3.0),
            (2.0, 5.0, 2.0, 5.0, 6.0),
        ];
        for (a, b, c, d, e) in cases {
            let once = (a + c).max(b + d) - e;
            let back = (a + c).max(b + d) - once;
            assert!((back - e).abs() < 1e-12, "{e} became {back}");
        }
    }

    /// Rotating three points about their midpoint, as one braid generator.
    fn twist(pts: &mut [[f64; 2]], i: usize, j: usize, sign: f64, frac: f64) {
        let (a, b) = (pts[i], pts[j]);
        let m = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let th = sign * std::f64::consts::PI * frac;
        let (c, s) = (th.cos(), th.sin());
        for &k in &[i, j] {
            let d = [pts[k][0] - m[0], pts[k][1] - m[1]];
            pts[k] = [m[0] + c * d[0] - s * d[1], m[1] + s * d[0] + c * d[1]];
        }
    }

    /// Points spread on a sphere, triangulated.
    fn fib_sphere(n: usize) -> Vec<[f64; 3]> {
        let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let th = ga * i as f64;
                [r * th.cos(), r * th.sin(), z]
            })
            .collect()
    }

    /// The sphere's triangulation closes.
    ///
    /// Every point on a sphere is a vertex of its own convex hull, so the
    /// Delaunay triangulation has `2V - 4` faces by Euler's formula whatever
    /// the points are. A predicate with the wrong sign gives a face count that
    /// misses this immediately.
    #[test]
    fn the_sphere_triangulation_closes() {
        for n in [12usize, 40, 120] {
            let pts = fib_sphere(n);
            let tris = delaunay_small(&Sphere, &pts);
            assert_eq!(tris.len(), 2 * n - 4, "{n} points gave {} faces", tris.len());
            let band = Band::new(Sphere, pts, tris);
            assert_eq!(band.n_edges(), 3 * n - 6, "edge count for {n} points");
            assert_eq!(band.inverted(), 0, "a face turned over at {n} points");
        }
    }

    /// A rigid rotation stretches nothing.
    ///
    /// The whole sphere turning together deforms no material, so the band's
    /// measure must not grow however long the rotation runs. This is the
    /// control that separates a measure of mixing from a measure of motion.
    #[test]
    fn a_rigid_rotation_grows_no_band() {
        let n = 80;
        let pts = fib_sphere(n);
        let tris = delaunay_small(&Sphere, &pts);
        let mut band = Band::new(Sphere, pts, tris);
        let inside: Vec<bool> = band.points.iter().map(|p| p[2] > 0.0).collect();
        band.encircle(&inside);

        let steps = 600;
        let dth = 6.0 * std::f64::consts::TAU / steps as f64;
        for _ in 0..steps {
            let (c, s) = (dth.cos(), dth.sin());
            for p in band.points.iter_mut() {
                *p = [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
            }
            band.repair(8);
            band.accumulate();
        }
        assert_eq!(band.inverted(), 0, "the triangulation turned over");
        println!(
            "rigid rotation on a sphere: log growth {:.3e} over {} turns, {} flips",
            band.log_growth, 6, band.flips
        );
        assert!(
            band.log_growth.abs() < 1e-9,
            "a rigid rotation should grow no band, got {}",
            band.log_growth
        );
    }

    /// The golden braid, against its exact entropy.
    ///
    /// `sigma_1 sigma_2^-1` on three strands is pseudo-Anosov with dilatation
    /// `(3 + sqrt 5) / 2`, so a band wrapped around the punctures grows by that
    /// factor every time the braid is executed and the entropy is
    /// `log((3 + sqrt 5) / 2) = 0.96242`. The number is exact and independent
    /// of everything about this implementation, which is what makes it a test
    /// of the implementation rather than a description of it.
    #[test]
    fn the_golden_braid_returns_its_own_entropy() {
        let n_bdy = 16;
        let r = 6.0;
        let mut pts: Vec<[f64; 2]> = vec![[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
        for k in 0..n_bdy {
            let th = std::f64::consts::TAU * k as f64 / n_bdy as f64;
            pts.push([r * th.cos(), r * th.sin()]);
        }
        let tris = delaunay_small(&Plane, &pts);
        let mut band = Band::new(Plane, pts.clone(), tris);

        // A curve around two of the three moving punctures.
        let mut inside = vec![false; band.points.len()];
        inside[0] = true;
        inside[1] = true;
        band.encircle(&inside);

        // Which puncture sits in which slot, so a generator always acts on
        // slots rather than on labels.
        let mut slot = [0usize, 1, 2];
        let sub = 60;
        let words = 12;
        let mut per_word = Vec::new();
        for _ in 0..words {
            let before = band.log_growth;
            for (i, j, sign) in [(0usize, 1usize, 1.0_f64), (1, 2, -1.0)] {
                for _ in 0..sub {
                    twist(&mut band.points, slot[i], slot[j], sign, 1.0 / sub as f64);
                    band.repair(8);
                    band.accumulate();
                }
                slot.swap(i, j);
            }
            per_word.push(band.log_growth - before);
        }
        assert_eq!(band.inverted(), 0, "the triangulation turned over");

        let want = ((3.0 + 5.0_f64.sqrt()) / 2.0).ln();
        // The first words are transient while the band aligns with the
        // unstable foliation, so the rate is read from the later ones.
        let tail: f64 = per_word[words / 2..].iter().sum::<f64>()
            / (words - words / 2) as f64;
        println!(
            "golden braid: {tail:.6} per word against {want:.6}, \
             per word {:?}",
            per_word.iter().map(|v| (v * 1e3).round() / 1e3).collect::<Vec<_>>()
        );
        assert!(
            (tail - want).abs() < 0.02,
            "entropy per braid word {tail:.6}, wanted {want:.6}"
        );
    }
}
