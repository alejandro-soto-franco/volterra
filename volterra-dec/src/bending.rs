//! Discrete Helfrich bending energy and its exact gradient.
//!
//! ## Why this module exists
//!
//! [`crate::helfrich::helfrich_forces`] and
//! [`crate::evolving_domain::EvolvingDomain::shape_velocity`] build a force by
//! evaluating the continuum shape equation with discrete curvature estimates:
//! they reconstruct a pointwise mean curvature `H` per vertex and then apply the
//! DEC Laplacian to it. The per-vertex error in that estimate is `O(h)` and
//! non-smooth from vertex to vertex (valence-5 against valence-6 sites on an
//! icosphere), while the Laplacian amplifies by `h^-2`, so the spurious normal
//! velocity on an exact unit sphere, where the analytic answer is zero,
//! *diverges* under refinement:
//!
//! ```text
//! icosphere level    1        2        3        4        5
//! rms  v_n        1.2e0    2.6e0    5.1e0    1.0e1    2.0e1     (~h^-1)
//! max |v_n|       1.8e0    8.7e0    3.5e1    1.4e2    5.6e2     (~h^-2)
//! ```
//!
//! This module takes the other route. It defines the energy on the mesh and
//! returns its exact gradient with respect to vertex positions. A discrete
//! equilibrium is then a true critical point at every resolution, and a
//! gradient flow built on it dissipates the discrete energy by construction.
//!
//! ## The discrete energy
//!
//! Per vertex `v`, with `A_v` the barycentric dual area, `K_v` the integrated
//! mean-curvature vector from the cotangent formula, `n_v` the area-weighted
//! unit normal and `d_v` the angle defect:
//!
//! ```text
//! A_v = (1/3) sum_{f ∋ v} area(f)
//! K_v = (1/2) sum_{j ∈ N(v)} (cot alpha_vj + cot beta_vj) (x_v - x_j)
//! H_v = (K_v . n_v) / (2 A_v)
//! d_v = 2 pi - sum_{f ∋ v} angle_f(v)
//!
//! E = sum_v [ (kappa/2) (H_v - H0_v)^2 A_v  +  kappa_bar d_v  +  sigma A_v ]
//! ```
//!
//! `H_v` is signed against the mesh orientation, so on a closed surface with
//! outward-oriented faces a sphere of radius `R` has `H = +1/R`. That is the
//! physical convention, in which the area gradient is `2 sigma H` and a
//! positive `H0` bends towards the outward normal. It is the opposite of the
//! sign produced by `EvolvingDomain::recompute_curvatures`, which returns
//! `H = -1` on the unit sphere.
//!
//! ## Gauss-Bonnet
//!
//! `sum_v d_v = 2 pi chi` is an exact combinatorial identity on a closed mesh,
//! so the `kappa_bar` term is constant in the vertex positions and contributes
//! nothing to the gradient. Saddle-splay does work only where the topology
//! changes, which no primitive in this workspace performs. `kappa_bar` is
//! carried here for energy bookkeeping across such a transition.
//!
//! ## Differentiation
//!
//! The energy is a sum of local terms, each depending only on the closed
//! one-ring of its vertex. Each local term is differentiated by forward-mode
//! dual arithmetic, seeding one ring position at a time, so the gradient is
//! exact to machine precision rather than to a difference-quotient tolerance,
//! and the cost stays linear in the vertex count.

use nalgebra::Vector3;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ─────────────────────────────────────────────────────────────────────────────
// Forward-mode dual scalar
// ─────────────────────────────────────────────────────────────────────────────

/// A real value carrying its gradient with respect to one seeded 3-vector.
#[derive(Clone, Copy, Debug)]
struct Dual {
    re: f64,
    du: [f64; 3],
}

impl Dual {
    /// A constant: zero gradient.
    #[inline]
    fn cst(re: f64) -> Self {
        Dual { re, du: [0.0; 3] }
    }

    /// Component `k` of the seeded vector: unit gradient in direction `k`.
    #[inline]
    fn var(re: f64, k: usize) -> Self {
        let mut du = [0.0; 3];
        du[k] = 1.0;
        Dual { re, du }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let r = self.re.sqrt();
        let s = if r > 0.0 { 0.5 / r } else { 0.0 };
        Dual {
            re: r,
            du: [self.du[0] * s, self.du[1] * s, self.du[2] * s],
        }
    }

    /// `atan2(self, x)`, used for the interior angle. Better conditioned near a
    /// degenerate corner than `acos`, whose derivative blows up at `+/-1`.
    #[inline]
    fn atan2(self, x: Dual) -> Self {
        let d = self.re * self.re + x.re * x.re;
        let s = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d1 = |k: usize| s * (x.re * self.du[k] - self.re * x.du[k]);
        Dual {
            re: self.re.atan2(x.re),
            du: [d1(0), d1(1), d1(2)],
        }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual {
            re: self.re + o.re,
            du: [
                self.du[0] + o.du[0],
                self.du[1] + o.du[1],
                self.du[2] + o.du[2],
            ],
        }
    }
}

impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual {
            re: self.re - o.re,
            du: [
                self.du[0] - o.du[0],
                self.du[1] - o.du[1],
                self.du[2] - o.du[2],
            ],
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        Dual {
            re: -self.re,
            du: [-self.du[0], -self.du[1], -self.du[2]],
        }
    }
}

impl Mul for Dual {
    type Output = Dual;
    // The product rule legitimately adds inside a multiplication impl.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        let d1 = |k: usize| self.du[k] * o.re + self.re * o.du[k];
        Dual {
            re: self.re * o.re,
            du: [d1(0), d1(1), d1(2)],
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let inv = 1.0 / o.re;
        let inv2 = inv * inv;
        let d1 = |k: usize| (self.du[k] * o.re - self.re * o.du[k]) * inv2;
        Dual {
            re: self.re * inv,
            du: [d1(0), d1(1), d1(2)],
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, s: f64) -> Dual {
        Dual {
            re: self.re * s,
            du: [self.du[0] * s, self.du[1] * s, self.du[2] * s],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dual 3-vectors
// ─────────────────────────────────────────────────────────────────────────────

type DVec3 = [Dual; 3];

#[inline]
fn dsub(a: DVec3, b: DVec3) -> DVec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn ddot(a: DVec3, b: DVec3) -> Dual {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn dcross(a: DVec3, b: DVec3) -> DVec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dnorm(a: DVec3) -> Dual {
    ddot(a, a).sqrt()
}

#[inline]
fn dscale(a: DVec3, s: Dual) -> DVec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dadd_assign(a: &mut DVec3, b: DVec3) {
    a[0] = a[0] + b[0];
    a[1] = a[1] + b[1];
    a[2] = a[2] + b[2];
}

/// Position of vertex `q` as a dual 3-vector, seeded when `q == seed`.
#[inline]
fn dpos(q: usize, seed: usize, verts: &[Vector3<f64>]) -> DVec3 {
    let p = verts[q];
    if q == seed {
        [Dual::var(p[0], 0), Dual::var(p[1], 1), Dual::var(p[2], 2)]
    } else {
        [Dual::cst(p[0]), Dual::cst(p[1]), Dual::cst(p[2])]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Material parameters of the discrete Helfrich energy.
pub struct BendingParams {
    /// Bending rigidity `kappa`, units of energy.
    pub kappa: f64,
    /// Gaussian (saddle-splay) modulus `kappa_bar`, units of energy. Constant
    /// at fixed topology by Gauss-Bonnet, so it never enters the gradient.
    pub kappa_bar: f64,
    /// Spontaneous curvature at each vertex, one entry per vertex, in the
    /// outward-positive convention (`H = +1/R` on a sphere of radius `R`).
    pub h0: Vec<f64>,
    /// Surface tension `sigma`, units of energy per area.
    pub tension: f64,
}

/// Vertex to incident-face map.
fn vertex_faces(n_vertices: usize, triangles: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut vf = vec![Vec::new(); n_vertices];
    for (f, t) in triangles.iter().enumerate() {
        for &v in t {
            vf[v].push(f);
        }
    }
    vf
}

/// The two other vertices of triangle `t`, in the winding order that keeps
/// `cross(x_a - x_v, x_b - x_v)` aligned with the face orientation.
#[inline]
fn others_oriented(t: [usize; 3], v: usize) -> (usize, usize) {
    if t[0] == v {
        (t[1], t[2])
    } else if t[1] == v {
        (t[2], t[0])
    } else {
        (t[0], t[1])
    }
}

/// Local energy contributed by vertex `v`, differentiated with respect to the
/// position of vertex `seed`. Pass `usize::MAX` for the value alone.
fn local_energy(
    v: usize,
    faces: &[usize],
    triangles: &[[usize; 3]],
    verts: &[Vector3<f64>],
    seed: usize,
    params: &BendingParams,
) -> Dual {
    let xv = dpos(v, seed, verts);

    let mut area = Dual::cst(0.0);
    let mut kvec: DVec3 = [Dual::cst(0.0); 3];
    let mut nvec: DVec3 = [Dual::cst(0.0); 3];
    let mut angle_sum = Dual::cst(0.0);

    for &f in faces {
        let (a, b) = others_oriented(triangles[f], v);
        let xa = dpos(a, seed, verts);
        let xb = dpos(b, seed, verts);

        let ea = dsub(xa, xv);
        let eb = dsub(xb, xv);
        let cr = dcross(ea, eb);
        // |cr| is twice the face area for any labelling of the corners.
        let two_area = dnorm(cr);

        // Barycentric dual area: a third of each incident face.
        area = area + two_area * (1.0 / 6.0);

        // Area-weighted normal accumulator: |cr| = 2 * area(f).
        dadd_assign(&mut nvec, cr);

        // Interior angle at v.
        angle_sum = angle_sum + two_area.atan2(ddot(ea, eb));

        // Cotangent weights. In triangle (v, a, b) the angle at b is opposite
        // edge (v, a) and the angle at a is opposite edge (v, b). Summing over
        // the incident faces gives each edge both of its opposite angles, so
        // kvec accumulates (1/2) sum_j (cot alpha + cot beta) (x_v - x_j).
        let cot_a = ddot(dsub(xv, xa), dsub(xb, xa)) / two_area;
        let cot_b = ddot(dsub(xv, xb), dsub(xa, xb)) / two_area;
        dadd_assign(&mut kvec, dscale(dsub(xv, xa), cot_b * 0.5));
        dadd_assign(&mut kvec, dscale(dsub(xv, xb), cot_a * 0.5));
    }

    let nhat = dscale(nvec, Dual::cst(1.0) / dnorm(nvec));
    let h = ddot(kvec, nhat) / (area * 2.0);
    let dh = h - Dual::cst(params.h0[v]);

    let bending = dh * dh * area * (0.5 * params.kappa);
    let defect = Dual::cst(2.0 * std::f64::consts::PI) - angle_sum;
    let saddle = defect * params.kappa_bar;
    let surface = area * params.tension;

    bending + saddle + surface
}

/// Total discrete Helfrich energy of a closed triangle mesh in R^3.
///
/// See the module documentation for the discretisation. `params.h0` must carry
/// one entry per vertex.
///
/// # Panics
///
/// Panics if `params.h0` has a different length than `vertices`.
pub fn bending_energy(
    vertices: &[Vector3<f64>],
    triangles: &[[usize; 3]],
    params: &BendingParams,
) -> f64 {
    assert_eq!(
        params.h0.len(),
        vertices.len(),
        "h0 must have one entry per vertex"
    );
    let vf = vertex_faces(vertices.len(), triangles);
    (0..vertices.len())
        .map(|v| local_energy(v, &vf[v], triangles, vertices, usize::MAX, params).re)
        .sum()
}

/// Exact gradient of [`bending_energy`] with respect to every vertex position.
///
/// The returned vector is `dE/dx`. A descent step therefore moves along its
/// negation, and the force on vertex `v` is `-grad[v]`.
///
/// Each local term depends only on the closed one-ring of its vertex, so the
/// gradient is assembled by seeding one ring position at a time under
/// forward-mode dual arithmetic. It is exact to machine precision.
///
/// # Panics
///
/// Panics if `params.h0` has a different length than `vertices`.
pub fn bending_gradient(
    vertices: &[Vector3<f64>],
    triangles: &[[usize; 3]],
    params: &BendingParams,
) -> Vec<Vector3<f64>> {
    assert_eq!(
        params.h0.len(),
        vertices.len(),
        "h0 must have one entry per vertex"
    );
    let nv = vertices.len();
    let vf = vertex_faces(nv, triangles);
    let mut grad = vec![Vector3::zeros(); nv];

    let mut ring: Vec<usize> = Vec::with_capacity(16);
    for (v, faces) in vf.iter().enumerate() {
        // Closed one-ring: v itself plus every vertex of every incident face.
        ring.clear();
        ring.push(v);
        for &f in faces {
            for &q in &triangles[f] {
                if !ring.contains(&q) {
                    ring.push(q);
                }
            }
        }

        for &p in &ring {
            let e = local_energy(v, faces, triangles, vertices, p, params);
            grad[p] += Vector3::new(e.du[0], e.du[1], e.du[2]);
        }
    }

    grad
}
