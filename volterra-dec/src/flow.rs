//! Time stepping for overdamped Helfrich flow.
//!
//! ## The flow
//!
//! With `E` the discrete Helfrich energy of [`crate::bending`], `A_v` the
//! barycentric dual area and `eta` a surface drag,
//!
//! ```text
//! eta A_v dx_v/dt = -(grad E)_v
//! ```
//!
//! ## Why explicit stepping is hopeless
//!
//! `grad E` is fourth order in position, so an explicit step is limited to
//! `dt ~ eta h^4 / kappa`, and the division by `A_v ~ h^2` costs another two
//! orders. Halving the edge length then costs a factor of 64 in step count on
//! top of the factor of 4 in vertex count.
//!
//! ## The scheme
//!
//! Treat the leading operator implicitly. Linearise the fully implicit step
//! and replace the Hessian of `E` by its dominant part, the discrete
//! bi-Laplacian `L M^-1 L`, where `L` is the cotangent stiffness and `M` the
//! diagonal matrix of dual areas. Writing the step as an increment,
//!
//! ```text
//! (eta M + dt (kappa L M^-1 L + sigma L)) dx = -dt grad E(x^n)
//! x^{n+1} = x^n + dx
//! ```
//!
//! `L` and `M` are lagged at `x^n`, so each step costs one linear solve. Two
//! properties follow. The operator is symmetric positive definite, since
//! `eta M` is positive diagonal and `L M^-1 L` and `L` are positive
//! semi-definite, so conjugate gradients apply. And a discrete equilibrium is
//! preserved exactly: `grad E = 0` gives `dx = 0` whatever `dt` is, so the
//! stabilising operator adds no bias to the steady state.
//!
//! The tension term contributes `sigma L`, which is second order, and folding
//! it in removes the `h^2` limit it would otherwise impose.
//!
//! The operator is applied matrix-free from the edge list, and the cotangent
//! weights are built by the same formula the energy uses, so the implicit
//! operator and the energy share one discretisation.

use crate::bending::{BendingParams, bending_gradient};
use nalgebra::Vector3;
use std::collections::HashMap;

/// Integrator settings for [`semi_implicit_step`].
pub struct FlowConfig {
    /// Time step.
    pub dt: f64,
    /// Surface drag coefficient `eta`.
    pub eta: f64,
    /// Relative residual at which the linear solve stops.
    pub cg_tol: f64,
    /// Iteration cap for the linear solve.
    pub cg_max_iter: usize,
}

/// What the linear solve did, per step.
pub struct StepReport {
    /// Conjugate-gradient iterations taken, summed over the three coordinates.
    pub cg_iters: usize,
    /// Largest relative residual left by the three solves.
    pub cg_residual: f64,
    /// Largest vertex displacement applied.
    pub max_displacement: f64,
}

/// Barycentric dual area at each vertex: a third of each incident face.
pub fn dual_areas(vertices: &[Vector3<f64>], triangles: &[[usize; 3]]) -> Vec<f64> {
    let mut a = vec![0.0; vertices.len()];
    for t in triangles {
        let area = (vertices[t[1]] - vertices[t[0]])
            .cross(&(vertices[t[2]] - vertices[t[0]]))
            .norm()
            / 2.0;
        for &v in t {
            a[v] += area / 3.0;
        }
    }
    a
}

/// Cotangent stiffness weights as a sorted edge list `(i, j, w_ij)`, with
/// `w_ij = (cot alpha + cot beta) / 2` summed over the faces on the edge.
///
/// Sorted so that the matrix-free products sum in a fixed order and a step is
/// reproducible bit for bit.
fn cotan_edges(vertices: &[Vector3<f64>], triangles: &[[usize; 3]]) -> Vec<(usize, usize, f64)> {
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let c = t[k];
            let a = t[(k + 1) % 3];
            let b = t[(k + 2) % 3];
            let ca = vertices[a] - vertices[c];
            let cb = vertices[b] - vertices[c];
            let cross = ca.cross(&cb).norm();
            let cot = if cross > 1e-30 {
                ca.dot(&cb) / cross
            } else {
                0.0
            };
            let key = if a < b { (a, b) } else { (b, a) };
            *map.entry(key).or_insert(0.0) += 0.5 * cot;
        }
    }
    let mut edges: Vec<(usize, usize, f64)> =
        map.into_iter().map(|((i, j), w)| (i, j, w)).collect();
    edges.sort_unstable_by_key(|&(i, j, _)| (i, j));
    edges
}

/// `(L y)_i = sum_j w_ij (y_i - y_j)`, the cotangent stiffness applied
/// matrix-free.
fn apply_l(edges: &[(usize, usize, f64)], y: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for &(i, j, w) in edges {
        let d = w * (y[i] - y[j]);
        out[i] += d;
        out[j] -= d;
    }
}

/// The step operator `eta M + dt (kappa L M^-1 L + sigma L)`, matrix-free.
struct StepOperator<'a> {
    edges: &'a [(usize, usize, f64)],
    areas: &'a [f64],
    eta: f64,
    dt_kappa: f64,
    dt_sigma: f64,
    /// Reciprocal of the Jacobi preconditioner diagonal.
    inv_diag: Vec<f64>,
}

impl<'a> StepOperator<'a> {
    fn new(
        edges: &'a [(usize, usize, f64)],
        areas: &'a [f64],
        eta: f64,
        dt_kappa: f64,
        dt_sigma: f64,
    ) -> Self {
        // diag(L)_ii = sum_j w_ij and L_ij = -w_ij, so
        // diag(L M^-1 L)_ii = L_ii^2 / A_i + sum_j w_ij^2 / A_j.
        let n = areas.len();
        let mut l_diag = vec![0.0; n];
        for &(i, j, w) in edges {
            l_diag[i] += w;
            l_diag[j] += w;
        }
        let mut bi_diag: Vec<f64> = (0..n)
            .map(|i| l_diag[i] * l_diag[i] / areas[i].max(1e-30))
            .collect();
        for &(i, j, w) in edges {
            bi_diag[i] += w * w / areas[j].max(1e-30);
            bi_diag[j] += w * w / areas[i].max(1e-30);
        }
        let inv_diag = (0..n)
            .map(|i| {
                let d = eta * areas[i] + dt_kappa * bi_diag[i] + dt_sigma * l_diag[i];
                if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }
            })
            .collect();
        Self {
            edges,
            areas,
            eta,
            dt_kappa,
            dt_sigma,
            inv_diag,
        }
    }

    fn apply(&self, y: &[f64], scratch: &mut [f64], scratch2: &mut [f64], out: &mut [f64]) {
        // eta M y
        for i in 0..y.len() {
            out[i] = self.eta * self.areas[i] * y[i];
        }
        if self.dt_kappa != 0.0 {
            // dt kappa L M^-1 L y
            apply_l(self.edges, y, scratch);
            for (s, a) in scratch.iter_mut().zip(self.areas) {
                *s /= a.max(1e-30);
            }
            apply_l(self.edges, scratch, scratch2);
            for (o, s) in out.iter_mut().zip(scratch2.iter()) {
                *o += self.dt_kappa * s;
            }
        }
        if self.dt_sigma != 0.0 {
            apply_l(self.edges, y, scratch);
            for (o, s) in out.iter_mut().zip(scratch.iter()) {
                *o += self.dt_sigma * s;
            }
        }
    }
}

/// Jacobi-preconditioned conjugate gradients on the step operator.
///
/// Returns the iteration count and the final relative residual.
fn pcg(op: &StepOperator, b: &[f64], x: &mut [f64], tol: f64, max_iter: usize) -> (usize, f64) {
    let n = b.len();
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    x.fill(0.0);
    if bnorm <= f64::MIN_POSITIVE {
        return (0, 0.0);
    }

    let mut r = b.to_vec();
    let mut z: Vec<f64> = (0..n).map(|i| op.inv_diag[i] * r[i]).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0; n];
    let mut s1 = vec![0.0; n];
    let mut s2 = vec![0.0; n];
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

    for it in 0..max_iter {
        op.apply(&p, &mut s1, &mut s2, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            // The operator is SPD by construction, so this only fires on a
            // degenerate mesh. Stop rather than divide by a non-positive pivot.
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            return (it, rn / bnorm);
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn / bnorm <= tol {
            return (it + 1, rn / bnorm);
        }
        for i in 0..n {
            z[i] = op.inv_diag[i] * r[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    (max_iter, rn / bnorm)
}

/// One semi-implicit step of overdamped Helfrich flow.
///
/// Solves `(eta M + dt (kappa L M^-1 L + sigma L)) dx = -dt grad E(x^n)` and
/// applies the increment. See the module documentation for the derivation.
///
/// The three coordinates decouple, since the operator is scalar valued, so
/// this performs three independent conjugate-gradient solves.
///
/// # Panics
///
/// Panics if `params.h0` has a different length than `vertices`.
pub fn semi_implicit_step(
    vertices: &mut [Vector3<f64>],
    triangles: &[[usize; 3]],
    params: &BendingParams,
    cfg: &FlowConfig,
) -> StepReport {
    let nv = vertices.len();
    let areas = dual_areas(vertices, triangles);
    let edges = cotan_edges(vertices, triangles);
    let grad = bending_gradient(vertices, triangles, params);

    let op = StepOperator::new(
        &edges,
        &areas,
        cfg.eta,
        cfg.dt * params.kappa,
        cfg.dt * params.tension,
    );

    let mut total_iters = 0;
    let mut worst_residual: f64 = 0.0;
    let mut delta = vec![Vector3::zeros(); nv];
    let mut rhs = vec![0.0; nv];
    let mut sol = vec![0.0; nv];

    for c in 0..3 {
        for i in 0..nv {
            rhs[i] = -cfg.dt * grad[i][c];
        }
        let (iters, residual) = pcg(&op, &rhs, &mut sol, cfg.cg_tol, cfg.cg_max_iter);
        total_iters += iters;
        worst_residual = worst_residual.max(residual);
        for i in 0..nv {
            delta[i][c] = sol[i];
        }
    }

    let mut max_displacement: f64 = 0.0;
    for (v, d) in vertices.iter_mut().zip(&delta) {
        max_displacement = max_displacement.max(d.norm());
        *v += d;
    }

    StepReport {
        cg_iters: total_iters,
        cg_residual: worst_residual,
        max_displacement,
    }
}
