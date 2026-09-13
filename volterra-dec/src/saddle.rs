//! MINRES against a Riesz-map block preconditioner, for the symmetric
//! indefinite saddle point of the bounded three-dimensional Stokes solver.
//!
//! The direct sparse LU is exact and its memory grows fastest, which is what
//! sets the ceiling on chamber resolution. The pressure is the slowest quantity
//! in the validation ladder and converges in the largest cell dimension, so a
//! thin chamber needs the in-plane mesh refined with the depth, and that is the
//! regime the factorisation cannot reach.
//!
//! # The preconditioner
//!
//! The saddle point is well posed in the natural norms of the de Rham complex:
//! `H(curl)` for the vorticity, `H(div)` for the velocity and `L2` for the
//! pressure. The Riesz map in those norms is therefore the preconditioner the
//! formulation asks for, and it has no parameter to tune:
//!
//! ```text
//! P_w = M1 + d1^T M2 d1        P_u = M2 + d2^T M3 d2        P_p = M3
//! ```
//!
//! Each block is symmetric positive definite, so each may be inverted by a
//! sparse Cholesky, which is far cheaper than the coupled indefinite
//! factorisation because the blocks are smaller and definite. The diagonal of
//! the same blocks is the cheap alternative, whose memory is linear and whose
//! iteration count is not mesh independent. Both are offered and both are
//! measured, since a preconditioner nobody has counted iterations for is a
//! preference rather than a choice.
//!
//! Replacing the block solves with multigrid is what makes the iteration count
//! mesh independent in memory as well as in count, and it sits behind this same
//! interface.

use faer::linalg::solvers::SolveCore;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};

/// A symmetric sparse operator in compressed-row form, applied matrix-free.
#[derive(Debug, Clone)]
pub struct SymOperator {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
    n: usize,
}

impl SymOperator {
    /// Build from triplets.
    ///
    /// Duplicate entries are kept rather than merged, and add up when the
    /// operator is applied. The assembly emits them by the thousand, so merging
    /// would cost a sort for nothing: `apply` sums a row either way.
    pub fn from_triplets(n: usize, trip: &[(usize, usize, f64)]) -> Self {
        let mut counts = vec![0usize; n];
        for &(r, _, _) in trip {
            counts[r] += 1;
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + counts[i];
        }
        let nnz = row_ptr[n];
        let mut col_idx = vec![0usize; nnz];
        let mut val = vec![0.0f64; nnz];
        let mut fill = row_ptr[..n].to_vec();
        for &(r, c, v) in trip {
            col_idx[fill[r]] = c;
            val[fill[r]] = v;
            fill[r] += 1;
        }
        Self {
            row_ptr,
            col_idx,
            val,
            n,
        }
    }

    /// Dimension of the square operator.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// `y = A x`.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for r in 0..self.n {
            let mut s = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                s += self.val[k] * x[self.col_idx[k]];
            }
            y[r] = s;
        }
        y
    }
}

/// How the Riesz blocks are inverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RieszMode {
    /// The diagonal of each block. Linear memory, and the iteration count grows
    /// with the mesh.
    Jacobi,
    /// A sparse Cholesky of each block, applied exactly, with `H(curl)` on the
    /// edges.
    #[default]
    Exact,
    /// The same, with `L2` on the edges rather than `H(curl)`.
    ///
    /// The `(1,1)` block of the saddle point is `-M1` exactly, so the `L2` norm
    /// inverts it exactly rather than approximately. Whether that or the graph
    /// norm gives the smaller iteration count is a measurement, and the answer
    /// is in `examples/stokes3d_minres.rs`.
    ExactMass,
}

impl RieszMode {
    /// Whether the edge block includes the `curl-curl` term.
    pub fn edge_block_has_curl(self) -> bool {
        matches!(self, Self::Jacobi | Self::Exact)
    }
}

/// The Riesz map of the three natural norms, as a block-diagonal preconditioner.
pub enum Riesz {
    /// Reciprocal diagonal of the whole block-diagonal operator.
    Jacobi(Vec<f64>),
    /// Exact solves on the vorticity and velocity blocks, and the reciprocal of
    /// the pressure block, which is diagonal already.
    Exact {
        w: faer::sparse::linalg::solvers::Llt<usize, f64>,
        u: faer::sparse::linalg::solvers::Llt<usize, f64>,
        p_inverse: Vec<f64>,
        n_w: usize,
        n_u: usize,
    },
}

impl Riesz {
    /// Assemble from the three blocks, each given as triplets over its own index
    /// range, together with the pressure block's diagonal.
    ///
    /// Returns `None` when a block cannot be factorised, which for a symmetric
    /// positive definite block means the mesh produced one that is not.
    pub fn new(
        mode: RieszMode,
        n_w: usize,
        n_u: usize,
        w_block: &[(usize, usize, f64)],
        u_block: &[(usize, usize, f64)],
        p_diagonal: &[f64],
    ) -> Option<Self> {
        match mode {
            RieszMode::Jacobi => {
                let n = n_w + n_u + p_diagonal.len();
                let mut d = vec![0.0f64; n];
                for &(r, c, v) in w_block {
                    if r == c {
                        d[r] += v;
                    }
                }
                for &(r, c, v) in u_block {
                    if r == c {
                        d[n_w + r] += v;
                    }
                }
                for (i, &v) in p_diagonal.iter().enumerate() {
                    d[n_w + n_u + i] = v;
                }
                if d.iter().any(|&x| x <= 0.0) {
                    return None;
                }
                Some(Self::Jacobi(d.into_iter().map(|x| 1.0 / x).collect()))
            }
            RieszMode::Exact | RieszMode::ExactMass => {
                let factor = |n: usize, t: &[(usize, usize, f64)]| {
                    let trip: Vec<Triplet<usize, usize, f64>> =
                        t.iter().map(|&(r, c, v)| Triplet::new(r, c, v)).collect();
                    SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip)
                        .ok()?
                        .sp_cholesky(Side::Lower)
                        .ok()
                };
                if p_diagonal.iter().any(|&x| x <= 0.0) {
                    return None;
                }
                Some(Self::Exact {
                    w: factor(n_w, w_block)?,
                    u: factor(n_u, u_block)?,
                    p_inverse: p_diagonal.iter().map(|&x| 1.0 / x).collect(),
                    n_w,
                    n_u,
                })
            }
        }
    }

    /// Apply the inverse of the preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        match self {
            Self::Jacobi(inv) => r.iter().zip(inv).map(|(a, b)| a * b).collect(),
            Self::Exact {
                w,
                u,
                p_inverse,
                n_w,
                n_u,
            } => {
                let mut out = vec![0.0f64; r.len()];
                let solve = |llt: &faer::sparse::linalg::solvers::Llt<usize, f64>,
                             src: &[f64],
                             dst: &mut [f64]| {
                    let mut m = Mat::<f64>::zeros(src.len(), 1);
                    for (i, &v) in src.iter().enumerate() {
                        m[(i, 0)] = v;
                    }
                    llt.solve_in_place_with_conj(faer::Conj::No, m.as_mut());
                    for (i, d) in dst.iter_mut().enumerate() {
                        *d = m[(i, 0)];
                    }
                };
                let (head, rest) = out.split_at_mut(*n_w);
                solve(w, &r[..*n_w], head);
                let (mid, tail) = rest.split_at_mut(*n_u);
                solve(u, &r[*n_w..*n_w + *n_u], mid);
                for (i, t) in tail.iter_mut().enumerate() {
                    *t = r[*n_w + *n_u + i] * p_inverse[i];
                }
                out
            }
        }
    }
}

/// What a MINRES solve did.
#[derive(Debug, Clone, Copy)]
pub struct MinresReport {
    /// Iterations taken.
    pub iterations: usize,
    /// Final residual in the preconditioner norm, relative to the initial one.
    pub relative_residual: f64,
    /// Whether the tolerance was reached before the iteration cap.
    pub converged: bool,
}

/// Preconditioned MINRES for a symmetric indefinite operator.
///
/// The Lanczos recurrence runs in the inner product the preconditioner defines,
/// which is why the preconditioner must be positive definite: an indefinite one
/// makes `beta` the square root of a negative number and the recurrence
/// meaningless. The quantity tested against the tolerance is the residual in
/// that same norm, relative to its initial value.
pub fn minres(
    a: &SymOperator,
    m: &Riesz,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, MinresReport) {
    let n = a.dim();
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();

    let mut x = vec![0.0f64; n];
    let mut r1 = b.to_vec();
    let mut y = m.apply(&r1);
    let beta1sq = dot(&r1, &y);
    if beta1sq <= 0.0 {
        return (
            x,
            MinresReport {
                iterations: 0,
                relative_residual: 0.0,
                converged: true,
            },
        );
    }
    let beta1 = beta1sq.sqrt();

    let (mut oldb, mut beta, mut dbar, mut epsln, mut phibar) = (0.0, beta1, 0.0, 0.0, beta1);
    let (mut cs, mut sn) = (-1.0f64, 0.0f64);
    let mut w = vec![0.0f64; n];
    let mut w2 = vec![0.0f64; n];
    let mut r2 = r1.clone();
    let mut iterations = 0usize;

    for _ in 0..max_iter {
        iterations += 1;
        let s = 1.0 / beta;
        let v: Vec<f64> = y.iter().map(|q| s * q).collect();
        y = a.apply(&v);
        if iterations >= 2 {
            let f = beta / oldb;
            for i in 0..n {
                y[i] -= f * r1[i];
            }
        }
        let alfa = dot(&v, &y);
        let f = alfa / beta;
        for i in 0..n {
            y[i] -= f * r2[i];
        }
        r1 = std::mem::replace(&mut r2, y.clone());
        y = m.apply(&r2);
        oldb = beta;
        let betasq = dot(&r2, &y);
        beta = betasq.max(0.0).sqrt();

        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        let gamma = (gbar * gbar + beta * beta).sqrt().max(f64::MIN_POSITIVE);
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar *= sn;

        let denom = 1.0 / gamma;
        let w1 = std::mem::replace(&mut w2, w.clone());
        for i in 0..n {
            w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
            x[i] += phi * w[i];
        }

        if phibar <= tol * beta1 {
            return (
                x,
                MinresReport {
                    iterations,
                    relative_residual: phibar / beta1,
                    converged: true,
                },
            );
        }
        if beta <= f64::MIN_POSITIVE {
            break;
        }
    }
    (
        x,
        MinresReport {
            iterations,
            relative_residual: phibar / beta1,
            converged: false,
        },
    )
}
