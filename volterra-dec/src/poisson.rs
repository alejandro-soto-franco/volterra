//! Sparse Poisson solver on DEC meshes.
//!
//! Returns `psi` with `-apply_laplace_beltrami(psi) = rhs`, i.e. `Delta psi = rhs`
//! (`apply_laplace_beltrami` is `L = -Delta`). This is the sign convention of the previous
//! direct solver, which the DEC runners compose against.
//!
//! The stored `Operators::laplace_beltrami` is the mass-normalised operator `M^{-1} S`
//! (`M` = dual-area mass, `S` = symmetric cotan stiffness). It is generically NON-symmetric
//! on curved meshes, so a symmetric direct factorisation cannot be applied to it (and a
//! non-pivoting `LDL^T` mis-solves it, returning ~0 on the sphere). This module instead
//! assembles the symmetric SPD stiffness `S = diag(star0) * laplace_beltrami` and solves
//! `S psi = -M rhs` by Jacobi-preconditioned conjugate gradient, which needs only the SPD
//! matvec and is robust on any triangulation.
//!
//! Two entry points:
//!
//! - [`PoissonSolver`]: precomputes the stiffness and preconditioner once, reused across
//!   many solves (each solve is a CG iteration over the sparse matvec).
//! - [`solve_poisson`]: one-shot convenience function.

use cartan_core::Manifold;
use cartan_dec::Operators;
use nalgebra::DVector;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};
use faer::linalg::solvers::SolveCore;
use sprs::CsMat;

use crate::ichol::IChol;

/// A preconditioner application: the incomplete Cholesky when one was built,
/// diagonal scaling when it was not.
type Preconditioner<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>;

/// Precomputed symmetric stiffness and preconditioner for repeated Poisson solves.
///
/// Assembled once and reused for each CG solve. This is the recommended path when solving
/// Poisson multiple times per timestep (e.g., vorticity-stream function Stokes).
///
/// Two modes:
/// - **Closed-manifold** (default, `dirichlet_vertices` empty): the stiffness is singular
///   (constant kernel); the RHS is range-projected and the solution returned with zero mean.
///   Correct for periodic/closed meshes (sphere, torus).
/// - **Dirichlet** (`dirichlet_vertices` non-empty): enforces ψ = 0 on all listed vertices
///   (identity rows in the CG operator). Use this for bounded domains (no-slip
///   stream-function).
pub struct PoissonSolver {
    /// Number of vertices.
    n: usize,
    /// Full symmetric stiffness `S = diag(star0) * laplace_beltrami` (both triangles),
    /// applied as a matvec operator in the CG solve.
    ///
    /// `laplace_beltrami` is the mass-normalised operator `M^{-1} S`, which is NOT symmetric
    /// when the dual areas vary (any curved mesh). Left-multiplying by the mass diagonal
    /// recovers the symmetric SPD stiffness `S = d0^T star1 d0`, which is what the iterative
    /// solve requires.
    s: CsMat<f64>,
    /// The same stiffness in compressed rows, for the matvec.
    ///
    /// The triplet form is a scatter: every entry writes to an arbitrary row,
    /// so the loop cannot be split across threads without a reduction. In rows
    /// each thread owns its own output entry and the product parallelises with
    /// no synchronisation at all, which matters because this matvec is the
    /// single largest cost in a run.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
    /// Jacobi preconditioner `1 / A_ii`, where `A` is `S` with identity rows on Dirichlet
    /// DOFs. One entry per vertex.
    inv_diag: Vec<f64>,
    /// Dual-area mass diagonal (star0), one per vertex; mass-weights the right-hand side.
    star0: Vec<f64>,
    /// Per-vertex shift `c`, so the operator is `-(Delta + c)` rather than `-Delta`.
    ///
    /// Zero for a plain Poisson solve. The surface Stokes stream function needs
    /// `c = 2K`: see [`Self::new_shifted`].
    shift: Vec<f64>,
    /// A sparse Cholesky of a nearby definite operator, as the preconditioner.
    ///
    /// The operator itself cannot be factorised: on a sphere `-(Delta + 2K)`
    /// has eigenvalue `-2` on the constant and `0` on the rotations, so it is
    /// indefinite and singular. Adding `sigma M` moves the whole spectrum
    /// positive without changing the sparsity, and the factorisation of THAT
    /// is an excellent preconditioner for the operator itself: the eigenvalues
    /// of the preconditioned system are `(lambda - shift) / (lambda - shift +
    /// sigma)`, which cluster tightly for any `sigma` of the size of the
    /// shift. The conjugate gradient still converges to the exact solution of
    /// the unmodified operator, so this changes the iteration count and not
    /// the answer.
    chol: Option<ShiftedCholesky>,
    /// A mass-orthonormal basis of the shifted operator's kernel, beyond the
    /// constant the closed-manifold mode already removes.
    ///
    /// Empty for a plain Poisson solve. At `c = 2K` on a sphere it is the three
    /// rigid rotations, whose stream functions are the linear coordinates.
    kernel: Vec<Vec<f64>>,
    /// Dirichlet vertex indices (ψ = 0 enforced here). Empty → closed-manifold mode.
    dirichlet_vertices: Vec<usize>,
    /// Boolean Dirichlet mask, length `n`.
    is_dirichlet: Vec<bool>,
    /// Zero-fill incomplete Cholesky factor of the CG operator, when one could
    /// be built, and `None` when it could not, which falls the solve back to
    /// `inv_diag`.
    ///
    /// The operator is fixed for the life of the solver, so this is one
    /// factorisation for a whole run against two triangular solves per
    /// iteration. On the lattice-matched nephroid the biharmonic's two Poisson
    /// solves were 415 seconds of a 416 second run under Jacobi alone.
    ichol: Option<IChol>,
}

/// The operator the conjugate gradient applies, as triples: the stiffness with
/// Dirichlet rows and columns eliminated and a unit diagonal put back on them.
/// This is what `apply_a` computes, so it is what a preconditioner has to
/// approximate.
fn cg_operator_triples(
    s: &CsMat<f64>,
    is_dirichlet: &[bool],
    shift_diag: &[f64],
    f: &mut dyn FnMut(usize, usize, f64),
) {
    for (&v, (r, c)) in s.iter() {
        if !is_dirichlet[r] && !is_dirichlet[c] {
            let d = if r == c { shift_diag[r] } else { 0.0 };
            f(r, c, v - d);
        }
    }
    for (i, &d) in is_dirichlet.iter().enumerate() {
        if d {
            f(i, i, 1.0);
        }
    }
}


/// A sparse Cholesky of `A + sigma M`, applied as a preconditioner.
struct ShiftedCholesky {
    llt: faer::sparse::linalg::solvers::Llt<usize, f64>,
    n: usize,
}

impl ShiftedCholesky {
    /// Factorise `A + sigma M`, raising `sigma` until it is definite.
    ///
    /// `sigma` starts a little above the largest shift, which is where the
    /// spectrum crosses zero, and doubles on failure. Returning `None` leaves
    /// the caller on the incomplete Cholesky, so a mesh this cannot factorise
    /// is slower and never wrong.
    fn new(
        row_ptr: &[usize],
        col_idx: &[usize],
        val: &[f64],
        star0: &[f64],
        shift: &[f64],
        is_dirichlet: &[bool],
    ) -> Option<Self> {
        let n = star0.len();
        let max_shift = shift.iter().cloned().fold(0.0_f64, f64::max);
        // A scale for the operator, so a flat mesh with no shift still gets a
        // positive sigma rather than the singular matrix `S` itself.
        let rate: f64 = (0..n)
            .map(|r| {
                let d = (row_ptr[r]..row_ptr[r + 1])
                    .find(|&k| col_idx[k] == r)
                    .map(|k| val[k])
                    .unwrap_or(0.0);
                d / star0[r].max(1e-300)
            })
            .sum::<f64>()
            / n as f64;
        let mut sigma = 2.0 * max_shift + 1e-3 * rate.abs().max(1e-12);
        for _ in 0..8 {
            let mut trip: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(val.len() + n);
            for r in 0..n {
                if is_dirichlet[r] {
                    trip.push(Triplet::new(r, r, 1.0));
                    continue;
                }
                for k in row_ptr[r]..row_ptr[r + 1] {
                    let c = col_idx[k];
                    if is_dirichlet[c] {
                        continue;
                    }
                    let d = if r == c { star0[r] * (shift[r] - sigma) } else { 0.0 };
                    trip.push(Triplet::new(r, c, val[k] - d));
                }
            }
            if let Ok(m) = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip) {
                if let Ok(llt) = m.sp_cholesky(Side::Lower) {
                    return Some(Self { llt, n });
                }
            }
            sigma *= 2.0;
        }
        None
    }

    fn apply(&self, r: &[f64]) -> Vec<f64> {
        let mut rhs = Mat::<f64>::zeros(self.n, 1);
        for i in 0..self.n {
            rhs[(i, 0)] = r[i];
        }
        self.llt.solve_in_place_with_conj(faer::Conj::No, rhs.as_mut());
        (0..self.n).map(|i| rhs[(i, 0)]).collect()
    }
}

/// Compressed-row arrays for `s`, for the parallel matvec.
fn to_csr(s: &CsMat<f64>, n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut counts = vec![0usize; n];
    for (_, (r, _)) in s.iter() {
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
    for (&v, (r, c)) in s.iter() {
        col_idx[fill[r]] = c;
        val[fill[r]] = v;
        fill[r] += 1;
    }
    (row_ptr, col_idx, val)
}

impl PoissonSolver {
    /// Build the solver from a DEC Operators struct (closed-manifold / periodic mode).
    ///
    /// Assembles the symmetric SPD stiffness `S = diag(star0) * laplace_beltrami` and a
    /// Jacobi preconditioner. Solves are performed by preconditioned conjugate gradient
    /// (see [`Self::solve`]): a direct `LDL^T` factorisation is not used because the DEC
    /// stiffness on a general (non-well-centred) triangulation is not reliably factorised
    /// by a non-pivoting `LDL^T`, whereas CG needs only the SPD matvec.
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();
        let star0: Vec<f64> = ops.hodge.star0().iter().copied().collect();
        let s = full_stiffness(&ops.laplace_beltrami, &star0);
        let is_dirichlet = vec![false; n];
        let inv_diag = jacobi_inv_diag(&s, &is_dirichlet);
        let zero = vec![0.0; n];
        let ichol = IChol::factor(n, |f| cg_operator_triples(&s, &is_dirichlet, &zero, f));
        let (row_ptr, col_idx, val) = to_csr(&s, n);
        let zero_shift = vec![0.0; n];
        let chol = ShiftedCholesky::new(
            &row_ptr, &col_idx, &val, &star0, &zero_shift, &is_dirichlet,
        );
        Ok(Self {
            n,
            s,
            row_ptr,
            col_idx,
            val,
            chol,
            inv_diag,
            star0,
            shift: vec![0.0; n],
            kernel: Vec::new(),
            dirichlet_vertices: Vec::new(),
            is_dirichlet,
            ichol,
        })
    }

    /// Build the solver with Dirichlet ψ = 0 boundary conditions.
    ///
    /// Enforces ψ = 0 on every vertex in `dirichlet_vertices` (identity rows in the CG
    /// operator, RHS forced to 0 there). The interior system is SPD. Use this for bounded
    /// domains (confined active nematics, no-slip Stokes).
    pub fn with_dirichlet<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        dirichlet_vertices: &[usize],
    ) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();
        let star0: Vec<f64> = ops.hodge.star0().iter().copied().collect();
        let s = full_stiffness(&ops.laplace_beltrami, &star0);
        let mut is_dirichlet = vec![false; n];
        for &d in dirichlet_vertices {
            is_dirichlet[d] = true;
        }
        let inv_diag = jacobi_inv_diag(&s, &is_dirichlet);
        let zero = vec![0.0; n];
        let ichol = IChol::factor(n, |f| cg_operator_triples(&s, &is_dirichlet, &zero, f));
        let (row_ptr, col_idx, val) = to_csr(&s, n);
        let zero_shift = vec![0.0; n];
        let chol = ShiftedCholesky::new(
            &row_ptr, &col_idx, &val, &star0, &zero_shift, &is_dirichlet,
        );
        Ok(Self {
            n,
            s,
            row_ptr,
            col_idx,
            val,
            chol,
            inv_diag,
            star0,
            shift: vec![0.0; n],
            kernel: Vec::new(),
            dirichlet_vertices: dirichlet_vertices.to_vec(),
            is_dirichlet,
            ichol,
        })
    }

    /// A closed-manifold solver for `-(Delta + shift) phi = rhs`.
    ///
    /// The surface Stokes stream function needs `shift = 2K`. The momentum
    /// equation has `(Delta_B + K) u` on the vector field with the Bochner
    /// Laplacian; rewriting a divergence-free `u = J grad psi` through its
    /// stream function turns that into the scalar Laplace-Beltrami operator
    /// shifted by TWICE the curvature.
    ///
    /// The factor is what makes a Killing field free. On the unit sphere a
    /// rigid rotation about `z` has stream function `z`, an `l = 1` harmonic
    /// with `Delta z = -2 z`, so `(Delta + 2K) z = 0` while `(Delta + K) z`
    /// is `-z`. At the right shift the operator is therefore singular on a
    /// sphere, and the solve projects that kernel out of both the right-hand
    /// side and the answer, which is what returns the solution of minimal
    /// `L2` norm.
    ///
    /// On a flat mesh the shift vanishes and this is the plain Poisson solver.
    pub fn new_shifted<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        shift: &[f64],
        coords: &[[f64; 3]],
    ) -> Result<Self, String> {
        let mut solver = Self::new(ops)?;
        if shift.len() != solver.n {
            return Err(format!(
                "shift has {} entries for {} vertices",
                shift.len(),
                solver.n
            ));
        }
        solver.shift = shift.to_vec();
        // Both preconditioners describe `S`, and the shift moves the diagonal,
        // so both are rebuilt against the operator that is actually solved.
        // Preconditioning the shifted system with the unshifted factorisation
        // costs about 30 per cent more iterations a step.
        let shift_diag: Vec<f64> =
            (0..solver.n).map(|i| solver.star0[i] * solver.shift[i]).collect();
        solver.inv_diag = (0..solver.n)
            .map(|i| {
                let d = 1.0 / solver.inv_diag[i] - shift_diag[i];
                if d.abs() > 1e-300 { 1.0 / d } else { 1.0 }
            })
            .collect();
        solver.ichol = IChol::factor(solver.n, |f| {
            cg_operator_triples(&solver.s, &solver.is_dirichlet, &shift_diag, f)
        });
        solver.chol = ShiftedCholesky::new(
            &solver.row_ptr, &solver.col_idx, &solver.val,
            &solver.star0, &solver.shift, &solver.is_dirichlet,
        );
        solver.kernel = solver.find_kernel(coords);
        Ok(solver)
    }

    /// The kernel directions the solve removes, for inspection in a test.
    ///
    /// This counts how many of three CANDIDATES survive, not the operator's
    /// true kernel. Use [`Self::apply_operator`] with [`Self::mass`] to take
    /// the spectrum when the actual dimension is the question.
    pub fn kernel_dimension(&self) -> usize {
        self.kernel.len()
    }

    /// Apply the operator the solve inverts, for inspection.
    ///
    /// The symmetric stiffness with the shift already in it, which is the
    /// operator whose spectrum decides how many directions the solve must
    /// remove. Exposed so a test can take that spectrum against the real
    /// assembly rather than against a second one written to agree with it.
    pub fn apply_operator(&self, x: &[f64]) -> Vec<f64> {
        self.apply_a(x)
    }

    /// The dual-area mass diagonal, the inner product the operator is
    /// symmetric in.
    pub fn mass(&self) -> &[f64] {
        &self.star0
    }

    /// The preconditioner to drive the conjugate gradient with: the incomplete
    /// Cholesky when one was built, and diagonal scaling when it was not.
    fn precond(&self) -> Preconditioner<'_> {
        // The complete factorisation of the nearby definite operator first, the
        // incomplete one next, and diagonal scaling only when neither built.
        if let Some(ch) = &self.chol {
            return Box::new(move |r: &[f64]| ch.apply(r));
        }
        match &self.ichol {
            Some(ic) => Box::new(move |r: &[f64]| ic.apply(r)),
            None => Box::new(move |r: &[f64]| {
                (0..self.n).map(|i| self.inv_diag[i] * r[i]).collect()
            }),
        }
    }

    /// Whether the incomplete Cholesky was built, and the shift it needed.
    ///
    /// Reported so a run can record which preconditioner it actually used
    /// rather than which one it asked for.
    pub fn preconditioner(&self) -> (bool, f64) {
        match &self.ichol {
            Some(ic) => (true, ic.shift),
            None => (false, 0.0),
        }
    }

    /// Apply the CG operator `A`: `S` on the interior, identity on Dirichlet rows.
    fn apply_a(&self, x: &[f64]) -> Vec<f64> {
        // Zero the input on Dirichlet DOFs (their columns are eliminated).
        let mut xz = x.to_vec();
        for (xzi, &d) in xz.iter_mut().zip(&self.is_dirichlet) {
            if d {
                *xzi = 0.0;
            }
        }
        // Compressed rows rather than the triplet scatter: each output entry is
        // written once, from contiguous values, which the triplet form cannot
        // do. Threading this is a loss, not a win. The matrix has 72k nonzeros
        // and the product runs in tens of microseconds, so a rayon fork and
        // join per call costs more than the work it splits, and the solve calls
        // it hundreds of times a step. Measured: 22 steps a second sequential
        // against 8 threaded.
        let mut y = vec![0.0f64; self.n];
        for r in 0..self.n {
            let mut acc = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                acc += self.val[k] * xz[self.col_idx[k]];
            }
            y[r] = acc - self.star0[r] * self.shift[r] * xz[r];
        }
        // Identity rows for Dirichlet DOFs.
        for (yi, (&d, &xi)) in y.iter_mut().zip(self.is_dirichlet.iter().zip(x)) {
            if d {
                *yi = xi;
            }
        }
        y
    }

    /// Apply the operator WITHOUT its shift, for inspection.
    ///
    /// The plain stiffness, whose Rayleigh quotient on a smooth field gives the
    /// `O(1)` scale that decides whether the shifted operator has annihilated
    /// that field.
    pub fn apply_unshifted(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for r in 0..self.n {
            let mut acc = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                acc += self.val[k] * x[self.col_idx[k]];
            }
            y[r] = acc;
        }
        y
    }

    /// The mass inner product `x^T M y`.
    fn m_dot(&self, x: &[f64], y: &[f64]) -> f64 {
        (0..self.n).map(|i| self.star0[i] * x[i] * y[i]).sum()
    }

    /// Remove every kernel direction from `x`, in the mass inner product.
    fn project_out_kernel(&self, x: &mut [f64]) {
        for b in &self.kernel {
            let d = self.m_dot(x, b);
            for i in 0..self.n {
                x[i] -= d * b[i];
            }
        }
    }

    /// The kernel of the shifted operator, from the candidates a shifted
    /// Laplacian on a surface can annihilate: the three linear coordinates,
    /// whose stream functions are the rigid rotations.
    ///
    /// A candidate is kept when the shifted operator leaves almost nothing of
    /// it, measured against what the PLAIN operator does to that same field.
    /// Comparing a field with itself is the point: `rho_plain` is the field's
    /// own Rayleigh quotient under `-Delta`, an `O(1)` number set by the
    /// field's frequency, so the ratio vanishes like the mesh's own order for a
    /// Killing direction and stays `O(1)` for anything else. A surface with no
    /// Killing field keeps none of them, and a flat mesh has no shift to make
    /// any of them a kernel in the first place.
    ///
    /// The numerator is a NORM rather than a quadratic form. A Rayleigh
    /// quotient can be small because positive and negative eigencomponents
    /// cancel, which is a live possibility once `2K` is negative somewhere and
    /// the operator is indefinite.
    ///
    /// Measured, at `|A v| / (rho_plain |v|)`: an icosphere gives 4.4e-2, 1.4e-2
    /// and 4.9e-3 at levels 1, 2 and 3, falling with the mesh, while a genus-2
    /// surface gives 5.4 to 36 at two resolutions. The threshold sits between,
    /// with a factor of 2 below it and 54 above at the worst case.
    ///
    /// This replaces a test that normalised by the response of a pseudo-random
    /// probe. A random vector is all high frequency, so its response is set by
    /// the mesh's smallest triangle rather than by the geometry: that scale ran
    /// from 3.1 on a coarse sphere to 2.2e4 on a fine genus-2 mesh, which made
    /// the test read 0 rotations on an icosphere at level 1 and 3 on a genus-2
    /// surface that has none.
    ///
    /// The survivors are mass-orthonormalised, since the linear coordinates are
    /// mass-orthogonal only up to the mesh's own asymmetry.
    fn find_kernel(&self, coords: &[[f64; 3]]) -> Vec<Vec<f64>> {
        if self.shift.iter().all(|c| c.abs() < 1e-300) {
            return Vec::new();
        }
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for axis in 0..3 {
            let v: Vec<f64> = coords.iter().map(|c| c[axis]).collect();
            let mm = self.m_dot(&v, &v);
            if mm <= 0.0 {
                continue;
            }
            // The plain stiffness on this same field: `A` with the shift added
            // back, so the Dirichlet handling is identical to the numerator's.
            let av = self.apply_a(&v);
            let shifted_back: Vec<f64> = (0..self.n)
                .map(|i| av[i] + self.star0[i] * self.shift[i] * v[i])
                .collect();
            let rho_plain = self.m_dot(&v, &shifted_back) / mm;
            if rho_plain.abs() < 1e-300 {
                continue;
            }
            let rel = self.m_norm(&av) / (rho_plain.abs() * self.m_norm(&v)).max(1e-300);
            if rel > 1e-1 {
                continue;
            }
            let mut w = v;
            for b in &basis {
                let d = self.m_dot(&w, b);
                for i in 0..self.n {
                    w[i] -= d * b[i];
                }
            }
            let nrm = self.m_norm(&w);
            if nrm > 1e-8 {
                for x in w.iter_mut() {
                    *x /= nrm;
                }
                basis.push(w);
            }
        }
        basis
    }

    fn m_norm(&self, x: &[f64]) -> f64 {
        self.m_dot(x, x).sqrt()
    }

    /// Apply the raw stiffness `S`, with no Dirichlet masking.
    ///
    /// [`Self::apply_a`] eliminates the Dirichlet columns, which is right for the
    /// homogeneous solve and wrong for lifting a prescribed boundary value,
    /// where the coupling `S_IB g_B` is exactly the term wanted.
    fn apply_s_raw(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.n];
        for (&v, (r, c)) in self.s.iter() {
            y[r] += v * x[c];
        }
        y
    }

    /// Solve `Delta psi = rhs` with `psi = g` on the Dirichlet set.
    ///
    /// [`Self::solve`] fixes `psi = 0` there. Static condensation lifts a
    /// prescribed value instead: writing `psi = psi_I + g_B` with `g_B` extended
    /// by zero into the interior,
    ///
    /// ```text
    ///   S_II psi_I = -M_I rhs_I - (S g_B)_I
    /// ```
    ///
    /// which is the homogeneous system this solver already assembles, with the
    /// coupling moved to the right-hand side. `g` is a full-length vector and
    /// only its Dirichlet entries are read.
    pub fn solve_with_boundary(
        &self,
        rhs: &DVector<f64>,
        g: &[f64],
        tol: f64,
    ) -> DVector<f64> {
        assert_eq!(rhs.len(), self.n);
        assert_eq!(g.len(), self.n);
        assert!(
            !self.dirichlet_vertices.is_empty(),
            "solve_with_boundary needs a Dirichlet set"
        );
        let mut lift = vec![0.0f64; self.n];
        for &d in &self.dirichlet_vertices {
            lift[d] = g[d];
        }
        let coupling = self.apply_s_raw(&lift);
        let mut b: Vec<f64> = (0..self.n)
            .map(|i| -self.star0[i] * rhs[i] - coupling[i])
            .collect();
        for &d in &self.dirichlet_vertices {
            b[d] = 0.0;
        }
        let pc = self.precond();
        let (mut x, _) =
            pcg_solve_from_pc(|q| self.apply_a(q), &pc, &b, self.n, false, None, tol);
        for &d in &self.dirichlet_vertices {
            x[d] = 0.0;
        }
        for i in 0..self.n {
            x[i] += lift[i];
        }
        DVector::from_vec(x)
    }

    /// Solve for `psi` with `-apply_laplace_beltrami(psi) = rhs`, by Jacobi-preconditioned CG.
    ///
    /// `apply_laplace_beltrami` is `L = -Delta` (positive), so this returns `psi` with
    /// `-L psi = rhs`, i.e. `Delta psi = rhs`. In symmetric-stiffness form the system is
    /// `S psi = -M rhs` with `S = diag(star0) * L` and `M = diag(star0)`. On an eigenfunction
    /// `Y_lm` (`L Y = l(l+1) Y`) this gives `solve(l(l+1) Y) = -Y`.
    ///
    /// This preserves the sign convention of the previous direct solver (the downstream DEC
    /// runners compose against it); the change here is the solve method (robust CG on the
    /// symmetric stiffness) rather than the convention.
    ///
    /// **Closed-manifold mode**: `S` is singular (kernel = constants); the RHS is projected
    /// onto the range and the CG iterates are kept mean-free, and the solution is returned
    /// with zero mean.
    ///
    /// **Dirichlet mode**: the RHS is forced to 0 at Dirichlet DOFs and the solution is 0
    /// there exactly.
    /// Solve, starting from `x0` and stopping at `tol`, returning the iterations.
    ///
    /// See [`pcg_solve_from_pc`]. The gauge fix and the Dirichlet rows are applied
    /// exactly as in [`Self::solve`], so the two differ only in where the
    /// iteration starts and where it stops.
    pub fn solve_from(
        &self,
        rhs: &DVector<f64>,
        x0: Option<&[f64]>,
        tol: f64,
    ) -> (DVector<f64>, usize) {
        assert_eq!(rhs.len(), self.n);
        let closed = self.dirichlet_vertices.is_empty();
        let mut b: Vec<f64> = (0..self.n).map(|i| -self.star0[i] * rhs[i]).collect();
        if closed {
            let mean = b.iter().sum::<f64>() / self.n as f64;
            for v in b.iter_mut() {
                *v -= mean;
            }
        } else {
            for &d in &self.dirichlet_vertices {
                b[d] = 0.0;
            }
        }
        if !self.kernel.is_empty() {
            // `b` is mass-weighted, so the projection is done on `M^-1 b` and
            // re-weighted, keeping it in the inner product the kernel was
            // orthonormalised in.
            let mut br: Vec<f64> = (0..self.n).map(|i| b[i] / self.star0[i]).collect();
            self.project_out_kernel(&mut br);
            for i in 0..self.n {
                b[i] = self.star0[i] * br[i];
            }
        }
        let pc = self.precond();
        let (mut x, its) = pcg_solve_from_pc(
            |p| self.apply_a(p),
            &pc,
            &b,
            self.n,
            closed,
            x0,
            tol,
        );
        if closed {
            let mean = x.iter().sum::<f64>() / self.n as f64;
            for v in x.iter_mut() {
                *v -= mean;
            }
        } else {
            for &d in &self.dirichlet_vertices {
                x[d] = 0.0;
            }
        }
        self.project_out_kernel(&mut x);
        (DVector::from_vec(x), its)
    }

    /// The dual-area mass diagonal the right-hand side is weighted by.
    ///
    /// Exposed so a force-driven solve can convert a nodal functional into the
    /// pointwise source `solve` expects, which is what keeps the resulting
    /// operator symmetric. See `SurfaceStokes::solve_force_warm`.
    pub fn mass_diagonal(&self) -> &[f64] {
        &self.star0
    }

    /// Solve `-(Delta + shift) phi = rhs` cold, at a fixed tolerance.
    ///
    /// A thin wrapper over [`Self::solve_from`], which is the single
    /// implementation. Keeping a second copy here is how the kernel projection
    /// came to be applied on one path and not the other.
    pub fn solve(&self, rhs: &DVector<f64>) -> DVector<f64> {
        self.solve_from(rhs, None, 1e-10).0
    }
}


/// Dot product of two slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}



/// The same iteration with an arbitrary symmetric positive definite
/// preconditioner `M^{-1}`, given as `precond`.
///
/// The preconditioner has to be symmetric and positive definite or the search
/// directions stop being conjugate and the method is no longer the conjugate
/// gradient. Diagonal scaling satisfies that trivially; an incomplete Cholesky
/// satisfies it because `(L L^T)^{-1}` is symmetric positive definite whenever
/// `L` has a positive diagonal, which the factorisation enforces by refusing a
/// non-positive pivot.
pub(crate) fn pcg_solve_from_pc<F: Fn(&[f64]) -> Vec<f64>>(
    apply: F,
    precond: &dyn Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    n: usize,
    project_kernel: bool,
    x0: Option<&[f64]>,
    tol: f64,
) -> (Vec<f64>, usize) {
    let demean = |v: &mut [f64]| {
        let m = v.iter().sum::<f64>() / n as f64;
        for x in v.iter_mut() {
            *x -= m;
        }
    };

    let mut x = match x0 {
        Some(v) if v.len() == n => v.to_vec(),
        _ => vec![0.0f64; n],
    };
    let mut r = if x0.is_some() {
        let ax = apply(&x);
        (0..n).map(|i| b[i] - ax[i]).collect()
    } else {
        b.to_vec()
    };
    if project_kernel {
        demean(&mut r);
    }
    let mut z: Vec<f64> = precond(&r);
    if project_kernel {
        demean(&mut z);
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let bnorm = dot(b, b).sqrt().max(1e-300);
    let max_iter = 10 * n + 100;
    if dot(&r, &r).sqrt() <= tol * bnorm {
        return (x, 0);
    }

    let mut used = 0usize;
    for _ in 0..max_iter {
        used += 1;
        let ap = apply(&p);
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-300 {
            break;
        }
        let alpha = rz / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if project_kernel {
            demean(&mut r);
        }
        if dot(&r, &r).sqrt() <= tol * bnorm {
            break;
        }
        let mut z_new: Vec<f64> = precond(&r);
        if project_kernel {
            demean(&mut z_new);
        }
        let rz_new = dot(&r, &z_new);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z_new[i] + beta * p[i];
        }
        rz = rz_new;
    }
    (x, used)
}

/// One-shot convenience: factorise and solve in one call.
///
/// For repeated solves, use [`PoissonSolver`] instead.
pub fn solve_poisson<M: Manifold>(
    ops: &Operators<M, 3, 2>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, String> {
    let solver = PoissonSolver::new(ops)?;
    Ok(solver.solve(rhs))
}

/// Build the FULL symmetric stiffness `S = diag(star0) * lap`, where `lap` is the
/// mass-normalised Laplace-Beltrami operator `M^{-1} S`.
///
/// `lap` is generically NON-symmetric on curved meshes (the dual-area mass `M` varies).
/// Left-multiplying by the mass diagonal recovers the symmetric SPD stiffness
/// `S = d0^T star1 d0`: entry `(r, c)` is `star0[r] * lap_{r,c} = S_{r,c}`, and entry
/// `(c, r)` is `star0[c] * lap_{c,r} = S_{c,r} = S_{r,c}`, so both triangles agree. The full
/// matrix is stored (both triangles) so it can be applied directly as a matvec.
pub(crate) fn full_stiffness(lap: &CsMat<f64>, star0: &[f64]) -> CsMat<f64> {
    let n = lap.rows();
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for (&val, (row, col)) in lap.iter() {
        rows.push(row);
        cols.push(col);
        vals.push(star0[row] * val);
    }

    let tri = sprs::TriMat::from_triplets((n, n), rows, cols, vals);
    tri.to_csc()
}

/// Jacobi preconditioner diagonal `1 / A_ii`, where `A` is the stiffness `S` with identity
/// rows on Dirichlet DOFs. Non-positive or missing diagonals fall back to `1.0`.
pub(crate) fn jacobi_inv_diag(s: &CsMat<f64>, is_dirichlet: &[bool]) -> Vec<f64> {
    let n = s.rows();
    let mut diag = vec![0.0f64; n];
    for (&v, (r, c)) in s.iter() {
        if r == c {
            diag[r] += v;
        }
    }
    (0..n)
        .map(|i| {
            if is_dirichlet[i] {
                1.0
            } else if diag[i].abs() > 1e-300 {
                1.0 / diag[i]
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::mesh::FlatMesh;
    use cartan_manifolds::euclidean::Euclidean;

    /// The incomplete Cholesky has to solve the same system Jacobi does, and in
    /// fewer iterations, or there is no reason to carry it.
    ///
    /// A unit square with Dirichlet edges is the shape the confined stream
    /// function actually solves on, and the gain grows with the mesh diameter,
    /// so a small grid understates it rather than flattering it.
    #[test]
    fn the_incomplete_cholesky_beats_jacobi_and_lands_in_the_same_place() {
        let mesh = FlatMesh::unit_square_grid(24);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        // Dirichlet on the boundary of the grid, which is what a confined solve has.
        let edge: Vec<usize> = (0..nv)
            .filter(|&i| {
                let v = mesh.vertex(i);
                v[0].abs() < 1e-9 || v[1].abs() < 1e-9
                    || (v[0] - 1.0).abs() < 1e-9 || (v[1] - 1.0).abs() < 1e-9
            })
            .collect();
        assert!(!edge.is_empty(), "grid has a boundary");
        let solver = PoissonSolver::with_dirichlet(&ops, &edge).unwrap();
        let (built, shift) = solver.preconditioner();
        assert!(built, "no incomplete Cholesky was built");

        let rhs = DVector::from_fn(nv, |i, _| (i as f64 * 0.37).sin());
        let (x_ic, its_ic) = solver.solve_from(&rhs, None, 1e-10);

        // The same system, driven by the diagonal alone.
        let closed = false;
        let mut b: Vec<f64> = (0..nv).map(|i| -solver.star0[i] * rhs[i]).collect();
        for &d in &edge {
            b[d] = 0.0;
        }
        let jacobi: Preconditioner = Box::new(|r: &[f64]| {
            (0..nv).map(|i| solver.inv_diag[i] * r[i]).collect()
        });
        let (mut x_j, its_j) = pcg_solve_from_pc(
            |p| solver.apply_a(p),
            &jacobi,
            &b,
            nv,
            closed,
            None,
            1e-10,
        );
        for &d in &edge {
            x_j[d] = 0.0;
        }

        // Same answer, to the tolerance both were asked for.
        let num: f64 = (0..nv).map(|i| (x_ic[i] - x_j[i]).powi(2)).sum::<f64>().sqrt();
        let den: f64 = (0..nv).map(|i| x_j[i] * x_j[i]).sum::<f64>().sqrt().max(1e-300);
        assert!(num / den < 1e-6, "the two preconditioners disagree by {}", num / den);

        // And fewer iterations. Two triangular solves cost about as much as the
        // matvec, so the count has to fall by more than a factor of two before
        // the factorisation pays for itself at all.
        assert!(
            (its_ic as f64) * 2.0 < its_j as f64,
            "incomplete Cholesky took {its_ic} iterations against Jacobi's {its_j}, shift {shift}"
        );
    }

    #[test]
    fn poisson_zero_rhs() {
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let rhs = DVector::zeros(mesh.n_vertices());
        let x = solve_poisson(&ops, &rhs).unwrap();
        assert!(x.norm() < 1e-12, "zero rhs should give zero solution");
    }

    #[test]
    fn poisson_self_consistency() {
        // Verify that solve(-Delta, rhs) gives x such that -Delta x = rhs
        // (up to the constant kernel). This tests the solver's internal
        // consistency without depending on boundary treatment.
        let n = 8;
        let mesh = FlatMesh::unit_square_grid(n);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();

        // Arbitrary zero-mean rhs.
        let mut rhs = DVector::from_fn(nv, |i, _| (i as f64 * 0.37).sin());
        let mean = rhs.sum() / nv as f64;
        for i in 0..nv {
            rhs[i] -= mean;
        }

        let x = solve_poisson(&ops, &rhs).unwrap();

        // Check: solve returns psi with -apply_laplace_beltrami(psi) = rhs (Delta psi = rhs).
        let neg_lap_x = -ops.apply_laplace_beltrami(&x);
        let mean_r = neg_lap_x.sum() / nv as f64;
        let mean_b = rhs.sum() / nv as f64;
        let residual_zm: DVector<f64> = (&neg_lap_x - &DVector::from_element(nv, mean_r))
            - (&rhs - &DVector::from_element(nv, mean_b));

        // Tolerance is 15% because unit_square_grid has physical boundaries
        // where the one-sided DEC stencil introduces discretisation error.
        // On periodic meshes (the actual Stokes target), this error vanishes.
        let rel_err = residual_zm.norm() / rhs.norm();
        assert!(
            rel_err < 0.15,
            "Poisson self-consistency: relative residual = {rel_err} (expected < 0.15)"
        );
    }

    #[test]
    fn poisson_solver_reuse() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();

        let rhs = DVector::from_fn(nv, |i, _| (i as f64 * 0.1).sin());

        let x_oneshot = solve_poisson(&ops, &rhs).unwrap();
        let solver = PoissonSolver::new(&ops).unwrap();
        let x_reuse = solver.solve(&rhs);

        let diff = (&x_oneshot - &x_reuse).norm();
        assert!(
            diff < 1e-12,
            "reuse solver should match one-shot: diff = {diff}"
        );
    }

    /// Dirichlet Poisson: solve -Delta psi = f with psi=0 on all boundary
    /// vertices of a unit_square_grid mesh.  Asserts that:
    /// (a) solution is exactly 0 on every boundary vertex (< 1e-10),
    /// (b) the interior solution is non-trivial for a non-zero RHS.
    #[test]
    fn dirichlet_poisson_boundary_is_zero() {
        use cartan_dec::mesh::FlatMesh;

        // Use a 6x6 grid (36 vertices); boundary = outer ring.
        let n = 6_usize;
        let mesh = FlatMesh::unit_square_grid(n);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();

        // Identify boundary vertices: those on x=0, x=1, y=0, or y=1.
        // unit_square_grid places vertex (i,j) at index i*(n+1)+j with
        // coordinates (i/n, j/n). Boundary = i==0 || i==n || j==0 || j==n.
        let boundary_vertices: Vec<usize> = (0..nv)
            .filter(|&k| {
                let i = k / (n + 1);
                let j = k % (n + 1);
                i == 0 || i == n || j == 0 || j == n
            })
            .collect();

        assert!(
            !boundary_vertices.is_empty(),
            "boundary should be non-empty for unit_square_grid(6)"
        );
        let n_interior = nv - boundary_vertices.len();
        assert!(n_interior > 0, "interior should be non-empty");

        // Smooth non-zero RHS: sin(pi*x)*sin(pi*y) (zero on all boundary).
        let rhs = DVector::from_fn(nv, |k, _| {
            let i = k / (n + 1);
            let j = k % (n + 1);
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
        });

        let solver = PoissonSolver::with_dirichlet(&ops, &boundary_vertices)
            .expect("Dirichlet Poisson solver should construct");
        let psi = solver.solve(&rhs);

        // (a) Boundary vertices must have psi ≈ 0.
        let max_boundary = boundary_vertices.iter()
            .map(|&bv| psi[bv].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_boundary < 1e-10,
            "psi on boundary should be ~0, got max = {max_boundary:.3e}"
        );

        // (b) Interior solution should be non-trivial.
        let boundary_set: std::collections::HashSet<usize> =
            boundary_vertices.iter().cloned().collect();
        let max_interior = (0..nv)
            .filter(|i| !boundary_set.contains(i))
            .map(|i| psi[i].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_interior > 1e-6,
            "interior psi should be non-trivial, got max = {max_interior:.3e}"
        );
    }
}
