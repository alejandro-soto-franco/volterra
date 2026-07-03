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
use sprs::CsMat;

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
    /// Jacobi preconditioner `1 / A_ii`, where `A` is `S` with identity rows on Dirichlet
    /// DOFs. One entry per vertex.
    inv_diag: Vec<f64>,
    /// Dual-area mass diagonal (star0), one per vertex; mass-weights the right-hand side.
    star0: Vec<f64>,
    /// Dirichlet vertex indices (ψ = 0 enforced here). Empty → closed-manifold mode.
    dirichlet_vertices: Vec<usize>,
    /// Boolean Dirichlet mask, length `n`.
    is_dirichlet: Vec<bool>,
}

impl PoissonSolver {
    /// Build the solver from a DEC Operators struct (closed-manifold / periodic mode).
    ///
    /// Assembles the symmetric SPD stiffness `S = diag(star0) * laplace_beltrami` and a
    /// Jacobi preconditioner. Solves are performed by preconditioned conjugate gradient
    /// (see [`Self::solve`]): a direct `LDL^T` factorisation is not used because the DEC
    /// stiffness on a general (non-well-centered) triangulation is not reliably factorised
    /// by a non-pivoting `LDL^T`, whereas CG needs only the SPD matvec.
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();
        let star0: Vec<f64> = ops.hodge.star0().iter().copied().collect();
        let s = full_stiffness(&ops.laplace_beltrami, &star0);
        let is_dirichlet = vec![false; n];
        let inv_diag = jacobi_inv_diag(&s, &is_dirichlet);
        Ok(Self { n, s, inv_diag, star0, dirichlet_vertices: Vec::new(), is_dirichlet })
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
        Ok(Self {
            n,
            s,
            inv_diag,
            star0,
            dirichlet_vertices: dirichlet_vertices.to_vec(),
            is_dirichlet,
        })
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
        let mut y = vec![0.0f64; self.n];
        for (&v, (r, c)) in self.s.iter() {
            y[r] += v * xz[c];
        }
        // Identity rows for Dirichlet DOFs.
        for (yi, (&d, &xi)) in y.iter_mut().zip(self.is_dirichlet.iter().zip(x)) {
            if d {
                *yi = xi;
            }
        }
        y
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
    pub fn solve(&self, rhs: &DVector<f64>) -> DVector<f64> {
        assert_eq!(rhs.len(), self.n);
        let closed = self.dirichlet_vertices.is_empty();

        // b = -M rhs (mass-weighted); S psi = -M rhs solves -L psi = rhs (Delta psi = rhs).
        let mut b: Vec<f64> = (0..self.n).map(|i| -self.star0[i] * rhs[i]).collect();
        if closed {
            // Project onto range(S) = {v : sum v = 0}.
            let mean = b.iter().sum::<f64>() / self.n as f64;
            for v in b.iter_mut() {
                *v -= mean;
            }
        } else {
            for &d in &self.dirichlet_vertices {
                b[d] = 0.0;
            }
        }

        let mut x = pcg_solve(|p| self.apply_a(p), &self.inv_diag, &b, self.n, closed);

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
        DVector::from_vec(x)
    }
}

/// Dot product of two slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Jacobi-preconditioned conjugate gradient for a symmetric operator `apply` with
/// preconditioner diagonal `inv_diag`, solving `A x = b`.
///
/// `project_kernel` keeps the residual and preconditioned residual mean-free, so the routine
/// converges to the minimum-norm solution when `A` is singular with the constant vector in
/// its kernel (closed-manifold Laplacian) OR when `A` is SPD only on the zero-mean subspace
/// (the shifted `(Delta + K)` operator, which is indefinite on constants but positive on the
/// zero-mean harmonics). Returns the raw iterate; callers apply any final gauge fix.
pub(crate) fn pcg_solve<F: Fn(&[f64]) -> Vec<f64>>(
    apply: F,
    inv_diag: &[f64],
    b: &[f64],
    n: usize,
    project_kernel: bool,
) -> Vec<f64> {
    let demean = |v: &mut [f64]| {
        let m = v.iter().sum::<f64>() / n as f64;
        for x in v.iter_mut() {
            *x -= m;
        }
    };

    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    if project_kernel {
        demean(&mut r);
    }
    let mut z: Vec<f64> = (0..n).map(|i| inv_diag[i] * r[i]).collect();
    if project_kernel {
        demean(&mut z);
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let bnorm = dot(b, b).sqrt().max(1e-300);
    let tol = 1e-10;
    let max_iter = 10 * n + 100;

    for _ in 0..max_iter {
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
        let mut z_new: Vec<f64> = (0..n).map(|i| inv_diag[i] * r[i]).collect();
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
    x
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
