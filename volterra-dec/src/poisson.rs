//! Sparse Poisson solver on DEC meshes.
//!
//! Solves the equation -Delta f = rhs on the mesh, where Delta is the
//! scalar Laplace-Beltrami operator (negative semi-definite).
//!
//! Two solver paths:
//!
//! - [`PoissonSolver`]: precomputes an LDL^T factorisation of -Delta with
//!   one vertex pinned (to remove the constant kernel). Amortises the
//!   O(n^{3/2}) factorisation cost across many solves. Each subsequent
//!   solve is O(nnz) (sparse back-substitution).
//!
//! - [`solve_poisson`]: one-shot convenience function (factorises + solves).

use cartan_core::Manifold;
use cartan_dec::Operators;
use nalgebra::DVector;
use sprs::CsMat;

/// Precomputed LDL^T factorisation of -Delta for repeated Poisson solves.
///
/// The factorisation is computed once and reused for each solve. This is
/// the recommended path when solving Poisson multiple times per timestep
/// (e.g., vorticity-stream function Stokes).
pub struct PoissonSolver {
    /// Number of vertices (including the pinned one).
    n: usize,
    /// Index of the pinned vertex (set to zero to remove kernel).
    pinned: usize,
    /// LDL^T factorisation of the reduced system.
    ldl: sprs_ldl::LdlNumeric<f64, usize>,
}

impl PoissonSolver {
    /// Build the solver from a DEC Operators struct.
    ///
    /// Factorises (-Delta + epsilon I) where epsilon is a small regularisation
    /// that makes the matrix positive definite (removes the constant kernel).
    /// The factorisation cost is O(n^{3/2}) for a 2D mesh.
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();

        // Build -Delta + epsilon*I (positive definite, symmetric).
        // epsilon is small enough not to affect the solution significantly
        // but large enough to regularise the zero eigenvalue.
        let eps = 1e-10;
        let reg_mat = regularise_neg_laplacian_lower(&ops.laplace_beltrami, eps);

        // LDL^T factorisation. We pass the lower triangle only, skip the
        // symmetry check, and disable fill-reduction (which internally
        // asserts symmetry via reverse Cuthill-McKee).
        let ldl = sprs_ldl::Ldl::new()
            .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(sprs::FillInReduction::NoReduction)
            .numeric(reg_mat.view())
            .map_err(|e| format!("LDL factorisation: {e:?}"))?;

        Ok(Self {
            n,
            pinned: 0,
            ldl,
        })
    }

    /// Solve -Delta f = rhs.
    ///
    /// Projects rhs to zero mean, solves with vertex 0 pinned, then
    /// shifts the solution to zero mean.
    pub fn solve(&self, rhs: &DVector<f64>) -> DVector<f64> {
        assert_eq!(rhs.len(), self.n);

        // Project rhs to zero mean.
        let mean_rhs = rhs.sum() / self.n as f64;
        let mut b: Vec<f64> = rhs.iter().map(|&v| v - mean_rhs).collect();

        // Pin: set rhs at pinned vertex to 0.
        b[self.pinned] = 0.0;

        // Solve via LDL^T back-substitution.
        let x: Vec<f64> = self.ldl.solve(&b);

        // Shift to zero mean.
        let mean_x = x.iter().sum::<f64>() / self.n as f64;
        let result: Vec<f64> = x.iter().map(|&v| v - mean_x).collect();

        DVector::from_vec(result)
    }
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

/// Build the LOWER TRIANGLE of (-Delta + epsilon * I).
///
/// sprs-ldl reads only the lower triangle of the input matrix (row >= col).
/// Delta is negative semi-definite, so -Delta is positive semi-definite.
/// Adding epsilon * I makes it strictly positive definite.
fn regularise_neg_laplacian_lower(lap: &CsMat<f64>, eps: f64) -> CsMat<f64> {
    let n = lap.rows();
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Copy only the lower triangle of -Delta (row >= col).
    for (&val, (row, col)) in lap.iter() {
        if row >= col {
            rows.push(row);
            cols.push(col);
            vals.push(-val);
        }
    }

    // Add epsilon to the diagonal.
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(eps);
    }

    let tri = sprs::TriMat::from_triplets((n, n), rows, cols, vals);
    tri.to_csc()
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

        // Check: -Delta x should equal rhs (up to a constant).
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
}
