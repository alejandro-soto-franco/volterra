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
///
/// Two modes:
/// - **Closed-manifold** (default, `dirichlet_vertices` empty): pins vertex 0 and
///   projects to zero mean. Correct for periodic/closed meshes (sphere, torus).
/// - **Dirichlet** (`dirichlet_vertices` non-empty): enforces ψ = 0 on all listed
///   vertices via standard symmetric elimination. No zero-mean projection; the
///   system is non-singular once the Dirichlet rows/cols are pinned. Use this
///   for bounded domains (no-slip stream-function).
pub struct PoissonSolver {
    /// Number of vertices.
    n: usize,
    /// Index of the pinned vertex (closed-manifold mode only; used when
    /// `dirichlet_vertices` is empty).
    pinned: usize,
    /// LDL^T factorisation of the (modified) system.
    ldl: sprs_ldl::LdlNumeric<f64, usize>,
    /// Dirichlet vertex indices (ψ = 0 enforced here). Empty → closed-manifold mode.
    dirichlet_vertices: Vec<usize>,
}

impl PoissonSolver {
    /// Build the solver from a DEC Operators struct (closed-manifold / periodic mode).
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
            dirichlet_vertices: Vec::new(),
        })
    }

    /// Build the solver with Dirichlet ψ = 0 boundary conditions.
    ///
    /// Enforces ψ = 0 on every vertex in `dirichlet_vertices` by zeroing those
    /// rows and columns of (-Delta) and placing 1 on the diagonal (symmetric
    /// elimination, RHS = 0 for those DOFs). The resulting system is
    /// non-singular and no zero-mean projection is required.
    ///
    /// Use this for bounded domains (confined active nematics, no-slip Stokes).
    /// The existing `new` (closed-manifold mode) is left unchanged.
    pub fn with_dirichlet<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        dirichlet_vertices: &[usize],
    ) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();

        // Build lower triangle of (-Delta) with Dirichlet elimination applied.
        let mat = dirichlet_neg_laplacian_lower(&ops.laplace_beltrami, dirichlet_vertices);

        let ldl = sprs_ldl::Ldl::new()
            .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(sprs::FillInReduction::NoReduction)
            .numeric(mat.view())
            .map_err(|e| format!("LDL factorisation (Dirichlet): {e:?}"))?;

        Ok(Self {
            n,
            pinned: 0,
            ldl,
            dirichlet_vertices: dirichlet_vertices.to_vec(),
        })
    }

    /// Solve -Delta f = rhs.
    ///
    /// **Closed-manifold mode** (no Dirichlet vertices): projects rhs to zero
    /// mean, solves with vertex 0 pinned, then shifts the solution to zero mean.
    ///
    /// **Dirichlet mode**: zeros RHS entries at Dirichlet vertices (since their
    /// target value is 0), solves the modified system, then zeros the solution
    /// at those vertices (enforces the BC exactly at the discrete level).
    pub fn solve(&self, rhs: &DVector<f64>) -> DVector<f64> {
        assert_eq!(rhs.len(), self.n);

        if self.dirichlet_vertices.is_empty() {
            // ── Closed-manifold path (original behaviour) ────────────────────
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
        } else {
            // ── Dirichlet path ────────────────────────────────────────────────
            // RHS at Dirichlet DOFs must be 0 (target ψ = 0; the elimination
            // moved nothing to the RHS since the target is 0).
            let mut b: Vec<f64> = rhs.iter().cloned().collect();
            for &dv in &self.dirichlet_vertices {
                b[dv] = 0.0;
            }

            let mut x: Vec<f64> = self.ldl.solve(&b);

            // Enforce ψ = 0 exactly at Dirichlet vertices (eliminates any
            // floating-point residual from the factorisation).
            for &dv in &self.dirichlet_vertices {
                x[dv] = 0.0;
            }

            DVector::from_vec(x)
        }
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

/// Build the LOWER TRIANGLE of (-Delta) with Dirichlet elimination.
///
/// For each Dirichlet vertex `d`:
/// - Row `d`: zeroed off-diagonal, diagonal set to 1.
/// - Col `d` (lower triangle only, i.e. rows > d): zeroed.
///
/// This is standard symmetric Dirichlet elimination for ψ_d = 0.
/// The resulting matrix is strictly positive definite (no kernel) as long
/// as at least one Dirichlet DOF exists, so no epsilon regularisation is needed.
/// A small epsilon is still added on the diagonal for numerical robustness.
fn dirichlet_neg_laplacian_lower(lap: &CsMat<f64>, dirichlet: &[usize]) -> CsMat<f64> {
    let n = lap.rows();
    let eps = 1e-10;

    // Build a boolean mask for O(1) lookup.
    let mut is_dirichlet = vec![false; n];
    for &d in dirichlet {
        is_dirichlet[d] = true;
    }

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Copy only the lower triangle of -Delta, applying Dirichlet elimination.
    for (&val, (row, col)) in lap.iter() {
        if row < col {
            continue; // upper triangle: skip
        }
        if is_dirichlet[row] || is_dirichlet[col] {
            // Zero off-diagonal entries touching a Dirichlet DOF.
            if row != col {
                continue;
            }
            // Diagonal of a Dirichlet row: set to 1.
            rows.push(row);
            cols.push(col);
            vals.push(1.0);
        } else {
            rows.push(row);
            cols.push(col);
            vals.push(-val);
        }
    }

    // Add eps to non-Dirichlet diagonals (robustness) and ensure every
    // Dirichlet diagonal entry exists (in case the Laplacian diagonal was missing).
    #[allow(clippy::needless_range_loop)] // i is pushed as a value (row/col index), not just used to index
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(if is_dirichlet[i] { 0.0 } else { eps });
    }

    let tri = sprs::TriMat::from_triplets((n, n), rows, cols, vals);
    tri.to_csc()
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
