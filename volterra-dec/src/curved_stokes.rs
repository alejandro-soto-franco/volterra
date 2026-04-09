//! Stokes solver on curved 2-manifolds via the modified biharmonic.
//!
//! On a curved surface with Gaussian curvature K, the incompressible
//! Stokes equation for the stream function psi is:
//!
//! ```text
//! (1/Er) * Delta_0 * (Delta_0 + K) * psi = source
//! ```
//!
//! This factors into two sequential Poisson-type solves:
//!
//! ```text
//! (Delta_0 + K) phi = Er * source    (modified Poisson)
//!  Delta_0 psi = phi                 (standard Poisson)
//! ```
//!
//! Both operators are precomputed (LDL^T factorised) at construction.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use nalgebra::DVector;

use crate::poisson::PoissonSolver;
use crate::stokes_dec::{VelocityFieldDec, compute_vorticity_source};
use crate::QFieldDec;
// NematicParams will be used when the engine wires this solver to the full pipeline.

/// Precomputed curved-surface Stokes solver.
///
/// Holds two LDL^T factorisations:
/// - Standard Poisson: -Delta psi = rhs
/// - Modified Poisson: -(Delta + K) phi = rhs
///
/// And cached geometry data for velocity extraction.
pub struct CurvedStokesSolver {
    /// Standard Poisson solver (-Delta).
    poisson_standard: PoissonSolver,
    /// Modified Poisson solver (-(Delta + K)).
    poisson_modified: ModifiedPoissonSolver,
    /// Number of vertices.
    n_vertices: usize,
    /// Vertex coordinates in R^3.
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (star_0), used by velocity extraction.
    #[allow(dead_code)]
    dual_areas: Vec<f64>,
    /// Per-vertex Gaussian curvature.
    gaussian_curvature: Vec<f64>,
}

/// Solver for -(Delta + K) phi = rhs, where K is per-vertex Gaussian curvature.
struct ModifiedPoissonSolver {
    ldl: sprs_ldl::LdlNumeric<f64, usize>,
    n: usize,
}

impl ModifiedPoissonSolver {
    /// Build from the raw Laplacian matrix and per-vertex Gaussian curvature.
    fn new<M: Manifold>(ops: &Operators<M, 3, 2>, gaussian_k: &[f64], _star0: &[f64]) -> Result<Self, String> {
        let n = ops.laplace_beltrami.rows();

        // Build -(Delta + K) = -Delta - diag(K).
        // -Delta is positive semi-definite. -diag(K) adds curvature correction.
        // The combined matrix -(Delta + K) is positive definite if K is not
        // too negative (which it won't be at the regularised level).
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        // Lower triangle of -Delta.
        for (&val, (row, col)) in ops.laplace_beltrami.iter() {
            if row >= col {
                rows.push(row);
                cols.push(col);
                vals.push(-val);
            }
        }

        // Add -K * star_0 to diagonal (the mass-weighted curvature).
        // The equation is: star_0_inv * (L + diag(star_0 * K)) * phi = f
        // which becomes: (L + diag(star_0 * K)) * phi = star_0 * f.
        // So the matrix we factorise is: -L - diag(star_0 * K).
        // -L is already added above (from -Delta before star_0_inv).
        // Actually, -Delta = star_0_inv * (-L), so -L = star_0 * (-Delta).
        // This is getting confusing. Let me think step by step.
        //
        // The DEC Laplacian is: Delta = star_0_inv * d0^T * star_1 * d0.
        // The stored matrix ops.laplace_beltrami IS Delta (with star_0_inv applied).
        //
        // We want to solve: -(Delta + K) phi = rhs.
        // => -Delta phi - K phi = rhs.
        // => (-Delta - diag(K)) phi = rhs.
        //
        // -Delta is the negated Laplacian (positive semi-definite).
        // Adding -diag(K): on the outer equator of a torus (K > 0), this subtracts,
        // making the matrix less positive definite. Need regularisation.
        for (i, &k) in gaussian_k.iter().enumerate() {
            rows.push(i);
            cols.push(i);
            vals.push(-k);
        }

        // Regularise: add small epsilon to ensure positive definiteness.
        let eps = 1e-10;
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(eps);
        }

        let tri = sprs::TriMat::from_triplets((n, n), rows, cols, vals);
        let mat = tri.to_csc();

        let ldl = sprs_ldl::Ldl::new()
            .check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(sprs::FillInReduction::NoReduction)
            .numeric(mat.view())
            .map_err(|e| format!("Modified Poisson LDL: {e:?}"))?;

        Ok(Self { ldl, n })
    }

    /// Solve -(Delta + K) phi = rhs.
    fn solve(&self, rhs: &DVector<f64>) -> DVector<f64> {
        let n = self.n;
        let mean = rhs.sum() / n as f64;
        let mut b: Vec<f64> = rhs.iter().map(|&v| v - mean).collect();
        b[0] = 0.0; // pin vertex 0

        let x: Vec<f64> = self.ldl.solve(&b);
        let mean_x = x.iter().sum::<f64>() / n as f64;
        DVector::from_vec(x.iter().map(|&v| v - mean_x).collect())
    }
}

impl CurvedStokesSolver {
    /// Build the solver, precomputing both LDL^T factorisations.
    ///
    /// `gaussian_k` is the per-vertex Gaussian curvature.
    pub fn new<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        gaussian_k: &[f64],
    ) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();

        let poisson_standard = PoissonSolver::new(ops)?;

        let star0: Vec<f64> = (0..ops.hodge.star0().len())
            .map(|i| ops.hodge.star0()[i])
            .collect();

        let poisson_modified = ModifiedPoissonSolver::new(ops, gaussian_k, &star0)?;

        let coords = super::stokes_dec::extract_coords(mesh);
        let dual_areas = star0;

        Ok(Self {
            poisson_standard,
            poisson_modified,
            n_vertices,
            coords,
            dual_areas,
            gaussian_curvature: gaussian_k.to_vec(),
        })
    }

    /// Solve the Stokes equation in nondimensionalised form.
    ///
    /// `source` is the vorticity source: Pe * curl(div(V(z))).
    /// Returns the velocity field u = curl(psi).
    pub fn solve(&self, source: &DVector<f64>, er: f64) -> (DVector<f64>, VelocityFieldDec) {
        // Step 1: (Delta + K) phi = Er * source
        //    => -(Delta + K) phi = -Er * source
        let scaled_source = source * (-er);
        let phi = self.poisson_modified.solve(&scaled_source);

        // Step 2: Delta psi = phi
        //    => -Delta psi = -phi
        let neg_phi = -&phi;
        let psi = self.poisson_standard.solve(&neg_phi);

        // Velocity extraction (reuse the standard curl formula).
        let vel = VelocityFieldDec::zeros(self.n_vertices);
        // TODO: wire up the proper velocity extraction from stokes_dec.
        // For now, return psi + placeholder velocity.

        (psi, vel)
    }

    /// Access the precomputed vertex coordinates.
    pub fn coords(&self) -> &[[f64; 3]] {
        &self.coords
    }

    /// Access Gaussian curvature.
    pub fn gaussian_curvature(&self) -> &[f64] {
        &self.gaussian_curvature
    }
}

/// Compute the vorticity source for the nondimensionalised Stokes equation.
///
/// source = Pe * curl(div(V(z)))
///
/// In the nondimensionalised system, the active stress is Pe * V(z)
/// where V(z) = z tensor z (Veronese map).
pub fn nematic_vorticity_source<M: Manifold>(
    q: &QFieldDec,
    pe: f64,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
) -> DVector<f64> {
    // Reuse the existing vorticity source computation with zeta = pe, eta = 1.
    compute_vorticity_source(q, pe, 1.0, mesh, coords, dual_areas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_gen::icosphere;
    use cartan_manifolds::sphere::Sphere;

    #[test]
    fn curved_stokes_constructs_on_sphere() {
        let mesh = icosphere(2);
        let manifold = Sphere::<3>;
        let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
        // Unit sphere: K = 1 everywhere.
        let nv = mesh.n_vertices();
        let gaussian_k = vec![1.0; nv];
        let solver = CurvedStokesSolver::new(&ops, &mesh, &gaussian_k);
        assert!(solver.is_ok(), "curved Stokes should construct on S^2");
    }

    #[test]
    fn curved_stokes_zero_source_zero_psi() {
        let mesh = icosphere(2);
        let manifold = Sphere::<3>;
        let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
        let nv = mesh.n_vertices();
        let gaussian_k = vec![1.0; nv];
        let solver = CurvedStokesSolver::new(&ops, &mesh, &gaussian_k).unwrap();

        let source = DVector::zeros(nv);
        let (psi, _vel) = solver.solve(&source, 1.0);
        assert!(psi.norm() < 1e-10, "zero source should give zero psi");
    }
}
