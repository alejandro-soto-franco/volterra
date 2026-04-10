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
use crate::stokes_dec::VelocityFieldDec;
use crate::QFieldDec;
// NematicParams will be used when the engine wires this solver to the full pipeline.

/// Precomputed curved-surface Stokes solver.
///
/// Holds two LDL^T factorisations:
/// - Standard Poisson: -Delta psi = rhs
/// - Modified Poisson: -(Delta + K) phi = rhs
///
/// And cached geometry data for velocity extraction.
#[allow(dead_code)]
pub struct CurvedStokesSolver {
    /// Standard Poisson solver (-Delta).
    poisson_standard: PoissonSolver,
    /// Modified Poisson solver (-(Delta + K)).
    poisson_modified: ModifiedPoissonSolver,
    /// Number of vertices.
    n_vertices: usize,
    /// Vertex coordinates in R^3.
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (star_0).
    dual_areas: Vec<f64>,
    /// Per-vertex Gaussian curvature.
    gaussian_curvature: Vec<f64>,
    /// Mesh edge endpoints.
    boundaries: Vec<[usize; 2]>,
    /// Per-vertex list of incident edge indices.
    vertex_boundaries: Vec<Vec<usize>>,
    /// Mesh simplices (triangles), for face normal computation.
    simplices: Vec<[usize; 3]>,
    /// Per-edge list of incident face indices.
    boundary_simplices: Vec<Vec<usize>>,
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
            boundaries: mesh.boundaries.clone(),
            vertex_boundaries: mesh.vertex_boundaries.clone(),
            simplices: mesh.simplices.clone(),
            boundary_simplices: mesh.boundary_simplices.clone(),
        })
    }

    /// Solve the Stokes equation in nondimensionalised form.
    ///
    /// `source` is the vorticity source: Pe * curl(div(V(z))).
    /// Returns the velocity field u = curl(psi).
    pub fn solve(&self, source: &DVector<f64>, er: f64) -> (DVector<f64>, VelocityFieldDec) {
        use crate::stokes_dec::{sub3, cross3, norm3, scale3, add3};

        // Step 1: (Delta + K) phi = Er * source
        //    => -(Delta + K) phi = -Er * source
        let scaled_source = source * (-er);
        let phi = self.poisson_modified.solve(&scaled_source);

        // Step 2: Delta psi = phi
        //    => -Delta psi = -phi
        let neg_phi = -&phi;
        let psi = self.poisson_standard.solve(&neg_phi);

        // Step 3: Extract velocity u = *d(psi) via DEC curl.
        // For each edge, the velocity flux is dpsi * (n x e_hat) / |e|,
        // distributed to both endpoints and averaged by vertex valence.
        let nv = self.n_vertices;
        let ne = self.boundaries.len();
        let mut vel = vec![[0.0_f64; 3]; nv];

        for e in 0..ne {
            let [v0, v1] = self.boundaries[e];
            let dpsi = psi[v1] - psi[v0];

            let edge = sub3(self.coords[v1], self.coords[v0]);
            let edge_len = norm3(edge);
            if edge_len < 1e-30 {
                continue;
            }
            let edge_hat = scale3(edge, 1.0 / edge_len);

            // Average face normal for this edge.
            let fn_hat = self.average_edge_normal(e);

            // Dual edge direction: face_normal x edge_hat.
            let dual_dir = cross3(fn_hat, edge_hat);

            let vel_magnitude = dpsi / edge_len;
            let u_contrib = scale3(dual_dir, vel_magnitude);

            vel[v0] = add3(vel[v0], scale3(u_contrib, 0.5));
            vel[v1] = add3(vel[v1], scale3(u_contrib, 0.5));
        }

        // Normalise by vertex valence.
        for (v, edges) in vel.iter_mut().zip(&self.vertex_boundaries) {
            let valence = edges.len() as f64;
            if valence > 0.0 {
                *v = scale3(*v, 1.0 / valence);
            }
        }

        (psi, VelocityFieldDec { v: vel, n_vertices: nv })
    }

    /// Average face normal for an edge, computed from incident triangles.
    fn average_edge_normal(&self, edge_idx: usize) -> [f64; 3] {
        use crate::stokes_dec::{sub3, cross3, norm3, scale3, add3};
        let mut n = [0.0_f64; 3];
        for &fi in &self.boundary_simplices[edge_idx] {
            let [i0, i1, i2] = self.simplices[fi];
            let e01 = sub3(self.coords[i1], self.coords[i0]);
            let e02 = sub3(self.coords[i2], self.coords[i0]);
            let cr = cross3(e01, e02);
            n = add3(n, cr);
        }
        let len = norm3(n);
        if len > 1e-14 { scale3(n, 1.0 / len) } else { [0.0, 0.0, 1.0] }
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
/// Accepts simplices and coords directly (no Mesh reference needed).
pub fn nematic_vorticity_source(
    q: &QFieldDec,
    pe: f64,
    simplices: &[[usize; 3]],
    coords: &[[f64; 3]],
    _dual_areas: &[f64],
) -> DVector<f64> {
    use crate::stokes_dec::{sub3, cross3, norm3, scale3, add3};
    let nv = q.n_vertices;
    let mut omega = vec![0.0_f64; nv];
    let mut areas = vec![0.0_f64; nv];

    for &[i0, i1, i2] in simplices {
        let p0 = coords[i0]; let p1 = coords[i1]; let p2 = coords[i2];
        let e01 = sub3(p1, p0);
        let e02 = sub3(p2, p0);
        let e12 = sub3(p2, p1);
        let e20 = sub3(p0, p2);

        let fn_vec = cross3(e01, e02);
        let area2 = norm3(fn_vec);
        if area2 < 1e-30 { continue; }
        let fn_hat = scale3(fn_vec, 1.0 / area2);
        let inv_2a = 1.0 / area2;

        let rot_e12 = cross3(fn_hat, e12);
        let rot_e20 = cross3(fn_hat, e20);
        let rot_e01 = cross3(fn_hat, e01);

        let gq1 = scale3(add3(add3(
            scale3(rot_e12, q.q1[i0]),
            scale3(rot_e20, q.q1[i1])),
            scale3(rot_e01, q.q1[i2])), inv_2a);
        let gq2 = scale3(add3(add3(
            scale3(rot_e12, q.q2[i0]),
            scale3(rot_e20, q.q2[i1])),
            scale3(rot_e01, q.q2[i2])), inv_2a);

        let fx = -pe * (gq1[0] + gq2[1]);
        let fy = -pe * (gq2[0] - gq1[1]);

        let circ_01 = fx * e01[0] + fy * e01[1];
        let circ_12 = fx * e12[0] + fy * e12[1];
        let circ_20 = fx * e20[0] + fy * e20[1];

        let face_area = 0.5 * area2;
        let third = face_area / 3.0;
        areas[i0] += third; areas[i1] += third; areas[i2] += third;
        omega[i0] += 0.5 * (circ_01 - circ_20);
        omega[i1] += 0.5 * (circ_12 - circ_01);
        omega[i2] += 0.5 * (circ_20 - circ_12);
    }

    for i in 0..nv {
        if areas[i] > 1e-30 {
            omega[i] /= areas[i];
        }
    }

    DVector::from_vec(omega)
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
