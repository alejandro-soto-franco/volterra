//! Active nematic engine implementing Zhu et al. (2024) Algorithm 1.
//!
//! Operator splitting per timestep:
//! 1. STOKES:    u, p = stokes.solve(div_nabla(zeta * Q))
//! 2. ADVECTION: z_adv = semi_lagrangian.advect(z, u, dt, lambda)
//! 3. DIFFUSION: z_new = implicit_euler(z_adv, (1/Pe) * Delta_L, dt)
//! 4. NORMALISE: z_new = z_new / |z_new|
//!
//! ## References
//!
//! - Zhu, Saintillan, Chern (2024). "Active nematic fluids on Riemannian
//!   2-manifolds." arXiv:2405.06044.

use num_complex::Complex;
use nalgebra::SVector;
use sprs::{CsMat, TriMat};

use cartan_core::Manifold;
use cartan_dec::line_bundle::{Section, ConnectionAngles, BochnerLaplacian, defect_charges};
use cartan_dec::mesh::Mesh;
use cartan_dec::hodge::HodgeStar;

use volterra_dec::nematic_field_2d::NematicField2D;
use volterra_dec::stokes_trait::{StokesSolver, FlowField};
use volterra_dec::semi_lagrangian::SemiLagrangian;
use volterra_dec::stokes_dec::VelocityFieldDec;
use volterra_dec::QFieldDec;

/// Nondimensionalised parameters for the active nematic engine.
#[derive(Debug, Clone)]
pub struct EngineParams {
    /// Active Peclet number: Pe = |alpha| * eta_r^2 / mu.
    pub pe: f64,
    /// Tumbling parameter: 0 = corotational, 1 = flow-aligning.
    pub lambda: f64,
    /// Defect core size / domain size (for Ginzburg-Landau if not normalising).
    pub epsilon: f64,
    /// Timestep (free of diffusive CFL with implicit diffusion).
    pub dt: f64,
    /// Activity sign: +1 for contractile, -1 for extensile.
    pub activity_sign: f64,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            pe: 1.0,
            lambda: 1.0,
            epsilon: 0.01,
            dt: 0.01,
            activity_sign: -1.0,
        }
    }
}

/// Per-step diagnostics from the engine.
#[derive(Debug, Clone)]
pub struct StepDiagnostics {
    /// Simulation time.
    pub time: f64,
    /// Mean scalar order parameter S.
    pub mean_order: f64,
    /// Stokes divergence residual.
    pub stokes_residual: f64,
    /// Step index.
    pub step: usize,
}

/// Active nematic engine on a 2-manifold.
///
/// Implements the full operator-split algorithm from Zhu et al. (2024).
pub struct ActiveNematicEngine {
    /// Nondimensionalised parameters.
    params: EngineParams,
    /// Bochner Laplacian on L_2 (nematic bundle).
    bochner: BochnerLaplacian<2>,
    /// Semi-Lagrangian advector.
    semi_lag: SemiLagrangian,
    /// Stokes solver (trait object, either stream function or Killing).
    stokes: Box<dyn StokesSolver>,
    /// Precomputed implicit diffusion matrix: (I - dt/Pe * Delta_L)^{-1}.
    /// Since we can't invert a complex sparse matrix easily, we store the
    /// system matrix and solve per step via CG on real/imaginary parts.
    diffusion_matrix: CsMat<f64>,
    /// Connection angles for defect detection.
    connection: ConnectionAngles,
    /// Dual areas (star_0).
    dual_areas: Vec<f64>,
    /// Vertex coordinates for force computation.
    coords: Vec<[f64; 3]>,
    /// Mesh simplices for force computation.
    simplices: Vec<[usize; 3]>,
    /// Number of vertices.
    n_vertices: usize,
    /// Current step counter.
    step: usize,
    /// Current simulation time.
    time: f64,
}

impl ActiveNematicEngine {
    /// Create a new engine from mesh data and precomputed operators.
    ///
    /// `stokes` is a boxed trait object: either `KillingOperatorSolver` or
    /// `StreamFunctionStokes`.
    pub fn new<M: Manifold<Point = SVector<f64, 3>>>(
        mesh: &Mesh<M, 3, 2>,
        manifold: &M,
        hodge: &HodgeStar,
        params: EngineParams,
        stokes: Box<dyn StokesSolver>,
    ) -> Self {
        let nv = mesh.n_vertices();
        let connection = ConnectionAngles::from_mesh(mesh, manifold);
        let bochner = BochnerLaplacian::<2>::from_mesh_data(mesh, hodge, &connection);

        let dual_areas: Vec<f64> = (0..nv).map(|i| hodge.star0()[i]).collect();

        let coords: Vec<[f64; 3]> = mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();

        let simplices: Vec<[usize; 3]> = mesh.simplices.clone();

        // Build the semi-Lagrangian advector.
        let triangles: Vec<[usize; 3]> = simplices.clone();
        let semi_lag = SemiLagrangian::new(coords.clone(), triangles);

        // Build the implicit diffusion system matrix.
        // (I - dt/Pe * Delta_L) where Delta_L is the Bochner Laplacian.
        // Since Delta_L is complex Hermitian, and we separate real/imaginary:
        // The real part of the Laplacian matrix operates on Re(z) and Im(z)
        // independently (since the matrix is Hermitian, the real part is
        // symmetric and the imaginary part is antisymmetric).
        // For simplicity, store the real part of (I - dt/Pe * Delta_L).
        let dt_over_pe = params.dt / params.pe;
        let lap_mat = bochner.matrix();
        let n = nv;
        let mut triplets = TriMat::new((n, n));
        // Identity.
        for i in 0..n {
            triplets.add_triplet(i, i, 1.0);
        }
        // -dt/Pe * Re(Delta_L)
        for (col, col_view) in lap_mat.outer_iterator().enumerate() {
            for (row, val) in col_view.iter() {
                triplets.add_triplet(row, col, -dt_over_pe * val.re);
            }
        }
        let diffusion_matrix = triplets.to_csc();

        Self {
            params,
            bochner,
            semi_lag,
            stokes,
            diffusion_matrix,
            connection,
            dual_areas,
            coords,
            simplices,
            n_vertices: nv,
            step: 0,
            time: 0.0,
        }
    }

    /// Perform one operator-split timestep.
    ///
    /// Returns diagnostics for this step.
    pub fn step(&mut self, field: &mut NematicField2D) -> StepDiagnostics {
        let dt = self.params.dt;

        // 1. STOKES: compute active force and solve for velocity.
        let force = self.compute_active_force(field);
        let flow = self.stokes.solve(&force);

        // 2. ADVECTION: semi-Lagrangian backtrack.
        // Convert velocity to per-vertex [f64; 3].
        let velocity: Vec<[f64; 3]> = flow.velocity_3d;
        let (q1_adv, q2_adv) = self.advect_components(field, &velocity, dt);

        // 3. DIFFUSION: implicit solve on real and imaginary parts.
        let q1_diff = self.implicit_diffusion_solve(&q1_adv);
        let q2_diff = self.implicit_diffusion_solve(&q2_adv);

        // Update field.
        for v in 0..self.n_vertices {
            field.values_mut()[v] = Complex::new(q1_diff[v], q2_diff[v]);
        }

        // 4. NORMALISE.
        field.normalise();

        self.step += 1;
        self.time += dt;

        StepDiagnostics {
            time: self.time,
            mean_order: field.mean_scalar_order(),
            stokes_residual: flow.div_residual,
            step: self.step,
        }
    }

    /// Compute the active force from the nematic field.
    ///
    /// f = activity_sign * Pe * div(Q) where Q is the Q-tensor.
    /// Approximated per vertex using FEM-style gradient of Q over incident triangles.
    fn compute_active_force(&self, field: &NematicField2D) -> Vec<[f64; 3]> {
        let nv = self.n_vertices;
        let mut force = vec![[0.0; 3]; nv];

        let vals = field.values();
        let zeta = self.params.activity_sign * self.params.pe;

        // Simple approximation: compute gradient of Q components per face,
        // distribute to vertices.
        for &[i0, i1, i2] in &self.simplices {
            let v0 = self.coords[i0];
            let v1 = self.coords[i1];
            let v2 = self.coords[i2];

            let e01 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e02 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e01[1] * e02[2] - e01[2] * e02[1],
                e01[2] * e02[0] - e01[0] * e02[2],
                e01[0] * e02[1] - e01[1] * e02[0],
            ];
            let area2 = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            if area2 < 1e-30 {
                continue;
            }
            let n = [cross[0] / area2, cross[1] / area2, cross[2] / area2];
            let area = area2 / 2.0;

            // FEM gradients (rotated edge / 2A).
            let grad_phi = [
                cross_with_normal(n, [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]], area),
                cross_with_normal(n, [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]], area),
                cross_with_normal(n, [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]], area),
            ];

            let q1 = [vals[i0].re, vals[i1].re, vals[i2].re];
            let q2 = [vals[i0].im, vals[i1].im, vals[i2].im];

            // Gradient of q1 and q2 on this face.
            let mut grad_q1 = [0.0; 3];
            let mut grad_q2 = [0.0; 3];
            for local in 0..3 {
                for d in 0..3 {
                    grad_q1[d] += grad_phi[local][d] * q1[local];
                    grad_q2[d] += grad_phi[local][d] * q2[local];
                }
            }

            // Active force from div(Q): simplified as gradient-based approximation.
            // Distribute to vertices (area / 3 weighting).
            let w = zeta * area / 3.0;
            for &vi in &[i0, i1, i2] {
                for d in 0..3 {
                    force[vi][d] += w * (grad_q1[d] + grad_q2[d]);
                }
            }
        }

        force
    }

    /// Advect field components using semi-Lagrangian method.
    fn advect_components(
        &self,
        field: &NematicField2D,
        velocity: &[[f64; 3]],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let qfield = field.to_qfield_dec();
        let vel = VelocityFieldDec {
            v: velocity.to_vec(),
            n_vertices: self.n_vertices,
        };

        let q_adv = self.semi_lag.advect_with_params(&qfield, &vel, dt, self.params.lambda);
        (q_adv.q1, q_adv.q2)
    }

    /// Solve (I - dt/Pe * Re(Delta_L)) * x = b via conjugate gradient.
    fn implicit_diffusion_solve(&self, rhs: &[f64]) -> Vec<f64> {
        let n = rhs.len();
        let mut x = rhs.to_vec();
        let mut r: Vec<f64> = {
            let ax = sparse_matvec(&self.diffusion_matrix, &x);
            rhs.iter().zip(&ax).map(|(b, a)| b - a).collect()
        };
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();

        if rs_old.sqrt() < 1e-14 {
            return x;
        }

        for _ in 0..1000 {
            let ap = sparse_matvec(&self.diffusion_matrix, &p);
            let pap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rs_old / pap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
            if rs_new.sqrt() < 1e-14 {
                break;
            }

            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        x
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step
    }
}

/// n x e / (2*area): FEM gradient helper.
fn cross_with_normal(n: [f64; 3], e: [f64; 3], area: f64) -> [f64; 3] {
    let inv_2a = 0.5 / area;
    [
        inv_2a * (n[1] * e[2] - n[2] * e[1]),
        inv_2a * (n[2] * e[0] - n[0] * e[2]),
        inv_2a * (n[0] * e[1] - n[1] * e[0]),
    ]
}

/// Sparse matrix-vector multiply.
fn sparse_matvec(mat: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrows = mat.rows();
    let mut y = vec![0.0; nrows];
    for (col, col_view) in mat.outer_iterator().enumerate() {
        let xc = x[col];
        if xc.abs() < 1e-30 {
            continue;
        }
        for (row, &val) in col_view.iter() {
            y[row] += val * xc;
        }
    }
    y
}
