//! Stokes solver trait and backends for active nematic simulations.
//!
//! Two backends:
//! - `StreamFunctionStokes`: modified biharmonic via stream function (2-manifolds only, fast).
//! - `KillingOperatorSolver`: augmented Lagrangian from cartan-dec (any dimension).

use nalgebra::{DVector, SVector};

use cartan_core::Manifold;
use cartan_dec::mesh::Mesh;
use cartan_dec::stokes::StokesSolverAL;
use cartan_dec::Operators;

use volterra_dec::curved_stokes::CurvedStokesSolver;
use volterra_dec::stokes_dec::{self, VelocityFieldDec};

/// Flow field result from a Stokes solve.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Velocity: R^3-valued per vertex.
    pub velocity_3d: Vec<[f64; 3]>,
    /// Divergence residual (only meaningful for Killing backend).
    pub div_residual: f64,
}

/// Which Stokes solver backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StokesBackend {
    /// Stream function formulation (2-manifolds only, direct solve).
    StreamFunction,
    /// Killing operator + augmented Lagrangian (any dimension, iterative).
    KillingOperator,
}

/// Trait for Stokes solver backends.
pub trait StokesSolver {
    /// Solve for the velocity field given a force per vertex.
    ///
    /// `force_3d` is R^3-valued per vertex: force_3d[v] = [fx, fy, fz].
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField;
}

/// Killing operator Stokes solver (wraps cartan-dec's StokesSolverAL).
pub struct KillingOperatorSolver {
    inner: StokesSolverAL,
    n_vertices: usize,
}

impl KillingOperatorSolver {
    /// Create a new Killing operator solver for a mesh.
    pub fn new<M: Manifold<Point = SVector<f64, 3>>>(
        mesh: &Mesh<M, 3, 2>,
        penalty: f64,
        tolerance: f64,
    ) -> Self {
        let nv = mesh.n_vertices();
        let inner = StokesSolverAL::new(mesh, penalty, tolerance, 100, 1000);
        Self { inner, n_vertices: nv }
    }
}

impl StokesSolver for KillingOperatorSolver {
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField {
        let nv = self.n_vertices;
        assert_eq!(force_3d.len(), nv);

        // Flatten to 3*nv vector.
        let mut force_flat = vec![0.0; 3 * nv];
        for v in 0..nv {
            force_flat[v * 3] = force_3d[v][0];
            force_flat[v * 3 + 1] = force_3d[v][1];
            force_flat[v * 3 + 2] = force_3d[v][2];
        }

        let result = self.inner.solve(&force_flat);

        // Unflatten velocity.
        let mut velocity_3d = vec![[0.0; 3]; nv];
        for v in 0..nv {
            velocity_3d[v][0] = result.velocity[v * 3];
            velocity_3d[v][1] = result.velocity[v * 3 + 1];
            velocity_3d[v][2] = result.velocity[v * 3 + 2];
        }

        FlowField {
            velocity_3d,
            div_residual: result.div_residual,
        }
    }
}

/// Stream function Stokes solver for 2-manifolds.
///
/// Converts the R^3 force field to a scalar vorticity source via discrete
/// curl, solves the modified biharmonic for the stream function psi, then
/// extracts velocity via the DEC curl u = *d(psi).
#[allow(dead_code)]
pub struct StreamFunctionStokes {
    inner: CurvedStokesSolver,
    n_vertices: usize,
    /// Ericksen number (viscosity ratio) for the biharmonic factorisation.
    er: f64,
    /// Mesh simplices (triangles).
    simplices: Vec<[usize; 3]>,
    /// Vertex coordinates.
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (barycentric, area/3 per incident triangle).
    dual_areas: Vec<f64>,
    /// Mesh edge endpoints.
    boundaries: Vec<[usize; 2]>,
    /// Per-vertex list of incident edge indices.
    vertex_boundaries: Vec<Vec<usize>>,
    /// Hodge star on 1-forms.
    star1: Vec<f64>,
}

impl StreamFunctionStokes {
    /// Build from mesh and precomputed operators.
    ///
    /// `er` is the Ericksen number (ratio of viscous to elastic stress).
    /// For the nondimensionalised system in Zhu et al. (2024), Er = 1.
    pub fn new<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        gaussian_k: &[f64],
        er: f64,
    ) -> Result<Self, String> {
        let n_vertices = mesh.n_vertices();
        let inner = CurvedStokesSolver::new(ops, mesh, gaussian_k)?;

        let coords = stokes_dec::extract_coords(mesh);

        // Compute barycentric dual areas.
        let mut dual_areas = vec![0.0_f64; n_vertices];
        for &[i0, i1, i2] in &mesh.simplices {
            let e01 = sub3(coords[i1], coords[i0]);
            let e02 = sub3(coords[i2], coords[i0]);
            let cr = cross3(e01, e02);
            let face_area = 0.5 * norm3(cr);
            let third = face_area / 3.0;
            dual_areas[i0] += third;
            dual_areas[i1] += third;
            dual_areas[i2] += third;
        }

        let s1 = ops.hodge.star1();
        let star1: Vec<f64> = (0..s1.len()).map(|i| s1[i]).collect();

        Ok(Self {
            inner,
            n_vertices,
            er,
            simplices: mesh.simplices.clone(),
            coords,
            dual_areas,
            boundaries: mesh.boundaries.clone(),
            vertex_boundaries: mesh.vertex_boundaries.clone(),
            star1,
        })
    }
}

impl StokesSolver for StreamFunctionStokes {
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField {
        let nv = self.n_vertices;
        assert_eq!(force_3d.len(), nv);

        // Step 1: Compute vorticity source = discrete curl of force field.
        // For each triangle, accumulate the circulation of f around the
        // boundary and distribute to vertices via dual area weighting.
        let omega = discrete_curl(force_3d, &self.simplices, &self.coords, &self.dual_areas, nv);

        // Step 2: Solve the modified biharmonic for stream function psi.
        let (psi, _) = self.inner.solve(&omega, self.er);

        // Step 3: Extract velocity from psi via DEC curl: u = *d(psi).
        let vel = velocity_from_psi(
            nv,
            &psi,
            &self.boundaries,
            &self.vertex_boundaries,
            &self.coords,
        );

        FlowField {
            velocity_3d: vel.v,
            div_residual: 0.0, // Stream function is divergence-free by construction.
        }
    }
}

/// Discrete curl of a vertex vector field on a triangle mesh.
///
/// Returns a scalar per vertex (vorticity), computed by accumulating
/// the circulation of f around each triangle and distributing to vertices
/// with dual area weighting.
fn discrete_curl(
    force: &[[f64; 3]],
    simplices: &[[usize; 3]],
    coords: &[[f64; 3]],
    dual_areas: &[f64],
    nv: usize,
) -> DVector<f64> {
    let mut omega = vec![0.0_f64; nv];

    for &[i0, i1, i2] in simplices {
        let p0 = coords[i0];
        let p1 = coords[i1];
        let p2 = coords[i2];

        let e01 = sub3(p1, p0);
        let e12 = sub3(p2, p1);
        let e20 = sub3(p0, p2);

        // Force at edge midpoints.
        let f01 = mid3(force[i0], force[i1]);
        let f12 = mid3(force[i1], force[i2]);
        let f20 = mid3(force[i2], force[i0]);

        // Circulation: f_mid . edge for each edge of the triangle.
        let circ_01 = dot3(f01, e01);
        let circ_12 = dot3(f12, e12);
        let circ_20 = dot3(f20, e20);

        // Total circulation = sum. Distribute 1/3 to each vertex.
        // This is the standard vertex-based dual cell integration of curl.
        omega[i0] += 0.5 * (circ_01 - circ_20);
        omega[i1] += 0.5 * (circ_12 - circ_01);
        omega[i2] += 0.5 * (circ_20 - circ_12);
    }

    // Normalise by dual area.
    for i in 0..nv {
        if dual_areas[i] > 1e-30 {
            omega[i] /= dual_areas[i];
        }
    }

    DVector::from_vec(omega)
}

/// Recover the 3D velocity field u = *d(psi) from the stream function.
///
/// For each edge, the velocity flux is dpsi * (n x e_hat) / |e|,
/// distributed to both endpoints and averaged by vertex valence.
fn velocity_from_psi(
    nv: usize,
    psi: &DVector<f64>,
    boundaries: &[[usize; 2]],
    vertex_boundaries: &[Vec<usize>],
    coords: &[[f64; 3]],
) -> VelocityFieldDec {
    let ne = boundaries.len();
    let mut vel = vec![[0.0_f64; 3]; nv];

    // Precompute average face normal per edge for the curl direction.
    // For interior edges this averages the two incident face normals;
    // for boundary edges it uses the single incident face normal.
    // We approximate by computing on-the-fly from incident triangles.
    // Since we don't have boundary_simplices here, use a simpler approach:
    // for each edge, the curl direction is face_normal x edge_hat.
    // We compute a consistent face normal from the edge neighbourhood.

    for e in 0..ne {
        let [v0, v1] = boundaries[e];
        let dpsi = psi[v1] - psi[v0];

        let edge = sub3(coords[v1], coords[v0]);
        let edge_len = norm3(edge);
        if edge_len < 1e-30 {
            continue;
        }
        let edge_hat = scale3(edge, 1.0 / edge_len);

        // Approximate face normal from the two endpoint positions and a
        // perpendicular direction. For a surface mesh we can get a good
        // normal by finding a third vertex that shares a face with this edge.
        // As a robust fallback, use the cross product of the edge with the
        // average position vector (works for convex surfaces like spheres).
        let mid = mid3(coords[v0], coords[v1]);
        let mid_len = norm3(mid);
        let approx_normal = if mid_len > 1e-14 {
            let n = scale3(mid, 1.0 / mid_len);
            // Ensure orthogonal to edge.
            let d = dot3(n, edge_hat);
            let corrected = [n[0] - d * edge_hat[0], n[1] - d * edge_hat[1], n[2] - d * edge_hat[2]];
            let cl = norm3(corrected);
            if cl > 1e-14 { scale3(corrected, 1.0 / cl) } else { [0.0, 0.0, 1.0] }
        } else {
            [0.0, 0.0, 1.0]
        };

        let dual_dir = cross3(approx_normal, edge_hat);
        let vel_magnitude = dpsi / edge_len;
        let u_contrib = scale3(dual_dir, vel_magnitude);

        vel[v0] = add3(vel[v0], scale3(u_contrib, 0.5));
        vel[v1] = add3(vel[v1], scale3(u_contrib, 0.5));
    }

    // Normalise by vertex valence.
    for (v, edges) in vel.iter_mut().zip(vertex_boundaries) {
        let valence = edges.len() as f64;
        if valence > 0.0 {
            *v = scale3(*v, 1.0 / valence);
        }
    }

    VelocityFieldDec { v: vel, n_vertices: nv }
}

// Vector helpers.
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn mid3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])] }
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0] * s, a[1] * s, a[2] * s] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
