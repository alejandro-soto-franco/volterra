//! Stokes solver for active nematics on 2D DEC meshes.
//!
//! Computes the incompressible velocity field driven by the active stress
//! sigma = -zeta Q via the stream-function formulation. The velocity is
//! a 3D tangent vector at each vertex (for surfaces embedded in R^3).

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use nalgebra::DVector;
use volterra_core::ActiveNematicParams;

use crate::poisson::PoissonSolver;
use crate::QFieldDec;

/// Precomputed Stokes solver with cached vertex coordinates and Poisson factorisation.
pub struct StokesSolverDec {
    poisson: PoissonSolver,
    n_vertices: usize,
    /// Vertex coordinates in R^3 (for 2D meshes, z = 0).
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (barycentric, 1/3 of incident triangle areas).
    dual_areas: Vec<f64>,
}

/// Velocity field on a DEC mesh: 3D tangent vector per vertex.
#[derive(Debug, Clone)]
pub struct VelocityFieldDec {
    /// Velocity components at each vertex (3D tangent vector).
    pub v: Vec<[f64; 3]>,
    pub n_vertices: usize,
}

impl VelocityFieldDec {
    pub fn zeros(nv: usize) -> Self {
        Self {
            v: vec![[0.0; 3]; nv],
            n_vertices: nv,
        }
    }

    /// Convenience accessors for backward compatibility.
    pub fn vx(&self, i: usize) -> f64 { self.v[i][0] }
    pub fn vy(&self, i: usize) -> f64 { self.v[i][1] }
    pub fn vz(&self, i: usize) -> f64 { self.v[i][2] }

    /// Velocity magnitude at vertex i.
    pub fn speed(&self, i: usize) -> f64 {
        let [x, y, z] = self.v[i];
        (x * x + y * y + z * z).sqrt()
    }
}

/// Extract vertex coordinates from a generic mesh by formatting via Debug.
/// All cartan manifold Point types are SVector<f64, N>.
fn extract_coords<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Vec<[f64; 3]> {
    mesh.vertices.iter().map(|v| {
        let s = format!("{:?}", v);
        let nums: Vec<f64> = s
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',' || *c == ' ' || *c == 'e' || *c == '+')
            .collect::<String>()
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        match nums.len() {
            2 => [nums[0], nums[1], 0.0],
            n if n >= 3 => [nums[0], nums[1], nums[2]],
            _ => [0.0, 0.0, 0.0],
        }
    }).collect()
}

/// Compute barycentric dual cell areas from triangles.
fn compute_dual_areas(nv: usize, simplices: &[[usize; 3]], coords: &[[f64; 3]]) -> Vec<f64> {
    let mut areas = vec![0.0_f64; nv];
    for &[i0, i1, i2] in simplices {
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let cr = cross3(e01, e02);
        let face_area = 0.5 * norm3(cr);
        let third = face_area / 3.0;
        areas[i0] += third;
        areas[i1] += third;
        areas[i2] += third;
    }
    areas
}

impl StokesSolverDec {
    /// Build the Stokes solver, caching vertex coordinates and dual areas.
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>, mesh: &Mesh<M, 3, 2>) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();
        let poisson = PoissonSolver::new(ops)?;
        let coords = extract_coords(mesh);
        let dual_areas = compute_dual_areas(n_vertices, &mesh.simplices, &coords);
        Ok(Self { poisson, n_vertices, coords, dual_areas })
    }

    /// Solve Stokes for the active velocity field.
    pub fn solve<M: Manifold>(
        &self,
        q: &QFieldDec,
        params: &ActiveNematicParams,
        _ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
    ) -> VelocityFieldDec {
        let nv = self.n_vertices;
        let zeta = params.zeta_eff;
        let eta = params.eta;

        if zeta.abs() < 1e-30 || eta.abs() < 1e-30 {
            return VelocityFieldDec::zeros(nv);
        }

        // Compute vorticity source from active stress.
        let omega = compute_vorticity_source(q, zeta, eta, mesh, &self.coords, &self.dual_areas);

        // Solve for stream function: -Delta psi = omega.
        let psi = self.poisson.solve(&omega);

        // Recover 3D velocity: u = curl(psi) on the surface.
        velocity_from_psi(nv, &psi, mesh, &self.coords, &self.dual_areas)
    }
}

/// Compute the vorticity source from active stress curl(div(-zeta Q)).
fn compute_vorticity_source<M: Manifold>(
    q: &QFieldDec,
    zeta: f64,
    eta: f64,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
) -> DVector<f64> {
    let nv = q.n_vertices;
    let mut omega = vec![0.0_f64; nv];

    for &[i0, i1, i2] in &mesh.simplices {
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

        // Rotated edges (n x e) for gradient computation.
        let rot_e12 = cross3(fn_hat, e12);
        let rot_e20 = cross3(fn_hat, e20);
        let rot_e01 = cross3(fn_hat, e01);

        // Gradient of q1 and q2 on this face.
        let gq1 = scale3(
            add3(add3(
                scale3(rot_e12, q.q1[i0]),
                scale3(rot_e20, q.q1[i1])),
                scale3(rot_e01, q.q1[i2])),
            inv_2a,
        );
        let gq2 = scale3(
            add3(add3(
                scale3(rot_e12, q.q2[i0]),
                scale3(rot_e20, q.q2[i1])),
                scale3(rot_e01, q.q2[i2])),
            inv_2a,
        );

        // Active force on this face: f = -zeta * div(Q)
        // f_x = -zeta*(gq1_x + gq2_y), f_y = -zeta*(gq2_x - gq1_y)
        // (using the local face tangent plane)
        let fx = -zeta * (gq1[0] + gq2[1]);
        let fy = -zeta * (gq2[0] - gq1[1]);

        // Circulation of f around the triangle edges, distributed to vertices.
        let circ_01 = fx * e01[0] + fy * e01[1];
        let circ_12 = fx * e12[0] + fy * e12[1];
        let circ_20 = fx * e20[0] + fy * e20[1];

        omega[i0] += 0.5 * (circ_01 - circ_20);
        omega[i1] += 0.5 * (circ_12 - circ_01);
        omega[i2] += 0.5 * (circ_20 - circ_12);
    }

    // Normalise by dual area and viscosity.
    for i in 0..nv {
        if dual_areas[i] > 1e-30 {
            omega[i] /= eta * dual_areas[i];
        }
    }

    DVector::from_vec(omega)
}

/// Recover the 3D velocity field u = curl(psi) from the stream function.
///
/// For each edge [v0, v1], the velocity contribution is a 3D tangent vector:
///   u_edge = (dpsi / |e|^2) * (face_normal x edge_tangent)
///
/// This gives a vector in the surface tangent plane, perpendicular to the edge.
fn velocity_from_psi<M: Manifold>(
    nv: usize,
    psi: &DVector<f64>,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
) -> VelocityFieldDec {
    let ne = mesh.n_boundaries();
    let mut vel = vec![[0.0_f64; 3]; nv];

    for e in 0..ne {
        let [v0, v1] = mesh.boundaries[e];
        let dpsi = psi[v1] - psi[v0];

        let edge = sub3(coords[v1], coords[v0]);
        let edge_len_sq = dot3(edge, edge);
        if edge_len_sq < 1e-30 { continue; }

        // Average face normal of the one or two triangles sharing this edge.
        let fn_hat = average_edge_normal(e, mesh, coords);

        // Rotated edge: face_normal x edge (perpendicular to edge, in the surface).
        let rot = cross3(fn_hat, edge);

        // Velocity contribution: dpsi * rot / |e|^2 (integrated form).
        let factor = dpsi / edge_len_sq;
        let u_contrib = scale3(rot, factor);

        // Distribute to both endpoints.
        vel[v0] = add3(vel[v0], u_contrib);
        vel[v1] = add3(vel[v1], u_contrib);
    }

    // Normalise by dual area.
    for i in 0..nv {
        if dual_areas[i] > 1e-30 {
            vel[i] = scale3(vel[i], 1.0 / dual_areas[i]);
        }
    }

    VelocityFieldDec { v: vel, n_vertices: nv }
}

/// Advect Q along velocity: computes (u · grad Q) at each vertex.
///
/// Uses edge-based directional derivative: for each edge [v0, v1],
/// the advective flux is (u · edge_tangent) * (Q_v1 - Q_v0) / |e|^2,
/// distributed to both vertices.
pub fn advect_q(
    q: &QFieldDec,
    vel: &VelocityFieldDec,
    mesh_boundaries: &[[usize; 2]],
    vertex_boundaries: &[Vec<usize>],
    coords: &[[f64; 3]],
) -> QFieldDec {
    let nv = q.n_vertices;
    let mut adv_q1 = vec![0.0; nv];
    let mut adv_q2 = vec![0.0; nv];

    for &[v0, v1] in mesh_boundaries {
        let edge = sub3(coords[v1], coords[v0]);
        let edge_len_sq = dot3(edge, edge);
        if edge_len_sq < 1e-30 { continue; }

        // Velocity at edge midpoint (average of endpoints).
        let u_mid = scale3(add3(vel.v[v0], vel.v[v1]), 0.5);

        // Directional derivative along the edge: (u · e_hat) * dQ / |e|
        let u_dot_e = dot3(u_mid, edge) / edge_len_sq;
        let dq1 = q.q1[v1] - q.q1[v0];
        let dq2 = q.q2[v1] - q.q2[v0];

        let flux1 = u_dot_e * dq1;
        let flux2 = u_dot_e * dq2;

        // Distribute to both vertices (central average).
        let n_e0 = vertex_boundaries[v0].len().max(1) as f64;
        let n_e1 = vertex_boundaries[v1].len().max(1) as f64;
        adv_q1[v0] += flux1 / n_e0;
        adv_q1[v1] += flux1 / n_e1;
        adv_q2[v0] += flux2 / n_e0;
        adv_q2[v1] += flux2 / n_e1;
    }

    QFieldDec { q1: adv_q1, q2: adv_q2, n_vertices: nv }
}

fn average_edge_normal<M: Manifold>(
    edge_idx: usize,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> [f64; 3] {
    let mut n = [0.0_f64; 3];
    for &fi in &mesh.boundary_simplices[edge_idx] {
        let [i0, i1, i2] = mesh.simplices[fi];
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let cr = cross3(e01, e02);
        n = add3(n, cr);
    }
    let len = norm3(n);
    if len > 1e-14 { scale3(n, 1.0 / len) } else { [0.0, 0.0, 1.0] }
}

// Vector helpers.
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s, a[1]*s, a[2]*s] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::mesh::FlatMesh;
    use cartan_manifolds::euclidean::Euclidean;

    #[test]
    fn stokes_zero_activity_zero_velocity() {
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 0.0;

        let nv = mesh.n_vertices();
        let q = QFieldDec::random_perturbation(nv, 0.5, 42);

        let solver = StokesSolverDec::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.v.iter().map(|[x, y, z]| x.abs() + y.abs() + z.abs()).sum();
        assert!(v_norm < 1e-12, "zero activity should give zero velocity");
    }

    #[test]
    fn stokes_nonzero_activity_nonzero_velocity() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 1.0;

        let nv = mesh.n_vertices();
        let q = QFieldDec::random_perturbation(nv, 0.3, 42);

        let solver = StokesSolverDec::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.v.iter()
            .map(|[x, y, z]| x * x + y * y + z * z)
            .sum::<f64>()
            .sqrt();
        assert!(v_norm > 1e-6, "nonzero activity should give nonzero velocity, got {v_norm}");
    }

    #[test]
    fn stokes_solver_constructs() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = StokesSolverDec::new(&ops, &mesh);
        assert!(solver.is_ok());
    }
}
