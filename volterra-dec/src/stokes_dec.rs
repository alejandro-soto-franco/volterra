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
    /// Hodge star on 1-forms: star_1[e] = |dual_edge| / |primal_edge|.
    star1: Vec<f64>,
    /// Per-vertex unit normal.
    normals: Vec<[f64; 3]>,
    /// Per-vertex tangent frame e1 direction (e2 = n x e1).
    e1_frames: Vec<[f64; 3]>,
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
pub fn extract_coords<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Vec<[f64; 3]> {
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
    /// Build the Stokes solver, caching vertex coordinates, dual areas, and tangent frames.
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>, mesh: &Mesh<M, 3, 2>) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();
        let poisson = PoissonSolver::new(ops)?;
        let coords = extract_coords(mesh);
        let dual_areas = compute_dual_areas(n_vertices, &mesh.simplices, &coords);
        let s1 = ops.hodge.star1();
        let star1: Vec<f64> = (0..s1.len()).map(|i| s1[i]).collect();
        let normals = compute_vertex_normals_stokes(&mesh.simplices, &coords);
        let e1_frames = compute_tangent_frames_stokes(&normals);
        Ok(Self { poisson, n_vertices, coords, dual_areas, star1, normals, e1_frames })
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

        // Compute vorticity source from active stress (covariant divergence).
        let omega = compute_vorticity_source(
            q, zeta, eta, mesh, &self.coords, &self.dual_areas,
            &self.normals, &self.e1_frames,
        );

        // Solve for stream function: -Delta psi = omega.
        let psi = self.poisson.solve(&omega);

        // Recover 3D velocity: u = curl(psi) on the surface.
        velocity_from_psi(nv, &psi, mesh, &self.coords, &self.dual_areas, &self.star1)
    }
}

/// Compute the vorticity source from active stress curl(div(-zeta Q)).
///
/// Uses the covariant tensor divergence: Q at each vertex is expanded into
/// its 3D ambient representation using the vertex tangent frame (e1, e2),
/// then FEM gradients give the surface divergence on each triangle. This
/// correctly handles curved surfaces where the tangent frames differ between
/// vertices (the previous version used only global (x,y) components, which
/// fails away from the equator on a sphere).
pub fn compute_vorticity_source<M: Manifold>(
    q: &QFieldDec,
    zeta: f64,
    eta: f64,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
    normals: &[[f64; 3]],
    e1_frames: &[[f64; 3]],
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

        // FEM gradient basis functions: grad(phi_a) = (n x e_opp) / (2*area).
        let grad_phi = [
            scale3(cross3(fn_hat, e12), inv_2a),
            scale3(cross3(fn_hat, e20), inv_2a),
            scale3(cross3(fn_hat, e01), inv_2a),
        ];

        let verts = [i0, i1, i2];

        // Compute 3D active force f = -zeta * div(Q) on this face.
        //
        // Q at vertex a in 3D ambient:
        //   Q_{kj} = q1_a*(e1_k*e1_j - e2_k*e2_j) + q2_a*(e1_k*e2_j + e2_k*e1_j)
        //
        // div(Q)_k = sum_a sum_j grad_phi_a[j] * Q_{kj}(a)
        //          = sum_a [q1_a*(e1_k*g1 - e2_k*g2) + q2_a*(e1_k*g2 + e2_k*g1)]
        //
        // where g1 = dot(grad_phi_a, e1_a), g2 = dot(grad_phi_a, e2_a).
        let mut f = [0.0_f64; 3];
        for local in 0..3 {
            let vi = verts[local];
            let e1 = e1_frames[vi];
            let e2 = cross3(normals[vi], e1);

            let g1 = dot3(grad_phi[local], e1);
            let g2 = dot3(grad_phi[local], e2);

            let q1_v = q.q1[vi];
            let q2_v = q.q2[vi];

            for k in 0..3 {
                f[k] += -zeta * (
                    q1_v * (e1[k] * g1 - e2[k] * g2)
                  + q2_v * (e1[k] * g2 + e2[k] * g1)
                );
            }
        }

        // Circulation of the 3D force around the triangle edges.
        let circ_01 = dot3(f, e01);
        let circ_12 = dot3(f, e12);
        let circ_20 = dot3(f, e20);

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

/// Recover the 3D velocity field u = *d(psi) from the stream function.
///
/// Uses the DEC discrete curl: u = star_0_inv * d0^T * star_1 * d0 * psi
/// is the Laplacian (already solved). The velocity 1-form is star_1 * d0 * psi.
/// To convert to a vertex vector field, for each edge e with endpoints v0, v1:
///
///   flux_e = star_1[e] * dpsi_e  (integrated velocity across dual edge)
///
/// The direction is perpendicular to the primal edge in the surface (the
/// dual edge direction): face_normal x edge_hat.
///
/// The vertex velocity is the area-weighted sum:
///   u_v = (1 / A_v) * sum_{e incident to v} flux_e * dual_edge_direction
fn velocity_from_psi<M: Manifold>(
    nv: usize,
    psi: &DVector<f64>,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    _dual_areas: &[f64],
    _star1: &[f64],
) -> VelocityFieldDec {
    let ne = mesh.n_boundaries();
    let mut vel = vec![[0.0_f64; 3]; nv];

    for e in 0..ne {
        let [v0, v1] = mesh.boundaries[e];
        let dpsi = psi[v1] - psi[v0];

        let edge = sub3(coords[v1], coords[v0]);
        let edge_len = norm3(edge);
        if edge_len < 1e-30 { continue; }
        let edge_hat = scale3(edge, 1.0 / edge_len);

        // Average face normal.
        let fn_hat = average_edge_normal(e, mesh, coords);

        // Dual edge direction: face_normal x edge_hat (unit vector, perpendicular
        // to the primal edge in the surface tangent plane).
        let dual_dir = cross3(fn_hat, edge_hat);

        // DEC velocity flux: star_1[e] * dpsi.
        // This is the integrated velocity across the dual edge.
        // Divide by dual_edge_length to get pointwise velocity magnitude.
        // dual_edge_length = star_1[e] * primal_edge_length.
        // So pointwise velocity = star_1[e] * dpsi / (star_1[e] * edge_len) = dpsi / edge_len.
        let vel_magnitude = dpsi / edge_len;

        let u_contrib = scale3(dual_dir, vel_magnitude);

        // Distribute equally to both endpoints.
        vel[v0] = add3(vel[v0], scale3(u_contrib, 0.5));
        vel[v1] = add3(vel[v1], scale3(u_contrib, 0.5));
    }

    // Normalise by the number of incident edges (vertex valence averaging).
    for (v, boundaries) in vel.iter_mut().zip(&mesh.vertex_boundaries) {
        let valence = boundaries.len() as f64;
        if valence > 0.0 {
            *v = scale3(*v, 1.0 / valence);
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

/// Covariant advection: computes (u . grad Q) with parallel transport.
///
/// Unlike [`advect_q`], this function parallel-transports Q along each edge
/// before computing directional derivatives, correctly handling the
/// frame-dependence of Q-tensor components on curved surfaces.
///
/// `edge_phases[e]` is the spin-2 connection phase for `mesh_boundaries[e]`,
/// obtained from [`crate::connection_laplacian::ConnectionLaplacian::edge_phases`].
pub fn advect_q_covariant(
    q: &QFieldDec,
    vel: &VelocityFieldDec,
    mesh_boundaries: &[[usize; 2]],
    vertex_boundaries: &[Vec<usize>],
    coords: &[[f64; 3]],
    edge_phases: &[f64],
) -> QFieldDec {
    let nv = q.n_vertices;
    let mut adv_q1 = vec![0.0; nv];
    let mut adv_q2 = vec![0.0; nv];

    for (e, &[v0, v1]) in mesh_boundaries.iter().enumerate() {
        let edge = sub3(coords[v1], coords[v0]);
        let edge_len_sq = dot3(edge, edge);
        if edge_len_sq < 1e-30 { continue; }

        let u_mid = scale3(add3(vel.v[v0], vel.v[v1]), 0.5);
        let u_dot_e = dot3(u_mid, edge) / edge_len_sq;

        let phase = edge_phases[e];
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // Transport Q from v1 to v0's frame, then difference.
        let q1_v1_in_v0 =  cos_p * q.q1[v1] + sin_p * q.q2[v1];
        let q2_v1_in_v0 = -sin_p * q.q1[v1] + cos_p * q.q2[v1];
        let dq1_at_v0 = q1_v1_in_v0 - q.q1[v0];
        let dq2_at_v0 = q2_v1_in_v0 - q.q2[v0];

        let n_e0 = vertex_boundaries[v0].len().max(1) as f64;
        adv_q1[v0] += u_dot_e * dq1_at_v0 / n_e0;
        adv_q2[v0] += u_dot_e * dq2_at_v0 / n_e0;

        // Transport Q from v0 to v1's frame, then difference.
        let q1_v0_in_v1 = cos_p * q.q1[v0] - sin_p * q.q2[v0];
        let q2_v0_in_v1 = sin_p * q.q1[v0] + cos_p * q.q2[v0];
        let dq1_at_v1 = q.q1[v1] - q1_v0_in_v1;
        let dq2_at_v1 = q.q2[v1] - q2_v0_in_v1;

        // Reversed edge direction: u_dot_e_rev = -u_dot_e, dq_rev = -dq_at_v1_reversed
        // But dq_at_v1 is already "Q_v1 - transported(Q_v0)", and the edge direction for v1
        // is -e, so the sign of u_dot_e flips and the dq sign flips, giving the same product.
        let n_e1 = vertex_boundaries[v1].len().max(1) as f64;
        adv_q1[v1] += u_dot_e * dq1_at_v1 / n_e1;
        adv_q2[v1] += u_dot_e * dq2_at_v1 / n_e1;
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

/// Compute area-weighted vertex normals from a triangle mesh.
fn compute_vertex_normals_stokes(simplices: &[[usize; 3]], coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let nv = coords.len();
    let mut normals = vec![[0.0_f64; 3]; nv];
    for &[i0, i1, i2] in simplices {
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let fn_vec = cross3(e01, e02);
        normals[i0] = add3(normals[i0], fn_vec);
        normals[i1] = add3(normals[i1], fn_vec);
        normals[i2] = add3(normals[i2], fn_vec);
    }
    for n in &mut normals {
        let len = norm3(*n);
        if len > 1e-14 {
            *n = scale3(*n, 1.0 / len);
        }
    }
    normals
}

/// Compute per-vertex tangent frame e1 from normals (e2 = n x e1).
///
/// Uses the same algorithm as `ConnectionLaplacian` so that the frames are
/// consistent with the covariant Laplacian used by the molecular field.
fn compute_tangent_frames_stokes(normals: &[[f64; 3]]) -> Vec<[f64; 3]> {
    normals.iter().map(|n| {
        let ref_dir = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let d = dot3(*n, ref_dir);
        let t = [ref_dir[0] - d * n[0], ref_dir[1] - d * n[1], ref_dir[2] - d * n[2]];
        let len = norm3(t);
        if len > 1e-14 { scale3(t, 1.0 / len) } else { [1.0, 0.0, 0.0] }
    }).collect()
}

// Vector helpers (pub(crate) for use by curved_stokes).
pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
pub(crate) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s, a[1]*s, a[2]*s] }
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
pub(crate) fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
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

    #[test]
    fn stokes_nonzero_velocity_on_sphere() {
        use crate::mesh_gen::icosphere;
        use cartan_manifolds::sphere::Sphere;

        let mesh = icosphere(3); // 642 vertices
        let ops = Operators::from_mesh_generic(&mesh, &Sphere::<3>).unwrap();
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 1.0;

        let nv = mesh.n_vertices();
        let q = QFieldDec::random_perturbation(nv, 0.3, 42);

        let solver = StokesSolverDec::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_rms: f64 = (v.v.iter()
            .map(|[x, y, z]| x * x + y * y + z * z)
            .sum::<f64>() / nv as f64)
            .sqrt();
        assert!(
            v_rms > 1e-6,
            "nonzero activity on sphere should give nonzero velocity, got v_rms = {v_rms:.3e}"
        );
    }
}
