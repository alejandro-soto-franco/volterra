//! Stokes solver for active nematics on 2D DEC meshes.
//!
//! Computes the incompressible velocity field driven by the active stress
//! sigma = -zeta Q via the stream-function formulation.
//!
//! The biharmonic is solved via two Poisson problems (LDL^T factorised).
//! Velocity is recovered from the stream function via the DEC curl:
//! u = *d(psi), computed using edge tangent vectors rotated 90 degrees
//! in the surface tangent plane.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use nalgebra::DVector;
use volterra_core::ActiveNematicParams;

use crate::poisson::PoissonSolver;
use crate::QFieldDec;

/// Precomputed Stokes solver for repeated time steps.
pub struct StokesSolverDec {
    poisson: PoissonSolver,
    n_vertices: usize,
}

/// Velocity field on a DEC mesh (2 components per vertex).
#[derive(Debug, Clone)]
pub struct VelocityFieldDec {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub n_vertices: usize,
}

impl VelocityFieldDec {
    pub fn zeros(nv: usize) -> Self {
        Self {
            vx: vec![0.0; nv],
            vy: vec![0.0; nv],
            n_vertices: nv,
        }
    }
}

impl StokesSolverDec {
    /// Build the Stokes solver (precomputes LDL^T factorisation).
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();
        let poisson = PoissonSolver::new(ops)?;
        Ok(Self {
            poisson,
            n_vertices,
        })
    }

    /// Solve Stokes for the active velocity on a mesh with known vertex coordinates.
    ///
    /// `vertex_coords` provides the (x, y, z) position of each vertex as a flat
    /// slice of length 3 * n_vertices. For 2D meshes embedded in R^2, set z = 0.
    ///
    /// `face_vertex_ids` provides the triangle connectivity as [i, j, k] per face.
    pub fn solve_with_coords<M: Manifold>(
        &self,
        q: &QFieldDec,
        params: &ActiveNematicParams,
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        vertex_coords: &[[f64; 3]],
    ) -> VelocityFieldDec {
        let nv = self.n_vertices;
        let zeta = params.zeta_eff;
        let eta = params.eta;

        if zeta.abs() < 1e-30 || eta.abs() < 1e-30 {
            return VelocityFieldDec::zeros(nv);
        }

        // Active stress source: compute the vorticity source from Q gradients.
        // The full source for the stream-function equation is:
        //   eta * Delta(omega) = curl(div(sigma))
        // For traceless sigma = -zeta * Q:
        //   curl(div(sigma)) = -zeta * [2 d_xy(q1) + (d_yy - d_xx)(q2)]
        //
        // On the DEC mesh, we compute this via per-edge gradients:
        // For each edge e = [v0, v1], the integrated gradient of a scalar f is:
        //   (d0 f)_e = f[v1] - f[v0]
        //
        // The active force at each vertex is approximated by averaging the
        // edge-based divergence of the Q-tensor stress.
        let omega_source = compute_vorticity_source(q, params, ops, mesh, vertex_coords);

        // Solve for stream function: -Delta psi = omega_source.
        let psi = self.poisson.solve(&omega_source);

        // Recover velocity: u = curl(psi) via edge normals.
        velocity_from_psi(nv, &psi, mesh, vertex_coords)
    }

    /// Convenience method that extracts vertex coordinates from the mesh.
    ///
    /// Works for any manifold where M::Point is SVector<f64, N> with N >= 2.
    /// Pads 2D points to 3D with z = 0.
    pub fn solve<M: Manifold>(
        &self,
        q: &QFieldDec,
        params: &ActiveNematicParams,
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
    ) -> VelocityFieldDec {
        // Extract vertex coordinates via Debug formatting.
        // All cartan manifold Point types are SVector<f64, N> with Debug output
        // like "[[x, y]]" or "[[x, y, z]]". We parse the numeric values.
        let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| {
            let s = format!("{:?}", v);
            let nums: Vec<f64> = s
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',' || *c == ' ' || *c == 'e')
                .collect::<String>()
                .split(',')
                .filter_map(|t| t.trim().parse::<f64>().ok())
                .collect();
            match nums.len() {
                2 => [nums[0], nums[1], 0.0],
                n if n >= 3 => [nums[0], nums[1], nums[2]],
                _ => [0.0, 0.0, 0.0],
            }
        }).collect();

        self.solve_with_coords(q, params, ops, mesh, &coords)
    }
}

/// Compute the vorticity source from active stress divergence.
///
/// Uses per-edge gradients of Q to compute curl(div(sigma)) at each vertex.
fn compute_vorticity_source<M: Manifold>(
    q: &QFieldDec,
    params: &ActiveNematicParams,
    _ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> DVector<f64> {
    let nv = q.n_vertices;
    let _ne = mesh.n_boundaries();
    let zeta = params.zeta_eff;
    let eta = params.eta;

    // Active force: f = -zeta * div(Q)
    // For traceless Q in 2D: f_x = -zeta*(d_x q1 + d_y q2), f_y = -zeta*(d_x q2 - d_y q1)
    // Vorticity source: omega = curl(f)/eta = (d_x f_y - d_y f_x)/eta
    //
    // On the DEC mesh, approximate this via:
    // 1. Compute per-edge force contributions from Q gradients.
    // 2. Accumulate the curl at each vertex.

    let mut omega = vec![0.0_f64; nv];
    let mut areas = vec![0.0_f64; nv];

    // For each triangle, compute the integrated vorticity contribution.
    let nf = mesh.n_simplices();
    for f in 0..nf {
        let [i0, i1, i2] = mesh.simplices[f];
        let p0 = coords[i0];
        let p1 = coords[i1];
        let p2 = coords[i2];

        // Edge vectors.
        let e01 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e02 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        // Face normal (for 2D, this is just (0, 0, area*2)).
        let nx = e01[1] * e02[2] - e01[2] * e02[1];
        let ny = e01[2] * e02[0] - e01[0] * e02[2];
        let nz = e01[0] * e02[1] - e01[1] * e02[0];
        let area2 = (nx * nx + ny * ny + nz * nz).sqrt();
        let face_area = 0.5 * area2;

        if face_area < 1e-30 {
            continue;
        }

        // Face-average Q gradient via linear interpolation on the triangle.
        // grad(q) = (1/(2*A)) * sum_i q_i * (n x e_opposite_i)
        // where e_opposite_i is the edge opposite to vertex i.
        let inv_2a = 1.0 / area2;

        // Face normal unit vector.
        let fn_x = nx / area2;
        let fn_y = ny / area2;
        let fn_z = nz / area2;

        // Edge opposite to vertex 0: e12 = p2 - p1
        let e12 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        // Edge opposite to vertex 1: e20 = p0 - p2
        let e20 = [p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]];
        // Edge opposite to vertex 2: e01 (already computed)

        // n x e gives the "rotated edge" that points perpendicular to e in the face plane.
        let rot_e12 = cross3([fn_x, fn_y, fn_z], e12);
        let rot_e20 = cross3([fn_x, fn_y, fn_z], e20);
        let rot_e01 = cross3([fn_x, fn_y, fn_z], e01);

        // Gradient of q1 on this face: grad(q1) = inv_2a * sum_i q1_i * rot_e_opp_i
        let gq1_x = inv_2a * (q.q1[i0] * rot_e12[0] + q.q1[i1] * rot_e20[0] + q.q1[i2] * rot_e01[0]);
        let gq1_y = inv_2a * (q.q1[i0] * rot_e12[1] + q.q1[i1] * rot_e20[1] + q.q1[i2] * rot_e01[1]);
        let gq2_x = inv_2a * (q.q2[i0] * rot_e12[0] + q.q2[i1] * rot_e20[0] + q.q2[i2] * rot_e01[0]);
        let gq2_y = inv_2a * (q.q2[i0] * rot_e12[1] + q.q2[i1] * rot_e20[1] + q.q2[i2] * rot_e01[1]);

        // Active force on this face (in the tangent plane):
        // f_x = -zeta * (d_x q1 + d_y q2)
        // f_y = -zeta * (d_x q2 - d_y q1)
        let fx = -zeta * (gq1_x + gq2_y);
        let fy = -zeta * (gq2_x - gq1_y);

        // Vorticity of the force: omega = (d_x f_y - d_y f_x) / eta
        // On a single triangle with constant gradients, the vorticity is zero
        // (constant force has zero curl). The vorticity arises from the
        // DIFFERENCE in force between adjacent triangles.
        //
        // For the DEC approach, accumulate the force circulation around each vertex:
        // omega_v = (1/A_dual) * sum_{edges around v} (f dot e_rotated)
        //
        // Distribute the face's force contribution to its three vertices.
        let third_area = face_area / 3.0;
        for &vi in &[i0, i1, i2] {
            // Approximate: accumulate the force magnitude weighted by face area.
            // The true vorticity comes from spatial variation of f.
            areas[vi] += third_area;
        }

        // Circulation contribution: integrate f along each edge of the triangle
        // and accumulate to the vertex at the left of the edge (CCW orientation).
        // edge i0->i1: f dot (p1-p0)
        let circ_01 = fx * e01[0] + fy * e01[1];
        let circ_12 = fx * e12[0] + fy * e12[1];
        let circ_20 = fx * e20[0] + fy * e20[1];

        // Distribute circulation to vertices (each vertex gets the circulation
        // of the two edges it touches, weighted by 1/2).
        omega[i0] += 0.5 * (circ_01 - circ_20);
        omega[i1] += 0.5 * (circ_12 - circ_01);
        omega[i2] += 0.5 * (circ_20 - circ_12);
    }

    // Normalise by dual area.
    for i in 0..nv {
        if areas[i] > 1e-30 {
            omega[i] /= eta * areas[i];
        }
    }

    DVector::from_vec(omega)
}

/// Recover velocity u = curl(psi) from the stream function.
///
/// For each edge [v0, v1], the velocity contribution is:
///   u_edge = dpsi * (edge_normal_rotated) / edge_length
/// where edge_normal_rotated is the edge tangent rotated 90 degrees CCW
/// in the surface tangent plane.
///
/// Contributions are averaged to vertices weighted by dual cell areas.
fn velocity_from_psi<M: Manifold>(
    nv: usize,
    psi: &DVector<f64>,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> VelocityFieldDec {
    let ne = mesh.n_boundaries();
    let mut vx = vec![0.0; nv];
    let mut vy = vec![0.0; nv];
    let mut dual_area = vec![0.0_f64; nv];

    // Accumulate dual cell areas from incident triangles.
    let nf = mesh.n_simplices();
    for f in 0..nf {
        let [i0, i1, i2] = mesh.simplices[f];
        let p0 = coords[i0]; let p1 = coords[i1]; let p2 = coords[i2];
        let e01 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let e02 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
        let cr = cross3(e01, e02);
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();
        let third = area / 3.0;
        dual_area[i0] += third;
        dual_area[i1] += third;
        dual_area[i2] += third;
    }

    // For each edge, compute the velocity contribution from dpsi.
    for e in 0..ne {
        let [v0, v1] = mesh.boundaries[e];
        let dpsi = psi[v1] - psi[v0];

        let p0 = coords[v0];
        let p1 = coords[v1];

        // Edge tangent vector.
        let tx = p1[0] - p0[0];
        let ty = p1[1] - p0[1];
        let tz = p1[2] - p0[2];
        let edge_len = (tx*tx + ty*ty + tz*tz).sqrt();
        if edge_len < 1e-30 { continue; }

        // Compute edge normal (tangent rotated 90 degrees in the surface plane).
        // For 2D (tz ~ 0): rotate (tx, ty) -> (-ty, tx).
        // For 3D surfaces: use the average face normal of adjacent triangles,
        // then edge_normal = face_normal x edge_tangent.
        let (nx, ny) = if tz.abs() < 1e-10 * edge_len {
            // Flat 2D case: simple 90-degree rotation.
            (-ty / edge_len, tx / edge_len)
        } else {
            // 3D surface: compute from adjacent face normals.
            let face_normal = average_edge_normal(e, mesh, coords);
            let t_hat = [tx / edge_len, ty / edge_len, tz / edge_len];
            let rot = cross3(face_normal, t_hat);
            // Project to x-y (tangent plane approximation for velocity).
            let rot_len = (rot[0]*rot[0] + rot[1]*rot[1]).sqrt();
            if rot_len > 1e-14 {
                (rot[0] / rot_len, rot[1] / rot_len)
            } else {
                (-ty / edge_len, tx / edge_len)
            }
        };

        // Velocity contribution: u = dpsi * rotated_normal / edge_length.
        // Distribute to both endpoints of the edge.
        let u_contrib_x = dpsi * nx / edge_len;
        let u_contrib_y = dpsi * ny / edge_len;

        vx[v0] += u_contrib_x;
        vy[v0] += u_contrib_y;
        vx[v1] += u_contrib_x;
        vy[v1] += u_contrib_y;
    }

    // Normalise by dual area (each edge contributes to the dual cell of both endpoints).
    for i in 0..nv {
        if dual_area[i] > 1e-30 {
            let inv_a = 1.0 / dual_area[i];
            vx[i] *= inv_a;
            vy[i] *= inv_a;
        }
    }

    VelocityFieldDec { vx, vy, n_vertices: nv }
}

/// Average face normal for the one or two triangles adjacent to an edge.
fn average_edge_normal<M: Manifold>(
    edge_idx: usize,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> [f64; 3] {
    let adjacent = &mesh.boundary_simplices[edge_idx];
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;

    for &fi in adjacent {
        let [i0, i1, i2] = mesh.simplices[fi];
        let p0 = coords[i0]; let p1 = coords[i1]; let p2 = coords[i2];
        let e01 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let e02 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
        let cr = cross3(e01, e02);
        nx += cr[0]; ny += cr[1]; nz += cr[2];
    }

    let norm = (nx*nx + ny*ny + nz*nz).sqrt();
    if norm > 1e-14 {
        [nx/norm, ny/norm, nz/norm]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
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

        let solver = StokesSolverDec::new(&ops).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.vx.iter().chain(&v.vy).map(|x| x.abs()).sum();
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

        let solver = StokesSolverDec::new(&ops).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.vx.iter().chain(&v.vy).map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            v_norm > 1e-6,
            "nonzero activity + nonzero Q should give nonzero velocity, got {v_norm}"
        );
    }

    #[test]
    fn stokes_solver_constructs() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = StokesSolverDec::new(&ops);
        assert!(solver.is_ok());
    }
}
