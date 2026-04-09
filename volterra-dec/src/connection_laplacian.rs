//! Parallel-transport connection Laplacian for spin-2 Q-tensors on DEC meshes.
//!
//! Computes the covariant Laplacian of a nematic Q-tensor field on a triangulated
//! surface using the Levi-Civita connection. The Q-tensor is represented as a
//! complex number z = q1 + i*q2 (section of the line bundle L^2). Parallel
//! transport along an edge rotates z by exp(i * 2 * alpha), where alpha is the
//! discrete connection angle.
//!
//! The curvature enters automatically through the holonomy: the accumulated
//! connection angles around a vertex sum to 2*pi - angle_deficit, where
//! angle_deficit = K * A_dual (discrete Gauss-Bonnet). No manual Weitzenboeck
//! correction is needed.
//!
//! ## References
//!
//! - Knoppel et al. "Globally Optimal Direction Fields." ACM TOG 2013.
//! - Crane et al. "Trivial Connections on Discrete Surfaces." SGP 2010.
//! - Zhu, Saintillan & Chern. "Active nematic fluids on Riemannian
//!   2-manifolds." arXiv:2405.06044, 2024.

use cartan_core::Manifold;
use cartan_dec::Mesh;

use crate::QFieldDec;

/// Precomputed connection data for the spin-2 Laplacian.
///
/// Stores the connection angle and cotangent weight for each edge,
/// plus the dual cell area at each vertex. All quantities are computed
/// once from the mesh geometry and reused at every time step.
pub struct ConnectionLaplacian {
    /// Number of vertices.
    pub n_vertices: usize,
    /// For each edge: (v0, v1, cot_weight, connection_angle_2x).
    /// connection_angle_2x = 2 * alpha (spin-2 transport).
    edges: Vec<EdgeData>,
    /// Dual cell area (star_0) at each vertex.
    dual_areas: Vec<f64>,
}

struct EdgeData {
    v0: usize,
    v1: usize,
    /// Cotangent weight: (cot alpha_ij + cot beta_ij) / 2.
    cot_weight: f64,
    /// Twice the connection angle (spin-2 transport phase).
    phase_2x: f64,
}

impl ConnectionLaplacian {
    /// Build the connection Laplacian from a triangle mesh.
    ///
    /// `coords[i]` is the 3D position of vertex i.
    /// `star0` and `star1` are the Hodge star diagonals from cartan-dec's
    /// Operators, ensuring consistency with the scalar Laplace-Beltrami.
    pub fn new<M: Manifold>(
        mesh: &Mesh<M, 3, 2>,
        coords: &[[f64; 3]],
        star0: &[f64],
        star1: &[f64],
    ) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();

        // Compute per-vertex reference tangent frames.
        let normals = compute_vertex_normals(&mesh.simplices, coords);
        let e1_frames = compute_tangent_frames(&normals);

        // Compute per-edge connection angles. Use star_1 as cotangent weights
        // (consistent with cartan-dec's scalar Laplacian).
        let mut edges = Vec::with_capacity(ne);
        for (e, &[v0, v1]) in mesh.boundaries.iter().enumerate() {

            // Connection angle: the rotation from v0's frame to v1's frame,
            // as measured by parallel-transporting e1 along the edge.
            //
            // On a flat mesh, both frames are [1,0,0], so alpha = 0.
            // On a curved mesh, the frames differ because the tangent planes
            // are tilted relative to each other.
            //
            // Compute: project e1[v0] onto v1's tangent plane, then measure
            // the angle between the projection and e1[v1].
            let e1_v0 = e1_frames[v0];
            let e1_v1 = e1_frames[v1];
            let n_v1 = normals[v1];
            let e2_v1 = cross3(n_v1, e1_v1);

            // Project e1[v0] onto v1's tangent plane (remove normal component).
            let proj_dot = dot3(e1_v0, n_v1);
            let proj = [
                e1_v0[0] - proj_dot * n_v1[0],
                e1_v0[1] - proj_dot * n_v1[1],
                e1_v0[2] - proj_dot * n_v1[2],
            ];
            let proj_len = norm3(proj);
            let alpha = if proj_len > 1e-14 {
                let proj_hat = scale3(proj, 1.0 / proj_len);
                let x = dot3(proj_hat, e1_v1);
                let y = dot3(proj_hat, e2_v1);
                y.atan2(x)
            } else {
                0.0
            };

            edges.push(EdgeData {
                v0,
                v1,
                cot_weight: star1[e], // use cartan-dec's Hodge star
                phase_2x: 2.0 * alpha,
            });
        }

        // Use star_0 as dual areas (consistent with cartan-dec).
        let dual_areas = star0.to_vec();

        ConnectionLaplacian {
            n_vertices: nv,
            edges,
            dual_areas,
        }
    }

    /// Apply the connection Laplacian to a Q-tensor field.
    ///
    /// Returns Delta_conn Q, where the parallel transport is built into
    /// the stencil weights via the spin-2 phase rotation.
    pub fn apply(&self, q: &QFieldDec) -> QFieldDec {
        let nv = self.n_vertices;
        let mut lap_q1 = vec![0.0_f64; nv];
        let mut lap_q2 = vec![0.0_f64; nv];

        for edge in &self.edges {
            let v0 = edge.v0;
            let v1 = edge.v1;
            let w = edge.cot_weight;
            let phase = edge.phase_2x;

            // On a flat mesh, phase = 0, so transport is identity.
            // On curved meshes, the phase encodes the Levi-Civita holonomy.
            let cos_p = phase.cos();
            let sin_p = phase.sin();

            // Transport Q from v1 to v0's frame: rotate by -phase (undo the frame rotation).
            let q1_v1 = q.q1[v1];
            let q2_v1 = q.q2[v1];
            let transported_q1 = cos_p * q1_v1 + sin_p * q2_v1;
            let transported_q2 = -sin_p * q1_v1 + cos_p * q2_v1;

            // DEC sign convention: d0[e,v0] = -1, d0[e,v1] = +1.
            // The Laplacian entry for v0 from edge e is:
            //   -star_1[e] * (Q_v1_transported - Q_v0) / star_0[v0]
            // which equals star_1[e] * (Q_v0 - Q_v1_transported) / star_0[v0].
            // We accumulate with the DEC sign (negative of naive difference).
            lap_q1[v0] -= w * (transported_q1 - q.q1[v0]);
            lap_q2[v0] -= w * (transported_q2 - q.q2[v0]);

            // Transport Q from v0 to v1's frame: rotate by +phase.
            let q1_v0 = q.q1[v0];
            let q2_v0 = q.q2[v0];
            let transported_q1_rev = cos_p * q1_v0 - sin_p * q2_v0;
            let transported_q2_rev = sin_p * q1_v0 + cos_p * q2_v0;

            lap_q1[v1] -= w * (transported_q1_rev - q.q1[v1]);
            lap_q2[v1] -= w * (transported_q2_rev - q.q2[v1]);
        }

        // Divide by dual area to get the pointwise Laplacian.
        for i in 0..nv {
            if self.dual_areas[i] > 1e-30 {
                let inv_a = 1.0 / self.dual_areas[i];
                lap_q1[i] *= inv_a;
                lap_q2[i] *= inv_a;
            }
        }

        QFieldDec {
            q1: lap_q1,
            q2: lap_q2,
            n_vertices: nv,
        }
    }
}

/// Compute the molecular field using the connection Laplacian.
///
/// H = K * Delta_conn Q + (-a_eff) * Q - 2c * Tr(Q^2) * Q
pub fn molecular_field_conn(
    q: &QFieldDec,
    k_frank: f64,
    a_eff: f64,
    c_landau: f64,
    conn_lap: &ConnectionLaplacian,
) -> QFieldDec {
    let nv = q.n_vertices;
    let lap = conn_lap.apply(q);
    let tr_q2 = q.trace_q_squared();
    let bulk_linear = -a_eff;

    let mut h = QFieldDec::zeros(nv);
    for (i, &tr) in tr_q2.iter().enumerate() {
        let bulk = bulk_linear - 2.0 * c_landau * tr;
        h.q1[i] = k_frank * lap.q1[i] + bulk * q.q1[i];
        h.q2[i] = k_frank * lap.q2[i] + bulk * q.q2[i];
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

fn compute_vertex_normals(simplices: &[[usize; 3]], coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
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

fn compute_tangent_frames(normals: &[[f64; 3]]) -> Vec<[f64; 3]> {
    normals.iter().map(|n| {
        // Pick a reference direction not aligned with n.
        let ref_dir = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        // Project onto tangent plane and normalise.
        let d = dot3(*n, ref_dir);
        let t = [ref_dir[0] - d * n[0], ref_dir[1] - d * n[1], ref_dir[2] - d * n[2]];
        let len = norm3(t);
        if len > 1e-14 { scale3(t, 1.0 / len) } else { [1.0, 0.0, 0.0] }
    }).collect()
}

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
    use crate::mesh_gen::icosphere;
    use cartan_dec::Operators;
    use cartan_manifolds::sphere::Sphere;

    fn extract_coords_sphere(mesh: &Mesh<Sphere<3>, 3, 2>) -> Vec<[f64; 3]> {
        mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect()
    }

    #[test]
    fn connection_laplacian_constructs_on_sphere() {
        let mesh = icosphere(2);
        let coords = extract_coords_sphere(&mesh);
        let manifold = Sphere::<3>;
        let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
        let star0: Vec<f64> = (0..ops.hodge.star0().len()).map(|i| ops.hodge.star0()[i]).collect();
        let star1: Vec<f64> = (0..ops.hodge.star1().len()).map(|i| ops.hodge.star1()[i]).collect();
        let cl = ConnectionLaplacian::new(&mesh, &coords, &star0, &star1);
        assert_eq!(cl.n_vertices, 162);
    }

    #[test]
    fn connection_laplacian_zero_q_gives_zero() {
        let mesh = icosphere(2);
        let coords = extract_coords_sphere(&mesh);
        let manifold = Sphere::<3>;
        let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
        let star0: Vec<f64> = (0..ops.hodge.star0().len()).map(|i| ops.hodge.star0()[i]).collect();
        let star1: Vec<f64> = (0..ops.hodge.star1().len()).map(|i| ops.hodge.star1()[i]).collect();
        let cl = ConnectionLaplacian::new(&mesh, &coords, &star0, &star1);
        let q = QFieldDec::zeros(162);
        let lap = cl.apply(&q);
        let norm: f64 = lap.q1.iter().chain(&lap.q2).map(|x| x.abs()).sum();
        assert!(norm < 1e-12, "lap of zero Q should be zero, got {norm}");
    }

    #[test]
    fn connection_laplacian_flat_mesh_matches_scalar() {
        // On a flat mesh (Euclidean<2>), the connection angles are all zero,
        // so the connection Laplacian should equal the scalar Laplacian.
        use cartan_dec::mesh::FlatMesh;
        use cartan_dec::Operators;
        use cartan_manifolds::euclidean::Euclidean;
        use nalgebra::DVector;

        let mesh = FlatMesh::unit_square_grid(8);
        let nv = mesh.n_vertices();
        let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v[0], v[1], 0.0]).collect();

        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let star0: Vec<f64> = (0..ops.hodge.star0().len()).map(|i| ops.hodge.star0()[i]).collect();
        let star1: Vec<f64> = (0..ops.hodge.star1().len()).map(|i| ops.hodge.star1()[i]).collect();
        let cl = ConnectionLaplacian::new(&mesh, &coords, &star0, &star1);

        // Random Q field.
        let q = QFieldDec::random_perturbation(nv, 0.1, 42);

        // Connection Laplacian.
        let lap_conn = cl.apply(&q);

        // Scalar Laplacian applied component-wise.
        let q1_vec = DVector::from_column_slice(&q.q1);
        let q2_vec = DVector::from_column_slice(&q.q2);
        let lap_q1 = ops.apply_laplace_beltrami(&q1_vec);
        let lap_q2 = ops.apply_laplace_beltrami(&q2_vec);

        // Compare interior vertices (boundary has different stencils).
        let ns = 9; // unit_square_grid(8) has 9 vertices per side
        let mut max_diff = 0.0_f64;
        for i in ns..(nv - ns) {
            let row = i / ns;
            let col = i % ns;
            if row == 0 || row == ns - 1 || col == 0 || col == ns - 1 {
                continue;
            }
            let d1 = (lap_conn.q1[i] - lap_q1[i]).abs();
            let d2 = (lap_conn.q2[i] - lap_q2[i]).abs();
            max_diff = max_diff.max(d1).max(d2);
        }
        assert!(
            max_diff < 0.1,
            "on flat mesh, connection Laplacian should match scalar: max_diff = {max_diff}"
        );
    }
}
