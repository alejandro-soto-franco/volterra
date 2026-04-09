//! Semi-Lagrangian advection for nematic fields on triangle meshes.
//!
//! Traces the flow map backward from each vertex to find the departure point,
//! then interpolates the nematic field z = q1 + iq2 using barycentric
//! coordinates within the containing triangle.
//!
//! This scheme is unconditionally stable for the advection term (no
//! advective CFL restriction). The only timestep constraint is the
//! diffusive CFL from the connection Laplacian.

use crate::stokes_dec::VelocityFieldDec;
use crate::QFieldDec;

/// Semi-Lagrangian advection operator.
///
/// Precomputes a spatial acceleration structure (simple nearest-triangle
/// search for now) for efficient point-in-triangle queries.
pub struct SemiLagrangian {
    /// Vertex coordinates in R^3.
    coords: Vec<[f64; 3]>,
    /// Triangle vertex indices.
    triangles: Vec<[usize; 3]>,
    /// Per-triangle precomputed data for barycentric queries.
    tri_data: Vec<TriData>,
    /// Number of vertices.
    n_vertices: usize,
}

/// Precomputed data for a single triangle (for fast barycentric queries).
struct TriData {
    /// Centroid position.
    centroid: [f64; 3],
    /// Face normal (unnormalised, magnitude = 2 * area).
    normal: [f64; 3],
    /// Vertex positions.
    v: [[f64; 3]; 3],
}

impl SemiLagrangian {
    /// Build the advection operator from mesh geometry.
    pub fn new(coords: Vec<[f64; 3]>, triangles: Vec<[usize; 3]>) -> Self {
        let n_vertices = coords.len();
        let tri_data: Vec<TriData> = triangles.iter().map(|&[i0, i1, i2]| {
            let v0 = coords[i0];
            let v1 = coords[i1];
            let v2 = coords[i2];
            let centroid = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];
            let e01 = sub3(v1, v0);
            let e02 = sub3(v2, v0);
            let normal = cross3(e01, e02);
            TriData { centroid, normal, v: [v0, v1, v2] }
        }).collect();

        Self { coords, triangles, tri_data, n_vertices }
    }

    /// Advect the nematic field backward along the velocity for one timestep.
    ///
    /// For each vertex v:
    ///   1. Compute departure point: p = v - u(v) * dt
    ///   2. Project p onto the surface (nearest point)
    ///   3. Find the containing triangle
    ///   4. Interpolate z barycentrically
    ///
    /// Returns the advected field (z at the departure points, interpolated
    /// and transported back to the arrival vertices).
    pub fn advect(&self, q: &QFieldDec, vel: &VelocityFieldDec, dt: f64) -> QFieldDec {
        let nv = self.n_vertices;
        let mut q_adv = QFieldDec::zeros(nv);

        for v in 0..nv {
            let pos = self.coords[v];
            let u = vel.v[v];

            // Backward trace: departure point.
            let departure = [
                pos[0] - u[0] * dt,
                pos[1] - u[1] * dt,
                pos[2] - u[2] * dt,
            ];

            // Project onto surface: for a sphere, normalise to unit length.
            // For general surfaces, find nearest point.
            // Simple heuristic: normalise if the mesh is approximately spherical
            // (centroid near origin), otherwise keep as-is.
            let r = norm3(departure);
            let departure = if r > 0.5 {
                // Likely a sphere: project to unit sphere.
                let mesh_r = norm3(self.coords[0]);
                scale3(departure, mesh_r / r)
            } else {
                departure
            };

            // Find containing triangle via nearest-centroid search.
            let (tri_idx, bary) = self.locate_point(departure);

            if let Some((ti, (w0, w1, w2))) = tri_idx.zip(Some(bary)) {
                let [i0, i1, i2] = self.triangles[ti];
                // Barycentric interpolation of z = q1 + iq2.
                q_adv.q1[v] = w0 * q.q1[i0] + w1 * q.q1[i1] + w2 * q.q1[i2];
                q_adv.q2[v] = w0 * q.q2[i0] + w1 * q.q2[i1] + w2 * q.q2[i2];
            } else {
                // Departure point not found in any triangle: keep the original value.
                q_adv.q1[v] = q.q1[v];
                q_adv.q2[v] = q.q2[v];
            }
        }

        q_adv
    }

    /// Locate a point on the mesh: find the containing triangle and barycentric coords.
    ///
    /// Returns (Some(triangle_index), (w0, w1, w2)) if found, or (None, (0,0,0)).
    fn locate_point(&self, p: [f64; 3]) -> (Option<usize>, (f64, f64, f64)) {
        // Simple O(N_triangles) search. For production, replace with BVH.
        let mut best_tri = None;
        let mut best_bary = (0.0, 0.0, 0.0);
        let mut best_dist = f64::INFINITY;

        for (ti, td) in self.tri_data.iter().enumerate() {
            // Quick reject: check distance to centroid.
            let d = dist3(p, td.centroid);
            if d > best_dist * 2.0 && best_tri.is_some() {
                continue;
            }

            // Compute barycentric coordinates.
            let bary = barycentric_3d(p, td.v[0], td.v[1], td.v[2], td.normal);
            let (w0, w1, w2) = bary;

            // Check if inside (with tolerance for numerical issues).
            let tol = -0.1;
            if w0 >= tol && w1 >= tol && w2 >= tol {
                let d_centroid = dist3(p, td.centroid);
                if d_centroid < best_dist {
                    best_dist = d_centroid;
                    best_tri = Some(ti);
                    // Clamp barycentrics to [0, 1] and renormalise.
                    let w0c = w0.max(0.0);
                    let w1c = w1.max(0.0);
                    let w2c = w2.max(0.0);
                    let sum = w0c + w1c + w2c;
                    if sum > 1e-30 {
                        best_bary = (w0c / sum, w1c / sum, w2c / sum);
                    }
                }
            }
        }

        // Fallback: if no triangle found, use the nearest centroid.
        if best_tri.is_none() {
            let mut nearest_dist = f64::INFINITY;
            for (ti, td) in self.tri_data.iter().enumerate() {
                let d = dist3(p, td.centroid);
                if d < nearest_dist {
                    nearest_dist = d;
                    best_tri = Some(ti);
                }
            }
            if let Some(ti) = best_tri {
                let td = &self.tri_data[ti];
                let bary = barycentric_3d(p, td.v[0], td.v[1], td.v[2], td.normal);
                let w0 = bary.0.max(0.0);
                let w1 = bary.1.max(0.0);
                let w2 = bary.2.max(0.0);
                let sum = w0 + w1 + w2;
                if sum > 1e-30 {
                    best_bary = (w0 / sum, w1 / sum, w2 / sum);
                }
            }
        }

        (best_tri, best_bary)
    }
}

/// Compute barycentric coordinates of point p in triangle (v0, v1, v2).
/// Uses the face normal for consistent area computation in 3D.
fn barycentric_3d(
    p: [f64; 3],
    v0: [f64; 3], v1: [f64; 3], v2: [f64; 3],
    face_normal: [f64; 3],
) -> (f64, f64, f64) {
    let area2 = dot3(face_normal, face_normal).sqrt();
    if area2 < 1e-30 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    // Sub-triangle areas via cross products projected onto face normal.
    let n0 = cross3(sub3(v1, p), sub3(v2, p));
    let n1 = cross3(sub3(v2, p), sub3(v0, p));
    let n2 = cross3(sub3(v0, p), sub3(v1, p));

    let w0 = dot3(n0, face_normal) / (area2 * area2);
    let w1 = dot3(n1, face_normal) / (area2 * area2);
    let w2 = dot3(n2, face_normal) / (area2 * area2);

    (w0, w1, w2)
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s, a[1]*s, a[2]*s] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 { norm3(sub3(a, b)) }
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barycentric_at_vertex() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let n = cross3(sub3(v1, v0), sub3(v2, v0));
        let (w0, w1, w2) = barycentric_3d(v0, v0, v1, v2, n);
        assert!((w0 - 1.0).abs() < 1e-10);
        assert!(w1.abs() < 1e-10);
        assert!(w2.abs() < 1e-10);
    }

    #[test]
    fn barycentric_at_centroid() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let n = cross3(sub3(v1, v0), sub3(v2, v0));
        let c = [1.0/3.0, 1.0/3.0, 0.0];
        let (w0, w1, w2) = barycentric_3d(c, v0, v1, v2, n);
        assert!((w0 - 1.0/3.0).abs() < 1e-10);
        assert!((w1 - 1.0/3.0).abs() < 1e-10);
        assert!((w2 - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn advect_zero_velocity_identity() {
        use crate::mesh_gen::icosphere;
        use cartan_manifolds::sphere::Sphere;

        let mesh = icosphere(2);
        let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
        let nv = coords.len();
        let sl = SemiLagrangian::new(coords, mesh.simplices.clone());

        let q = QFieldDec::random_perturbation(nv, 0.3, 42);
        let vel = VelocityFieldDec::zeros(nv);

        let q_adv = sl.advect(&q, &vel, 0.001);

        // With zero velocity, the advected field should equal the original.
        let diff: f64 = q.q1.iter().zip(&q_adv.q1)
            .chain(q.q2.iter().zip(&q_adv.q2))
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-10, "zero velocity advection should be identity, diff = {diff}");
    }
}
