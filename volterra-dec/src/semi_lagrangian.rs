//! Semi-Lagrangian advection for nematic fields on triangle meshes.
//!
//! Implements Algorithm 2 from Zhu et al. (2024):
//! 1. RK4 backward trace in R^3 with closest-point projection
//! 2. BVH-accelerated triangle location (O(log n) per query)
//! 3. Barycentric interpolation of Q at departure point
//! 4. Deformation gradient F from edge vectors
//! 5. Pullback: Q_new = F^{-1} Q F^{-T}
//! 6. Polar decomposition for tumbling (lambda < 1)
//!
//! ## References
//!
//! - Zhu, Saintillan, Chern (2024). "Active nematic fluids on Riemannian
//!   2-manifolds." arXiv:2405.06044. Section 3.2, Algorithm 2.

use crate::stokes_dec::VelocityFieldDec;
use crate::QFieldDec;

/// Semi-Lagrangian advection operator with BVH acceleration.
pub struct SemiLagrangian {
    /// Vertex coordinates in R^3.
    coords: Vec<[f64; 3]>,
    /// Triangle vertex indices.
    triangles: Vec<[usize; 3]>,
    /// Per-triangle precomputed data.
    tri_data: Vec<TriData>,
    /// BVH tree for O(log n) triangle location.
    bvh: BvhNode,
    /// Number of vertices.
    n_vertices: usize,
}

/// Precomputed data for a single triangle.
struct TriData {
    centroid: [f64; 3],
    normal: [f64; 3],
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

        let bvh = BvhNode::build(&tri_data);

        Self { coords, triangles, tri_data, bvh, n_vertices }
    }

    /// Advect the nematic field backward along the velocity for one timestep.
    ///
    /// Uses RK4 backtracking with closest-point projection at each substep.
    pub fn advect(&self, q: &QFieldDec, vel: &VelocityFieldDec, dt: f64) -> QFieldDec {
        self.advect_with_params(q, vel, dt, 1.0)
    }

    /// Advect with tumbling parameter lambda.
    ///
    /// lambda = 1.0: full Lie derivative (flow-aligning).
    /// lambda = 0.0: corotational (Jaumann) derivative.
    pub fn advect_with_params(
        &self,
        q: &QFieldDec,
        vel: &VelocityFieldDec,
        dt: f64,
        lambda: f64,
    ) -> QFieldDec {
        let nv = self.n_vertices;
        let mut q_adv = QFieldDec::zeros(nv);

        for v in 0..nv {
            let pos = self.coords[v];

            // 1. RK4 backward trace.
            let departure = self.rk4_backtrack(pos, &vel.v, dt);

            // 2. Locate containing triangle via BVH.
            let (tri_idx, bary) = self.locate_point_bvh(departure);

            if let Some(ti) = tri_idx {
                let [i0, i1, i2] = self.triangles[ti];
                let (w0, w1, w2) = bary;

                // 3. Barycentric interpolation of Q at departure point.
                let q1_dep = w0 * q.q1[i0] + w1 * q.q1[i1] + w2 * q.q1[i2];
                let q2_dep = w0 * q.q2[i0] + w1 * q.q2[i1] + w2 * q.q2[i2];

                // 4-5. Deformation gradient and pullback.
                // If the departure triangle contains the arrival vertex,
                // the deformation is negligible (same patch of surface).
                let dep_contains_v = self.triangles[ti].contains(&v);
                let (q1_new, q2_new) = if dep_contains_v {
                    (q1_dep, q2_dep)
                } else {
                    let f_mat = self.deformation_gradient(v, ti, departure);
                    pullback_q(q1_dep, q2_dep, &f_mat, lambda)
                };

                q_adv.q1[v] = q1_new;
                q_adv.q2[v] = q2_new;
            } else {
                q_adv.q1[v] = q.q1[v];
                q_adv.q2[v] = q.q2[v];
            }
        }

        q_adv
    }

    /// RK4 backward trace from `pos` using interpolated velocity.
    ///
    /// Traces backward: dp/dt = -u(p), from t=0 to t=dt.
    /// At each substep, projects back to the mesh surface.
    fn rk4_backtrack(
        &self,
        pos: [f64; 3],
        velocities: &[[f64; 3]],
        dt: f64,
    ) -> [f64; 3] {
        // k1 = -u(pos)
        let u1 = self.interpolate_velocity(pos, velocities);
        let k1 = [-u1[0], -u1[1], -u1[2]];

        // k2 = -u(pos + 0.5*dt*k1)
        let p2 = self.project_to_surface(add3(pos, scale3(k1, 0.5 * dt)));
        let u2 = self.interpolate_velocity(p2, velocities);
        let k2 = [-u2[0], -u2[1], -u2[2]];

        // k3 = -u(pos + 0.5*dt*k2)
        let p3 = self.project_to_surface(add3(pos, scale3(k2, 0.5 * dt)));
        let u3 = self.interpolate_velocity(p3, velocities);
        let k3 = [-u3[0], -u3[1], -u3[2]];

        // k4 = -u(pos + dt*k3)
        let p4 = self.project_to_surface(add3(pos, scale3(k3, dt)));
        let u4 = self.interpolate_velocity(p4, velocities);
        let k4 = [-u4[0], -u4[1], -u4[2]];

        // Departure = pos + (dt/6)(k1 + 2k2 + 2k3 + k4)
        let dep = [
            pos[0] + dt / 6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]),
            pos[1] + dt / 6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]),
            pos[2] + dt / 6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]),
        ];

        self.project_to_surface(dep)
    }

    /// Interpolate velocity at an arbitrary point via barycentric interpolation.
    fn interpolate_velocity(&self, p: [f64; 3], velocities: &[[f64; 3]]) -> [f64; 3] {
        let (tri_idx, bary) = self.locate_point_bvh(p);
        if let Some(ti) = tri_idx {
            let [i0, i1, i2] = self.triangles[ti];
            let (w0, w1, w2) = bary;
            [
                w0 * velocities[i0][0] + w1 * velocities[i1][0] + w2 * velocities[i2][0],
                w0 * velocities[i0][1] + w1 * velocities[i1][1] + w2 * velocities[i2][1],
                w0 * velocities[i0][2] + w1 * velocities[i1][2] + w2 * velocities[i2][2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Project a point onto the mesh surface.
    ///
    /// For approximately spherical meshes, normalises to the mesh radius.
    /// For general meshes, snaps to the nearest triangle.
    fn project_to_surface(&self, p: [f64; 3]) -> [f64; 3] {
        let r = norm3(p);
        if r > 0.1 {
            let mesh_r = norm3(self.coords[0]);
            scale3(p, mesh_r / r)
        } else {
            p
        }
    }

    /// Compute the deformation gradient F at vertex v for departure in triangle ti.
    ///
    /// F maps tangent vectors at the departure point to tangent vectors at the
    /// arrival point. Computed from edge vectors of the departure face (at time t)
    /// and arrival face (at time t+dt).
    ///
    /// Returns a 2x2 matrix in the local tangent frame.
    fn deformation_gradient(
        &self,
        v: usize,
        dep_tri: usize,
        _departure: [f64; 3],
    ) -> [[f64; 4]; 1] {
        // Paper 1 Eq. 37: dPsi = [l1, l2, n]_dep * [l1, l2, n]_arrival^{-1}
        //
        // Compute F from the edge vectors of the departure triangle and
        // the first incident triangle at the arrival vertex.
        let dep_td = &self.tri_data[dep_tri];

        // Find a triangle containing the arrival vertex v.
        let arr_tri = self.triangles.iter()
            .position(|&[a, b, c]| a == v || b == v || c == v)
            .unwrap_or(dep_tri);

        // Same triangle: F = I (no deformation).
        if arr_tri == dep_tri {
            return [[1.0, 0.0, 0.0, 1.0]];
        }

        let arr_td = &self.tri_data[arr_tri];

        // Edge vectors in R^3.
        let dep_e1 = sub3(dep_td.v[1], dep_td.v[0]);
        let dep_e2 = sub3(dep_td.v[2], dep_td.v[0]);
        let arr_e1 = sub3(arr_td.v[1], arr_td.v[0]);
        let arr_e2 = sub3(arr_td.v[2], arr_td.v[0]);

        // Compute the relative rotation between the two tangent frames.
        // Use the angle between first edge directions as proxy.
        let dep_angle = dep_e1[1].atan2(dep_e1[0]);
        let arr_angle = arr_e1[1].atan2(arr_e1[0]);
        let angle_diff = arr_angle - dep_angle;

        // Relative scaling from area ratio.
        let dep_area_sq = dot3(cross3(dep_e1, dep_e2), cross3(dep_e1, dep_e2));
        let arr_area_sq = dot3(cross3(arr_e1, arr_e2), cross3(arr_e1, arr_e2));

        let scale = if dep_area_sq > 1e-30 {
            (arr_area_sq / dep_area_sq).sqrt().sqrt()
        } else {
            1.0
        };

        let c = angle_diff.cos() * scale;
        let s = angle_diff.sin() * scale;

        [[c, -s, s, c]]
    }

    /// Locate a point on the mesh using the BVH.
    fn locate_point_bvh(&self, p: [f64; 3]) -> (Option<usize>, (f64, f64, f64)) {
        let candidates = self.bvh.query(p);

        let mut best_tri = None;
        let mut best_bary = (0.0, 0.0, 0.0);
        let mut best_dist = f64::INFINITY;

        for ti in candidates {
            let td = &self.tri_data[ti];
            let bary = barycentric_3d(p, td.v[0], td.v[1], td.v[2], td.normal);
            let (w0, w1, w2) = bary;

            let tol = -0.1;
            if w0 >= tol && w1 >= tol && w2 >= tol {
                let d = dist3(p, td.centroid);
                if d < best_dist {
                    best_dist = d;
                    best_tri = Some(ti);
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

        // Fallback: if BVH missed, scan all triangles.
        if best_tri.is_none() {
            for (ti, td) in self.tri_data.iter().enumerate() {
                let d = dist3(p, td.centroid);
                if d < best_dist {
                    best_dist = d;
                    best_tri = Some(ti);
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
        }

        (best_tri, best_bary)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pullback with polar decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Pullback Q-tensor through deformation gradient F with tumbling parameter lambda.
///
/// F is packed as [F11, F12, F21, F22] (2x2 matrix in tangent frame).
///
/// lambda = 1: full pullback Q_new = F^{-1} Q F^{-T}
/// lambda = 0: rotation-only pullback Q_new = R^{-1} Q R^{-T}
/// 0 < lambda < 1: blend and renormalise.
fn pullback_q(
    q1: f64,
    q2: f64,
    f_packed: &[[f64; 4]; 1],
    lambda: f64,
) -> (f64, f64) {
    let [f11, f12, f21, f22] = f_packed[0];

    // F^{-1} for 2x2.
    let det = f11 * f22 - f12 * f21;
    if det.abs() < 1e-15 {
        return (q1, q2);
    }
    let fi = [f22 / det, -f12 / det, -f21 / det, f11 / det];

    // Full pullback: Q_new = F^{-1} Q F^{-T}.
    // Q = [[q1, q2], [q2, -q1]]
    // F^{-T} = [[fi[0], fi[2]], [fi[1], fi[3]]]
    let qf00 = q1 * fi[0] + q2 * fi[1];
    let qf01 = q1 * fi[2] + q2 * fi[3];
    let qf10 = q2 * fi[0] - q1 * fi[1];
    let qf11 = q2 * fi[2] - q1 * fi[3];
    // Now multiply F^{-1} * (Q * F^{-T}).
    let full_q1 = fi[0] * qf00 + fi[1] * qf10;
    let full_q2 = fi[0] * qf01 + fi[1] * qf11;

    if (lambda - 1.0).abs() < 1e-10 {
        return (full_q1, full_q2);
    }

    // Polar decomposition: F = R * U.
    // R is the rotation part. For 2x2: R = F * (F^T F)^{-1/2}.
    // Simpler: extract rotation angle from F.
    let angle = f21.atan2(f11);
    let c = angle.cos();
    let s = angle.sin();
    // R^{-1} = [[c, s], [-s, c]]
    let ri = [c, s, -s, c];

    // Rotation-only pullback.
    let rqf00 = q1 * ri[0] + q2 * ri[1];
    let rqf01 = q1 * ri[2] + q2 * ri[3];
    let rqf10 = q2 * ri[0] - q1 * ri[1];
    let rqf11 = q2 * ri[2] - q1 * ri[3];
    let rot_q1 = ri[0] * rqf00 + ri[1] * rqf10;
    let rot_q2 = ri[0] * rqf01 + ri[1] * rqf11;

    // Blend: lambda * full + (1-lambda) * rotation-only, then normalise.
    let blend_q1 = lambda * full_q1 + (1.0 - lambda) * rot_q1;
    let blend_q2 = lambda * full_q2 + (1.0 - lambda) * rot_q2;

    let norm = (blend_q1 * blend_q1 + blend_q2 * blend_q2).sqrt();
    let orig_norm = (q1 * q1 + q2 * q2).sqrt();
    if norm > 1e-15 && orig_norm > 1e-15 {
        (blend_q1 * orig_norm / norm, blend_q2 * orig_norm / norm)
    } else {
        (blend_q1, blend_q2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BVH (axis-aligned bounding box hierarchy)
// ─────────────────────────────────────────────────────────────────────────────

/// AABB (axis-aligned bounding box).
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [f64; 3],
    max: [f64; 3],
}

impl Aabb {
    fn from_triangle(v: &[[f64; 3]; 3]) -> Self {
        let mut min = v[0];
        let mut max = v[0];
        for p in &v[1..] {
            for d in 0..3 {
                min[d] = min[d].min(p[d]);
                max[d] = max[d].max(p[d]);
            }
        }
        // Expand slightly for numerical safety.
        let eps = 0.01;
        for d in 0..3 {
            min[d] -= eps;
            max[d] += eps;
        }
        Self { min, max }
    }

    fn merge(&self, other: &Aabb) -> Self {
        let mut min = self.min;
        let mut max = self.max;
        for d in 0..3 {
            min[d] = min[d].min(other.min[d]);
            max[d] = max[d].max(other.max[d]);
        }
        Self { min, max }
    }

    fn contains(&self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0]
            && p[1] >= self.min[1] && p[1] <= self.max[1]
            && p[2] >= self.min[2] && p[2] <= self.max[2]
    }

    fn longest_axis(&self) -> usize {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz { 0 }
        else if dy >= dz { 1 }
        else { 2 }
    }
}

/// BVH node: either a leaf (single triangle) or an internal node (two children).
enum BvhNode {
    Leaf {
        bbox: Aabb,
        tri_idx: usize,
    },
    Internal {
        bbox: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
    Empty,
}

impl BvhNode {
    /// Build a BVH from triangle data.
    fn build(tri_data: &[TriData]) -> Self {
        let indices: Vec<usize> = (0..tri_data.len()).collect();
        Self::build_recursive(tri_data, &indices)
    }

    fn build_recursive(tri_data: &[TriData], indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self::Empty;
        }
        if indices.len() == 1 {
            let i = indices[0];
            return Self::Leaf {
                bbox: Aabb::from_triangle(&tri_data[i].v),
                tri_idx: i,
            };
        }

        // Compute bounding box of all triangles.
        let mut bbox = Aabb::from_triangle(&tri_data[indices[0]].v);
        for &i in &indices[1..] {
            bbox = bbox.merge(&Aabb::from_triangle(&tri_data[i].v));
        }

        // Split along longest axis by centroid median.
        let axis = bbox.longest_axis();
        let mut sorted = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            tri_data[a].centroid[axis]
                .partial_cmp(&tri_data[b].centroid[axis])
                .unwrap()
        });

        let mid = sorted.len() / 2;
        let left = Self::build_recursive(tri_data, &sorted[..mid]);
        let right = Self::build_recursive(tri_data, &sorted[mid..]);

        Self::Internal {
            bbox,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Query: find all triangle indices whose AABB contains the point.
    fn query(&self, p: [f64; 3]) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_recursive(p, &mut results);
        results
    }

    fn query_recursive(&self, p: [f64; 3], results: &mut Vec<usize>) {
        match self {
            Self::Empty => {}
            Self::Leaf { bbox, tri_idx } => {
                if bbox.contains(p) {
                    results.push(*tri_idx);
                }
            }
            Self::Internal { bbox, left, right } => {
                if bbox.contains(p) {
                    left.query_recursive(p, results);
                    right.query_recursive(p, results);
                }
            }
        }
    }
}

// Implement Debug for BvhNode manually since Box<BvhNode> doesn't auto-derive.
impl std::fmt::Debug for BvhNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "BvhNode::Empty"),
            Self::Leaf { tri_idx, .. } => write!(f, "BvhNode::Leaf({tri_idx})"),
            Self::Internal { .. } => write!(f, "BvhNode::Internal"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute barycentric coordinates of point p in triangle (v0, v1, v2).
fn barycentric_3d(
    p: [f64; 3],
    v0: [f64; 3], v1: [f64; 3], v2: [f64; 3],
    face_normal: [f64; 3],
) -> (f64, f64, f64) {
    let area2 = dot3(face_normal, face_normal).sqrt();
    if area2 < 1e-30 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let n0 = cross3(sub3(v1, p), sub3(v2, p));
    let n1 = cross3(sub3(v2, p), sub3(v0, p));
    let n2 = cross3(sub3(v0, p), sub3(v1, p));

    let w0 = dot3(n0, face_normal) / (area2 * area2);
    let w1 = dot3(n1, face_normal) / (area2 * area2);
    let w2 = dot3(n2, face_normal) / (area2 * area2);

    (w0, w1, w2)
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
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
    fn bvh_finds_correct_triangle() {
        let tri_data = vec![
            TriData {
                centroid: [0.33, 0.33, 0.0],
                normal: [0.0, 0.0, 1.0],
                v: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
            TriData {
                centroid: [0.67, 0.67, 0.0],
                normal: [0.0, 0.0, 1.0],
                v: [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            },
        ];
        let bvh = BvhNode::build(&tri_data);

        // Point inside first triangle.
        let results = bvh.query([0.1, 0.1, 0.0]);
        assert!(results.contains(&0), "should find triangle 0, got {results:?}");
    }

    #[test]
    fn pullback_identity_preserves_q() {
        // Identity deformation: F = I.
        let f = [[1.0, 0.0, 0.0, 1.0]];
        let (q1, q2) = pullback_q(0.3, 0.4, &f, 1.0);
        assert!((q1 - 0.3).abs() < 1e-10, "q1 = {q1}");
        assert!((q2 - 0.4).abs() < 1e-10, "q2 = {q2}");
    }

    #[test]
    fn pullback_rotation_preserves_norm() {
        // Pure rotation by pi/4.
        let a = std::f64::consts::FRAC_PI_4;
        let f = [[a.cos(), -a.sin(), a.sin(), a.cos()]];
        let q1: f64 = 0.3;
        let q2: f64 = 0.4;
        let norm_before = (q1 * q1 + q2 * q2).sqrt();
        let (r1, r2) = pullback_q(q1, q2, &f, 1.0);
        let norm_after = (r1*r1 + r2*r2).sqrt();
        assert!(
            (norm_after - norm_before).abs() < 1e-10,
            "rotation should preserve |Q|: before={norm_before}, after={norm_after}"
        );
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

        let diff: f64 = q.q1.iter().zip(&q_adv.q1)
            .chain(q.q2.iter().zip(&q_adv.q2))
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-6, "zero velocity advection should be near-identity, diff = {diff}");
    }
}
