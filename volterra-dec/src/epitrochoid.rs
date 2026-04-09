//! Epitrochoid domain mesh generation for confined active nematics.
//!
//! Generates triangulated 2D domains bounded by epitrochoid curves
//! (cardioid, nephroid, trefoiloid, etc.) with boundary vertex tagging
//! for anchoring boundary conditions.
//!
//! ## Epitrochoid parametrisation (from arXiv:2503.10880 Eqs. 3-4)
//!
//! ```text
//! x(u) = r/(2q) [(2q-1) cos(u) + cos((2q-1)u)]
//! y(u) = r/(2q) [(2q-1) sin(u) + sin((2q-1)u)]
//! ```
//!
//! for u in [0, 2pi]. The half-integer winding number q determines the geometry:
//! - q = 3/2: cardioid (1 cusp)
//! - q = 2:   nephroid (2 cusps)
//! - q = 5/2: trefoiloid (3 cusps)
//!
//! Number of cusps = 2(q - 1).
//!
//! ## Disk (special case)
//!
//! A circular disk of radius r with winding number q in the anchoring
//! direction is also supported (used for the steady-winding BC benchmarks).

use cartan_dec::mesh::FlatMesh;

/// Result of epitrochoid (or disk) mesh generation.
pub struct ConfinedMesh {
    /// The triangle mesh.
    pub mesh: FlatMesh,
    /// Indices of boundary vertices (on the domain boundary).
    pub boundary_vertices: Vec<usize>,
    /// Anchoring direction (n_x, n_y) at each boundary vertex.
    /// For tangential anchoring: n is the boundary tangent.
    /// For steady-winding: n(theta) = (-sin(q*theta), cos(q*theta)).
    pub anchoring_directions: Vec<[f64; 2]>,
}

/// Sample points along the epitrochoid boundary curve.
///
/// Returns `n_boundary` points (x, y) and the corresponding parameter values u.
pub fn sample_epitrochoid(q: f64, r: f64, n_boundary: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut points = Vec::with_capacity(n_boundary);
    let mut params = Vec::with_capacity(n_boundary);
    let coeff = 2.0 * q - 1.0;

    for i in 0..n_boundary {
        let u = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
        let x = r / (2.0 * q) * (coeff * u.cos() + ((coeff) * u).cos());
        let y = r / (2.0 * q) * (coeff * u.sin() + ((coeff) * u).sin());
        points.push([x, y]);
        params.push(u);
    }

    (points, params)
}

/// Sample points along a circular boundary.
pub fn sample_disk(r: f64, n_boundary: usize) -> Vec<[f64; 2]> {
    (0..n_boundary)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
            [r * theta.cos(), r * theta.sin()]
        })
        .collect()
}

/// Compute the tangential anchoring direction at each boundary point
/// of an epitrochoid curve.
///
/// The tangent vector dx/du, dy/du is normalised to unit length.
pub fn epitrochoid_tangent_anchoring(q: f64, r: f64, params: &[f64]) -> Vec<[f64; 2]> {
    let coeff = 2.0 * q - 1.0;
    params
        .iter()
        .map(|&u| {
            let dx = r / (2.0 * q) * (-coeff * u.sin() - coeff * (coeff * u).sin());
            let dy = r / (2.0 * q) * (coeff * u.cos() + coeff * (coeff * u).cos());
            let norm = (dx * dx + dy * dy).sqrt();
            if norm < 1e-14 {
                // At a cusp, the tangent is undefined. Use a neighbouring direction.
                [0.0, 1.0]
            } else {
                [dx / norm, dy / norm]
            }
        })
        .collect()
}

/// Compute the steady-winding anchoring direction on a circular boundary.
///
/// n(theta) = (-sin(q*theta), cos(q*theta))
///
/// This imposes a total winding number 2*pi*q in the director at the boundary,
/// forcing n = 2q excess +1/2 defects in the interior.
pub fn steady_winding_anchoring(q: f64, n_boundary: usize) -> Vec<[f64; 2]> {
    (0..n_boundary)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
            let angle = q * theta;
            [-angle.sin(), angle.cos()]
        })
        .collect()
}

/// Generate a triangulated disk mesh with interior points.
///
/// The mesh has:
/// - `n_boundary` vertices on the circle of radius `r`
/// - Additional interior points on a grid, filtered to be inside the circle
/// - Delaunay triangulation of all points
///
/// Returns `ConfinedMesh` with boundary vertices tagged.
pub fn disk_mesh(r: f64, q_winding: f64, n_boundary: usize, interior_spacing: f64) -> ConfinedMesh {
    let boundary_pts = sample_disk(r, n_boundary);

    // Generate interior points on a grid.
    let mut all_points: Vec<[f64; 2]> = boundary_pts.clone();
    let n_boundary_actual = boundary_pts.len();

    let n_grid = (2.0 * r / interior_spacing).ceil() as i32;
    for ix in -n_grid..=n_grid {
        for iy in -n_grid..=n_grid {
            let x = ix as f64 * interior_spacing;
            let y = iy as f64 * interior_spacing;
            let dist2 = x * x + y * y;
            // Inside the circle (with margin to avoid near-boundary points).
            if dist2 < (r - interior_spacing * 0.5).powi(2) {
                all_points.push([x, y]);
            }
        }
    }

    // Delaunay triangulation.
    let flat: Vec<delaunator::Point> = all_points
        .iter()
        .map(|[x, y]| delaunator::Point { x: *x, y: *y })
        .collect();
    let result = delaunator::triangulate(&flat);

    // Convert to FlatMesh triangle indices.
    let n_tri = result.triangles.len() / 3;
    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n_tri);
    for t in 0..n_tri {
        triangles.push([
            result.triangles[3 * t],
            result.triangles[3 * t + 1],
            result.triangles[3 * t + 2],
        ]);
    }

    // Filter out triangles whose centroid is outside the disk.
    let triangles: Vec<[usize; 3]> = triangles
        .into_iter()
        .filter(|[a, b, c]| {
            let cx = (all_points[*a][0] + all_points[*b][0] + all_points[*c][0]) / 3.0;
            let cy = (all_points[*a][1] + all_points[*b][1] + all_points[*c][1]) / 3.0;
            cx * cx + cy * cy < r * r * 1.01 // small tolerance
        })
        .collect();

    let vertices: Vec<[f64; 2]> = all_points;
    let mesh = FlatMesh::from_triangles(vertices, triangles);

    let boundary_vertices: Vec<usize> = (0..n_boundary_actual).collect();
    let anchoring_directions = steady_winding_anchoring(q_winding, n_boundary_actual);

    ConfinedMesh {
        mesh,
        boundary_vertices,
        anchoring_directions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epitrochoid_cardioid_samples_correct() {
        let q = 1.5; // cardioid
        let r = 1.0;
        let (pts, _params) = sample_epitrochoid(q, r, 200);
        assert_eq!(pts.len(), 200);
        // Cardioid with q=3/2, r=1: the cusp is at u=pi.
        // x(pi) = 1/3 * (2*cos(pi) + cos(2*pi)) = 1/3*(-2+1) = -1/3
        // y(pi) = 1/3 * (2*sin(pi) + sin(2*pi)) = 0
        // Check the point nearest u=pi (index 100 of 200).
        let cusp = pts[100];
        assert!(
            (cusp[0] - (-1.0 / 3.0)).abs() < 0.05,
            "cusp x should be near -1/3, got {}",
            cusp[0]
        );
        assert!(
            cusp[1].abs() < 0.05,
            "cusp y should be near 0, got {}",
            cusp[1]
        );
    }

    #[test]
    fn disk_boundary_on_circle() {
        let pts = sample_disk(5.0, 100);
        for [x, y] in &pts {
            let r = (x * x + y * y).sqrt();
            assert!((r - 5.0).abs() < 1e-10, "boundary point at r = {r}, expected 5.0");
        }
    }

    #[test]
    fn steady_winding_unit_vectors() {
        let dirs = steady_winding_anchoring(1.5, 100);
        for [nx, ny] in &dirs {
            let norm = (nx * nx + ny * ny).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "anchoring direction should be unit vector, got norm = {norm}"
            );
        }
    }

    #[test]
    fn disk_mesh_constructs() {
        let cm = disk_mesh(5.0, 1.5, 64, 0.5);
        assert!(cm.mesh.n_vertices() > 64, "should have boundary + interior vertices");
        assert!(cm.mesh.n_simplices() > 0, "should have triangles");
        assert_eq!(cm.boundary_vertices.len(), 64);
        assert_eq!(cm.anchoring_directions.len(), 64);
    }
}
