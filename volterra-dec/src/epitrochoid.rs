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

/// Test whether a 2D point is inside a polygon using the ray-casting algorithm.
///
/// Casts a ray in the +x direction from `point` and counts crossings with
/// the polygon edges. Returns true for an odd crossing count (inside).
fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let [px, py] = point;
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let [xi, yi] = polygon[i];
        let [xj, yj] = polygon[j];
        // Does the edge [j, i] cross the horizontal ray at y=py going right?
        if (yi > py) != (yj > py) {
            let x_cross = xj + (py - yj) * (xi - xj) / (yi - yj);
            if px < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Generate a triangulated epitrochoid mesh with interior points.
///
/// The mesh has:
/// - `n_boundary` vertices on the epitrochoid curve with winding number `q` and scale `r`
/// - Additional interior points on a grid, filtered to be inside the epitrochoid
/// - Delaunay triangulation of all points, with exterior triangles removed
///
/// Returns a [`ConfinedMesh`] with boundary vertices tagged and tangential anchoring
/// directions at each boundary vertex.
///
/// # Nephroid (q = 2.0)
///
/// For the nephroid, set `q = 2.0`. The domain has 2 cusps.
/// A typical call for the confined nephroid run:
///
/// ```rust,ignore
/// let cm = epitrochoid_mesh(2.0, 3.0, 128, 0.3);
/// ```
pub fn epitrochoid_mesh(q: f64, r: f64, n_boundary: usize, interior_spacing: f64) -> ConfinedMesh {
    let (boundary_pts, params) = sample_epitrochoid(q, r, n_boundary);

    // Bounding box of the epitrochoid for grid generation.
    let x_min = boundary_pts.iter().map(|[x, _]| *x).fold(f64::INFINITY, f64::min);
    let x_max = boundary_pts.iter().map(|[x, _]| *x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = boundary_pts.iter().map(|[_, y]| *y).fold(f64::INFINITY, f64::min);
    let y_max = boundary_pts.iter().map(|[_, y]| *y).fold(f64::NEG_INFINITY, f64::max);

    // Start with boundary points; interior points appended after.
    let mut all_points: Vec<[f64; 2]> = boundary_pts.clone();
    let n_boundary_actual = boundary_pts.len();

    // Generate candidate interior points on a regular grid.
    // Use a half-spacing inset margin to avoid placing points too close to the boundary.
    let margin = interior_spacing * 0.5;
    let nx_grid = ((x_max - x_min) / interior_spacing).ceil() as i64 + 1;
    let ny_grid = ((y_max - y_min) / interior_spacing).ceil() as i64 + 1;

    for ix in 0..=nx_grid {
        for iy in 0..=ny_grid {
            let x = x_min + ix as f64 * interior_spacing;
            let y = y_min + iy as f64 * interior_spacing;
            // Must be inside the epitrochoid polygon with inset margin.
            // We test both the raw point and a margin-inset check using a slightly
            // contracted polygon test (done by testing multiple offsets).
            // Simpler: just test whether the point is strictly inside the polygon
            // and not too close to any boundary vertex (Euclidean guard).
            if !point_in_polygon([x, y], &boundary_pts) {
                continue;
            }
            // Reject points too close to the boundary to avoid degenerate triangles.
            let too_close = boundary_pts.iter().any(|[bx, by]| {
                let dx = x - bx;
                let dy = y - by;
                (dx * dx + dy * dy).sqrt() < margin
            });
            if too_close {
                continue;
            }
            all_points.push([x, y]);
        }
    }

    // Delaunay triangulation of all points.
    let flat: Vec<delaunator::Point> = all_points
        .iter()
        .map(|[x, y]| delaunator::Point { x: *x, y: *y })
        .collect();
    let result = delaunator::triangulate(&flat);

    let n_tri = result.triangles.len() / 3;
    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n_tri);
    for t in 0..n_tri {
        triangles.push([
            result.triangles[3 * t],
            result.triangles[3 * t + 1],
            result.triangles[3 * t + 2],
        ]);
    }

    // Filter out triangles whose centroid is outside the epitrochoid polygon.
    // This removes the "convex hull" triangles that fill the space between cusps
    // but lie outside the actual epitrochoid domain.
    let triangles: Vec<[usize; 3]> = triangles
        .into_iter()
        .filter(|[a, b, c]| {
            let cx = (all_points[*a][0] + all_points[*b][0] + all_points[*c][0]) / 3.0;
            let cy = (all_points[*a][1] + all_points[*b][1] + all_points[*c][1]) / 3.0;
            point_in_polygon([cx, cy], &boundary_pts)
        })
        .collect();

    let vertices: Vec<[f64; 2]> = all_points;
    let mesh = cartan_dec::mesh::FlatMesh::from_triangles(vertices, triangles);

    let boundary_vertices: Vec<usize> = (0..n_boundary_actual).collect();
    let anchoring_directions = epitrochoid_tangent_anchoring(q, r, &params);

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

    /// Deliverable 2 test 1: epitrochoid_mesh with nephroid (q=2) builds correctly.
    #[test]
    fn epitrochoid_mesh_nephroid_constructs() {
        let q = 2.0; // nephroid
        let r = 3.0;
        let n_boundary = 80;
        let spacing = 0.4;
        let cm = epitrochoid_mesh(q, r, n_boundary, spacing);

        // Has more vertices than boundary (interior points exist).
        assert!(
            cm.mesh.n_vertices() > n_boundary,
            "should have boundary + interior vertices, got {}",
            cm.mesh.n_vertices()
        );
        // Triangles are non-empty.
        assert!(
            cm.mesh.n_simplices() > 0,
            "should have triangles, got {}",
            cm.mesh.n_simplices()
        );
        // Boundary vertices are tagged.
        assert_eq!(cm.boundary_vertices.len(), n_boundary);
        // Anchoring directions match boundary count and are unit vectors.
        assert_eq!(cm.anchoring_directions.len(), n_boundary);
        for [nx, ny] in &cm.anchoring_directions {
            let norm = (nx * nx + ny * ny).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "anchoring direction should be unit vector, got norm = {norm}"
            );
        }

        // All triangle centroids should be inside the epitrochoid.
        let (boundary_pts, _) = sample_epitrochoid(q, r, n_boundary);
        for &[a, b, c] in &cm.mesh.simplices {
            let pa = cm.mesh.vertices[a];
            let pb = cm.mesh.vertices[b];
            let pc = cm.mesh.vertices[c];
            let cx = (pa.x + pb.x + pc.x) / 3.0;
            let cy = (pa.y + pb.y + pc.y) / 3.0;
            assert!(
                point_in_polygon([cx, cy], &boundary_pts),
                "triangle centroid ({cx:.3}, {cy:.3}) is outside the epitrochoid"
            );
        }
    }

    /// Deliverable 2 test 2: finer spacing yields more vertices.
    #[test]
    fn epitrochoid_mesh_scales_with_spacing() {
        let q = 2.0;
        let r = 3.0;
        let n_boundary = 60;

        let cm_coarse = epitrochoid_mesh(q, r, n_boundary, 0.6);
        let cm_fine   = epitrochoid_mesh(q, r, n_boundary, 0.3);

        assert!(
            cm_fine.mesh.n_vertices() > cm_coarse.mesh.n_vertices(),
            "finer spacing should give more vertices: fine={}, coarse={}",
            cm_fine.mesh.n_vertices(),
            cm_coarse.mesh.n_vertices()
        );
    }
}
