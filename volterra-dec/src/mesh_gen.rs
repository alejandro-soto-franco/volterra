//! Mesh generation for standard geometries.
//!
//! Produces `Mesh<M, 3, 2>` instances for common manifolds using
//! cartan-dec's generic mesh constructor and cartan-remesh's
//! manifold-generic edge splitting for refinement.
//!
//! ## Supported geometries
//!
//! - [`icosphere`]: triangulated unit sphere S^2 via icosahedral subdivision
//! - [`torus_mesh`]: triangulated torus of given radii embedded in R^3

use cartan_dec::Mesh;
use cartan_manifolds::sphere::Sphere;
use nalgebra::SVector;

/// Icosahedral seed: 12 vertices and 20 faces of a regular icosahedron
/// inscribed in the unit sphere.
fn icosahedron_seed() -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // golden ratio
    let a = 1.0 / (1.0 + phi * phi).sqrt(); // normalise to unit sphere
    let b = phi * a;

    #[rustfmt::skip]
    let verts: Vec<SVector<f64, 3>> = vec![
        SVector::from([-a,  b, 0.0]), SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]), SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]), SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]), SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]), SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]), SVector::from([-b, 0.0,  a]),
    ];

    #[rustfmt::skip]
    let faces: Vec<[usize; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    (verts, faces)
}

/// Subdivide each triangle into 4 by inserting midpoints on each edge,
/// projecting the new vertices onto the unit sphere.
fn subdivide_sphere(
    verts: &[SVector<f64, 3>],
    faces: &[[usize; 3]],
) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    use std::collections::HashMap;

    let mut new_verts = verts.to_vec();
    let mut midpoint_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(faces.len() * 4);

    let get_midpoint = |a: usize, b: usize,
                            new_verts: &mut Vec<SVector<f64, 3>>,
                            cache: &mut HashMap<(usize, usize), usize>|
     -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&idx) = cache.get(&key) {
            return idx;
        }
        let mid = (new_verts[a] + new_verts[b]) * 0.5;
        let norm = mid.norm();
        let proj = if norm > 1e-14 { mid / norm } else { mid };
        let idx = new_verts.len();
        new_verts.push(proj);
        cache.insert(key, idx);
        idx
    };

    for &[a, b, c] in faces {
        let ab = get_midpoint(a, b, &mut new_verts, &mut midpoint_cache);
        let bc = get_midpoint(b, c, &mut new_verts, &mut midpoint_cache);
        let ca = get_midpoint(c, a, &mut new_verts, &mut midpoint_cache);
        new_faces.push([a, ab, ca]);
        new_faces.push([b, bc, ab]);
        new_faces.push([c, ca, bc]);
        new_faces.push([ab, bc, ca]);
    }

    (new_verts, new_faces)
}

/// Generate a triangulated unit sphere (S^2) by icosahedral subdivision.
///
/// `refinement` controls the number of subdivision passes:
/// - 0: 12 vertices, 20 faces (raw icosahedron)
/// - 1: 42 vertices, 80 faces
/// - 2: 162 vertices, 320 faces
/// - 3: 642 vertices, 1280 faces
/// - 4: 2562 vertices, 5120 faces
///
/// Returns a `Mesh<Sphere<3>, 3, 2>` with all vertices on the unit sphere.
pub fn icosphere(refinement: usize) -> Mesh<Sphere<3>, 3, 2> {
    let (mut verts, mut faces) = icosahedron_seed();

    for _ in 0..refinement {
        let (v, f) = subdivide_sphere(&verts, &faces);
        verts = v;
        faces = f;
    }

    let manifold = Sphere::<3>;
    Mesh::from_simplices(&manifold, verts, faces)
}

/// Generate a triangulated torus embedded in R^3.
///
/// The torus has major radius `R` (distance from the centre of the tube
/// to the centre of the torus) and minor radius `r` (radius of the tube).
///
/// `n_major` and `n_minor` control the resolution in the major and minor
/// directions respectively. Total vertices = n_major * n_minor, total
/// triangles = 2 * n_major * n_minor.
///
/// The mesh is constructed as `Mesh<Euclidean<3>, 3, 2>` (embedded in R^3)
/// since cartan does not have a native Torus manifold type. The Gaussian
/// curvature must be computed externally from the vertex positions.
pub fn torus_mesh(
    major_radius: f64,
    minor_radius: f64,
    n_major: usize,
    n_minor: usize,
) -> Mesh<cartan_manifolds::euclidean::Euclidean<3>, 3, 2> {
    let n_verts = n_major * n_minor;
    let mut verts: Vec<SVector<f64, 3>> = Vec::with_capacity(n_verts);

    for i in 0..n_major {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / n_major as f64;
        for j in 0..n_minor {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_minor as f64;
            let x = (major_radius + minor_radius * phi.cos()) * theta.cos();
            let y = (major_radius + minor_radius * phi.cos()) * theta.sin();
            let z = minor_radius * phi.sin();
            verts.push(SVector::from([x, y, z]));
        }
    }

    // Triangulate: each quad (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) -> 2 triangles.
    // Periodic in both directions.
    let mut faces: Vec<[usize; 3]> = Vec::with_capacity(2 * n_verts);
    for i in 0..n_major {
        let i1 = (i + 1) % n_major;
        for j in 0..n_minor {
            let j1 = (j + 1) % n_minor;
            let v00 = i * n_minor + j;
            let v10 = i1 * n_minor + j;
            let v01 = i * n_minor + j1;
            let v11 = i1 * n_minor + j1;
            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);
        }
    }

    let manifold = cartan_manifolds::euclidean::Euclidean::<3>;
    Mesh::from_simplices(&manifold, verts, faces)
}

/// Compute the Gaussian curvature at each vertex of a torus mesh.
///
/// For a torus with major radius R and minor radius r, the Gaussian
/// curvature is K = cos(phi) / (r * (R + r cos(phi))), where phi is
/// the minor angle at the vertex.
///
/// Returns a vector of K values, one per vertex, in the same order
/// as the mesh vertices.
pub fn torus_gaussian_curvature(
    major_radius: f64,
    minor_radius: f64,
    n_major: usize,
    n_minor: usize,
) -> Vec<f64> {
    let mut curvatures = Vec::with_capacity(n_major * n_minor);
    for _i in 0..n_major {
        for j in 0..n_minor {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / n_minor as f64;
            let k = phi.cos() / (minor_radius * (major_radius + minor_radius * phi.cos()));
            curvatures.push(k);
        }
    }
    curvatures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_level_0() {
        let mesh = icosphere(0);
        assert_eq!(mesh.n_vertices(), 12);
        assert_eq!(mesh.n_simplices(), 20);
    }

    #[test]
    fn icosphere_level_3() {
        let mesh = icosphere(3);
        assert_eq!(mesh.n_vertices(), 642);
        assert_eq!(mesh.n_simplices(), 1280);
    }

    #[test]
    fn icosphere_vertices_on_unit_sphere() {
        let mesh = icosphere(2);
        for v in &mesh.vertices {
            let r = v.norm();
            assert!(
                (r - 1.0).abs() < 1e-12,
                "vertex not on unit sphere: r = {r}"
            );
        }
    }

    #[test]
    fn icosphere_euler_characteristic() {
        // S^2: chi = V - E + F = 2.
        let mesh = icosphere(2);
        assert_eq!(mesh.euler_characteristic(), 2);
    }

    #[test]
    fn torus_mesh_counts() {
        let mesh = torus_mesh(3.0, 1.0, 20, 10);
        assert_eq!(mesh.n_vertices(), 200);
        assert_eq!(mesh.n_simplices(), 400);
    }

    #[test]
    fn torus_euler_characteristic() {
        // T^2: chi = V - E + F = 0.
        let mesh = torus_mesh(3.0, 1.0, 12, 8);
        assert_eq!(mesh.euler_characteristic(), 0);
    }

    #[test]
    fn torus_curvature_sign() {
        let curvatures = torus_gaussian_curvature(3.0, 1.0, 20, 10);
        // Outer equator (phi=0): K > 0. Inner equator (phi=pi): K < 0.
        assert!(curvatures[0] > 0.0, "outer equator should have K > 0");
        assert!(curvatures[5] < 0.0, "inner equator should have K < 0");
    }
}
