//! Meshes of implicit surfaces, and the topology they must reproduce.

use volterra_dec::implicit::genus2_mesh;

/// A genus-2 surface has `chi = -2`, and the mesh must say so.
///
/// The Euler characteristic is combinatorial, so this checks the marching and
/// the remeshing together: neither is allowed to open a hole, weld two sheets
/// or drop a triangle.
#[test]
fn the_genus_two_mesh_has_euler_characteristic_minus_two() {
    let (verts, tris) = genus2_mesh(0.24, 26, 6);
    let mut edges = std::collections::HashSet::new();
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    let chi = verts.len() as i64 - edges.len() as i64 + tris.len() as i64;
    assert_eq!(chi, -2, "V {} - E {} + F {}", verts.len(), edges.len(), tris.len());
}

/// Every edge has two faces and every triangle has area.
///
/// A cut landing exactly on a grid corner puts two vertices at one point, and
/// the triangle between them has no area. That leaves `chi` correct and breaks
/// the angle-defect identity, so counting `chi` alone would not catch it.
#[test]
fn the_genus_two_mesh_is_closed_and_has_no_degenerate_triangle() {
    let (verts, tris) = genus2_mesh(0.24, 26, 6);
    let mut faces_on_edge = std::collections::HashMap::new();
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *faces_on_edge.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    let open = faces_on_edge.values().filter(|&&c| c != 2).count();
    assert_eq!(open, 0, "{open} edge(s) without exactly two faces");

    for t in &tris {
        assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2], "repeated vertex in {t:?}");
        let (p, q, r) = (verts[t[0]], verts[t[1]], verts[t[2]]);
        let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let c = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let area = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        assert!(area > 1e-12, "triangle {t:?} has area {area:e}");
    }
}

/// The angle defects sum to `2 pi chi`.
///
/// Discrete Gauss-Bonnet is an identity for a closed triangulation, so this
/// reads the topology back out of the geometry and fails on exactly the
/// degeneracies the count of `chi` cannot see.
#[test]
fn the_genus_two_mesh_satisfies_gauss_bonnet() {
    let (verts, tris) = genus2_mesh(0.24, 26, 6);
    let mut angle = vec![0.0f64; verts.len()];
    for t in &tris {
        for a in 0..3 {
            let (i, j, k) = (t[a], t[(a + 1) % 3], t[(a + 2) % 3]);
            let (p, q, r) = (verts[i], verts[j], verts[k]);
            let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
            let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
            angle[i] += d.clamp(-1.0, 1.0).acos();
        }
    }
    let total: f64 = angle.iter().map(|a| std::f64::consts::TAU - a).sum();
    let want = -4.0 * std::f64::consts::PI;
    assert!((total - want).abs() < 1e-8, "defect sum {total}, wanted {want}");
}

/// The cotangent weights are non-negative, which is what DEC needs.
///
/// Marching output has slivers with angles under a tenth of a degree, and the
/// remeshing exists to remove them. Without it the genus-2 mesh had over two
/// thousand non-Delaunay edges.
#[test]
fn the_genus_two_mesh_is_delaunay() {
    use cartan_manifolds::euclidean::Euclidean;
    let (verts, tris) = genus2_mesh(0.24, 26, 6);
    let sv: Vec<nalgebra::SVector<f64, 3>> =
        verts.iter().map(|p| nalgebra::SVector::from(*p)).collect();
    let mesh = cartan_dec::mesh::Mesh::from_simplices(&Euclidean::<3>, sv, tris);
    let q = cartan_dec::mesh_quality::quality_report(&mesh, &Euclidean::<3>);
    assert_eq!(q.non_delaunay_edges, 0, "non-Delaunay edges remain");
    assert!(
        q.min_angle.to_degrees() > 15.0,
        "min angle {:.2} deg is too small for the cotangent weights",
        q.min_angle.to_degrees()
    );
}
