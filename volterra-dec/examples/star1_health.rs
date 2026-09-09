//! Negative `star1` entries and obtuse triangles, per production geometry.
//!
//! `star1[e] = |dual edge| / |primal edge|` is the unique diagonal Hodge star on
//! 1-forms in two dimensions, and it turns negative exactly when a triangle is
//! obtuse. `cartan-mimetic`'s `local_star` is the replacement, consistent and
//! positive definite at every degree, but `star1` is read in five files here, so
//! swapping a diagonal `Vec<f64>` for a sparse matrix reaches far beyond the
//! Stokes solver.
//!
//! The spec therefore measures before it changes. A cusp is where a mesher
//! produces an obtuse triangle, so the expectation is a positive count on the
//! cusped families and zero on the circle. Whatever comes out goes into
//! `2026-09-09-screened-stokes-and-mimetic-star-design.md` and decides whether
//! the swap is needed at production resolution.

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_dec::epitrochoid::{disk_mesh, epitrochoid_mesh};

/// The angles of a triangle, in radians.
fn angles(p: [f64; 3], q: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let d = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let (a, b, c) = (d(q, r), d(r, p), d(p, q));
    let ang = |o: f64, x: f64, y: f64| {
        let v = ((x * x + y * y - o * o) / (2.0 * x * y)).clamp(-1.0, 1.0);
        v.acos()
    };
    [ang(a, b, c), ang(b, c, a), ang(c, a, b)]
}

fn main() {
    println!(
        "{:>28} {:>8} {:>8} {:>10} {:>10} {:>12} {:>10}",
        "geometry", "verts", "edges", "neg star1", "obtuse", "min star1", "max angle"
    );

    // Production resolutions, taken from the call sites in the crate.
    let cases: Vec<(String, _)> = vec![
        ("circle r=1, 240 bdy, h=0.04".to_string(), disk_mesh(1.0, 1.0, 240, 0.04)),
        ("circle r=5, 32 bdy, h=1.0".to_string(), disk_mesh(5.0, 1.5, 32, 1.0)),
        ("cardioid q=1, r=3, 128 bdy".to_string(), epitrochoid_mesh(1.0, 3.0, 128, 0.3)),
        ("nephroid q=2, r=3, 128 bdy".to_string(), epitrochoid_mesh(2.0, 3.0, 128, 0.3)),
        ("epitrochoid q=3, r=3, 128 bdy".to_string(), epitrochoid_mesh(3.0, 3.0, 128, 0.3)),
    ];

    for (name, cm) in cases {
        let mesh = cm.mesh;
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let s1 = ops.hodge.star1();
        let n_edges = s1.len();
        let mut neg = 0usize;
        let mut min_s1 = f64::INFINITY;
        for e in 0..n_edges {
            let v = s1[e];
            min_s1 = min_s1.min(v);
            if v < 0.0 {
                neg += 1;
            }
        }

        let coords: Vec<[f64; 3]> = (0..mesh.n_vertices())
            .map(|v| {
                let p = mesh.vertices[v];
                [p[0], p[1], if p.len() > 2 { p[2] } else { 0.0 }]
            })
            .collect();
        let right = std::f64::consts::FRAC_PI_2;
        let mut obtuse = 0usize;
        let mut max_angle = 0.0_f64;
        for &[i0, i1, i2] in &mesh.simplices {
            let a = angles(coords[i0], coords[i1], coords[i2]);
            let m = a.iter().cloned().fold(0.0_f64, f64::max);
            max_angle = max_angle.max(m);
            if m > right + 1e-12 {
                obtuse += 1;
            }
        }

        println!(
            "{:>28} {:>8} {:>8} {:>10} {:>10} {:>12.4e} {:>9.1}d",
            name,
            mesh.n_vertices(),
            n_edges,
            neg,
            obtuse,
            min_s1,
            max_angle.to_degrees()
        );
    }
}
