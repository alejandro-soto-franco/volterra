//! Does `star1` lose the sign the cotangent formula gives it?
//!
//! The circumcentric Hodge star on 1-forms is the SIGNED dual length over the
//! primal length, and that equals `(cot a + cot b) / 2` for the two angles
//! opposite the edge. It is negative exactly when `a + b > pi`, which is the
//! obtuse case. `cartan-dec` forms the dual length as `manifold.dist(c1, c2)`
//! between the two circumcentres, and a distance is unsigned, so an obtuse edge
//! would come back with the right magnitude and the wrong sign.
//!
//! `star1_health` measured zero negative entries on meshes with 96, 80, 68 and
//! 64 obtuse triangles, which is what that would look like. This compares the
//! two formulas edge by edge and reports where they disagree in sign, so the
//! reading is measured rather than inferred from an absence.

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_dec::epitrochoid::{disk_mesh, epitrochoid_mesh};

fn main() {
    println!(
        "{:>28} {:>7} {:>10} {:>10} {:>12} {:>12}",
        "geometry", "edges", "cot<0", "sign flip", "worst gap", "max |cot|"
    );

    let cases: Vec<(String, _)> = vec![
        (
            "circle r=1, 240 bdy, h=0.04".into(),
            disk_mesh(1.0, 1.0, 240, 0.04),
        ),
        (
            "cardioid q=1, r=3, 128 bdy".into(),
            epitrochoid_mesh(1.0, 3.0, 128, 0.3),
        ),
        (
            "nephroid q=2, r=3, 128 bdy".into(),
            epitrochoid_mesh(2.0, 3.0, 128, 0.3),
        ),
    ];

    for (name, cm) in cases {
        let mesh = cm.mesh;
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let s1 = ops.hodge.star1();

        let p = |v: usize| {
            let x = mesh.vertices[v];
            [x[0], x[1]]
        };

        // Cotangent of the angle at `apex` in the triangle (apex, u, v).
        let cot_at = |apex: usize, u: usize, v: usize| -> f64 {
            let (a, b, c) = (p(apex), p(u), p(v));
            let e1 = [b[0] - a[0], b[1] - a[1]];
            let e2 = [c[0] - a[0], c[1] - a[1]];
            let dot = e1[0] * e2[0] + e1[1] * e2[1];
            let cross = (e1[0] * e2[1] - e1[1] * e2[0]).abs();
            if cross < 1e-30 { 0.0 } else { dot / cross }
        };

        // Sum the two opposite cotangents for each edge, over its cofaces.
        let ne = s1.len();
        let mut cot = vec![0.0_f64; ne];
        for (e, cot_e) in cot.iter_mut().enumerate() {
            let ends = mesh.boundaries[e];
            let (u, v) = (ends[0], ends[1]);
            let mut acc = 0.0;
            for &t in &mesh.boundary_simplices[e] {
                if let Some(&apex) = mesh.simplices[t].iter().find(|&&w| w != u && w != v) {
                    acc += cot_at(apex, u, v);
                }
            }
            *cot_e = 0.5 * acc;
        }

        let mut neg = 0usize;
        let mut flips = 0usize;
        let mut worst = 0.0_f64;
        let mut max_cot = 0.0_f64;
        for e in 0..ne {
            let c = cot[e];
            let s = s1[e];
            max_cot = max_cot.max(c.abs());
            if c < -1e-12 {
                neg += 1;
                // A sign flip is the star being positive where the cotangent
                // formula is negative, with a magnitude that agrees.
                if s > 1e-12 && (s - c.abs()).abs() < 0.05 * c.abs().max(1e-30) {
                    flips += 1;
                }
            }
            worst = worst.max((s - c).abs());
        }

        println!(
            "{:>28} {:>7} {:>10} {:>10} {:>12.4e} {:>12.4e}",
            name, ne, neg, flips, worst, max_cot
        );
    }
}
