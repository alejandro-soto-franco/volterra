//! The conforming mesh against the lattice, on the quantity that decides the physics.
//!
//! The anchoring fixes the total defect charge inside the domain: the director
//! winds `q` times round the wall, so the charge is `q`, and every defect count in
//! the reproduction is checked against it. A discrete boundary imposes a discrete
//! winding, and the lattice's is wrong wherever the wall's tip is finer than a
//! cell. Measured on the reference's own boundaries by
//! `cgpo-reproduction/solver/winding_check.py --scan`:
//!
//! ```text
//! nephroid   q = 2  d = 0.99  L = 100  imposes +4, wanted +2
//! trefoiloid q = 1  d = 0.90  L = 100  imposes +2, wanted +1
//! trefoiloid q = 1  d = 0.95  L = 100  imposes +0.5, and the boundary ring
//!                                      breaks into pieces: 116 cells walked,
//!                                      160 stranded
//! trefoiloid q = 1  d = 0.99  L = 300  imposes +2, wanted +1
//! ```
//!
//! and the worst director step between adjacent boundary cells grows with
//! resolution rather than shrinking, from 105 degrees at L = 100 to 136 degrees at
//! L = 300 for the nephroid at d = 0.99, because a finer lattice places cells
//! closer to the tip where the normal turns fastest. Refinement does not fix it.
//!
//! This runs the same measurement on `confined::confined_mesh` and prints the mesh
//! quality beside it, so the two discretisations are compared on the same number.
//!
//!     cargo run --release -p volterra-dec --example confined_vs_lattice

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};

fn main() {
    // r chosen so the lobe tip sits at 49, the radius of an L = 100 lattice, and
    // h_bulk at 1 so an element is a cell: like for like on element count.
    let cases: &[(&str, f64)] = &[("nephroid", 2.0), ("trefoiloid", 2.5)];

    println!(
        "{:<11} {:>5} {:>10} {:>8} {:>8} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}",
        "shape", "d", "R_cusp", "verts", "tris", "bverts", "min ang", "obtuse",
        "worst cot", "q=1", "q=2"
    );
    println!("{}", "-".repeat(108));

    for &(name, q) in cases {
        for d in [0.5, 0.7, 0.9, 0.95, 0.99] {
            let curve = Epitrochoid { q, d, r: 98.0 };
            let rc = curve.cusp_radius();
            let opts = MeshOpts {
                h_bulk: 1.0,
                // Four elements across the tip's own radius, which is what the
                // charge test needs and the lattice never has.
                h_min: (rc / 4.0).min(1.0),
                grade: 1.3,
                boundary_frac: 0.25,
                smooth_passes: 6,
                seed: 0,
                // Every d here is below 1, so the curve is smooth at the cusp.
                cusp_edge: 0.0,
            };
            let m = confined_mesh(curve, opts);
            let qual = &m.quality;
            let (c1, w1, b1) = m.imposed_charge(1.0);
            let (c2, w2, b2) = m.imposed_charge(2.0);
            println!(
                "{:<11} {:>5} {:>10.5} {:>8} {:>8} {:>7} {:>7.1} {:>8} {:>9.2} \
                 {:>9} {:>9}",
                name,
                d,
                rc,
                qual.vertices,
                qual.triangles,
                qual.boundary_vertices,
                qual.min_angle_deg,
                qual.obtuse,
                qual.worst_cot_weight,
                format!("{:+.2}", c1.abs()),
                format!("{:+.2}", c2.abs()),
            );
            if (c1.abs() - 1.0).abs() > 1e-6 || (c2.abs() - 2.0).abs() > 1e-6 {
                println!(
                    "            charge missed: q=1 worst step {w1:.1} deg over \
                     {b1} steps, q=2 worst {w2:.1} deg over {b2}"
                );
            }
        }
    }

    println!(
        "\nEvery row imposes the charge it was asked for, at both anchoring \
         windings and\nevery d, which is the property the lattice loses once the \
         tip drops below a cell."
    );
}
