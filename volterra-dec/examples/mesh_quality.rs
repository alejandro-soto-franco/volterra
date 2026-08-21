//! Quality of the confined epitrochoid meshes, against what DEC needs of them.
//!
//! The lattice solver tests cell centres against a radial inequality, so its wall
//! is a staircase and its cusp is a staircase corner. A mesh can put vertices on
//! the curve exactly, which is the point of moving to one. Whether that helps
//! depends on three things a mesh can get wrong, and all three are measured here
//! rather than assumed:
//!
//!   angles          the DEC cotangent Laplacian has weight (cot a + cot b)/2 per
//!                   edge, which goes negative once an opposite angle exceeds a
//!                   right angle. Negative weights cost the discrete maximum
//!                   principle, and for a nematic that means the order parameter
//!                   can overshoot and manufacture a core. Obtuse triangles are
//!                   counted, and so are the ones that are not well centred,
//!                   which is the stronger condition the diagonal Hodge star
//!                   wants.
//!
//!   boundary        distance from each boundary vertex to the exact curve, and
//!                   the arclength spacing between neighbours against the local
//!                   radius of curvature. A boundary edge longer than the local
//!                   curvature radius cannot represent the wall there, and near a
//!                   cusp that radius collapses.
//!
//!   cusp            the curvature radius at the tip is 3 R (1 - d)^2 / 8 for the
//!                   two-cusped curve, so it vanishes quadratically as the cusp
//!                   sharpens. This reports it against the smallest edge actually
//!                   present, which is the ratio that says whether the geometry
//!                   is resolved at all.
//!
//!     cargo run --release -p volterra-dec --example mesh_quality

use std::f64::consts::PI;

use volterra_dec::epitrochoid::{epitrochoid_mesh, sample_epitrochoid};

/// Point, first and second derivative of the epitrochoid at parameter u.
///
/// The module's own curve is the true epicycloid, `d = 1`; the derivatives are
/// written with `d` explicit so the cusp curvature can be reported for the
/// regularised family the lattice runs actually use.
fn curve(q: f64, r: f64, d: f64, u: f64) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let c = 2.0 * q - 1.0;
    let a = r / (2.0 * q);
    (
        [a * (c * u.cos() + d * (c * u).cos()), a * (c * u.sin() + d * (c * u).sin())],
        [
            a * (-c * u.sin() - d * c * (c * u).sin()),
            a * (c * u.cos() + d * c * (c * u).cos()),
        ],
        [
            a * (-c * u.cos() - d * c * c * (c * u).cos()),
            a * (-c * u.sin() - d * c * c * (c * u).sin()),
        ],
    )
}

fn curvature_radius(q: f64, r: f64, d: f64, u: f64) -> f64 {
    let (_, p1, p2) = curve(q, r, d, u);
    let cross = (p1[0] * p2[1] - p1[1] * p2[0]).abs();
    let speed = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
    if cross <= 0.0 { f64::INFINITY } else { speed.powi(3) / cross }
}

/// Shortest distance from a point to the curve, by dense sampling then a local
/// refinement, which is enough to tell an exact vertex from an approximate one.
fn dist_to_curve(q: f64, r: f64, p: [f64; 2], n: usize) -> f64 {
    let mut best = f64::INFINITY;
    let mut best_u = 0.0;
    for i in 0..n {
        let u = 2.0 * PI * i as f64 / n as f64;
        let (c, _, _) = curve(q, r, 1.0, u);
        let d2 = (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2);
        if d2 < best {
            best = d2;
            best_u = u;
        }
    }
    let mut lo = best_u - 2.0 * PI / n as f64;
    let mut hi = best_u + 2.0 * PI / n as f64;
    for _ in 0..80 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let f = |u: f64| {
            let (c, _, _) = curve(q, r, 1.0, u);
            (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)
        };
        if f(m1) < f(m2) { hi = m2 } else { lo = m1 }
    }
    f64::sqrt(f64::min(best, {
        let (c, _, _) = curve(q, r, 1.0, 0.5 * (lo + hi));
        (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)
    }))
}

fn angles(p: [[f64; 2]; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for k in 0..3 {
        let a = p[k];
        let b = p[(k + 1) % 3];
        let c = p[(k + 2) % 3];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - a[0], c[1] - a[1]];
        let dot = u[0] * v[0] + u[1] * v[1];
        let nu = (u[0] * u[0] + u[1] * u[1]).sqrt();
        let nv = (v[0] * v[0] + v[1] * v[1]).sqrt();
        out[k] = (dot / (nu * nv)).clamp(-1.0, 1.0).acos();
    }
    out
}

fn main() {
    let q = 2.0;          // nephroid
    let r = 98.0;         // so the lobe tip sits at 49, the L = 100 lattice radius
    println!(
        "nephroid, q = {q}, r = {r} (lobe tip at {:.1}, matching the L = 100 lattice)\n",
        r * (2.0 * q) / (2.0 * q) / 2.0
    );

    println!("cusp curvature radius against the regularisation d, in lattice units:");
    println!("  {:>6}  {:>14}  {:>26}", "d", "R_cusp", "lattice cells across it");
    for d in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999] {
        let rc = curvature_radius(q, r, d, PI / 2.0);
        println!("  {d:>6}  {rc:>14.5}  {rc:>26.4}");
    }
    println!(
        "\nso at the reference's d = 0.99 the wall's tip is {:.0}x finer than one \
         lattice cell,\nand the staircase the lattice integrates is not the curve \
         at all.\n",
        1.0 / curvature_radius(q, r, 0.99, PI / 2.0)
    );

    println!("{:>9} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>10}",
             "spacing", "verts", "tris", "bverts", "min ang", "max ang", "obtuse",
             "%obtuse", "max dev");
    for spacing in [4.0, 2.0, 1.0, 0.5] {
        let cm = epitrochoid_mesh(q, r, (2.0 * PI * r / spacing) as usize, spacing);
        let m = &cm.mesh;
        let mut amin = f64::INFINITY;
        let mut amax: f64 = 0.0;
        let mut obtuse = 0usize;
        for t in 0..m.n_simplices() {
            let s = m.simplices[t];
            let p = [
                [m.vertices[s[0]].x, m.vertices[s[0]].y],
                [m.vertices[s[1]].x, m.vertices[s[1]].y],
                [m.vertices[s[2]].x, m.vertices[s[2]].y],
            ];
            let a = angles(p);
            let lo = a.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = a.iter().cloned().fold(0.0_f64, f64::max);
            amin = amin.min(lo);
            amax = amax.max(hi);
            if hi > PI / 2.0 + 1e-12 {
                obtuse += 1;
            }
        }
        // Boundary vertices should sit on the curve to rounding, since they are
        // sampled from it; any deviation would mean the mesher moved them.
        let mut dev: f64 = 0.0;
        for &i in cm.boundary_vertices.iter().step_by(7) {
            let p = [m.vertices[i].x, m.vertices[i].y];
            dev = dev.max(dist_to_curve(q, r, p, 4000));
        }
        println!("{:>9.2} {:>8} {:>8} {:>8} {:>9.2} {:>9.2} {:>8} {:>8.1} {:>10.2e}",
                 spacing, m.n_vertices(), m.n_simplices(),
                 cm.boundary_vertices.len(), amin.to_degrees(), amax.to_degrees(),
                 obtuse, 100.0 * obtuse as f64 / m.n_simplices() as f64, dev);
    }

    // Boundary spacing against the local curvature radius. Uniform sampling in
    // the parameter is not uniform in arclength: the speed |r'(u)| collapses at a
    // cusp, so points crowd there on their own. Whether they crowd enough is the
    // question.
    println!("\nboundary edge length against the local curvature radius, \
              at 512 samples, d = 0.99:");
    let (pts, params) = sample_epitrochoid(q, r, 512);
    println!("  {:>10} {:>12} {:>12} {:>12}", "u/pi", "ds", "R_curv", "ds/R_curv");
    for i in [0usize, 32, 64, 96, 120, 126, 128, 130, 136, 160] {
        let j = (i + 1) % pts.len();
        let ds = ((pts[j][0] - pts[i][0]).powi(2) + (pts[j][1] - pts[i][1]).powi(2)).sqrt();
        let rc = curvature_radius(q, r, 0.99, params[i]);
        println!("  {:>10.4} {:>12.5} {:>12.5} {:>12.2}",
                 params[i] / PI, ds, rc, ds / rc);
    }
}
