//! Does the clamped screened solve converge to its closed form?
//!
//! `stokes_reproduces_the_clamped_screened_solution` measured a relative L2
//! error of 0.1076 at 240 boundary vertices, against a 5 per cent band chosen
//! by analogy with the simply supported test. The pre-existing clamped test
//! allows 12 per cent, so 10.8 per cent may be the accuracy of the clamped path
//! rather than an error in the closed form. Refinement separates the two: an
//! error that falls with the mesh belongs to the discretisation, and one that
//! sits still belongs to the formula.

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::Screening;
use volterra_dec::epitrochoid::disk_mesh;
use volterra_dec::stokes::{SurfaceStokes, extract_coords};

fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        ax * (0.5
            + t * (0.87890594
                + t * (0.51498869
                    + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
    } else {
        let t = 3.75 / ax;
        let a = 0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059));
        let b = 0.39894228
            + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * a))));
        (ax.exp() / ax.sqrt()) * b
    };
    if x >= 0.0 { r } else { -r }
}

fn psi_clamped(r: f64, rad: f64, k: f64, s: f64) -> f64 {
    let k2 = k * k;
    let a = s * rad / (2.0 * k * bessel_i1(k * rad));
    let c = s * rad * rad / (4.0 * k2) - (a / k2) * bessel_i0(k * rad);
    -s * r * r / (4.0 * k2) + (a / k2) * bessel_i0(k * r) + c
}

fn main() {
    let rad = 1.0_f64;
    let s_src = -4.0_f64;
    println!(
        "{:>6} {:>7} {:>8} {:>8} {:>12} {:>10}",
        "k", "n_bdy", "h", "verts", "rel L2", "ratio"
    );

    for &k in &[1.0_f64, 2.0, 4.0] {
        let mut prev = f64::NAN;
        for &(n_b, h) in &[(120usize, 0.08_f64), (240, 0.04), (480, 0.02)] {
            let cm = disk_mesh(rad, 1.0, n_b, h);
            let mesh = cm.mesh;
            let bverts = cm.boundary_vertices;
            let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
            let nv = mesh.n_vertices();
            let solver = SurfaceStokes::new_confined_clamped_screened(
                &ops,
                &mesh,
                &bverts,
                Screening::Length(1.0 / k),
            )
            .unwrap();
            let source = nalgebra::DVector::from_element(nv, s_src);
            let (_v, psi) = solver.stream_and_velocity(&source, &mesh);
            let coords = extract_coords(&mesh);
            let exact: Vec<f64> = coords
                .iter()
                .map(|p| psi_clamped((p[0] * p[0] + p[1] * p[1]).sqrt(), rad, k, s_src))
                .collect();
            let num: f64 = psi.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f64 = exact.iter().map(|b| b * b).sum();
            let err = (num / den).sqrt();
            let ratio = if prev.is_nan() { f64::NAN } else { prev / err };
            println!("{k:>6.1} {n_b:>7} {h:>8.3} {nv:>8} {err:>12.4e} {ratio:>10.2}");
            prev = err;
        }
        println!();
    }
}
