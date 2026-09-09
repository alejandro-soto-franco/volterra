//! How the clamped wall's response matrix conditions against screening.
//!
//! `new_confined_clamped_screened` builds one response basis column per
//! boundary vertex: a unit wall value of the vorticity is lifted through the
//! FIRST solve, the stream function it drives comes from the SECOND, and the
//! column records that stream function's inward normal derivative at every
//! boundary vertex. The clamped condition is then a solve against that matrix.
//!
//! `the_clamped_wall_holds_under_screening` reported an infinite condition
//! number at `l_s = 0.1` on a disc of radius 1. This sweeps the screening
//! length and the boundary resolution together, so the answer separates into
//! one of two: the matrix loses rank once `l_s` falls below the boundary
//! spacing, which is a resolution limit and is fixable by refining; or it loses
//! rank at a fixed `l_s / R` whatever the resolution, which condemns the
//! method in the Hele-Shaw regime the chip needs.

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::Screening;
use volterra_dec::stokes::SurfaceStokes;

fn main() {
    let rad = 1.0_f64;
    println!(
        "{:>10} {:>7} {:>9} {:>9} {:>12} {:>12} {:>12}",
        "n_boundary", "spacing", "l_s", "l_s/space", "cond", "sigma_max", "sigma_min"
    );

    for &(n_b, spacing) in &[(120usize, 0.06_f64), (240, 0.03), (480, 0.015)] {
        let cm = volterra_dec::epitrochoid::disk_mesh(rad, 1.0, n_b, spacing);
        let mesh = cm.mesh;
        let bverts = cm.boundary_vertices;
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let arc = 2.0 * std::f64::consts::PI * rad / bverts.len() as f64;

        for &ls in &[f64::INFINITY, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025] {
            let screening = if ls.is_infinite() {
                Screening::None
            } else {
                Screening::Length(ls)
            };
            let solver =
                SurfaceStokes::new_confined_clamped_screened(&ops, &mesh, &bverts, screening);
            match solver {
                Ok(s) => {
                    let cond = s.clamped_condition().unwrap_or(f64::NAN);
                    let spec = s.clamped_spectrum().unwrap_or(&[]);
                    let hi = spec.first().copied().unwrap_or(f64::NAN);
                    let lo = spec.last().copied().unwrap_or(f64::NAN);
                    println!(
                        "{:>10} {:>7.3} {:>9} {:>9.2} {:>12.3e} {:>12.3e} {:>12.3e}",
                        bverts.len(),
                        arc,
                        if ls.is_infinite() { "inf".to_string() } else { format!("{ls}") },
                        if ls.is_infinite() { f64::INFINITY } else { ls / arc },
                        cond,
                        hi,
                        lo
                    );
                }
                Err(e) => println!("{:>10} {:>7.3} {:>9} construction failed: {e}", bverts.len(), arc, ls),
            }
        }
        println!();
    }
}
