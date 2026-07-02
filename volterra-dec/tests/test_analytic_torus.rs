//! Fast-tier analytic invariants on a torus.
//!
//! The torus is the natural companion to the sphere: it is a genus-1 surface, so
//! Gauss-Bonnet gives `integral K dA = 2 pi chi = 0` (the positive outer rim and negative
//! inner rim must cancel), which is a sharper test of the curvature/area weighting than the
//! sphere's `4 pi`. The analytic Gaussian curvature is known in closed form.

use std::f64::consts::PI;

use cartan_manifolds::euclidean::Euclidean;
use volterra_dec::mesh_gen::{torus_gaussian_curvature, torus_mesh};
use volterra_dec::DecDomain;

const R: f64 = 1.0; // major radius
const R_MINOR: f64 = 0.4; // minor radius
const N_MAJOR: usize = 64;
const N_MINOR: usize = 32;

fn torus_domain() -> DecDomain<Euclidean<3>> {
    let mesh = torus_mesh(R, R_MINOR, N_MAJOR, N_MINOR);
    DecDomain::new(mesh, Euclidean::<3>).expect("torus DecDomain construction")
}

#[test]
fn dual_areas_sum_to_torus_surface_area() {
    let d = torus_domain();
    let total: f64 = d.dual_areas.iter().sum();
    let exact = 4.0 * PI * PI * R * R_MINOR; // 4 pi^2 R r
    let rel = (total - exact).abs() / exact;
    assert!(
        rel < 0.02,
        "dual-area sum {total:.5} should match torus area {exact:.5} (rel err {rel:.4})"
    );
}

#[test]
fn gauss_bonnet_torus_integrates_to_zero() {
    // integral of K over a genus-1 surface = 2 pi * chi = 0.
    let d = torus_domain();
    let k = torus_gaussian_curvature(R, R_MINOR, N_MAJOR, N_MINOR);
    assert_eq!(k.len(), d.n_vertices());

    let integral_k: f64 = k.iter().zip(&d.dual_areas).map(|(&ki, &ai)| ki * ai).sum();
    // Normalise by the integral of |K| so the tolerance is dimensionless.
    let integral_abs_k: f64 = k.iter().zip(&d.dual_areas).map(|(&ki, &ai)| ki.abs() * ai).sum();
    let rel = integral_k.abs() / integral_abs_k;
    assert!(
        rel < 0.02,
        "integral K should vanish on the torus: got {integral_k:.4e} \
         (relative to integral|K| = {integral_abs_k:.4}, rel {rel:.4})"
    );
}
