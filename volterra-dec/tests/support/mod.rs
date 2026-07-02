//! Shared harness for the analytic-oracle integration tests.
//!
//! These helpers turn `volterra-dec`'s operators into checks against *closed-form*
//! results (spherical-harmonic spectra, Gauss-Bonnet, manufactured Stokes solutions)
//! rather than single-resolution regression thresholds. The point is an independent
//! oracle: a discretisation that is subtly wrong fails these even when it "looks right".
//!
//! Nothing here depends on a second implementation. The spherical harmonics are the
//! Cartesian monomials restricted to the unit sphere, which are exact eigenfunctions of
//! the Laplace-Beltrami operator with eigenvalue `-l(l+1)`.

#![allow(dead_code)] // helpers are shared across several test binaries; not all use all

use cartan_manifolds::sphere::Sphere;
use nalgebra::DVector;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::DecDomain;

/// Build a unit-sphere `DecDomain` at the given icosphere refinement level.
///
/// Vertex counts: level 2 -> 162, 3 -> 642, 4 -> 2562, 5 -> 10242.
pub fn sphere_domain(level: usize) -> DecDomain<Sphere<3>> {
    let mesh = icosphere(level);
    DecDomain::new(mesh, Sphere::<3>).expect("icosphere DecDomain construction")
}

/// Vertex coordinates as `[x, y, z]` triples (unit-length on the icosphere).
pub fn coords_of<M: cartan_core::Manifold>(domain: &DecDomain<M>) -> Vec<[f64; 3]> {
    volterra_dec::stokes_dec::extract_coords(&domain.mesh)
}

/// The Laplace-Beltrami eigenvalue magnitude `l(l+1)` for degree `l`.
///
/// On the unit sphere `-Delta Y_lm = l(l+1) Y_lm`, so `Delta Y_lm = -l(l+1) Y_lm`.
pub fn sph_eigenvalue(l: usize) -> f64 {
    (l * (l + 1)) as f64
}

/// A real spherical harmonic of degree `l`, order `m`, sampled at unit-sphere `coords`.
///
/// Returned unnormalised (the eigenfunction relation is scale-invariant). Supported
/// pairs are the ones the tests exercise; each is an exact `-Delta` eigenfunction with
/// eigenvalue `l(l+1)`:
///
/// - `(1, 0) -> z`,             `(1, 1) -> x`         (eigenvalue 2)
/// - `(2, 0) -> 3 z^2 - 1`,     `(2, 2) -> x^2 - y^2` (eigenvalue 6)
pub fn sph_harmonic(coords: &[[f64; 3]], l: usize, m: usize) -> DVector<f64> {
    let f = |p: &[f64; 3]| -> f64 {
        let (x, y, z) = (p[0], p[1], p[2]);
        match (l, m) {
            (1, 0) => z,
            (1, 1) => x,
            (2, 0) => 3.0 * z * z - 1.0,
            (2, 2) => x * x - y * y,
            _ => panic!("sph_harmonic: unsupported (l, m) = ({l}, {m})"),
        }
    };
    DVector::from_iterator(coords.len(), coords.iter().map(f))
}

/// Subtract the (unweighted) mean, projecting out the constant kernel.
///
/// The Poisson solver returns zero-mean solutions and the continuum harmonics
/// (l >= 1) are zero-mean, so comparisons are done on the zero-mean projection.
pub fn zero_mean(v: &DVector<f64>) -> DVector<f64> {
    let mean = v.sum() / v.len() as f64;
    v.map(|x| x - mean)
}

/// Area-weighted L2 norm `sqrt( sum_i w_i a_i^2 / sum_i w_i )` (the correct DEC norm).
pub fn l2_norm(a: &DVector<f64>, w: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (ai, &wi) in a.iter().zip(w) {
        num += wi * ai * ai;
        den += wi;
    }
    (num / den).sqrt()
}

/// Area-weighted relative L2 error `||a - b||_w / ||b||_w`.
pub fn l2_rel_error(a: &DVector<f64>, b: &DVector<f64>, w: &[f64]) -> f64 {
    let diff = a - b;
    l2_norm(&diff, w) / l2_norm(b, w)
}

/// Mean primal edge length, used as the mesh scale `h` for convergence fits.
pub fn mean_edge_length<M: cartan_core::Manifold>(domain: &DecDomain<M>) -> f64 {
    let n = domain.edge_lengths.len().max(1);
    domain.edge_lengths.iter().sum::<f64>() / n as f64
}

/// Least-squares slope of `ln(err)` against `ln(h)`: the observed convergence order.
///
/// A second-order-accurate operator gives ~2. Requires at least two points.
pub fn convergence_order(hs: &[f64], errs: &[f64]) -> f64 {
    assert_eq!(hs.len(), errs.len());
    assert!(hs.len() >= 2, "need >= 2 refinements to estimate an order");
    let n = hs.len() as f64;
    let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (xi, yi) in lx.iter().zip(&ly) {
        cov += (xi - mx) * (yi - my);
        var += (xi - mx) * (xi - mx);
    }
    cov / var
}
