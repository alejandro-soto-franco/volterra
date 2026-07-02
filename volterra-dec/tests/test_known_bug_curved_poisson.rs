//! KNOWN-BUG oracles: `PoissonSolver` / `CurvedStokesSolver` on curved meshes.
//!
//! These tests assert the CORRECT behaviour (a manufactured/spectral solution recovered at
//! `O(h^2)`). They currently FAIL, because `PoissonSolver` is incorrect on meshes with
//! non-uniform dual areas. They are therefore `#[ignore]`d and parked here so they cannot
//! break the default gate, while standing as the oracle that will confirm the fix.
//!
//! ## The defect
//!
//! `Operators::laplace_beltrami` is the mass-normalised operator `M^{-1} S` (`M` = dual-area
//! mass, `S` = symmetric cotan stiffness). Its forward action is the correct `-Delta`
//! (`apply_laplace_beltrami(Y_lm) = +l(l+1) Y_lm`, second-order — see `test_convergence.rs`).
//! But `M^{-1} S` is NOT symmetric when the dual areas vary (as on any curved mesh):
//! measured asymmetry on `icosphere(3)` is `max|A_ij - A_ji| = 8.2` against `max|A| = 239`.
//!
//! `PoissonSolver::{new, with_dirichlet}` build the LOWER TRIANGLE of `-laplace_beltrami`
//! and hand it to `sprs_ldl` (a symmetric `LDL^T` factorisation). That silently symmetrises
//! a non-symmetric matrix, so the factorisation solves the wrong system: on `icosphere(3)`,
//! `solve(2 z)` returns a field of norm `~0.006` (the true answer has norm `~0.29`) with a
//! ~100% residual. Flat `unit_square_grid` meshes have `M ~ const*I`, so `M^{-1} S` is
//! nearly symmetric and the existing (flat-only) Poisson tests pass — which is why this was
//! never caught.
//!
//! ## The fix (for review, not applied here)
//!
//! Factorise the SYMMETRIC stiffness `S = diag(star0) * laplace_beltrami` (SPD after pinning
//! / zero-mean) and mass-weight the right-hand side (`solve` should form `b = star0 .* rhs`
//! before back-substitution). On flat uniform grids this reduces to the current behaviour.
//! Once implemented, drop the `#[ignore]`s below and delete this header.

mod support;
use support::*;

use volterra_dec::curved_stokes::CurvedStokesSolver;
use volterra_dec::poisson::PoissonSolver;

const LEVELS: [usize; 3] = [2, 3, 4];

fn report(name: &str, hs: &[f64], errs: &[f64]) -> f64 {
    let order = convergence_order(hs, errs);
    eprintln!("\n[{name}] convergence ladder:");
    for (h, e) in hs.iter().zip(errs) {
        eprintln!("    h = {h:.5}   rel L2 err = {e:.3e}");
    }
    eprintln!("    least-squares order = {order:.3}");
    order
}

#[test]
#[ignore = "KNOWN BUG: PoissonSolver is incorrect on non-uniform dual areas; see file header"]
fn poisson_recovers_l1_harmonic_coarse() {
    // Single-resolution round-trip: solve(-Delta psi = l(l+1) Y) should return Y.
    let d = sphere_domain(3);
    let coords = coords_of(&d);
    let y = sph_harmonic(&coords, 1, 0);
    let rhs = &y * sph_eigenvalue(1);
    let psi = PoissonSolver::new(&d.ops).expect("Poisson solver").solve(&rhs);
    let err = l2_rel_error(&zero_mean(&psi), &zero_mean(&y), &d.dual_areas);
    assert!(err < 0.05, "Poisson round-trip rel L2 error = {err:.4} (expected < 0.05)");
}

#[test]
#[ignore = "KNOWN BUG: PoissonSolver is incorrect on non-uniform dual areas; see file header"]
fn poisson_solve_second_order() {
    let (l, m) = (2usize, 0usize);
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &level in &LEVELS {
        let d = sphere_domain(level);
        let coords = coords_of(&d);
        let y = sph_harmonic(&coords, l, m);
        let rhs = &y * sph_eigenvalue(l);
        let psi = PoissonSolver::new(&d.ops).expect("Poisson solver").solve(&rhs);
        hs.push(mean_edge_length(&d));
        errs.push(l2_rel_error(&zero_mean(&psi), &zero_mean(&y), &d.dual_areas));
    }
    let order = report("poisson_solve", &hs, &errs);
    assert!(order > 1.7, "Poisson solve order {order:.3} should be ~2 (> 1.7)");
    assert!(*errs.last().unwrap() < 5e-3, "finest error: {:.3e}", errs.last().unwrap());
}

#[test]
#[ignore = "KNOWN BUG: CurvedStokesSolver inherits the PoissonSolver defect; see file header"]
fn curved_stokes_mms_second_order() {
    // Manufactured solution for the full two-Poisson stream-function chain.
    // psi = Y_lm, unit-sphere K = 1: feeding source = mu (mu - 1) / Er * Y_lm returns psi.
    let (l, m) = (2usize, 0usize);
    let mu = sph_eigenvalue(l);
    let er = 1.0;
    let src_coeff = mu * (mu - 1.0) / er;

    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &level in &LEVELS {
        let d = sphere_domain(level);
        let coords = coords_of(&d);
        let y = sph_harmonic(&coords, l, m);
        let source = &y * src_coeff;
        let solver = CurvedStokesSolver::new(&d.ops, &d.mesh, &vec![1.0; d.n_vertices()])
            .expect("curved Stokes solver");
        let (psi, _vel) = solver.solve(&source, er);
        hs.push(mean_edge_length(&d));
        errs.push(l2_rel_error(&zero_mean(&psi), &zero_mean(&y), &d.dual_areas));
    }
    let order = report("curved_stokes_mms", &hs, &errs);
    assert!(order > 1.5, "curved Stokes MMS order {order:.3} should be ~2 (> 1.5)");
    assert!(*errs.last().unwrap() < 1e-2, "finest error: {:.3e}", errs.last().unwrap());
}
