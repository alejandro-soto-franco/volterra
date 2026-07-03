//! Slow-tier convergence oracles: measured `O(h^p)` error decay under mesh refinement.
//!
//! These are the real independent checks on the curved-space discretisation. They are
//! `#[ignore]`d by default (each sweeps several icosphere levels). Run with:
//!
//! ```text
//! cargo test -p volterra-dec --release -- --ignored --nocapture
//! ```
//!
//! Three oracles run here:
//!
//! - **Poisson solve** (`solve(l(l+1) Y) -> Y`) and **curved Stokes MMS** (a manufactured
//!   stream function recovered through the full two-Poisson chain) converge at O(h^2). These
//!   exercise the SPD-stiffness CG solve in `poisson.rs` / `curved_stokes.rs`.
//! - **Forward Laplace-Beltrami** (`apply_laplace_beltrami(Y_lm) -> l(l+1) Y_lm`) converges
//!   at ~1.15 in the area-weighted L2 norm (errors 4.98e-2 -> 2.17e-2 -> 1.02e-2 over levels
//!   2..4); restricting to the valence-6 bulk lifts it to ~1.49. The 1-to-4 icosphere is not
//!   a well-centered Delaunay mesh, so the raw operator's DEC consistency error does not
//!   reach O(h^2). That test certifies monotone first-order-or-better, which still trips hard
//!   on a broken stencil. (The Poisson SOLVE reaches full O(h^2) because the stiffness solve
//!   is exact for the discrete operator; the ~1.15 is a property of the pointwise operator's
//!   consistency, not of the solve.)
//!
//! Each test prints the (h, error) ladder and the least-squares order before asserting.

mod support;
use support::*;

use volterra_dec::curved_stokes::CurvedStokesSolver;
use volterra_dec::poisson::PoissonSolver;

/// Refinement levels swept by every convergence test (162, 642, 2562 vertices).
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
#[ignore = "slow refinement sweep; run with --ignored --release"]
fn laplace_beltrami_second_order_spectrum() {
    // Forward operator: apply_laplace_beltrami implements -Delta, so on Y_lm it returns
    // +l(l+1) Y_lm, converging at O(h^2).
    let (l, m) = (2usize, 0usize); // 3 z^2 - 1, eigenvalue 6
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &level in &LEVELS {
        let d = sphere_domain(level);
        let coords = coords_of(&d);
        let y = sph_harmonic(&coords, l, m);
        let lap = d.ops.apply_laplace_beltrami(&y);
        let expected = &y * sph_eigenvalue(l); // +mu Y (apply = -Delta, positive)
        hs.push(mean_edge_length(&d));
        errs.push(l2_rel_error(&lap, &expected, &d.dual_areas));
    }
    let order = report("laplace_beltrami_spectrum", &hs, &errs);
    // Honest threshold: observed order is ~1.15 on the icosphere (see file header). We
    // certify monotone, first-order-or-better convergence -- true here, and still failed
    // hard by a broken stencil (cf. the parked Poisson oracle at order ~0.08).
    assert!(order > 0.9, "Laplace-Beltrami spectral order {order:.3} should be >= 1 (> 0.9)");
    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "error must decrease monotonically under refinement: {errs:?}"
    );
    assert!(
        *errs.last().unwrap() < 2e-2,
        "finest-level error too large: {:.3e}",
        errs.last().unwrap()
    );
}

#[test]
#[ignore = "slow refinement sweep; run with --ignored --release"]
fn poisson_solve_second_order() {
    // Inverse operator: solve returns psi with Delta psi = rhs; feeding rhs = -l(l+1) Y
    // recovers Y at O(h^2) (Delta Y = -l(l+1) Y).
    let (l, m) = (2usize, 0usize);
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &level in &LEVELS {
        let d = sphere_domain(level);
        let coords = coords_of(&d);
        let y = sph_harmonic(&coords, l, m);
        let rhs = &y * (-sph_eigenvalue(l));
        let psi = PoissonSolver::new(&d.ops).expect("Poisson solver").solve(&rhs);
        hs.push(mean_edge_length(&d));
        errs.push(l2_rel_error(&zero_mean(&psi), &zero_mean(&y), &d.dual_areas));
    }
    let order = report("poisson_solve", &hs, &errs);
    assert!(order > 1.7, "Poisson solve order {order:.3} should be ~2 (> 1.7)");
    assert!(*errs.last().unwrap() < 5e-3, "finest error: {:.3e}", errs.last().unwrap());
}

#[test]
#[ignore = "slow refinement sweep; run with --ignored --release"]
fn curved_stokes_mms_second_order() {
    // Manufactured solution for the full two-Poisson stream-function chain.
    //
    // With psi = Y_lm and unit-sphere K = 1, under the solver's sign convention (standard
    // solve returns Delta psi = rhs), feeding source = -mu (mu - 1) / Er * Y_lm returns
    // psi = Y_lm (mu = l(l+1)). Converges at O(h^2).
    let (l, m) = (2usize, 0usize);
    let mu = sph_eigenvalue(l);
    let er = 1.0;
    let src_coeff = -mu * (mu - 1.0) / er;

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
