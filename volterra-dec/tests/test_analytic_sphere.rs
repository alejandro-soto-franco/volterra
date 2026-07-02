//! Fast-tier analytic invariants on the unit sphere.
//!
//! These run in the default `cargo test` gate at a single coarse resolution. They assert
//! structural properties that hold for *any* correct discretisation (self-adjointness,
//! sign-definiteness, kernel content, linearity), so they are cheap regression guards that
//! do not need a mesh-refinement sweep. The `O(h^p)` convergence oracles live in
//! `test_convergence.rs` behind `#[ignore]`.

mod support;
use support::*;

use nalgebra::DVector;
use volterra_dec::connection_laplacian::ConnectionLaplacian;
use volterra_dec::curved_stokes::CurvedStokesSolver;
use volterra_dec::qfield_dec::QFieldDec;

/// Dual-area-weighted inner product of two scalar fields.
fn dot_w(a: &DVector<f64>, b: &DVector<f64>, w: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .zip(w)
        .map(|((ai, bi), &wi)| wi * ai * bi)
        .sum()
}

/// Dual-area-weighted inner product of two Q-tensor (spin-2) fields.
fn dot_w_q(a: &QFieldDec, b: &QFieldDec, w: &[f64]) -> f64 {
    (0..w.len())
        .map(|i| w[i] * (a.q1[i] * b.q1[i] + a.q2[i] * b.q2[i]))
        .sum()
}

fn connection_laplacian(domain: &volterra_dec::DecDomain<cartan_manifolds::sphere::Sphere<3>>)
    -> ConnectionLaplacian
{
    let coords = coords_of(domain);
    let star0: Vec<f64> = domain.ops.hodge.star0().iter().copied().collect();
    let star1: Vec<f64> = domain.ops.hodge.star1().iter().copied().collect();
    ConnectionLaplacian::new(&domain.mesh, &coords, &star0, &star1)
}

// ── Scalar Laplace-Beltrami ────────────────────────────────────────────────

#[test]
fn laplace_beltrami_annihilates_constants() {
    let d = sphere_domain(3);
    let ones = DVector::from_element(d.n_vertices(), 1.0);
    let lap = d.ops.apply_laplace_beltrami(&ones);
    let max = lap.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(max < 1e-9, "Laplacian of a constant should vanish, got max |L1| = {max:.3e}");
}

#[test]
fn laplace_beltrami_self_adjoint() {
    // <L u, v>_w == <u, L v>_w for the dual-area inner product (L = M^{-1} S, S symmetric).
    let d = sphere_domain(3);
    let n = d.n_vertices();
    let coords = coords_of(&d);
    let u = &sph_harmonic(&coords, 1, 0) + &sph_harmonic(&coords, 2, 2);
    let v = &sph_harmonic(&coords, 1, 1) - &sph_harmonic(&coords, 2, 0);
    assert_eq!(u.len(), n);

    let lu = d.ops.apply_laplace_beltrami(&u);
    let lv = d.ops.apply_laplace_beltrami(&v);
    let lhs = dot_w(&lu, &v, &d.dual_areas);
    let rhs = dot_w(&u, &lv, &d.dual_areas);
    let scale = lhs.abs().max(rhs.abs()).max(1e-30);
    assert!(
        (lhs - rhs).abs() / scale < 1e-9,
        "self-adjointness broken: <Lu,v> = {lhs:.6e}, <u,Lv> = {rhs:.6e}"
    );
}

#[test]
fn laplace_beltrami_positive_semidefinite() {
    // apply_laplace_beltrami implements -Delta (positive), so <L u, u>_w >= 0 for every
    // field. This is the discrete Dirichlet energy; a sign error would flip it.
    let d = sphere_domain(3);
    let coords = coords_of(&d);
    for (l, m) in [(1usize, 0usize), (1, 1), (2, 0), (2, 2)] {
        let u = sph_harmonic(&coords, l, m);
        let lu = d.ops.apply_laplace_beltrami(&u);
        let quad = dot_w(&lu, &u, &d.dual_areas);
        assert!(
            quad >= -1e-9,
            "L (=-Delta) not positive semi-definite for (l,m)=({l},{m}): <Lu,u> = {quad:.6e}"
        );
    }
}

// ── Connection (spin-2) Laplacian ──────────────────────────────────────────

#[test]
fn connection_laplacian_zero_field() {
    let d = sphere_domain(3);
    let cl = connection_laplacian(&d);
    let zero = QFieldDec::zeros(d.n_vertices());
    let out = cl.apply(&zero);
    let max = (0..d.n_vertices()).fold(0.0_f64, |m, i| m.max(out.q1[i].abs()).max(out.q2[i].abs()));
    assert!(max < 1e-12, "connection Laplacian of zero field should vanish, got {max:.3e}");
}

#[test]
fn connection_laplacian_self_adjoint() {
    let d = sphere_domain(3);
    let cl = connection_laplacian(&d);
    let u = QFieldDec::random_perturbation(d.n_vertices(), 1.0, 11);
    let v = QFieldDec::random_perturbation(d.n_vertices(), 1.0, 29);
    let lu = cl.apply(&u);
    let lv = cl.apply(&v);
    let lhs = dot_w_q(&lu, &v, &d.dual_areas);
    let rhs = dot_w_q(&u, &lv, &d.dual_areas);
    let scale = lhs.abs().max(rhs.abs()).max(1e-30);
    assert!(
        (lhs - rhs).abs() / scale < 1e-9,
        "connection Laplacian not self-adjoint: <Lu,v> = {lhs:.6e}, <u,Lv> = {rhs:.6e}"
    );
}

#[test]
fn connection_laplacian_positive_semidefinite() {
    // Same convention as the scalar operator: apply implements -Delta_conn (positive),
    // so the spin-2 Dirichlet energy <L u, u>_w >= 0 for every field.
    let d = sphere_domain(3);
    let cl = connection_laplacian(&d);
    for seed in [3u64, 7, 13, 101] {
        let u = QFieldDec::random_perturbation(d.n_vertices(), 1.0, seed);
        let lu = cl.apply(&u);
        let quad = dot_w_q(&lu, &u, &d.dual_areas);
        assert!(
            quad >= -1e-9,
            "connection Laplacian not positive semi-definite (seed {seed}): <Lu,u> = {quad:.6e}"
        );
    }
}

// ── Curved Stokes solver (structural) ──────────────────────────────────────

#[test]
fn stokes_solution_is_finite_and_linear() {
    // The stream-function/velocity map is linear in the source. Assert finiteness and
    // solve(2 s) == 2 solve(s) to machine precision (both Poisson solves are linear).
    let d = sphere_domain(3);
    let coords = coords_of(&d);
    let solver = CurvedStokesSolver::new(&d.ops, &d.mesh, &vec![1.0; d.n_vertices()])
        .expect("curved Stokes solver");
    let source = sph_harmonic(&coords, 2, 2);
    let er = 1.0;

    let (psi1, vel1) = solver.solve(&source, er);
    let (psi2, vel2) = solver.solve(&(&source * 2.0), er);

    assert!(psi1.iter().all(|v| v.is_finite()), "psi must be finite");
    assert!(
        (0..d.n_vertices()).all(|i| vel1.v[i].iter().all(|c| c.is_finite())),
        "velocity must be finite"
    );

    let psi_lin = (&psi2 - &(&psi1 * 2.0)).amax();
    assert!(psi_lin < 1e-9, "stream function not linear in source: max dev = {psi_lin:.3e}");

    let vel_lin = (0..d.n_vertices())
        .flat_map(|i| (0..3).map(move |c| (i, c)))
        .fold(0.0_f64, |m, (i, c)| m.max((vel2.v[i][c] - 2.0 * vel1.v[i][c]).abs()));
    assert!(vel_lin < 1e-9, "velocity not linear in source: max dev = {vel_lin:.3e}");
}
