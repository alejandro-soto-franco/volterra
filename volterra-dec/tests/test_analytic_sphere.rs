//! Fast-tier analytic invariants on the unit sphere.
//!
//! These run in the default `cargo test` run at a single coarse resolution. They assert
//! structural properties that hold for *any* correct discretisation (self-adjointness,
//! sign-definiteness, kernel content, linearity), so they are cheap regression checks that
//! do not need a mesh-refinement sweep. The `O(h^p)` convergence oracles live in
//! `test_convergence.rs` behind `#[ignore]`.

mod support;
use support::*;

use nalgebra::DVector;
use volterra_dec::connection_laplacian::ConnectionLaplacian;
use volterra_dec::stokes::SurfaceStokes;
use volterra_dec::poisson::PoissonSolver;
use volterra_dec::qfield::QField;

/// Dual-area-weighted inner product of two scalar fields.
fn dot_w(a: &DVector<f64>, b: &DVector<f64>, w: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .zip(w)
        .map(|((ai, bi), &wi)| wi * ai * bi)
        .sum()
}

/// Dual-area-weighted inner product of two Q-tensor (spin-2) fields.
fn dot_w_q(a: &QField, b: &QField, w: &[f64]) -> f64 {
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
fn poisson_recovers_l1_harmonic() {
    // Round-trip at a single resolution. solve returns psi with Delta psi = rhs, and
    // Delta Y = -l(l+1) Y, so feeding rhs = -l(l+1) Y recovers Y (up to a constant).
    // O(h^2) convergence is asserted in test_convergence.rs.
    let d = sphere_domain(3);
    let coords = coords_of(&d);
    let y = sph_harmonic(&coords, 1, 0); // z, eigenvalue 2
    let rhs = &y * (-sph_eigenvalue(1));
    let psi = PoissonSolver::new(&d.ops).expect("Poisson solver").solve(&rhs);
    let err = l2_rel_error(&zero_mean(&psi), &zero_mean(&y), &d.dual_areas);
    assert!(err < 0.05, "Poisson round-trip rel L2 error = {err:.4} (expected < 0.05)");
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
    let zero = QField::zeros(d.n_vertices());
    let out = cl.apply(&zero);
    let max = (0..d.n_vertices()).fold(0.0_f64, |m, i| m.max(out.q1[i].abs()).max(out.q2[i].abs()));
    assert!(max < 1e-12, "connection Laplacian of zero field should vanish, got {max:.3e}");
}

#[test]
fn connection_laplacian_self_adjoint() {
    let d = sphere_domain(3);
    let cl = connection_laplacian(&d);
    let u = QField::random_perturbation(d.n_vertices(), 1.0, 11);
    let v = QField::random_perturbation(d.n_vertices(), 1.0, 29);
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
        let u = QField::random_perturbation(d.n_vertices(), 1.0, seed);
        let lu = cl.apply(&u);
        let quad = dot_w_q(&lu, &u, &d.dual_areas);
        assert!(
            quad >= -1e-9,
            "connection Laplacian not positive semi-definite (seed {seed}): <Lu,u> = {quad:.6e}"
        );
    }
}

// ── Surface Stokes solver (structural) ─────────────────────────────────────

#[test]
fn stokes_solution_is_finite_and_linear() {
    // The stream-function/velocity map is linear in the source. Assert finiteness and
    // solve(2 s) == 2 solve(s) to machine precision (both Poisson solves are linear).
    let d = sphere_domain(3);
    let coords = coords_of(&d);
    let solver = SurfaceStokes::new(&d.ops, &d.mesh).expect("surface Stokes solver");
    let source = sph_harmonic(&coords, 2, 2);
    let er = 1.0;

    let (vel1, psi1v) = solver.stream_and_velocity(&(&source * er), &d.mesh);
    let (vel2, psi2v) = solver.stream_and_velocity(&(&source * (2.0 * er)), &d.mesh);
    let psi1 = nalgebra::DVector::from_vec(psi1v);
    let psi2 = nalgebra::DVector::from_vec(psi2v);

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

/// How much of the active flow is a rigid spin of the whole sphere.
///
/// A rigid rotation is a Killing field: it deforms nothing, so the viscous
/// operator annihilates it and no forcing can be spent on it. On a sphere the
/// Killing fields are the three rotations `Omega x r`, so a correct solver
/// returns a velocity with no component in that three-dimensional subspace.
///
/// The measure is the fraction of `|u|^2` that the projection onto the three
/// rotations accounts for, in the dual-area inner product. A rigid spin moves
/// defects around the sphere without moving them relative to one another, so
/// whatever sits in this subspace is added to every defect trajectory and
/// contributes nothing to the braid they write.
#[test]
fn the_active_flow_has_no_rigid_spin() {
    use volterra_core::ActiveNematicParams;
    use volterra_dec::stokes::SurfaceStokes;

    let domain = sphere_domain(4);
    let coords = coords_of(&domain);
    let nv = coords.len();
    let w: Vec<f64> = (0..nv).map(|i| domain.ops.hodge.star0()[i]).collect();

    let solver = SurfaceStokes::new(&domain.ops, &domain.mesh).expect("stokes solver");
    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = 1.0;
    params.eta = 1.0;

    // An ordered field with structure at l = 2, so the forcing is not itself
    // symmetric in a way that would zero the projection for the wrong reason.
    let q = QField::random_perturbation(nv, 0.5, 7);
    let vel = solver.solve(&q, &params, &domain.ops, &domain.mesh);

    let dot = |a: &[[f64; 3]], b: &[[f64; 3]]| -> f64 {
        (0..nv).map(|i| w[i] * (a[i][0] * b[i][0] + a[i][1] * b[i][1] + a[i][2] * b[i][2])).sum()
    };

    // The three rigid rotations, mass-orthonormalised.
    let mut basis: Vec<Vec<[f64; 3]>> = Vec::new();
    for axis in 0..3 {
        let mut e = [0.0; 3];
        e[axis] = 1.0;
        let mut r: Vec<[f64; 3]> = coords
            .iter()
            .map(|p| {
                [
                    e[1] * p[2] - e[2] * p[1],
                    e[2] * p[0] - e[0] * p[2],
                    e[0] * p[1] - e[1] * p[0],
                ]
            })
            .collect();
        for b in &basis {
            let d = dot(&r, b);
            for i in 0..nv {
                for k in 0..3 {
                    r[i][k] -= d * b[i][k];
                }
            }
        }
        let n = dot(&r, &r).sqrt();
        for p in r.iter_mut() {
            for k in 0..3 {
                p[k] /= n;
            }
        }
        basis.push(r);
    }

    let frac_of = |u: &[[f64; 3]]| -> f64 {
        let total = dot(u, u);
        let spin: f64 = basis.iter().map(|b| dot(u, b).powi(2)).sum();
        spin / total
    };

    // The measure must report 1 on a field that IS a rigid rotation, and it
    // must report a spin when one is added to the solver's own answer. Without
    // both controls a zero reading cannot be told from a blind measure.
    let pure_spin: Vec<[f64; 3]> = coords.iter().map(|p| [-p[1], p[0], 0.0]).collect();
    let f_pure = frac_of(&pure_spin);
    assert!(
        (f_pure - 1.0).abs() < 1e-8,
        "the measure should report 1 on a pure rotation, got {f_pure}"
    );
    let scale = dot(&vel.v, &vel.v).sqrt() / dot(&pure_spin, &pure_spin).sqrt();
    let spiked: Vec<[f64; 3]> = (0..nv)
        .map(|i| {
            [
                vel.v[i][0] + scale * pure_spin[i][0],
                vel.v[i][1] + scale * pure_spin[i][1],
                vel.v[i][2] + scale * pure_spin[i][2],
            ]
        })
        .collect();
    let f_spiked = frac_of(&spiked);
    assert!(
        f_spiked > 0.3,
        "the measure should see an injected spin, got {f_spiked}"
    );

    let fraction = frac_of(&vel.v);
    println!(
        "rigid-spin fraction: solver {fraction:.3e}, pure rotation {f_pure:.6}, \
         solver plus an equal spin {f_spiked:.4}"
    );
    assert!(
        fraction < 1e-3,
        "the flow should have no rigid-spin component, got {fraction:.4} of |u|^2"
    );
}

/// Discrete Gauss-Bonnet, on the curvature the Stokes solver builds for itself.
///
/// `sum_i K_i A_i = 2 pi chi` is `4 pi` on a sphere whatever the triangulation,
/// so it tests the curvature rather than the mesh.
#[test]
fn the_curvature_integrates_to_the_euler_characteristic() {
    use volterra_dec::stokes::{compute_dual_areas, extract_coords, gaussian_curvature};

    for level in [2usize, 3, 4] {
        let domain = sphere_domain(level);
        let coords = extract_coords(&domain.mesh);
        let nv = coords.len();
        let areas = compute_dual_areas(nv, &domain.mesh.simplices, &coords);
        let k = gaussian_curvature(nv, &domain.mesh.simplices, &coords, &areas);
        let total: f64 = (0..nv).map(|i| k[i] * areas[i]).sum();
        let want = 4.0 * std::f64::consts::PI;
        assert!(
            (total - want).abs() < 1e-9,
            "level {level}: curvature integrates to {total}, wanted {want}"
        );
    }
}

/// The rigid rotations sit in the kernel of the outer factor, and nothing else does.
#[test]
fn the_outer_factor_has_the_rotations_in_its_kernel() {
    use volterra_dec::poisson::PoissonSolver;
    use volterra_dec::stokes::{compute_dual_areas, extract_coords, gaussian_curvature};

    let domain = sphere_domain(4);
    let coords = extract_coords(&domain.mesh);
    let nv = coords.len();
    let areas = compute_dual_areas(nv, &domain.mesh.simplices, &coords);
    let k = gaussian_curvature(nv, &domain.mesh.simplices, &coords, &areas);

    // At twice the curvature the three rotations are free.
    let shift: Vec<f64> = k.iter().map(|x| 2.0 * x).collect();
    let solver = PoissonSolver::new_shifted(&domain.ops, &shift, &coords).unwrap();
    assert_eq!(
        solver.kernel_dimension(),
        3,
        "a sphere has three Killing fields"
    );

    // At the curvature itself, or with no shift, none of them is.
    let half: Vec<f64> = k.to_vec();
    let s1 = PoissonSolver::new_shifted(&domain.ops, &half, &coords).unwrap();
    assert_eq!(s1.kernel_dimension(), 0, "at K rather than 2K none is free");
    let zero = vec![0.0; nv];
    let s0 = PoissonSolver::new_shifted(&domain.ops, &zero, &coords).unwrap();
    assert_eq!(s0.kernel_dimension(), 0, "an unshifted solve has no rotation kernel");
}

/// The Killing count is a property of the surface, not of the mesh.
///
/// A sphere has three rotations at every refinement, so the number the solve
/// projects out must not move with the mesh. The count was once normalised by
/// the response of a pseudo-random probe, and a random vector is all high
/// frequency, so its response is set by the smallest triangle rather than by
/// anything geometric. That read 0 here at level 1 while reading 3 at levels 2
/// and 3, and read 3 on a genus-2 surface, which has no Killing field at all.
#[test]
fn the_killing_count_does_not_move_with_the_mesh() {
    use volterra_dec::poisson::PoissonSolver;
    use volterra_dec::stokes::{compute_dual_areas, extract_coords, gaussian_curvature};

    for level in [1usize, 2, 3, 4] {
        let domain = sphere_domain(level);
        let coords = extract_coords(&domain.mesh);
        let nv = coords.len();
        let areas = compute_dual_areas(nv, &domain.mesh.simplices, &coords);
        let k = gaussian_curvature(nv, &domain.mesh.simplices, &coords, &areas);
        let shift: Vec<f64> = k.iter().map(|x| 2.0 * x).collect();
        let solver = PoissonSolver::new_shifted(&domain.ops, &shift, &coords).unwrap();
        assert_eq!(
            solver.kernel_dimension(),
            3,
            "level {level} ({nv} vertices) found a different number of rotations"
        );
    }
}

/// The active stress does positive net work on the fluid.
///
/// In steady Stokes flow the power the nematic delivers is spent entirely on
/// viscous dissipation, so `integral u . f` is positive for any flow that is
/// not at rest. It is an energy balance rather than a convention, so a sign
/// error anywhere between the stress and the velocity fails it.
///
/// The reference reports positive power around each `+1/2` defect, which is
/// the same statement made locally.
#[test]
fn the_active_stress_does_positive_work() {
    use volterra_core::ActiveNematicParams;
    use volterra_dec::stokes::{SurfaceStokes, vertex_force_from_stress};

    let domain = sphere_domain(3);
    let coords = coords_of(&domain);
    let nv = coords.len();
    let w: Vec<f64> = (0..nv).map(|i| domain.ops.hodge.star0()[i]).collect();
    let solver = SurfaceStokes::new(&domain.ops, &domain.mesh).expect("stokes");

    let zeta = 0.5;
    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = zeta;
    params.eta = 1.0;

    for seed in [1u64, 7, 42] {
        let q = QField::random_perturbation(nv, 0.5, seed);
        let vel = solver.solve(&q, &params, &domain.ops, &domain.mesh);
        let sym1: Vec<f64> = q.q1.iter().map(|v| -zeta * v).collect();
        let sym2: Vec<f64> = q.q2.iter().map(|v| -zeta * v).collect();
        let anti = vec![0.0; nv];
        let f = vertex_force_from_stress(
            &sym1, &sym2, &anti, &domain.mesh, &coords,
            solver.normals(), solver.e1_frames(),
        );
        let power: f64 = (0..nv)
            .map(|i| {
                w[i] * (vel.v[i][0] * f[i][0] + vel.v[i][1] * f[i][1] + vel.v[i][2] * f[i][2])
            })
            .sum();
        let speed: f64 = (0..nv)
            .map(|i| w[i] * (vel.v[i].iter().map(|c| c * c).sum::<f64>()))
            .sum();
        assert!(speed > 1e-12, "seed {seed}: the flow should not be at rest");
        assert!(
            power > 0.0,
            "seed {seed}: the active stress should do positive work, got {power:.3e} \
             against a flow of {speed:.3e}"
        );
    }
}
