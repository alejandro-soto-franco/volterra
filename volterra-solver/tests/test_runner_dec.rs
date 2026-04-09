//! Integration tests for the DEC dry active nematic runner.

use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::QFieldDec;
use volterra_solver::run_dry_active_nematic_dec;

#[test]
fn dec_dry_nematic_order_grows() {
    // On a flat torus in the active turbulent phase (a_eff < 0),
    // a small random Q perturbation should grow.
    //
    // CFL constraint: gamma_r * k_r * dt / dx^2 < 1.
    // unit_square_grid(4) has dx = 0.25, so max stable dt ~ dx^2 / (gamma_r * k_r) = 0.0625.
    // We use dt = 0.005 for a comfortable margin.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.005;

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.001, 42);
    let s_before = q0.mean_order_param();

    assert!(
        params.a_eff() < 0.0,
        "need active regime for this test (a_eff = {})",
        params.a_eff()
    );

    let (q_fin, stats) = run_dry_active_nematic_dec(&q0, &params, &ops, None, 200, 100);

    let s_after = q_fin.mean_order_param();
    assert!(
        s_after > s_before,
        "order should grow in active phase: s_before={s_before}, s_after={s_after}"
    );
    assert_eq!(stats.len(), 3, "expected snapshots at steps 0, 100, 200");
    assert!(stats[0].time < 1e-10, "first snapshot should be at t=0");
}

#[test]
fn dec_uniform_q_laplacian_vanishes() {
    // The Laplacian of a uniform Q on a flat mesh should be zero.
    let mesh = FlatMesh::unit_square_grid(8);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let nv = mesh.n_vertices();
    let q = QFieldDec::uniform(nv, 0.3, 0.1);
    let q_layout = q.to_lichnerowicz_layout();
    let lap = ops.apply_lichnerowicz_laplacian(&q_layout, None);

    let lap_norm = lap.norm();
    assert!(
        lap_norm < 1e-10,
        "Laplacian of uniform Q on flat mesh should vanish, got norm = {lap_norm}"
    );
}

#[test]
fn dec_zero_activity_relaxes_to_equilibrium() {
    // With zeta_eff = 0 (passive) and a_landau < 0 (ordered nematic),
    // Q should relax toward equilibrium, not blow up.
    //
    // CFL: unit_square_grid(4) dx=0.25, dt=0.001 => CFL = 1*1*0.001/0.0625 = 0.016. Safe.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = 0.0; // passive
    params.a_landau = -0.5; // ordered phase
    params.dt = 0.001;

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.3, 99);

    let (q_fin, _stats) = run_dry_active_nematic_dec(&q0, &params, &ops, None, 500, 500);

    let s_fin = q_fin.mean_order_param();
    assert!(s_fin.is_finite(), "mean_s must be finite, got {s_fin}");
    assert!(
        s_fin < 10.0,
        "passive nematic should not blow up: mean_s = {s_fin}"
    );
}

#[test]
fn dec_molecular_field_zero_q() {
    // H(Q=0) = 0 identically (no constant term in LdG).
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);
    let params = ActiveNematicParams::default_test();
    let q = QFieldDec::zeros(mesh.n_vertices());

    let h = volterra_dec::molecular_field_dec(&q, &params, &ops, None);
    let h_norm: f64 = h.q1.iter().chain(&h.q2).map(|x| x.abs()).sum();
    assert!(h_norm < 1e-12, "H(Q=0) should vanish, got {h_norm}");
}
