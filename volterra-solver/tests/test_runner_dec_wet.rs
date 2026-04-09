//! Integration tests for the wet active nematic DEC runner.

use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::QFieldDec;
use volterra_solver::run_wet_active_nematic_dec;

#[test]
fn wet_dec_nematic_runs_without_nan() {
    // Smoke test: the wet runner should complete without NaN or panic.
    // Uses a small mesh and few steps.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.001; // small dt for stability
    params.zeta_eff = 1.0; // moderate activity

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.001, 42);

    let (q_fin, stats) = run_wet_active_nematic_dec(
        &q0, &params, &ops, &mesh, None, 50, 50,
    ).unwrap();

    assert!(q_fin.mean_order_param().is_finite(), "mean_s should be finite");
    assert_eq!(stats.len(), 2, "snapshots at step 0 and 50");
}

#[test]
fn wet_dec_zero_activity_matches_dry() {
    // With zeta_eff = 0, the Stokes velocity is zero and the wet runner
    // should produce the same result as the dry runner.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.005;
    params.zeta_eff = 0.0; // no activity -> no flow -> dry equivalent

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.001, 42);

    let (q_wet, _) = run_wet_active_nematic_dec(
        &q0, &params, &ops, &mesh, None, 100, 100,
    ).unwrap();

    let (q_dry, _) = volterra_solver::run_dry_active_nematic_dec(
        &q0, &params, &ops, None, 100, 100,
    );

    // Should be identical (both reduce to pure molecular field RK4).
    let diff: f64 = q_wet.q1.iter().zip(&q_dry.q1)
        .chain(q_wet.q2.iter().zip(&q_dry.q2))
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff < 1e-10,
        "zero activity: wet and dry should agree, diff = {diff}"
    );
}

#[test]
fn wet_dec_order_grows_with_activity() {
    // With activity, the order parameter should grow from a small perturbation.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.005;

    assert!(params.a_eff() < 0.0, "need active regime");

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.001, 42);
    let s_before = q0.mean_order_param();

    let (q_fin, _) = run_wet_active_nematic_dec(
        &q0, &params, &ops, &mesh, None, 100, 100,
    ).unwrap();

    let s_after = q_fin.mean_order_param();
    assert!(
        s_after > s_before,
        "order should grow: s_before={s_before}, s_after={s_after}"
    );
}
