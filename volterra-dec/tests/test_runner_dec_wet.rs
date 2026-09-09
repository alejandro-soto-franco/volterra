//! Integration tests for the wet active nematic DEC runner.

use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::QField;
use volterra_dec::run_wet_active_nematic_dec;

#[test]
fn wet_dec_nematic_runs_without_nan() {
    // Smoke test: the wet runner should complete without NaN or panic.
    // Uses a small mesh and few steps.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.00005; // very small dt for CFL stability with real 3D flow
    params.zeta_eff = 0.5;

    let nv = mesh.n_vertices();
    let q0 = QField::random_perturbation(nv, 0.001, 42);

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
    let q0 = QField::random_perturbation(nv, 0.001, 42);

    let (q_wet, _) = run_wet_active_nematic_dec(
        &q0, &params, &ops, &mesh, None, 100, 100,
    ).unwrap();

    let (q_dry, _) = volterra_dec::run_dry_active_nematic_dec(
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
    // Use K_r = 0.01 so elastic damping doesn't dominate the bulk growth.
    // CFL: gamma_r * k_r * dt / dx^2 = 1 * 0.01 * 0.001 / 0.0625 = 0.00016. Safe.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.k_r = 0.01;
    params.dt = 0.001;

    assert!(params.a_eff() < 0.0, "need active regime");

    let nv = mesh.n_vertices();
    let q0 = QField::random_perturbation(nv, 0.01, 42);
    let s_before = q0.mean_order_param();

    // Run 2000 steps (t=2.0) so the uniform mode grows by e^(|a_eff|*t) = e^3 ~ 20x.
    let (q_fin, _) = run_wet_active_nematic_dec(
        &q0, &params, &ops, &mesh, None, 2000, 2000,
    ).unwrap();

    let s_after = q_fin.mean_order_param();
    assert!(
        s_after > s_before,
        "order should grow: s_before={s_before}, s_after={s_after}"
    );
}

/// The chamber depth in the params reaches the confined solver.
///
/// Screening suppresses the flow: in the Darcy limit the peak speed goes as
/// `l_s^2`, so a strongly screened run sits closer to the zero-activity run than
/// an unscreened one does. Both halves matter. The first asserts the two runs
/// differ at all, which fails if the parameter never reaches the solver. The
/// second asserts they differ in the right DIRECTION, which fails if the
/// screening is applied with the wrong sign, since an inverted shift would
/// amplify the flow rather than suppress it.
#[test]
fn the_wet_runner_screens_when_the_params_say_so() {
    use volterra_core::Screening;
    use volterra_dec::run_wet_active_nematic_dec_confined;

    let cm = volterra_dec::epitrochoid::disk_mesh(1.0, 1.0, 60, 0.12);
    let mesh = cm.mesh;
    let bverts = cm.boundary_vertices;
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();
    let q0 = QField::random_perturbation(nv, 0.001, 42);

    let run = |screening: Screening, zeta: f64| {
        let mut p = ActiveNematicParams::default_test();
        p.dt = 0.0005;
        p.zeta_eff = zeta;
        p.screening = screening;
        run_wet_active_nematic_dec_confined(&q0, &p, &ops, &mesh, &bverts, None, 40, 40)
            .unwrap()
            .0
    };

    let unscreened = run(Screening::None, 0.5);
    let screened = run(Screening::Length(0.02), 0.5);
    let still = run(Screening::None, 0.0);

    let distance = |a: &QField, b: &QField| -> f64 {
        a.q1.iter()
            .zip(&b.q1)
            .chain(a.q2.iter().zip(&b.q2))
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };

    let d_unscreened = distance(&unscreened, &still);
    let d_screened = distance(&screened, &still);

    assert!(
        distance(&unscreened, &screened) > 0.0,
        "the screening never reached the solver: the two runs are identical"
    );
    assert!(
        d_screened < d_unscreened,
        "screening should suppress the flow, so the screened run should sit closer to the \
         zero-activity run: screened {d_screened:.6e} against unscreened {d_unscreened:.6e}"
    );
}
