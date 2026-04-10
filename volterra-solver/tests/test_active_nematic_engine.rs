use nalgebra::SVector;

use cartan_dec::mesh_gen::icosphere;
use cartan_dec::hodge::HodgeStar;
use cartan_manifolds::sphere::Sphere;

use volterra_solver::nematic_field_2d::NematicField2D;
use volterra_solver::stokes_trait::KillingOperatorSolver;
use volterra_solver::{ActiveNematicEngine, EngineParams};

#[test]
fn test_engine_runs_on_sphere() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let nv = mesh.n_vertices();
    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();

    let params = EngineParams {
        pe: 1.0,
        lambda: 1.0,
        epsilon: 0.01,
        dt: 0.001,
        activity_sign: -1.0,
    };

    let stokes = Box::new(KillingOperatorSolver::new(&mesh, 1e3, 1e-4));

    let mut engine = ActiveNematicEngine::new(&mesh, &manifold, &hodge, params, stokes);

    // Start with a small random-ish perturbation.
    let mut field = NematicField2D::from_section(
        cartan_dec::line_bundle::Section::<2>::from_real_components(
            &(0..nv).map(|i| 0.01 * (i as f64 * 0.7).sin()).collect::<Vec<_>>(),
            &(0..nv).map(|i| 0.01 * (i as f64 * 1.3).cos()).collect::<Vec<_>>(),
        ),
    );

    // Run a few steps.
    let mut last_diag = None;
    for _ in 0..5 {
        let diag = engine.step(&mut field);
        assert!(diag.mean_order.is_finite(), "mean order should be finite");
        assert!(diag.time > 0.0);
        last_diag = Some(diag);
    }

    let diag = last_diag.unwrap();
    assert_eq!(diag.step, 5);
    assert!((diag.time - 0.005).abs() < 1e-10);

    // After normalisation, all |z| should be 1.
    for z in field.values() {
        assert!(
            (z.norm() - 1.0).abs() < 1e-10 || z.norm() < 1e-10,
            "after normalisation, |z| should be 1 or 0, got {}",
            z.norm()
        );
    }
}

#[test]
fn test_engine_zero_activity_preserves_field() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let nv = mesh.n_vertices();
    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();

    let params = EngineParams {
        pe: 1e10,  // Very large Pe: almost no diffusion.
        lambda: 1.0,
        epsilon: 0.01,
        dt: 0.001,
        activity_sign: 0.0,  // No activity.
    };

    let stokes = Box::new(KillingOperatorSolver::new(&mesh, 1e3, 1e-4));
    let mut engine = ActiveNematicEngine::new(&mesh, &manifold, &hodge, params, stokes);

    // Uniform unit nematic.
    let mut field = NematicField2D::uniform(nv, 0.5, 0.0);
    field.normalise();

    let order_before = field.mean_scalar_order();

    for _ in 0..3 {
        engine.step(&mut field);
    }

    let order_after = field.mean_scalar_order();

    // With zero activity and negligible diffusion, order should be preserved.
    assert!(
        (order_after - order_before).abs() < 0.1,
        "order should be roughly preserved: before={order_before}, after={order_after}"
    );
}

#[test]
fn test_engine_stream_function_backend() {
    use cartan_dec::Operators;
    use volterra_solver::stokes_trait::StreamFunctionStokes;

    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let nv = mesh.n_vertices();
    let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();

    // Unit sphere: K = 1 everywhere.
    let gaussian_k = vec![1.0; nv];

    let params = EngineParams {
        pe: 1.0,
        lambda: 1.0,
        epsilon: 0.01,
        dt: 0.001,
        activity_sign: -1.0,
    };

    let sf = StreamFunctionStokes::new(&ops, &mesh, &gaussian_k, 1.0).unwrap();
    let stokes: Box<dyn volterra_solver::StokesSolver> = Box::new(sf);

    let mut engine = ActiveNematicEngine::new(&mesh, &manifold, &hodge, params, stokes);

    let mut field = NematicField2D::from_section(
        cartan_dec::line_bundle::Section::<2>::from_real_components(
            &(0..nv).map(|i| 0.01 * (i as f64 * 0.7).sin()).collect::<Vec<_>>(),
            &(0..nv).map(|i| 0.01 * (i as f64 * 1.3).cos()).collect::<Vec<_>>(),
        ),
    );

    // Run a few steps with stream function backend.
    for _ in 0..5 {
        let diag = engine.step(&mut field);
        assert!(diag.mean_order.is_finite(), "mean order should be finite");
        assert_eq!(diag.stokes_residual, 0.0, "stream function is div-free by construction");
    }
}
