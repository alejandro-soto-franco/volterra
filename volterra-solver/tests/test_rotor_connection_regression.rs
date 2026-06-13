//! Regression: the rotor-backed connection Laplacian keeps a short S^2
//! active-nematic run finite and stable over 20 steps. Guards the cutover at
//! the solver level, not just the operator level.

use volterra_solver::run_dry_active_nematic_dec_smoke;

#[test]
fn sim_sphere_rotor_path_stays_finite_20_steps() {
    // icosphere refinement 2: 162 vertices, 320 faces.
    let result = run_dry_active_nematic_dec_smoke(2, 20);
    let nv = 162;
    assert_eq!(result.len(), nv, "unexpected field length");
    assert!(
        result.iter().all(|v| v.is_finite()),
        "non-finite field value after 20 steps"
    );
}
