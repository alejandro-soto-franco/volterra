//! Integration test for the `CgpoStep` `PhysicsStep` adapter.
//!
//! Verifies that `CgpoStep` routes one CGPO finite-difference advance through
//! the shared `volterra_core::sim::PhysicsStep` trait and that the state
//! remains finite after several steps.

use volterra_core::sim::PhysicsStep;
use volterra_cgpo::{
    nephroid_boundary,
    sim_step::CgpoStep,
    step::State,
    Params,
};

fn make_params(lx: usize) -> Params {
    Params::new(lx, 1.0, 9.0, 0.7, 1e-4, 20)
}

#[test]
fn cgpo_step_advances_and_reports_finite_stats() {
    let lx = 16;
    let ly = 16;

    let params = make_params(lx);
    let boundary = nephroid_boundary(lx, ly);
    let mut state = State::new(lx, ly);

    // Deterministic IC matching test_smoke.rs: small uniform Q inside the domain.
    let amplitude = 0.1 * params.s0;
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if boundary.inside[idx] {
                state.q[idx * 2]     =  amplitude;
                state.q[idx * 2 + 1] = -amplitude * 0.5;
            }
        }
    }

    let mut physics = CgpoStep {
        params,
        boundary,
        target_rel_change: 1e-4,
    };

    // Run five steps through the PhysicsStep trait.
    for step_idx in 0..5 {
        let q_before: Vec<f64> = state.q.clone();
        let stats = physics.step(&mut state, step_idx as f64 * 1e-4);

        // The state must have changed (at least one Q component differs).
        assert!(
            state.q.iter().zip(q_before.iter()).any(|(a, b)| a != b),
            "step {step_idx}: state.q did not change"
        );

        // Any reported order_param must be finite.
        assert!(
            stats.order_param.is_none_or(f64::is_finite),
            "step {step_idx}: order_param is not finite: {:?}", stats.order_param
        );

        // All Q, u, p fields must remain finite.
        assert!(
            state.q.iter().all(|v| v.is_finite()),
            "step {step_idx}: state.q contains non-finite value"
        );
        assert!(
            state.u.iter().all(|v| v.is_finite()),
            "step {step_idx}: state.u contains non-finite value"
        );
        assert!(
            state.p.iter().all(|v| v.is_finite()),
            "step {step_idx}: state.p contains non-finite value"
        );
    }
}
