//! Library-level smoke test: five CGPO steps on a 16×16 grid, in-process.
//!
//! Drives the solver through the public step API and asserts every field
//! stays finite after each step. Complements `test_cli_smoke.rs` (which
//! uses a subprocess); this one exercises the library directly.

use volterra_cgpo::{
    guard::check_finite,
    nephroid_boundary,
    step::{update_step_inner, State},
    Params,
};

/// `Params::new` args that match the canonical test constants used elsewhere:
///
/// - als = 1.0   → zeta = K / 1^2 = K = 16384
/// - ncl = 9.0   → c_landau = K / 81,  a_landau = -K / 81
/// - lambda = 0.7
/// - dt    = 1e-4
/// - max_p_iters = 20
fn make_params(lx: usize) -> Params {
    Params::new(lx, 1.0, 9.0, 0.7, 1e-4, 20)
}

#[test]
fn five_steps_keep_all_fields_finite() {
    let lx = 16;
    let ly = 16;

    let params = make_params(lx);
    let bnd = nephroid_boundary(lx, ly);
    let mut state = State::new(lx, ly);

    // Deterministic IC: set Q to a small uniform value inside the domain so
    // the solver has a non-trivial field to evolve, without needing any RNG.
    // Only interior cells get a non-zero value; boundary BCs will anchor
    // the outer layer on the first step.
    let s0 = params.s0;
    let amplitude = 0.1 * s0;
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if bnd.inside[idx] {
                // Q has 2 components packed as [Q0_00, Q1_00, Q0_01, Q1_01, ...]
                state.q[idx * 2]     =  amplitude;
                state.q[idx * 2 + 1] = -amplitude * 0.5;
            }
        }
    }

    let target_rel_change = 1e-4;

    for step in 0..5 {
        let _ = update_step_inner(&mut state, &params, &bnd, target_rel_change);
        check_finite(&state.q, "Q", step).expect("Q must be finite");
        check_finite(&state.u, "u", step).expect("u must be finite");
        check_finite(&state.p, "p", step).expect("p must be finite");
    }
}
