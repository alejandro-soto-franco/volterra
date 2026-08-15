//! Milestone-1 validation: FIRE must find the same equilibrium a long
//! Euler run finds, on a case with a closed-form answer.
//!
//! Case: a spatially uniform director along z, so the Laplacian is exactly
//! zero everywhere and only the bulk Landau-de Gennes term acts. With
//! `zeta_eff = 0` (passive) the bulk molecular field for a uniaxial state
//! `Q = S(nn - I/3)` is `H = Q * (-a_eff - (4/3) c S^2)`, using
//! `Tr((nn-I/3)^2) = 2/3` for a unit director in 3D. The nonzero-`H` root is
//!
//! ```text
//! S0 = sqrt(-3 a_eff / (4 c))
//! ```
//!
//! With `default_test()`'s `a_landau = -0.5`, `c_landau = 4.5`, `zeta_eff =
//! 0` (so `a_eff = a_landau`), `S0 = sqrt(1.5 / 18) = 0.288675...`.
//!
//! This is deliberately a case "where the answer is known": the closed-form
//! `S0`, not just self-consistency between the two integrators.

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_solver::{
    beris_edwards_rhs_3d_par_dry, euler_step_par, fire_minimize_3d_par, FireParams,
};

fn uniform_z_director(s: f64) -> [f64; 5] {
    [-s / 3.0, 0.0, 0.0, -s / 3.0, 0.0]
}

fn scalar_order_param(q: &[f64; 5]) -> f64 {
    // Recover S from Q = S(nn - I/3) along z: q11 = -S/3.
    -3.0 * q[0]
}

#[test]
fn fire_and_long_euler_agree_on_the_analytic_equilibrium() {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = 6;
    p.ny = 6;
    p.nz = 6;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;

    let a_eff = p.a_eff();
    let c = p.c_landau;
    let s0_analytic = (-3.0 * a_eff / (4.0 * c)).sqrt();
    assert!(
        s0_analytic.is_finite() && s0_analytic > 0.0,
        "test setup must have a real, nonzero equilibrium S0"
    );

    let s_start = 0.1; // well away from s0_analytic
    let q0 = QField3D::uniform(p.nx, p.ny, p.nz, p.dx, uniform_z_director(s_start));

    // FIRE
    let fire_params = FireParams::open_qmin_defaults(p.dt, 1e-9, 5000);
    let fire_result = fire_minimize_3d_par(&q0, &p, &fire_params, 0.0);
    assert!(
        fire_result.converged,
        "FIRE did not converge: force_max={}",
        fire_result.force_max
    );
    let s_fire = scalar_order_param(&fire_result.q.q[0]);

    // long Euler run
    let mut q_euler = q0.clone();
    let mut max_rhs = f64::INFINITY;
    for _ in 0..200_000 {
        let rhs = beris_edwards_rhs_3d_par_dry(&q_euler, &p, 0.0);
        max_rhs = rhs
            .q
            .iter()
            .map(|c| c.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
            .fold(0.0_f64, f64::max);
        if max_rhs < 1e-9 {
            break;
        }
        q_euler = euler_step_par(&q_euler, p.dt, &rhs);
    }
    assert!(
        max_rhs < 1e-6,
        "long Euler run did not settle: max|dQ/dt|={max_rhs}"
    );
    let s_euler = scalar_order_param(&q_euler.q[0]);

    // cross-checks, all against the SAME closed-form S0
    let tol_vs_analytic = 1e-4;
    let tol_cross = 1e-5;

    assert!(
        (s_fire - s0_analytic).abs() < tol_vs_analytic,
        "FIRE equilibrium S={s_fire} disagrees with analytic S0={s0_analytic}"
    );
    assert!(
        (s_euler - s0_analytic).abs() < tol_vs_analytic,
        "Euler equilibrium S={s_euler} disagrees with analytic S0={s0_analytic}"
    );
    assert!(
        (s_fire - s_euler).abs() < tol_cross,
        "FIRE and Euler equilibria disagree: fire={s_fire} euler={s_euler}"
    );

    // Every site must agree, not just site 0 (the field started uniform and
    // has no spatial coupling in this case, so it must stay uniform).
    for k in 0..fire_result.q.len() {
        let s_k = scalar_order_param(&fire_result.q.q[k]);
        assert!(
            (s_k - s0_analytic).abs() < tol_vs_analytic,
            "FIRE site {k} disagrees with analytic S0: S={s_k}"
        );
    }
}
