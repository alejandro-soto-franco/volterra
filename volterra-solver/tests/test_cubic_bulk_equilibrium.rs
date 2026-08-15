//! Correctness check for the cubic bulk term added to the 3D molecular field
//! (`volterra-solver/src/mol_field_3d.rs`), mirroring
//! `test_fire_matches_euler_equilibrium.rs` but with `b_landau != 0`.
//!
//! Case: a spatially uniform director along z (Laplacian exactly zero), so
//! only the bulk term acts. For a uniaxial state `Q = S(nn - I/3)`,
//! `Tr(Q^2) = (2/3)S^2` and `Tr(Q^3) = (2/9)S^3` (`mol_field_3d.rs`'s module
//! header derives the cubic contribution to `H`; this test derives the
//! resulting scalar equilibrium condition from the free energy directly).
//! The bulk free energy density is
//!
//! ```text
//! F(S) = (a_eff/3) S^2 + (2 b_landau/9) S^3 + (2 c_landau/9) S^4
//! ```
//!
//! `dF/dS = 0` at `S != 0` gives
//!
//! ```text
//! 3 a_eff + 3 b_landau S + 4 c_landau S^2 = 0
//! S0 = [-3 b_landau +/- sqrt(9 b_landau^2 - 48 a_eff c_landau)] / (8 c_landau)
//! ```
//!
//! reducing to the existing `S0 = sqrt(-3 a_eff / (4 c_landau))` when
//! `b_landau = 0` (confirmed directly:
//! `test_b_zero_reduces_to_the_existing_formula`, below).

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_solver::{
    beris_edwards_rhs_3d_par_dry, euler_step_fused_par, fire_minimize_3d_par, FireParams,
};

fn uniform_z_director(s: f64) -> [f64; 5] {
    [-s / 3.0, 0.0, 0.0, -s / 3.0, 0.0]
}

fn scalar_order_param(q: &[f64; 5]) -> f64 {
    -3.0 * q[0]
}

/// The positive root of `3 a_eff + 3 b S + 4 c S^2 = 0`.
fn analytic_s0_cubic(a_eff: f64, b_landau: f64, c_landau: f64) -> f64 {
    let disc = 9.0 * b_landau * b_landau - 48.0 * a_eff * c_landau;
    assert!(disc >= 0.0, "no real equilibrium for these constants");
    (-3.0 * b_landau + disc.sqrt()) / (8.0 * c_landau)
}

#[test]
fn test_b_zero_reduces_to_the_existing_formula() {
    let a_eff = -0.5_f64;
    let c_landau = 4.5_f64;
    let s0_cubic = analytic_s0_cubic(a_eff, 0.0, c_landau);
    let s0_old = (-3.0 * a_eff / (4.0 * c_landau)).sqrt();
    assert!(
        (s0_cubic - s0_old).abs() < 1e-14,
        "b_landau=0 must reduce exactly to the old S0 formula: {} vs {}",
        s0_cubic,
        s0_old
    );
}

#[test]
fn fire_and_long_euler_agree_on_the_cubic_analytic_equilibrium() {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = 6;
    p.ny = 6;
    p.nz = 6;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.dt = 0.005;
    p.b_landau = -1.5;

    let a_eff = p.a_eff();
    let s0_analytic = analytic_s0_cubic(a_eff, p.b_landau, p.c_landau);
    assert!(
        s0_analytic.is_finite() && s0_analytic > 0.0,
        "test setup must have a real, positive equilibrium S0, got {}",
        s0_analytic
    );

    // Confirm this is a stable minimum (F''(S0) > 0), not the unstable
    // branch or the S=0 saddle: F''(S) = 2a_eff/3 + 4b/3 S + 8c/3 S^2.
    let f_pp = 2.0 * a_eff / 3.0 + 4.0 * p.b_landau / 3.0 * s0_analytic
        + 8.0 * p.c_landau / 3.0 * s0_analytic * s0_analytic;
    assert!(
        f_pp > 0.0,
        "test setup's chosen root must be a stable minimum, F''(S0)={}",
        f_pp
    );

    let s_start = 0.15; // away from s0_analytic, same basin (no other critical point between 0 and s0_analytic)
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

    // Long Euler run
    let mut q_euler = q0.clone();
    for _ in 0..200_000 {
        euler_step_fused_par(&mut q_euler, &p, 0.0);
    }
    let s_euler = scalar_order_param(&q_euler.q[0]);

    assert!(
        (s_fire - s0_analytic).abs() < 1e-4,
        "FIRE equilibrium {} does not match analytic S0 {}",
        s_fire,
        s0_analytic
    );
    assert!(
        (s_euler - s0_analytic).abs() < 1e-3,
        "Euler equilibrium {} does not match analytic S0 {}",
        s_euler,
        s0_analytic
    );
    assert!(
        (s_fire - s_euler).abs() < 1e-3,
        "FIRE and Euler equilibria disagree: {} vs {}",
        s_fire,
        s_euler
    );

    // Every site must agree (uniform IC, uniform dynamics: no spatial variation).
    for k in 0..q0.len() {
        let s_k = scalar_order_param(&fire_result.q.q[k]);
        assert!((s_k - s_fire).abs() < 1e-12);
    }

    // The molecular field at the converged state must be (numerically) zero.
    let h = beris_edwards_rhs_3d_par_dry(&fire_result.q, &p, 0.0);
    let max_h = h
        .q
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(max_h < 1e-6, "molecular field not zero at equilibrium: {}", max_h);
}
