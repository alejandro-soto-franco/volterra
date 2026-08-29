//! Enhanced nematic locking on a periodic domain, against the two papers that
//! specify it.
//!
//! - Mitchell, Sabbir, Klein and Beller, "Modelling active nematics via the
//!   nematic locking principle", Soft Matter (2025), arXiv:2506.20996.
//! - Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing
//!   active nematics", Phys. Rev. E 109, 014606 (2024).

use rand::{RngExt, SeedableRng, rngs::StdRng};
use volterra_fd::{
    Dimensionless, Locking, Params,
    boundary::periodic_boundary,
    locking::{rms_and_median, rotation_rates},
    step::{State, update_step_inner},
};

fn random_director(q: &mut [f64], s0: f64, lx: usize, ly: usize, seed: u64) {
    use std::f64::consts::PI;
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..lx * ly {
        let theta: f64 = PI * rng.random::<f64>();
        let (s, c) = theta.sin_cos();
        q[i * 2] = s0 * (c * c - 0.5);
        q[i * 2 + 1] = s0 * (c * s);
    }
}

/// The reference states its own constants: `gamma = 5 * 256`, `C = 256^2`,
/// `K = 256^2`, `eta = 2560`, `zeta = (256/3)^2`, with `Re = 0.01`,
/// `ell_a = 3` and `ell_n = 1` (arXiv:2506.20996, Sect. V A). Reading them back
/// out of the dimensionless groups is what fixes the parameter mapping.
#[test]
fn from_dimensionless_reproduces_the_reference_constants() {
    let d = Dimensionless::nematic_locking(3.0);
    let p = Params::from_dimensionless(200, 200, d, Dimensionless::MITCHELL_K, 7.5e-5, 50);

    assert!((p.k_elastic - 65536.0).abs() < 1e-9, "K = {}", p.k_elastic);
    assert!((p.eta - 2560.0).abs() < 1e-9, "eta = {}", p.eta);
    assert!((p.gamma - 5.0 * 256.0).abs() < 1e-9, "gamma = {}", p.gamma);
    assert!((p.zeta - (256.0_f64 / 3.0).powi(2)).abs() < 1e-9, "zeta = {}", p.zeta);
    assert!((p.c_landau - 65536.0).abs() < 1e-9, "C = {}", p.c_landau);
    // C = -2A puts equilibrium at S = 1, the convention the switch width is in.
    assert!((p.a_landau + 0.5 * p.c_landau).abs() < 1e-9, "A = {}", p.a_landau);
    assert!((p.s0 - 1.0).abs() < 1e-12, "S_eq = {}", p.s0);

    assert!((p.active_length() - 3.0).abs() < 1e-9);
    assert!((p.coherence_length() - 1.0).abs() < 1e-9);
    // Re = K / (rho nu^2) and gamma_tilde = gamma nu / K, read back.
    assert!((p.k_elastic / (p.rho * p.eta * p.eta) - 0.01).abs() < 1e-12);
    assert!((p.gamma * p.eta / p.k_elastic - 50.0).abs() < 1e-9);
    assert!((p.c_landau / p.zeta - 9.0).abs() < 1e-9);
    // t_a = K / (zeta nu), the factor the entropy is made dimensionless with.
    assert!((p.active_time() - 3.515625e-3).abs() < 1e-12, "t_a = {}", p.active_time());
}

/// Mitchell et al. (2024) take `C = -A`, so the same groups give `S_eq = sqrt 2`.
#[test]
fn the_two_papers_differ_only_in_the_order_parameter_normalisation() {
    let a = Params::from_dimensionless(
        100, 100, Dimensionless::mitchell(3.0), Dimensionless::MITCHELL_K, 1e-4, 50);
    let b = Params::from_dimensionless(
        100, 100, Dimensionless::nematic_locking(3.0), Dimensionless::MITCHELL_K, 1e-4, 50);
    assert_eq!(a.k_elastic, b.k_elastic);
    assert_eq!(a.eta, b.eta);
    assert_eq!(a.gamma, b.gamma);
    assert_eq!(a.zeta, b.zeta);
    assert_eq!(a.c_landau, b.c_landau);
    assert!((a.a_landau - 2.0 * b.a_landau).abs() < 1e-9);
    assert!((a.s0 - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((b.s0 - 1.0).abs() < 1e-12);
}

/// Every constructor leaves the model standard, and a config written before the
/// field existed still reads back as the standard model.
#[test]
fn locking_is_off_by_default_and_absent_from_a_config_means_off() {
    assert!(Params::new(32, 2.8, 4.8, 1.0, 1e-4, -1).locking.is_none());
    assert!(
        Params::from_dimensionless(
            32, 32, Dimensionless::mitchell(3.0), Dimensionless::MITCHELL_K, 1e-4, 50)
        .locking
        .is_none()
    );

    let p = Params::new(32, 2.8, 4.8, 1.0, 1e-4, -1);
    let mut v: serde_json::Value = serde_json::to_value(&p).unwrap();
    v.as_object_mut().unwrap().remove("locking").expect("field present");
    let back: Params = serde_json::from_value(v).expect("deserialises without the field");
    assert!(back.locking.is_none());

    let with = p.with_locking(Locking::REFERENCE);
    let round: Params = serde_json::from_str(&serde_json::to_string(&with).unwrap()).unwrap();
    assert_eq!(round.locking, Some(Locking::REFERENCE));
}

/// Switching locking on must change the trajectory, and switching it off must
/// leave the step exactly as it was.
#[test]
fn the_switch_changes_the_step_only_when_it_is_on() {
    let (lx, ly) = (32, 32);
    let bnd = periodic_boundary(lx, ly);
    let d = Dimensionless::nematic_locking(3.0);
    let base = Params::from_dimensionless(lx, ly, d, Dimensionless::MITCHELL_K, 5e-5, 20);
    let benl = base.clone().with_locking(Locking::REFERENCE);

    let mut a = State::new(lx, ly);
    random_director(&mut a.q, base.s0, lx, ly, 7);
    let mut b = State::new(lx, ly);
    b.q.copy_from_slice(&a.q);
    let mut c = State::new(lx, ly);
    c.q.copy_from_slice(&a.q);

    for _ in 0..40 {
        update_step_inner(&mut a, &base, &bnd, 1e-6);
        update_step_inner(&mut b, &base, &bnd, 1e-6);
        update_step_inner(&mut c, &benl, &bnd, 1e-6);
    }
    // Two standard runs from the same state agree bit for bit.
    assert_eq!(a.q, b.q);
    // The modified one does not.
    let diff: f64 = a
        .q
        .iter()
        .zip(c.q.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(diff > 1e-6, "enhanced locking changed nothing: max diff {diff}");
}

/// The reference's central measurement, in miniature: enhanced locking must
/// collapse the *median* fracturing rate while leaving the advective rate alone.
///
/// The reference reports median `|omega_F|` falling from `0.0687` to
/// `7.14e-7`, five orders of magnitude, with median `|omega_A|` essentially
/// unchanged at `0.149` against `0.168`. This test asserts the direction and
/// the scale of the collapse on a grid small enough to run in a unit test, not
/// the reference's own numbers, which need its `200 x 200` domain and a
/// developed turbulent state.
#[test]
fn enhanced_locking_collapses_the_median_fracturing_rate() {
    let (lx, ly) = (64, 64);
    let bnd = periodic_boundary(lx, ly);
    let d = Dimensionless::nematic_locking(3.0);
    let be = Params::from_dimensionless(lx, ly, d, Dimensionless::MITCHELL_K, 5e-5, 20);
    let benl = be.clone().with_locking(Locking::REFERENCE);

    let run = |p: &Params| {
        let mut st = State::new(lx, ly);
        random_director(&mut st.q, p.s0, lx, ly, 11);
        for _ in 0..4000 {
            update_step_inner(&mut st, p, &bnd, 1e-6);
        }
        let r = rotation_rates(&st.u, &st.q, &st.h, p.gamma, p.s0, p.locking, &bnd);
        let (_, a_med) = rms_and_median(&r.omega_a, &bnd);
        let (_, f_med) = rms_and_median(&r.omega_f, &bnd);
        (a_med, f_med)
    };

    let (a_be, f_be) = run(&be);
    let (a_benl, f_benl) = run(&benl);

    // The advective rate is a property of the flow and stays the same order.
    assert!(
        a_benl > 0.2 * a_be && a_benl < 5.0 * a_be,
        "advective median moved too far: {a_be} -> {a_benl}"
    );
    // The fracturing rate collapses. Two orders of magnitude on this grid, over
    // a run far too short to develop turbulence; the reference sees five on its
    // own domain, in a developed state.
    assert!(
        f_benl < 1e-2 * f_be,
        "fracturing median did not collapse: {f_be} -> {f_benl}"
    );
    // And under the standard model it is a sizeable fraction of the advective
    // rate, which is the reference's complaint about that model.
    assert!(
        f_be > 0.05 * a_be,
        "standard model showed no bulk fracturing: {f_be} against {a_be}"
    );
}
