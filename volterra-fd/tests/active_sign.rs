//! The sign of the two-dimensional activity, read from what it does to a +1/2 disclination.
//!
//! With the director at half the polar angle, theta = phi / 2, the divergence
//! of Q points along +x, towards the tail where the director runs straight, and
//! the rounded head lies along -x. The wet solver forces the flow with
//! `-zeta_eff div Q`, the active stress `-zeta_eff Q` every volterra solver
//! uses, so a positive activity drives the core towards its head, the
//! extensile sense, and a negative one towards its tail, the contractile sense.
//! Without activity the core stays where it is.

use volterra_core::{ActiveNematicParams, QField2D};
use volterra_fd::run_active_nematic_hydro;

const N: usize = 64;

fn plus_half() -> QField2D {
    let mut q = vec![[0.0; 2]; N * N];
    for i in 0..N {
        for j in 0..N {
            // Off-node, so the core sits inside one plaquette.
            let x = i as f64 - N as f64 / 2.0 + 0.5;
            let y = j as f64 - N as f64 / 2.0 + 0.5;
            let theta = y.atan2(x) / 2.0;
            let s = 0.5 * (x.hypot(y) / 3.0).tanh();
            q[i * N + j] = [s * (2.0 * theta).cos(), s * (2.0 * theta).sin()];
        }
    }
    QField2D {
        q,
        nx: N,
        ny: N,
        dx: 1.0,
    }
}

/// The x coordinate of the +1/2 core, from the plaquettes winding by half a turn.
fn core_x(field: &QField2D) -> f64 {
    let angle = |i: usize, j: usize| {
        let [a, b] = field.q[i * N + j];
        b.atan2(a)
    };
    let wrap = |d: f64| {
        (d + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
    };
    let (mut sum, mut count) = (0.0, 0.0);
    for i in 0..N - 1 {
        for j in 0..N - 1 {
            let w = wrap(angle(i + 1, j) - angle(i, j))
                + wrap(angle(i + 1, j + 1) - angle(i + 1, j))
                + wrap(angle(i, j + 1) - angle(i + 1, j + 1))
                + wrap(angle(i, j) - angle(i, j + 1));
            if w / (4.0 * std::f64::consts::PI) > 0.2 {
                sum += i as f64 + 0.5 - (N as f64 / 2.0 - 0.5);
                count += 1.0;
            }
        }
    }
    assert!(count > 0.0, "no +1/2 core found");
    sum / count
}

fn moved(zeta: f64) -> f64 {
    let mut p = ActiveNematicParams::default_test();
    p.nx = N;
    p.ny = N;
    p.zeta_eff = zeta;
    p.a_landau = -0.5 + zeta / 2.0;
    p.validate().unwrap();
    let start = plus_half();
    let x0 = core_x(&start);
    let (end, _) = run_active_nematic_hydro(&start, &p, 300, 300);
    core_x(&end) - x0
}

#[test]
fn positive_activity_drives_a_plus_half_core_towards_its_head() {
    let dx = moved(0.3);
    assert!(dx < -2.0, "core moved {dx} along x");
}

#[test]
fn negative_activity_drives_a_plus_half_core_towards_its_tail() {
    let dx = moved(-0.3);
    assert!(dx > 2.0, "core moved {dx} along x");
}

#[test]
fn without_activity_the_core_stays() {
    let dx = moved(0.0);
    assert!(dx.abs() < 0.5, "core moved {dx} along x");
}

#[test]
fn a_negative_activity_is_a_valid_parameter_and_a_non_finite_one_is_not() {
    let mut p = ActiveNematicParams::default_test();
    p.zeta_eff = -1.0;
    assert!(p.validate().is_ok());
    assert!(p.defect_length().is_finite());
    p.zeta_eff = f64::NAN;
    assert!(p.validate().is_err());
}
