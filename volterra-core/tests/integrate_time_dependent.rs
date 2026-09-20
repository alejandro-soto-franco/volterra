//! Order of the integrator on a right-hand side that depends on the time.
//!
//! The magnetic drive makes the molecular field explicitly time dependent, and
//! the drive frequency is the axis the design map is swept along. An integrator
//! that holds the forcing at the step's start across all four stages is first
//! order in it however good its treatment of the rest, so the order is measured
//! here rather than assumed.
//!
//! `dy/dt = cos(t)`, `y(0) = 0`, exact `y(1) = sin(1)`.

use volterra_core::sim::integrate::FieldVec;
use volterra_core::sim::integrate::{rk4, rk4_t};

#[derive(Clone, Debug)]
struct Scalar(f64);

impl FieldVec for Scalar {
    fn add_scaled(&self, other: &Self, s: f64) -> Self {
        Scalar(self.0 + s * other.0)
    }
}

fn integrate_staged(n: usize) -> f64 {
    let dt = 1.0 / n as f64;
    let mut y = Scalar(0.0);
    for i in 0..n {
        y = rk4_t(&y, i as f64 * dt, dt, |_y, t| Scalar(t.cos()));
    }
    (y.0 - 1.0_f64.sin()).abs()
}

fn integrate_frozen(n: usize) -> f64 {
    let dt = 1.0 / n as f64;
    let mut y = Scalar(0.0);
    for i in 0..n {
        let t = i as f64 * dt;
        y = rk4(&y, dt, |_y| Scalar(t.cos()));
    }
    (y.0 - 1.0_f64.sin()).abs()
}

fn order(f: impl Fn(usize) -> f64) -> f64 {
    let (coarse, fine) = (f(32), f(64));
    (coarse / fine).log2()
}

#[test]
fn staging_the_time_gives_fourth_order() {
    let p = order(integrate_staged);
    assert!(
        p > 3.8,
        "a staged right-hand side must converge at fourth order, measured {p}"
    );
}

#[test]
fn freezing_the_time_costs_three_orders() {
    // The control that gives the first test its meaning: the same integrator on
    // the same problem, with the forcing held at the step's start, is first
    // order. Without this the fourth-order claim could be read off any
    // implementation that happened to be accurate at these two resolutions.
    let p = order(integrate_frozen);
    assert!(
        p < 1.5,
        "holding the forcing fixed is expected to be first order, measured {p}"
    );
}

#[test]
fn an_autonomous_right_hand_side_matches_the_existing_rk4_exactly() {
    // rk4_t must not become a second implementation that drifts from rk4. On a
    // right-hand side that ignores the time the two are the same arithmetic and
    // must agree to the last bit.
    let dt = 0.017;
    let mut a = Scalar(0.3);
    let mut b = Scalar(0.3);
    for i in 0..50 {
        a = rk4(&a, dt, |y| Scalar(-2.0 * y.0));
        b = rk4_t(&b, i as f64 * dt, dt, |y, _t| Scalar(-2.0 * y.0));
    }
    assert_eq!(
        a.0, b.0,
        "the two integrators diverged on an autonomous problem"
    );
}
