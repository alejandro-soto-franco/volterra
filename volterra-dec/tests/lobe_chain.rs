//! The lobe chain's imposed charge, and its independence from the neck.
//!
//! The device rests on the boundary forcing its defect count. A chain of `n`
//! lobes has `n - 1` interior cusps on each of its two walls, so `2(n - 1)`
//! in all, and a closed curve with `k` cusps imposes `1 + k/2`. That is `n`.
//!
//! The neck is what the design varies, so the charge has to be the same number
//! at every neck width. An earlier family tied the cusp to the neck and its
//! charge collapsed to zero the moment the neck opened; these tests exist so
//! that cannot happen again unnoticed.

use volterra_dec::confined::{MeshOpts, confined_mesh};
use volterra_dec::lobe_chain::LobeChain;

/// Imposed charge, with the cusp treatment named so a test can vary it.
///
/// `big`, the count of boundary steps past a right angle, is returned for
/// reporting and is never asserted on. A true cusp swings the tangent through
/// a half turn over one element, so it produces such a step by construction
/// and refining does not remove it: measured at 76.5 degrees at neck 0.35 and
/// 97.3 at neck 0.50, unchanged across three cusp treatments. `Epitrochoid`
/// documents the same thing on its own tip.
fn charge_with(n: usize, neck: f64, cusp_edge: f64, h_min: f64) -> (f64, usize) {
    let c = LobeChain::new(n, neck);
    let cm = confined_mesh(
        c,
        MeshOpts {
            h_bulk: 0.13,
            h_min,
            cusp_edge,
            ..Default::default()
        },
    );
    let (charge, _worst, big) = cm.imposed_charge(1.0);
    (charge, big)
}

fn charge_of(n: usize, neck: f64) -> (f64, usize) {
    charge_with(n, neck, 0.005, 0.004)
}

#[test]
fn a_two_lobe_chain_imposes_two() {
    let (charge, _big) = charge_of(2, 0.15);
    assert!(
        (charge - 2.0).abs() < 1e-6,
        "two lobes have two interior cusps and must impose 1 + 2/2 = 2, read {charge}"
    );
}

#[test]
fn the_charge_counts_the_lobes() {
    for n in 1..=4 {
        let (charge, _big) = charge_of(n, 0.15);
        assert!(
            (charge - n as f64).abs() < 1e-6,
            "{n} lobes must impose {n}, read {charge}"
        );
    }
}

#[test]
fn the_charge_does_not_move_with_the_neck() {
    // The failure this exists for: a family whose cusp is its neck loses both
    // together, and the charge falls to zero the moment the neck opens.
    let n = 3;
    for neck in [0.02, 0.08, 0.20, 0.35, 0.50, 0.65] {
        let (charge, _big) = charge_of(n, neck);
        assert!(
            (charge - 3.0).abs() < 1e-6,
            "neck = {neck} must leave the charge at 3, read {charge}"
        );
    }
}

#[test]
fn the_tangent_reverses_at_an_interior_cusp() {
    use volterra_dec::curve::PlaneCurve;
    let c = LobeChain::new(3, 0.2);
    let cusp = c.features()[0]; // the first interior cusp of the lower wall
    let eps = 1e-7;
    let before = c.tangent(cusp - eps);
    let after = c.tangent(cusp + eps);
    let dot = before[0] * after[0] + before[1] * after[1];
    assert!(
        dot < -0.999,
        "a cusp reverses the tangent; got t.t = {dot} across the feature"
    );
    assert!(
        c.speed(cusp) < 1e-6,
        "the speed collapses at a cusp; got {}",
        c.speed(cusp)
    );
}

#[test]
fn the_end_junctions_leave_the_tangent_continuous() {
    // A right-angle corner puts 2*d(theta) at exactly pi, the branch point of
    // the wrap in `imposed_charge`, and costs a half turn per corner. The end
    // walls are vertical and the train's tangent is vertical where they meet,
    // so there is no corner to mis-book.
    use std::f64::consts::TAU;
    use volterra_dec::curve::PlaneCurve;
    let c = LobeChain::new(2, 0.25);
    let eps = 1e-7;
    for f in [0.45, 0.50, 0.95, 0.0] {
        let u = f * TAU;
        let before = c.tangent(u - eps);
        let after = c.tangent(u + eps);
        let dot = before[0] * after[0] + before[1] * after[1];
        assert!(
            dot > 0.999,
            "the junction at {f} of the circuit turned the tangent: t.t = {dot}"
        );
    }
}

#[test]
fn the_cusp_tips_face_each_other_across_the_neck() {
    use volterra_dec::curve::PlaneCurve;
    let c = LobeChain::new(2, 0.3);
    let lower = c.point(c.features()[0]);
    let upper_param = c.features()[3];
    let upper = c.point(upper_param);
    assert!(
        (lower[0] - upper[0]).abs() < 1e-9,
        "opposing tips sit at the same station: {} against {}",
        lower[0],
        upper[0]
    );
    assert!(
        (upper[1] - lower[1] - c.neck_width()).abs() < 1e-9,
        "the gap between the tips is the neck width {}, measured {}",
        c.neck_width(),
        upper[1] - lower[1]
    );
}

#[test]
fn a_wide_neck_needs_a_finer_cusp_treatment() {
    // The charge survives to a neck of 0.80, and reading it there takes a
    // cusp element five times smaller. The coarse treatment does not read a
    // wrong-but-close number, it books the tip's half turn on the wrong branch
    // and loses a whole unit, so this is a cliff rather than a drift.
    let coarse = charge_with(3, 0.80, 0.02, 0.012).0;
    let fine = charge_with(3, 0.80, 0.001, 0.001).0;
    assert!(
        (coarse - 1.0).abs() < 1e-6,
        "the coarse cusp treatment is expected to mis-book at this neck, read {coarse}"
    );
    assert!(
        (fine - 3.0).abs() < 1e-6,
        "the fine cusp treatment must still read 3 at a neck of 0.80, read {fine}"
    );
}
