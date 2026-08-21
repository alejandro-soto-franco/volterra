//! The index law on the epitrochoid family, end to end.
//!
//! A tangentially anchored line field in a simply connected domain has total
//! index 1. A re-entrant cusp has interior angle `2 pi` and holds `-1/2` of that
//! on the boundary, so a domain with `k` cusps leaves `1 + k/2` for the
//! interior. Smoothing the cusps at `d < 1` removes the boundary contribution
//! and the interior is back to `1`.
//!
//! This exercises the curve, the mesher and the anchoring together, and it is
//! the property every confined run depends on: it fixes how many defects the
//! domain must hold before any physics runs.

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};

/// `q = 1 + k/2`, so `k = 2(q - 1)`: cardioid, nephroid, trefoiloid,
/// quatrefoiloid, quintefoiloid.
const SHAPES: [(f64, usize, f64); 5] = [
    (1.5, 1, 54.271546),
    (2.0, 2, 49.778694),
    (2.5, 3, 49.776248),
    (3.0, 4, 49.776195),
    (3.5, 5, 49.776153),
];

fn imposed(q: f64, d: f64, r: f64, cusp_edge: f64) -> (f64, f64) {
    let mesh = confined_mesh(
        Epitrochoid { q, d, r },
        MeshOpts { h_bulk: 1.5, h_min: 1.5, cusp_edge, ..Default::default() },
    );
    let (charge, worst_step, _) = mesh.imposed_charge(1.0);
    (charge, worst_step)
}

#[test]
fn a_cusped_domain_imposes_one_plus_half_the_cusp_count() {
    for (q, k, r) in SHAPES {
        let (charge, worst) = imposed(q, 1.0, r, 1.5);
        let want = 1.0 + k as f64 / 2.0;
        assert!(
            (charge - want).abs() < 1e-9,
            "k = {k}: imposed {charge:+.6} where the index law wants {want:+.2}"
        );
        // A boundary sampled too coarsely to resolve the anchoring would give
        // the right number by luck, so the step is checked as well.
        assert!(worst < 60.0, "k = {k}: worst boundary step {worst:.1} deg is too coarse");
    }
}

#[test]
fn a_smooth_domain_imposes_one_whatever_its_lobe_count() {
    for (q, k, r) in SHAPES {
        let (charge, _) = imposed(q, 0.72, r, 0.0);
        assert!(
            (charge - 1.0).abs() < 1e-9,
            "k = {k} at d = 0.72: imposed {charge:+.6} where a smooth boundary wants +1"
        );
    }
}

#[test]
fn the_cusp_radius_vanishes_only_at_d_equals_one() {
    // R_cusp = a c (1-d)^2 / |1 - c d| with c = k+1 and a = r/(k+2). It is the
    // radius the mesh grades towards, so a run at d < 1 has a finite tip and a
    // run at d = 1 has a genuine cusp.
    for (q, k, r) in SHAPES {
        let sharp = Epitrochoid { q, d: 1.0, r }.cusp_radius();
        let round = Epitrochoid { q, d: 0.72, r }.cusp_radius();
        assert!(sharp < 1e-12, "k = {k}: d = 1 should be a true cusp, got R = {sharp:.3e}");
        assert!(round > 1.0, "k = {k}: d = 0.72 should have a finite tip, got R = {round:.3e}");
    }
}
