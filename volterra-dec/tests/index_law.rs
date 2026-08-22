//! The index law on the epitrochoid family, end to end.
//!
//! A tangentially anchored line field in a simply connected domain has total
//! index equal to the winding of the boundary line field, so the wall fixes how
//! many defects the domain must hold before any physics runs. For the
//! epitrochoid the tangent traces `w(u) = exp(i u)(1 + d exp(i k u))` and that
//! winding is
//!
//! ```text
//!   d < 1    1            the second factor has real part at least 1 - d > 0
//!   d = 1    (k + 2)/2    w = 2 cos(k u/2) exp(i (k+2) u/2), the k sign
//!                         changes being cusps a line field cannot see
//! ```
//!
//! **The two are separated by resolution, not by `d`.** An earlier revision of
//! this file premised that "smoothing the cusps at `d < 1` removes the boundary
//! contribution and the interior is back to 1", and exercised only `d = 0.72`,
//! where it happens to hold. It fails at the `d = 0.9` and `d = 0.99` every
//! production run uses. Near a blunted tip the tangent line swings through
//! `2 arcsin(d)` and back, and a boundary step spanning the swing books the
//! wrong branch and adds a half turn. `Epitrochoid::aliasing_deficit` measures
//! how much one step can accumulate, and the reading is spacing-dependent
//! exactly when that exceeds a half turn, which fixes a threshold
//! `d_c(k)` bracketed by `1/sqrt(2)` and `sin(pi (k+2)/(4(k+1)))`.
//!
//! So there are three regimes and all three are tested here: a true cusp, a
//! boundary that resolves a blunted tip, and the production spacing, which does
//! not. Every `d = 0.99` run therefore integrates a cusped wall, and its
//! recorded `imposed_charge` says so.
//!
//! The exact statements are machine-checked in
//! `cgpo-reproduction/symbolic-review/forms/sympy/index_law.py`, 107 checks.

use std::f64::consts::PI;

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};

/// `q = 1 + k/2`, so `k = 2(q - 1)`: cardioid, nephroid, trefoiloid,
/// quatrefoiloid, quintefoiloid. Radii are the paper's own per-shape scales.
const SHAPES: [(f64, usize, f64); 5] = [
    (1.5, 1, 54.271546),
    (2.0, 2, 49.778694),
    (2.5, 3, 49.776248),
    (3.0, 4, 49.776195),
    (3.5, 5, 49.776153),
];

fn imposed(curve: Epitrochoid, o: MeshOpts) -> (f64, f64) {
    let mesh = confined_mesh(curve, o);
    let (charge, worst_step, _) = mesh.imposed_charge(1.0);
    (charge, worst_step)
}

/// The spacing a production run uses: `ACT_H=1.0 ACT_HMIN=1.0`, so `h_min` is
/// pinned at `h_bulk` and nothing refines towards the tip.
fn production_opts() -> MeshOpts {
    MeshOpts { h_bulk: 1.0, h_min: 1.0, cusp_edge: 0.0, ..Default::default() }
}

/// A boundary that refines to a quarter of the tip's own radius of curvature.
fn resolving_opts(c: &Epitrochoid) -> MeshOpts {
    MeshOpts { h_bulk: 4.0, h_min: c.cusp_radius() / 4.0, ..Default::default() }
}

#[test]
fn a_cusped_domain_imposes_one_plus_half_the_cusp_count() {
    for (q, k, r) in SHAPES {
        let c = Epitrochoid { q, d: 1.0, r };
        let (charge, worst) = imposed(
            c,
            MeshOpts { h_bulk: 1.5, h_min: 1.5, cusp_edge: 1.5, ..Default::default() },
        );
        let want = 1.0 + k as f64 / 2.0;
        assert!(
            (charge - want).abs() < 1e-9,
            "k = {k}: imposed {charge:+.6} where the index law wants {want:+.2}"
        );
        assert!((c.exact_winding() - want).abs() < 1e-12, "k = {k}: exact_winding disagrees");
        // A boundary sampled too coarsely to follow the anchoring would give the
        // right number by luck, so the step is checked as well.
        assert!(worst < 60.0, "k = {k}: worst boundary step {worst:.1} deg is too coarse");
    }
}

#[test]
fn a_resolved_boundary_imposes_one_whatever_its_lobe_count() {
    // The claim the old test made, now stated with the condition it needs: a
    // boundary refined to the tip reads 1 at every lobe count AND at every d
    // below 1, including the 0.9 and 0.99 the old premise failed at.
    for (q, k, r) in SHAPES {
        for d in [0.72, 0.9, 0.99] {
            let c = Epitrochoid { q, d, r };
            let o = resolving_opts(&c);
            assert!(
                o.h_min < c.cusp_radius(),
                "k = {k}, d = {d}: fixture does not resolve the tip"
            );
            let (charge, _) = imposed(c, o);
            assert!(
                (charge - 1.0).abs() < 1e-9,
                "k = {k} at d = {d}: imposed {charge:+.6} where a resolved smooth \
                 boundary wants +1"
            );
        }
    }
}

#[test]
fn the_production_spacing_at_d_0_99_imposes_the_cusped_winding() {
    // What every d = 0.99 run actually integrates. Uniform elements at h = 1
    // against a tip radius near 1e-3, so no step comes close to following the
    // swing and each tip contributes the half turn it does not have.
    for (q, k, r) in SHAPES {
        let c = Epitrochoid { q, d: 0.99, r };
        assert!(
            c.cusp_radius() < 1e-2,
            "k = {k}: fixture tip radius {} is not the production case",
            c.cusp_radius()
        );
        let (charge, worst) = imposed(c, production_opts());
        let want = 1.0 + k as f64 / 2.0;
        assert!(
            (charge - want).abs() < 1e-9,
            "k = {k} at d = 0.99, h = 1: imposed {charge:+.6}, wanted the cusped \
             {want:+.2}; worst step {worst:.1} deg"
        );
        // and the same curve resolved gives the other answer, so this is the
        // sampling and not the shape.
        let (fine, _) = imposed(c, resolving_opts(&c));
        assert!(
            (fine - 1.0).abs() < 1e-9,
            "k = {k}: the same d = 0.99 curve resolved should impose 1, got {fine:+.6}"
        );
    }
    // Recorded values, from each run's own consts.json:
    //   gold_card_d0.99_ncl6.5_als1.0_s0     imposed_charge 1.5
    //   prod_trefoiloid_d0.99_ncl11.47_s0    imposed_charge 2.5
    //   prod_neph_d0.72_s0                   imposed_charge 1.0
}

#[test]
fn below_the_alias_threshold_every_spacing_agrees() {
    // `1/sqrt(2)` is the universal floor: no epitrochoid at or below it can be
    // misread at any lobe count or any sampling, so coarse and fine must agree.
    let d = 0.70;
    assert!(d < 1.0 / 2.0_f64.sqrt());
    for (q, k, r) in SHAPES {
        let c = Epitrochoid { q, d, r };
        assert!(
            c.winding_is_sampling_independent(),
            "k = {k}: deficit {} exceeds a half turn below 1/sqrt(2)",
            c.aliasing_deficit()
        );
        for o in [
            MeshOpts { h_bulk: 4.0, h_min: 4.0, ..Default::default() },
            production_opts(),
            resolving_opts(&c),
        ] {
            let (charge, worst) = imposed(c, o);
            assert!(
                (charge - 1.0).abs() < 1e-9,
                "k = {k} at d = {d}: imposed {charge:+.6} at h_bulk {}, worst step \
                 {worst:.1} deg",
                o.h_bulk
            );
        }
    }
}

#[test]
fn the_alias_threshold_sits_inside_its_closed_form_brackets() {
    // Measured 2026-08-22 by bisecting the deficit, and cross-checked against
    // the same bisection in `index_law.py`.
    const WANT: [f64; 5] = [0.896316, 0.847487, 0.818864, 0.800000, 0.786612];
    let mut prev = 1.0;
    for (i, (_, k, _)) in SHAPES.iter().enumerate() {
        let dc = Epitrochoid::alias_threshold(*k);
        let upper = (PI * (*k as f64 + 2.0) / (4.0 * (*k as f64 + 1.0))).sin();
        assert!(
            1.0 / 2.0_f64.sqrt() < dc && dc < upper,
            "k = {k}: d_c {dc:.6} outside [1/sqrt(2), {upper:.6}]"
        );
        assert!((dc - WANT[i]).abs() < 1e-5, "k = {k}: d_c {dc:.6}, wanted {:.6}", WANT[i]);
        assert!(dc < prev, "k = {k}: d_c should fall with the lobe count");
        prev = dc;
    }
}

#[test]
fn a_misreading_is_always_a_whole_number_of_half_turns() {
    // The shape of the error, across the family. A boundary either follows a
    // tip's swing or steps over it, so what it reports is `1 + j/2` for `j` the
    // count of tips it stepped over, and nothing between. `j = 0` whenever the
    // deficit stays below a half turn, which is the direction the criterion
    // states without qualification.
    for (q, k, r) in SHAPES {
        let dc = Epitrochoid::alias_threshold(k);
        for d in [0.5, 0.7, dc - 0.02, dc + 0.02, 0.95, 0.99] {
            let c = Epitrochoid { q, d, r };
            for h in [1.0, 2.0, 4.0] {
                let o = MeshOpts { h_bulk: h, h_min: h, cusp_edge: 0.0, ..Default::default() };
                let (charge, worst) = imposed(c, o);
                let j = 2.0 * (charge - 1.0);
                assert!(
                    (j - j.round()).abs() < 1e-9 && j.round() >= 0.0 && j.round() <= k as f64,
                    "k = {k}, d = {d:.4}, h = {h}: imposed {charge:+.6} is not 1 + j/2 for                      0 <= j <= {k}; worst step {worst:.1} deg"
                );
                if c.winding_is_sampling_independent() {
                    assert!(
                        (charge - 1.0).abs() < 1e-9,
                        "k = {k}, d = {d:.4}, h = {h}: deficit {:.4} is below a half turn,                          so no spacing may misread, yet this one imposed {charge:+.6}",
                        c.aliasing_deficit()
                    );
                }
            }
        }
    }
}

#[test]
fn the_production_spacing_starts_misreading_once_the_tip_falls_below_a_fortieth() {
    // Above the threshold a misreading is POSSIBLE, and whether a given boundary
    // realises it depends on where its samples land, so the onset in `d` is
    // measured rather than derived. This mesher samples the wall at
    // `boundary_frac` of the local radius of curvature, which follows the swing
    // on its own until the tip is small enough that the sampler stops keeping
    // up. Measured 2026-08-22 at the production spacing `h = 1`:
    //
    // ```text
    //   k = 1   reads 1 through d = 0.97 (R 3.5e-2), cusped from d = 0.98 (R 1.5e-2)
    //   k = 2   reads 1 through d = 0.95 (R 5.0e-2), cusped from d = 0.96 (R 3.2e-2)
    //   k = 3   reads 1 at     d = 0.90 (R 1.5e-1), cusped from d = 0.95 (R 3.6e-2)
    //   k = 4   already cusped at d = 0.90 (R 1.2e-1)
    // ```
    //
    // so the onset sits near `R_cusp` of a few times `1e-2`, and the `d = 0.9`
    // runs are on both sides of it depending on the lobe count. Only the two
    // ends are asserted, since the middle is where the sampling decides.
    for (q, k, r) in SHAPES {
        let coarse = MeshOpts { h_bulk: 1.0, h_min: 1.0, cusp_edge: 0.0, ..Default::default() };
        let (loose, _) = imposed(Epitrochoid { q, d: 0.995, r }, coarse);
        assert!(
            (loose - (1.0 + k as f64 / 2.0)).abs() < 1e-9,
            "k = {k} at d = 0.995: a tip near 1e-3 should read cusped, got {loose:+.6}"
        );
        let c = Epitrochoid { q, d: 0.85, r };
        if c.winding_is_sampling_independent() {
            let (tight, _) = imposed(c, coarse);
            assert!(
                (tight - 1.0).abs() < 1e-9,
                "k = {k} at d = 0.85: below the threshold, got {tight:+.6}"
            );
        }
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
        // The tip scale and the aliasing criterion are different things: R_cusp
        // has no lobe-count dependence and the threshold does.
        assert!(Epitrochoid { q, d: 1.0, r }.aliasing_deficit().is_infinite());
    }
}
