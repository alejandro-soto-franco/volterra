use volterra_fd::nephroid_boundary;

/// Interior-cell count for Lx=Ly=100 must be within ±5 of the Python reference (5621).
#[test]
fn interior_count_100() {
    let b = nephroid_boundary(100, 100);
    let count = b.interior_count();
    println!("interior_count(100,100) = {count}");
    assert!(
        (count as i64 - 5621).abs() <= 5,
        "expected ~5621, got {count}"
    );
}

/// Interior-cell count for Lx=Ly=60 must be within ±5 of the Python reference (1965).
#[test]
fn interior_count_60() {
    let b = nephroid_boundary(60, 60);
    let count = b.interior_count();
    println!("interior_count(60,60) = {count}");
    assert!(
        (count as i64 - 1965).abs() <= 5,
        "expected ~1965, got {count}"
    );
}

/// Every outer-boundary cell must have a unit normal (|n| ≈ 1, tol 1e-2).
/// Every inner-boundary cell must have a unit normal.
/// Off-boundary cells must have [0, 0] normals.
#[test]
fn normals_unit_length_and_zero_off_boundary() {
    let b = nephroid_boundary(60, 60);
    let tol = 1e-2_f64;

    for x in 0..b.lx {
        for y in 0..b.ly {
            let idx = x * b.ly + y;

            let on = b.outer_normals[idx];
            let inn = b.inner_normals[idx];

            if b.is_outer[idx] {
                let mag = (on[0] * on[0] + on[1] * on[1]).sqrt();
                assert!(
                    (mag - 1.0).abs() < tol,
                    "outer normal at ({x},{y}) has |n|={mag}, expected ~1"
                );
            } else {
                assert_eq!(
                    on,
                    [0.0, 0.0],
                    "outer_normals[{x},{y}] should be [0,0] for non-outer cell"
                );
            }

            if b.is_inner[idx] {
                let mag = (inn[0] * inn[0] + inn[1] * inn[1]).sqrt();
                assert!(
                    (mag - 1.0).abs() < tol,
                    "inner normal at ({x},{y}) has |n|={mag}, expected ~1"
                );
            } else {
                assert_eq!(
                    inn,
                    [0.0, 0.0],
                    "inner_normals[{x},{y}] should be [0,0] for non-inner cell"
                );
            }
        }
    }
}

/// Every inner cell must have at least one outer 4-neighbour (consistency check).
#[test]
fn inner_cells_have_outer_neighbour() {
    let b = nephroid_boundary(60, 60);
    for x in 0..b.lx {
        for y in 0..b.ly {
            if !b.is_inner[x * b.ly + y] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let has_outer = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)]
                .iter()
                .any(|&(nx, ny)| {
                    if nx < 0 || ny < 0 || nx >= b.lx as i64 || ny >= b.ly as i64 {
                        return false;
                    }
                    b.is_outer[nx as usize * b.ly + ny as usize]
                });
            assert!(
                has_outer,
                "inner cell ({x},{y}) has no outer 4-neighbour"
            );
        }
    }
}

/// Every outer cell must have at least one non-inside 4-neighbour.
#[test]
fn outer_cells_have_non_inside_neighbour() {
    let b = nephroid_boundary(60, 60);
    for x in 0..b.lx {
        for y in 0..b.ly {
            if !b.is_outer[x * b.ly + y] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let has_non_inside = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)]
                .iter()
                .any(|&(nx, ny)| {
                    if nx < 0 || ny < 0 || nx >= b.lx as i64 || ny >= b.ly as i64 {
                        return true; // out of grid = non-inside
                    }
                    !b.inside[nx as usize * b.ly + ny as usize]
                });
            assert!(
                has_non_inside,
                "outer cell ({x},{y}) has no non-inside 4-neighbour"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The epitrochoid family: cardioid (q=3/2), nephroid (q=2), trefoiloid (q=5/2)
//
// arXiv:2503.10880 Eq. SI.6. The nephroid cases above pin the ported behaviour
// against the Python reference; these pin the generalisation that carries it to
// the other members of the family.
// ---------------------------------------------------------------------------

use std::f64::consts::PI;
use volterra_fd::{cardioid_boundary, epitrochoid_boundary, trefoiloid_boundary, Epitrochoid};

/// The regularised outward normal winds through exactly `2 pi`, at every `q`.
///
/// This is worth pinning because it is the opposite of the natural guess. The
/// normal is `e^{iu} (1 + d e^{i k u})` up to scale, and for `d < 1` the second
/// factor never encircles the origin, so the winding is one turn whatever the
/// cusp count `k`. Only the sharp epicycloid at `d = 1` winds `q` times, by
/// picking up a jump of `pi` at each cusp.
///
/// So the boundary condition these normals build is a charge-1 condition, the
/// same as a disk's, and `net_charge` must be `1.0` for an epitrochoid run. The
/// excess defects are not imposed: each regularised cusp pins a `-1/2` defect
/// dynamically, and `n(+1/2) - k(-1/2) = 1` gives `n = 2 + k = 2q` mobile
/// defects, three in the cardioid and four in the nephroid. That count is a
/// result of a run, not a property of the mesh, so this test cannot check it.
#[test]
fn regularised_normal_winds_once() {
    for q in [1.5_f64, 2.0, 2.5] {
        let epi = Epitrochoid::new(q);
        let n_samples = 200_000;
        let mut total = 0.0_f64;
        let mut prev = f64::atan2(epi.normal(0.0)[1], epi.normal(0.0)[0]);
        for i in 1..=n_samples {
            let u = 2.0 * PI * i as f64 / n_samples as f64;
            let n = epi.normal(u);
            let angle = f64::atan2(n[1], n[0]);
            let mut delta = angle - prev;
            while delta > PI {
                delta -= 2.0 * PI;
            }
            while delta < -PI {
                delta += 2.0 * PI;
            }
            total += delta;
            prev = angle;
        }
        let turns = total / (2.0 * PI);
        assert!(
            (turns - 1.0).abs() < 1e-6,
            "q={q}: normal winds {turns} turns, expected 1"
        );
    }
}

/// The counted interior agrees with the closed-form area of Eq. SI.6.
///
/// A lattice count and a continuum area differ by a boundary term of order the
/// perimeter, so the tolerance is a percent rather than a rounding error. What
/// this catches is a wrong `q`, a wrong `d`, or an interior test that traces
/// some other curve, all of which move the area by tens of percent.
#[test]
fn interior_count_matches_closed_form_area() {
    for (label, q, lx) in [("cardioid", 1.5_f64, 200_usize), ("nephroid", 2.0, 100), ("trefoiloid", 2.5, 200)] {
        let epi = Epitrochoid::new(q);
        let b = epitrochoid_boundary(lx, lx, epi);
        let counted = b.interior_count() as f64;
        let closed_form = epi.area((lx / 2 - 1) as f64);
        let rel = (counted - closed_form).abs() / closed_form;
        println!(
            "{label} lx={lx}: counted {counted}, closed form {closed_form:.1}, \
             rel {rel:.4}, sqrt(A_sys) {:.2}",
            b.sqrt_area()
        );
        assert!(
            rel < 0.01,
            "{label}: counted {counted} against closed form {closed_form:.1}, rel {rel:.4}"
        );
    }
}

/// Every boundary normal is a unit vector for every member of the family.
///
/// At `q = 3/2` the pre-normalisation magnitude drops to `1 - d = 0.01` beside
/// the single cusp, where the polar-angle solve is at its worst conditioned; a
/// normal that failed to come back unit there would mean `solve_u` returned a
/// parameter for the wrong point on the curve.
#[test]
fn family_normals_are_unit() {
    for (label, b) in [
        ("cardioid", cardioid_boundary(100, 100)),
        ("trefoiloid", trefoiloid_boundary(100, 100)),
    ] {
        for idx in 0..b.lx * b.ly {
            for (layer, n) in [("outer", b.outer_normals[idx]), ("inner", b.inner_normals[idx])] {
                let on_layer = if layer == "outer" { b.is_outer[idx] } else { b.is_inner[idx] };
                if !on_layer {
                    continue;
                }
                let mag = (n[0] * n[0] + n[1] * n[1]).sqrt();
                assert!(
                    (mag - 1.0).abs() < 1e-9,
                    "{label} {layer} normal at flat index {idx} has |n|={mag}"
                );
            }
        }
    }
}

/// `d = 0` degenerates to a disk, whatever `q` is.
#[test]
fn zero_regularisation_is_a_disk() {
    let epi = Epitrochoid { q: 2.0, d: 0.0 };
    let b = epitrochoid_boundary(100, 100, epi);
    // The disk this reduces to has radius (2q-1) r / 2q = 3 * 49 / 4.
    let expected = PI * (3.0 * 49.0 / 4.0_f64).powi(2);
    let rel = (b.interior_count() as f64 - expected).abs() / expected;
    assert!(rel < 0.01, "counted {}, expected ~{expected:.1}", b.interior_count());
}
