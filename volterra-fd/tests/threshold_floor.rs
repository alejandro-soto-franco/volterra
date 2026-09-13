//! The default disclination threshold floor, against the two states it separates.
//!
//! A threshold taken purely as a fraction of the field's own peak reports lines
//! in the noise of a field whose defects have annihilated, since `s` has a
//! largest value wherever the field is. A 32^3 dry run watched its threshold
//! fall from 6.3e-3 to 3.6e-9 while the order parameter sat at 0.499 and the
//! line count stayed near 38.
//!
//! The floor that stops that has to mean the same thing on any grid, so it is
//! scaled: `s` is quadratic in a Q gradient and a core turns the director
//! through a fixed angle over a lattice spacing, which makes `(q_eq / dx)^2` the
//! natural unit. These tests pin the floor between the two states rather than
//! against a number someone chose.

use volterra_braid::disclination::{disclination_lines_at_fraction, disclination_magnitude};
use volterra_core::{ActiveNematicParams3D, DEFAULT_DISCLINATION_FLOOR_COEFFICIENT};

fn uniaxial(n: [f64; 3], q_mag: f64) -> [f64; 5] {
    let t = 1.0 / 3.0;
    [
        q_mag * (n[0] * n[0] - t),
        q_mag * (n[0] * n[1]),
        q_mag * (n[0] * n[2]),
        q_mag * (n[1] * n[1] - t),
        q_mag * (n[1] * n[2]),
    ]
}

/// A `+1/2` wedge line through the box, cored between voxels.
fn wedge(n: usize, q_mag: f64) -> Vec<[f64; 5]> {
    let c = n as f64 / 2.0 - 0.5;
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let theta = 0.5 * (j as f64 - c).atan2(i as f64 - c);
                q[((i * n) + j) * n + l] = uniaxial([theta.cos(), theta.sin(), 0.0], q_mag);
            }
        }
    }
    q
}

/// A uniformly ordered field with a trace of numerical noise on it.
fn ordered(n: usize, q_mag: f64, noise: f64) -> Vec<[f64; 5]> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut q = vec![[0.0; 5]; n * n * n];
    for k in 0..q.len() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        let d = [1.0, noise * r, noise * r];
        let m = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        q[k] = uniaxial([d[0] / m, d[1] / m, d[2] / m], q_mag);
    }
    q
}

#[test]
fn the_equilibrium_order_is_the_positive_root_of_the_landau_quartic() {
    let p = ActiveNematicParams3D::default_test();
    let q = p.equilibrium_q();
    // 6a + 3b q + 8c q^2 = 0 at the equilibrium.
    let residual = 6.0 * p.a_landau + 3.0 * p.b_landau * q + 8.0 * p.c_landau * q * q;
    assert!(
        residual.abs() < 1e-12,
        "q_eq = {q} left residual {residual}"
    );
    assert!(q > 0.0, "the ordered root should be positive, got {q}");
}

#[test]
fn the_default_floor_scales_as_the_inverse_square_of_the_spacing() {
    // Scaling it is what makes the threshold mean the same thing on any grid.
    // `s` is quadratic in a Q gradient, so halving `dx` quadruples the density
    // at a core, and the floor has to follow or it means something different.
    let mut p = ActiveNematicParams3D::default_test();
    p.dx = 1.0;
    let coarse = p.disclination_floor();
    p.dx = 0.5;
    let fine = p.disclination_floor();
    assert!(
        ((fine / coarse) - 4.0).abs() < 1e-9,
        "halving the spacing moved the floor by {}, expected 4",
        fine / coarse
    );
}

#[test]
fn an_explicit_floor_overrides_the_scaled_default() {
    let mut p = ActiveNematicParams3D::default_test();
    p.disclination_threshold_floor = Some(0.0);
    assert_eq!(
        p.disclination_floor(),
        0.0,
        "an explicit zero should disable the floor"
    );
    p.disclination_threshold_floor = Some(7.5e-3);
    assert_eq!(p.disclination_floor(), 7.5e-3);
}

#[test]
fn the_default_floor_sits_between_a_resolved_core_and_an_ordered_field() {
    // The two states the floor has to separate, measured rather than assumed.
    let p = ActiveNematicParams3D::default_test();
    let q_eq = p.equilibrium_q();
    let floor = p.disclination_floor();
    let n = 32;

    let peak = |field: &[[f64; 5]]| {
        let s = disclination_magnitude(field, n, n, n, p.dx);
        let mut m = 0.0_f64;
        for i in 2..n - 2 {
            for j in 2..n - 2 {
                for l in 2..n - 2 {
                    m = m.max(s[((i * n) + j) * n + l]);
                }
            }
        }
        m
    };

    let core = peak(&wedge(n, q_eq));
    let quiet = peak(&ordered(n, q_eq, 1e-6));

    assert!(
        quiet < floor && floor < core,
        "floor {floor:.3e} should lie between an ordered field's {quiet:.3e} and a core's {core:.3e}"
    );
    // And with room to spare on both sides, so neither a slightly noisier field
    // nor a slightly coarser core crosses it.
    assert!(
        core / floor > 100.0,
        "only {:.1}x of headroom below a core",
        core / floor
    );
    assert!(
        floor / quiet > 100.0,
        "only {:.1}x of headroom above the noise",
        floor / quiet
    );
}

#[test]
fn the_default_floor_empties_an_ordered_field_and_keeps_a_real_line() {
    let p = ActiveNematicParams3D::default_test();
    let q_eq = p.equilibrium_q();
    let floor = p.disclination_floor();
    let n = 32;

    let (quiet_lines, _) =
        disclination_lines_at_fraction(&ordered(n, q_eq, 1e-6), n, n, n, p.dx, 0.25, floor);
    assert!(
        quiet_lines.is_empty(),
        "an ordered field returned {} lines above the default floor",
        quiet_lines.len()
    );

    let (core_lines, _) =
        disclination_lines_at_fraction(&wedge(n, q_eq), n, n, n, p.dx, 0.25, floor);
    assert!(
        !core_lines.is_empty(),
        "the wedge line was lost to the default floor"
    );
}

#[test]
fn the_coefficient_is_the_one_the_calibration_gave() {
    // A resolved core reads 0.647 (q_eq/dx)^2 and an ordered field's noise sat
    // at 1.8e-7 of it, so the coefficient belongs between those two by orders of
    // magnitude. This fails if someone retunes it without redoing the measurement.
    assert!(
        (1e-5..1e-2).contains(&DEFAULT_DISCLINATION_FLOOR_COEFFICIENT),
        "the coefficient left the band the calibration supports"
    );
}
