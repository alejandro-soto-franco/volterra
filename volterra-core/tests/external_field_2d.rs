//! The external-field term in the two-dimensional molecular field.
//!
//! The three-dimensional lane has had this since `test_external_fields.rs`, and
//! these are the same four properties restated for the planar packing: inert at
//! zero amplitude, quadratic in the amplitude, of a magnitude the drive phase
//! does not change, and signed so the director goes where the anisotropy says.
//!
//! The planar packing stores `(Q_xx, Q_xy)` with `Q_yy = -Q_xx`, so
//! tracelessness is structural here rather than something to assert.

use volterra_core::ActiveNematicParams;

fn params() -> ActiveNematicParams {
    let mut p = ActiveNematicParams::default_test();
    p.chi_a = 0.0;
    p.b0 = 0.0;
    p.omega_b = 0.0;
    p
}

#[test]
fn a_coefficient_with_no_field_behind_it_changes_nothing() {
    let off = params();
    let mut coefficient_only = off.clone();
    coefficient_only.chi_a = 5.0;
    for t in [0.0, 0.3, 7.5] {
        assert_eq!(
            off.field_term(t),
            coefficient_only.field_term(t),
            "a coefficient with zero amplitude moved the molecular field"
        );
    }
}

#[test]
fn an_amplitude_with_no_coefficient_behind_it_changes_nothing() {
    let off = params();
    let mut amplitude_only = off.clone();
    amplitude_only.b0 = 3.0;
    assert_eq!(off.field_term(1.1), amplitude_only.field_term(1.1));
}

#[test]
fn the_term_is_quadratic_in_the_amplitude() {
    let mut p = params();
    p.chi_a = 0.7;
    p.b0 = 1.0;
    let one = p.field_term(0.0);
    p.b0 = 3.0;
    let three = p.field_term(0.0);
    assert!(
        (three[0] - 9.0 * one[0]).abs() < 1e-12 && (three[1] - 9.0 * one[1]).abs() < 1e-12,
        "trebling the amplitude must multiply the term by nine: {one:?} against {three:?}"
    );
}

#[test]
fn a_rotating_drive_turns_the_term_without_changing_its_size() {
    let mut p = params();
    p.chi_a = 0.7;
    p.b0 = 1.3;
    p.omega_b = 2.0;
    let size = 0.5 * p.chi_a * p.b0 * p.b0;
    for t in [0.0, 0.21, 1.0, 4.4] {
        let [a, b] = p.field_term(t);
        assert!(
            (a.hypot(b) - size).abs() < 1e-12,
            "the drive phase changed the size of the term at t = {t}"
        );
    }
}

#[test]
fn a_static_drive_points_the_term_along_x() {
    // omega_b = 0 leaves the field along x, and a director along x is
    // (Q_xx, Q_xy) = (+S/2, 0), so a positive anisotropy gives a positive
    // first component and a vanishing second.
    let mut p = params();
    p.chi_a = 0.9;
    p.b0 = 1.0;
    p.omega_b = 0.0;
    let [a, b] = p.field_term(3.7);
    assert!(
        a > 0.0,
        "a positive anisotropy must drive the director to the field axis"
    );
    assert!(
        b.abs() < 1e-15,
        "a field along x contributes no shear component"
    );
}

#[test]
fn the_phase_advances_at_twice_the_drive_rate() {
    // Q carries the spin-2 representation, so a field turned by a right angle
    // returns a term of the opposite sign rather than an orthogonal one.
    let mut p = params();
    p.chi_a = 0.9;
    p.b0 = 1.0;
    p.omega_b = 1.0;
    let at_zero = p.field_term(0.0);
    let quarter = p.field_term(std::f64::consts::FRAC_PI_2);
    assert!(
        (quarter[0] + at_zero[0]).abs() < 1e-12 && (quarter[1] + at_zero[1]).abs() < 1e-12,
        "a quarter turn of the field must negate the term: {at_zero:?} then {quarter:?}"
    );
}
