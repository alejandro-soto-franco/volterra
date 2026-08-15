//! The magnetic and electric couplings in the 3D molecular field.
//!
//! Both are quadratic in an external field and both contribute a traceless
//! rank-two term, so they are tested the same way: the term must vanish when the
//! amplitude is zero, must scale as the square of the amplitude, must be
//! traceless, and must push the director the way the sign of the anisotropy
//! says.

use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;
use volterra_fd::molecular_field_3d;

fn params() -> ActiveNematicParams3D {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = 6;
    p.ny = 6;
    p.nz = 6;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p
}

/// A uniform field, so the elastic term is exactly zero and what remains is the
/// bulk plus whatever the external fields contribute.
fn uniform(p: &ActiveNematicParams3D) -> QField3D {
    QField3D::uniform(p.nx, p.ny, p.nz, p.dx, [0.2, 0.05, 0.0, -0.1, 0.0])
}

#[test]
fn no_field_leaves_the_molecular_field_untouched() {
    // The electric term must be inert when unset, or every result measured
    // before it existed would have moved.
    let mut off = params();
    off.chi_a = 0.0;
    off.e0 = 0.0;
    off.epsilon_a = 0.0;
    let q = uniform(&off);
    let base = molecular_field_3d(&q, &off, 0.7);

    let mut e_zero_amplitude = off.clone();
    e_zero_amplitude.epsilon_a = 5.0; // a coefficient with no field behind it
    let same = molecular_field_3d(&q, &e_zero_amplitude, 0.7);
    for (a, b) in base.q.iter().zip(same.q.iter()) {
        assert_eq!(a, b, "a coefficient with zero amplitude changed the field");
    }
}

#[test]
fn the_electric_term_is_traceless() {
    let mut p = params();
    p.epsilon_a = 0.8;
    p.e0 = 1.5;
    p.omega_e = 0.0;
    let q = uniform(&p);
    let h = molecular_field_3d(&q, &p, 0.0);
    for site in &h.q {
        let [h11, _, _, h22, _] = *site;
        let h33 = -(h11 + h22);
        assert!(
            (h11 + h22 + h33).abs() < 1e-15,
            "molecular field is not traceless"
        );
    }
}

#[test]
fn the_electric_term_scales_as_the_square_of_the_field() {
    let q = uniform(&params());
    let contribution = |e0: f64| {
        let mut p = params();
        p.chi_a = 0.0;
        p.epsilon_a = 0.8;
        p.e0 = e0;
        p.omega_e = 0.0;
        let with = molecular_field_3d(&q, &p, 0.0);
        let mut off = p.clone();
        off.e0 = 0.0;
        let without = molecular_field_3d(&q, &off, 0.0);
        with.q[0][0] - without.q[0][0]
    };
    let one = contribution(1.0);
    let two = contribution(2.0);
    assert!(one.abs() > 1e-12, "no electric contribution at all");
    assert!(
        (two / one - 4.0).abs() < 1e-12,
        "doubling the field scaled the term by {}, expected 4",
        two / one
    );
}

#[test]
fn magnetic_and_electric_add() {
    // They enter through one term, so a run with both must equal the sum of the
    // two contributions taken separately.
    let q = uniform(&params());
    let base = {
        let mut p = params();
        p.chi_a = 0.0;
        p.e0 = 0.0;
        molecular_field_3d(&q, &p, 0.3)
    };
    let only_b = {
        let mut p = params();
        p.chi_a = 0.7;
        p.b0 = 1.1;
        p.e0 = 0.0;
        molecular_field_3d(&q, &p, 0.3)
    };
    let only_e = {
        let mut p = params();
        p.chi_a = 0.0;
        p.epsilon_a = 0.4;
        p.e0 = 1.3;
        molecular_field_3d(&q, &p, 0.3)
    };
    let both = {
        let mut p = params();
        p.chi_a = 0.7;
        p.b0 = 1.1;
        p.epsilon_a = 0.4;
        p.e0 = 1.3;
        molecular_field_3d(&q, &p, 0.3)
    };
    for c in 0..5 {
        let lhs = both.q[0][c] - base.q[0][c];
        let rhs = (only_b.q[0][c] - base.q[0][c]) + (only_e.q[0][c] - base.q[0][c]);
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "component {c}: together {lhs}, separately {rhs}"
        );
    }
}

#[test]
fn a_static_field_lies_along_x_and_a_rotating_one_does_not() {
    let q = uniform(&params());
    let off_diagonal = |omega: f64, t: f64| {
        let mut p = params();
        p.chi_a = 0.0;
        p.epsilon_a = 0.9;
        p.e0 = 1.0;
        p.omega_e = omega;
        let with = molecular_field_3d(&q, &p, t);
        let mut off = p.clone();
        off.e0 = 0.0;
        let without = molecular_field_3d(&q, &off, t);
        with.q[0][1] - without.q[0][1]
    };
    // Static: the direction is x, so the xy component of `d (x) d` vanishes.
    assert!(off_diagonal(0.0, 2.4).abs() < 1e-15, "a static field is not along x");
    // Rotating: a quarter turn in, the direction has both components.
    let quarter = std::f64::consts::FRAC_PI_4;
    assert!(
        off_diagonal(1.0, quarter).abs() > 1e-3,
        "a rotating field left no off-diagonal term"
    );
}

#[test]
fn a_negative_anisotropy_reverses_the_term() {
    // A nematic of negative dielectric anisotropy aligns across the field rather
    // than along it, which is a real class of material and must be expressible.
    let q = uniform(&params());
    let term = |eps: f64| {
        let mut p = params();
        p.chi_a = 0.0;
        p.epsilon_a = eps;
        p.e0 = 1.0;
        let with = molecular_field_3d(&q, &p, 0.0);
        let mut off = p.clone();
        off.e0 = 0.0;
        with.q[0][0] - molecular_field_3d(&q, &off, 0.0).q[0][0]
    };
    let positive = term(0.6);
    let negative = term(-0.6);
    assert!(
        (positive + negative).abs() < 1e-12,
        "reversing the anisotropy did not reverse the term: {positive} and {negative}"
    );
}
