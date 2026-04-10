use volterra_solver::nematic_field_2d::NematicField2D;
use volterra_dec::QFieldDec;

#[test]
fn test_nematic_field_2d_zeros() {
    let f = NematicField2D::zeros(10);
    assert_eq!(f.n_vertices(), 10);
    assert!(f.mean_scalar_order() == 0.0);
}

#[test]
fn test_nematic_field_2d_scalar_order() {
    let f = NematicField2D::uniform(5, 0.3, 0.4);
    // |z| = sqrt(0.09 + 0.16) = 0.5, S = 2*0.5 = 1.0
    assert!((f.mean_scalar_order() - 1.0).abs() < 1e-14);
}

#[test]
fn test_nematic_field_2d_qfield_roundtrip() {
    let original = NematicField2D::uniform(5, 0.3, 0.4);
    let qfield = original.to_qfield_dec();
    assert!((qfield.q1[0] - 0.3).abs() < 1e-14);
    assert!((qfield.q2[0] - 0.4).abs() < 1e-14);

    let recovered = NematicField2D::from_qfield_dec(&qfield);
    let (q1, q2) = recovered.section.to_real_components();
    assert!((q1[0] - 0.3).abs() < 1e-14);
    assert!((q2[0] - 0.4).abs() < 1e-14);
}

#[test]
fn test_nematic_field_2d_trace_q_squared() {
    let f = NematicField2D::uniform(3, 0.3, 0.4);
    let tr = f.trace_q_squared();
    let expected = 2.0 * (0.3_f64.powi(2) + 0.4_f64.powi(2));
    assert!((tr[0] - expected).abs() < 1e-14);
}

#[test]
fn test_nematic_field_2d_normalise() {
    let mut f = NematicField2D::uniform(3, 3.0, 4.0);
    // |z| = 5
    assert!((f.values()[0].norm() - 5.0).abs() < 1e-14);
    f.normalise();
    assert!((f.values()[0].norm() - 1.0).abs() < 1e-14);
}
