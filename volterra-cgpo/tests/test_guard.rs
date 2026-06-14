use volterra_cgpo::guard::{check_cfl, check_finite};

#[test]
fn cfl_passes_when_dt_is_safe() {
    // u packed as [x0_c0, x0_c1, x1_c0, ...]; max|u| = sqrt(3^2+4^2) = 5.
    let u = vec![3.0, 4.0, 0.0, 0.0];
    // dx = 1.0, safety 0.5 -> safe dt = 0.5 * 1 / 5 = 0.1
    assert!(check_cfl(&u, 0.05, 1.0, 1.0, 0.5, 0).is_ok());
}

#[test]
fn cfl_fails_when_dt_too_large() {
    let u = vec![3.0, 4.0, 0.0, 0.0];
    let err = check_cfl(&u, 0.2, 1.0, 1.0, 0.5, 7).unwrap_err();
    assert!(err.to_string().contains("step 7"));
}

#[test]
fn finite_passes_on_clean_field() {
    assert!(check_finite(&[1.0, -2.0, 3.5], "Q", 0).is_ok());
}

#[test]
fn finite_fails_on_nan_naming_step_and_field() {
    let err = check_finite(&[1.0, f64::NAN], "u", 99).unwrap_err();
    let m = err.to_string();
    assert!(m.contains("99") && m.contains('u'));
}
