use volterra_cgpo::CgpoError;

#[test]
fn io_error_carries_path_context() {
    let e = CgpoError::Io {
        path: "/nope/frame.txt".into(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
    };
    let msg = e.to_string();
    assert!(msg.contains("/nope/frame.txt"), "error must name the path: {msg}");
}

#[test]
fn non_finite_error_names_step() {
    let e = CgpoError::NonFinite { step: 42, field: "u" };
    assert!(e.to_string().contains("42"));
    assert!(e.to_string().contains('u'));
}
