use volterra_cgpo::Params;

#[test]
fn params_roundtrips_through_json() {
    let p = Params::new(32, 2.8, 4.8, 1.0, 0.001, -1);
    let json = serde_json::to_string(&p).expect("serialize");
    let back: Params = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(p.lx, back.lx);
    assert_eq!(p.ly, back.ly);
    assert!((p.dt - back.dt).abs() < 1e-15);
    assert_eq!(p.max_p_iters, back.max_p_iters);
}
