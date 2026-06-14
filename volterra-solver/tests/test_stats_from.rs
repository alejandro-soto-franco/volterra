use volterra_core::sim::stats::StepStats;
use volterra_solver::SnapStats;

#[test]
fn snapstats_from_stepstats_fills_shared_fields() {
    let s = StepStats::default()
        .with_time(2.0)
        .with_order_param(0.7)
        .with_defect_count(4);
    let snap = SnapStats::from(s);
    assert_eq!(snap.time, 2.0);
    assert_eq!(snap.mean_s, 0.7);
    assert_eq!(snap.n_defects, 4);
}
