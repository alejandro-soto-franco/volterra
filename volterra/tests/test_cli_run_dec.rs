use std::process::Command;

#[test]
fn run_dec_sphere_dry_exits_zero_and_writes_stats() {
    let tmp = std::env::temp_dir().join("vcli_dec");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_volterra"))
        .args([
            "run", "dec",
            "--mesh", "sphere",
            "--mode", "dry",
            "--steps", "4",
            "--snap-every", "2",
            "--out", tmp.to_str().unwrap(),
        ])
        .status()
        .expect("spawn volterra");
    assert!(status.success(), "volterra run dec exited non-zero: {status}");
    assert!(tmp.join("stats.json").exists(), "stats.json not written");
}
