use std::process::Command;

#[test]
fn run_cartesian2d_dry_exits_zero() {
    let tmp = std::env::temp_dir().join("vcli_2d");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_volterra"))
        .args([
            "run", "cartesian2d",
            "--mode", "dry",
            "--nx", "16",
            "--ny", "16",
            "--steps", "4",
            "--snap-every", "2",
            "--out", tmp.to_str().unwrap(),
        ])
        .status()
        .expect("spawn volterra");
    assert!(status.success(), "volterra run cartesian2d exited non-zero: {status}");
    assert!(tmp.join("stats.json").exists(), "stats.json not written");
}
