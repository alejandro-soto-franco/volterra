use std::process::Command;

#[test]
fn run_cartesian3d_dry_exits_zero_and_writes_output() {
    let tmp = std::env::temp_dir().join("vcli_3d");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_volterra"))
        .args([
            "run", "cartesian3d",
            "--mode", "dry",
            "--nx", "8",
            "--ny", "8",
            "--nz", "8",
            "--steps", "4",
            "--snap-every", "2",
            "--out", tmp.to_str().unwrap(),
        ])
        .status()
        .expect("spawn volterra");
    assert!(status.success(), "volterra run cartesian3d exited non-zero: {status}");
    // The 3D runner writes q_*.npy frames and stats.json into out_dir.
    assert!(tmp.join("stats.json").exists(), "stats.json not written");
    let npy_count = std::fs::read_dir(&tmp)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("q_")
        })
        .count();
    assert!(npy_count >= 1, "no q_*.npy files written into {}", tmp.display());
}
