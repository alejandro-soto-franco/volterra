//! Smoke test for `volterra run cgpo`.
//!
//! Spawns the binary on a tiny 16x16 grid for 2 steps with snap_every=1 and
//! asserts exit 0 plus the expected CGPO frame output structure.

use std::process::Command;

#[test]
fn run_cgpo_exits_zero_and_writes_frame_files() {
    let tmp = std::env::temp_dir().join("vcli_cgpo_smoke");
    let _ = std::fs::remove_dir_all(&tmp);

    let status = Command::new(env!("CARGO_BIN_EXE_volterra"))
        .args([
            "run", "cgpo",
            "--lx", "16",
            "--als", "1.5",
            "--ncl", "1",
            "--steps", "2",
            "--snap-every", "1",
            "--out", tmp.to_str().unwrap(),
        ])
        .status()
        .expect("spawn volterra");

    assert!(status.success(), "volterra run cgpo exited non-zero: {status}");

    // The run dir is <out>/als_1.5_ncl_1/
    let run_dir = tmp.join("als_1.5_ncl_1");

    // Q frames at steps 0, 1, 2
    assert!(
        run_dir.join("Q").join("Q_0000000000.txt").exists(),
        "Q frame at step 0 missing"
    );
    assert!(
        run_dir.join("Q").join("Q_0000000001.txt").exists(),
        "Q frame at step 1 missing"
    );
    assert!(
        run_dir.join("Q").join("Q_0000000002.txt").exists(),
        "Q frame at step 2 missing"
    );

    // u frames
    assert!(
        run_dir.join("u").join("u_0000000000.txt").exists(),
        "u frame at step 0 missing"
    );
    assert!(
        run_dir.join("u").join("u_0000000002.txt").exists(),
        "u frame at step 2 missing"
    );

    // p frames (gauge-fixed)
    assert!(
        run_dir.join("p").join("p_0000000000.txt").exists(),
        "p frame at step 0 missing"
    );
    assert!(
        run_dir.join("p").join("p_0000000002.txt").exists(),
        "p frame at step 2 missing"
    );
}
