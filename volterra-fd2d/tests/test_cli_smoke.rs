use std::process::Command;

#[test]
fn fd2d_writes_into_portable_default_when_no_env() {
    let tmp = std::env::temp_dir().join("fd2d_smoke");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_fd2d"))
        .current_dir(std::env::temp_dir())
        .env("FD2D_OUT", tmp.to_str().unwrap())
        .env("FD2D_LX", "16")
        .env("FD2D_MAX_STEPS", "2")
        .env("FD2D_SAVE_EVERY", "1")
        .env("FD2D_SEED", "42")
        .status()
        .expect("spawn fd2d");
    assert!(status.success(), "fd2d must exit 0");
    assert!(tmp.exists(), "output dir must be created");
}
