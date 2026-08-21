use std::process::Command;

#[test]
fn fd_writes_into_portable_default_when_no_env() {
    let tmp = std::env::temp_dir().join("fd_smoke");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_fd"))
        .current_dir(std::env::temp_dir())
        .env("FD_OUT", tmp.to_str().unwrap())
        .env("FD_LX", "16")
        .env("FD_MAX_STEPS", "2")
        .env("FD_SAVE_EVERY", "1")
        .env("FD_SEED", "42")
        .status()
        .expect("spawn fd");
    assert!(status.success(), "fd must exit 0");
    assert!(tmp.exists(), "output dir must be created");
}
