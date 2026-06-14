use std::process::Command;

#[test]
fn cgpo_fd_writes_into_portable_default_when_no_env() {
    let tmp = std::env::temp_dir().join("cgpo_fd_smoke");
    let _ = std::fs::remove_dir_all(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_cgpo_fd"))
        .current_dir(std::env::temp_dir())
        .env("CGPO_OUT", tmp.to_str().unwrap())
        .env("CGPO_LX", "16")
        .env("CGPO_MAX_STEPS", "2")
        .env("CGPO_SAVE_EVERY", "1")
        .env("CGPO_SEED", "42")
        .status()
        .expect("spawn cgpo_fd");
    assert!(status.success(), "cgpo_fd must exit 0");
    assert!(tmp.exists(), "output dir must be created");
}
