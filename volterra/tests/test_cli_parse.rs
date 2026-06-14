use volterra::cli::{Cli, Command, RunTarget};
use clap::Parser;

#[test]
fn parses_cartesian2d_dry_with_common_flags() {
    let cli = Cli::try_parse_from([
        "volterra", "run", "cartesian2d", "--mode", "dry",
        "--steps", "10", "--snap-every", "2", "--seed", "7", "--out", "/tmp/out",
    ])
    .expect("parse");
    match cli.command {
        Command::Run(RunTarget::Cartesian2d(a)) => {
            assert_eq!(a.common.steps, 10);
            assert_eq!(a.common.snap_every, 2);
            assert_eq!(a.common.seed, 7);
            assert_eq!(a.mode, "dry");
        }
        _ => panic!("wrong subcommand"),
    }
}

#[test]
fn defaults_out_dir_per_subcommand() {
    let cli = Cli::try_parse_from(["volterra", "run", "cgpo"]).expect("parse");
    if let Command::Run(RunTarget::Cgpo(a)) = cli.command {
        assert_eq!(
            a.common.out_or_default("cgpo"),
            std::path::PathBuf::from("./output/cgpo")
        );
    } else {
        panic!("expected cgpo");
    }
}
