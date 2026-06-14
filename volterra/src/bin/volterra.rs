use clap::Parser;
use volterra::cli::Cli;

fn main() {
    let cli = Cli::parse();
    if let Err(e) = volterra::cli::dispatch(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
