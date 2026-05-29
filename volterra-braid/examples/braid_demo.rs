//! Demo: extract braid words from synthetic golden/silver orbits and report the
//! topological entropy, checking against the analytic values.
//!
//! Run with: `cargo run --release --example braid_demo -p volterra-braid`

use volterra_braid::{BraidWord, GOLDEN_H, RealizeOpts, SILVER_H, golden_orbit, silver_orbit};

fn main() {
    let opts = RealizeOpts {
        frames_per_gen: 12,
        periods: 3,
    };

    let cases = [
        ("golden (cardioid)", golden_orbit(&opts), GOLDEN_H),
        ("silver (nephroid)", silver_orbit(&opts), SILVER_H),
    ];

    for (name, frames, expected_h) in cases {
        let word = BraidWord::from_frames(&frames);
        let period_codes: Vec<i32> = word.fundamental_period().iter().map(|g| g.code()).collect();
        let period = BraidWord::from_codes(word.n_strands, &period_codes);

        println!("{name}:");
        println!("  frames        {}", frames.len());
        println!("  braid (1 period) {period}");
        println!("  codes         {period_codes:?}");
        println!("  permutation   {:?}", period.permutation());
        println!("  exponent sum  {}", period.exponent_sum());
        println!(
            "  entropy       {:.6}  (analytic {:.6}, |diff| {:.2e})",
            period.topological_entropy(),
            expected_h,
            (period.topological_entropy() - expected_h).abs()
        );
    }
}
