# volterra-braid

Braid-group analysis of active-nematic defect trajectories: detection, tracking,
braid-word extraction, and topological entropy.

This crate is the analysis layer for the "chaos-generating periodic orbits" work
(Klein, Soto Franco et al. 2026, [arXiv:2503.10880](https://arxiv.org/abs/2503.10880)).
It is **decoupled from the PDE solver**: the input is geometry (defect positions,
or a Q-tensor grid), and the output is a braid word and its topological entropy.
So the braid algebra is fast, deterministic, and testable without running the
(expensive) confined active-nematic simulation.

## Pipeline

```text
Q grid --detect_defects--> defects --track--> worldlines --extract_braidword--> BraidWord
                                                                                    |
                                                                  topological_entropy --> h
```

- `detect_defects` -- defect density `ss = (2 dxQxy)(2 dyQxx) - (2 dxQxx)(2 dyQxy)`,
  threshold, 8-connected flood fill, centroid; matches the reference scheme.
- `track` -- greedy nearest-neighbour with removal (per-frame bijection).
- `extract_braidword` -- sort defects by x; an adjacent swap at gap `k` emits
  `sigma_{k+1}` (1-based), with sign from the y-ordering at the crossing.
- `topological_entropy` -- `log` of the dilatation, from the Burau matrix spectral
  radius at `t = -1`. Exact for the orbits of interest; a lower bound in general.

## Example

```rust
use volterra_braid::{golden_orbit, BraidWord, RealizeOpts, GOLDEN_H};

let frames = golden_orbit(&RealizeOpts { frames_per_gen: 12, periods: 3 });
let word = BraidWord::from_frames(&frames);
assert_eq!(word.fundamental_period().iter().map(|g| g.code()).collect::<Vec<_>>(), vec![-2, 1]);
assert!((BraidWord::from_codes(3, &[-2, 1]).topological_entropy() - GOLDEN_H).abs() < 1e-9);
```

Run the demo: `cargo run --release --example braid_demo -p volterra-braid`.

## Analytic gates

- **golden braid** (cardioid) `{sigma_2^-1 sigma_1}` on 3 strands:
  `h = 2 log phi ≈ 0.96242` (dilatation `phi^2`).
- **silver braid** (nephroid)
  `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` on 4 strands:
  `h = log(3 + 2 sqrt 2) ≈ 1.76275` (dilatation `(1 + sqrt 2)^2`).

`tests/golden.rs` and `tests/silver.rs` verify the synthetic orbits round-trip to
these words and entropies.

## Python

The `volterra` PyO3 extension exposes `BraidWord` plus the plain-list functions
`braid_detect_defects` / `braid_word_from_frames` / `braid_topological_entropy`.
The `oracle/` directory holds `braid_tracker_v2.py` (a cleaned reimplementation of
the published CGPO tracker) and `cross_check.py` (a Rust-vs-Python differential
gate). See `oracle/README.md`.

## License

MIT.
