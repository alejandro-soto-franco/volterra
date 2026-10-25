# Verified grid index maps

Every field type in volterra-core indexes its grid through
`volterra-core/src/fields/index.rs`. Charon and Aeneas translate that file to Lean 4,
and `lean/IndexProofs.lean` proves the following for every grid whose vertex count fits
in `usize`, on 32-bit and 64-bit targets:

- `wrap_signed` returns the Euclidean residue `i % n` and never panics.
- `periodic_idx2`, `periodic_idx2_signed` and `periodic_idx3` never overflow, return the
  row-major closed form and stay below the vertex count. The signed index depends on each
  coordinate only through its residue and has period `nx` in `i` and `ny` in `j`.
- `coords3` and `periodic_idx3` are inverse: `periodic_idx3_of_coords3` and
  `coords3_of_periodic_idx3`.

Each theorem depends on Lean's three standard axioms alone. `index_core/` compiles the
shipped `index.rs` through a `#[path]` include. The proofs therefore cover the code the
simulations run. Five planted bugs in `index.rs`, one per function, each break a proof.

## Running

`./run.sh` runs the Rust tests, extracts with Charon, translates with Aeneas, compares
the result with the committed Lean in `lean/IndexCore/`, builds the proofs and checks
their axioms. It fails on a test failure, on drift between the Rust and the committed
translation and on any axiom beyond the standard three.

Tools: the Aeneas nightly `nightly-2026.09.24-557f7a1` at `~/local/aeneas` (override
with `AENEAS_HOME`), which bundles the Charon its Lean backend expects. Charon needs the
Rust toolchain `nightly-2026-09-17`. The Lean project uses Lean and Mathlib v4.31.0 and
takes Aeneas from git at the same commit.
