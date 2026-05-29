# volterra-braid Python oracle

Differential validation of the Rust `volterra-braid` crate against a cleaned
Python reimplementation of the CGPO reference braid tracker.

## Files

- `braid_tracker_v2.py` — cleaned, documented reimplementation of
  `Chaos-Generating-Periodic-Orbits/braid_tracker.py` (defect detection,
  tracking, braid-word extraction, topological entropy). The in-file docstring
  lists every deliberate deviation from the original. Library + self-test.
- `cross_check.py` — feeds identical inputs to volterra (Rust, via the
  `volterra` PyO3 extension) and to `braid_tracker_v2`, and asserts they agree on
  defect detection, braid words, and topological entropy.

## Running

The v2 reference is self-contained (PEP 723 inline deps); run its self-test with
uv, no setup:

```bash
uv run braid_tracker_v2.py
```

The cross-check imports the compiled `volterra` extension, so build it first:

```bash
cd <workspace root>            # ~/volterra
maturin develop --release      # builds + installs `volterra` into .venv
.venv/bin/python volterra-braid/oracle/cross_check.py
# or, with the venv active:  uv run --active volterra-braid/oracle/cross_check.py
```

A passing cross-check means the Rust extraction and the Python reference produce
identical braid words and entropies on the same data — the analytic braid-group
gate, decoupled from the active-nematic PDE solve.

## Analytic targets

- golden braid `{sigma_2^-1 sigma_1}` (3 strands): `h = 2 log phi ≈ 0.96242`
- silver braid `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}`
  (4 strands): `h = log(3 + 2 sqrt 2) ≈ 1.76275`
