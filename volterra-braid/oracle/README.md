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
- `compare_cgpo.py` — the end-to-end comparison against the **unmodified**
  published `braid_tracker.py`. Renders Q-tensor grids for the golden/silver
  braiding configurations, runs the real script via uv, and checks volterra and
  the script extract the same braid word and topological entropy.

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

### Against the unmodified published script

```bash
cd <workspace root>            # ~/volterra
maturin develop --release
.venv/bin/python volterra-braid/oracle/compare_cgpo.py            # golden + silver
.venv/bin/python volterra-braid/oracle/compare_cgpo.py --no-real  # volterra side only (fast)
```

`compare_cgpo.py` runs `Chaos-Generating-Periodic-Orbits/braid_tracker.py`
verbatim via `uv run --with "matplotlib==3.7.3"` (pinned `<3.8` because the
script uses `ax.w_xaxis`, removed in matplotlib 3.8). It renders well-separated,
tie-free 2D defect orbits realising each braid, feeds the identical Q-tensor
grids to both, and confirms agreement. Result: the published script and volterra
extract the **identical** braid word for both configurations, both with the
analytic entropy (golden `2 log phi`, silver `log(3 + 2 sqrt 2)`).

Two practical notes for reproducing this, learned the hard way:

- The braid axis must match. `braid_tracker.py` orders the braid by the grid
  **y** coordinate (it stores `X<-y, Y<-x`); volterra orders by its first
  coordinate. Feed volterra the swapped `(y, x)` coordinates so both read the
  same axis.
- `braid_tracker.py` writes inverse generators as `sigma_K^-1` (its f-string
  `^{-1}` evaluates the field `{-1}` to the integer `-1`), **not** `sigma_K^{-1}`.
  Parse accordingly.

## Analytic targets

- golden braid `{sigma_2^-1 sigma_1}` (3 strands): `h = 2 log phi ≈ 0.96242`
- silver braid `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}`
  (4 strands): `h = log(3 + 2 sqrt 2) ≈ 1.76275`
