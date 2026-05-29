# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24"]
# ///
"""Differential cross-check: volterra (Rust/PyO3) vs braid_tracker_v2 (Python).

Feeds identical inputs to both implementations and asserts they agree on:

  1. **Defect detection** from a Q-tensor grid (positions + charges).
  2. **Braid-word extraction** from a defect-position time series.
  3. **Topological entropy** of the extracted braid.

This is the literal "volterra vs the CGPO Python code" gate, decoupled from the
PDE. It must be run in an environment where the ``volterra`` extension is
installed (``maturin develop`` from the workspace root), e.g.::

    cd volterra && maturin develop --release
    uv run --active volterra-braid/oracle/cross_check.py
    # or: .venv/bin/python volterra-braid/oracle/cross_check.py

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import braid_tracker_v2 as v2

try:
    import volterra
except ImportError:
    sys.exit(
        "ERROR: `import volterra` failed. Build the extension first:\n"
        "  cd <workspace> && maturin develop --release"
    )


def winding_grid(nx, ny, cx, cy, plus):
    """Row-major (qxx, qxy, mask) for one +/-1/2 defect at (cx, cy)."""
    qxx = [0.0] * (nx * ny)
    qxy = [0.0] * (nx * ny)
    mask = [True] * (nx * ny)
    for x in range(nx):
        for y in range(ny):
            i = x * ny + y
            if x in (0, nx - 1) or y in (0, ny - 1):
                mask[i] = False
            dx, dy = x - cx, y - cy
            r = math.hypot(dx, dy)
            if r < 0.5:
                continue
            qxx[i] = dx / r
            qxy[i] = (dy / r) if plus else (-dy / r)
    return qxx, qxy, mask


def check_detection():
    nx = ny = 41
    fails = 0
    for plus in (True, False):
        qxx, qxy, mask = winding_grid(nx, ny, 20.0, 20.0, plus)
        rs = volterra.braid_detect_defects(qxx, qxy, nx, ny, 0.5, mask)
        py = v2.detect_defects(qxx, qxy, nx, ny, 0.5, mask)
        rs = sorted((round(x, 6), round(y, 6), c) for x, y, c in rs)
        py = sorted((round(x, 6), round(y, 6), c) for x, y, c in py)
        if rs != py:
            print(f"  detection MISMATCH (plus={plus}):\n    rust={rs}\n    py  ={py}")
            fails += 1
        else:
            print(f"  detection plus={plus}: {len(rs)} defect(s), rust==py OK -> {rs}")
    return fails


def check_words_and_entropy():
    cases = {
        "golden": (3, [-2, 1]),
        "silver": (4, [3, 1, 2, -3, -1, -2]),
        "braid-relation": (3, [1, 2, 1]),
        "mixed": (4, [1, -2, 3, -1, 2]),
        "power": (3, [2, 2, 2]),
    }
    fails = 0
    for name, (n, codes) in cases.items():
        frames = v2.realize_braid(n, codes, frames_per_gen=10, periods=2)
        n_rs, codes_rs = volterra.braid_word_from_frames(frames)
        n_py, codes_py = v2.braidword_from_frames(frames)
        if (n_rs, codes_rs) != (n_py, codes_py):
            print(f"  word MISMATCH ({name}):\n    rust=({n_rs},{codes_rs})\n    py  =({n_py},{codes_py})")
            fails += 1
            continue
        h_rs = volterra.braid_topological_entropy(n, codes)
        h_py = v2.topological_entropy(n, codes)
        if abs(h_rs - h_py) > 1e-12:
            print(f"  entropy MISMATCH ({name}): rust={h_rs} py={h_py}")
            fails += 1
            continue
        print(f"  {name}: word rust==py {codes_rs}, entropy {h_rs:.6f} (rust==py) OK")
    return fails


def main():
    print("Detection cross-check (Q grid -> defects):")
    fails = check_detection()
    print("Braid-word + entropy cross-check (frames -> word -> h):")
    fails += check_words_and_entropy()
    if fails:
        print(f"\nFAILED: {fails} mismatch(es) between volterra and braid_tracker_v2")
        sys.exit(1)
    print("\nPASS: volterra (Rust) and braid_tracker_v2 (Python) agree on all inputs")


if __name__ == "__main__":
    main()
