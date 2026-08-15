#!/usr/bin/env python3
"""Extract the braid of the mobile +1/2 defects from an `fd` run directory.

Generalises `extract_braid.py` from the steady-winding circle to any
confinement the `fd` driver can build. Two things change with the geometry:

- The interior mask is read from the run's own `mask.txt` rather than
  recomputed. An epitrochoid's interior test is a root solve per cell, so
  recomputing it here would mean porting that solve into Python.
- Defects are split by charge, and only the positive ones enter the braid. On
  the circle every defect in the braiding regime is a +1/2, so the split is a
  no-op there. On an epitrochoid each regularised cusp pins a -1/2 defect,
  which is stationary, sits against the boundary, and is not a braid strand.

Usage:
    python analyse_run.py <run_dir> [threshold] [--min-window N] [--winding]

`run_dir` is the directory holding `Q/`, `mask.txt` and `meta.json`.

`--winding` swaps the published Jacobian-and-threshold detector for plaquette
winding of the director, which carries no threshold. A threshold calibrated at
one coherence length is wrong at another, and every epitrochoid point in
arXiv:2503.10880 Fig. 7 sits an order of magnitude in coherence length away from
the circle runs the published threshold of 0.1 was set for. This implementation
is deliberately independent of `volterra_braid::detect_defects_winding`, so
running both on the same frames checks one against the other.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import braid_tracker_v2 as v2  # noqa: E402


def load_frame(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def detect_defects_winding(qxx, qxy, nx, ny, mask):
    """Defects by the winding of the director around each lattice plaquette.

    Returns (x, y, charge) triples with charge +/-1, in the same convention as
    ``braid_tracker_v2.detect_defects``. Vectorised over plaquettes, so it is
    not a transcription of the Rust and disagreement means one of them is wrong.
    """
    qxx = np.asarray(qxx, float).reshape(nx, ny)
    qxy = np.asarray(qxy, float).reshape(nx, ny)
    mask = np.asarray(mask, bool).reshape(nx, ny)

    phi = 0.5 * np.arctan2(qxy, qxx)

    def wrap_half(d):
        # Into (-pi/2, pi/2]: the director is a line, so a corner-to-corner turn
        # past a right angle reads as the shorter turn the other way.
        return -((-d + np.pi / 2) % np.pi - np.pi / 2)

    corners = [phi[:-1, :-1], phi[1:, :-1], phi[1:, 1:], phi[:-1, 1:]]
    total = sum(wrap_half(corners[(i + 1) % 4] - corners[i]) for i in range(4))
    inside = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    charge = np.where(inside, np.rint(total / np.pi), 0.0).astype(int)

    # 8-connected same-sign clusters, centroid of plaquette centres.
    visited = np.zeros_like(charge, bool)
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    px, py = charge.shape
    for x in range(px):
        for y in range(py):
            if charge[x, y] == 0 or visited[x, y]:
                continue
            sign = int(np.sign(charge[x, y]))
            stack = [(x, y)]
            visited[x, y] = True
            sx = sy = 0.0
            count = 0
            while stack:
                cx, cy = stack.pop()
                sx += cx + 0.5
                sy += cy + 0.5
                count += 1
                for dx, dy in neigh:
                    a, b = cx + dx, cy + dy
                    if not (0 <= a < px and 0 <= b < py):
                        continue
                    if not visited[a, b] and charge[a, b] != 0 and np.sign(charge[a, b]) == sign:
                        visited[a, b] = True
                        stack.append((a, b))
            out.append((sx / count, sy / count, sign))
    return out


def trailing_runs(counts):
    """Maximal runs of a constant count, latest first."""
    runs = []
    i = len(counts) - 1
    while i >= 0:
        j = i
        while j - 1 >= 0 and counts[j - 1] == counts[i]:
            j -= 1
        runs.append((j, i, counts[i]))
        i = j - 1
    return runs


def fmt(codes):
    return " ".join(
        (f"sigma_{abs(g)}^-1" if g < 0 else f"sigma_{g}") for g in codes
    )


def main():
    run_dir = Path(sys.argv[1])
    threshold = 0.1
    min_window = 8
    rest = sys.argv[2:]
    if rest and not rest[0].startswith("--"):
        threshold = float(rest[0])
        rest = rest[1:]
    if "--min-window" in rest:
        min_window = int(rest[rest.index("--min-window") + 1])
    winding = "--winding" in rest

    meta = json.loads((run_dir / "meta.json").read_text())
    lx, ly = meta["lx"], meta["ly"]

    mask_path = run_dir / "mask.txt"
    if not mask_path.exists():
        print(f"no mask.txt under {run_dir}; the run predates mask output")
        sys.exit(2)
    mask = np.loadtxt(mask_path).astype(bool)
    if mask.size != lx * ly:
        print(f"mask has {mask.size} cells, grid is {lx}x{ly}")
        sys.exit(2)

    files = sorted(glob.glob(str(run_dir / "Q" / "Q_*.txt")))
    if not files:
        print(f"no Q frames under {run_dir}/Q")
        sys.exit(2)

    positive, negative = [], []
    for f in files:
        qxx, qxy = load_frame(f)
        if winding:
            found = detect_defects_winding(qxx, qxy, lx, ly, mask)
        else:
            found = v2.detect_defects(qxx, qxy, lx, ly, threshold, mask)
        positive.append([d for d in found if d[2] > 0])
        negative.append([d for d in found if d[2] < 0])

    pos_counts = [len(d) for d in positive]
    neg_counts = [len(d) for d in negative]
    print(f"{len(files)} frames over {meta['n_steps']} steps, "
          f"{meta['lx']}x{meta['ly']}, als={meta['als']:.4f} ncl={meta['ncl']:.4f}")
    print(f"  +1/2 counts: {pos_counts}")
    print(f"  -1/2 counts: {neg_counts}")

    # Where the pinned negatives sit, averaged over the trailing half of the
    # run. A cusp-pinned defect barely moves, so a large spread means it is not
    # pinned.
    tail = negative[len(negative) // 2:]
    if any(tail):
        pts = np.array([[d[0], d[1]] for frame in tail for d in frame])
        centre = (lx / 2 - 1, ly / 2 - 1)
        radii = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
        print(f"  -1/2 radial position over the trailing half: "
              f"mean {radii.mean():.1f}, sd {radii.std():.1f} px "
              f"(domain radius ~{lx / 2 - 1:.0f})")

    runs = trailing_runs(pos_counts)
    chosen = next(
        ((s, e, c) for (s, e, c) in runs if e - s + 1 >= min_window and c >= 2),
        None,
    )
    if chosen is None:
        print(f"no run of >={min_window} frames with a stable +1/2 count and "
              f"at least 2 defects")
        sys.exit(3)

    start, end, count = chosen
    print(f"  braiding frames [{start}, {end}] ({end - start + 1} frames), "
          f"{count} mobile +1/2 defects")

    n, codes = v2.braidword_from_frames(positive[start:end + 1])
    period = v2.period_word(codes) if codes else []
    h_period = v2.topological_entropy(n, period) if period else 0.0
    h_window = v2.topological_entropy(n, codes) if codes else 0.0
    repeats = len(codes) / len(period) if period else 0.0

    print(f"n_strands={n}")
    print(f"window word ({len(codes)} generators, {repeats:.2f} periods)")
    print(f"braid word: {{{fmt(period)}}}" if period else "braid word: {} (trivial)")
    print(f"topological entropy: {h_period:.6f}")
    print(f"whole-window entropy (not comparable to a published value): {h_window:.6f}")


if __name__ == "__main__":
    main()
