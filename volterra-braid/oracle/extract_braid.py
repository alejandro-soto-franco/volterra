#!/usr/bin/env python3
"""Extract a braid word and topological entropy from a fd2d Q-frame trajectory.

Reads every saved Q_*.txt frame from a fd2d run directory, detects defects
per frame with braid_tracker_v2's Jacobian detector, restricts to the trailing
window of frames whose defect count is stable (past any initial transient),
tracks worldlines, and extracts the braid word and its topological entropy.

Usage:
    python extract_braid.py <run_dir> <lx> <expected_n> [threshold]
"""
import sys
from pathlib import Path
import glob
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import braid_tracker_v2 as v2  # noqa: E402


def circular_mask(lx, ly):
    radius = lx // 2 - 1
    xs, ys = np.meshgrid(np.arange(lx), np.arange(ly), indexing="ij")
    return (xs - radius) ** 2 + (ys - radius) ** 2 <= radius * radius


def load_frame(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def main():
    run_dir = sys.argv[1]
    lx = int(sys.argv[2])
    expected_n = int(sys.argv[3])
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    ly = lx
    mask = circular_mask(lx, ly).ravel()

    files = sorted(glob.glob(f"{run_dir}/Q/Q_*.txt"))
    if not files:
        print(f"no Q frames found under {run_dir}/Q")
        sys.exit(2)

    all_defects = []
    for f in files:
        qxx, qxy = load_frame(f)
        defects = v2.detect_defects(qxx, qxy, lx, ly, threshold, mask)
        all_defects.append(defects)

    counts = [len(d) for d in all_defects]
    print(f"{len(files)} frames, defect counts: {counts}")

    # Trailing run of frames with a constant count, at least 8 frames long,
    # preferring a count equal to expected_n if such a run exists.
    def trailing_runs():
        runs = []
        i = len(counts) - 1
        while i >= 0:
            j = i
            while j - 1 >= 0 and counts[j - 1] == counts[i]:
                j -= 1
            runs.append((j, i, counts[i]))
            i = j - 1
        return runs

    runs = trailing_runs()
    chosen = None
    for (start, end, c) in runs:
        if c == expected_n and end - start + 1 >= 8:
            chosen = (start, end, c)
            break
    if chosen is None:
        for (start, end, c) in runs:
            if end - start + 1 >= 8 and c >= 2:
                chosen = (start, end, c)
                break
    if chosen is None:
        print("no stable run of >=8 frames with a consistent defect count found")
        sys.exit(3)

    start, end, c = chosen
    print(f"using frames [{start}, {end}] ({end - start + 1} frames), "
          f"defect count {c} (expected {expected_n})")

    window = all_defects[start:end + 1]
    n, codes = v2.braidword_from_frames(window)

    def fmt(cs):
        return " ".join(
            (f"sigma_{abs(g)}^-1" if g < 0 else f"sigma_{g}") for g in cs
        )

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
