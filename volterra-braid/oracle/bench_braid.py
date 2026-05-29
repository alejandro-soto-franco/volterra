"""Performance benchmark: volterra braid pipeline (Rust/PyO3) vs the CGPO Python.

Times the full braid-analysis pipeline -- defect detection from Q-tensor grids,
tracking, and braid-word extraction -- on identical input, comparing:

  - volterra (Rust, via the `volterra` PyO3 extension), and
  - braid_tracker_v2.py (a faithful pure-Python transcription of the CGPO
    braid_tracker.py algorithm; its per-cell ss loop + flood fill + tracking are
    the same work the published script does, minus the plotting/IO).

Detection dominates (an Lx*Ly per-cell Jacobian + connected-component flood fill
per frame), so this is essentially a native-vs-Python throughput comparison of
the defect-tracking analytics.

Run (needs `maturin develop --release` first):
    .venv/bin/python volterra-braid/oracle/bench_braid.py
    .venv/bin/python volterra-braid/oracle/bench_braid.py 150 128   # frames grid
"""

from __future__ import annotations

import sys
import time

import braid_tracker_v2 as v2
import compare_cgpo as cc

try:
    import volterra
except ImportError:
    sys.exit("ERROR: import volterra failed; run `maturin develop --release` first.")


def build_grids(n_frames, lx):
    """Render `n_frames` Q-grids of a 3-defect golden orbit at lx*lx resolution."""
    cc.LX = cc.LY = lx
    cc.RADIUS = lx // 2 - 1
    # Scale the orbit geometry to the grid so defects stay well inside.
    cc.GY_SPACING = {3: lx * 0.24, 4: lx * 0.2}
    pts = cc.disk_orbit(3, [-2, 1], fpg=10, periods=max(1, n_frames // 20))
    pts = pts[:n_frames]
    grids = [cc.render_q(p) for p in pts]  # list of (qxx_flat, qxy_flat)
    mask = cc.circular_mask()
    return grids, mask


def time_volterra(grids, mask, lx):
    frames = []
    t0 = time.perf_counter()
    for (qxx, qxy) in grids:
        defs = volterra.braid_detect_defects(qxx.tolist(), qxy.tolist(), lx, lx, cc.THRESHOLD, mask)
        frames.append([(y, x, c) for (x, y, c) in defs])  # (gy, gx) ordering
    t_det = time.perf_counter() - t0
    t1 = time.perf_counter()
    _, codes = volterra.braid_word_from_frames(frames)
    t_word = time.perf_counter() - t1
    return t_det, t_word, codes


def time_python(grids, mask, lx):
    frames = []
    t0 = time.perf_counter()
    for (qxx, qxy) in grids:
        defs = v2.detect_defects(qxx.tolist(), qxy.tolist(), lx, lx, cc.THRESHOLD, mask)
        frames.append([(y, x, c) for (x, y, c) in defs])
    t_det = time.perf_counter() - t0
    t1 = time.perf_counter()
    _, codes = v2.braidword_from_frames(frames)
    t_word = time.perf_counter() - t1
    return t_det, t_word, codes


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    lx = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print(f"Braid pipeline benchmark: {n_frames} frames at {lx}x{lx} ({lx*lx} sites/frame)\n")

    grids, mask = build_grids(n_frames, lx)

    # Warm-up (JIT-free, but warms caches / first-call allocation).
    time_volterra(grids[:5], mask, lx)
    time_python(grids[:5], mask, lx)

    v_det, v_word, v_codes = time_volterra(grids, mask, lx)
    p_det, p_word, p_codes = time_python(grids, mask, lx)

    sites = n_frames * lx * lx
    print(f"{'stage':<22}{'volterra (Rust)':>20}{'braid_tracker_v2 (Py)':>24}{'speedup':>10}")
    print("-" * 76)
    for label, vt, pt in (("detection (total)", v_det, p_det), ("track+word (total)", v_word, p_word)):
        print(f"{label:<22}{vt*1e3:>17.2f} ms{pt*1e3:>21.2f} ms{pt/max(vt,1e-12):>9.1f}x")
    v_tot, p_tot = v_det + v_word, p_det + p_word
    print(f"{'TOTAL':<22}{v_tot*1e3:>17.2f} ms{p_tot*1e3:>21.2f} ms{p_tot/max(v_tot,1e-12):>9.1f}x")
    print()
    print(f"  volterra: {v_det/n_frames*1e6:8.2f} us/frame detection, {v_det/sites*1e9:6.2f} ns/site")
    print(f"  python:   {p_det/n_frames*1e6:8.2f} us/frame detection, {p_det/sites*1e9:6.2f} ns/site")
    print(f"\n  braid words match: {v_codes == p_codes}  (volterra {v_codes[:6]}...)")


if __name__ == "__main__":
    main()
