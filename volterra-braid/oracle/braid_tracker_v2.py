# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24"]
# ///
"""braid_tracker_v2 -- cleaned reference for braid-word extraction.

This is an adapted, corrected reimplementation of the defect-detection,
tracking, and braid-word-extraction logic in the published
``Chaos-Generating-Periodic-Orbits/braid_tracker.py``. It is the Python oracle
that volterra's Rust implementation (crate ``volterra-braid``, exposed via the
``volterra`` PyO3 extension) is validated against on identical inputs.

Run directly for a self-test::

    uv run braid_tracker_v2.py

Deltas from the original ``braid_tracker.py`` (all deliberate, the original is
quirky):

1. **Flood-fill bug fixed.** The original ``cluster`` recursion (line 14) has
   ``cluster(...,-1,y+1)`` where ``(x-1,y+1)`` was intended, so one of the eight
   diagonal neighbours is mis-sampled. v2 uses a correct, bounds-checked,
   iterative 8-connected flood fill.
2. **Tracking is a per-frame bijection.** The original greedy nearest-neighbour
   tracker (lines 274-282) never removes a claimed defect, so two worldlines can
   latch onto the same defect. v2 removes a defect once claimed, giving a proper
   assignment (each defect used at most once per frame).
3. **No global I/O side effects.** The original walks the filesystem, writes
   PNGs, runs ffmpeg/scp, and hardcodes ``out_dir='5_2fa'``, ``Lx=Ly=100``,
   ``bc_label='circular'``. v2 is a pure library: arrays in, braid word out.
4. **Generator index convention preserved.** Same as the original: positions are
   sorted by x; an adjacent swap at 0-based gap ``k`` emits generator
   ``sigma_{k+1}`` (1-based). Sign preserved: positive when, in the new x-order,
   the strand now on the left of the swapped pair has the larger y.
5. **Degenerate-frame skip preserved.** Frames where two strands share an exact
   x or y coordinate are skipped (original line 346).
6. **Detection density preserved.** Same Jacobian
   ``ss = (2 dxQxy)(2 dyQxx) - (2 dxQxx)(2 dyQxy)`` with periodic central
   differences, same ``|ss| > threshold`` candidate test, same charge
   ``= -sign(ss)``.

These match the volterra-braid Rust crate exactly, so a passing cross-check
means the two implementations agree.
"""

from __future__ import annotations

import math
import numpy as np


def detect_defects(qxx, qxy, nx, ny, threshold, mask):
    """Detect defects in a row-major nx*ny Q grid.

    qxx[x*ny + y], qxy[x*ny + y] are the components; mask[x*ny + y] False zeroes
    ss at that cell. Returns a list of (x, y, charge) triples.
    """
    qxx = np.asarray(qxx, float).reshape(nx, ny)
    qxy = np.asarray(qxy, float).reshape(nx, ny)
    mask = np.asarray(mask, bool).reshape(nx, ny)

    field = np.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y]:
                continue
            xup, xdn = (x + 1) % nx, (x - 1) % nx
            yup, ydn = (y + 1) % ny, (y - 1) % ny
            dx_qxx = qxx[xup, y] - qxx[xdn, y]
            dx_qxy = qxy[xup, y] - qxy[xdn, y]
            dy_qxx = qxx[x, yup] - qxx[x, ydn]
            dy_qxy = qxy[x, yup] - qxy[x, ydn]
            ss = dx_qxy * dy_qxx - dx_qxx * dy_qxy
            if abs(ss) > threshold:
                field[x, y] = ss

    visited = np.zeros((nx, ny), bool)
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    defects = []
    for x in range(nx):
        for y in range(ny):
            if field[x, y] == 0.0 or visited[x, y]:
                continue
            sign = math.copysign(1.0, field[x, y])
            stack = [(x, y)]
            visited[x, y] = True
            sx = sy = 0.0
            count = 0
            while stack:
                cx, cy = stack.pop()
                sx += cx
                sy += cy
                count += 1
                for dx, dy in neigh:
                    nxp, nyp = cx + dx, cy + dy
                    if not (0 <= nxp < nx and 0 <= nyp < ny):
                        continue
                    if (not visited[nxp, nyp] and field[nxp, nyp] != 0.0
                            and math.copysign(1.0, field[nxp, nyp]) == sign):
                        visited[nxp, nyp] = True
                        stack.append((nxp, nyp))
            defects.append((sx / count, sy / count, int(-sign)))
    return defects


def track(frames):
    """Greedy nearest-neighbour tracking with removal (per-frame bijection).

    frames: list of frames, each a list of (x, y, charge). Returns a list of
    worldlines, each a dict {'positions': [(x, y), ...], 'charge': int}.
    """
    assert frames, "track requires at least one frame"
    dim = len(frames[0])
    worldlines = [{"positions": [(d[0], d[1])], "charge": int(d[2])} for d in frames[0]]
    for frame in frames[1:]:
        assert len(frame) >= dim, "frame has fewer defects than tracked worldlines"
        claimed = [False] * len(frame)
        for wl in worldlines:
            px, py = wl["positions"][-1]
            best, best_d2 = -1, float("inf")
            for j, d in enumerate(frame):
                if claimed[j]:
                    continue
                d2 = (d[0] - px) ** 2 + (d[1] - py) ** 2
                if d2 < best_d2:
                    best_d2, best = d2, j
            claimed[best] = True
            wl["positions"].append((frame[best][0], frame[best][1]))
    return worldlines


def _has_exact_duplicate(vals):
    return len(set(vals)) != len(vals)


def extract_braidword(worldlines):
    """Extract (n_strands, codes) from worldlines. codes are signed 1-based."""
    dim = len(worldlines)
    if dim < 2:
        return (max(dim, 1), [])
    n_frames = len(worldlines[0]["positions"])
    gens = []
    prev_order = None
    for t in range(n_frames):
        xs = [w["positions"][t][0] for w in worldlines]
        ys = [w["positions"][t][1] for w in worldlines]
        if _has_exact_duplicate(xs) or _has_exact_duplicate(ys):
            continue
        order = sorted(range(dim), key=lambda s: xs[s])
        if prev_order is None:
            prev_order = order
            continue
        if order == prev_order:
            continue
        cur = list(prev_order)
        for k in range(dim):
            while cur[k] != order[k]:
                p = next(p for p in range(k + 1, dim) if cur[p] == order[k])
                j = p - 1
                cur[j], cur[j + 1] = cur[j + 1], cur[j]
                positive = ys[cur[j]] > ys[cur[j + 1]]
                gens.append((j + 1) if positive else -(j + 1))
        prev_order = order
    return (dim, gens)


def braidword_from_frames(frames):
    """track + extract_braidword in one call."""
    return extract_braidword(track(frames))


def topological_entropy(n_strands, codes):
    """log of the Burau-at-t=-1 spectral radius (the dilatation)."""
    n = n_strands
    m = np.eye(n)
    for c in codes:
        i = abs(c) - 1
        b = np.eye(n)
        if c < 0:  # sigma_i^-1
            b[i, i], b[i, i + 1], b[i + 1, i], b[i + 1, i + 1] = 0.0, 1.0, -1.0, 2.0
        else:  # sigma_i
            b[i, i], b[i, i + 1], b[i + 1, i], b[i + 1, i + 1] = 2.0, -1.0, 1.0, 0.0
        m = m @ b
    lam = max(abs(ev) for ev in np.linalg.eigvals(m))
    # See volterra-braid entropy.rs DILATATION_TOL: a reducible braid's Burau
    # matrix can have a defective eigenvalue at 1, so the solver returns 1 + ~1e-4;
    # the smallest pseudo-Anosov dilatation is well above 1 + 1e-3.
    return 0.0 if lam <= 1.0 + 1e-3 else math.log(lam)


def realize_braid(n_strands, codes, frames_per_gen=8, periods=1):
    """Render a braid word as a defect-position time series.

    Mirrors the Rust ``volterra_braid::synthetic::realize_braid`` exactly so the
    same frames can be fed to both implementations.
    """
    n = n_strands
    m = max(frames_per_gen, 2)
    eps, h = 1e-3, 2.0
    strand_at_slot = list(range(n))

    def baseline(s):
        return s * eps

    def frame_from(xs, ys):
        return [(xs[s], ys[s], 1) for s in range(n)]

    def rest(sas):
        xs = [0.0] * n
        ys = [0.0] * n
        for slot, s in enumerate(sas):
            xs[s] = slot + 1
            ys[s] = baseline(s)
        return frame_from(xs, ys)

    frames = [rest(strand_at_slot)]
    for _ in range(max(periods, 1)):
        for c in codes:
            i = abs(c)
            inverse = c < 0
            left_slot = i - 1
            a = strand_at_slot[left_slot]
            b = strand_at_slot[left_slot + 1]
            for k in range(m):
                tau = (k + 1) / m
                pert = h * math.sin(math.pi * tau)
                xs = [0.0] * n
                ys = [0.0] * n
                for slot, s in enumerate(strand_at_slot):
                    xs[s] = slot + 1
                    ys[s] = baseline(s)
                xs[a] = (left_slot + 1) + tau
                xs[b] = (left_slot + 2) - tau
                if inverse:
                    ys[a] = baseline(a) + pert
                    ys[b] = baseline(b) - pert
                else:
                    ys[a] = baseline(a) - pert
                    ys[b] = baseline(b) + pert
                frames.append(frame_from(xs, ys))
            strand_at_slot[left_slot], strand_at_slot[left_slot + 1] = (
                strand_at_slot[left_slot + 1],
                strand_at_slot[left_slot],
            )
    return frames


def _fundamental_period(codes):
    n = len(codes)
    for period in range(1, n + 1):
        if n % period == 0 and all(codes[i] == codes[i % period] for i in range(n)):
            return codes[:period]
    return codes


def _self_test():
    cases = [
        ("golden", 3, [-2, 1], 0.962_423_650_119_205_8),
        ("silver", 4, [3, 1, 2, -3, -1, -2], 1.762_747_174_039_086),
    ]
    for name, n, codes, h_exact in cases:
        frames = realize_braid(n, codes, frames_per_gen=10, periods=2)
        nn, got = braidword_from_frames(frames)
        period = _fundamental_period(got)
        assert nn == n, f"{name}: n_strands {nn} != {n}"
        assert period == codes, f"{name}: extracted {period} != {codes}"
        h = topological_entropy(n, codes)
        assert abs(h - h_exact) < 1e-9, f"{name}: entropy {h} != {h_exact}"
        print(f"  {name}: word {period} entropy {h:.6f} OK")
    print("braid_tracker_v2 self-test PASS")


if __name__ == "__main__":
    _self_test()
