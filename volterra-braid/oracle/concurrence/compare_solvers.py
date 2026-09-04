# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scipy>=1.10"]
# ///
"""Topological concurrence: the reference lattice against the conforming mesh.

Both solvers are run on the nephroid at the published phase point and read for
the same three quantities: the defect complement, the net charge, and the braid
the positive cores trace. The fields diverge pointwise, since the system is
chaotic, so the comparison is on the invariants rather than on the trajectory.

The point. The published pair `(l_a, l_c) = (0.0128, 0.0766)` maps to lattice
units through the per-shape divisor `0.764031 Lx`, giving `als = 2` and
`ncl = 11.7` at `Lx = 200`. Both solvers take those two lengths, the same fixed
constants, `dt = 1e-4`, and the same 500-step frame cadence.

Matching the WALL is what makes the two comparable. The lattice mask at
`d = 0.99` imposes a charge of `+1`, so the mesh is run at `d = 0.9`, which
imposes `+1` as well and needs no cusp treatment; at `d = 1` the mesh imposes
`+2` and holds four positive cores with no negative ones, which is a different
boundary condition rather than a different answer.

Measured over 120000 steps, 240 frames:

    reference lattice   (4, 2) in 212 frames, net +1, braid entropy 0.000000
    conforming mesh     (4, 2) in 216 frames, net +1, braid entropy 1.762747

The complement and the net charge agree. The braid does not yet: the lattice's
four cores have not begun to exchange by `t = 12`, where the mesh reaches the
silver braid by `t = 1.65`. The published runs are 1.5e6 steps, so the lattice
needs about 3.4 hours at 8.2 ms a step to reach the same window.

Run the reference first, then this:

    mkdir -p run && cd run
    FD_LX=200 FD_LY=200 FD_K=2 FD_D=0.99 FD_ALS=2 FD_NCL=12 \\
    FD_MAX_STEPS=120000 FD_SAVE_EVERY=500 \\
      uv run --python 3.11 --with numpy --with numba \\
             --with matplotlib==3.7.3 --with scipy ../flow_solver_run.py
    uv run compare_solvers.py run/als_2_ncl_12/Q

Passing no directory runs the mesh side alone.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
from collections import Counter

import numpy as np

# The silver braid, up to a cyclic rotation and the one relation that reorders
# it: sigma_1 and sigma_3 act on disjoint strand pairs and commute.
_SILVER = [3, 1, 2, -3, -1, -2]
_SWAPPED = [1, 3, 2, -1, -3, -2]
SILVER_VARIANTS = {
    tuple(w[i:] + w[:i]) for w in (_SILVER, _SWAPPED) for i in range(6)
}
SILVER_ENTROPY = math.log(3 + 2 * math.sqrt(2))

RADIUS_AT_LX_100 = 49.778694002
DIVISOR_PER_LX = 0.764031
PUBLISHED = (0.0128, 0.0766)


def lattice_point(lx):
    """The published dimensionless pair in lattice units at this grid side."""
    d = DIVISOR_PER_LX * lx
    return PUBLISHED[0] * d, PUBLISHED[1] * d


def braid_of(frames, label):
    """The complement, the net charge, and the braid the positive cores trace."""
    counts = [
        (sum(1 for t in f if t[2] > 0), sum(1 for t in f if t[2] < 0)) for f in frames
    ]
    hist = Counter(counts)
    top, n_top = hist.most_common(1)[0]
    net = 0.5 * (top[0] - top[1])
    print(f"{label}: {len(frames)} frames")
    print(f"  complement {top} in {n_top} frames, net charge {net:+.1f}")
    print(f"  histogram {hist.most_common(4)}")

    best = cur = []
    for i, c in enumerate(counts):
        cur = cur + [i] if c[0] == 4 else []
        if len(cur) > len(best):
            best = cur
    print(f"  longest run of four positive cores: {len(best)} frames")
    if len(best) < 15:
        return None

    import volterra as v

    seq = [[t for t in frames[i] if t[2] > 0] for i in best]
    word = v.BraidWord.from_frames(seq)
    block = tuple(word.fundamental_period()[-6:])
    entropy = (
        v.BraidWord(4, list(block)).entropy() if len(block) == 6 else float("nan")
    )
    verdict = "silver" if block in SILVER_VARIANTS else "not silver"
    print(
        f"  braid: {word.n_strands} strands, {len(word)} generators, "
        f"block {list(block)}, entropy {entropy:.6f} ({verdict})"
    )
    return block, entropy


def read_reference(qdir, lx, k=2, d=0.99):
    """The reference's frames, read with the winding detector.

    The mask is the reference's own in-bound test, transcribed. Reading these
    with `braid_detect_defects` and an angle-sized threshold returns nothing on
    a settled field, which is why the winding detector is used here.
    """
    import volterra as v
    from scipy.optimize import fsolve

    radius = lx // 2 - 1
    mask = np.zeros(lx * lx, dtype=bool)
    for x in range(lx):
        for y in range(lx):
            f = lambda u: np.arctan2(y - radius, x - radius) - np.arctan2(
                (k + 1) * np.sin(u) + d * np.sin((k + 1) * u),
                (k + 1) * np.cos(u) + d * np.cos((k + 1) * u),
            )
            u = fsolve(f, 0.1)[0]
            r2 = (
                radius**2
                / (k + 2) ** 2
                * ((k + 1) ** 2 + d**2 + 2 * (k + 1) * d * np.cos(k * u))
            )
            mask[x * lx + y] = (x - radius) ** 2 + (y - radius) ** 2 <= r2

    frames = []
    for fp in sorted(glob.glob(os.path.join(qdir, "Q_*.txt"))):
        a = np.loadtxt(fp)
        frames.append(
            v.braid_detect_defects_winding(
                a[:, 0].tolist(), a[:, 1].tolist(), lx, lx, mask.tolist()
            )
        )
    return frames


def run_mesh(lx=200, steps=120_000, every=500, seed=0):
    """The conforming mesh at the same point, on a wall that imposes +1."""
    import volterra as v

    als, ncl = lattice_point(lx)
    curve = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=RADIUS_AT_LX_100 * lx / 100.0)
    first = curve.features[0]
    r_cusp = min(
        curve.curvature_radius(u) for u in np.linspace(first - 0.3, first + 0.3, 400)
    )
    # `h_bulk = 4.0` sits well inside the cap the core sets, `ncl / 2 = 5.85`,
    # and it is the value the numbers in the README were measured at. Element
    # quality is not monotone in it: 3.90 gives a minimum angle of 12.4 degrees
    # where 4.0 gives 24.3, so the value is pinned rather than derived.
    h_bulk = 4.0
    mesh = v.confined_mesh(curve, h_bulk=h_bulk, h_min=min(max(r_cusp / 4, 0.05), h_bulk))
    charge, worst, over = mesh.imposed_charge(1.0)
    print(
        f"mesh: {mesh.n_vertices} vertices, min angle {mesh.min_angle_deg:.2f} deg, "
        f"imposed charge {charge:+.3f} (worst step {worst:.1f} deg, {over} over 90)"
    )
    run = v.ConfinedRun(
        mesh,
        active_length=round(als),
        coherence_length=round(ncl),
        resolution=lx,
        q_anchor=1.0,
        wall="noslip",
        dt=1e-4,
        seed=seed,
    )
    t0 = time.time()
    frames = []
    for _ in range(steps // every):
        run.step(every)
        frames.append([tuple(t) for t in run.defects()])
    dt = time.time() - t0
    print(f"  {steps} steps in {dt:.1f} s ({1e3 * dt / steps:.2f} ms a step)")
    return frames


def main(argv):
    lx = 200
    als, ncl = lattice_point(lx)
    print(f"published point at Lx = {lx}: als {als:.3f}, ncl {ncl:.3f}\n")

    if len(argv) > 1:
        ref = read_reference(argv[1], lx)
        braid_of(ref, "reference lattice")
        print()
    mine = run_mesh(lx)
    braid_of(mine, "conforming mesh")


if __name__ == "__main__":
    main(sys.argv)
