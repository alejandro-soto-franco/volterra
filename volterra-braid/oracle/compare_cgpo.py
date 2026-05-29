# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24"]
# ///
"""End-to-end comparison: the UNMODIFIED CGPO braid_tracker.py vs volterra.

For each braiding configuration (golden cardioid, silver nephroid) this:

  1. Builds a genuinely 2D, well-separated, tie-free defect orbit that realises
     that braid word (each generator is a 180-degree rotation of the adjacent
     pair about their midpoint; a small per-rank y-tilt keeps resting defects at
     distinct y so neither implementation skips frames).
  2. Renders each frame as a 100x100 Q-tensor grid (superposed +1/2 winding
     fields) and writes `5_2fa/Q/frame_XXXX.txt` in the exact format the
     published script reads (Lx*Ly lines "Qxx Qxy", x-outer y-inner).
  3. Runs the real `Chaos-Generating-Periodic-Orbits/braid_tracker.py` via uv
     (matplotlib pinned <3.8 so the script runs unmodified) and parses its
     `5_2fa/out.txt` braid word.
  4. Runs volterra on the SAME files / window (frames 80-139) / circular mask.
  5. Compares the two braid-word sequences and the topological entropy, and
     checks the orbit realises the analytic braid.

Run:
    .venv/bin/python volterra-braid/oracle/compare_cgpo.py            # both configs
    .venv/bin/python volterra-braid/oracle/compare_cgpo.py golden     # one config
    .venv/bin/python volterra-braid/oracle/compare_cgpo.py --no-real  # skip the slow script

Exits non-zero on any disagreement.
"""

from __future__ import annotations

import math
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

import braid_tracker_v2 as v2

try:
    import volterra
except ImportError:
    sys.exit("ERROR: `import volterra` failed; run `maturin develop --release` first.")

LX = LY = 100
RADIUS = LX // 2 - 1  # 49, matching braid_tracker.py's circular BC
THRESHOLD = 0.1
CGPO_DIR = pathlib.Path.home() / "Chaos-Generating-Periodic-Orbits"
BRAID_TRACKER = CGPO_DIR / "braid_tracker.py"

# braid_tracker.py: sorted(files)[40:140] then cutoff=40 -> braid word from
# frames at sorted indices 80..139.
WINDOW = slice(80, 140)

# fpg=10 makes the 60-frame window (frames 80-139) hold exactly 6 crossings, an
# integer number of braid periods (3 for golden, 1 for silver), so each window
# word reduces cleanly to the analytic braid.
CONFIGS = {
    "golden": dict(n=3, codes=[-2, 1], fpg=10, periods=8),
    "silver": dict(n=4, codes=[3, 1, 2, -3, -1, -2], fpg=10, periods=3),
}

GY_SPACING = {3: 24.0, 4: 18.0}  # spacing along the braid (grid-y) axis
GX_TILT = 5.0   # distinct resting grid-x per rank (cells) -> no ties at rest
GX_DETOUR = 12.0  # over/under amplitude in grid-x during a crossing


def disk_orbit(n: int, codes: list[int], fpg: int, periods: int):
    """Realise the braid as a 2D defect orbit, braiding along the grid-y axis.

    Defects rest at distinct (grid-x, grid-y); each generator slides the adjacent
    pair (by grid-y rank) past each other along grid-y while one detours far in
    grid-x (the over strand) and the other the opposite way. The large grid-x gap
    at the crossing makes the over/under -- hence the generator sign -- robust to
    detection noise. The braid axis is grid-y, which is the axis braid_tracker.py
    reads. Returns frames of strand-indexed (grid-x, grid-y) centres.
    """
    cx = cy = float(RADIUS)
    gysp = GY_SPACING.get(n, 22.0)

    def rest_gx(rank):
        return cx + (rank - (n - 1) / 2.0) * GX_TILT

    def rest_gy(rank):
        return cy + (rank - (n - 1) / 2.0) * gysp

    strand_at_rank = list(range(n))
    pos = {k: [rest_gx(k), rest_gy(k)] for k in range(n)}

    def snapshot():
        return [tuple(pos[s]) for s in range(n)]

    frames = [snapshot()]
    for _ in range(periods):
        for code in codes:
            rl = abs(code) - 1
            a = strand_at_rank[rl]      # at the lower grid-y rank
            b = strand_at_rank[rl + 1]  # at the higher grid-y rank
            gya0, gyb0 = rest_gy(rl), rest_gy(rl + 1)
            gxa0, gxb0 = rest_gx(rl), rest_gx(rl + 1)
            sgn = 1.0 if code > 0 else -1.0
            for k in range(1, fpg + 1):
                t = k / fpg
                bump = GX_DETOUR * math.sin(math.pi * t)
                pos[a] = [gxa0 + t * (gxb0 - gxa0) + sgn * bump, gya0 + t * (gyb0 - gya0)]
                pos[b] = [gxb0 + t * (gxa0 - gxb0) - sgn * bump, gyb0 + t * (gya0 - gyb0)]
                frames.append(snapshot())
            strand_at_rank[rl], strand_at_rank[rl + 1] = b, a
    return frames


def render_q(points):
    """Superpose +1/2 winding fields -> (qxx_flat, qxy_flat), row-major x*LY+y."""
    gx, gy = np.meshgrid(np.arange(LX), np.arange(LY), indexing="ij")
    ang = np.zeros((LX, LY))
    for (px, py) in points:
        ang += np.arctan2(gy - py, gx - px)
    return np.cos(ang).reshape(-1), np.sin(ang).reshape(-1)


def circular_mask():
    """braid_tracker.py's ss-zeroing for circular BC -> flat bool mask."""
    sim = {
        (x, y)
        for x in range(LX)
        for y in range(LY)
        if (x - RADIUS) ** 2 + (y - RADIUS) ** 2 <= RADIUS ** 2
    }
    four = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    outer = {(x, y) for (x, y) in sim if any((x + dx, y + dy) not in sim for dx, dy in four)}
    inner = {
        (x, y)
        for (x, y) in sim
        if (x, y) not in outer and any((x + dx, y + dy) in outer for dx, dy in four)
    }
    mask = [False] * (LX * LY)
    for (x, y) in sim:
        if (x, y) not in outer and (x, y) not in inner:
            mask[x * LY + y] = True
    return mask


# disk_orbit braids along the grid-y axis, which is the axis braid_tracker.py
# reads (it stores X<-y, Y<-x at lines 283-286 and orders by X). volterra orders
# by its first coordinate, so we feed it the swapped (grid-y, grid-x) so both
# read the same braid axis with the sign taken from grid-x.
def write_fixture(qdir: pathlib.Path, frames_points):
    qdir.mkdir(parents=True, exist_ok=True)
    for i, pts in enumerate(frames_points):
        qxx, qxy = render_q(pts)
        np.savetxt(qdir / f"frame_{i:04d}.txt", np.column_stack([qxx, qxy]), fmt="%.6f")


def parse_out_txt(path: pathlib.Path):
    # braid_tracker.py writes "sigma_K" or "sigma_K^-1" (its f-string `^{-1}`
    # evaluates the field {-1} to the integer -1, so there are no braces).
    codes = []
    pat = re.compile(r"sigma_(\d+)(\^-1)?")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            codes.append(-int(m.group(1)) if m.group(2) else int(m.group(1)))
    return codes


def run_real_braid_tracker(workdir: pathlib.Path):
    cmd = [
        "uv", "run", "--with", "numpy", "--with", "matplotlib==3.7.3",
        "--python", "3.11", str(BRAID_TRACKER),
    ]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
    out_txt = workdir / "5_2fa" / "out.txt"
    if not out_txt.exists():
        print("    braid_tracker.py produced no out.txt; stderr tail:")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-15:]))
        return None
    return parse_out_txt(out_txt)


def volterra_detect_frames(qdir: pathlib.Path, mask, window):
    files = sorted(qdir.glob("frame_*.txt"))
    if window is not None:
        files = files[window]
    frames, counts = [], []
    for f in files:
        data = np.loadtxt(f)
        defs = volterra.braid_detect_defects(
            data[:, 0].tolist(), data[:, 1].tolist(), LX, LY, THRESHOLD, mask
        )
        counts.append(len(defs))
        # Swap to (y, x): volterra then orders by the grid-y axis that
        # braid_tracker.py braids along, with the sign taken from x.
        frames.append([(y, x, c) for (x, y, c) in defs])
    return frames, counts


def cyclic_period(codes):
    """Shortest block whose cyclic repetition reproduces codes (else codes)."""
    n = len(codes)
    for p in range(1, n + 1):
        if n % p == 0 and all(codes[i] == codes[i % p] for i in range(n)):
            return codes[:p]
    return codes


def is_cyclic_rotation(seq, period):
    """True if seq is a cyclic rotation of whole repetitions of `period`."""
    if not period or len(seq) % len(period) != 0:
        return False
    reps = len(seq) // len(period)
    full = period * reps
    return any(seq == full[r:] + full[:r] for r in range(len(period)))


def compare_config(name: str, cfg: dict, mask, run_real: bool) -> int:
    n, codes = cfg["n"], cfg["codes"]
    analytic_h = v2.topological_entropy(n, codes)
    print(f"\n=== {name} ({n} strands, target {codes}, analytic h={analytic_h:.6f}) ===")
    frames_points = disk_orbit(n, codes, cfg["fpg"], cfg["periods"])
    print(f"  generated {len(frames_points)} frames")

    # Ideal-position word over the full orbit: validates the kinematics realise
    # the target braid, independent of detection.
    # Swap to (grid-y, grid-x) so volterra orders by the grid-y braid axis.
    ideal_frames = [[(gy, gx, 1) for (gx, gy) in pts] for pts in frames_points]
    _, ideal_codes = volterra.braid_word_from_frames(ideal_frames)
    ideal_p = cyclic_period(ideal_codes)
    ok_ideal = is_cyclic_rotation(ideal_codes, codes) or ideal_p == codes
    print(f"  ideal-position word period {ideal_p} (realises target: {ok_ideal})")

    fails = 0
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        qdir = workdir / "5_2fa" / "Q"
        write_fixture(qdir, frames_points)

        det_frames, counts = volterra_detect_frames(qdir, mask, WINDOW)
        print(f"  volterra per-frame defect counts in window: {sorted(set(counts))}")
        if set(counts) != {n}:
            print(f"  WARNING: detection not stable at {n} defects/frame")
        _, vol_codes = volterra.braid_word_from_frames(det_frames)

        if not run_real:
            print(f"  volterra (detected) window word: {vol_codes}")
            print("  [--no-real: skipped braid_tracker.py]")
            return 0

        print("  running real braid_tracker.py via uv (matplotlib<3.8)...")
        real_codes = run_real_braid_tracker(workdir)
        if real_codes is None:
            return 1

    print(f"  braid_tracker.py window word: {real_codes}")
    print(f"  volterra        window word: {vol_codes}")

    if vol_codes == real_codes:
        print("  AGREE: identical braid-word sequences")
    else:
        print("  MISMATCH: braid-word sequences differ")
        fails += 1

    # Braiding analytics: each window word, reduced to its period, should have
    # the analytic topological entropy (a conjugacy invariant; robust to a
    # mirrored sign convention which gives the same entropy).
    for label, seq in (("braid_tracker.py", real_codes), ("volterra", vol_codes)):
        per = cyclic_period(seq)
        h = v2.topological_entropy(n, per) if per else 0.0
        matches = abs(h - analytic_h) < 1e-6
        print(f"    {label}: period {per}, entropy(period) {h:.6f}, matches analytic {analytic_h:.6f}: {matches}")
        if not matches:
            fails += 1

    return fails


def main():
    args = sys.argv[1:]
    run_real = "--no-real" not in args
    only = next((a for a in args if not a.startswith("-")), None)
    if not BRAID_TRACKER.exists() and run_real:
        sys.exit(f"ERROR: {BRAID_TRACKER} not found")
    mask = circular_mask()
    fails = 0
    for nm, cfg in CONFIGS.items():
        if only and nm != only:
            continue
        fails += compare_config(nm, cfg, mask, run_real)
    if fails:
        print(f"\nFAILED: {fails} disagreement(s) between volterra and braid_tracker.py")
        sys.exit(1)
    print("\nPASS: volterra and the CGPO braid_tracker.py agree on the braiding analytics")


if __name__ == "__main__":
    main()
