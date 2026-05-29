# /// script
# requires-python = ">=3.10"
# ///
"""SP2: generate a single-phase-point, reproducible reference runner from the
unmodified CGPO flow-solver.py.

Reads ~/Chaos-Generating-Periodic-Orbits/flow-solver.py and applies a small set
of documented, asserted string patches to produce `flow_solver_run.py` in this
directory:

  * run ONE (active_length_scale, nematic_coherence_length) point from env
    (CGPO_ALS / CGPO_NCL) instead of the 4x5 sweep;
  * grid size and step budget from env (CGPO_LX/CGPO_LY/CGPO_MAX_STEPS/CGPO_MAX_T)
    so a short smoke run is possible before the full 1.5e6-step run;
  * dump the initial director field theta (theta_ic_<runname>.txt) for the
    matched-initial-condition pointwise comparison (SP3);
  * disable the ffmpeg encode and the `rm -r` results cleanup so Q_*.txt and
    u_*.txt persist.

The physics, numerics, and output format are otherwise byte-identical to the
published solver. Run:  uv run patch_flow_solver.py  ->  flow_solver_run.py
Then (smoke):   CGPO_LX=60 CGPO_MAX_STEPS=2000 uv run --with numpy --with numba flow_solver_run.py
Then (full golden, once the als/ncl map is fixed):  CGPO_ALS=.. CGPO_NCL=.. uv run --with numpy --with numba flow_solver_run.py
"""

from __future__ import annotations

import pathlib

SRC = pathlib.Path.home() / "Chaos-Generating-Periodic-Orbits" / "flow-solver.py"
DST = pathlib.Path(__file__).parent / "flow_solver_run.py"

PATCHES = [
    # single phase point from env instead of the 4x5 sweep
    ("for active_length_scale in [1, 2, 3, 4]:",
     "for active_length_scale in [int(os.environ.get('CGPO_ALS', '2'))]:  # patched: single point"),
    ("    for nematic_coherence_length in [4, 5, 6, 7, 8]:",
     "    for nematic_coherence_length in [int(os.environ.get('CGPO_NCL', '6'))]:  # patched: single point"),
    # grid + step budget from env (for smoke runs)
    ("Lx = 200  # number of lattice sites along x direction",
     "Lx = int(os.environ.get('CGPO_LX', '200'))  # patched: env-overridable"),
    ("Ly = 200  # number of lattice sites along y direction",
     "Ly = int(os.environ.get('CGPO_LY', '200'))  # patched: env-overridable"),
    ("max_steps = 1500000",
     "max_steps = int(os.environ.get('CGPO_MAX_STEPS', '1500000'))  # patched"),
    ("max_t = 1500000",
     "max_t = int(os.environ.get('CGPO_MAX_T', '1500000'))  # patched"),
    # dump the matched initial condition, just after the sim is launched
    ("        run_active_nematic_sim(u, Q, p, boundary, bounds, consts_dict, runname)",
     "        np.savetxt(f'theta_ic_{runname}.txt', theta_initial)  # patched: matched IC for SP3\n"
     "        run_active_nematic_sim(u, Q, p, boundary, bounds, consts_dict, runname)"),
    # disable ffmpeg + results cleanup so Q_*.txt / u_*.txt persist
    ("        os.system(f\"ffmpeg -framerate 15 -pattern_type glob -i '{imgpath}{runname}/*.png'   -c:v libx264 -pix_fmt yuv420p {imgpath}{runname}/{runname}.mp4\")",
     "        pass  # patched: ffmpeg disabled"),
    ("        os.system(f\"rm {imgpath}{runname}/*.png\")",
     "        pass  # patched: keep PNGs"),
    ("        os.system(f\"rm -r {resultspath}\")",
     "        pass  # patched: KEEP results (Q_*.txt, u_*.txt)"),
]


def main():
    src = SRC.read_text()
    if "import os" not in src.splitlines()[0:40].__str__():
        # flow-solver.py already imports os; assert so the env patches work.
        assert "import os" in src, "flow-solver.py must import os for the env patches"
    out = src
    for old, new in PATCHES:
        assert out.count(old) == 1, f"patch target not found exactly once: {old[:60]!r} (count={out.count(old)})"
        out = out.replace(old, new)
    DST.write_text(out)
    print(f"wrote {DST}  ({len(PATCHES)} patches applied)")


if __name__ == "__main__":
    main()
