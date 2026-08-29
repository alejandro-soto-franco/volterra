"""Figure 5 of Mitchell et al. (2024), reproduced.

Dimensionless topological entropy against the active length, with the bands the
paper shades by how many `+1/2` defects survive: red where none or one remain,
blue where two do, green where three or more. The blue curve is the braid
prediction `log(phi + sqrt phi) / (T_tilde / 4)`, which needs the measured
period and is drawn only where a period was found.

Two series are drawn, the standard model and the enhanced-locking one, from
runs that differ in nothing else.

    uv run --with numpy,matplotlib plot_fig5.py <sweep-dir> [<out.pdf>]

The sweep directory holds `be_ella<L>` and `lock_ella<L>` run directories.
"""
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "axes.grid": False,
    "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000",
    "ytick.color": "#000000", "axes.edgecolor": "#000000",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
})
BLUE = "#1f4e9c"
RED = "#d81e05"
GREEN = "#1a9a3a"
BANDS = {"red": "#f4c9c4", "blue": "#c6d4ea", "green": "#c9e6cf"}


def series(sweep, prefix):
    """One model's `(ell_a, h_tilde, mean n_plus, braid prediction)` rows."""
    out = []
    for d in sorted(sweep.glob(f"{prefix}_ella*")):
        m = re.search(r"_ella([0-9.]+)$", d.name)
        if not m or not (d / "stats.csv").exists():
            continue
        la = float(m.group(1))
        rows = [l.split(",") for l in (d / "stats.csv").read_text().splitlines()[1:]
                if l.strip()]
        if len(rows) < 20:
            continue
        t = np.array([float(r[1]) for r in rows])
        npl = np.array([float(r[4]) for r in rows])
        late = t >= t[-1] - 0.25 * (t[-1] - t[0])
        h = np.nan
        ef = d / "entropy.json"
        if ef.exists():
            h = json.loads(ef.read_text()).get("h_tilde", np.nan)
        pred = np.nan
        bf = d / "braid.json"
        if bf.exists():
            b = json.loads(bf.read_text())
            # Only where a period was actually found: a prediction from a lag
            # the autocorrelation never peaked at is not a prediction.
            if b.get("period_peak", 0.0) > 0.5:
                pred = b.get("h_tilde_max", np.nan)
        out.append((la, h, float(npl[late].mean()), pred, float(t[-1])))
    return sorted(out)


def band_colour(n):
    if n >= 2.5:
        return "green"
    if n >= 1.5:
        return "blue"
    return "red"


def main():
    sweep = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else sweep / "figure5.pdf"
    be = series(sweep, "be")
    lock = series(sweep, "lock")
    if not be:
        sys.exit(f"no be_ella* runs in {sweep}")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Bands from the standard model's defect census, which is what the paper
    # shades by.
    xs = [r[0] for r in be]
    for k, r in enumerate(be):
        lo = xs[k] - (xs[k] - xs[k - 1]) / 2 if k else xs[k] - 0.125
        hi = xs[k] + (xs[k + 1] - xs[k]) / 2 if k + 1 < len(xs) else xs[k] + 0.125
        ax.axvspan(lo, hi, color=BANDS[band_colour(r[2])], lw=0, zorder=0)

    ax.plot([r[0] for r in be], [r[1] for r in be], color="#000000", lw=1.4,
            marker="o", ms=4, label="standard model", zorder=3)
    if lock:
        ax.plot([r[0] for r in lock], [r[1] for r in lock], color=RED, lw=1.4,
                marker="s", ms=4, label="enhanced locking", zorder=3)
    pb = [(r[0], r[3]) for r in be if np.isfinite(r[3])]
    if pb:
        ax.plot([p[0] for p in pb], [p[1] for p in pb], color=BLUE, lw=1.4,
                ls="--", marker="^", ms=4, label="braid prediction", zorder=4)

    ax.set_xlabel(r"active length $\ell_a$")
    ax.set_ylabel(r"$\tilde h$")
    ax.set_xlim(min(xs) - 0.125, max(xs) + 0.125)
    finite = [r[1] for r in be if np.isfinite(r[1])] + \
             [r[1] for r in lock if np.isfinite(r[1])]
    if finite:
        ax.set_ylim(0, max(finite) * 1.25)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color("#000000")

    # The bands, named once in the corner rather than in a second legend.
    ax.text(0.02, 0.98,
            "shading: red 0 or 1 defect, blue 2, green 3 or more",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{'ell_a':>6s} {'model':>10s} {'t_end':>6s} {'n+':>6s} "
          f"{'h_tilde':>10s} {'braid':>10s}")
    for name, rows in (("standard", be), ("locking", lock)):
        for r in rows:
            print(f"{r[0]:6.2f} {name:>10s} {r[4]:6.1f} {r[2]:6.2f} "
                  f"{r[1]:10.3e} {r[3]:10.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
