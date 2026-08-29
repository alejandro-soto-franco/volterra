"""Figure 2 of Mitchell et al. (2024), reproduced.

    a  the maximal mixing braid on the torus, drawn from its own construction:
       each +1/2 rod (open and closed dots) runs on the circle of radius L/2
       about a -1/2 defect (red), the two circles meet at the rods' own sites,
       and with the periodic images each rod meets the other's track four times
       a revolution
    b  the defect orbits of a run, lifted and tiled over the plane, with the
       fundamental cell in grey

Panel a is the published construction and needs no run. Panel b needs a
`braid.json` written by `braid_report`.

    uv run --with numpy,matplotlib plot_braid.py <run-dir> [<out.pdf>]
    uv run --with numpy,matplotlib plot_braid.py --cartoon <out.pdf>
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "axes.grid": False,
    "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000",
    "ytick.color": "#000000", "axes.edgecolor": "#000000",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

BLUE = "#1f4e9c"
RED = "#d81e05"
CELL = "#d9d9d9"


def draw_cartoon(ax, L=100.0, tiles=2):
    """The construction of Fig. 2a, over a `(2 tiles + 1)^2` tiling."""
    th = np.linspace(0, 2 * np.pi, 401)
    for m in range(-tiles, tiles + 1):
        for n in range(-tiles, tiles + 1):
            for c in ([0.0, 0.0], [0.5 * L, 0.5 * L]):
                cx, cy = c[0] + m * L, c[1] + n * L
                ax.plot(cx + 0.5 * L * np.cos(th), cy + 0.5 * L * np.sin(th),
                        color=BLUE, lw=0.7, ls=(0, (2, 2)), zorder=1)
                ax.plot([cx], [cy], marker="o", ms=4.5, color=RED, zorder=5)
            # The rods' sites: filled on the vertical edge midpoints, open on
            # the horizontal ones. Every filled dot is one point of the torus,
            # and so is every open dot.
            ax.plot([m * L], [0.5 * L + n * L], marker="o", ms=7.5,
                    markerfacecolor="#000000", markeredgecolor="#000000", zorder=6)
            ax.plot([0.5 * L + m * L], [n * L], marker="o", ms=7.5,
                    markerfacecolor="#ffffff", markeredgecolor="#000000",
                    markeredgewidth=1.1, zorder=6)

    # One quarter arc of each circle, arrowed, to fix the sense.
    for m in range(-tiles, tiles + 1):
        for n in range(-tiles, tiles + 1):
            for c in ([0.0, 0.0], [0.5 * L, 0.5 * L]):
                cx, cy = c[0] + m * L, c[1] + n * L
                a = np.linspace(0.5 * np.pi, np.pi, 60)
                x, y = cx + 0.5 * L * np.cos(a), cy + 0.5 * L * np.sin(a)
                ax.plot(x, y, color=BLUE, lw=1.9, zorder=3)
                ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-6], y[-6]),
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.9),
                            zorder=4)

    ax.add_patch(Rectangle((0, 0), L, L, facecolor=CELL, edgecolor="none", zorder=0))
    ax.set_xlim(-1.4 * L, 1.4 * L)
    ax.set_ylim(-1.4 * L, 1.4 * L)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def draw_orbits(ax, b, tiles=1):
    """Panel b: the run's lifted worldlines, tiled over the plane."""
    L = b["lx"]
    wl = b["worldlines"]
    charge = b["charge"]
    for m in range(-tiles, tiles + 2):
        for n in range(-tiles, tiles + 2):
            for s, w in enumerate(wl):
                p = np.asarray(w)
                # Lifted worldlines already run in the plane; wrap them back to
                # the cell before tiling, so a bounded orbit draws one closed
                # curve in every copy rather than one long drift.
                x = p[:, 0] % L + m * L
                y = p[:, 1] % b["ly"] + n * b["ly"]
                # A wrap puts a spurious segment across the cell; break there.
                cut = (np.abs(np.diff(x)) > 0.5 * L) | (np.abs(np.diff(y)) > 0.5 * L)
                xs, ys = np.split(x, np.where(cut)[0] + 1), np.split(y, np.where(cut)[0] + 1)
                for xx, yy in zip(xs, ys):
                    ax.plot(xx, yy, color=BLUE if charge[s] > 0 else RED,
                            lw=1.0, zorder=2 if charge[s] > 0 else 3)
    ax.add_patch(Rectangle((0, 0), L, b["ly"], facecolor=CELL, edgecolor="none",
                           zorder=0))
    ax.set_xlim(-0.9 * L, 2.0 * L)
    ax.set_ylim(-0.9 * b["ly"], 2.0 * b["ly"])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "--cartoon":
        out = Path(args[1]) if len(args) > 1 else Path("figure2a.pdf")
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        draw_cartoon(ax)
        ax.set_title("maximal mixing braid", fontsize=14)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
        return 0

    run = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else run / "figure2.pdf"
    b = json.loads((run / "braid.json").read_text())

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 5.3))
    draw_cartoon(ax[0], L=b["lx"])
    ax[0].set_title("maximal mixing braid", fontsize=14)
    draw_orbits(ax[1], b)
    ax[1].set_title("defect orbits", fontsize=14)

    def tex(v, digits=3):
        if not math.isfinite(v):
            return "undefined"
        e = int(math.floor(math.log10(abs(v)))) if v != 0 else 0
        if -3 <= e <= 3:
            return f"{v:.{max(0, digits - e)}f}"
        return f"{v / 10 ** e:.{digits}f}" + r" \times 10^{" + f"{e}" + "}"

    sub = (f"locking {'on' if b['locking'] else 'off'},   "
           f"{b['n_plus']} $+1/2$ and {b['n_minus']} $-1/2$,   "
           f"{b['encounters_per_period']:.2f} encounters per period,   "
           f"$\\tilde T = {tex(b['t_tilde'], 1)}$,   "
           f"$\\tilde h_{{\\max}} = {tex(b['h_tilde_max'])}$,   "
           f"maximal mixing braid: {'yes' if b['verdict'] else 'no'}")
    fig.suptitle(sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
