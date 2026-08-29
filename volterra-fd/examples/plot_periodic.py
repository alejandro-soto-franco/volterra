#!/usr/bin/env python3
"""Figures for a periodic active-nematic run.

Reproduces the two figure panels Mitchell, Sabbir, Geumhan, Smith, Klein and
Beller, Phys. Rev. E 109, 014606 (2024) report: the defect orbits of their
Fig. 2b, and the RMS velocity trace of their Fig. 3.

Usage:  uv run --with numpy,matplotlib plot_periodic.py <run-dir> [<out.pdf>]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.grid": False,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.titlecolor": "#000000",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(run):
    cfg = json.loads((run / "config.json").read_text())
    rows = [l.split(",") for l in (run / "stats.csv").read_text().splitlines()]
    head = rows[0]
    s = {n: np.array([float(r[i]) if r[i] not in ("", "NaN") else np.nan
                      for r in rows[1:]]) for i, n in enumerate(head)}
    drows = [l.split(",") for l in (run / "defects.csv").read_text().splitlines()][1:]
    d = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in drows])
    return cfg, s, d


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "figure.pdf"
    cfg, s, d = load(run)
    p = cfg["params"]
    lx = p["lx"]
    ell_a = (p["k_elastic"] / p["zeta"]) ** 0.5

    # Late window, where the state has settled.
    t_cut = s["t"][-1] - (s["t"][-1] - s["t"][0]) / 3.0
    late = d[:, 0] >= t_cut

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.0))

    plus = d[late & (d[:, 3] > 0)]
    minus = d[late & (d[:, 3] < 0)]
    ax[0].scatter(plus[:, 1], plus[:, 2], s=1.2, c="#1f4e9c", marker="o",
                  label=r"$+1/2$", linewidths=0)
    ax[0].scatter(minus[:, 1], minus[:, 2], s=1.2, c="#b02020", marker="^",
                  label=r"$-1/2$", linewidths=0)
    ax[0].set_xlim(0, lx)
    ax[0].set_ylim(0, lx)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    ax[0].set_title(rf"Defect orbits, $\ell_a = {ell_a:.1f}$", fontsize=11)
    leg = ax[0].legend(frameon=False, markerscale=6, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color("#000000")

    m = s["t"] >= t_cut
    ax[1].plot(s["t"][m], s["rms_u"][m], color="#000000", lw=0.8)
    ax[1].set_xlabel(r"integration time $t$")
    ax[1].set_ylabel(r"$u_{\mathrm{rms}}$")
    ax[1].set_title("Root-mean-square velocity", fontsize=11)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
