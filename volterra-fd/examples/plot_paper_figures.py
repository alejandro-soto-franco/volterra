"""Figures 3 and 4 of Mitchell et al. (2024), reproduced.

    fig3  RMS velocity against integration time, with two director-field
          snapshots inset and arrowed to the instants they are taken at. Their
          panel a is the chaotic case, panel b the confined one, where the two
          snapshots are a period apart and show the same field.
    fig4  Contour length of an advected material line against time, on a semilog
          axis whose slope is the topological entropy `h`, with the final curve
          inset.

Both take one run directory per panel.

    uv run --with numpy,matplotlib plot_paper_figures.py fig3 <runA> <runB> <out>
    uv run --with numpy,matplotlib plot_paper_figures.py fig4 <runA> <runB> <out>
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "axes.grid": False,
    "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000",
    "ytick.color": "#000000", "axes.edgecolor": "#000000",
    "axes.labelsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
})
BLUE = "#1f4e9c"
RED = "#d81e05"


def load(run):
    cfg = json.loads((run / "config.json").read_text())
    rows = [l.split(",") for l in (run / "stats.csv").read_text().splitlines()]
    head = rows[0]
    s = {n: np.array([float(r[i]) if r[i] not in ("", "NaN") else np.nan
                      for r in rows[1:]]) for i, n in enumerate(head)}
    return cfg, s


def ell_a(cfg):
    p = cfg["params"]
    return math.sqrt(p["k_elastic"] / p["zeta"])


def director_inset(fig, host, rect, run, step, cfg):
    """Director field and defects at one frame, drawn into an inset."""
    p = cfg["params"]
    lx, ly = p["lx"], p["ly"]
    q = np.load(run / f"q_{step:08}.npy").reshape(lx, ly, 2)
    ax = host.inset_axes(rect)
    gi, gj = np.meshgrid(np.arange(2, lx, 4), np.arange(2, ly, 4), indexing="ij")
    th = 0.5 * np.arctan2(q[gi, gj, 1], q[gi, gj, 0])
    ctr = np.stack([gi + 0.5, gj + 0.5], -1).reshape(-1, 2).astype(float)
    d = np.stack([np.cos(th), np.sin(th)], -1).reshape(-1, 2) * 1.5
    ax.add_collection(LineCollection(np.stack([ctr - d, ctr + d], 1),
                                     colors="#000000", linewidths=0.35))
    drows = [[float(v) for v in l.split(",")]
             for l in (run / "defects.csv").read_text().splitlines()[1:] if l.strip()]
    dfc = np.array(drows) if drows else np.zeros((0, 5))
    now = dfc[np.isclose(dfc[:, 0], step)] if len(dfc) else dfc
    if len(now):
        pos, neg = now[now[:, 4] > 0], now[now[:, 4] < 0]
        ax.scatter(pos[:, 2], pos[:, 3], s=14, c=BLUE, linewidths=0, zorder=4)
        ax.scatter(neg[:, 2], neg[:, 3], s=14, c=RED, marker="^", linewidths=0,
                   zorder=4)
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#000000")
    return ax


def fig3(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs):
        cfg, s = load(run)
        ax.plot(s["t"], s["rms_u"], color="#000000", lw=0.8)
        ax.set_xlabel("integration time $t$")
        ax.set_ylabel(r"RMS velocity $u_{\mathrm{rms}}$")
        ax.set_xlim(s["t"][0], s["t"][-1])
        ax.set_ylim(0, float(np.nanmax(s["rms_u"])) * 1.12)
        ax.set_title(rf"$\ell_a = {ell_a(cfg):.1f}$", fontsize=14)

        # Two snapshots from the developed state, a third of the run apart.
        frames = sorted(int(f.stem.split("_")[1]) for f in run.glob("q_*.npy"))
        if len(frames) >= 4:
            dt = cfg["params"]["dt"]
            pick = [frames[len(frames) // 2], frames[(5 * len(frames)) // 6]]
            for k, (st, rect) in enumerate(zip(pick, [(0.06, 0.06, 0.36, 0.44),
                                                      (0.56, 0.06, 0.36, 0.44)])):
                director_inset(fig, ax, rect, run, st, cfg)
                t = st * dt
                ax.annotate("", xy=(t, np.interp(t, s["t"], s["rms_u"])),
                            xytext=(0.24 + 0.5 * k, 0.52), textcoords=ax.transAxes,
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig4(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs):
        cfg, _ = load(run)
        lf = run / "line_lengths.csv"
        if not lf.exists():
            ax.set_title("no material lines in this run", fontsize=12)
            continue
        rows = np.array([[float(v) for v in l.split(",")]
                         for l in lf.read_text().splitlines()[1:] if l.strip()])
        ent = json.loads((run / "entropy.json").read_text()) \
            if (run / "entropy.json").exists() else None
        for i in sorted(set(rows[:, 0].astype(int))):
            m = rows[:, 0] == i
            ax.semilogy(rows[m, 1], rows[m, 2], lw=1.0)
        if ent:
            t0, t1 = ent["fit_window"]
            tt = np.linspace(t0, t1, 2)
            # Anchor the fitted line on the first curve at the window start.
            m = rows[:, 0] == 0
            l0 = float(np.interp(t0, rows[m, 1], rows[m, 2]))
            ax.semilogy(tt, l0 * np.exp(ent["h"] * (tt - t0)), color="#000000",
                        lw=1.4, ls="--")
            ax.text(0.03, 0.95,
                    rf"$h = {ent['h']:.3f} \pm {ent['h_sem']:.3f}$" "\n"
                    rf"$\tilde h = {ent['h_tilde']:.3e}$".replace("e-0", r"\times 10^{-")
                    + ("}" if "e-0" in f"{ent['h_tilde']:.3e}" else ""),
                    transform=ax.transAxes, va="top", ha="left", fontsize=11)
        ax.set_xlabel("integration time $t$")
        ax.set_ylabel("contour length")
        ax.set_title(rf"$\ell_a = {ell_a(cfg):.1f}$", fontsize=14)

        # The final advected curve, inset.
        pts = sorted(run.glob("line_*.csv"))
        if pts:
            p = np.array([[float(v) for v in l.split(",")]
                          for l in pts[-1].read_text().splitlines()[1:] if l.strip()])
            axi = ax.inset_axes((0.58, 0.08, 0.38, 0.42))
            first = p[p[:, 0] == 0]
            axi.plot(first[:, 1], first[:, 2], color=BLUE, lw=0.25)
            axi.set_xlim(0, cfg["params"]["lx"])
            axi.set_ylim(0, cfg["params"]["ly"])
            axi.set_aspect("equal")
            axi.set_xticks([])
            axi.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    which, *rest = sys.argv[1:]
    runs = [Path(r) for r in rest[:-1]]
    out = Path(rest[-1])
    {"fig3": fig3, "fig4": fig4}[which](runs, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
