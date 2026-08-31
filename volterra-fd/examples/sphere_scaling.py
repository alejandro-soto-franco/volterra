#!/usr/bin/env python3
"""Entropy against activity, across a family of sphere runs.

Left, the ensemble topological entropy and the pair separation rate against the
measured Peclet number, on log axes, with a power law fitted to each. Right, the
entropy against the size of the tracer ensemble that measured it, which shows
where the reading settles and so which ensemble the left panel may be read from.

    python sphere_scaling.py <run-dir> [<run-dir> ...] [--sweep FILE] [--out FILE]

Each run directory needs `stats.csv`, `etec.json` and `stretch.json`. The sweep
file is three columns, the run's tag, the tracer count and the rate.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The standing figure style: serif through LaTeX, no grid, every text element
# fully black.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.grid": False,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
})


def measured_pe(run: Path) -> float:
    """The Peclet number the run actually reached, over its second half."""
    with open(run / "stats.csv") as fh:
        rows = list(csv.DictReader(fh))
    tail = [float(r["pe_measured"]) for r in rows[len(rows) // 2:] if r.get("pe_measured")]
    return sum(tail) / len(tail)


def power_law(x, y):
    """Exponent and prefactor of `y = a x^m`, by least squares on the logs."""
    m, c = np.polyfit(np.log(x), np.log(y), 1)
    return m, float(np.exp(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sweep", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pe, etec, ftle, period = [], [], [], []
    by_tag: dict[str, float] = {}
    for run in args.runs:
        pe.append(measured_pe(run))
        by_tag[run.name.removeprefix("sphere_braid_")] = pe[-1]
        etec.append(json.loads((run / "etec.json").read_text())["rate"])
        ftle.append(json.loads((run / "stretch.json").read_text())["rate_median"])
        period.append(json.loads((run / "braid.json").read_text())["period"])
    order = np.argsort(pe)
    pe = np.array(pe)[order]
    etec = np.array(etec)[order]
    ftle = np.array(ftle)[order]
    period = np.array(period, dtype=float)[order]

    n_panels = 3 if args.sweep else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.1))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    grid = np.geomspace(pe.min() * 0.8, pe.max() * 1.25, 64)
    for vals, label, marker, colour in (
        (etec, r"ensemble entropy $h$", "o", "#1b1b1b"),
        (ftle, r"separation rate $\lambda$", "s", "#8a8a8a"),
    ):
        m, a = power_law(pe, vals)
        ax.plot(grid, a * grid**m, "-", color=colour, lw=1.0, zorder=1)
        ax.plot(pe, vals, marker, color=colour, ms=5.5, ls="none", zorder=2,
                label=rf"{label}, $\propto \mathrm{{Pe}}^{{{m:.2f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Pe}$")
    ax.set_ylabel(r"rate per unit time")
    ax.set_title(r"Entropy against activity")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]
    m, _ = power_law(pe, period)
    ax.plot(pe, etec * period, "o-", color="#1b1b1b", ms=5.5, lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel(r"$\mathrm{Pe}$")
    ax.set_ylabel(r"$hT$ per defect orbit")
    ax.set_title(r"Stretching over one orbit")
    ax.text(0.05, 0.12,
            rf"$T \propto \mathrm{{Pe}}^{{{m:.2f}}}$", transform=ax.transAxes,
            fontsize=9)

    if args.sweep:
        ax = axes[2]
        series: dict[str, list[tuple[int, float]]] = {}
        with open(args.sweep) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 3:
                    continue
                series.setdefault(parts[0], []).append((int(parts[1]), float(parts[2])))
        # Ordered by activity, so the shading runs with it.
        ordered = sorted(series.items(), key=lambda kv: by_tag.get(kv[0], float("inf")))
        shades = np.linspace(0.0, 0.66, max(len(ordered), 1))
        for (tag, pts), shade in zip(ordered, shades):
            pts.sort()
            n = [p[0] for p in pts]
            r = [p[1] for p in pts]
            label = (rf"$\mathrm{{Pe}} = {by_tag[tag]:.2f}$" if tag in by_tag
                     else tag.replace("_", r"\_"))
            ax.plot(n, np.array(r) / r[-1], "o-", ms=4.5, lw=1.0,
                    color=str(shade), label=label)
        ax.axhline(1.0, color="#000000", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"tracers")
        ax.set_ylabel(r"$h$ relative to the largest ensemble")
        ax.set_title(r"Ensemble convergence")
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = args.out or Path("output/scaling.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")

    print(f"\n{'Pe':>8} {'h':>12} {'lambda':>12} {'h/lambda':>9}")
    for p, e, f in zip(pe, etec, ftle):
        print(f"{p:8.4f} {e:12.4e} {f:12.4e} {e / f:9.3f}")
    for vals, name in ((etec, "h"), (ftle, "lambda"), (period, "T"),
                       (etec * period, "hT")):
        m, a = power_law(pe, vals)
        print(f"{name}: exponent {m:+.4f}, prefactor {a:.4e}")


if __name__ == "__main__":
    main()
