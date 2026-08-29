"""Nine-panel film of a periodic active-nematic run, fields and rates together.

    row 1, the nematic and the flow it drives
      A  order parameter, director glyphs, defect markers
      E  Q isocontours: the zero sets of Qxx and Qxy, which CROSS at every
         defect, over isolines of S
      V  velocity, with the flow speed under the arrows

    row 2, what the active stress makes and what it mixes
      C  vorticity
      W  RMS vorticity ACCUMULATED over the run, where the persistent gyres sit
      M  passive tracers, coloured by the column each started in

    row 3, the two director rotation rates and the discriminator
      RA advective rotation rate, the part of dn/dt the flow turns
      RF fracturing rotation rate, the part the molecular field turns, which is
         the term enhanced nematic locking switches off in the ordered bulk
      G  defect count and RMS velocity against time, whose periodicity is what
         separates the periodic orbit of Phys. Rev. E 109, 014606 (2024) from
         the chaotic state

THE LAYOUT IS FIXED. Every run is filmed in this arrangement so two films can be
put side by side, which is the whole point of the standard-against-enhanced pair
at one `ell_a`.

RA and RF share one colour scale, taken from the advective rate. On the same
scale a run under enhanced locking draws RF blank, which is the measurement of
arXiv:2506.20996 as a picture rather than as a median in a log.

Vorticity is centrally differenced from `u` on the torus. Every colour scale is
fixed across the film, from a prepass over the frames that will be drawn: a
scale that follows the frame makes a film where no two instants compare.

    uv run --with numpy,matplotlib panels_periodic_video.py <run> <out.mp4> [--stride N]
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

VOLTERRA_S = LinearSegmentedColormap.from_list("volterra_s", ["#1a9a3a", "#ffffff"])
VORT = LinearSegmentedColormap.from_list("vort", ["#1f4e9c", "#ffffff", "#d81e05"])
SPEEDMAP = LinearSegmentedColormap.from_list("speed", ["#ffffff", "#1f4e9c", "#0b1c3a"])
DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#1f4e9c", "#ffffff", "#d81e05"])
# Four bands, not a gradient. A continuous colouring over sixty columns is a
# speckle from the first frame and says nothing about how far the mixing has
# gone; four blocks stay legible until they are stirred through.
NBAND = 4
TRACER = ListedColormap(["#d81e05", "#c98a00", "#1a9a3a", "#1f4e9c"])
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "axes.grid": False,
    "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000",
    "ytick.color": "#000000", "axes.edgecolor": "#000000", "figure.dpi": 120,
    "axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

run, out_mp4 = Path(sys.argv[1]), sys.argv[2]
opt = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[3:] if "=" in a}
stride = int(opt.get("--stride", 1))
# Skip the quench when accumulating and when fixing scales. The field is still
# condensing before this and its vorticity is cell-scale noise, which would
# dominate an accumulation started at t = 0 and hide the gyres the panel exists
# to show.
start_t = float(opt.get("--start", 1.0))
png = Path(out_mp4).with_suffix("")
png.mkdir(parents=True, exist_ok=True)

cfg = json.loads((run / "config.json").read_text())
p = cfg["params"]
d = cfg["dimensionless"]
lx, ly = p["lx"], p["ly"]
dt = p["dt"]
S0 = p["s0"]
ell_a = math.sqrt(p["k_elastic"] / p["zeta"])
locking = p.get("locking")
NSIDE = int(cfg.get("tracers", 0))

steps = sorted(int(f.stem.split("_")[1]) for f in run.glob("q_*.npy"))[::stride]
times = np.array(steps) * dt
if not steps:
    sys.exit(f"no q_*.npy frames in {run}")


def load(prefix, step, cols=1):
    a = np.load(run / f"{prefix}_{step:08}.npy")
    return a.reshape(lx, ly, cols) if cols > 1 else a.reshape(lx, ly)


def vorticity(u):
    """`omega = d_x u_y - d_y u_x`, centrally differenced on the torus."""
    return (0.5 * (np.roll(u[:, :, 1], -1, 0) - np.roll(u[:, :, 1], 1, 0))
            - 0.5 * (np.roll(u[:, :, 0], -1, 1) - np.roll(u[:, :, 0], 1, 1)))


# ===================== stats
rows = [l.split(",") for l in (run / "stats.csv").read_text().splitlines()]
head = rows[0]
S = {n: np.array([float(r[i]) if r[i] not in ("", "NaN") else np.nan for r in rows[1:]])
     for i, n in enumerate(head)}
_drows = [[float(v) for v in l.split(",")]
          for l in (run / "defects.csv").read_text().splitlines()[1:] if l.strip()]
dfc = np.array(_drows) if _drows else np.zeros((0, 5))


def texnum(v, digits=2):
    """`v` as LaTeX maths, with an exponent where one is warranted."""
    if v == 0.0 or not np.isfinite(v):
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if -3 <= e <= 3:
        return f"{v:.{max(0, digits - e)}f}"
    return f"{v / 10 ** e:.{digits}f}" + r" \times 10^{" + f"{e}" + "}"

# ===================== fixed scales
w_acc = np.zeros((lx, ly))
w_cnt = 0
sp_hi, w_hi, wa_hi = [], [], []
for st in steps:
    u = load("u", st, 2)
    w = vorticity(u)
    sp_hi.append(np.percentile(np.hypot(u[:, :, 0], u[:, :, 1]), 98))
    w_hi.append(np.percentile(np.abs(w), 98))
    wa_hi.append(np.percentile(np.abs(load("wa", st)), 98))
    if st * dt >= start_t:
        w_acc += w * w
        w_cnt += 1
V_HI = float(np.max(sp_hi))
W_HI = float(np.max(w_hi))
# One scale for both rates. The advective one sets it, so a collapsed fracturing
# rate reads as a blank panel rather than as a rescaled copy of its own noise.
R_HI = float(np.max(wa_hi))
# An accumulated RMS is bounded away from zero everywhere the flow ever moved,
# so anchoring it at zero paints the whole panel at the top of the scale and the
# gyres it exists to show disappear. The window is the field's own decile range.
_wr = np.sqrt(w_acc / max(w_cnt, 1))
WRMS_LO = float(np.percentile(_wr, 10)) if w_cnt else 0.0
WRMS_HI = float(np.percentile(_wr, 98)) if w_cnt else 1.0
print(f"  scales: |u| to {V_HI:.1f}, omega to +-{W_HI:.1f}, "
      f"rates to +-{R_HI:.3f}, rms omega {WRMS_LO:.1f} to {WRMS_HI:.1f} "
      f"over {w_cnt} frames")

# ===================== glyphs
gx, gy = np.meshgrid(np.arange(2, lx, 4) + 0.5, np.arange(2, ly, 4) + 0.5,
                     indexing="ij")
GI, GJ = gx.astype(int), gy.astype(int)
GLEN = 3.0
qx, qy = np.meshgrid(np.arange(0, lx, 5) + 0.5, np.arange(0, ly, 5) + 0.5,
                     indexing="ij")
QI, QJ = qx.astype(int), qy.astype(int)
XC = np.arange(lx) + 0.5
YC = np.arange(ly) + 0.5
EXT = [0, lx, 0, ly]


def dress(ax, title):
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title, fontsize=15)


def show(ax, fig, f, cmap, vmin, vmax, label):
    im = ax.imshow(f.T, origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", zorder=0)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.set_title(
        label, fontsize=13, pad=7)


if locking:
    HEAD = "Enhanced nematic locking on the torus"
else:
    HEAD = "Beris--Edwards on the torus"
SEQ = r"\sqrt{2}" if abs(S0 - math.sqrt(2)) < 1e-9 else f"{S0:g}"
SUP = (f"{HEAD}\n"
       f"$L = {lx}$,   $\\ell_a = {ell_a:.3g}$,   $\\ell_n = {cfg['ell_n']:.3g}$,   "
       f"$\\mathrm{{Re}} = {d['re']:g}$,   $\\tilde\\gamma = {d['gamma_tilde']:g}$,   "
       f"$\\tilde C = {d['c_tilde']:g}$,   $\\lambda = {p['lambda']:g}$,   "
       f"$S_{{\\mathrm{{eq}}}} = {SEQ}$,   $\\Delta t = {dt:.2e}$,   "
       f"seed {cfg['seed']},   {cfg['ic']} initial condition")
SUP = re.sub(r"e-0?(\d+)", r"\\times 10^{-\1}", SUP)

# Both traces are scaled on the SETTLED part. The quench puts several hundred
# defects on the lattice in the first few hundredths of a time unit, and an axis
# that fits that spike flattens everything the film is about.
_st = S["t"] >= start_t
NPLUS_HI = max(4.0, float(np.nanmax(S["n_plus"][_st])) * 1.2)
RMSU_HI = float(np.nanmax(S["rms_u"][_st])) * 1.15

w_acc = np.zeros((lx, ly))
w_cnt = 0
for k, st in enumerate(steps):
    t = times[k]
    q = load("q", st, 2)
    u = load("u", st, 2)
    wa = load("wa", st)
    wf = load("wf", st)
    w = vorticity(u)
    Sf = 2.0 * np.hypot(q[:, :, 0], q[:, :, 1])
    now = dfc[np.isclose(dfc[:, 0], st)]

    fig = plt.figure(figsize=(21.0, 16.6))
    gs = fig.add_gridspec(3, 3, hspace=0.24, wspace=0.30,
                          left=0.045, right=0.955, top=0.930, bottom=0.040)
    A = fig.add_subplot(gs[0, 0]); E = fig.add_subplot(gs[0, 1])
    V = fig.add_subplot(gs[0, 2])
    C = fig.add_subplot(gs[1, 0]); W = fig.add_subplot(gs[1, 1])
    M = fig.add_subplot(gs[1, 2])
    RA = fig.add_subplot(gs[2, 0]); RF = fig.add_subplot(gs[2, 1])
    G = fig.add_subplot(gs[2, 2])

    # A: order parameter, director, defects
    show(A, fig, Sf, VOLTERRA_S, 0.0, S0, "$S$")
    th = 0.5 * np.arctan2(q[GI, GJ, 1], q[GI, GJ, 0])
    ctr = np.stack([XC[GI], YC[GJ]], -1).reshape(-1, 2)
    dv = np.stack([np.cos(th), np.sin(th)], -1).reshape(-1, 2) * (GLEN / 2)
    A.add_collection(LineCollection(np.stack([ctr - dv, ctr + dv], 1),
                                    colors="#000000", linewidths=0.5, zorder=2))
    pos, neg = now[now[:, 4] > 0], now[now[:, 4] < 0]
    A.scatter(pos[:, 2], pos[:, 3], s=46, facecolor="#d81e05", edgecolor="#000000",
              linewidths=0.7, zorder=6)
    A.scatter(neg[:, 2], neg[:, 3], s=42, marker="^", facecolor="#1f4e9c",
              edgecolor="#000000", linewidths=0.6, zorder=5)
    dress(A, "director field")
    A.text(0.02, 0.98, f"$t = {t:.2f}$", transform=A.transAxes, va="top", ha="left",
           fontsize=11, bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.8,
                                  pad=1.6))

    # E: the zero sets of the two Q components, which cross at every defect.
    # A +-1/2 core is where BOTH components vanish, so the crossings of the two
    # curves are the defect set, found without a detector and without a merge
    # radius. The S isolines behind them are the cores those crossings sit in.
    E.contour(XC, YC, Sf.T, levels=np.linspace(0.15 * S0, 0.9 * S0, 6),
              colors="#9a9a9a", linewidths=0.5, zorder=1)
    E.contour(XC, YC, q[:, :, 0].T, levels=[0.0], colors="#d81e05", linewidths=1.0,
              zorder=3)
    E.contour(XC, YC, q[:, :, 1].T, levels=[0.0], colors="#1f4e9c", linewidths=1.0,
              zorder=3)
    E.scatter(pos[:, 2], pos[:, 3], s=44, facecolor="#ffffff", edgecolor="#000000",
              linewidths=0.8, zorder=6)
    E.scatter(neg[:, 2], neg[:, 3], s=42, marker="^", facecolor="#ffffff",
              edgecolor="#000000", linewidths=0.8, zorder=6)
    dress(E, "$Q$ isocontours")
    E.plot([], [], color="#d81e05", lw=1.0, label="$Q_{xx} = 0$")
    E.plot([], [], color="#1f4e9c", lw=1.0, label="$Q_{xy} = 0$")
    E.plot([], [], color="#9a9a9a", lw=0.5, label=r"$S$ isolines, 0.15--0.9 $S_0$")
    E.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.85,
             edgecolor="none", handlelength=1.3, borderpad=0.25, labelspacing=0.25)

    # V: velocity
    show(V, fig, np.hypot(u[:, :, 0], u[:, :, 1]), SPEEDMAP, 0.0, V_HI, "$|u|$")
    V.quiver(XC[QI], YC[QJ], u[QI, QJ, 0], u[QI, QJ, 1], color="#000000",
             width=0.004, scale=18 * V_HI, zorder=3)
    dress(V, "velocity field")

    # C: vorticity
    show(C, fig, w, VORT, -W_HI, W_HI, r"$\omega$")
    dress(C, "vorticity field")

    # W: RMS vorticity accumulated over the settled frames drawn so far
    if t >= start_t:
        w_acc += w * w
        w_cnt += 1
    show(W, fig, np.sqrt(w_acc / max(w_cnt, 1)), SPEEDMAP, WRMS_LO, WRMS_HI,
         r"$\omega_{\mathrm{rms}}$")
    dress(W, "RMS vorticity")
    # A static string, so usetex renders it once for the whole film.
    W.text(0.02, 0.98, f"accumulated from $t = {start_t:g}$", transform=W.transAxes,
           va="top", ha="left", fontsize=10,
           bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.8, pad=1.6))

    # M: passive tracers, coloured by the column each started in
    tf = run / f"tracer_{st:08}.csv"
    if tf.exists():
        tp = np.array([[float(v) for v in l.split(",")]
                       for l in tf.read_text().splitlines()[1:]])
        band = (tp[:, 0] * NBAND // max(NSIDE, 1)).clip(0, NBAND - 1)
        M.scatter(tp[:, 1], tp[:, 2], c=band, cmap=TRACER, vmin=-0.5,
                  vmax=NBAND - 0.5, s=2.6, linewidths=0, zorder=2)
    dress(M, "passive tracers")

    # RA, RF: the two rotation rates, on ONE scale
    show(RA, fig, wa, DIVERGE, -R_HI, R_HI, r"$\omega_A$")
    dress(RA, "advective rotation rate")
    show(RF, fig, wf, DIVERGE, -R_HI, R_HI, r"$\omega_F$")
    dress(RF, "fracturing rotation rate")
    RF.text(0.02, 0.98,
            r"median $|\omega_F| = " + texnum(float(np.median(np.abs(wf)))) + "$",
            transform=RF.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.8, pad=1.6))

    # G: the discriminator
    G.plot(S["t"], S["n_plus"], color="#1f4e9c", lw=1.2, label="$n_{+1/2}$")
    G.set_xlim(times[0], times[-1])
    G.set_ylim(0, NPLUS_HI)
    G.set_xlabel("$t$")
    G.set_ylabel("$n_{+1/2}$")
    G2 = G.twinx()
    G2.plot(S["t"], S["rms_u"], color="#d81e05", lw=1.2, label=r"$u_{\mathrm{rms}}$")
    G2.set_ylabel(r"$u_{\mathrm{rms}}$")
    G2.set_ylim(0, RMSU_HI)
    for ax in (G, G2):
        ax.tick_params(colors="#000000")
        ax.yaxis.label.set_color("#000000")
    G.axvline(t, color="#000000", lw=0.8, ls=":")
    G.set_title("defect count and RMS velocity", fontsize=15)
    h1, l1 = G.get_legend_handles_labels()
    h2, l2 = G2.get_legend_handles_labels()
    G.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10, framealpha=0.92,
             borderpad=0.5, handlelength=1.9, labelcolor="#000000")

    for ax in (G, G2):
        for sp in ax.spines.values():
            sp.set_color("#000000")

    fig.suptitle(SUP, fontsize=16)
    # No bbox_inches="tight": it resizes every frame to its own content, which
    # makes the film jitter and the layout read as cropped.
    fig.savefig(png / f"panel_{k:05d}.png")
    plt.close(fig)
    if (k + 1) % 25 == 0:
        print(f"  {k + 1}/{len(steps)} frames")

print(f"{len(steps)} frames -> {png}")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "14",
                "-pattern_type", "glob", "-i", str(png / "panel_*.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4], check=True)
print(out_mp4)
