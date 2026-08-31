#!/usr/bin/env python3
"""Film of the defect braid on a sphere.

Left: the sphere itself, shaded by the scalar order parameter, with the
director drawn as short rods and the four `+1/2` defects as red dots. A filled
dot is on the near face and an open dot is on the far one, so a defect passing
round the back stays legible. Each defect trails its own past path.

Right, top: the braid diagram. The strands are the defects' positions in the
stereographic chart the braid word is read in, plotted against time, so a
crossing in the diagram is a generator in the word. The vertical line is the
frame on the left.

Right, bottom: the E-tec rate the tracer ensemble measures over the same
window, against the braid above it.

    python sphere_braid_video.py <run-dir> [--out FILE] [--stride N]

The run must already have a `braid.json` from `sphere_braid_report`.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

# The standing film style, the same block the CGPO videos use. usetex is NOT
# set here: it comes from ~/.config/matplotlib/matplotlibrc, so every frame goes
# through LaTeX and sets in Latin Modern. Setting it False falls back to the
# DejaVu face, which is not the house serif.
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.grid": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
                           r"\usepackage{amsfonts}\usepackage{lmodern}",
    "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000",
    "ytick.color": "#000000", "axes.edgecolor": "#000000", "figure.dpi": 150,
    "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
})

BASE_DPI = 110    # the resolution the raster and glyph counts below are set for
PX = 620          # sphere raster, pixels across, scaled by the requested dpi
N_GLYPH = 900     # director rods over the whole sphere, culled to the visible face


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def vertex_frames(verts, tris):
    """The per-vertex normal and `e1`, matching the solver's own convention.

    The normal is area-weighted over incident triangles; `e1` is `x` projected
    into the tangent plane, or `y` where the normal is too close to `x` for that
    projection to be stable. `e2` is `n x e1`.
    """
    n = np.zeros_like(verts)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tris[:, k], cr)
    n = unit(n)
    ref = np.where((np.abs(n[:, 0]) < 0.9)[:, None],
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    e1 = unit(ref - (n * ref).sum(1)[:, None] * n)
    e2 = np.cross(n, e1)
    return n, e1, e2


def fibonacci(m):
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--dpi", type=int, default=BASE_DPI,
                    help="output resolution; 256 gives 3840 across, which is 4K")
    args = ap.parse_args()
    run = args.run
    out = args.out or run / "braid.mp4"

    # The sphere raster and the glyph count follow the output resolution, or a
    # 4K frame carries an upscaled sphere and the same sparse rods.
    scale = args.dpi / BASE_DPI
    raster = int(round(PX * scale))
    n_glyph = int(round(N_GLYPH * scale ** 1.5))

    mesh = json.loads((run / "mesh.json").read_text())
    verts = np.asarray(mesh["vertices"], float)
    tris = np.asarray(mesh["triangles"], int)
    meta = json.loads((run / "meta.json").read_text())
    braid = json.loads((run / "braid.json").read_text())

    times = np.asarray(braid["times"], float)
    W = np.asarray(braid["worldlines"], float)        # (strand, frame, 3)
    proj = np.asarray(braid["projected"], float)      # (strand, frame, 2)
    crossings = braid["crossings"]
    word = braid["word"]
    pole = np.asarray(braid["pole"], float)
    n_strand = W.shape[0]

    # The snapshot index for each braid frame. Snapshots are on a uniform grid
    # in step number; the braid window is a contiguous slice of it.
    snaps = sorted(int(p.stem.split("_")[1]) for p in run.glob("q_*.npy"))
    dt = meta["dt"]
    step_of = {s: s * dt for s in snaps}
    snap_for = []
    for t in times:
        k = min(snaps, key=lambda s: abs(step_of[s] - t))
        snap_for.append(k)

    # Camera: look along the chart's deleted point, so the strands spread across
    # the visible face rather than piling up near the puncture.
    d = unit(-pole)                      # eye -> origin
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(tmp, d))
    v = np.cross(d, u)

    # Pixel -> vertex, computed once: neither the sphere nor the camera moves.
    g = np.linspace(-1.0, 1.0, raster)
    X, Y = np.meshgrid(g, -g)
    R2 = X * X + Y * Y
    disc = R2 <= 1.0
    Z = -np.sqrt(np.maximum(0.0, 1.0 - R2))
    P = (X[..., None] * u + Y[..., None] * v + Z[..., None] * d)
    tree = cKDTree(verts)
    idx = np.zeros(X.shape, int)
    idx[disc] = tree.query(P[disc])[1]

    # Lambertian shading from over the viewer's shoulder, so the rim reads as a
    # sphere rather than as a flat disc.
    light = unit(np.array([-0.4, 0.5, -1.0]) @ np.stack([u, v, d]))
    shade = np.clip((P * light).sum(-1), 0.0, 1.0) ** 0.6
    shade = 0.45 + 0.55 * shade

    # Glyph seats, culled to the face that is comfortably visible.
    seats = fibonacci(n_glyph)
    keep = (seats * d).sum(1) < -0.12
    seats = seats[keep]
    seat_v = tree.query(seats)[1]
    sx = (seats * u).sum(1)
    sy = (seats * v).sum(1)

    _, e1, e2 = vertex_frames(verts, tris)

    fig = plt.figure(figsize=(15.0, 8.0), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.62],
                          left=0.02, right=0.975, top=0.93, bottom=0.07,
                          wspace=0.16, hspace=0.28)
    AX = fig.add_subplot(gs[:, 0])
    AB = fig.add_subplot(gs[0, 1])
    AS = fig.add_subplot(gs[1, 1])

    AX.set_xlim(-1.05, 1.05)
    AX.set_ylim(-1.05, 1.05)
    AX.set_aspect("equal")
    AX.axis("off")

    # Braid diagram, drawn whole; a cursor marks the frame.
    xs = proj[:, :, 0]
    lo, hi = np.percentile(xs, 1), np.percentile(xs, 99)
    pad = 0.12 * (hi - lo)
    cols = ["#c0392b", "#2471a3", "#1e8449", "#8e44ad", "#b7950b", "#17a589"]
    for s in range(n_strand):
        AB.plot(times, xs[s], color=cols[s % len(cols)], lw=1.5, zorder=3)
    for fr, code in crossings:
        AB.axvline(times[fr], color="#00000022", lw=6, zorder=1)
    AB.set_xlim(times[0], times[-1])
    AB.set_ylim(lo - pad, hi + pad)
    AB.set_xlabel("time")
    AB.set_ylabel("position in the chart")
    AB.set_title(f"braid diagram: {len(word)} generators on {n_strand} strands",
                 fontsize=11)
    AB.spines[["top", "right"]].set_visible(False)
    cursor = AB.axvline(times[0], color="k", lw=1.4, zorder=5)

    # The E-tec rate: what the tracer ensemble is doing while the defects braid.
    # Four defects see less than four hundred tracers do, and the two panels
    # side by side are what shows it.
    etec_path = run / "etec.json"
    if etec_path.exists():
        ej = json.loads(etec_path.read_text())
        conv = np.asarray(ej.get("convergence", []), dtype=float)
        # The opening tenth is the band finding the unstable direction, and its
        # rate is meaningless. Drop it from the curve rather than letting the
        # y-limits clip it, so the line starts where the measurement does and
        # runs to the end of the window.
        k = len(conv) // 10
        AS.plot(conv[k:, 0] + times[0], conv[k:, 1], color="#1a1a1a", lw=1.4,
                zorder=3)
        AS.axhline(ej["rate"], color="#8a8a8a", ls="--", lw=1.0, zorder=2)
        shown = conv[k:, 1]
        lo_, hi_ = float(shown.min()), float(shown.max())
        pad_ = 0.25 * max(hi_ - lo_, 1e-12)
        AS.set_ylim(lo_ - pad_, hi_ + pad_)
        AS.text(0.985, 0.90, f"{ej['rate']:.3e}", fontsize=11, va="center",
                ha="right", color="#000000", transform=AS.transAxes)
        AS.set_xlim(times[0], times[-1])
        AS.set_ylabel("E-tec rate")
        AS.set_title(f"ensemble entropy rate, {ej['tracers']} tracers", fontsize=11)
    else:
        AS.text(0.5, 0.5, "no etec.json", ha="center", va="center",
                transform=AS.transAxes, fontsize=12)
        AS.set_xlim(times[0], times[-1])
    AS.set_xlabel("time")
    AS.spines[["top", "right"]].set_visible(False)
    cursor2 = AS.axvline(times[0], color="k", lw=1.4)

    # The `--pe` flag is not the Peclet number the run reaches; `stats.csv`
    # records the measured one, and that is what a title should carry.
    pe = float("nan")
    stats = run / "stats.csv"
    if stats.exists():
        import csv as _csv
        rows_ = list(_csv.DictReader(open(stats)))
        tail_ = [float(r["pe_measured"]) for r in rows_[len(rows_) // 2:]
                 if r.get("pe_measured")]
        if tail_:
            pe = sum(tail_) / len(tail_)
    sup = fig.suptitle("", fontsize=13)

    im = AX.imshow(np.zeros((raster, raster, 4)), extent=(-1, 1, -1, 1),
                   origin="upper", interpolation="bilinear", zorder=1)
    rods = LineCollection([], colors="#1a1a1a", linewidths=1.0, alpha=0.75, zorder=3)
    AX.add_collection(rods)
    trails = [AX.plot([], [], color=cols[s % len(cols)], lw=1.1, alpha=0.7,
                      zorder=4)[0] for s in range(n_strand)]
    near = AX.scatter([], [], s=95, facecolors="#d62828", edgecolors="k",
                      linewidths=0.9, zorder=6)
    far = AX.scatter([], [], s=95, facecolors="none", edgecolors="#d62828",
                     linewidths=1.6, zorder=5)

    frames = range(0, len(times), args.stride)
    tmpdir = run / "_frames"
    tmpdir.mkdir(exist_ok=True)
    for n, fi in enumerate(frames):
        q = np.load(run / f"q_{snap_for[fi]:06d}.npy")
        amp = np.hypot(q[:, 0], q[:, 1])
        S = 2.0 * amp

        # Order parameter: white where ordered, dark green in a core.
        val = np.clip(S / max(1e-9, np.percentile(S, 99)), 0.0, 1.0)
        f = val[idx]
        rgb = np.stack([0.04 + 0.96 * f ** 1.4,
                        0.24 + 0.76 * f ** 0.9,
                        0.04 + 0.96 * f ** 1.4], -1)
        rgb *= shade[..., None]
        rgba = np.concatenate([rgb, disc[..., None].astype(float)], -1)
        im.set_data(np.clip(rgba, 0, 1))

        # Director rods: half the spin-2 angle, in the vertex's own frame,
        # pushed forward to R^3 and then projected to the screen.
        th = 0.5 * np.arctan2(q[seat_v, 1], q[seat_v, 0])
        dir3 = np.cos(th)[:, None] * e1[seat_v] + np.sin(th)[:, None] * e2[seat_v]
        dx = (dir3 * u).sum(1)
        dy = (dir3 * v).sum(1)
        L = 0.021
        segs = np.stack([np.stack([sx - L * dx, sy - L * dy], 1),
                         np.stack([sx + L * dx, sy + L * dy], 1)], 1)
        rods.set_segments(segs)

        pts = W[:, fi, :]
        depth = (pts * d).sum(1)
        px, py = (pts * u).sum(1), (pts * v).sum(1)
        near.set_offsets(np.c_[px[depth < 0], py[depth < 0]])
        far.set_offsets(np.c_[px[depth >= 0], py[depth >= 0]])
        for s in range(n_strand):
            tr = W[s, max(0, fi - 120):fi + 1, :]
            vis = (tr * d).sum(1) < 0
            tx, ty = (tr * u).sum(1), (tr * v).sum(1)
            tx = np.where(vis, tx, np.nan)
            trails[s].set_data(tx, np.where(vis, ty, np.nan))

        cursor.set_xdata([times[fi], times[fi]])
        cursor2.set_xdata([times[fi], times[fi]])
        done = sum(1 for fr, _ in crossings if fr <= fi)
        sup.set_text(f"active nematic on a sphere,  Pe = {pe:.2f},  "
                     f"t = {times[fi]:.0f},  crossings so far: {done} of {len(word)}")
        fig.savefig(tmpdir / f"f{n:05d}.png", facecolor="white")
        if n % 25 == 0:
            print(f"  frame {n}/{len(frames)}", flush=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", str(tmpdir / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)], check=True)
    for p in tmpdir.glob("f*.png"):
        p.unlink()
    tmpdir.rmdir()
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
