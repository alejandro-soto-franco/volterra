#!/usr/bin/env python3
"""A disclination loop evolving under hydrodynamics.

One 3D panel of the loop and the flow it drives, and three panels of what the
detector measures as the film runs. The colour is the winding character
`cos(beta)`, as Head, Negro and co-authors colour theirs: `+1` a comet, the
`+1/2` profile, `-1` a triradius, the `-1/2` profile, `0` a twist between them.

The scale is fixed at `[-1, 1]` rather than normalised per frame, because
`cos(beta)` means the same thing in every frame and a moving scale would make a
loop that never changed appear to.

One camera, set once. A film that orbits shows the camera rather than the
physics, and every frame is gated on identical pixel extent before encoding.

Usage: loop_film.py <run_dir> [--out film.mp4] [--preview]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".claude/skills/comparison-panel-video"))
from panelkit import apply_style, encode, render_frames  # noqa: E402

CHARACTER_MAP = "viridis"
# The scalar order parameter has its own convention in this project: white where
# the nematic is ordered, green where it melts, so a defect core reads as a green
# spot against a white bulk. It is deliberately not viridis, which belongs to the
# winding character and would be read as the same quantity.
ORDER_LOW = "#1a9a3a"
ORDER_HIGH = "#ffffff"
FLOW = "#000000"
HOLD_SECONDS = 4
FPS = 30


def load_frames(run_dir: Path):
    """Every frame's derived fields, in order."""
    metas = sorted(run_dir.glob("curves_*.json"))
    frames = []
    for m in metas:
        tag = m.stem.split("_")[1]
        meta = json.loads(m.read_text())
        n = meta["n"]
        order_path = run_dir / f"order_{tag}.npy"
        director_path = run_dir / f"director_{tag}.npy"
        frames.append({
            "meta": meta,
            "density": np.load(run_dir / f"density_{tag}.npy").reshape(n, n, n),
            "character": np.load(run_dir / f"cos_beta_{tag}.npy").reshape(n, n, n),
            "order": np.load(order_path).reshape(n, n, n) if order_path.exists() else None,
            "director": (np.load(director_path).reshape(n, n, n, 3)
                         if director_path.exists() else None),
        })
    return frames


def interior(field, margin, axis_radius, fill=0.0):
    """Blank the seam at the faces and the seed's singular axis.

    `fill` is what "nothing here" means for the field being masked. Zero is right
    for a disclination density, which vanishes where there is no defect. It is
    wrong for the scalar order parameter, where zero means fully melted and hence
    fully opaque: filling a mask with it wraps the box in an opaque shell.
    """
    out = field.copy()
    # `out[-0:]` is the whole array, so a zero margin would blank the field
    # rather than leave it alone.
    if margin > 0:
        out[:margin] = out[-margin:] = fill
        out[:, :margin] = out[:, -margin:] = fill
        out[:, :, :margin] = out[:, :, -margin:] = fill
    if axis_radius > 0:
        n = out.shape[0]
        c = (n - 1) / 2.0
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        out[np.hypot(i - c, j - c) < axis_radius, :] = fill
    return out


def order_map():
    """White where the nematic is ordered, green where it melts."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("volterra_s", [ORDER_LOW, ORDER_HIGH])


def render_panels(frames, margin, axis_radius, stride=4, size=(1700, 1700)):
    """One 3D image per frame, from one camera that is set once.

    Every image is checked for the same pixel extent before any of them is
    encoded. A frame that differs in size makes the film jump, and the cause is
    usually a camera that was reset rather than held.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    n = frames[0]["meta"]["n"]
    dx = frames[0]["meta"]["dx"]
    centre = (n - 1) * dx / 2.0
    # One camera, set once and never reset. Placed close enough that the box
    # fills its panel: a film of mostly white space wastes the frame it has.
    camera = [
        (1.95 * n * dx, -1.52 * n * dx, 1.34 * n * dx),
        (centre, centre, centre),
        (0.0, 0.0, 1.0),
    ]

    images = []
    for k, fr in enumerate(frames):
        meta = fr["meta"]
        grid = pv.ImageData(dimensions=(n, n, n), spacing=(dx, dx, dx))
        grid.point_data["density"] = interior(fr["density"], margin, axis_radius).ravel(order="F")
        grid.point_data["character"] = fr["character"].ravel(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background("#ffffff")

        # The nematic, rendered as a volume rather than a shell, so the texture
        # is visible everywhere. Kept faint: it is the medium the loop sits in,
        # and an opaque one would hide the loop entirely.
        if fr["order"] is not None:
            order = np.clip(fr["order"] / max(fr["order"].max(), 1e-12), 0.0, 1.0)
            grid.point_data["order"] = interior(order, margin, axis_radius, fill=1.0).ravel(order="F")
            plotter.add_volume(
                grid, scalars="order", cmap=order_map(),
                opacity=[0.62, 0.34, 0.17, 0.08, 0.03, 0.0],
                clim=(0.0, 1.0), shade=False, show_scalar_bar=False,
            )

        surface = grid.contour([meta["threshold"]], scalars="density")
        if surface.n_points:
            plotter.add_mesh(surface, scalars="character", cmap=CHARACTER_MAP,
                             clim=(-1.0, 1.0), opacity=0.30, smooth_shading=True,
                             specular=0.25, show_scalar_bar=False)

        for curve in meta["curves"]:
            pts = np.asarray(curve["points"])
            lo, hi = margin * dx, (n - 1 - margin) * dx
            if pts.min() < lo or pts.max() > hi or len(pts) < 4:
                continue
            if np.hypot(pts[:, 0] - centre, pts[:, 1] - centre).min() < axis_radius * dx:
                continue
            beta = list(curve["cos_beta"])
            if curve["is_loop"]:
                pts = np.vstack([pts, pts[:1]])
                beta = beta + beta[:1]
            line = pv.PolyData(pts, lines=np.concatenate([[len(pts)], np.arange(len(pts))]))
            line.point_data["character"] = np.asarray(beta)
            radius = 0.55 * dx if curve["is_loop"] else 0.34 * dx
            plotter.add_mesh(line.tube(radius=radius, n_sides=20), scalars="character",
                             cmap=CHARACTER_MAP, clim=(-1.0, 1.0), smooth_shading=True,
                             specular=0.4, show_scalar_bar=False)

        # The nematic itself, as rods. A director is apolar, so the glyph is a
        # cylinder rather than an arrow and the sign the eigensolver returns
        # does not matter. Coloured on the house convention for S, white where
        # the nematic is ordered and green where it melts, so a core reads as a
        # patch of green rods. White rods stay legible against white because the
        # lighting shades them.
        if fr["director"] is not None and fr["order"] is not None:
            step = max(1, stride)
            sl = slice(margin, n - margin, step)
            idx = np.mgrid[sl, sl, sl].reshape(3, -1).T.astype(float)
            keep = np.hypot(idx[:, 0] - (n - 1) / 2.0, idx[:, 1] - (n - 1) / 2.0) >= axis_radius
            idx = idx[keep]
            ii, jj, ll = idx[:, 0].astype(int), idx[:, 1].astype(int), idx[:, 2].astype(int)
            rods = pv.PolyData(idx * dx)
            rods.point_data["director"] = fr["director"][ii, jj, ll]
            rods.point_data["order"] = np.clip(
                fr["order"][ii, jj, ll] / max(fr["order"].max(), 1e-12), 0.0, 1.0)
            glyphs = rods.glyph(orient="director", scale=False, factor=2.4 * step * dx,
                                geom=pv.Cylinder(radius=0.12, height=1.0, resolution=10))
            plotter.add_mesh(glyphs, scalars="order", cmap=order_map(), clim=(0.0, 1.0),
                             smooth_shading=True, specular=0.35, opacity=0.85,
                             show_scalar_bar=False)

        plotter.add_mesh(grid.outline(), color="#000000", line_width=1.2)
        plotter.camera_position = camera
        images.append(plotter.screenshot(return_img=True))
        plotter.close()
        if k % 10 == 0:
            print(f"  panel {k}/{len(frames)}")

    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise SystemExit(f"the frames differ in extent: {shapes}")
    print(f"{len(images)} panels, all {images[0].shape}")
    return images


def series(frames):
    """The measured quantities, tracking one loop by continuity.

    Taking the longest closed curve each frame is not a persistent identity: on
    this run it switches to a different curve at one snapshot and back, which
    reads as the loop jumping by fifty in length and recovering. Following the
    closed curve nearest the previous frame's centre keeps the same object and
    removes the jump, which is a tracking artefact rather than physics.
    """
    out = {"length": [], "curvature": [], "reference": [], "displacement": [], "lines": []}
    previous = None
    origin = None
    for fr in frames:
        loops = [c for c in fr["meta"]["curves"] if c["is_loop"] and len(c["points"]) > 8]
        out["lines"].append(len(fr["meta"]["curves"]))
        ring = None
        if loops:
            centres = [np.asarray(c["points"]).mean(axis=0) for c in loops]
            if previous is None:
                k = int(np.argmax([c["length"] for c in loops]))
            else:
                k = int(np.argmin([np.linalg.norm(c - previous) for c in centres]))
            ring, previous = loops[k], centres[k]
            if origin is None:
                origin = previous.copy()
        if ring is None:
            for key in ("length", "curvature", "reference", "displacement"):
                out[key].append(np.nan)
            continue
        out["length"].append(ring["length"])
        out["curvature"].append(ring["mean_curvature"])
        out["reference"].append(2 * np.pi / ring["length"])
        out["displacement"].append(float(np.linalg.norm(previous - origin)))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def build(run_dir: Path, out: Path, margin: int, axis_radius: float, stride: int,
          preview: bool):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frames = load_frames(run_dir)
    if not frames:
        raise SystemExit(f"no frames in {run_dir}")

    images = render_panels(frames, margin, axis_radius, stride)
    data = series(frames)
    steps = np.arange(len(frames), dtype=float)

    apply_style(font_size=13.0)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 2.45], hspace=0.48, wspace=0.06,
                          left=0.050, right=0.995, top=0.905, bottom=0.115)
    ax_len = fig.add_subplot(gs[0, 0])
    ax_curv = fig.add_subplot(gs[1, 0])
    ax_char = fig.add_subplot(gs[2, 0])
    ax_3d = fig.add_subplot(gs[:, 1])
    bar_ax = fig.add_axes([0.50, 0.048, 0.44, 0.016])
    bar = fig.colorbar(cm.ScalarMappable(norm=Normalize(-1, 1), cmap=CHARACTER_MAP),
                       cax=bar_ax, orientation="horizontal")
    bar.set_ticks([-1, 0, 1])
    bar.set_ticklabels([r"$-1$, triradius", r"$0$, twist", r"$+1$, comet"])
    bar.outline.set_edgecolor("#000000")
    bar_ax.set_title(r"winding character $\cos\beta$", fontsize=13, pad=6)

    total = len(frames) + HOLD_SECONDS * FPS

    def draw(frame: int):
        k = min(frame, len(frames) - 1)
        for ax in (ax_len, ax_curv, ax_char, ax_3d):
            ax.clear()

        ax_3d.imshow(images[k])
        ax_3d.set_axis_off()
        # The colour bar names the loop's colouring. These are the other two
        # things in the panel, and without a line saying so the grey curves are
        # unreadable. A legend rather than a caption, which the panel has no
        # room for and the house style does not take.
        ax_3d.legend(
            handles=[
                Patch(facecolor="#e8eaec", edgecolor="#000000",
                      label="director, one rod every four voxels"),
                Patch(facecolor=ORDER_LOW, edgecolor="#000000",
                      label=r"melted nematic, low $S$"),
            ],
            loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=12,
            handlelength=1.8, borderpad=0.4,
        )

        upto = slice(0, k + 1)
        ax_len.plot(steps[upto], data["lines"][upto], color="#000000", lw=2.0)
        ax_len.set_title("Disclination lines", loc="left")
        ax_len.set_xlim(0, len(frames) - 1)
        ax_len.set_ylim(0, max(data["lines"]) * 1.08)

        ax_curv.plot(steps[upto], data["curvature"][upto], color="#000000", lw=2.0,
                     label="measured")
        ax_curv.plot(steps[upto], data["reference"][upto], color="#000000", lw=1.2,
                     ls="--", alpha=0.55, label=r"$2\pi/\ell$, a circle")
        ax_curv.set_title("Line curvature", loc="left")
        ax_curv.set_xlim(0, len(frames) - 1)
        lo = min(np.nanmin(data["curvature"]), np.nanmin(data["reference"]))
        hi = max(np.nanmax(data["curvature"]), np.nanmax(data["reference"]))
        ax_curv.set_ylim(lo * 0.92, hi * 1.08)
        ax_curv.legend(loc="upper left")

        ax_char.plot(steps[upto], data["displacement"][upto], color="#000000", lw=2.0)
        ax_char.set_title("Displacement of the loop centre", loc="left")
        ax_char.set_xlabel("snapshot")
        ax_char.set_xlim(0, len(frames) - 1)
        ax_char.set_ylim(0, max(1.0, np.nanmax(data["displacement"]) * 1.1))

        fig.suptitle("Disclination loop evolving under hydrodynamics", y=0.965, fontsize=19)

    return fig, total, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("loop_film.mp4"))
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--axis-radius", type=float, default=9.0)
    ap.add_argument("--stride", type=int, default=4,
                    help="draw a rod every this many voxels")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    fig, total, draw = build(args.run_dir, args.out, args.margin, args.axis_radius,
                             args.stride, args.preview)
    frames_dir = args.frames_dir or args.out.parent / "frames"
    render_frames(fig, total, draw, frames_dir, preview=args.preview)
    if not args.preview:
        encode(frames_dir, args.out)


if __name__ == "__main__":
    main()
