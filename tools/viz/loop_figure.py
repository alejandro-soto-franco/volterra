#!/usr/bin/env python3
"""A disclination loop in a 3D active nematic, coloured by winding character.

The tube is the isosurface of the disclination density, the scalar that vanishes
in the ordered bulk and peaks on a core. It is coloured by the local `cos(beta)`,
which is what Head, Negro and co-authors colour theirs by: `+1` is a comet, the
`+1/2` profile, `-1` a triradius, the `-1/2` profile, and `0` a twist between
them. The map is viridis, identified by fitting their Fig. 1 rather than assumed.

Both fields come from the same density tensor the line detector reads, so the
surface drawn is the surface measured.

Usage: loop_figure.py <run_dir> [--out figure.png] [--margin 3]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ── House colours ────────────────────────────────────────────────────────────
INK = "#000000"
PAPER = "#ffffff"
RULE = "#4a4a4a"
# The flow is drawn in a neutral grey. Colour is reserved for the winding
# character, so a second scale would compete with the thing the figure is about.
FLOW = "#6f7378"
# The winding character runs on viridis, dark purple at a triradius through teal
# at a twist to yellow at a comet.
CHARACTER_MAP = "viridis"


def load_run(run_dir: Path):
    """The two fields and the curves the solver wrote."""
    meta = json.loads((run_dir / "curves.json").read_text())
    n = meta["n"]
    density = np.load(run_dir / "density.npy").reshape(n, n, n)
    character = np.load(run_dir / "cos_beta.npy").reshape(n, n, n)
    flow_path = run_dir / "velocity.npy"
    flow = np.load(flow_path).reshape(n, n, n, 3) if flow_path.exists() else None
    return meta, density, character, flow


def interior(field: np.ndarray, margin: int, axis_radius: float = 0.0) -> np.ndarray:
    """Blank the two features the seed puts in that the physics does not.

    The derivative stencil wraps, so a field that is not periodic reads a seam at
    the box faces. And the seeded texture refers to the radial direction, which
    is undefined on the box axis, so a line forms there too. Both come from the
    initial condition rather than from the solver, and both are stated in the
    caption rather than quietly removed.
    """
    out = field.copy()
    out[:margin] = out[-margin:] = 0.0
    out[:, :margin] = out[:, -margin:] = 0.0
    out[:, :, :margin] = out[:, :, -margin:] = 0.0
    if axis_radius > 0.0:
        n = out.shape[0]
        c = (n - 1) / 2.0
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        near = np.hypot(i - c, j - c) < axis_radius
        out[near, :] = 0.0
    return out


def render_volume(meta, density, character, flow, margin, axis_radius, size=(1600, 1400)):
    """The 3D panel: the density isosurface, coloured by winding character."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    n, dx = meta["n"], meta["dx"]

    grid = pv.ImageData(dimensions=(n, n, n), spacing=(dx, dx, dx), origin=(0, 0, 0))
    grid.point_data["density"] = interior(density, margin, axis_radius).ravel(order="F")
    grid.point_data["character"] = character.ravel(order="F")

    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(PAPER)

    surface = grid.contour([meta["threshold"]], scalars="density")
    if surface.n_points:
        plotter.add_mesh(
            surface,
            scalars="character",
            cmap=CHARACTER_MAP,
            clim=(-1.0, 1.0),
            opacity=0.45,
            smooth_shading=True,
            specular=0.25,
            show_scalar_bar=False,
        )

    # The detected curve down the middle of that tube, drawn opaque so the line
    # and the surface it is the axis of are both legible.
    for curve in meta["curves"]:
        pts = np.asarray(curve["points"])
        lo, hi = margin * dx, (n - 1 - margin) * dx
        if pts.min() < lo or pts.max() > hi:
            continue
        centre = (n - 1) * dx / 2.0
        if np.hypot(pts[:, 0] - centre, pts[:, 1] - centre).min() < axis_radius * dx:
            continue
        if not curve["is_loop"]:
            continue
        if len(pts) < 3:
            continue
        if curve["is_loop"]:
            pts = np.vstack([pts, pts[:1]])
        cells = np.concatenate([[len(pts)], np.arange(len(pts))])
        line = pv.PolyData(pts, lines=cells)
        line.point_data["character"] = np.asarray(
            curve["cos_beta"] + curve["cos_beta"][:1]
            if curve["is_loop"]
            else curve["cos_beta"]
        )
        plotter.add_mesh(
            line.tube(radius=0.55 * dx, n_sides=24),
            scalars="character",
            cmap=CHARACTER_MAP,
            clim=(-1.0, 1.0),
            smooth_shading=True,
            specular=0.4,
            show_scalar_bar=False,
        )

    if flow is not None:
        # Streamlines seeded on a sphere about the loop, which is where the
        # active stress of the texture actually drives the fluid.
        grid.point_data["flow"] = flow.reshape(-1, 3, order="F")
        centre = (n - 1) * dx / 2.0
        # Few seeds and short traces. A dense streamline set is a picture of the
        # integrator rather than of the flow, and it hides the loop the figure
        # is about.
        seed = pv.Sphere(radius=0.30 * n * dx, center=(centre, centre, centre),
                         theta_resolution=6, phi_resolution=5)
        try:
            streams = grid.streamlines_from_source(
                seed, vectors="flow",
                max_length=0.45 * n * dx,
                integration_direction="both",
            )
        except Exception:
            streams = None
        if streams is not None and streams.n_points > 2:
            plotter.add_mesh(
                streams.tube(radius=0.09 * dx),
                color=FLOW,
                opacity=0.45,
                smooth_shading=True,
                show_scalar_bar=False,
            )

    plotter.add_mesh(grid.outline(), color=RULE, line_width=1.5)
    # One fixed viewpoint. A film never orbits and neither does a still, so the
    # geometry is read from the same angle every time this is regenerated.
    plotter.camera_position = [
        (2.4 * n * dx, -1.9 * n * dx, 1.7 * n * dx),
        (n * dx / 2, n * dx / 2, n * dx / 2),
        (0.0, 0.0, 1.0),
    ]
    return plotter.screenshot(return_img=True)


def compose(meta, panel, out_path: Path, has_flow: bool):
    """Title the panel and label the character scale."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "axes.grid": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    })

    loops = [c for c in meta["curves"] if c["is_loop"]]
    ring = max(loops, key=lambda c: c["length"]) if loops else None

    fig = plt.figure(figsize=(9.2, 10.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[24, 1], hspace=0.10,
        left=0.05, right=0.95, top=0.90, bottom=0.26,
    )

    ax = fig.add_subplot(gs[0])
    ax.imshow(panel)
    ax.set_axis_off()

    fig.suptitle(
        "Disclination loop in a 3D active nematic with hydrodynamics"
        if has_flow
        else "Disclination loop in a 3D active nematic",
        fontsize=16, y=0.965,
    )
    if ring is not None:
        radius = ring["length"] / (2 * np.pi)
        fig.text(
            0.5, 0.925,
            rf"closed curve of contour length ${ring['length']:.2f}$ and radius "
            rf"${radius:.2f}$, seeded at $16$",
            fontsize=11, ha="center", va="center",
        )

    cax = fig.add_subplot(gs[1])
    mappable = cm.ScalarMappable(norm=Normalize(-1, 1), cmap=CHARACTER_MAP)
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    bar.set_ticklabels([
        r"$-1$, triradius",
        r"$-0.5$",
        r"$0$, twist",
        r"$0.5$",
        r"$+1$, comet",
    ])
    bar.ax.tick_params(labelsize=10)
    bar.outline.set_edgecolor(INK)
    cax.set_title(
        r"winding character $\cos\beta = \hat{\Omega}\cdot\hat{T}$",
        fontsize=12, pad=8,
    )

    import textwrap

    notes = []
    if ring is not None:
        radius = ring["length"] / (2 * np.pi)
        notes.append(
            rf"Line curvature ${ring['mean_curvature']:.4f}$ against $1/r = {1 / radius:.4f}$. "
            rf"The tube around it curves at ${ring['surface_mean_curvature']:+.4f}$, the larger "
            rf"of the two, as a tube tighter than the ring it follows must."
        )
        notes.append(
            rf"Mean $\cos\beta = {ring['mean_cos_beta']:+.3f}$: the character sweeps the whole "
            rf"scale once around the loop and averages to nothing. Isosurface at "
            rf"$s = {meta['threshold']:.2e}$, a quarter of the interior peak."
        )
    if has_flow:
        notes.append(
            r"Grey lines are the flow. Each step solves the steady Stokes problem driven by the "
            r"active stress $-\zeta Q$ and advects the texture in the result, so the loop moves "
            r"in a flow its own winding generates."
        )
    notes.append(
        r"The wrapped stencil reads a seam at the box faces, and the seeded texture is singular "
        r"on the box axis. Both are masked here, and both come from the initial condition rather "
        r"than from the solver."
    )

    # Wrap by hand: matplotlib's own wrapping does not respect a text box that
    # was never given a width, and a caption running off the page is worse than
    # one that is a line longer.
    y = 0.175
    for note in notes:
        for line in textwrap.wrap(note, width=112):
            fig.text(0.05, y, line, fontsize=9.5, ha="left", va="top")
            y -= 0.018
        y -= 0.006

    fig.savefig(out_path, dpi=200)
    return ring


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("loop_figure.png"))
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--axis-radius", type=float, default=9.0,
                    help="mask this radius about the box axis, where the seed is singular")
    args = ap.parse_args()

    meta, density, character, flow = load_run(args.run_dir)
    panel = render_volume(meta, density, character, flow, args.margin, args.axis_radius)
    ring = compose(meta, panel, args.out, flow is not None)
    print(f"wrote {args.out}")
    if ring is not None:
        print(
            f"loop: length {ring['length']:.2f}, radius {ring['length'] / (2 * np.pi):.2f}, "
            f"mean cos(beta) {ring['mean_cos_beta']:+.3f}, "
            f"line curvature {ring['mean_curvature']:.4f}"
        )


if __name__ == "__main__":
    main()
