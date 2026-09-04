"""Setting up a new choice of boundary geometry.

A confined run needs three things of a wall: a mesh conforming to it, an inward
normal at every wall vertex, and the total defect charge the anchoring imposes.
`volterra.PlaneCurve` supplies the wall and `volterra.confined_mesh` supplies
the other two.

Four walls are built here, in increasing order of how much the caller has to
say:

  1. the analytic epitrochoid, by its winding number and cusp regularisation;
  2. a five-lobed flower, from its parametrisation;
  3. a rounded square, from a table of points;
  4. an ellipse with a wall bump, also from a table.

Each is meshed, checked for element quality, and read for the charge tangential
anchoring imposes on it. The figure at the end shows all four with their
anchored directors.

Run:  python new_boundary_geometry.py [outfile.png]
"""

import math
import sys

import numpy as np

import volterra as v


# ---------------------------------------------------------------------------
# 1. The analytic wall
# ---------------------------------------------------------------------------

# q = 1 + k/2 for k cusps: 1.5 cardioid, 2 nephroid, 2.5 trefoiloid.
# d regularises the cusp; the tip radius of curvature goes as (1 - d)^2, so
# d < 1 is what a mesh of bounded aspect ratio can resolve.
nephroid = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=60.0)


# ---------------------------------------------------------------------------
# 2. From a parametrisation
# ---------------------------------------------------------------------------

def flower(u, lobes=5, amp=0.3, r=60.0):
    rad = r * (1.0 + amp * math.cos(lobes * u))
    return (rad * math.cos(u), rad * math.sin(u))


# `samples` sets the resolution of the description; the mesh takes its element
# size from `h_bulk`. Raise `samples` where the wall turns inside one interval.
flower_curve = v.PlaneCurve.from_callable(flower, samples=1200)


# ---------------------------------------------------------------------------
# 3. From a table of points
# ---------------------------------------------------------------------------

def rounded_square(n=800, half=45.0, radius=12.0):
    """A square with filleted corners, tabulated once round."""
    pts = []
    centres = [(half - radius, half - radius), (-half + radius, half - radius),
               (-half + radius, -half + radius), (half - radius, -half + radius)]
    # Four straight sides and four quarter arcs, walked anticlockwise.
    per_arc = n // 8
    per_side = n // 8
    for k in range(4):
        cx, cy = centres[k]
        a0 = 0.5 * math.pi * k
        for i in range(per_arc):
            a = a0 + 0.5 * math.pi * i / per_arc
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        nx, ny = centres[(k + 1) % 4]
        a1 = a0 + 0.5 * math.pi
        sx, sy = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
        ex, ey = nx + radius * math.cos(a1), ny + radius * math.sin(a1)
        for i in range(per_side):
            t = i / per_side
            pts.append((sx + t * (ex - sx), sy + t * (ey - sy)))
    return np.array(pts)


square_curve = v.PlaneCurve.from_points(rounded_square())


# ---------------------------------------------------------------------------
# 4. A wall with one local feature
# ---------------------------------------------------------------------------

def bumped_ellipse(n=1000, a=70.0, b=40.0, bump=18.0, width=0.25, at=0.6 * math.pi):
    """An ellipse pushed inward over a short arc, which is the shape a wall
    obstacle takes. The bump's own width sets the local element size, and the
    sampling has to resolve it before the mesher can."""
    u = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    dent = bump * np.exp(-((np.angle(np.exp(1j * (u - at)))) ** 2) / (2.0 * width ** 2))
    return np.column_stack([(a - dent) * np.cos(u), (b - dent) * np.sin(u)])


bump_curve = v.PlaneCurve.from_points(bumped_ellipse())


# ---------------------------------------------------------------------------
# Mesh each of them, and read the imposed charge
# ---------------------------------------------------------------------------

cases = [
    ("nephroid", nephroid, dict(h_bulk=1.6, h_min=0.4)),
    ("flower", flower_curve, dict(h_bulk=1.6, h_min=0.5)),
    ("square", square_curve, dict(h_bulk=1.6, h_min=0.6)),
    ("bump", bump_curve, dict(h_bulk=1.6, h_min=0.4)),
]

meshes = []
print(f"{'wall':10s} {'vertices':>9s} {'triangles':>10s} {'min angle':>10s} "
      f"{'charge':>8s} {'worst step':>11s}")
for name, curve, opts in cases:
    m = v.confined_mesh(curve, **opts)
    charge, worst_deg, over = m.imposed_charge(q_anchor=1.0)
    meshes.append((name, curve, m, charge))
    print(f"{name:10s} {m.n_vertices:9d} {m.n_triangles:10d} "
          f"{m.min_angle_deg:9.2f} {charge:8.4f} {worst_deg:10.2f} deg"
          + ("" if over == 0 else f"  [{over} steps over a quarter turn]"))

# The imposed charge is what the anchoring will actually put in the interior,
# measured on this mesh's own boundary rather than assumed from the geometry. A
# boundary too coarse for its wall books the wrong branch, which is the failure
# a lattice mask has at a cusp, and the worst boundary step beside the charge is
# the reading that says so. Element shape is min_angle_deg; the quantity that
# reaches the DEC operator is worst_cot_weight, which a triangle past a right
# angle makes negative.
for name, _, m, charge in meshes:
    assert m.min_area > 0.0, f"{name}: degenerate element"
    assert abs(charge - 1.0) < 1e-6, f"{name}: imposed charge {charge}"


# ---------------------------------------------------------------------------
# The anchoring itself
# ---------------------------------------------------------------------------

# Strong planar anchoring pins Q at every wall vertex to the value set by the
# outward normal's angle. `anchoring_q` returns exactly the Dirichlet data the
# solver imposes after every step.
name, curve, m, _ = meshes[0]
q_wall = m.anchoring_q(q_anchor=1.0, s0=1.0)
print(f"\n{name}: Dirichlet Q on {q_wall.shape[0]} wall vertices, "
      f"sqrt(Tr Q^2) = {math.sqrt(2.0 * (q_wall[0] ** 2).sum()):.6f}")

# A winding anchoring is the same call with a different q. It forces 2q excess
# +1/2 cores into the interior, which is what the steady-winding benchmarks use.
for q_anchor in (1.0, 2.0, 3.0):
    charge, _, _ = m.imposed_charge(q_anchor)
    print(f"  q_anchor = {q_anchor}: imposed charge {charge:.4f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
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

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.0))
    for ax, (name, curve, m, charge) in zip(axes, meshes):
        xy = m.vertices
        ax.triplot(xy[:, 0], xy[:, 1], m.triangles, lw=0.18, color="#8899aa")

        b = m.boundary_vertices
        wall = xy[b]
        ax.plot(np.append(wall[:, 0], wall[0, 0]),
                np.append(wall[:, 1], wall[0, 1]),
                lw=1.0, color="#000000")

        # The anchored director, drawn as a headless segment every few vertices.
        # Its half-length is a fixed fraction of the domain, so the segments read
        # as directors at any scale in place of thickening into a second wall.
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        step = max(1, m.n_boundary // 44)
        d = m.anchoring_director(1.0)[::step]
        q = wall[::step]
        half = 0.055 * span
        seg = np.stack([q - half * d, q + half * d], axis=1)
        ax.add_collection(LineCollection(list(seg), colors="#b03030", linewidths=1.3))

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.10)

    # An equal-aspect axis shrinks to its own content, so a per-axes title sits
    # at a different height in each panel. Place the captions in figure
    # coordinates instead, on one line across the four.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.03)
    for ax, (name, curve, m, charge) in zip(axes, meshes):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), 0.90,
                 rf"{name}\quad ${m.n_triangles}$ triangles"
                 "\n"
                 rf"min angle ${m.min_angle_deg:.1f}^\circ$, "
                 rf"imposed charge ${charge:.2f}$",
                 ha="center", va="bottom", fontsize=11)
    fig.savefig(path, dpi=220)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "new_boundary_geometry.png"
    figure(out)
