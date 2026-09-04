"""A confined active nematic run, driven from Python.

The same scheme the Rust driver runs: a Stokes solve for the velocity from the
Beris-Edwards stress, then a semi-implicit Q update with the Frank term
implicit, with the wall pinned to the anchored value after every step.

This runs one geometry twice, once with each velocity wall condition, and draws
the director, the flow and the defects side by side.

  no-slip    the clamped plate, psi = 0 with dpsi/dn = 0
  free slip  the simply supported plate, psi = 0 with Laplacian psi = 0

Both take the same anchoring and the same initial field, so the panels differ
only in the wall.

Run:  python confined_run.py [outfile.png]
"""

import sys
import time

import numpy as np

import volterra as v


# ---------------------------------------------------------------------------
# Domain and parameters
# ---------------------------------------------------------------------------

# A nephroid: two cusps, regularised at d < 1 so a mesh of bounded aspect ratio
# resolves the tip. `imposed_charge` is the reading that says whether the
# boundary sampling can see the winding it is being asked to impose.
CURVE = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=40.0)
MESH = v.confined_mesh(CURVE, h_bulk=1.4, h_min=0.5, seed=0)

# The two lengths that set the physics, in lattice units. `active_length` is
# sqrt(K / zeta) and `coherence_length` is sqrt(K / C), the defect core size.
# The core has to sit above about twice the bulk element for a defect count to
# mean anything, so 4.0 against h_bulk 1.4 is comfortable.
PHYSICS = dict(active_length=3.5, coherence_length=4.0, resolution=80)

STEPS = 4000
SETTLE = 200
SEED = 1


def build(wall):
    return v.ConfinedRun(MESH, wall=wall, dt=2e-4, seed=SEED, wall_h=0.05, **PHYSICS)


charge, worst_step, over = MESH.imposed_charge(q_anchor=1.0)
print(f"mesh: {MESH.n_vertices} vertices, {MESH.n_triangles} triangles, "
      f"{MESH.n_boundary} on the wall, min angle {MESH.min_angle_deg:.1f} deg")
print(f"anchoring imposes charge {charge:.4f}, worst boundary step "
      f"{worst_step:.1f} deg, {over} steps over a quarter turn")

runs = {}
for wall in ("noslip", "freeslip"):
    r = build(wall)
    print(f"\n{wall}: {r.wall_vertices} vertices held at the wall, "
          f"{r.elastic_mask_vertices} with the elastic stress suppressed")
    print(f"  dt {r.dt:.1e}, explicit diffusive limit {r.diffusive_dt_limit:.2e}")

    # A short passive settle, so the active run starts from an ordered field
    # rather than from noise. Both walls settle the same way, since the flow is
    # off; it is the active phase that separates them.
    n, last = r.relax(SETTLE)
    print(f"  settled in {n} passive steps, last change {last:.2e}")

    t0 = time.time()
    r.step(STEPS)
    dt_wall = time.time() - t0
    s = r.stats()
    print(f"  {STEPS} steps in {dt_wall:.1f} s ({1e3 * dt_wall / STEPS:.2f} ms a step), "
          f"t = {r.time:.3f}")
    print(f"  defects {s['n_plus']:+d}/-{s['n_minus']}, charge {s['charge']:+.1f}, "
          f"S median {s['s_median']:.4f}, |u| max {s['speed_max']:.3f}, "
          f"Courant {s['courant']:.3f}")
    runs[wall] = r

# The wall changes the flow it drives, and the defect state it settles into.
u = {w: np.linalg.norm(r.velocity, axis=1) for w, r in runs.items()}
print(f"\nmean speed: no-slip {u['noslip'].mean():.4f}, "
      f"free slip {u['freeslip'].mean():.4f}")


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
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    wallxy = MESH.vertices[MESH.boundary_vertices]

    for ax, wall in zip(axes, ("noslip", "freeslip")):
        r = runs[wall]
        xy = r.mesh.vertices
        s = r.order_parameter
        theta = r.director_angle

        # Order parameter as the ground, so a defect core reads as a dip.
        ax.tripcolor(xy[:, 0], xy[:, 1], r.mesh.triangles, s,
                     cmap="Greys_r", vmin=0.0, vmax=1.0, shading="gouraud")

        # The director, on a thinned subset so the segments stay legible.
        step = max(1, len(xy) // 700)
        p = xy[::step]
        d = np.column_stack([np.cos(theta[::step]), np.sin(theta[::step])])
        half = 0.012 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        ax.add_collection(LineCollection(
            list(np.stack([p - half * d, p + half * d], axis=1)),
            colors="#2a4d8f", linewidths=0.7))

        for x, y, charge in r.defects():
            ax.plot(x, y, marker="o" if charge > 0 else "s", ms=6.0,
                    mfc="#b03030" if charge > 0 else "#2f7f4f", mec="#000000",
                    mew=0.6, ls="none")

        ax.plot(np.append(wallxy[:, 0], wallxy[0, 0]),
                np.append(wallxy[:, 1], wallxy[0, 1]),
                lw=1.0, color="#000000")

        st = r.stats()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            rf"{'no slip' if wall == 'noslip' else 'free slip'}"
            "\n"
            rf"$t = {r.time:.2f}$, ${st['n_plus']}$ plus and ${st['n_minus']}$ minus, "
            rf"$|u|_{{\max}} = {st['speed_max']:.2f}$",
            fontsize=11, pad=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.05)
    fig.savefig(path, dpi=220)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else "confined_run.png")
