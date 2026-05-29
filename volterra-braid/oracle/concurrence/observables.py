# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scipy>=1.10"]
# ///
"""Dynamical concurrence observables for volterra vs CGPO (arXiv:2503.10880).

These are the chaos-appropriate, statistical/topological measures the paper
itself uses, computed from a velocity-field (and Q-field) time series so the same
function applies to either solver's output:

  - time-averaged vorticity <omega> and its gyre count (paper Fig. 6/SI1; the
    number of gyres is set by boundary topology, = 4|q-1|);
  - line-stretching topological entropy h (paper Methods: advect a contour, slope
    of log L(t)).

Pure NumPy/SciPy so it runs on either solver's (nx,ny,2) field arrays. Defect
statistics reuse the volterra extension's detector elsewhere.

Self-test (synthetic flows with known answers): uv run observables.py
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def vorticity(u: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """omega = d u_y/dx - d u_x/dy for a velocity field u of shape (nx, ny, 2)."""
    ux, uy = u[..., 0], u[..., 1]
    duy_dx = np.gradient(uy, dx, axis=0)
    dux_dy = np.gradient(ux, dx, axis=1)
    return duy_dx - dux_dy


def time_averaged_vorticity(u_frames, dx: float = 1.0) -> np.ndarray:
    """Mean vorticity over a sequence of velocity fields."""
    return np.mean([vorticity(u, dx) for u in u_frames], axis=0)


def gyre_count(omega_avg: np.ndarray, mask: np.ndarray | None = None,
               threshold_frac: float = 0.15) -> int:
    """Number of gyres = connected sign-coherent vortical regions in <omega>.

    Counts connected components of {omega > thr} and {omega < -thr} where
    thr = threshold_frac * max|omega| (over the active region).
    """
    field = omega_avg.copy()
    if mask is not None:
        field = np.where(mask, field, 0.0)
    thr = threshold_frac * np.max(np.abs(field))
    if thr == 0:
        return 0
    n_pos = label(field > thr)[1]
    n_neg = label(field < -thr)[1]
    return n_pos + n_neg


def _bilinear(u: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of field u (nx,ny,2) at points pts (...,2), clamped."""
    nx, ny = u.shape[:2]
    x = np.clip(pts[..., 0], 0, nx - 1.001)
    y = np.clip(pts[..., 1], 0, ny - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = x0 + 1, y0 + 1
    out = np.empty(pts.shape[:-1] + (2,))
    for c in range(2):
        out[..., c] = (
            u[x0, y0, c] * (1 - fx) * (1 - fy) + u[x1, y0, c] * fx * (1 - fy)
            + u[x0, y1, c] * (1 - fx) * fy + u[x1, y1, c] * fx * fy
        )
    return out


def line_stretching_entropy(u_frames, dt: float, p0: np.ndarray, p1: np.ndarray,
                            n_pts: int = 400, fit_from: float = 0.2):
    """Topological entropy via the line-stretching method (paper Methods).

    Advect a line segment [p0, p1] (n_pts samples) through the velocity-field
    time series with RK4 + adaptive resampling, tracking contour length L(t).
    Returns (h, times, log_lengths) where h is the slope of log L(t) fit over the
    last (1 - fit_from) fraction of the trajectory (the asymptotic stretching rate).
    """
    seg = np.linspace(p0, p1, n_pts)
    lengths, times = [], []
    L0 = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    for k, u in enumerate(u_frames):
        # RK4 advection of every point through this frame's (frozen) field.
        v = lambda q: _bilinear(u, q)
        k1 = v(seg)
        k2 = v(seg + 0.5 * dt * k1)
        k3 = v(seg + 0.5 * dt * k2)
        k4 = v(seg + dt * k3)
        seg = seg + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Resample to keep point spacing bounded as the contour stretches.
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        L = float(np.sum(d))
        s = np.concatenate([[0], np.cumsum(d)])
        snew = np.linspace(0, s[-1], n_pts)
        seg = np.stack([np.interp(snew, s, seg[:, 0]), np.interp(snew, s, seg[:, 1])], axis=1)
        lengths.append(L / L0)
        times.append((k + 1) * dt)
    times = np.asarray(times)
    logL = np.log(np.asarray(lengths))
    i0 = int(fit_from * len(times))
    h = np.polyfit(times[i0:], logL[i0:], 1)[0] if len(times) - i0 >= 2 else 0.0
    return float(h), times, logL


# --------------------------------------------------------------------------- #
def _self_test():
    fails = 0

    # 1. Double-gyre <omega> -> 2 gyres. Classic steady double-gyre stream
    #    function psi = sin(pi x) sin(pi y) on [0,2]x[0,1] gives two counter-
    #    rotating cells; vorticity has two sign-coherent lobes.
    nx, ny = 120, 60
    X = np.linspace(0, 2, nx)[:, None] * np.ones((1, ny))
    Y = np.ones((nx, 1)) * np.linspace(0, 1, ny)[None, :]
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u = np.empty((nx, ny, 2))
    u[..., 0] = np.gradient(psi, axis=1)   # ux =  d psi/dy
    u[..., 1] = -np.gradient(psi, axis=0)  # uy = -d psi/dx
    omega = time_averaged_vorticity([u])
    g = gyre_count(omega)
    ok = g == 2
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] double-gyre flow -> gyre_count == 2  (got {g})")

    # 2. Linear extensional (hyperbolic) flow u = (lambda x, -lambda y) about the
    #    grid centre stretches a material line at rate lambda -> h ~= lambda. Keep
    #    the contour inside the grid (confined-flow regime; no boundary clamping).
    lam = 0.7
    gx, gy = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
    uext = np.stack([lam * (gx - 50.0), -lam * (gy - 50.0)], axis=-1)
    dt = 0.01
    nsteps = 200  # t=2: max excursion 50 + 6 e^{1.4} ~= 74 < 100, stays in-grid
    h, t, logL = line_stretching_entropy([uext] * nsteps, dt,
                                         p0=np.array([44.0, 50.0]), p1=np.array([56.0, 50.0]))
    ok2 = abs(h - lam) < 0.12 * lam
    fails += not ok2
    print(f"  [{'PASS' if ok2 else 'FAIL'}] extensional flow -> h ~= lambda={lam}  (got {h:.3f})")

    # 3. Rigid rotation -> no net stretching -> h ~= 0.
    omega_rot = 0.5
    urot = np.stack([-omega_rot * (gy - 50.0), omega_rot * (gx - 50.0)], axis=-1)
    h0, *_ = line_stretching_entropy([urot] * 200, 0.01,
                                     p0=np.array([35.0, 50.0]), p1=np.array([65.0, 50.0]))
    ok3 = abs(h0) < 0.05
    fails += not ok3
    print(f"  [{'PASS' if ok3 else 'FAIL'}] rigid rotation -> h ~= 0  (got {h0:.3f})")

    print("observables self-test " + ("PASS" if fails == 0 else f"FAIL ({fails})"))
    return fails


if __name__ == "__main__":
    raise SystemExit(_self_test())
