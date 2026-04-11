#!/usr/bin/env python3
"""
Render DEC surface snapshots with two panels:

1. **Director field**: surface coloured by S/S_0 (dark green = 0, white = 1),
   director streamlines overlaid in dark grey.

2. **Vorticity field**: surface coloured by omega (blue = negative, green = 0,
   red = positive) using a diverging rainbow scheme.

Usage:
    python render_surface_pv.py ~/.volterra-bench/viz/sphere --video out.mp4 --orbit
"""

import argparse
import json
import subprocess
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

pv.OFF_SCREEN = True


# ─── Custom colourmaps ───────────────────────────────────────────────────────

def _make_s_cmap():
    """Dark green (S=0) to white (S=1)."""
    return LinearSegmentedColormap.from_list(
        "nematic_order",
        [(0.0, "#0b3d0b"), (0.4, "#1a6e1a"), (0.7, "#6fbf6f"), (1.0, "#ffffff")],
    )

def _make_omega_cmap():
    """Blue (negative) -> green (zero) -> red (positive)."""
    return LinearSegmentedColormap.from_list(
        "vorticity_rainbow",
        [(0.0, "#0000cc"), (0.25, "#0088ff"), (0.5, "#00cc00"),
         (0.75, "#ff8800"), (1.0, "#cc0000")],
    )

S_CMAP = _make_s_cmap()
OMEGA_CMAP = _make_omega_cmap()


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_mesh(mesh_path):
    with open(mesh_path) as f:
        data = json.load(f)
    verts = np.array(data["vertices"], dtype=np.float64)
    tris = np.array(data["triangles"], dtype=np.int64)
    n_faces = len(tris)
    faces = np.column_stack([np.full(n_faces, 3, dtype=np.int64), tris]).ravel()
    return verts, faces, tris


def load_q(path, nv):
    data = np.load(path)
    if data.shape == (nv, 2):
        return data[:, 0], data[:, 1]
    elif data.ndim == 1 and data.shape[0] == 2 * nv:
        return data[:nv], data[nv:]
    else:
        flat = data.ravel()
        if flat.shape[0] == 2 * nv:
            return flat[0::2], flat[1::2]
        raise ValueError(f"unexpected shape {data.shape} for nv={nv}")


def load_velocity(path, nv):
    """Load a velocity snapshot (n_vertices, 3) -> array of [vx, vy, vz]."""
    data = np.load(path)
    if data.shape == (nv, 3):
        return data
    elif data.ndim == 1 and data.shape[0] == 3 * nv:
        return data.reshape(nv, 3)
    raise ValueError(f"unexpected velocity shape {data.shape} for nv={nv}")


# ─── Geometry ────────────────────────────────────────────────────────────────

def compute_vertex_normals(verts, tris):
    normals = np.zeros_like(verts)
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for idx in tri:
            normals[idx] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    return normals / norms


def compute_tangent_frame(verts, normals):
    nv = len(verts)
    e1 = np.zeros((nv, 3))
    e2 = np.zeros((nv, 3))
    for i in range(nv):
        n = normals[i]
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        t = ref - np.dot(ref, n) * n
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-14:
            t = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), n) * n
            t_norm = np.linalg.norm(t)
        e1[i] = t / t_norm
        e2[i] = np.cross(n, e1[i])
    return e1, e2


def compute_vorticity(vel, verts, tris, normals):
    """Compute per-vertex vorticity (scalar on surface) from velocity field.

    omega = (curl v) . n, computed via discrete Stokes theorem on the 1-ring.
    """
    nv = len(verts)
    omega = np.zeros(nv)
    area = np.zeros(nv)

    for tri in tris:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        e01 = p1 - p0
        e12 = p2 - p1
        e20 = p0 - p2

        fn = np.cross(e01, p2 - p0)
        face_area = 0.5 * np.linalg.norm(fn)
        if face_area < 1e-30:
            continue

        # Circulation of velocity around the triangle.
        v0, v1, v2 = vel[i0], vel[i1], vel[i2]
        circ = (np.dot(0.5 * (v0 + v1), e01)
              + np.dot(0.5 * (v1 + v2), e12)
              + np.dot(0.5 * (v2 + v0), e20))

        # Distribute to vertices (1/3 each).
        third_area = face_area / 3.0
        for idx in [i0, i1, i2]:
            omega[idx] += circ / 3.0
            area[idx] += third_area

    # Normalise by dual area.
    mask = area > 1e-30
    omega[mask] /= area[mask]
    return omega


# ─── Triangle locator (BVH) ──────────────────────────────────────────────────

class TriangleLocator:
    """Locate the containing triangle for a point on a triangle mesh.

    Uses a simple spatial hash for O(1) average lookup instead of brute-force
    O(n_faces) per query.
    """

    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris
        self._build_grid(verts, tris)

    def _build_grid(self, verts, tris, cells_per_dim=20):
        lo = verts.min(axis=0) - 1e-6
        hi = verts.max(axis=0) + 1e-6
        self.lo = lo
        self.cell_size = (hi - lo) / cells_per_dim
        self.n = cells_per_dim
        self.grid = {}
        for fi, tri in enumerate(tris):
            tri_verts = verts[tri]
            tri_lo = tri_verts.min(axis=0)
            tri_hi = tri_verts.max(axis=0)
            ilo = np.clip(((tri_lo - lo) / self.cell_size).astype(int), 0, self.n - 1)
            ihi = np.clip(((tri_hi - lo) / self.cell_size).astype(int), 0, self.n - 1)
            for ix in range(ilo[0], ihi[0] + 1):
                for iy in range(ilo[1], ihi[1] + 1):
                    for iz in range(ilo[2], ihi[2] + 1):
                        key = (ix, iy, iz)
                        if key not in self.grid:
                            self.grid[key] = []
                        self.grid[key].append(fi)

    def locate(self, pos):
        """Find containing triangle and barycentric coords for pos.

        Returns (face_idx, bary) or (None, None) if not found.
        """
        cell = tuple(np.clip(((pos - self.lo) / self.cell_size).astype(int), 0, self.n - 1))
        candidates = self.grid.get(cell, [])
        best_fi = None
        best_bary = None
        best_dist = 1e30

        for fi in candidates:
            i0, i1, i2 = self.tris[fi]
            bary = _barycentric(pos, self.verts[i0], self.verts[i1], self.verts[i2])
            if bary is None:
                continue
            # Accept if inside (all coords >= -eps).
            if bary[0] >= -0.05 and bary[1] >= -0.05 and bary[2] >= -0.05:
                # Prefer the triangle where point is most interior.
                dist = -min(bary)
                if dist < best_dist:
                    best_dist = dist
                    best_fi = fi
                    best_bary = bary

        return best_fi, best_bary


def _barycentric(p, a, b, c):
    """Compute barycentric coordinates of p in triangle (a, b, c) in 3D."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


# ─── Streamlines ─────────────────────────────────────────────────────────────

def _interp_director(pos, locator, verts, e1, e2, normals, q1, q2, tris):
    """Interpolate director at an arbitrary surface point using barycentric coords.

    Returns (director_3d, order_s, normal) or (None, 0, None) if location fails.
    """
    fi, bary = locator.locate(pos)
    if fi is None:
        # Fallback to nearest vertex.
        dists = np.linalg.norm(verts - pos, axis=1)
        nearest = np.argmin(dists)
        theta = 0.5 * np.arctan2(q2[nearest], q1[nearest])
        s = 2.0 * np.sqrt(q1[nearest]**2 + q2[nearest]**2)
        d = np.cos(theta) * e1[nearest] + np.sin(theta) * e2[nearest]
        return d, s, normals[nearest]

    i0, i1, i2 = tris[fi]
    w0, w1, w2 = bary

    # Interpolate q1, q2 with barycentric weights.
    q1_interp = w0 * q1[i0] + w1 * q1[i1] + w2 * q1[i2]
    q2_interp = w0 * q2[i0] + w1 * q2[i1] + w2 * q2[i2]
    s = 2.0 * np.sqrt(q1_interp**2 + q2_interp**2)

    if s < 1e-6:
        return None, 0.0, None

    # Interpolate tangent frame.
    e1_interp = w0 * e1[i0] + w1 * e1[i1] + w2 * e1[i2]
    e2_interp = w0 * e2[i0] + w1 * e2[i1] + w2 * e2[i2]
    n_interp = w0 * normals[i0] + w1 * normals[i1] + w2 * normals[i2]
    n_len = np.linalg.norm(n_interp)
    if n_len > 1e-14:
        n_interp /= n_len

    # Orthonormalise e1, e2 against interpolated normal.
    e1_interp -= np.dot(e1_interp, n_interp) * n_interp
    e1_len = np.linalg.norm(e1_interp)
    if e1_len > 1e-14:
        e1_interp /= e1_len
    e2_interp = np.cross(n_interp, e1_interp)

    theta = 0.5 * np.arctan2(q2_interp, q1_interp)
    d = np.cos(theta) * e1_interp + np.sin(theta) * e2_interp
    return d, s, n_interp


def _hermite_smooth(points, n_output=None):
    """Smooth a polyline using Catmull-Rom (cubic Hermite) interpolation.

    Upsamples each segment into sub-segments for a smooth curve.
    """
    if len(points) < 3:
        return points
    if n_output is None:
        n_output = max(len(points) * 3, 10)

    pts = np.array(points)
    n = len(pts)

    # Compute cumulative arc length for parameterisation.
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(dists)])
    total = arc[-1]
    if total < 1e-14:
        return pts

    # Evaluate Catmull-Rom at uniform arc-length samples.
    t_out = np.linspace(0, total, n_output)
    result = np.zeros((n_output, 3))

    for k, t in enumerate(t_out):
        # Find segment.
        seg = np.searchsorted(arc, t, side="right") - 1
        seg = max(0, min(seg, n - 2))
        seg_len = arc[seg + 1] - arc[seg]
        if seg_len < 1e-14:
            result[k] = pts[seg]
            continue
        u = (t - arc[seg]) / seg_len  # local param in [0, 1]

        # Catmull-Rom tangents (using neighbouring points).
        p0 = pts[max(seg - 1, 0)]
        p1 = pts[seg]
        p2 = pts[min(seg + 1, n - 1)]
        p3 = pts[min(seg + 2, n - 1)]

        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)

        # Hermite basis.
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2

        result[k] = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2

    return result


def trace_streamline(start_pos, locator, verts, normals, e1, e2, q1, q2, tris,
                     n_steps=60, step_size=None, direction=1.0):
    """Trace a director streamline with barycentric interpolation.

    Uses sub-edge step size and barycentric lookup for smooth curves.
    """
    if step_size is None:
        edge_lens = []
        for tri in tris[:200]:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge_lens.append(np.linalg.norm(verts[tri[a]] - verts[tri[b]]))
        step_size = 0.2 * np.mean(edge_lens)  # 1/5 edge length for fine resolution

    pos = start_pos.copy()
    points = [pos.copy()]

    for _ in range(n_steps):
        d, s, n = _interp_director(pos, locator, verts, e1, e2, normals, q1, q2, tris)
        if d is None or s < 0.01:
            break

        # Step along director.
        pos = pos + direction * step_size * d

        # Project back to surface (normal correction + radial for sphere).
        if n is not None:
            nearest_dist = np.linalg.norm(verts - pos, axis=1)
            nearest = np.argmin(nearest_dist)
            n_near = normals[nearest]
            pos = pos - np.dot(pos - verts[nearest], n_near) * n_near

        r = np.linalg.norm(pos)
        if r > 0.5:
            pos = pos / r

        points.append(pos.copy())

    return np.array(points)


def build_streamlines_polydata(verts, normals, e1, e2, q1, q2, tris,
                               s, n_seeds=500, n_steps=60):
    """Build a single PolyData with all director streamlines.

    Uses barycentric interpolation for smooth tracing and Catmull-Rom
    spline smoothing to eliminate triangle-boundary kinks.
    """
    nv = len(verts)
    locator = TriangleLocator(verts, tris)
    rng = np.random.RandomState(42)
    seeds = rng.choice(nv, size=min(n_seeds, nv), replace=False)

    all_points = []
    all_lines = []
    offset = 0

    for seed in seeds:
        if s[seed] < 0.05:
            continue

        start = verts[seed].copy()
        pts_fwd = trace_streamline(start, locator, verts, normals, e1, e2,
                                   q1, q2, tris, n_steps=n_steps, direction=1.0)
        pts_bwd = trace_streamline(start, locator, verts, normals, e1, e2,
                                   q1, q2, tris, n_steps=n_steps, direction=-1.0)

        if len(pts_bwd) > 1:
            pts = np.vstack([pts_bwd[::-1], pts_fwd[1:]])
        else:
            pts = pts_fwd
        if len(pts) < 4:
            continue

        # Catmull-Rom smoothing: 3x upsample for silky curves.
        pts = _hermite_smooth(pts, n_output=len(pts) * 3)

        # Re-project smoothed points to sphere surface.
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-14)
        pts = pts / norms

        # Offset slightly above surface for visibility.
        pt_normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        pts_offset = pts + 0.002 * pt_normals

        n_pts = len(pts_offset)
        all_points.append(pts_offset)
        line = np.zeros(n_pts + 1, dtype=np.int64)
        line[0] = n_pts
        line[1:] = np.arange(offset, offset + n_pts)
        all_lines.append(line)
        offset += n_pts

    if not all_points:
        return None
    return pv.PolyData(np.vstack(all_points), lines=np.concatenate(all_lines))


# ─── Rendering ───────────────────────────────────────────────────────────────

def render_frame(verts, faces, tris, q1, q2, vel, frame_path,
                 normals, e1, e2, s_ref,
                 streamline_seeds=500, streamline_steps=25,
                 camera_position=None, window_size=(1920, 1080)):
    """Render a dual-panel frame: director (left) + vorticity (right)."""
    s = 2.0 * np.sqrt(q1**2 + q2**2)
    s_norm = np.clip(s / max(s_ref, 1e-6), 0.0, 1.0)

    # ── Left panel: director field with S colouring ──
    plotter = pv.Plotter(off_screen=True, window_size=window_size,
                         shape=(1, 2), border=False)
    plotter.subplot(0, 0)
    plotter.set_background("#0a0a0a")

    surf_s = pv.PolyData(verts, faces)
    surf_s.point_data["S_norm"] = s_norm

    plotter.add_mesh(surf_s, scalars="S_norm", cmap=S_CMAP,
                     clim=[0.0, 1.0],
                     smooth_shading=True, specular=0.2, opacity=1.0,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "S / S\u2080",
                         "color": "white",
                         "title_font_size": 14,
                         "label_font_size": 11,
                         "width": 0.25,
                         "position_x": 0.05,
                         "position_y": 0.05,
                         "vertical": True,
                     })

    streamlines = build_streamlines_polydata(
        verts, normals, e1, e2, q1, q2, tris, s,
        n_seeds=streamline_seeds, n_steps=streamline_steps,
    )
    if streamlines is not None:
        plotter.add_mesh(streamlines, color="#333333", line_width=0.8, opacity=0.7)

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    plotter.add_text("Director field", position="upper_left",
                     font_size=10, color="white")

    # ── Right panel: vorticity field ──
    plotter.subplot(0, 1)
    plotter.set_background("#0a0a0a")

    if vel is not None:
        omega = compute_vorticity(vel, verts, tris, normals)
        omega_max = max(np.percentile(np.abs(omega), 98), 1e-10)
    else:
        omega = np.zeros(len(verts))
        omega_max = 1.0

    surf_w = pv.PolyData(verts, faces)
    surf_w.point_data["omega"] = omega

    plotter.add_mesh(surf_w, scalars="omega", cmap=OMEGA_CMAP,
                     clim=[-omega_max, omega_max],
                     smooth_shading=True, specular=0.2, opacity=1.0,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "\u03c9",
                         "color": "white",
                         "title_font_size": 14,
                         "label_font_size": 11,
                         "width": 0.25,
                         "position_x": 0.05,
                         "position_y": 0.05,
                         "vertical": True,
                     })

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    plotter.add_text("Vorticity \u03c9", position="upper_left",
                     font_size=10, color="white")

    plotter.screenshot(str(frame_path))
    plotter.close()


def main():
    parser = argparse.ArgumentParser(
        description="Render surface nematic: director (S colourmap) + vorticity")
    parser.add_argument("snapshot_dir", help="Directory with q_*.npy + mesh.json")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--orbit", action="store_true")
    parser.add_argument("--orbit-revs", type=float, default=0.25)
    parser.add_argument("--streamline-seeds", type=int, default=500)
    parser.add_argument("--streamline-steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None)
    parser.add_argument("--s-ref", type=float, default=None,
                        help="Reference S for normalisation (auto-detected if omitted)")
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    mesh_path = snap_dir / "mesh.json"
    if not mesh_path.exists():
        print(f"No mesh.json in {snap_dir}")
        return

    verts, faces, tris = load_mesh(mesh_path)
    nv = len(verts)

    normals = compute_vertex_normals(verts, tris)
    e1, e2 = compute_tangent_frame(verts, normals)

    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy in {snap_dir}")
        return

    # Check for velocity snapshots (may be one fewer than Q if last step has no vel).
    vel_snaps = sorted(snap_dir.glob("vel_*.npy"))
    vel_by_stem = {p.stem.replace("vel_", ""): p for p in vel_snaps}
    has_vel = len(vel_snaps) > 0
    if not has_vel:
        print(f"  No velocity snapshots (vel_*.npy); vorticity panel will be blank.")
    else:
        print(f"  Found {len(vel_snaps)} velocity snapshots.")

    # Auto-detect S reference from last snapshot.
    s_ref = args.s_ref
    if s_ref is None:
        q1_last, q2_last = load_q(snaps[-1], nv)
        s_last = 2.0 * np.sqrt(q1_last**2 + q2_last**2)
        s_ref = np.percentile(s_last, 95)
        if s_ref < 0.01:
            s_ref = 1.0
        print(f"  Auto S_ref = {s_ref:.4f}")

    out_dir = Path(args.output) if args.output else snap_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(snaps)} frames ({nv} vertices)...")
    n = len(snaps)
    window_size = (args.width, args.height)

    bbox_size = verts.max(axis=0) - verts.min(axis=0)
    cam_radius = np.linalg.norm(bbox_size) * 1.5

    for i, snap in enumerate(snaps):
        q1, q2 = load_q(snap, nv)
        q_stem = snap.stem.replace("q_", "")
        vel_path = vel_by_stem.get(q_stem)
        vel = load_velocity(vel_path, nv) if vel_path else None
        out_path = out_dir / f"frame_{i:06d}.png"

        cam_pos = None
        if args.orbit:
            angle = 2 * np.pi * args.orbit_revs * i / max(n - 1, 1)
            center = verts.mean(axis=0)
            cam_pos = [
                (center[0] + cam_radius * np.cos(angle),
                 center[1] + cam_radius * np.sin(angle),
                 center[2] + cam_radius * 0.35),
                tuple(center),
                (0, 0, 1),
            ]

        render_frame(verts, faces, tris, q1, q2, vel, out_path,
                     normals, e1, e2, s_ref,
                     streamline_seeds=args.streamline_seeds,
                     streamline_steps=args.streamline_steps,
                     camera_position=cam_pos, window_size=window_size)

        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"  {i+1}/{n}")

    if args.video:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(out_dir / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "slow", str(args.video),
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(args.video):
            mb = os.path.getsize(args.video) / 1e6
            print(f"Video: {args.video} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
