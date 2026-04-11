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


# ─── Streamlines ─────────────────────────────────────────────────────────────

def trace_streamline(start_idx, verts, normals, e1, e2, q1, q2, tris,
                     n_steps=30, step_size=None, direction=1.0):
    """Trace a director streamline from a vertex, projecting onto the surface."""
    if step_size is None:
        edge_lens = []
        for tri in tris[:100]:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge_lens.append(np.linalg.norm(verts[tri[a]] - verts[tri[b]]))
        step_size = 0.5 * np.mean(edge_lens)

    pos = verts[start_idx].copy()
    points = [pos.copy()]

    for _ in range(n_steps):
        dists = np.linalg.norm(verts - pos, axis=1)
        nearest = np.argmin(dists)

        theta = 0.5 * np.arctan2(q2[nearest], q1[nearest])
        s = 2.0 * np.sqrt(q1[nearest]**2 + q2[nearest]**2)
        if s < 0.01:
            break

        dx = np.cos(theta) * e1[nearest] + np.sin(theta) * e2[nearest]
        pos = pos + direction * step_size * dx
        n = normals[nearest]
        pos = pos - np.dot(pos - verts[nearest], n) * n

        r = np.linalg.norm(pos)
        if r > 0.5:
            pos = pos / r

        points.append(pos.copy())

    return np.array(points)


def build_streamlines_polydata(verts, normals, e1, e2, q1, q2, tris,
                               s, n_seeds=500, n_steps=25):
    """Build a single PolyData with all director streamlines."""
    nv = len(verts)
    rng = np.random.RandomState(42)
    seeds = rng.choice(nv, size=min(n_seeds, nv), replace=False)

    all_points = []
    all_lines = []
    offset = 0

    for seed in seeds:
        if s[seed] < 0.05:
            continue
        pts_fwd = trace_streamline(seed, verts, normals, e1, e2, q1, q2, tris,
                                   n_steps=n_steps, direction=1.0)
        pts_bwd = trace_streamline(seed, verts, normals, e1, e2, q1, q2, tris,
                                   n_steps=n_steps, direction=-1.0)

        if len(pts_bwd) > 1:
            pts = np.vstack([pts_bwd[::-1], pts_fwd[1:]])
        else:
            pts = pts_fwd
        if len(pts) < 3:
            continue

        n_pts = len(pts)
        nearest_idx = np.array([np.argmin(np.linalg.norm(verts - p, axis=1)) for p in pts])
        pts_offset = pts + 0.003 * normals[nearest_idx]

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

    # Check for velocity snapshots.
    vel_snaps = sorted(snap_dir.glob("vel_*.npy"))
    has_vel = len(vel_snaps) == len(snaps)
    if not has_vel:
        print(f"  No velocity snapshots (vel_*.npy); vorticity panel will be blank.")

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
        vel = load_velocity(vel_snaps[i], nv) if has_vel else None
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
