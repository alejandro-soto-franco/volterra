#!/usr/bin/env python3
"""
Render DEC surface snapshots using PyVista (off-screen).

Shows the nematic director field as headless rod segments on the mesh
surface, coloured by the scalar order parameter S.

Usage:
    python render_surface_pv.py ~/.volterra-bench/viz/sphere --video sphere.mp4
    python render_surface_pv.py ~/.volterra-bench/viz/torus --video torus.mp4 --orbit
"""

import argparse
import json
import subprocess
import os
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


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


def compute_vertex_normals(verts, tris):
    """Compute area-weighted vertex normals from triangle mesh."""
    normals = np.zeros_like(verts)
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        for idx in tri:
            normals[idx] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    return normals / norms


def compute_tangent_frame(verts, normals):
    """Build a local tangent frame (e1, e2) at each vertex.

    e1 is chosen by projecting a reference direction onto the tangent plane
    and normalising. e2 = n x e1.
    """
    nv = len(verts)
    e1 = np.zeros((nv, 3))
    e2 = np.zeros((nv, 3))

    # Reference direction: pick the axis least aligned with the normal.
    for i in range(nv):
        n = normals[i]
        # Choose reference: x-axis unless normal is nearly x-aligned.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        # Project reference onto tangent plane.
        t = ref - np.dot(ref, n) * n
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-14:
            t = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), n) * n
            t_norm = np.linalg.norm(t)
        e1[i] = t / t_norm
        e2[i] = np.cross(n, e1[i])

    return e1, e2


def director_vectors(q1, q2, verts, normals, e1, e2):
    """Compute 3D director vectors from Q-tensor components and tangent frame.

    theta = atan2(q2, q1) / 2 gives the director angle in the tangent plane.
    The 3D director is cos(theta)*e1 + sin(theta)*e2, scaled by S.
    """
    theta = 0.5 * np.arctan2(q2, q1)
    s = 2.0 * np.sqrt(q1**2 + q2**2)

    ct = np.cos(theta)
    st = np.sin(theta)

    # Director in 3D: d = cos(theta)*e1 + sin(theta)*e2, scaled by S.
    dx = ct[:, None] * e1 + st[:, None] * e2
    # Scale length by S (normalised to max S for visibility).
    s_max = max(np.percentile(s, 95), 0.01)
    scale = np.clip(s / s_max, 0.1, 1.0)
    dx *= scale[:, None]

    return dx, s


def render_frame(verts, faces, tris, q1, q2, frame_path,
                 normals=None, e1=None, e2=None,
                 rod_stride=1, rod_length=0.06,
                 camera_position=None, window_size=(1920, 1080)):
    """Render one frame with S colourmap and director rod field."""
    directors, s = director_vectors(q1, q2, verts, normals, e1, e2)

    surf = pv.PolyData(verts, faces)
    surf.point_data["S"] = s

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#0d1117")

    # Surface coloured by S.
    plotter.add_mesh(surf, scalars="S", cmap="coolwarm",
                     clim=[0, np.percentile(s, 98) * 1.05],
                     smooth_shading=True, specular=0.3, opacity=0.85,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "Scalar order $S$",
                         "color": "white",
                         "title_font_size": 16,
                         "label_font_size": 12,
                         "width": 0.3,
                         "position_x": 0.35,
                     })

    # Director field as rod segments (headless lines).
    # Subsample vertices for readability.
    idx = np.arange(0, len(verts), rod_stride)
    pts = verts[idx]
    dirs = directors[idx] * rod_length
    s_sub = s[idx]

    # Offset rods slightly above the surface so they don't z-fight.
    offset = normals[idx] * rod_length * 0.15
    starts = pts + offset - dirs * 0.5
    ends = pts + offset + dirs * 0.5

    n_rods = len(idx)
    rod_points = np.empty((2 * n_rods, 3))
    rod_points[0::2] = starts
    rod_points[1::2] = ends
    lines = np.column_stack([
        np.full(n_rods, 2, dtype=np.int64),
        np.arange(0, 2 * n_rods, 2),
        np.arange(1, 2 * n_rods, 2),
    ]).ravel()

    rods = pv.PolyData(rod_points, lines=lines)
    # Colour rods white with varying opacity by S.
    plotter.add_mesh(rods, color="white", line_width=2, opacity=0.9)

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    plotter.screenshot(str(frame_path))
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description="Render surface nematic with director rods (PyVista)")
    parser.add_argument("snapshot_dir", help="Directory with q_*.npy + mesh.json")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--orbit", action="store_true")
    parser.add_argument("--orbit-revs", type=float, default=0.5,
                        help="Number of full revolutions over the video (default 0.5)")
    parser.add_argument("--rod-stride", type=int, default=1,
                        help="Show every Nth vertex director (default: all)")
    parser.add_argument("--rod-length", type=float, default=None,
                        help="Rod segment length (auto-scaled if omitted)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    mesh_path = snap_dir / "mesh.json"
    if not mesh_path.exists():
        print(f"No mesh.json in {snap_dir}")
        return

    verts, faces, tris = load_mesh(mesh_path)
    nv = len(verts)

    # Precompute normals and tangent frame (constant across all frames).
    normals = compute_vertex_normals(verts, tris)
    e1, e2 = compute_tangent_frame(verts, normals)

    # Auto-scale rod length from mean edge length.
    if args.rod_length is None:
        edge_lengths = []
        for tri in tris:
            for a, b in [(0,1), (1,2), (2,0)]:
                edge_lengths.append(np.linalg.norm(verts[tri[a]] - verts[tri[b]]))
        rod_length = 0.8 * np.mean(edge_lengths)
    else:
        rod_length = args.rod_length

    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy in {snap_dir}")
        return

    out_dir = Path(args.output) if args.output else snap_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(snaps)} frames ({nv} vertices, rod_length={rod_length:.3f})...")
    n = len(snaps)
    window_size = (args.width, args.height)

    # Determine camera orbit radius from mesh bounding box.
    bbox_size = verts.max(axis=0) - verts.min(axis=0)
    cam_radius = np.linalg.norm(bbox_size) * 1.5

    for i, snap in enumerate(snaps):
        q1, q2 = load_q(snap, nv)
        out_path = out_dir / f"frame_{i:06d}.png"

        cam_pos = None
        if args.orbit:
            # Slow orbit: args.orbit_revs full turns over all frames.
            angle = 2 * np.pi * args.orbit_revs * i / max(n - 1, 1)
            center = verts.mean(axis=0)
            cam_pos = [
                (center[0] + cam_radius * np.cos(angle),
                 center[1] + cam_radius * np.sin(angle),
                 center[2] + cam_radius * 0.35),
                tuple(center),
                (0, 0, 1),
            ]

        render_frame(verts, faces, tris, q1, q2, out_path,
                     normals=normals, e1=e1, e2=e2,
                     rod_stride=args.rod_stride, rod_length=rod_length,
                     camera_position=cam_pos, window_size=window_size)

        if (i + 1) % 10 == 0 or i == n - 1:
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
