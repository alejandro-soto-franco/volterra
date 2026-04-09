#!/usr/bin/env python3
"""
Render DEC surface snapshots using PyVista (off-screen).

Reads Q-tensor .npy snapshots and a mesh.json, renders the surface
coloured by scalar order S with director field overlay.

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
    verts = None
    tris = None
    with open(mesh_path) as f:
        data = json.load(f)
    verts = np.array(data["vertices"], dtype=np.float64)
    tris = np.array(data["triangles"], dtype=np.int64)
    # PyVista face format: [n_pts, i0, i1, i2, ...]
    n_faces = len(tris)
    faces = np.column_stack([np.full(n_faces, 3, dtype=np.int64), tris]).ravel()
    return verts, faces, n_faces


def load_q(path, nv):
    data = np.load(path)
    if data.shape == (nv, 2):
        return data[:, 0], data[:, 1]
    elif data.ndim == 1 and data.shape[0] == 2 * nv:
        return data[:nv], data[nv:]
    else:
        # Try interleaved [q1_0, q2_0, q1_1, q2_1, ...]
        flat = data.ravel()
        if flat.shape[0] == 2 * nv:
            q1 = flat[0::2]
            q2 = flat[1::2]
            return q1, q2
        raise ValueError(f"unexpected shape {data.shape} for nv={nv}")


def render_frame(verts, faces, q1, q2, frame_path,
                 camera_position=None, window_size=(1920, 1080)):
    s = 2.0 * np.sqrt(q1**2 + q2**2)

    surf = pv.PolyData(verts, faces)
    surf.point_data["S"] = s

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#0d1117")

    plotter.add_mesh(surf, scalars="S", cmap="coolwarm",
                     smooth_shading=True, specular=0.4,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "Scalar order S",
                         "color": "white",
                         "title_font_size": 14,
                         "label_font_size": 12,
                     })

    # Wireframe overlay (subtle).
    plotter.add_mesh(surf, style="wireframe", color="white",
                     opacity=0.05, line_width=0.5)

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    plotter.screenshot(str(frame_path))
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description="Render surface nematic snapshots (PyVista)")
    parser.add_argument("snapshot_dir", help="Directory with q_*.npy + mesh.json")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--orbit", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    mesh_path = snap_dir / "mesh.json"
    if not mesh_path.exists():
        print(f"No mesh.json in {snap_dir}")
        return

    verts, faces, _ = load_mesh(mesh_path)
    nv = len(verts)

    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy in {snap_dir}")
        return

    out_dir = Path(args.output) if args.output else snap_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(snaps)} frames ({nv} vertices)...")
    n = len(snaps)
    window_size = (args.width, args.height)

    for i, snap in enumerate(snaps):
        q1, q2 = load_q(snap, nv)
        out_path = out_dir / f"frame_{i:06d}.png"

        cam_pos = None
        if args.orbit:
            angle = 2 * np.pi * i / max(n, 1)
            r = 8.0 if "torus" in str(snap_dir) else 3.5
            cam_pos = [
                (r * np.cos(angle), r * np.sin(angle), r * 0.4),
                (0, 0, 0),
                (0, 0, 1),
            ]

        render_frame(verts, faces, q1, q2, out_path,
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
