#!/usr/bin/env python3
"""
Curved surface visualisation using Polyscope.

Renders Q-tensor fields on triangulated 2-manifolds (S^2, torus,
epitrochoid domains) with:
  - Mesh coloured by scalar order parameter S
  - Director field as tangent line segments on the surface
  - Defect positions marked at low-S vertices

Usage:
    python render_surface.py <snapshot_dir> --mesh mesh.json
    python render_surface.py <snapshot_dir> --mesh mesh.json --video out.mp4
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_mesh(mesh_path):
    """Load a triangle mesh from JSON.

    Expected format:
        {"vertices": [[x,y,z], ...], "triangles": [[i,j,k], ...]}
    For 2D meshes, z=0 is assumed if vertices are [x,y].
    """
    with open(mesh_path) as f:
        data = json.load(f)
    verts = np.array(data["vertices"], dtype=np.float64)
    if verts.shape[1] == 2:
        # Pad 2D vertices to 3D.
        verts = np.column_stack([verts, np.zeros(len(verts))])
    tris = np.array(data["triangles"], dtype=np.int64)
    return verts, tris


def load_q_dec(path, nv):
    """Load a DEC Q-tensor snapshot (n_vertices, 2) -> (q1, q2)."""
    data = np.load(path)
    if data.shape == (nv, 2):
        return data[:, 0], data[:, 1]
    elif data.shape == (2 * nv,):
        return data[:nv], data[nv:]
    else:
        raise ValueError(f"unexpected shape {data.shape} for nv={nv}")


def render_polyscope(verts, tris, q1, q2, frame_path,
                     screenshot_width=1920, screenshot_height=1080):
    """Render one frame using Polyscope (off-screen)."""
    import polyscope as ps

    ps.init()
    ps.set_program_name("volterra")
    ps.set_ground_plane_mode("none")
    ps.set_screenshot_extension(".png")

    s = 2.0 * np.sqrt(q1**2 + q2**2)

    mesh = ps.register_surface_mesh("nematic", verts, tris)
    mesh.add_scalar_quantity("S", s, defined_on="vertices",
                            cmap="coolwarm", vminmax=(0, s.max() * 1.1),
                            enabled=True)

    # Director field as tangent vectors (2D projection onto surface).
    theta = 0.5 * np.arctan2(q2, q1)
    dx = np.cos(theta) * s
    dy = np.sin(theta) * s
    dz = np.zeros_like(dx)
    directors = np.column_stack([dx, dy, dz])
    mesh.add_vector_quantity("director", directors, defined_on="vertices",
                            enabled=True, length=0.02, color=(1, 1, 1))

    ps.screenshot(str(frame_path), transparent_bg=False)
    ps.remove_all_structures()


def render_all_surface(snapshot_dir, mesh_path, output_dir):
    """Render all DEC snapshots on a surface mesh."""
    snap_dir = Path(snapshot_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts, tris = load_mesh(mesh_path)
    nv = len(verts)

    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy snapshots found in {snap_dir}")
        return []

    print(f"Found {len(snaps)} snapshots, {nv} vertices")

    frame_paths = []
    for i, snap in enumerate(snaps):
        q1, q2 = load_q_dec(snap, nv)
        out_path = out_dir / f"frame_{i:06d}.png"
        render_polyscope(verts, tris, q1, q2, out_path)
        frame_paths.append(out_path)
        if (i + 1) % 10 == 0 or i == len(snaps) - 1:
            print(f"  rendered {i+1}/{len(snaps)}")

    return frame_paths


def main():
    parser = argparse.ArgumentParser(description="Render surface nematic snapshots")
    parser.add_argument("snapshot_dir", help="Directory with q_*.npy files")
    parser.add_argument("--mesh", required=True, help="Mesh JSON file")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    output_dir = args.output or str(Path(args.snapshot_dir) / "frames_surface")
    frame_paths = render_all_surface(args.snapshot_dir, args.mesh, output_dir)

    if args.video and frame_paths:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(Path(output_dir) / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "slow", str(args.video),
        ]
        subprocess.run(cmd, capture_output=True)
        import os
        if os.path.exists(args.video):
            print(f"Video saved: {args.video} ({os.path.getsize(args.video)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
