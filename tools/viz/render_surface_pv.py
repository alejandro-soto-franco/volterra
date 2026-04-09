#!/usr/bin/env python3
"""
Render DEC surface snapshots with director streamlines and defect colouring.

Director field rendered as connected streamlines (not disconnected rods).
Surface coloured by defect charge density:
  - Red: +1/2 defect regions (high winding, low S)
  - Blue: -1/2 defect regions (high winding, low S, opposite sense)
  - White/neutral: ordered nematic regions

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


def defect_charge_density(q1, q2, tris, verts):
    """Compute per-vertex defect charge density from the winding of the director
    angle around each vertex's 1-ring.

    Returns an array of values in [-1, 1] where:
      +0.5 at a +1/2 defect core
      -0.5 at a -1/2 defect core
      ~0 in ordered regions
    """
    nv = len(verts)
    theta = 0.5 * np.arctan2(q2, q1)

    # Build vertex adjacency (ordered 1-ring neighbours).
    # Use a simpler approach: accumulate angle jumps from incident triangles.
    winding = np.zeros(nv)
    count = np.zeros(nv)

    for tri in tris:
        i0, i1, i2 = tri
        # Angle differences around the triangle (mod pi for nematic).
        for (a, b, c) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)]:
            dtheta = theta[b] - theta[a]
            # Wrap to [-pi/2, pi/2] (nematic: director is headless).
            while dtheta > np.pi / 2:
                dtheta -= np.pi
            while dtheta < -np.pi / 2:
                dtheta += np.pi
            winding[a] += dtheta
            count[a] += 1

    # Normalise: total winding around a vertex = sum of angle jumps.
    # For a +1/2 defect: winding ~ pi. For -1/2: winding ~ -pi.
    # Charge = winding / (2*pi) for a full loop, but we have partial
    # coverage, so normalise by the expected full-loop contribution.
    charge = np.zeros(nv)
    for i in range(nv):
        if count[i] > 0:
            # The winding is accumulated from all incident triangles.
            # Divide by 2*pi to get the topological charge.
            charge[i] = winding[i] / np.pi

    # Smooth the charge field: average over 1-ring neighbours (Laplacian smoothing).
    # Repeat a few times to remove mesh-scale noise.
    adj = [[] for _ in range(nv)]
    for tri in tris:
        for a, b in [(0,1), (1,2), (2,0)]:
            adj[tri[a]].append(tri[b])
            adj[tri[b]].append(tri[a])

    for _ in range(3):  # 3 smoothing passes
        smoothed = np.zeros(nv)
        for i in range(nv):
            if len(adj[i]) > 0:
                neighbour_avg = np.mean([charge[j] for j in adj[i]])
                smoothed[i] = 0.5 * charge[i] + 0.5 * neighbour_avg
            else:
                smoothed[i] = charge[i]
        charge = smoothed

    return charge


def trace_streamline(start_idx, verts, normals, e1, e2, q1, q2, tris,
                     n_steps=30, step_size=None, vertex_adjacency=None):
    """Trace a director streamline starting from a vertex, projecting
    onto the surface at each step.

    Returns an array of 3D points along the streamline.
    """
    if step_size is None:
        # Auto: half the mean edge length.
        edge_lens = []
        for tri in tris[:100]:  # sample
            for a, b in [(0,1), (1,2), (2,0)]:
                edge_lens.append(np.linalg.norm(verts[tri[a]] - verts[tri[b]]))
        step_size = 0.5 * np.mean(edge_lens)

    nv = len(verts)
    pos = verts[start_idx].copy()
    points = [pos.copy()]

    for _ in range(n_steps):
        # Find nearest vertex to current position.
        dists = np.linalg.norm(verts - pos, axis=1)
        nearest = np.argmin(dists)

        # Director at nearest vertex.
        theta = 0.5 * np.arctan2(q2[nearest], q1[nearest])
        s = 2.0 * np.sqrt(q1[nearest]**2 + q2[nearest]**2)
        if s < 0.01:
            break  # inside defect core, stop

        # Director in 3D.
        dx = np.cos(theta) * e1[nearest] + np.sin(theta) * e2[nearest]

        # Step along the director (project back to surface).
        pos = pos + step_size * dx
        # Project onto surface: move toward nearest vertex's normal plane.
        n = normals[nearest]
        pos = pos - np.dot(pos - verts[nearest], n) * n

        # For sphere: re-normalise to unit sphere.
        r = np.linalg.norm(pos)
        if r > 0.5:  # only for sphere-like meshes
            pos = pos / r

        points.append(pos.copy())

    return np.array(points)


def render_frame(verts, faces, tris, q1, q2, frame_path,
                 normals, e1, e2,
                 streamline_seeds=500, streamline_steps=25,
                 camera_position=None, window_size=(1920, 1080)):
    """Render a frame with defect-coloured surface and director streamlines."""
    s = 2.0 * np.sqrt(q1**2 + q2**2)
    charge = defect_charge_density(q1, q2, tris, verts)

    # Colour field: red (+1/2), blue (-1/2), white (ordered).
    # Map charge to a diverging colourmap.
    surf = pv.PolyData(verts, faces)
    surf.point_data["charge"] = charge
    surf.point_data["S"] = s

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#0d1117")

    # Surface coloured by defect charge density.
    plotter.add_mesh(surf, scalars="charge", cmap="RdBu_r",
                     clim=[-0.6, 0.6],
                     smooth_shading=True, specular=0.3, opacity=0.9,
                     show_scalar_bar=True,
                     scalar_bar_args={
                         "title": "Defect charge",
                         "color": "white",
                         "title_font_size": 14,
                         "label_font_size": 11,
                         "width": 0.3,
                         "position_x": 0.35,
                     })

    # Director streamlines: trace from random seed vertices.
    nv = len(verts)
    rng = np.random.RandomState(42)
    seeds = rng.choice(nv, size=min(streamline_seeds, nv), replace=False)

    all_points = []
    all_lines = []
    offset = 0

    for seed in seeds:
        if s[seed] < 0.1:
            continue  # skip defect cores
        pts = trace_streamline(seed, verts, normals, e1, e2, q1, q2, tris,
                               n_steps=streamline_steps)
        if len(pts) < 2:
            continue

        n_pts = len(pts)
        # Offset points slightly above the surface.
        nearest_normals = np.array([normals[np.argmin(np.linalg.norm(verts - p, axis=1))] for p in pts])
        pts_offset = pts + 0.003 * nearest_normals

        all_points.append(pts_offset)
        line = np.zeros(n_pts + 1, dtype=np.int64)
        line[0] = n_pts
        line[1:] = np.arange(offset, offset + n_pts)
        all_lines.append(line)
        offset += n_pts

    if all_points:
        all_pts = np.vstack(all_points)
        all_ln = np.concatenate(all_lines)
        streamlines = pv.PolyData(all_pts, lines=all_ln)
        plotter.add_mesh(streamlines, color="white", line_width=1.2, opacity=0.7)

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.3)

    plotter.screenshot(str(frame_path))
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description="Render surface nematic with streamlines and defects")
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

    out_dir = Path(args.output) if args.output else snap_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(snaps)} frames ({nv} vertices, {args.streamline_seeds} streamlines)...")
    n = len(snaps)
    window_size = (args.width, args.height)

    bbox_size = verts.max(axis=0) - verts.min(axis=0)
    cam_radius = np.linalg.norm(bbox_size) * 1.5

    for i, snap in enumerate(snaps):
        q1, q2 = load_q(snap, nv)
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

        render_frame(verts, faces, tris, q1, q2, out_path,
                     normals, e1, e2,
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
