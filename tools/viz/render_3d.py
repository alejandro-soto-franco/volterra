#!/usr/bin/env python3
"""
3D active nematic visualisation using PyVista.

Renders Q-tensor snapshots as:
  - Scalar order parameter S as volume colourmap (transparent high-S, opaque low-S)
  - Isosurface of S at a configurable threshold (reveals defect cores)
  - Director field as oriented glyphs on a subsampled grid
  - Velocity streamlines (when velocity data is available)

Usage:
    python render_3d.py <snapshot_dir> [--output <frame_dir>] [--threshold 0.15]
    python render_3d.py /tmp/volterra_sim --video out.mp4 --orbit
"""

import argparse
import json
from pathlib import Path

import numpy as np

# PyVista must be imported after setting the backend for off-screen rendering.
import pyvista as pv
pv.OFF_SCREEN = True


def load_q_3d(path, nx, ny, nz):
    """Load a 3D Q-tensor snapshot. Returns (q11, q12, q13, q22, q23) arrays."""
    data = np.load(path)
    if data.shape == (nx, ny, nz, 5):
        return tuple(data[..., c] for c in range(5))
    elif data.shape == (nx * ny * nz, 5):
        data = data.reshape(nx, ny, nz, 5)
        return tuple(data[..., c] for c in range(5))
    else:
        raise ValueError(f"unexpected shape {data.shape} for ({nx},{ny},{nz})")


def scalar_order_3d(q11, q12, q13, q22, q23):
    """Compute S from 5-component Q-tensor (3D).

    S = (3/2) * max eigenvalue of Q. For speed, use the Frobenius norm
    approximation: S ~ sqrt(3/2 * Tr(Q^2)) which is exact for uniaxial Q.
    """
    q33 = -(q11 + q22)
    tr_q2 = q11**2 + q22**2 + q33**2 + 2*(q12**2 + q13**2 + q23**2)
    return np.sqrt(1.5 * tr_q2)


def render_isosurface(grid, s_data, threshold, frame_path,
                      camera_position=None, window_size=(1920, 1080)):
    """Render an isosurface of S at the given threshold."""
    grid.point_data["S"] = s_data.ravel(order="F")

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("black")

    # Volume rendering: transparent where S is high, opaque where S is low
    # (defect cores have low S).
    contour = grid.contour([threshold], scalars="S")
    if contour.n_points > 0:
        plotter.add_mesh(contour, color="#e74c3c", opacity=0.7,
                         smooth_shading=True, specular=0.5)

    # Semi-transparent volume of the full S field.
    plotter.add_mesh(grid.outline(), color="white", line_width=1)

    # Add a scalar bar.
    plotter.add_mesh(grid, scalars="S", cmap="coolwarm",
                     opacity="sigmoid_5", show_scalar_bar=True,
                     scalar_bar_args={"title": "Scalar order S"})

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.2)

    plotter.screenshot(str(frame_path))
    plotter.close()


def render_all_3d(snapshot_dir, output_dir, nx, ny, nz, dx=1.0,
                  threshold=None, window_size=(1920, 1080),
                  orbit=False):
    """Render all 3D snapshots to PNG frames."""
    snap_dir = Path(snapshot_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy snapshots found in {snap_dir}")
        return []

    print(f"Found {len(snaps)} 3D snapshots")

    # Create the structured grid (same for all frames).
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dx, dx))

    # Auto-detect threshold from the last snapshot if not specified.
    if threshold is None:
        q_last = load_q_3d(snaps[-1], nx, ny, nz)
        s_last = scalar_order_3d(*q_last)
        threshold = max(0.05, 0.3 * np.median(s_last))
        print(f"Auto threshold: S = {threshold:.3f}")

    frame_paths = []
    n_frames = len(snaps)
    for i, snap in enumerate(snaps):
        q_components = load_q_3d(snap, nx, ny, nz)
        s = scalar_order_3d(*q_components)

        out_path = out_dir / f"frame_{i:06d}.png"

        # Camera orbit: rotate around the z-axis.
        cam_pos = None
        if orbit:
            angle = 2 * np.pi * i / max(n_frames, 1)
            r = 2.5 * nx * dx
            cam_pos = [
                (r * np.cos(angle), r * np.sin(angle), 0.8 * nx * dx),
                (nx*dx/2, ny*dx/2, nz*dx/2),
                (0, 0, 1),
            ]

        render_isosurface(grid, s, threshold, out_path,
                          camera_position=cam_pos, window_size=window_size)
        frame_paths.append(out_path)

        if (i + 1) % 5 == 0 or i == n_frames - 1:
            print(f"  rendered {i+1}/{n_frames}")

    return frame_paths


def frames_to_video(frame_dir, output_path, fps=30):
    """Stitch frames into MP4."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(frame_dir) / "frame_%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[:300]}")
        return False
    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Video saved: {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render 3D active nematic snapshots")
    parser.add_argument("snapshot_dir", help="Directory containing q_*.npy files")
    parser.add_argument("--output", "-o", default=None,
                        help="Frame output directory (default: <snapshot_dir>/frames_3d)")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="S isosurface threshold (auto-detected if omitted)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--orbit", action="store_true",
                        help="Orbit the camera around the simulation box")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video", default=None, help="Output video path")
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)
    meta_path = snap_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        nx = args.nx or meta.get("nx")
        ny = args.ny or meta.get("ny")
        nz = args.nz or meta.get("nz", nx)
        dx = meta.get("dx", args.dx)
    else:
        nx, ny, nz = args.nx, args.ny, args.nz
        dx = args.dx

    if not all([nx, ny, nz]):
        print("Grid dimensions required. Pass --nx, --ny, --nz or provide meta.json.")
        return

    output_dir = args.output or str(snap_dir / "frames_3d")
    window_size = (args.width, args.height)

    frame_paths = render_all_3d(snap_dir, output_dir, nx, ny, nz, dx,
                                threshold=args.threshold,
                                window_size=window_size,
                                orbit=args.orbit)

    if args.video and frame_paths:
        frames_to_video(output_dir, args.video, fps=args.fps)


if __name__ == "__main__":
    main()
