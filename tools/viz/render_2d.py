#!/usr/bin/env python3
"""
Render 2D active nematic simulation snapshots as publication-quality frames.

Reads Q-tensor .npy snapshots from a simulation output directory and
produces .png frames showing:
  - Scalar order parameter S as a colourmap (blue = low, red = high)
  - Director field as headless line segments
  - Defect positions marked with triangles (+1/2) and pentagons (-1/2)

Usage:
    python render_2d.py <snapshot_dir> [--output <frame_dir>] [--dpi 150]
    python render_2d.py /tmp/volterra_sim --fps 30 --video out.mp4

Snapshot format:
    q_{step:06}.npy : shape (nx*ny, 2) float64, columns [q1, q2]
    OR shape (nx, ny, 2) float64
    meta.json : {"nx": N, "ny": N, "dx": 1.0, "dt": 0.01, ...}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_snapshot(path, nx, ny):
    """Load a Q-tensor snapshot and return (q1, q2) arrays of shape (nx, ny)."""
    data = np.load(path)
    if data.ndim == 1:
        # Flat [q1_0, q2_0, q1_1, q2_1, ...] or [q1_all, q2_all]
        if data.shape[0] == 2 * nx * ny:
            q1 = data[:nx*ny].reshape(nx, ny)
            q2 = data[nx*ny:].reshape(nx, ny)
        else:
            raise ValueError(f"unexpected 1D shape {data.shape} for nx={nx}, ny={ny}")
    elif data.ndim == 2 and data.shape == (nx * ny, 2):
        q1 = data[:, 0].reshape(nx, ny)
        q2 = data[:, 1].reshape(nx, ny)
    elif data.ndim == 2 and data.shape == (nx * ny, 5):
        # 3D Q-tensor stored as 5 components; take q11 and q12 for 2D projection
        q1 = data[:, 0].reshape(nx, ny)
        q2 = data[:, 1].reshape(nx, ny)
    elif data.ndim == 3 and data.shape == (nx, ny, 2):
        q1 = data[:, :, 0]
        q2 = data[:, :, 1]
    elif data.ndim == 4 and data.shape[3] == 5:
        # 3D Q-tensor (nx, ny, nz, 5): take the middle z-slice, use q11 and q12
        nz = data.shape[2]
        z_mid = nz // 2
        q1 = data[:, :, z_mid, 0]
        q2 = data[:, :, z_mid, 1]
    else:
        raise ValueError(f"unexpected shape {data.shape} for nx={nx}, ny={ny}")
    return q1, q2


def scalar_order(q1, q2):
    """Compute S = 2 sqrt(q1^2 + q2^2)."""
    return 2.0 * np.sqrt(q1**2 + q2**2)


def director_angle(q1, q2):
    """Director angle theta = atan2(q2, q1) / 2."""
    return 0.5 * np.arctan2(q2, q1)


def detect_defects_2d(q1, q2, threshold=0.3):
    """Simple defect detection: low-S regions below threshold.

    Returns list of (i, j, charge_sign) tuples.
    For proper holonomy-based detection, use volterra's scan_defects.
    """
    s = scalar_order(q1, q2)
    defects = []
    nx, ny = q1.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if s[i, j] < threshold and s[i, j] == np.min(s[i-1:i+2, j-1:j+2]):
                # Estimate charge from winding number around the defect
                angles = []
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    angles.append(director_angle(q1[i+di, j+dj], q2[i+di, j+dj]))
                winding = 0.0
                for k in range(4):
                    diff = angles[(k+1) % 4] - angles[k]
                    # Wrap to [-pi/2, pi/2] (nematic symmetry)
                    while diff > np.pi / 2:
                        diff -= np.pi
                    while diff < -np.pi / 2:
                        diff += np.pi
                    winding += diff
                charge = round(winding / np.pi * 2) / 2  # quantise to half-integer
                if abs(charge) > 0.1:
                    defects.append((i, j, int(np.sign(charge))))
    return defects


def render_frame(q1, q2, dx, frame_idx, output_path, title=None,
                 dpi=150, director_stride=4, s_range=(0.0, 1.0)):
    """Render a single frame to PNG."""
    nx, ny = q1.shape
    s = scalar_order(q1, q2)
    theta = director_angle(q1, q2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Scalar order colourmap
    extent = [0, ny * dx, 0, nx * dx]
    im = ax.imshow(s, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=s_range[0], vmax=s_range[1], interpolation="bilinear")

    # Director field: headless line segments with length proportional to S.
    stride = director_stride
    y_grid, x_grid = np.mgrid[0:nx:stride, 0:ny:stride]
    x_pos = (x_grid + 0.5) * dx
    y_pos = (y_grid + 0.5) * dx
    th = theta[::stride, ::stride]
    s_sub = s[::stride, ::stride]
    s_max = max(s_range[1], 0.01)
    length = 0.4 * stride * dx * np.clip(s_sub / s_max, 0.1, 1.0)
    dx_dir = length * np.cos(th)
    dy_dir = length * np.sin(th)

    ax.quiver(x_pos, y_pos, dx_dir, dy_dir,
              headaxislength=0, headlength=0, headwidth=0,
              pivot="middle", scale_units="xy", scale=1,
              color="white", alpha=0.8, linewidth=0.6)

    # Defect markers
    defects = detect_defects_2d(q1, q2)
    for di, dj, sign in defects:
        marker = "^" if sign > 0 else "v"
        colour = "#e74c3c" if sign > 0 else "#3498db"
        ax.plot((dj + 0.5) * dx, (di + 0.5) * dx,
                marker=marker, color=colour, markersize=8,
                markeredgecolor="white", markeredgewidth=0.5)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Scalar order $S$")

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"Step {frame_idx}", fontsize=12)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_all(snapshot_dir, output_dir, nx, ny, dx=1.0, dpi=150,
               director_stride=4, s_range=None):
    """Render all snapshots in a directory to PNG frames."""
    snap_dir = Path(snapshot_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all snapshot files, sorted by step number
    snaps = sorted(snap_dir.glob("q_*.npy"))
    if not snaps:
        print(f"No q_*.npy snapshots found in {snap_dir}")
        return []

    print(f"Found {len(snaps)} snapshots in {snap_dir}")

    # Auto-detect S range from first and last snapshot
    if s_range is None:
        q1_first, q2_first = load_snapshot(snaps[0], nx, ny)
        q1_last, q2_last = load_snapshot(snaps[-1], nx, ny)
        s_max = max(scalar_order(q1_first, q2_first).max(),
                    scalar_order(q1_last, q2_last).max())
        s_range = (0.0, min(s_max * 1.1, 2.0))

    frame_paths = []
    for i, snap in enumerate(snaps):
        step = int(snap.stem.split("_")[1])
        q1, q2 = load_snapshot(snap, nx, ny)
        out_path = out_dir / f"frame_{i:06d}.png"
        render_frame(q1, q2, dx, step, out_path, dpi=dpi,
                     director_stride=director_stride, s_range=s_range)
        frame_paths.append(out_path)
        if (i + 1) % 10 == 0 or i == len(snaps) - 1:
            print(f"  rendered {i+1}/{len(snaps)}")

    return frame_paths


def frames_to_video(frame_dir, output_path, fps=30):
    """Stitch PNG frames into an MP4 video using ffmpeg."""
    import subprocess
    frame_pattern = str(Path(frame_dir) / "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(output_path),
    ]
    print(f"Encoding video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    print(f"Video saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render 2D active nematic snapshots")
    parser.add_argument("snapshot_dir", help="Directory containing q_*.npy files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for frames (default: <snapshot_dir>/frames)")
    parser.add_argument("--nx", type=int, default=None, help="Grid size in x")
    parser.add_argument("--ny", type=int, default=None, help="Grid size in y")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PNG frames")
    parser.add_argument("--stride", type=int, default=4, help="Director field stride")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate")
    parser.add_argument("--video", default=None, help="Output video path (triggers encoding)")
    args = parser.parse_args()

    snap_dir = Path(args.snapshot_dir)

    # Try to load metadata
    meta_path = snap_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        nx = args.nx or meta.get("nx")
        ny = args.ny or meta.get("ny")
        dx = meta.get("dx", args.dx)
    else:
        nx = args.nx
        ny = args.ny
        dx = args.dx

    if nx is None or ny is None:
        # Try to infer from first snapshot
        snaps = sorted(snap_dir.glob("q_*.npy"))
        if snaps:
            data = np.load(snaps[0])
            if data.ndim == 2:
                total = data.shape[0]
                nx = ny = int(np.sqrt(total))
                print(f"Inferred nx=ny={nx} from snapshot shape {data.shape}")
            else:
                print("Cannot infer grid size. Pass --nx and --ny.")
                sys.exit(1)
        else:
            print("No snapshots found and no grid size specified.")
            sys.exit(1)

    output_dir = args.output or str(snap_dir / "frames")

    frame_paths = render_all(snap_dir, output_dir, nx, ny, dx,
                             dpi=args.dpi, director_stride=args.stride)

    if args.video and frame_paths:
        frames_to_video(output_dir, args.video, fps=args.fps)


if __name__ == "__main__":
    main()
