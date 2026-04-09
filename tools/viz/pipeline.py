#!/usr/bin/env python3
"""
Automated simulation-to-video pipeline for volterra.

One command to: run simulation -> render frames -> encode video.

Usage:
    # 2D active nematic on flat torus (Cartesian solver)
    python pipeline.py cartesian-2d --nx 128 --ny 128 --zeta 3.0 --steps 10000

    # 2D dry active nematic on DEC flat mesh
    python pipeline.py dec-2d --n 16 --steps 1000

    # 3D dry active nematic (Cartesian)
    python pipeline.py cartesian-3d --n 50 --steps 5000

    # Render existing snapshots only (no simulation)
    python pipeline.py render --dir /path/to/snapshots --nx 128 --ny 128

All output goes to ~/.volterra-bench/viz/<run_id>/ by default.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIZ_ROOT = Path.home() / ".volterra-bench" / "viz"
VOLTERRA_ROOT = Path(__file__).resolve().parent.parent.parent


def run_id(prefix, params):
    """Generate a deterministic run ID from parameters."""
    param_str = json.dumps(params, sort_keys=True)
    h = hashlib.sha256(param_str.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def ensure_built():
    """Ensure volterra is built in release mode."""
    print("[pipeline] Building volterra (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--workspace"],
        cwd=str(VOLTERRA_ROOT),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    print("[pipeline] Build OK")


def run_simulation_3d(out_dir, params):
    """Run a 3D Cartesian dry active nematic simulation."""
    n = params["n"]
    steps = params["steps"]
    snap = params.get("snap_every", max(1, steps // 200))
    zeta = params.get("zeta", 2.0)
    dt = params.get("dt", 0.001)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a small Rust example that runs the simulation and writes snapshots.
    # The existing run_dry_active_nematic_3d already writes .npy files.
    # Write the metadata JSON separately to avoid format! brace escaping issues.
    meta_json = json.dumps({"nx": n, "ny": n, "nz": n, "dx": 1.0, "dt": dt,
                            "zeta": zeta, "steps": steps, "snap_every": snap})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(meta_json)

    out_dir_str = str(out_dir).replace("\\", "/")
    example_src = f"""
use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::run_dry_active_nematic_3d;
fn main() {{
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = {n}; p.ny = {n}; p.nz = {n};
    p.zeta_eff = {zeta}_f64; p.noise_amp = 0.01; p.dt = {dt}_f64;
    let q0 = QField3D::random_perturbation({n}, {n}, {n}, p.dx, 0.01, 42);
    let out = std::path::Path::new("{out_dir_str}");
    std::fs::create_dir_all(out).ok();
    let _ = run_dry_active_nematic_3d(&q0, &p, {steps}, {snap}, out, false);
}}
"""
    example_path = VOLTERRA_ROOT / "volterra-solver" / "examples" / "_pipeline_sim.rs"
    example_path.write_text(example_src)

    print(f"[pipeline] Running 3D simulation: N={n}, steps={steps}, zeta={zeta}")
    t0 = time.time()
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "volterra-solver", "--example", "_pipeline_sim"],
        cwd=str(VOLTERRA_ROOT),
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0

    example_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"Simulation failed:\n{result.stderr}")
        sys.exit(1)

    n_snaps = len(list(out_dir.glob("q_*.npy")))
    print(f"[pipeline] Simulation complete: {n_snaps} snapshots in {elapsed:.1f}s")
    return n_snaps


def run_render(snap_dir, frame_dir, params):
    """Render snapshots to PNG frames."""
    render_script = Path(__file__).parent / "render_2d.py"
    nx = params.get("nx", params.get("n"))
    ny = params.get("ny", params.get("n"))

    cmd = [
        sys.executable, str(render_script),
        str(snap_dir),
        "--output", str(frame_dir),
        "--nx", str(nx),
        "--ny", str(ny),
        "--dpi", str(params.get("dpi", 150)),
        "--stride", str(params.get("stride", max(1, nx // 32))),
    ]

    print(f"[pipeline] Rendering frames: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Render failed:\n{result.stderr}")
        return False
    print(result.stdout)
    return True


def encode_video(frame_dir, video_path, fps=30):
    """Encode frames to MP4."""
    if not shutil.which("ffmpeg"):
        print("[pipeline] ffmpeg not found, skipping video encoding")
        return False

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
        str(video_path),
    ]

    print(f"[pipeline] Encoding video at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[:500]}")
        return False
    size_mb = os.path.getsize(video_path) / 1e6
    print(f"[pipeline] Video saved: {video_path} ({size_mb:.1f} MB)")
    return True


def cmd_cartesian_3d(args):
    params = {
        "n": args.n, "steps": args.steps, "zeta": args.zeta,
        "dt": args.dt, "snap_every": args.snap_every,
        "dpi": args.dpi, "stride": max(1, args.n // 32),
    }
    rid = run_id("cart3d", params)
    out = VIZ_ROOT / rid
    snap_dir = out / "snapshots"
    frame_dir = out / "frames"
    video_path = out / "active_nematic.mp4"

    ensure_built()
    run_simulation_3d(snap_dir, params)

    # For 3D, render a z-slice (middle plane)
    params["nx"] = args.n
    params["ny"] = args.n
    run_render(snap_dir, frame_dir, params)

    if not args.no_video:
        encode_video(frame_dir, video_path, fps=args.fps)

    print(f"\n[pipeline] Output directory: {out}")


def cmd_render(args):
    snap_dir = Path(args.dir)
    frame_dir = snap_dir / "frames"
    params = {"nx": args.nx, "ny": args.ny, "dpi": args.dpi, "stride": args.stride}
    run_render(snap_dir, frame_dir, params)
    if args.video:
        encode_video(frame_dir, args.video, fps=args.fps)


def main():
    parser = argparse.ArgumentParser(description="volterra simulation-to-video pipeline")
    sub = parser.add_subparsers(dest="command")

    # 3D Cartesian
    p3d = sub.add_parser("cartesian-3d", help="3D periodic active nematic")
    p3d.add_argument("--n", type=int, default=50, help="Grid size per dimension")
    p3d.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    p3d.add_argument("--zeta", type=float, default=2.0, help="Activity")
    p3d.add_argument("--dt", type=float, default=0.001, help="Time step")
    p3d.add_argument("--snap-every", type=int, default=25, help="Snapshot interval")
    p3d.add_argument("--fps", type=int, default=30, help="Video FPS")
    p3d.add_argument("--dpi", type=int, default=150, help="Frame DPI")
    p3d.add_argument("--no-video", action="store_true", help="Skip video encoding")

    # Render existing snapshots
    pr = sub.add_parser("render", help="Render existing snapshots")
    pr.add_argument("--dir", required=True, help="Snapshot directory")
    pr.add_argument("--nx", type=int, required=True)
    pr.add_argument("--ny", type=int, required=True)
    pr.add_argument("--dpi", type=int, default=150)
    pr.add_argument("--stride", type=int, default=4)
    pr.add_argument("--fps", type=int, default=30)
    pr.add_argument("--video", default=None, help="Output video path")

    args = parser.parse_args()
    if args.command == "cartesian-3d":
        cmd_cartesian_3d(args)
    elif args.command == "render":
        cmd_render(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
