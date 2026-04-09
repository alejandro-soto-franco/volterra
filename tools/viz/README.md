# Visualisation Pipeline

Automated simulation-to-video pipeline for volterra active nematic simulations.

## Tools

| Script | Purpose | Dependencies |
|--------|---------|-------------|
| `pipeline.py` | One-command orchestrator: simulate, render, encode | matplotlib, ffmpeg |
| `render_2d.py` | 2D frame renderer (director field, S colourmap, defects) | matplotlib, numpy |
| `render_3d.py` | 3D frame renderer (isosurfaces, volume rendering) | pyvista, numpy |
| `render_surface.py` | Curved surface renderer (DEC meshes) | polyscope, numpy |

## Quick Start

```bash
# Full pipeline: simulate + render + encode
python tools/viz/pipeline.py cartesian-3d --n 50 --steps 5000 --zeta 3.0

# Render existing 2D snapshots
python tools/viz/render_2d.py /path/to/snapshots --nx 128 --ny 128 --video out.mp4

# Render existing 3D snapshots with PyVista
python tools/viz/render_3d.py /path/to/snapshots --nx 50 --ny 50 --nz 50 --orbit --video out.mp4

# Render DEC surface snapshots with Polyscope
python tools/viz/render_surface.py /path/to/snapshots --mesh mesh.json --video out.mp4
```

## Output

All pipeline output goes to `~/.volterra-bench/viz/<run_id>/`:
- `snapshots/`: `.npy` files from the simulation
- `meta.json`: simulation parameters
- `frames/`: PNG frames
- `active_nematic.mp4`: encoded video

## Rendering Conventions

Following the active nematic visualisation standards from the literature:

- **Director field**: white headless line segments, length proportional to S
- **Scalar order S**: coolwarm colourmap (blue = low S / defect cores, red = high S)
- **+1/2 defects**: red triangles pointing along self-propulsion direction
- **-1/2 defects**: blue inverted triangles
- **3D defect cores**: red isosurface at configurable S threshold
- **Vorticity**: diverging colourmap (blue CW, red CCW)

## Dependencies

```bash
pip install matplotlib numpy pyvista polyscope
# ffmpeg must be on PATH for video encoding
```
