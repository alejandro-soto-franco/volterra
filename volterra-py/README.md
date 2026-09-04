# volterra-py

Python bindings for the volterra active nematics simulation library.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-py` exposes volterra's Rust simulation engine to Python via PyO3. The PyPI package is published as `volterra-nematic`, but the Python module is imported as `import volterra`. NumPy arrays are used for all field data interchange with zero-copy where possible.

## Installation

```bash
pip install volterra-nematic
```

## Exposed API

| Python name | Description |
|-------------|-------------|
| `volterra.ActiveNematicParams` | Physical and numerical parameters |
| `volterra.QField2D` | 2D Q-tensor field with NumPy interop |
| `volterra.DefectInfo` | Detected disclination (position, charge, frame) |
| `volterra.SnapStats` | Per-snapshot statistics |
| `volterra.run_dry_active_nematic` | Component 1 (dry active nematic) runner |
| `volterra.k0_convolution` | K0 transfer map (Component 2) |
| `volterra.scan_defects` | Holonomy-based defect detection |
| `volterra.PlaneCurve` | A closed wall: analytic, tabulated, or from a parametrisation |
| `volterra.ConfinedMesh` | Boundary-conforming graded mesh of its interior |
| `volterra.confined_mesh` | Build one from a curve |
| `volterra.ConfinedRun` | A confined active nematic run, stepped from Python |

## Example

```python
import numpy as np
import volterra

params = volterra.ActiveNematicParams(
    nx=128, ny=128, dx=1.0, dt=0.005,
    k_r=0.04, gamma_r=0.5, zeta_eff=0.07, eta=1.0,
    a_landau=-0.1, c_landau=0.1, lambda_=0.7,
    k_l=0.01, gamma_l=0.1, xi_l=5.0,
)
q0 = volterra.QField2D.random_perturbation(params.nx, params.ny, params.dx, 0.001, 42)
q_final, snapshots = volterra.run_dry_active_nematic(q0, params, n_steps=10000, snap_every=500)
S = np.asarray(q_final.order_param()).reshape(params.nx, params.ny)
```

## Confined boundary geometry

A confined run is set up by naming the wall. `PlaneCurve` takes it three ways
and `confined_mesh` returns the mesh conforming to it, with every boundary
vertex on the curve and its inward normal beside it.

```python
import numpy as np
import volterra as v

# Analytic: 2(q - 1) cusps, regularised by d, at outer scale r.
nephroid = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=60.0)

# From a parametrisation, for a wall with no closed form in the library.
flower = v.PlaneCurve.from_callable(
    lambda u: ((1 + 0.3 * np.cos(5 * u)) * 60 * np.cos(u),
               (1 + 0.3 * np.cos(5 * u)) * 60 * np.sin(u)),
    samples=1200,
)

# From a table of points, closed on itself, with the corners named. Any
# (n, 2) sequence works, a list of pairs included.
square = v.PlaneCurve.from_points(xy, features=[0.0, 200.0, 400.0, 600.0])

m = v.confined_mesh(flower, h_bulk=1.6, h_min=0.5)
charge, worst_step_deg, over = m.imposed_charge(q_anchor=1.0)
q_wall = m.anchoring_q(q_anchor=1.0, s0=1.0)   # Dirichlet Q, one row per wall vertex
```

`imposed_charge` is the reading to take before spending time on a run. It
returns the total defect charge the anchoring puts in the interior, measured on
this mesh's own boundary, with the worst doubled-angle step beside it. A step
past 90 degrees means the sampling has booked the wrong branch, which is what a
lattice mask does at a cusp and what a right-angle corner does at any sampling
density. Round such a corner over a radius the boundary resolves.

`volterra-py/examples/new_boundary_geometry.py` builds four walls this way and
draws them with their anchored directors.

## Running a confined nematic

`ConfinedRun` steps the same scheme the Rust driver runs. Each step solves the
Stokes problem for the velocity from the Beris-Edwards stress, then advances Q
semi-implicitly with the Frank term implicit, and re-imposes the anchoring on
the wall.

```python
run = v.ConfinedRun(
    mesh,
    active_length=3.5,        # sqrt(K / zeta)
    coherence_length=4.0,     # sqrt(K / C), the defect core size
    resolution=80,
    q_anchor=1.0,             # 1 is planar anchoring along the wall tangent
    wall="noslip",            # or "freeslip"
    dt=2e-4,
    seed=1,
)

run.relax(200)                # optional passive settle
run.step(4000)

run.q                         # (n_vertices, 2), the Q field
run.velocity                  # (n_vertices, 2)
run.order_parameter           # (n_vertices,)
run.director_angle            # (n_vertices,)
run.defects()                 # [(x, y, charge), ...]
run.stats()                   # step, time, n_plus, n_minus, charge, speeds
```

The `wall` argument is the velocity condition. `"noslip"` is the clamped plate,
`psi = 0` with `dpsi/dn = 0`. `"freeslip"` is the simply supported plate,
`psi = 0` with `Laplacian psi = 0`. Both take the same anchoring, so a pair of
runs from one seed is an A/B on the wall alone: measured on a nephroid at
`d = 0.85`, free slip reaches a mean speed of 4.62 against 2.17 and a maximum
of 15.5 against 5.5, while both settle to the same two `+1/2` cores.

`full_stress=False` drops the elastic backflow and keeps the active term.
`elastic_h` sets where the elastic stress is suppressed, and `wall_h` how far
the wall condition reaches into the graded layer behind the boundary.

A field that runs away raises `RuntimeError` naming the step, the worst vertex
and whether it sits on the wall, rather than returning a saturated field that
reads like an arrested run.

`volterra-py/examples/confined_run.py` runs one geometry under both wall
conditions and draws the two side by side.

## Development and testing

The bindings are built with [maturin](https://www.maturin.rs/) and tested with
pytest. Using [uv](https://docs.astral.sh/uv/):

```bash
cd volterra-py
uv venv
uv pip install maturin pytest numpy
source .venv/bin/activate
maturin develop          # compile the extension into the venv
pytest tests/ -q         # run the smoke suite
```

The same recipe runs in CI (the `pytest` job in `.github/workflows/ci.yml`).

## License

[MIT](../LICENSE-MIT)
