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
