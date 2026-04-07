# volterra-fields

Tensor field types for active nematics simulations: Q-tensor, velocity, pressure, concentration, and scalar fields.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-fields` defines the data structures that hold simulation state on regular Cartesian grids and DEC meshes. All fields use structure-of-arrays (SoA) layout for cache locality and SIMD vectorization.

2D types (`QField2D`, `VelocityField2D`, `ScalarField2D`) store data on periodic `(nx, ny)` grids with row-major indexing `k = i * ny + j`. 3D types (`QField3D`, `VelocityField3D`, `ScalarField3D`, `PressureField3D`, `ConcentrationField3D`) extend this to `(nx, ny, nz)` grids. The Q-tensor is symmetric and traceless: two independent components in 2D, five in 3D.

## Key types

| Type | Components |
|------|------------|
| `QField2D` / `QField3D` | Symmetric traceless Q-tensor (2 / 5 independent entries) |
| `VelocityField2D` / `VelocityField3D` | Velocity vector at each vertex |
| `ScalarField2D` / `ScalarField3D` | Generic scalar field |
| `PressureField3D` | Pressure field |
| `ConcentrationField3D` | Concentration field (Cahn-Hilliard) |

## Example

```rust,no_run
use volterra_fields::QField3D;

let nx = 64;
let ny = 64;
let nz = 64;
let q = QField3D::random(nx, ny, nz, 0.3);
let s_max = q.max_scalar_order();
```

## License

[MIT](../LICENSE-MIT)
