# volterra-core

Trait definitions, error types, and shared parameters for the volterra active nematics framework.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-core` defines the foundational types shared across all volterra crates. It contains the `MarsParams` parameter struct (grid dimensions, physics coefficients, Landau-de Gennes constants, magnetic coupling, lipid parameters, and curvature settings), the `VError` error enum, and the `Integrator<S>` trait that all time-stepping schemes implement.

Every other crate in the workspace depends on `volterra-core`.

## Key types

| Type | Purpose |
|------|---------|
| `MarsParams` | All physical and numerical parameters for the MARS + lipid system |
| `VError` | Unified error enum: `DimensionMismatch`, `ConvergenceFailure`, `InvalidParams`, `Io` |
| `Integrator<S>` | Trait for time-integration schemes (Euler, RK4, ...) |

## Example

```rust,no_run
use volterra_core::{MarsParams, VError};

let params = MarsParams {
    nx: 128, ny: 128, nz: 1,
    dx: 1.0, dt: 0.005,
    // ... remaining fields
    ..Default::default()
};
```

## License

[MIT](../LICENSE-MIT)
