# volterra-solver

Beris-Edwards nematohydrodynamics solver for active nematics on 3D Cartesian grids.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-solver` implements the equations of motion for active liquid crystals. The Q-tensor evolves under the Beris-Edwards equation with co-rotation, strain coupling, and Landau-de Gennes molecular field. An incompressible velocity field is obtained via an FFT-based Stokes pressure solve. Concentration dynamics follow Cahn-Hilliard with exponential time differencing (ETD).

Topological disclination lines (charge +1/2 and -1/2) are detected by computing the holonomy of the nematic frame field around mesh plaquettes, using `cartan-geo`.

## Modules

| Module | Contents |
|--------|----------|
| `molecular_field` | Landau-de Gennes free energy variation H = -dF/dQ |
| `beris_edwards_rhs` | Full time derivative of Q (dry and wet active models) |
| `stokes_solve` | FFT pressure projection for incompressibility |
| `ch_step_etd` | Cahn-Hilliard ETD integrator |
| `scan_defects` | Holonomy-based disclination detection |
| `EulerIntegrator` / `RK4Integrator` | Time-stepping schemes |

## Example

```rust,no_run
use volterra_solver::run_dry_active_nematic;
use volterra_core::ActiveNematicParams;

let params = ActiveNematicParams { ..Default::default() };
let snapshots = run_dry_active_nematic(&params, 10_000)?;
```

## License

[MIT](../LICENSE-MIT)
