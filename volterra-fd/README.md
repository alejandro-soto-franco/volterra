# volterra-fd

Finite-difference nematohydrodynamics for volterra: Beris-Edwards evolution in
two and three dimensions on a uniform lattice, with the Stokes flow that couples
to it.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-fd` is the lattice backend. It evolves the tensor order parameter `Q`
under the Beris-Edwards equation with a Landau-de Gennes free energy, solves the
Stokes problem for the flow the active stress drives, and advects `Q` through
it. Where `volterra-dec` discretises on a simplicial mesh, this crate uses a
uniform grid, so it suits periodic and rectangular domains and it is the cheaper
of the two to sweep over parameters.

Spectral routines go through `rustfft`, and the per-step work is parallelised
with `rayon`.

## Modules

| Module | Contents |
|--------|----------|
| `step` | one Beris-Edwards step, 2D and 3D |
| `stokes` | Stokes solve for the active and elastic stress |
| `molecular_field` | the molecular field `H`, Landau and Frank terms |
| `defects` | defect detection and charge from the director winding |
| `runner` | run loops, snapshots and the diagnostic series |
| `bin/fd` | the command line driver |

## Choosing between the backends

Use this crate for a periodic or rectangular domain, for parameter sweeps, and
wherever a uniform lattice represents the boundary adequately. Use
`volterra-dec` where the boundary sets the physics, since a mesh samples a
curve exactly and a lattice approximates it with a staircase.

## License

[MIT](../LICENSE-MIT)
