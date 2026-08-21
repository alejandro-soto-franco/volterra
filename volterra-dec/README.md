# volterra-dec

Discrete exterior calculus nematohydrodynamics for volterra: simplicial meshes,
Hodge operators, confined domains with anchored boundaries, and the Stokes flow
an active nematic drives on them.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-dec` is the mesh backend. It bridges `cartan-dec` geometry to the
physics, and its reason to exist is the boundary: a mesh samples a curve
exactly, so the topological charge the anchoring imposes is exact and
controllable, where a lattice approximates the same curve with a staircase and
imposes whatever that staircase happens to give.

That matters for confined active nematics, where the interior defect charge is
fixed by the turning of the tangent along the wall. For an epitrochoid with `k`
cusps the crate reproduces the index law `1 + k/2` at `d = 1` and `1` for a
smooth boundary, to machine precision.

## Modules

| Module | Contents |
|--------|----------|
| `confined` | epitrochoid domains, graded mesh generation, imposed winding |
| `confined_ldg` | Landau-de Gennes problem on a confined mesh, anchoring, Beris-Edwards stress |
| `stokes_dec` | biharmonic stream-function Stokes, clamped and free-slip walls, pressure recovery, vorticity |
| `poisson` | conjugate-gradient Poisson with Dirichlet or closed boundaries, incomplete Cholesky |
| `qfield_dec` | the tensor order parameter on mesh vertices |
| `molecular_field_dec` | molecular field, Landau and Frank terms |
| `semi_lagrangian` | backward-trace transport, so the step is not bound by the smallest element |
| `domain`, `mesh_gen` | `DecDomain<M>`, well-centred meshes and precomputed operators |
| `helfrich`, `bending`, `variational` | membrane energetics and a BAOAB integrator |
| `runner_dec`, `snapshot` | run loops, checkpoints and the diagnostic series |

## Pressure

Steady Stokes is solved in stream-function form, which eliminates the pressure,
so no run ever forms one. `StokesSolverDec::pressure_from_stress` recovers it
from the same assembled stress by the Poisson problem `Delta p = div f` with
`dp/dn = f.n`, gauge-fixed to the area-weighted interior mean.

Vorticity comes from `vorticity_from_psi` as `Delta psi`, since the velocity is
itself a discrete curl of `psi` and differencing it again converges at
`O(h^0.4)` on a graded mesh.

## Example

```rust,no_run
use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};

// A nephroid with true cusps, meshed at a uniform element size.
let mesh = confined_mesh(
    Epitrochoid { q: 2.0, d: 1.0, r: 49.78 },
    MeshOpts { h_bulk: 1.0, h_min: 1.0, ..Default::default() },
);
let (imposed, worst_step, _) = mesh.imposed_charge(1.0);
assert!((imposed - 2.0).abs() < 1e-9);   // 1 + k/2 with k = 2
```

## License

[MIT](../LICENSE-MIT)
