# volterra-dec

Discrete exterior calculus layer for volterra: simplicial meshes, Hodge operators, and covariant differential operators on Riemannian manifolds.

Part of the [volterra](https://github.com/alejandro-soto-franco/volterra) workspace.

## Overview

`volterra-dec` bridges `cartan-dec` geometry to the physics solver. It provides `DecDomain<M>`, which bundles a well-centered Delaunay mesh with precomputed DEC operators (exterior derivatives, Hodge stars), dual cell areas, and per-vertex curvature arrays.

The crate also implements Helfrich membrane energetics (bending rigidity, Gaussian curvature modulus, spontaneous curvature) and a BAOAB variational integrator for membrane dynamics. The BAOAB splitting uses manifold-preserving exponential-map position updates and supports adaptive time stepping.

## Modules

| Module | Contents |
|--------|----------|
| `domain` | `DecDomain<M>`, mesh + precomputed DEC operators |
| `helfrich` | Helfrich bending energy and surface forces |
| `variational` | BAOAB splitting integrator with adaptive stepping |

## Example

```rust,no_run
use volterra_dec::DecDomain;

// Build a DEC domain from a cartan-dec simplicial complex
let domain = DecDomain::from_complex(complex, params)?;
let laplacian = domain.laplacian_0();
```

## License

[MIT](../LICENSE-MIT)
