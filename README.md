# volterra

Covariant active nematics simulation on Riemannian manifolds, in Rust.

[![CI](https://github.com/alejandro-soto-franco/volterra/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/volterra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![PyPI](https://img.shields.io/pypi/v/volterra-nematic.svg)](https://pypi.org/project/volterra-nematic/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

> **Python users:** the PyPI distribution is `volterra-nematic` (`pip install volterra-nematic`), but the Python module is imported as `import volterra`. The name `volterra` was already taken on PyPI.

**volterra** solves the Beris-Edwards nematohydrodynamics equations for active liquid crystals on curved spaces. It is built on [cartan](https://github.com/alejandro-soto-franco/cartan), a Riemannian geometry library, and uses discrete exterior calculus (DEC) to discretize covariant differential operators on arbitrary manifolds.

The primary objects of study are the topological defects of the nematic order parameter field and their long-term dynamical behavior in confined and curved geometries.

## Physics

Active nematics are materials composed of self-driven elongated units (collections of cytoskeletal filaments driven by molecular motors, for instance) that spontaneously develop orientational order and generate internal stresses. The nematic order parameter Q is a symmetric traceless rank-2 tensor field. Its topological defects (points in 2D, lines in 3D) carry a topological charge of +/-1/2 and are the primary objects of physical and mathematical interest.

The equations of motion couple Q-tensor dynamics to an incompressible flow field:

```
(d/dt + u . nabla) Q - S(W, Q) = Gamma * H
rho (d/dt + u . nabla) u = -nabla p + eta Delta u + nabla . sigma
nabla . u = 0
```

where `H = -delta F / delta Q` is the molecular field from the Landau-de Gennes free energy, `S(W, Q)` encodes co-rotation and strain coupling, and the active stress `sigma_active = -zeta * Q` drives spontaneous flow.

All derivatives are covariant with respect to the Levi-Civita connection of the background Riemannian manifold.

## Design

volterra is designed around three principles:

**Correctness of the geometry.** The Laplacian acting on Q is the Lichnerowicz operator (the Bochner connection Laplacian plus a Riemann curvature correction). The background manifold geometry enters the equations of motion exactly, not as a perturbation. This makes volterra suitable for studying systems where curvature is itself the variable of interest.

**Cache-friendly DEC.** Differential operators are built from DEC on well-centered Delaunay meshes. Well-centered meshes give diagonal Hodge stars, so the full Laplace-Beltrami operator is two sparse {0, +1, -1} matrix-vector products interleaved with diagonal scalings. Simplices are Hilbert-curve reordered for spatial locality. Field arrays use structure-of-arrays layout for SIMD vectorization.

**Ludwig-quality numerics.** Time integration uses at minimum fourth-order Runge-Kutta. The pressure Poisson solve enforces the incompressibility constraint exactly at each step.

## Crate structure

```
volterra          facade crate (start here)
volterra-core     trait definitions and error types
volterra-fields   Q-tensor, velocity, pressure, and stress field types
volterra-solver   equations of motion, time integration, defect tracking
volterra-dec      DEC mesh, Hodge operators, covariant differential operators (in progress)
volterra-mars     MARS experimental system presets and dimensionless groups
volterra-py       PyO3 bindings  (pip install volterra-nematic)
```

## Substrate

volterra depends on [cartan](https://github.com/alejandro-soto-franco/cartan) for Riemannian geometry. Any type implementing `cartan_core::Manifold` can serve as the simulation domain. This includes the manifolds shipped with cartan (`Sphere<N>`, `SpecialOrthogonal<N>`, `SpecialEuclidean<N>`, flat `Euclidean<N>`) and any user-defined manifold.

## Status

The 3D Cartesian-grid simulation stack is implemented and tested. The DEC layer (curved-manifold discretization) is the next major milestone.

| Crate | Status |
|-------|--------|
| volterra-core | implemented: `MarsParams3D`, `VError`, `Integrator` |
| volterra-fields | implemented: `QField3D`, `VelocityField3D`, `ScalarField3D` |
| volterra-solver | implemented: Beris-Edwards, Stokes (FFT), Cahn-Hilliard (ETD), disclination detection |
| volterra-py | implemented: PyO3 bindings for all 3D types and runners |
| volterra-dec | implemented: `DecDomain<M>`, Helfrich energy, BAOAB variational integrator |

The Cartesian-grid solver uses a periodic grid with 7-point finite differences. The DEC layer (`volterra-dec`) generalizes this to arbitrary well-centered Riemannian meshes via [cartan](https://github.com/alejandro-soto-franco/cartan) discrete exterior calculus.

## License

[MIT](LICENSE-MIT)
