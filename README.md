# volterra

Covariant active matter simulation on Riemannian manifolds, in Rust.

[![CI](https://github.com/alejandro-soto-franco/volterra/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/volterra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![PyPI](https://img.shields.io/pypi/v/volterra-py.svg)](https://pypi.org/project/volterra-py/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

> **Python users:** the PyPI distribution is `volterra-py` (`pip install volterra-py`), and the Python module is imported as `import volterra`. The bare name `volterra` belongs to an unrelated package on PyPI. Releases through 0.5.0 were published as `volterra-nematic`.

**volterra** solves the Beris-Edwards nematohydrodynamics equations for active liquid crystals on curved spaces. It is built on [cartan](https://github.com/alejandro-soto-franco/cartan), a Riemannian geometry library, and uses discrete exterior calculus (DEC) to discretise covariant differential operators on arbitrary manifolds.

The primary objects of study are the topological defects of the nematic order parameter field and their long-term dynamical behaviour in confined and curved geometries.

## Physics

Active nematics are materials composed of self-driven elongated units (collections of cytoskeletal filaments driven by molecular motors, for instance) that spontaneously develop orientational order and generate internal stresses. The nematic order parameter Q is a symmetric traceless rank-2 tensor field. Its topological defects (points in 2D, lines in 3D) carry a topological charge of +/-1/2 and are the primary objects of physical and mathematical interest.

The equations of motion couple Q-tensor dynamics to an incompressible flow field:

```text
(d/dt + u . nabla) Q - S(W, Q) = Gamma * H
rho (d/dt + u . nabla) u = -nabla p + eta Delta u + nabla . sigma
nabla . u = 0
```

where `H = -delta F / delta Q` is the molecular field from the Landau-de Gennes free energy, `S(W, Q)` encodes co-rotation and strain coupling, and the active stress `sigma_active = -zeta * Q` drives spontaneous flow.

All derivatives are covariant with respect to the Levi-Civita connection of the background Riemannian manifold.

## Design

volterra is designed around three principles:

**Correctness of the geometry.** The Laplacian acting on Q is the Lichnerowicz operator (the Bochner connection Laplacian plus a Riemann curvature correction). The background manifold geometry enters the equations of motion exactly, not as a perturbation. This makes volterra suitable for studying systems where curvature is itself the variable of interest.

**Cache-friendly DEC.** Differential operators are built from DEC on well-centred Delaunay meshes. Well-centred meshes give diagonal Hodge stars, so the full Laplace-Beltrami operator is two sparse {0, +1, -1} matrix-vector products interleaved with diagonal scalings. Simplices are Hilbert-curve reordered for spatial locality. Field arrays use structure-of-arrays layout for SIMD vectorisation.

**Ludwig-quality numerics.** Time integration uses at minimum fourth-order Runge-Kutta. The pressure Poisson solve enforces the incompressibility constraint exactly at each step.

## Crate structure

The workspace has six members:

```
volterra          facade crate (start here)
volterra-core     trait definitions and error types
volterra-fd       finite-difference solver on confined Cartesian domains
volterra-dec      DEC physics on triangle meshes: Q-tensor fields, Stokes,
                  semi-Lagrangian advection, Helfrich energy, variational integrators
volterra-braid    defect detection and tracking, braid words, topological entropy
volterra-py       PyO3 bindings  (pip install volterra-py)
```

The CUDA paths live in two further crates. Each is its own workspace, because the
cuda-oxide codegen backend needs a pinned nightly toolchain that must not reach
the stable workspace above, and neither is published:

```
volterra-cuda     double-precision CUDA FIRE minimiser for the passive
                  Landau-de Gennes relaxation
volterra-fd-cuda  double-precision CUDA path for the finite-difference solver,
                  checked kernel by kernel against the CPU crate
```

## Substrate

volterra depends on [cartan](https://github.com/alejandro-soto-franco/cartan) for Riemannian geometry. Any type implementing `cartan_core::Manifold` can serve as the simulation domain. This includes the manifolds shipped with cartan (`Sphere<N>`, `SpecialOrthogonal<N>`, `SpecialEuclidean<N>`, flat `Euclidean<N>`) and any user-defined manifold.

## Status

The Cartesian-grid solver, DEC manifold solver, and the `ActiveNematicEngine` (operator-split active nematohydrodynamics on Riemannian 2-manifolds) are all implemented. Active turbulence with defect dynamics has been demonstrated on flat periodic domains (128x128, full Stokes coupling) and on S^2 (icosphere, 10242 vertices). The engine supports both intrinsic (complex line bundle) and extrinsic (Killing operator) discretisation paths, with switchable Stokes backends and implicit diffusion (no diffusive CFL).

| Crate | Status |
|-------|--------|
| volterra-core | `ActiveNematicParams`, `NematicParams` (dimensionless Pe/Er/La/Lc), `VError`, `Integrator` |
| volterra-fields | `QField2D`, `QField3D`, `VelocityField2D/3D`, `ScalarField2D/3D` |
| volterra-solver | Beris-Edwards (Cartesian + DEC), Stokes (FFT + DEC stream-function), Cahn-Hilliard (ETD), disclination detection, `NematicEngine`, `ActiveNematicEngine` (operator splitting), `NematicField2D` (complex Section\<2\>), `StokesSolver` trait (stream-function + Killing backends) |
| volterra-dec | `DecDomain<M>`, `QFieldDec`, `ConnectionLaplacian` (spin-2), `CurvedStokesSolver` (biharmonic), `SemiLagrangian` (BVH + RK4 + deformation gradient pullback), epitrochoid/icosphere/torus mesh, `.npy` snapshot export |
| volterra-mars | MARS experimental system presets and dimensionless groups |
| volterra-py | PyO3 bindings for all 2D/3D types and runners |

## Performance

volterra's fused parallel Rust solver (rayon) outperforms both open-Qmin (C++/CUDA) and Ludwig (C/MPI) on passive Landau-de Gennes relaxation benchmarks. See [BENCHMARKS.md](BENCHMARKS.md) for full data.

| N | Sites | volterra (CPU) | open-Qmin GPU | Speedup |
|---|-------|---------------|---------------|---------|
| 50 | 125K | 0.005 us/site/step | 0.027 | 5.5x |
| 100 | 1M | 0.009 | 0.042 | 4.6x |
| 200 | 8M | 0.009 | 0.048 | 5.3x |

## Visualisation

Automated simulation-to-video pipeline in `tools/viz/`:
- 2D: matplotlib (director field, S colourmap, defect markers)
- 3D: PyVista (isosurfaces, disclination tubes, volume rendering)
- Surfaces: PyVista (director streamlines, defect charge density)
- Video encoding: ffmpeg (h264)

## License

[MIT](LICENSE-MIT)
