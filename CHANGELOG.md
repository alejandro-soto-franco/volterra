# Changelog

All notable changes to volterra are documented here.

---

## [Unreleased]

### Breaking

- Renamed `MarsParams` to `ActiveNematicParams`, `MarsParams3D` to `ActiveNematicParams3D`.
- Renamed runner functions: `run_mars_component1` to `run_dry_active_nematic`,
  `run_mars_component1_hydro` to `run_active_nematic_hydro`,
  `run_mars_bech` to `run_bech`, `run_mars_3d` to `run_dry_active_nematic_3d`,
  `run_mars_3d_full` to `run_bech_3d`.
- Python API: all class and function names updated to match.

### Added

- New `volterra-mars` crate with MARS-specific parameter presets (`MarsPreset`)
  and dimensionless group helpers (`MarsLnpDimensionless`).

---

## [0.2.0] - 2026-04-08

### Added

- **volterra-dec**: replaced the 49-line stub with three production modules.
  - `domain::DecDomain<M>`: bundles a triangle mesh with precomputed DEC operators (exterior derivative, Hodge star, Laplacian), dual cell areas, edge lengths, and curvature arrays. Constructor assembles operators via `cartan_dec::Operators::from_mesh_generic`.
  - `helfrich`: discrete Helfrich bending energy `E = sum_v A_v [kb/2 (H - H0)^2 + kg K]` with per-vertex spontaneous curvature. Force computation placeholder (analytical shape operator gradient is a follow-up).
  - `variational`: BAOAB splitting scheme for membrane dynamics. `baoab_ba_step` performs deterministic B-A-A-B steps with manifold-preserving `exp` map position updates. The O step (Shardlow edge sweep from pathwise-geo) is inserted by the caller between the two A half-steps. `kinetic_energy` via manifold inner product. `compute_dt` with diffusive + force CFL bounds.

### Changed

- volterra workspace now patches cartan crates to local paths for the new Phase A/B APIs (`cartan-dec` 0.2.0, `cartan-remesh` 0.2.0).

---

## [0.1.0] - 2026-03-25

### Added

- **volterra-core**: `ActiveNematicParams3D` (grid, physics, Landau, magnetic, lipid, curvature parameters), `VError` (4 variants), `Integrator<S>` trait.
- **volterra-fields**: `QField2D`, `QField3D` (traceless symmetric Q-tensor fields), `VelocityField2D/3D`, `ScalarField2D/3D`, `PressureField3D`, `ConcentrationField3D`. Structure-of-arrays layout.
- **volterra-solver**: 3D Cartesian-grid Beris-Edwards solver with molecular field, Stokes FFT pressure solve, Cahn-Hilliard ETD, disclination detection via cartan holonomy.
- **volterra-py**: PyO3 bindings (PyPI: `volterra-nematic`, import as `volterra`).
- **volterra-dec**: crate scaffold (doc comments only, no implementation).
