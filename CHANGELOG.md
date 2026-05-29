# Changelog

All notable changes to volterra are documented here.

---

## [Unreleased]

### Added

- **volterra-braid**: new crate for braid-group analysis of defect trajectories,
  decoupled from the PDE solver. `detect_defects` (Q grid -> defects), `track`
  (greedy nearest-neighbour), `extract_braidword` (Artin generators from defect
  worldlines), `topological_entropy` (Burau spectral radius at `t = -1`; exact
  for the golden `2 log phi` and silver `log(3 + 2 sqrt 2)` braids), and
  `realize_braid` / `golden_orbit` / `silver_orbit` synthetic orbits.
- **volterra-py**: `BraidWord` class plus `braid_detect_defects`,
  `braid_word_from_frames`, `braid_topological_entropy` functions; `.pyi` stubs.
- **volterra-braid/oracle**: `braid_tracker_v2.py` (cleaned reimplementation of
  the CGPO reference tracker), `cross_check.py` (Rust-vs-Python differential
  validation), and `compare_cgpo.py` (comparison against the unmodified published
  `braid_tracker.py`, run via uv). volterra and the published script extract the
  identical braid word and entropy for the golden and silver configurations.
- **BENCHMARKS.md**: braid-analysis throughput section (golden and silver
  configs) -- volterra's defect detection is ~150-180x faster (native) / ~14x
  (via PyO3) than the published CGPO Python scheme, independent of defect count.
  Native bench: `examples/bench_braid.rs`; vs-Python: `oracle/bench_braid.py`.

---

## [0.3.0] - 2026-04-11

### Breaking

- Renamed `MarsParams` to `ActiveNematicParams`, `MarsParams3D` to `ActiveNematicParams3D`.
- **Fixed elastic sign** in `molecular_field_conn` and `molecular_field_dec`: `+K*lap` -> `-K*lap`. The DEC Laplacian is positive-semidefinite; elastic smoothing requires the minus sign.
- **Fixed active force** in `compute_vorticity_source`: flat (x,y) projection -> covariant 3D tensor divergence with per-vertex tangent frames.

### Added

- **volterra-dec**: `EvolvingDomain<M>` with mesh deformation and automatic operator rebuild. Discrete Levi-Civita connection and `CovLaplacian` rebuilt on each `deform()`.
- **volterra-dec**: curvature computation, shape equation (Helfrich + tension + active stress), `vn_correction()` for Q-tensor material derivative on evolving surfaces.
- **volterra-dec**: `active_stress_normal()` (`-zeta * Q:b`), `advect_q_covariant()` with parallel transport, `write_velocity_snapshot()`.
- **volterra-solver**: Zhu-parameterised S^2 simulation (`sim_sphere_zhu --pe N`).
- **volterra-solver**: coupled shape + nematic examples (`sim_deforming_sphere`, `sim_active_deforming`).
- **volterra-solver**: 3D Beris-Edwards via fiber bundle (`sim_3d_fiber_bundle`).
- **volterra-mars**: MARS-specific parameter presets and dimensionless groups.
- **tools/viz**: dark-green-to-white S colourmap, blue-green-red vorticity panel, barycentric streamline interpolation with Catmull-Rom smoothing.

### Changed

- Depends on cartan 0.4 (fiber bundle traits).

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
