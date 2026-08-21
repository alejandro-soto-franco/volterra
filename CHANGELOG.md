# Changelog

All notable changes to volterra are documented here.

---

## [Unreleased]

## [0.4.0] - 2026-08-22

### Added

- **Pressure and vorticity recovered from a stream-function Stokes solve.** The
  biharmonic solve is for `psi` alone, so the pressure is eliminated and no run
  has ever formed one. `StokesSolverDec::pressure_from_stress` recovers it from
  the same assembled stress by `Delta p = div f` with `dp/dn = f.n`, which the
  weak form imposes for itself, gauge-fixed to the area-weighted interior mean
  so it agrees with the gauge `volterra-fd` pins its pressure in. An unweighted
  vertex mean sits off the interior mean by a few per cent of a standard
  deviation, because the boundary is sampled more densely than the bulk.
  `pressure_rhs_from_force` and `vertex_force_from_stress` are the pieces.

- **`vorticity_from_psi`**, which returns `Delta psi`. Differencing the
  recovered `u` instead chains two vertex-gradient operators and converges at
  `O(h^0.4)` on a graded mesh, the same failure the co-rotational term has.
  Against an independent FEM curl of the velocity the two agree at a
  correlation of 0.9925 and a slope of 1.012.

- **`examples/replay_fields`**, which recovers `u`, `psi`, `p` and `omega` for a
  finished run from its saved `Q` frames, through the solver's own Stokes path.
  `Q` is the state and the flow is an instantaneous functional of it, so a saved
  frame determines the velocity exactly. It rebuilds the run's mesh and refuses
  to proceed unless the vertices reproduce, and it checks its own output against
  the run's recorded `speed_max`, so a run made before a change to the solver
  is reported as stale rather than replayed into fields it never had. On the production runs it reproduces the
  solver to 2.1e-8 relative, which is f32 storage precision.

- **`quintefoiloid`**, five cusps, `q = 7/2`. The geometry already derived
  `k = 2(q-1)` generically, so only the shape name needed adding.

- **Integration tests for the index law and the field recovery.**
  `tests/index_law.rs` asserts that a cusped domain imposes `1 + k/2` and a
  smooth one imposes `1`, for every member of the family, which is the property
  every confined run depends on before any physics runs.
  `tests/field_recovery.rs` checks the pressure against a force whose potential
  is known and the vorticity against a manufactured `psi`, both of which fail at
  a relative error of 2 if a sign is flipped.

- **README for `volterra-fd`**, which had none, and a rewritten one for
  `volterra-dec`, which still described three modules where the crate now has
  thirty.

- **Electric-field coupling on the 3D molecular field**, alongside the magnetic
  coupling that was already there. `ActiveNematicParams3D` gains `epsilon_a`,
  `e0` and `omega_e`, mirroring `chi_a`, `b0` and `omega_b`: both fields couple
  quadratically to a direction and contribute a traceless rank-two term, so both
  now go through one `field_term`, which replaced the magnetic expression
  duplicated at four call sites. A negative `epsilon_a` is allowed, since a
  nematic of negative dielectric anisotropy aligns across the field rather than
  along it. Exposed through the Python bindings as optional arguments.

  A zero amplitude removes the term identically, so every result measured before
  this existed is unchanged to the last bit, which a test asserts.

  What remains of open-Qmin's version of this capability is a *spatially
  varying* field loaded from file; `docs/SUBSUMPTION.md` now records the row as
  partial rather than a gap.


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

### Removed

- **`volterra-mars`**, which was a shim. Every one of its dimensionless groups
  delegated a single line to a method already on `ActiveNematicParams`
  (`pi_number`, `defect_length`, `a_eff`, `ch_coherence_length`, `phi_eq`), and
  its two presets were `default_test()` with a grid size and an activity
  changed. Nothing in the workspace depended on it. It had no field coupling
  of its own: the magnetic actuation the MARS system is built on lives in
  `ActiveNematicParams3D` and the molecular field, and is untouched by this.

  `volterra-mars` 0.3.2 stays on crates.io; a published name cannot be
  withdrawn.

### Changed

- **`volterra-solver` is dissolved.** It was not a crate with a subject: 6,982
  lines that split three ways with nothing left over. The finite-difference
  physics, in two dimensions and three, went to `volterra-fd`; the engine layer
  and the runners written against DEC meshes went to `volterra-dec`; and its
  tests, examples and benches followed the code they exercise. Its dependencies
  were redistributed with it.

  Anything that imported `volterra_solver::X` now imports it from
  `volterra_fd::X` or `volterra_dec::X`, depending on which discretisation X
  belongs to. `volterra-solver` 0.3.2 stays on crates.io.

- **`volterra-cgpo` is renamed `volterra-fd`**, and `volterra-cgpo-cuda`
  becomes `volterra-fd-cuda`. CGPO is the acronym of one paper, and having
  it in a crate name, a CLI subcommand, fifteen environment variables and six
  type names made a general solver read as that paper's code. The new name is
  the method, alongside `volterra-dec`, which is the discrete-exterior-calculus
  discretisation of the same physics. Neither has a dimension: what a
  discretisation is does not depend on how many dimensions it is applied in,
  and both are meant to grow into however many the engine supports. What
  `volterra-fd` implements today is two-dimensional.

  The rename reaches `volterra run cgpo` (now `volterra run fd`), the
  `cgpo_fd` binary (now `fd`), the `CGPO_*` environment variables (now
  `FD_*`), the `Cgpo*` types (now `Fd*`), and the default output directory
  `./output/cgpo` (now `./output/fd`).

  References to Klein et al.'s own released code keep the name, in
  `volterra-braid/oracle` and in the benchmark tables that compare against it,
  because that is what that code is called.

  `volterra-cgpo` 0.3.2 stays on crates.io: a published name cannot be reused or
  withdrawn. Releases continue under `volterra-fd`.


### Fixed

- **CI ran a test in a crate that no longer exists.** The perf floor step still
  named `volterra-solver`, dissolved above, so the job could not pass. It now
  runs against `volterra-fd`.

- **Three lints that `-D warnings` turned into a red build.** Two are NaN-safe
  `!(x > 0.0)` checks where clippy's `partial_cmp` suggestion would drop the NaN
  branch, so the intent is stated and the lint allowed rather than the logic
  changed. The third factored a boxed closure into a `Preconditioner` alias.

- **The facade listed its own crates wrongly**, naming `volterra-dec` twice and
  omitting `volterra-braid`.
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
- **volterra-solver**: 3D Beris-Edwards via fibre bundle (`sim_3d_fiber_bundle`).
- **volterra-mars**: MARS-specific parameter presets and dimensionless groups.
- **tools/viz**: dark-green-to-white S colourmap, blue-green-red vorticity panel, barycentric streamline interpolation with Catmull-Rom smoothing.

### Changed

- Depends on cartan 0.4 (fibre bundle traits).

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
