# Changelog

All notable changes to volterra are documented here.

---

## [Unreleased]

### Changed

- **The PyPI distribution is `volterra-py`**, which is the crate's own name.
  It was `volterra-nematic` through 0.5.0. The module is still imported as
  `import volterra`, so no code changes, and the bare name `volterra` belongs
  to an unrelated package on PyPI. The old distribution keeps its releases up
  to 0.5.0 and takes no new ones.

  The engine is not restricted to nematics, and a distribution named for one
  phase says otherwise. The `k`-atic order parameters are the intended scope.

  A distribution rename makes a new PyPI project, and trusted publishing is
  registered per project. The pending publisher for `volterra-py` has to be
  created by hand on pypi.org, against this repository and `release.yml`,
  before the first tag under the new name.

## [0.5.0] - 2026-09-05

### Added

- **A periodic domain.** `volterra_fd::boundary::periodic_boundary` returns a
  flat torus: every cell interior, neither boundary ring populated, so all four
  boundary-condition passes in `update_step_inner` visit nothing and the domain
  closes through the modular neighbour indexing every stencil in `ops` already
  used. Nothing else in the solver changes. This is the domain of both papers
  below.

- **Enhanced nematic locking** (`volterra_fd::locking`), after Mitchell,
  Sabbir, Klein and Beller, "Modelling active nematics via the nematic locking
  principle", Soft Matter (2025), arXiv:2506.20996. In 2D the molecular field is
  spanned by `Q` and `U = JQ`; the `Q` part changes only `S` and the `U` part
  rotates the director at the fracturing rate `omega_F = Tr(HU)/(gamma S^2)`,
  which is the only term that breaks locking. `add_fracture_switch` multiplies
  that term's mobility by `exp(-S^2/(2 sigma^2))`, so fracturing turns on only
  where `S` has fallen. `H` itself is untouched, so the Navier-Stokes side sees
  the same molecular field and the same stress, and the whole modification is
  one term added after `get_q_update`.

  Switched on through `Params::locking`, which every constructor leaves as
  `None`; the field is `#[serde(default)]`, so a config written before it
  existed still reads back as the standard model, and the golden concurrence
  tests are unaffected. `rotation_rates` returns `omega_A` and `omega_F` per
  cell for the reference's own diagnostic.

  The reference's `sigma = 0.2` is quoted in its own `C = -2A`, `S_eq = 1`
  convention, while this crate and `flow-solver.py` use `C = -A`, `S_eq = sqrt 2`.
  `Locking::sigma` is therefore given in units of the equilibrium `S`.

- **`Params::from_dimensionless` and `Dimensionless`**, the five groups Mitchell
  et al. state their runs in: `Re = K/(rho nu^2)`, `gamma_tilde = gamma nu / K`,
  `C_tilde = C/zeta = (ell_a/ell_n)^2`, the flow-alignment parameter, and the
  confinement ratio, plus `s_eq` because the two papers normalise the order
  parameter differently. At `K = 256^2` and `ell_a = 3` this returns
  arXiv:2506.20996's own stated constants digit for digit
  (`gamma = 5*256`, `C = 256^2`, `eta = 2560`, `zeta = (256/3)^2`), which
  `tests/test_locking.rs` asserts. `Params` also gains `active_length`,
  `coherence_length` and `active_time`.

- **Topological entropy by material-line stretching** (`volterra_fd::stretching`),
  the measurement of Mitchell, Sabbir, Geumhan, Smith, Klein and Beller,
  "Maximally mixing active nematics", Phys. Rev. E 109, 014606 (2024). A
  `MaterialLine` is advected with frozen-field RK4 and bilinear velocity
  sampling, refined by midpoint insertion so no segment exceeds a tolerance, and
  measured with minimum-image segment lengths on the torus. `fit` returns the
  slope of `log(length)` against time, which is `h`; the paper's dimensionless
  form is `h t_a`. Refinement stops at a point cap and the curve then freezes,
  which is where a fit must end. The point count grows like `exp(h t)`, which
  is why the reference moves to an ensemble algorithm for its own sweep.

  Validated against a uniform-strain field, where a segment stretches as
  `exp(a t)` exactly and the fit recovers `a` to `1e-3`.

- **`examples/periodic_active_nematic.rs`**, one runner for both papers, and
  `examples/analyse_periodic.py`, which reports the period, the steady defect
  count, the entropy and the rotation-rate statistics from a finished run.

- **`examples/panels_periodic_video.py`**, a nine-panel film of a periodic run:
  the director field, the `Q` isocontours and the velocity on the first row; the
  vorticity, the accumulated RMS vorticity and a passive-tracer lattice on the
  second; the advective and fracturing rotation rates and the defect-count trace
  on the third. The two rates share one colour scale, taken from the advective
  one, so enhanced locking draws the fracturing panel blank rather than
  rescaling its own noise. The layout is fixed, so the standard and the enhanced
  film at one `ell_a` can be put side by side.

  The runner writes what the film needs at `VP_FRAME_EVERY`: `p`, the two
  rotation-rate fields, the material-line vertices, and a lattice of passive
  tracers set by `VP_TRACERS`. A tracer never refines and never saturates, so it
  is four velocity samples a step for the life of a run, which is what lets
  the mixing panel stay live where a material line has long since stopped being
  resolved.

- **Braids on a flat torus** (`volterra_braid::torus`), after Mitchell et al.
  (2024). Worldlines are lifted to the universal cover, so a bounded orbit is a
  closed loop and a winding defect draws an open path; encounters are found over the
  periodic image lattice, which is where the four an orbit of the maximal mixing
  braid come from; and `is_maximal_mixing` applies the paper's own criterion of
  two `+1/2` defects, bounded unwound orbits, and four same-sense encounters a
  period. `h_tepo_maximal_mixing` is `log(phi + sqrt phi)` and
  `braid_prediction` is `log(phi + sqrt phi) / (T_tilde / 4)`.
  `ideal_figure_2a` reconstructs the published cartoon from its geometry, and
  the reader returns 4.00 same-sense encounters an orbit on it.

- **`Params::stress`** (`StressModel::{Full, Giomi}`), the two published elastic
  stresses. `Full` is the Beris-Edwards form of Klein et al. Eq. (11), which is
  `flow-solver.py`'s force; `Giomi` is `-lambda H + [Q, H] - alpha Q`, the form
  Mitchell et al. state and take from Giomi, Phys. Rev. X 5, 031003 (2015),
  where the Ericksen stress is dropped as higher order. `#[serde(default)]` and
  `Full` everywhere, so a run made before the field existed is reproduced bit
  for bit. Which one a run uses changes whether defects survive at
  `ell_a = 3`.

- **`VP_Q_INIT`**, continuation from a saved `q_*.npy` frame, the protocol
  Mitchell et al. build their Fig. 5 with. The reader accepts the header this
  crate writes and refuses anything else, rather than reinterpreting an `f32` or
  a Fortran-ordered array as the state.

- **`examples/braid_report`**, which reads a run's braid over the longest window
  in its developed half whose defect census never changes, and writes
  `braid.json`. The window must come from the developed state: the longest one
  over a whole run lands in the quench, where the census sits still and the
  defects wind rather than orbit.

- **`examples/plot_braid.py`, `plot_fig5.py` and `plot_paper_figures.py`**,
  which reproduce Figs. 2, 5, 3 and 4 of Mitchell et al.

- **`volterra_dec::curve::PlaneCurve`**, the wall a confined run is meshed
  against. A closed plane curve answers four questions and the mesher needs no
  others: where it is, how fast the parametrisation runs along it, how sharply
  it turns, and which side is the interior. `Epitrochoid` is the analytic
  family; `PolyCurve` splines a closed table of points with a periodic cubic,
  so a wall measured from an image or written as a parametrisation meshes on
  the same path. `confined_mesh` is generic over the trait rather than over the
  one shape it used to take.

- **The confined domain and its run in the Python bindings.** `PlaneCurve`
  (`epitrochoid`, `from_points`, `from_callable`), `confined_mesh` returning a
  `ConfinedMesh`, and `ConfinedRun`, which steps the same scheme the Rust driver
  runs and hands back the fields, the defects and a census. A collaborator
  reproduces a confined run without building the Rust.

- **The velocity wall as a toggle.** `ConfinedRun(wall=...)` takes `"noslip"`,
  the clamped plate with `psi = 0` and `dpsi/dn = 0`, or `"freeslip"`, the
  simply supported one with `psi = 0` and `Laplacian psi = 0`. Both take the
  same anchoring and the same seed, so a pair of runs separates the wall from
  everything else.

- **`ConfinedMesh::imposed_charge`**, the total charge the anchoring puts in the
  interior, measured on that mesh's own boundary, with the worst doubled-angle
  step beside it. A step past a quarter turn means the sampling booked the wrong
  branch, which a corner does at any sampling density. The number a mesh reports
  is the boundary condition a run has, and it is what to read rather than the
  regularisation parameter.

- **`braid_detect_defects_winding`**, detection by the director's holonomy, with
  no threshold to choose. `braid_detect_defects` thresholds the saddle-splay
  density, whose scale follows the field's gradients, and an angle-sized
  threshold silently returns nothing on a settled field. Its docstring now says
  what it bounds.

- **A typed wheel.** The bindings ship as a mixed maturin layout, so
  `__init__.pyi` and `py.typed` install beside the extension and a caller's type
  checker sees the whole API.

- **`volterra-py/examples/nephroid_braid.py`**, the shortest run that means
  something: the nephroid meshed, stepped to its periodic orbit, and read back
  as the silver braid at entropy `log(3 + 2 sqrt 2)`.

### Changed

- **The confined mesher samples the wall in arc length.** The step along the
  boundary was a parameter-space cap applied to an arc-length target, which
  compressed the spacing wherever the parametrisation ran fast and left every
  sub-15-degree triangle with exactly two wall vertices. The interior fill and
  the layer stride now follow the wall's own spacing, and the boundary step is
  graded against the previous one. On the production family the minimum angle
  goes from 5 to 10 degrees to 22 to 34, with the imposed charge exactly 1.000
  in all twelve configurations.

- **`NematicParams::klein` is `NematicParams::from_length_scales`**, and
  `Parameterisation::Klein { als, ncl, lx }` is
  `Parameterisation::LengthScales { active_length, coherence_length, resolution }`.
  A constructor named after a person says nothing about what it takes. The
  module `klein` is `constants`. Breaking for any caller of those three names.

- **`entropy.json` and `line_lengths.csv` are written at every observation**
  rather than when a run ends. A material-line entropy is a fit to a history
  that is complete the moment the line saturates, so a run stopped part way is
  as informative about mixing as one that reaches its last step. `entropy.json`
  now also records whether the line saturated.


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
  varying* field loaded from file.


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
