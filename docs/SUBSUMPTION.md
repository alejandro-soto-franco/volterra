# Subsumption matrix: volterra against three reference codes

This matrix decides one claim: does volterra subsume the liquid-crystal and
active-nematic codes from Daniel Beller's group. It does not. The matrix
below is strict about why: a row is marked `covered` only where a callable
volterra entry point was located in the source and cited by file and line;
`partial` where volterra has something related but narrower, untested at the
relevant scope, or interface-incompatible; `gap` where nothing exists, split
into `gap, additive` (more terms or more geometry inside volterra's existing
design, needing no change to how it is built) and `gap, structural` (needing
a different design). The split carries an asymmetry: nearly everything
volterra lacks relative to open-Qmin is additive, unwritten rather than out
of reach, while nearly everything open-Qmin lacks relative to volterra is
structural, out of reach without a redesign of a solver built to minimise an
energy on a simple-cubic lattice. See the Verdict section for the count and
the reasoning row by row.

**Read the open-Qmin rows against what open-Qmin is, not what its name
suggests.** Direct source search (grep across `src/` for Navier-Stokes,
Beris-Edwards, active stress, and any velocity field) finds none. open-Qmin
is a passive Landau-de Gennes free-energy minimiser; every one of its six
updaters descends from `equationOfMotion` and minimises energy, and its
`Time` field is a step counter, not physical time. volterra is a
time-integrated active-nematohydrodynamics solver. The two solve different
problems, and no row below should be read as a like-for-like exchange of one
for the other; the comparison is which named capabilities each side has, not
which architecture is faster or better.

**Read the open-zetar rows as notebooks, not an interface.** The repository
(`github.com/Brandonkl/open-zetar`, cloned at `/home/downloads-bulk/open-zetar`,
commit `1e8c740`) is three Jupyter notebooks (`BE_NS_2D.ipynb`, `BE_NS_3D.ipynb`,
`Fmin_2D.ipynb`) plus three plotting scripts and a `requirements.txt`. There
is no package, no CLI, no test suite, and no `LICENSE` file, a fact recorded
here because it bears on whether any of that code could be reused, not as a
comment on the author. "volterra covers what open-zetar does" is a claim
about numerical capability only, never about interface compatibility.

**One gap belongs to neither side.** `flow-solver.py`, `braid_tracker.py` and
`viscometric_analysis.py` (the reference code behind Klein et al.,
arXiv:2503.10880) implement none of periodic-orbit finding, shooting,
Poincare sections, return maps, or Floquet analysis; the word "golden" or
"silver" appears nowhere in those files, which label runs only by active
length and coherence length. The golden/silver classification is downstream
analysis the paper's authors did by hand, not code either side ships. Credit
neither open-zetar nor volterra with periodic-orbit discovery below.

## 1. Free energy terms

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| L1 one-constant elastic term, exposed as a runtime coefficient | open-Qmin | `volterra-core::ActiveNematicParams{,3D}::k_r` (`volterra-core/src/lib.rs:84,342`) | partial: same one-constant approximation, no multi-constant path on either side to compare beyond L1 |
| L2, L3, L4, L6 multi-constant elasticity (`--L2 --L3 --L4 --L6` CLI flags) | open-Qmin | none | **gap, additive**: every volterra crate uses one scalar elastic constant; the extra terms would enter the molecular field alongside the existing one-constant term, needing no change to the domain representation |
| Bulk phase constants a, b, c (cubic B term in the bulk potential) | open-Qmin | `ActiveNematicParams3D::b_landau`, wired into the 3D molecular field (`mol_field_3d::cubic_bulk_term`, `volterra-solver/src/mol_field_3d.rs`) | partial: volterra's 3D bulk carries the cubic term, derived and validated against a closed-form uniaxial equilibrium (`BENCHMARKS.md` section 1, "The cubic bulk term"); its 2D bulk still does not (`volterra-core/src/lib.rs:71`, no `b` field) |
| E-field and H-field coupling, including a spatially varying field loaded from file | open-Qmin | none | **gap, additive**: a linear term added to the molecular field, on top of the file-loading path volterra already has for `--theta-ic` |
| One-constant LdG bulk + elastic term, 2D (no cubic term) | open-zetar (`BE_NS_2D.ipynb`) | `volterra-solver::molecular_field` (`volterra-solver/src/lib.rs:133-149`) | covered |
| One-constant LdG bulk + elastic term, 3D, with cubic B term | open-zetar (`BE_NS_3D.ipynb`) | `mol_field_3d::molecular_field_3d`/`_par` (`volterra-solver/src/mol_field_3d.rs:34-151`) | covered |
| Active stress `zeta*Q`, uniform | open-zetar | `zeta_at` via `ActiveNematicParams{,3D}` | covered |
| Flow-alignment/co-rotation term `lambda` | open-zetar | `S(W,Q)` term in the same molecular-field/stress kernels | covered |
| One-constant LdG bulk + elastic term, `dt=0.005`-scale explicit Euler gradient-descent minimiser (`Fmin_2D.ipynb`) | open-zetar | none | **gap, additive**: a steepest-descent updater against forces volterra already computes (Section 5) |
| One-constant LdG bulk (`A`), no B/C cubic term, `chi=1` flow-alignment (paper's name; code's `lambda`), Frank constant `K=2^14` | flow-solver.py / Klein et al. | `volterra-cgpo::Params` (`volterra-cgpo/src/lib.rs:34-133`) | covered: bit-for-bit kernel concurrence against this exact reference, `COMPARISON.md` section 1 |
| Active stress `zeta = K/als^2`, spatial coherence length `ncl` | flow-solver.py | `Params::new` (`lib.rs:108-109`) | covered |
| Spatially varying activity field `zeta(x)` (not in flow-solver.py; volterra-only) | n/a | `volterra-core::ActiveNematicParams::zeta_field` | covered, exceeds the reference codes |
| Cahn-Hilliard phase field with Maier-Saupe coupling to Q, spontaneous curvature and Gaussian-curvature coupling | none of the three reference codes | `volterra-solver::ch_step_etd_3d`, `ch_step_etd_enriched_3d` (`volterra-solver/src/ch_3d.rs:86-401`) | covered, exceeds the reference codes; not benchmarked against any of the three, since none has this term |
| Helfrich bending energy, curved-surface DEC | none of the three | `volterra-dec::bending::{bending_energy,bending_gradient}` (`volterra-dec/src/bending.rs:359,387`) | covered on volterra's own terms; a second, superseded implementation (`helfrich.rs:64,89`) is self-documented as broken (wrong sign, wrong prefactor) and kept only for one legacy caller |

## 2. Boundary conditions and anchoring

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Homeotropic anchoring | open-Qmin | none | **gap, additive**: a different Dirichlet target on the same enforced-BC mechanism volterra already applies |
| Degenerate-planar anchoring | open-Qmin | none | **gap, additive** (only two anchoring types exist on the open-Qmin side; volterra has neither, and both are more Dirichlet-target cases on the existing mechanism) |
| Nobili-Durand fixed anchoring, Fournier-Galatola degenerate-planar anchoring as an energy functional (`f_boundary`, strength `W`) | open-Qmin | none | **gap, additive**: all volterra anchoring is hard Dirichlet, never an anchoring free-energy term, but the term would enter the same molecular-field machinery as the bulk free energy already does |
| Periodic boundaries only (BE_NS_2D/3D; `make_boundary` raises `ValueError` for anything else) | open-zetar | `QField2D`/`QField3D` modulo indexing (`volterra-fields/src/lib.rs:105-115`, `qfield3d.rs:78-90`) | covered |
| No-slip velocity at a domain wall | open-zetar (implicit; open-zetar has no walls, only periodic) | `apply_u_boundary_conditions`, DEC (`volterra-dec/src/boundary_conditions.rs`) and CGPO (`volterra-cgpo/src/bc.rs:101-116`) | covered, exceeds open-zetar (which has no wall BC at all, periodic only) |
| Tangential strong (Dirichlet) anchoring, winding-index-parameterised, on a circular boundary (Eq. 1, arXiv:2503.10880) | flow-solver.py | `volterra-cgpo::bc::apply_q_boundary_conditions` with `net_charge` (`volterra-cgpo/src/bc.rs:242-281`), boundary geometry `boundary::circular_boundary` (`boundary.rs:299`) | covered, **added during this dispatch**; not previously present, not validated against a captured Python reference the way the nephroid path was (see `docs/REPLICATION.md`) |
| Lions-slip velocity BC at the wall (as derived in the reference's own comments) | flow-solver.py | n/a | n/a: checked directly, the reference's own code never executes this derivation (`apply_u_boundary_conditions` computes it then overwrites with `u=0`); both sides are actually plain no-slip, see `docs/REPLICATION.md` |
| No-slip velocity, tangential Dirichlet Q anchoring, Neumann pressure, epitrochoid/nephroid confinement (`bc_label='epitrochoid'`) | flow-solver.py | `volterra-cgpo::bc::{apply_u_boundary_conditions,apply_q_boundary_conditions,apply_p_boundary_conditions}`, `boundary::nephroid_boundary` (`bc.rs:101,247,152`; `boundary.rs:168`) | covered, bit-for-bit kernel concurrence (`COMPARISON.md`) |
| Colloidal inclusions embedded in the domain (spheres, spherocylinders) with their own anchoring | open-Qmin | none | **gap, structural**: volterra's domain representation is either a full periodic box or a single confining outer boundary; an interior excluded region with its own boundary bookkeeping is not a case of the existing mechanism |
| Patterned/lithographic substrate anchoring islands | open-Qmin | none | **gap, additive**: a spatially varying anchoring specification on the same flat boundary, analogous to the spatially varying activity field (`zeta_field`) volterra already carries |

## 3. Geometry and confinement

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Rectangular lattice, optionally with a colloidal sphere, spherocylinder, wall, cavity, cylinder, or capillary, or an arbitrary shape loaded from a boundary file | open-Qmin | none of the inclusion geometries; flat periodic only | **gap, structural** for every inclusion/cavity/capillary shape, for the same domain-representation reason as the colloidal-inclusion row above; volterra has no curved-manifold support in open-Qmin's sense either (open-Qmin has zero curved-manifold code, so this is not a place volterra trails open-Qmin, it is a place neither side has the other's geometry) |
| Flat periodic 2D and 3D lattice | open-zetar, flow-solver.py | `QField2D`/`QField3D` (`volterra-fields/src/lib.rs:56`, `qfield3d.rs:21`) | covered |
| Circular confinement, tangential winding anchoring, arbitrary half-integer topological charge | flow-solver.py (Eq. 1 steady-winding circle) | `boundary::circular_boundary` (`boundary.rs:299`), **added this dispatch** | covered, added; unvalidated against a captured Python reference beyond the q=1 sanity check (`docs/REPLICATION.md`) |
| Epitrochoid family: cardioid, nephroid, trefoiloid confinement with tangential anchoring | flow-solver.py | `volterra-dec::epitrochoid::{disk_mesh,epitrochoid_mesh}` (`volterra-dec/src/epitrochoid.rs:115-302`, DEC solver) and, separately, `volterra-cgpo::boundary::nephroid_boundary` (FD solver, `boundary.rs:168`) | covered, by two independent, non-interoperating volterra solvers rather than one |
| Sphere (icosphere), intrinsically curved 2-manifold, with Q-tensor evolution and Stokes flow | none of the three reference codes | `volterra-dec::mesh_gen::icosphere` (`mesh_gen.rs:96-107`), extensively tested (`test_convergence.rs`, `test_helfrich_exact.rs`, `test_evolving_domain.rs`) | covered, exceeds all three reference codes, none of which has curved-manifold support |
| Torus, curved 2-manifold | none of the three | `mesh_gen::torus_mesh` (`mesh_gen.rs:121-184`) | partial: geometry and curvature operators are tested (`tests/test_analytic_torus.rs`), but no Q-tensor evolution or Stokes flow has ever been run on it |
| A volumetric (tetrahedral) curved 3-manifold | none of the three | none | **gap, structural**: every "curved manifold" in volterra is a 2-manifold embedded in ambient R^3 (`Mesh<M,3,2>`); no tetrahedral mesh type exists anywhere in the workspace, and the DEC operator stack assumes a simplicial surface, not a volume |
| Deforming/evolving curved domain, shape velocity coupled to active stress | none of the three | `EvolvingDomain::{deform,displace_normal,shape_velocity_active}` (`volterra-dec/src/evolving_domain.rs:69-266,428`) | covered, exceeds all three; only exercised on the sphere |

## 4. Initial conditions

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Random Q-tensor perturbation | open-Qmin, open-zetar, flow-solver.py | `QField2D/3D::random_perturbation`, `QFieldDec::random_perturbation` (`volterra-fields/src/lib.rs:82-101`, `qfield3d.rs:44-63`, `volterra-dec/src/qfield_dec.rs:52`) | covered |
| File-loaded initial configuration (`--initialConfigurationFile`) | open-Qmin | `--theta-ic` (`volterra/src/cli.rs:170-172,482-517`) | partial: file-loaded IC exists, but only for the CGPO subcommand's scalar theta grid, not the Cartesian/DEC Q-tensor solvers |
| "Completely random director everywhere" per the code comment; a structured sin-wave field in practice, for 2D (`BE_NS_2D.ipynb`; `Fmin_2D.ipynb` shares the comment and matches it, its field is uniformly random) | open-zetar | n/a | n/a: recorded here as a reference-code discrepancy between comment and behaviour (the 2D active/NS notebook's `randomize` overwrites its own uniform-random `phi` with a deterministic sine pattern before using it), not a volterra claim |
| A director drawn uniformly on the unit sphere, 3D (`BE_NS_3D.ipynb`) | open-zetar | `QField3D::random_perturbation` | covered |
| Random per-site theta on `[0, pi)`, masked to the interior | flow-solver.py | `volterra-cgpo/src/bin/cgpo_fd.rs` random-theta IC path (mirrors `flow-solver.py`'s masked random init) | covered |
| Defect-seeded initial condition (N prescribed +-1/2 disclinations placed analytically) | none of the three reference codes ship one either | none | **gap, additive** on both sides: a field-construction function using the same `QField` types volterra already has, not a place volterra trails the references, since none of them has this |
| numpy-array-loaded field | none of the three (each loads its own native format) | `PyQField2D/3D::from_numpy` (`volterra-py/src/lib.rs:207-221`, `bindings_3d.rs:208-223`) | covered, Python-only, no Rust/CLI equivalent |

## 5. Integrators and minimisers

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| FIRE (fast inertial relaxation engine), the only minimiser wired into the binary | open-Qmin | none | **gap, additive**: an updater against the forces volterra already computes each step |
| Adam, Nesterov accelerated gradient, plain gradient descent, L-BFGS minimisers, present in source but not wired into the `openQmin` binary's CLI | open-Qmin | none | **gap, additive**, same reasoning as FIRE |
| Explicit Euler, gradient descent to a free-energy minimum (`Fmin_2D.ipynb`) | open-zetar | none | **gap, additive**: no minimiser of any kind exists anywhere in volterra, confirmed by a workspace-wide search for `todo!`/`unimplemented!`/minimiser-shaped code, but a steepest-descent updater is additive for the reason given above; volterra's own documentation (`BENCHMARKS.md` Sections 1 and 6) already states its Euler relaxation settles at a nonzero residual floor rather than true equilibrium |
| Explicit Euler with Navier-Stokes coupling, uncapped Jacobi pressure relaxation | open-zetar | `volterra-solver::EulerIntegrator{,3D}` (`volterra-solver/src/lib.rs:312-322`, `beris_3d.rs:86-101`) and the fused rayon path `mol_field_3d::euler_step_fused_par` (`mol_field_3d.rs:159-251`) | covered |
| Explicit Euler, upwind advection, capped pressure-Poisson relaxation (`max_p_iters`), `dt=1e-4` | flow-solver.py | `volterra-cgpo::step` (`volterra-cgpo/src/step.rs`), driven by `cgpo_fd` | covered, bit-for-bit kernel concurrence |
| RK4 time integration | none of the three | `sim::integrate::rk4` (`volterra-core/src/sim/integrate.rs:10-35`), 2D `RK4Integrator` (`volterra-solver/src/lib.rs:333-352`) | covered, exceeds the reference codes; `RK4Integrator3D` (`beris_3d.rs:103-148`) is defined and exported but has zero production or test consumers |
| Semi-implicit / ETD time-stepping for a phase field or bending flow | none of the three | `ch_step_etd{,_3d,_enriched_3d}`, `flow::semi_implicit_step` (`volterra-solver/src/lib.rs:943-1058`, `ch_3d.rs`, `volterra-dec/src/flow.rs:264`) | covered, exceeds the reference codes |

## 6. Defect detection

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Three per-site scalar defect measures, no line tracking or topological classification | open-Qmin | `volterra-solver::scan_defects` (2D, `lib.rs:715-739`), `defects_3d::scan_defects_3d` (3D disclination lines, `defects_3d.rs:38-41`) | covered, exceeds open-Qmin (volterra tracks 3D disclination lines and frame-to-frame topological events; `defects_3d::track_defect_events`, `defects_3d.rs:61-68`) |
| Saddle-splay ("ss") Jacobian scalar field, thresholded and clustered, for plotting only; not exported, tracked, or used for braid analysis | open-zetar (`plot_2D.py`, `plot_2D_Fmin.py`) | `volterra_braid::detect_defects` (`volterra-braid/src/defect.rs:32-111`) | covered, exceeds open-zetar (volterra's detector output is a first-class, tracked, braid-analysable data type, not a plotting side effect) |
| Per-cell Jacobian defect detection, greedy nearest-neighbour tracking (fixed defect count from frame 0, no creation/annihilation handling), braid-word extraction, topological entropy | braid_tracker.py (Klein et al. reference) | `volterra_braid::{detect_defects,track,extract_braidword,topological_entropy}` (`volterra-braid/src/defect.rs:32`, `track.rs:23-64`, `braidword.rs:242`, `entropy.rs:56`) | covered, verified to 1e-9 against the paper's closed-form golden/silver entropies (`entropy.rs`); volterra's tracker has the same fixed-count, no-creation/annihilation limitation as the reference (`track.rs`), so this is a shared limitation, not a volterra deficit relative to the reference |
| 3D disclination-line or -loop braid/topology extraction | none of the three (all defect-braid analysis in this corpus is 2D) | none | **gap, structural**: `volterra-braid` is built around 2D point defects end to end (detection, tracking, braid-word extraction); a 3D line/loop tracker is a different data model, not an extension of the existing one; needed for the 3D "novel material" paper (arXiv:2607.10234), not attempted this dispatch |

## 7. Observables

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Per-term and total free energy, reported at each save | open-Qmin | `StepStats.energy: Option<f64>` (`volterra-core/src/sim/stats.rs:13`) | **gap, additive** in practice: the field exists but is never populated by any `PhysicsStep` implementation in the workspace (checked across every Cartesian, 3D, DEC and CGPO runner); populating it is summing a density already computed pointwise in the molecular-field kernel |
| Order parameter statistics (mean S, variance) | open-Qmin, open-zetar, flow-solver.py | `QField2D::mean_order_param`, `QField3D::scalar_order_s`, `ScalarField2D::variance` (`volterra-fields/src/lib.rs:221-229,424-464`, `qfield3d.rs:140-188`) | covered |
| Biaxiality | open-Qmin | `QField3D::biaxiality_p` (`qfield3d.rs`) | covered |
| Velocity/flow diagnostics: divergence, strain rate, vorticity | open-zetar, flow-solver.py | `VelocityField3D::{divergence,velocity_gradient_at}` (`volterra-fields/src/fields3d.rs:45-132`) | covered |
| Stress tensor (`Pi_S`, `Pi_A`), computed internally, never exported from a runner | flow-solver.py, open-zetar | none exported | **gap, additive**: volterra's CGPO stress kernels compute the same quantities internally but no runner returns them as an observable, matching the reference's own limitation rather than exceeding it; exporting an already-computed field is not a redesign |
| Vorticity noise-to-signal ratio, Lyapunov exponent, line-stretching/E-tec entropy per swap | flow-solver.py / Klein et al. (the paper's own analysis, not the released code) | none | **gap, additive**: these are the paper's own downstream statistical analyses, not part of the released `flow-solver.py`/`braid_tracker.py`; each is a post-processing function over an existing trajectory, not a solver redesign |

## 8. Output formats

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| Plain ASCII output only | open-Qmin | `.npy` (`volterra-core::sim::snapshot::write_npy`, `snapshot.rs:13-46`), JSON metadata | covered, exceeds open-Qmin in format range, not compatible with it (different formats entirely) |
| `.npz` raw arrays, automatic PNG/MP4 rendering via matplotlib/pyvista and ffmpeg | open-zetar | `.npy` snapshots, no automatic rendering | partial: volterra writes a comparable binary array format but has no built-in plotting/video pipeline; `tools/viz` is a separate, not-automatically-invoked path |
| Plain-text `%.18e` frames, numpy-`savetxt`-compatible layout | flow-solver.py | `volterra-cgpo::output::{write_2col_txt,write_1col_gauge_fixed}` (`output.rs:29,46`) | covered, bit-for-bit format match |
| VTK/VTU, HDF5, or XDMF, directly consumable by ParaView | none of the three (all three use raw arrays or ASCII, not a visualisation-standard format) | none | **gap, additive** on all sides: a new writer over existing field data, not a place volterra trails the references |
| Checkpoint/restart from a saved mid-run state | none of the three expose this either | none | **gap, additive** on all sides: a state serialiser/deserialiser over the same field types the existing `.npy` writer already touches |

## 9. Parallelism

| Reference capability | Code | volterra entry point | State |
|---|---|---|---|
| MPI domain decomposition (hard dependency, `find_package(MPI REQUIRED)` unconditional even without CUDA), rank-to-GPU binding via `chooseGPU` | open-Qmin | none | **gap, structural**: no MPI anywhere in volterra; a rank-based halo-exchange model is a different execution model from rayon shared-memory threading, not an extension of it |
| CUDA, optional at build time | open-Qmin | none | **gap, structural**: no GPU code of any kind (CUDA, wgpu, OpenCL) anywhere in volterra; a GPU kernel is a different execution model from a CPU-threaded loop, not an extension of one |
| numba `parallel=True`/`prange` CPU threading across the whole grid (`os.cpu_count() - 1` threads) | open-zetar | rayon `.into_par_iter()`, but only on the 3D **dry** path (`mol_field_3d.rs:77-251`) | partial: volterra's 2D Cartesian solver, 3D wet/BECH path, the entire DEC engine, and `volterra-braid` are single-threaded; only the 3D dry path and CGPO's row-chunked kernels are parallel |
| Single-threaded, numba JIT compiled | flow-solver.py | `volterra-cgpo`'s row-chunked rayon kernels, which switch to rayon above a 250,000-cell threshold (`par_gate.rs:29,62-78`) | covered, exceeds flow-solver.py (measured 4.6x single-thread-vs-32-core-numba speedup, `COMPARISON.md` section 2), single-threaded below the threshold |

## Verdict

Counting every row above (`n/a` rows excluded, since they resolve a specific
correction rather than assert coverage): **35 covered, 6 partial, 22 gap**
(16 additive, 6 structural).

The split carries the asymmetry in this matrix. Nearly everything volterra
lacks relative to open-Qmin is additive: a minimiser is an updater against
forces volterra already computes each step; multi-constant elasticity and
field coupling are more terms in a molecular field volterra already
assembles; homeotropic and degenerate-planar anchoring are different
Dirichlet targets on a mechanism volterra already has; an anchoring
free-energy functional, a populated energy observable, and file-format
writers all sit on data volterra already has in hand. Only colloidal
inclusions and patterned/lithographic substrates are structural on
volterra's side, since they need an interior excluded region volterra's
domain representation does not have. Read the other direction, the same
structural line falls on open-Qmin: it minimises an energy on a simple-cubic
lattice, and activity, hydrodynamic flow, or a curved background would each
make it a different solver, not an extension of the one that exists. The
gap most likely to close with ordinary engineering effort is the
minimiser and multi-constant-elasticity family; the gap least likely to
close without a design change is domain topology, on both sides.

The gaps that matter for the subsumption claim, in descending order of how
much they cost the claim:

1. **No minimiser of any kind (additive).** open-Qmin's and open-zetar's
   entire equilibrium-physics offering (FIRE, gradient descent, and the
   wired-but-unused Adam/Nesterov/L-BFGS family) has no volterra
   counterpart. This is the single largest gap by row count: it is not a
   missing feature at the margins, it is a different mode of solving the
   same equations that volterra cannot do at all, and volterra's own
   documentation already says so. It is additive in the sense above (an
   updater against existing forces), which bounds how much engineering it
   would take, not whether it currently exists.
2. **No colloidal inclusions, cavities, capillaries, or patterned
   substrates (structural for inclusions/cavities/capillaries; additive for
   patterned substrates alone).** open-Qmin's headline results (Saturn-ring
   metastability, patterned-boundary defect arrays) are unreachable in
   volterra on geometric grounds alone, independent of the minimiser gap.
3. **No multi-constant Frank elasticity, no anchoring free-energy
   functional, no homeotropic or degenerate-planar anchoring (additive).**
   Every volterra crate uses one scalar elastic constant and hard Dirichlet
   anchoring; open-Qmin's L2/L3/L4/L6 and Rapini-Papoular-style anchoring
   terms have no volterra equivalent, though each enters machinery volterra
   already has.
4. **No 3D disclination-line braid/topology extraction (structural).**
   volterra's braid machinery, its most distinctive capability relative to
   all three reference codes in 2D, is strictly 2D. This blocks any claim on
   the 2026 3D "novel material" paper (arXiv:2607.10234) regardless of the
   solver-level 3D capability volterra already has.
5. **The new circular-boundary and variable-`net_charge` code is
   unvalidated.** Added during this dispatch to make the paper's
   steady-winding-circle boundary reachable at all, it has not been checked
   against a captured Python reference the way the nephroid path has; see
   `docs/REPLICATION.md` for the attempted PDE-level reproduction and where
   it currently fails.

Set against these: volterra's 3D solver, DEC curved-manifold engine
(sphere, tested; torus, geometry-only), Cahn-Hilliard/Maier-Saupe coupling,
RK4 and semi-implicit integrators, and rayon-threaded throughput have no
counterpart in any of the three reference codes, and are not claimed as
subsuming them since none of the three has the capability to compare
against.
