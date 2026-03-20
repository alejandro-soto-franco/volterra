# Volterra 3D Sprint Design

**Date:** 2026-03-20
**Scope:** volterra + cartan-geo
**Strategy:** Approach A — extend alongside 2D, nothing existing is touched

---

## Context

volterra is a simulation substrate for the mars-lnp research project: de novo synthesis
of lipid nanoparticles (LNPs) driven by magnetically actuated nematic rotor suspensions
(MARS). The MARS rods are silica-coated Fe3O4 nanorods (aspect ratio >= 5), primarily
uniaxial with weak biaxiality from their oblate cross-section. The governing physics is
fully 3D on a flat periodic domain (R^3).

The existing volterra stack implements a complete 2D solver (Beris-Edwards, Stokes,
Cahn-Hilliard, holonomy defect detection, PyO3 bindings). This sprint adds a parallel
3D layer without modifying anything in the 2D code.

volterra-dec remains a stub this sprint. The flat R^3 regular grid does not need DEC;
that layer becomes relevant when moving to curved manifolds or unstructured meshes.

---

## Crate Responsibilities

### volterra-core

Add `MarsParams3D` alongside `MarsParams` (2D). New fields:

- `nz: usize` (grid depth)
- `b_landau: f64` — cubic LdG coefficient b/3 Tr(Q^3); default 0.0 (uniaxial runs),
  nonzero for biaxial extension
- `chi_b: f64` — biaxial susceptibility; default 0.0
- `chi_a: f64` — magnetic susceptibility anisotropy (dimensionless)
- `b0: f64` — applied field magnitude
- `omega_b: f64` — field rotation frequency

Derived quantity accessors mirror MarsParams: `defect_length()`, `pi_number()`,
`ch_coherence_length()`, `phi_eq()`. Validation covers all new params.

`Integrator<S>` trait is dimension-agnostic; no changes.

### volterra-fields

Three new types alongside existing 2D types:

**`QField3D`**
- Storage: `Vec<[f64; 5]>` of components `[q11, q12, q13, q22, q23]` per vertex,
  row-major index `k = (i*ny + j)*nz + l`
- `q33` recovered as `-(q11 + q22)` on demand
- `embed_matrix3(k)` returns `nalgebra::SMatrix<f64,3,3>` for vertex k
- `laplacian()` component-wise 3D finite difference with periodic wrapping
- `gradient()` returns `Vec<[[f64;3];5]>` (gradient of each component)
- `scalar_order_s()` largest eigenvalue * 3/2 at each vertex
- `biaxiality_p()` eigenvalue spread (secondary observable)
- `director()` eigenvector of largest eigenvalue, shape (N, 3)
- `zeros`, `uniform`, `random_perturbation` constructors

**`VelocityField3D`**
- Storage: `Vec<[f64; 3]>` per vertex
- `divergence()`, `curl()`, `velocity_gradient_tensor()` returning (D, Omega) —
  symmetric strain rate and antisymmetric vorticity

**`ConcentrationField3D`**
- Storage: `Vec<f64>` per vertex
- `laplacian()`, `gradient()`, `mean()`, `max()`

### volterra-solver

New 3D functions alongside existing 2D functions. No existing symbols modified.

**Core PDE:**

`molecular_field_3d(q, params, t) -> QField3D`
- Elastic: H_elastic = K_r nabla^2 Q
- Active LdG: H_active = (zeta_eff/2 - a) Q - 2c Tr(Q^2) Q
  (optional cubic term b/3 Tr(Q^3) when b_landau != 0)
- Magnetic torque: H_mag(t) = chi_a B0^2 [b_hat(t) otimes b_hat(t) - I/3]
  where b_hat(t) = (cos(omega_b t), sin(omega_b t), 0)
- Returns total H = H_elastic + H_active + H_mag

`co_rotation_3d(w, q, lambda) -> QField3D`
- S(W,Q) = lambda(DQ + QD) - (OmegaQ - QOmega) - lambda Tr(DQ) I
- Full 3x3 matrix arithmetic via nalgebra SMatrix

`beris_edwards_rhs_3d(q, vel, params, t) -> QField3D`
- dQ/dt = -u.nabla Q + S(W,Q) + Gamma_r * H
- vel=None skips advection and co-rotation (dry active model)

`stokes_solve_3d(q, params) -> (VelocityField3D, ScalarField3D)`
- Active stress: sigma = -zeta Q
- Divergence of stress as body force
- Spectral solve via 3D FFT: project onto divergence-free modes
- Returns (u, p)

`ch_step_etd_3d(phi, q, params, dt) -> ConcentrationField3D`
- mu = a_ch phi + b_ch phi^3 - kappa_ch nabla^2 phi - chi_ms Tr(Q^2)
- ETD (exponential time differencing) for the stiff kappa_ch nabla^4 phi term
- Maier-Saupe coupling drives lipid to regions of high Q

`EulerIntegrator3D`, `RK4Integrator3D` — implement `Integrator<QField3D>`

**Runners:**

`run_mars_3d(q_init, params, n_steps, snap_every, out_dir, track_defects) -> (QField3D, Vec<SnapStats3D>)`
- Dry active turbulence: Beris-Edwards only, no Stokes, no CH
- Writes `q_{step:06d}.npy` and `defects_{step:06d}.json` to out_dir
- Returns final Q and stats vector

`run_mars_3d_full(q_init, phi_init, params, n_steps, snap_every, out_dir, track_defects) -> (QField3D, ConcentrationField3D, Vec<BechStats3D>)`
- Full coupled: Beris-Edwards + Stokes + Cahn-Hilliard
- Writes q, phi, vel, defects, stats

`SnapStats3D` fields: `time`, `mean_s`, `biaxiality_p`, `n_disclination_lines`,
`total_line_length`, `mean_line_curvature`, `n_events`.

**Defect detection (delegates to cartan-geo):**

`scan_defects_3d(q) -> Vec<DisclinationLine>`
- Wraps cartan_geo::disclination::scan_disclination_lines_3d
- Then cartan_geo::disclination::connect_disclination_lines

`track_defect_events(lines_a, lines_b) -> Vec<DisclinationEvent>`
- Wraps cartan_geo::disclination::track_disclination_events

### cartan-geo (new module: disclination)

Three layered functions:

**Layer 1 — `scan_disclination_lines_3d(q: &QField3D) -> Vec<DisclinationSegment>`**
- For each primal edge in the regular grid, compute holonomy of Q around the dual
  loop (the 4 faces surrounding that edge)
- Holonomy angle ~= pi flags the edge as a disclination segment
- Returns raw edge list with associated DisclinationCharge

**Layer 2 — `connect_disclination_lines(segments) -> Vec<DisclinationLine>`**
- Graph traversal connecting segments into ordered vertex sequences
- Produces closed loops or open arcs (terminating at domain boundary)
- Arc-length parameterizes each line
- Computes per-vertex: tangent t(s) via finite difference, curvature kappa(s) = |dt/ds|,
  torsion tau(s) via discrete Frenet-Serret frame

**Layer 3 — `track_disclination_events(lines_a, lines_b) -> Vec<DisclinationEvent>`**
- Detects topology-changing events between two frames:
  - Creation: new loop appears (no matching loop in prev frame)
  - Annihilation: loop disappears (charge-neutral pair gone)
  - Reconnection: two lines exchange endpoints (proximity threshold + connectivity change)
- Line velocity: mean displacement of matched vertices between frames
- Returns Vec<DisclinationEvent> with kind, frame, position, participating line IDs

**Types:**

```
DisclinationCharge { Half(Sign), Anti }   // Sign = Pos | Neg
                                          // Designed for Q8 extension later

DisclinationSegment { edge: (usize, usize), charge: DisclinationCharge }

DisclinationLine {
    vertices: Vec<[f64; 3]>,
    tangents: Vec<[f64; 3]>,
    curvatures: Vec<f64>,
    torsions: Vec<f64>,
    charge: DisclinationCharge,
    is_loop: bool,
}

DisclinationEvent {
    kind: EventKind,  // Creation | Annihilation | Reconnection
    position: [f64; 3],
    line_ids: Vec<usize>,
}
```

### volterra-py

New Python classes alongside existing 2D bindings:

- `PyMarsParams3D` — keyword constructor, all derived accessors
- `PyQField3D` — `.q` as numpy `(nx, ny, nz, 5)`, `.scalar_order()` as `(nx, ny, nz)`,
  `.biaxiality()`, `.director()` as `(nx, ny, nz, 3)`, `.from_numpy(arr)` classmethod
- `PyVelocityField3D` — `.u` as `(nx, ny, nz, 3)`
- `PyConcentrationField3D` — `.phi` as `(nx, ny, nz)`
- `PySnapStats3D`, `PyBechStats3D`
- `PyDisclinationLine` — numpy arrays for vertices, tangent, curvature, torsion; charge
  as str; is_loop as bool
- `PyDisclinationEvent` — kind, frame, position, line_ids

Runner bindings:

```python
run_mars_3d(params, q_init, n_steps, snap_every, out_dir, track_defects=False)
run_mars_3d_full(params, q_init, phi_init, n_steps, snap_every, out_dir, track_defects=True)
scan_defects_3d(q) -> list[PyDisclinationLine]
track_defect_events(lines_a, lines_b) -> list[PyDisclinationEvent]
```

---

## Output Layout

Snapshots written to out_dir:

```
q_{step:06d}.npy        shape (nx, ny, nz, 5)
phi_{step:06d}.npy      shape (nx, ny, nz)        [CH runs only]
vel_{step:06d}.npy      shape (nx, ny, nz, 3)     [Stokes runs only]
defects_{step:06d}.json list of DisclinationLine   [if track_defects]
stats.json              list of SnapStats3D / BechStats3D
```

Consistent with existing mars-lnp analysis script conventions.

---

## What Does Not Change

- volterra-core: `MarsParams`, `VError`, `Integrator` — untouched
- volterra-fields: `QField2D`, `VelocityField2D`, `ScalarField2D` — untouched
- volterra-solver: all existing 2D functions — untouched
- volterra-py: all existing 2D Python bindings — untouched
- volterra-dec: remains a stub (DEC on manifolds is post-sprint)
- cartan: all existing crates untouched except cartan-geo gains `disclination` module

---

## Out of Scope (this sprint)

- Q8 biaxial charge classification (MARS rods are uniaxially dominant)
- Immersed boundary method for explicit MARS rod geometry (future extension)
- volterra-dec implementation
- 4D nematics
- Curved manifold support
