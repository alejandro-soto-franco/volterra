# Volterra Benchmark Suite and Documentation Site

**Date:** 2026-04-09
**Scope:** DEC-based active nematic solver on arbitrary Riemannian manifolds, benchmark suite validating correctness and performance, documentation site at volterra.sotofranco.dev.
**Motivation:** Dan Beller (JHU) asked for evidence that volterra reproduces the expected behaviour of active nematics. The goal is a comprehensive suite covering flat, confined, and curved geometries, with quantitative comparison against known results, the arXiv:2503.10880 paper (Klein, Soto Franco et al.), and open-Qmin.

---

## 1. DEC Solver Architecture

### 1.1 Core principle

A single DEC-based solver handles all geometries: flat periodic domains (T^2, T^3), confined planar domains with boundary (disk, epitrochoids), closed surfaces (S^2, torus), and 3D manifolds. The solver discretises the Beris-Edwards nematohydrodynamics equations on simplicial complexes using discrete exterior calculus with gauge-invariant parallel transport.

### 1.2 Equations of motion

On a Riemannian manifold (M, g), the Q-tensor evolves via Beris-Edwards:

$$
\partial_t Q_{ij} + u_k \nabla_k Q_{ij} = \frac{1}{\gamma} H_{ij} + \chi S E_{ij} + [Q, \omega]_{ij} - 2\,\mathrm{Tr}[QE]\,Q_{ij}
$$

where $H_{ij} = -Q_{ij} A(1 - \mathrm{Tr}[Q^2]) + K_{\mathrm{Frank}} \nabla^2_{\mathrm{conn}} Q_{ij}$ is the molecular field, $\nabla^2_{\mathrm{conn}}$ is the connection Laplacian on symmetric traceless (0,2) tensors, $E_{ij}$ and $\omega_{ij}$ are the rate-of-strain and vorticity tensors, and the velocity field $u$ satisfies the incompressible Stokes equations with active stress.

### 1.3 Connection Laplacian (Lichnerowicz operator)

The connection Laplacian is discretised via DEC with parallel transport of Q along mesh edges. No manual Weitzenboeck curvature correction is needed; the curvature enters automatically through the transport.

**2D surfaces:** Q is stored as a complex number $z = q_1 + i q_2$ per vertex (section of the line bundle $L^2$). Parallel transport along edge $(v, w)$ multiplies by $\exp(i \cdot 2\alpha_{vw})$ where $\alpha_{vw}$ is the Levi-Civita connection angle. The DEC Laplacian is:

$$
(\Delta_{\mathrm{conn}} Q)_v = \sum_{w \sim v} w_{vw} \left[ e^{i \cdot 2\alpha_{vw}} z_w - z_v \right]
$$

where $w_{vw}$ are cotangent weights. This is mathematically equivalent to the gauge-invariant discretisation of Zhu, Saintillan & Chern (arXiv:2405.06044).

**Verified properties (SymPy):**
- On S^2: eigenvalues $-(l(l+1)-4)/R^2$, shift of $+4K$ from scalar Laplacian (spin-2)
- On flat space ($K=0$): reduces to component-wise cotangent Laplacian
- Weitzenboeck endomorphism $W = -2n\kappa\,h$ on constant-curvature $n$-manifolds (verified $n = 2, 3, 4, 5$)

**3D manifolds:** Q is stored as a 5-component vector (symmetric traceless $3 \times 3$) per vertex. Parallel transport along an edge lifts the SO(3) frame rotation to a $5 \times 5$ orthogonal matrix acting on the symmetric traceless representation. This matrix is precomputed per edge.

**Verified properties (NumPy):**
- $5 \times 5$ representation is orthogonal ($\|R^T R - I\| = 10^{-16}$) and a group homomorphism ($\|R(g_1 g_2) - R(g_1)R(g_2)\| = 8 \times 10^{-16}$)
- Correct in arbitrary dimension: transport encodes the full Riemann tensor, not just scalar curvature

### 1.4 Stokes solver

**Closed 2D surfaces (S^2, torus):** Stream-function formulation. Incompressible flow $u = *\mathrm{d}\psi$ for a 0-form $\psi$. The Stokes equation reduces to:

$$
\eta\,\Delta_0(\Delta_0 + K)\,\psi = *\mathrm{d}(F_{\mathrm{active}})
$$

where $\Delta_0$ is the scalar Laplace-Beltrami and $K$ is the Gaussian curvature. Verified: eigenvalues on S^2 match the Hodge Laplacian on divergence-free 1-forms.

**Bounded 2D domains (disk, epitrochoids):** Biharmonic equation $\eta \Delta^2 \psi = (\nabla \times F)_z$ with boundary conditions $\psi = 0$ (no penetration) and $\partial\psi/\partial n = 0$ (no slip). Solved via vorticity-stream function splitting on the DEC mesh.

**3D periodic (T^3):** FFT solver (existing, used for validation against DEC).

### 1.5 Boundary conditions (confined 2D)

**Strong planar anchoring (Dirichlet on Q):** At boundary vertex with position parameter $\theta$, the anchoring direction is prescribed by the boundary tangent (for tangential anchoring) or by a winding-number formula $\mathbf{n}(\theta) = (-\sin(q\theta), \cos(q\theta))$ (for steady-winding circle). The Q-tensor is fixed: $Q_{ij} = S_0(n_i n_j - \delta_{ij}/2)$.

**No-slip (Dirichlet on velocity):** $\psi = 0$ and $\partial\psi/\partial n = 0$ at boundary vertices.

**Weak anchoring (Rapini-Papoular):** Surface energy penalty $W_s \int_{\partial\Omega} (Q_{ij} - Q^0_{ij})^2 \,\mathrm{d}s$ added to the free energy. Enters as a Robin-type BC on Q.

**Free boundary:** Q unconstrained at boundary, no anchoring energy.

### 1.6 Epitrochoid boundary generation

Parametric curve from Eqs. 3-4 of arXiv:2503.10880:

$$
x(u) = \frac{r}{2q}\left[(2q-1)\cos u + \cos((2q-1)u)\right], \quad y(u) = \frac{r}{2q}\left[(2q-1)\sin u + \sin((2q-1)u)\right]
$$

for $u \in [0, 2\pi]$. Number of cusps $= 2(q-1)$. Mesh generation: sample boundary, add interior points, constrained Delaunay triangulation, adaptive refinement near cusps (curvature diverges at cusp points; no vertex placed exactly at a cusp).

### 1.7 Time integration

RK4 (primary). Euler retained as a secondary integrator for validation (confirm both converge to the same steady state; Euler at smaller timestep).

### 1.8 New types

| Type | Location | Description |
|------|----------|-------------|
| `QFieldDec<D>` | `volterra-dec` | Q-tensor on a simplicial complex, $D = 2$ or $3$ |
| `ConnectionLaplacianDec` | `volterra-dec` | Precomputed transport matrices + cotangent weights |
| `StokesSolverDec` | `volterra-solver` | Stream-function biharmonic solver on DEC meshes |
| `EpitrochoidMesh` | `volterra-solver` or utility | Parametric boundary + triangulated interior |
| `AnchoringBC` | `volterra-core` | Enum: `StrongPlanar(n)`, `StrongHomeotropic`, `WeakRP(W_s, n)`, `Free` |

---

## 2. Benchmark Suite

### 2.1 Tier 1: Passive LdG validation (analytical comparisons)

| ID | Geometry | Observable | Expected | Reference |
|----|----------|------------|----------|-----------|
| 1a | S^2 | Equilibrium defect config | 4 half-charge defects, tetrahedral arrangement | Lubensky & Prost (1992) |
| 1b | T^2 (flat torus) | Ground state | Defect-free, uniform $S = S_{\mathrm{eq}}$ | Exact |
| 1c | Embedded torus | Defect positions | Defects at outer equator (max $K > 0$) | Vitelli & Nelson (2006) |
| 1d | Disk, $q = 1$ anchoring | Defect position + energy | +1 defect at centre, matches analytical Frank energy | de Gennes & Prost |
| 1e | Saturn ring (colloidal sphere) | Ring vs. hedgehog crossover | Matches open-Qmin at equivalent resolution | Sussman & Beller (2019) |

### 2.2 Tier 2: DEC vs. Cartesian self-consistency

| ID | Geometry | Observable | Tolerance |
|----|----------|------------|-----------|
| 2a | T^2 (DEC triangulation) | Defect density $\rho_d$ vs. $\zeta_{\mathrm{eff}}$ | Within 5% of Cartesian solver |
| 2b | T^3 (DEC tetrahedralisation) | Disclination line density vs. activity | Within 5% |
| 2c | T^2 with flow | Defect density, velocity RMS, mean vorticity | Within 5% |

### 2.3 Tier 3: Reproduction of arXiv:2503.10880

| ID | Geometry | Observable | Expected |
|----|----------|------------|----------|
| 3a | Disk, $q = 3/2$ | Braid word | $\{\sigma_2^{-1}\sigma_1\}$ (golden braid) |
| 3b | Disk, $q = 4/2$ | Braid word | $\{\sigma_1\sigma_3\sigma_2\sigma_1^{-1}\sigma_3^{-1}\sigma_2^{-1}\}$ (silver braid) |
| 3c | Disk, $q = 5/2$ | Dynamics | Aperiodic |
| 3d | Cardioid, tangential anchoring | Braid | Golden braid (matches paper Fig. 5) |
| 3e | Nephroid, tangential anchoring | Braid | Silver braid (matches paper Fig. 7) |
| 3f | Trefoiloid, tangential anchoring | Dynamics | Matches paper predictions |
| 3g | All confined geometries | Time-averaged vorticity | Gyre count $= 4|q - 1|$ |
| 3h | Disk, $q = 3/2$ and $q = 4/2$ | Topological entropy per swap | $h_{\mathrm{swap}} \approx 0.53$ ($q = 3/2$), $\approx 0.86$ ($q = 4/2$) |

### 2.4 Tier 4: Curved manifold active nematics (new results)

| ID | Geometry | Observable | Significance |
|----|----------|------------|-------------|
| 4a | S^2, wet active | Defect dynamics, steady-state turbulence | First open-source wet active nematics on S^2 |
| 4b | Embedded torus, wet active | Defect pinning, curvature-driven migration | Curvature-activity interplay |
| 4c | S^2 x R (thin shell) | 3D disclination lines on curved substrate | 3D Q-tensor on curved surface |
| 4d | Cylinder ($S^1 \times \mathbb{R}$, periodic in $z$) | Channel defect dynamics | Intermediate geometry |

### 2.5 Performance benchmarks

| ID | Comparison | Metric |
|----|-----------|--------|
| P1 | DEC vs. Cartesian on T^2 | Wall-clock at matched resolution |
| P2 | volterra vs. open-Qmin on Saturn ring (passive) | Wall-clock at equivalent resolution |
| P3 | DEC on S^2 | Convergence order (error vs. mesh spacing) |
| P4 | Thread scaling | Wall-clock vs. rayon thread count |

---

## 3. Benchmark Cache Architecture

### 3.1 Cache directory structure

```
~/.volterra-bench/
├── cache/
│   ├── {geometry}_{params_hash}_{mesh_hash}_{solver_version}/
│   │   ├── meta.json           # parameters, solver git hash, timestamp
│   │   ├── q_snapshots/        # Q-field .npy files at each snapshot
│   │   ├── defects/            # defect trajectories per snapshot
│   │   ├── stats.json          # per-snapshot statistics (S, rho_d, vorticity, ...)
│   │   └── mesh.json           # mesh definition (vertices, triangles, boundary tags)
│   └── ...
├── figures/                    # generated plots (consumed by volterra-docs)
│   ├── tier1/
│   ├── tier2/
│   ├── tier3/
│   ├── tier4/
│   └── performance/
└── openqmin/                   # cached open-Qmin results
    └── saturn_ring_{params_hash}/
```

### 3.2 Cache invalidation

A simulation result is valid if:
1. The `solver_version` field in `meta.json` matches the current `volterra` git commit hash (or a tagged release)
2. The parameter hash matches (SHA-256 of the JSON-serialised parameter struct)
3. The mesh hash matches (SHA-256 of the sorted vertex/triangle arrays)

Analysis scripts check these before re-running. A `--force` flag bypasses the cache.

### 3.3 Parallelism

Benchmark simulations are independent per (geometry, parameter point). The benchmark runner dispatches them in parallel:
- On a single machine: one tmux pane per simulation, or rayon-parallel within a single process for small grids
- Each simulation writes to its own cache directory (no contention)
- Analysis/plotting scripts run after all simulations complete, reading from cache

---

## 4. Documentation Site (volterra.sotofranco.dev)

### 4.1 Framework

Ported from cartan-docs: Next.js 15, Tailwind v4, KaTeX (client-side), Shiki (build-time code highlighting), Three.js + D3 for interactive visualisations. Computer Modern Serif fonts. Vercel deployment.

### 4.2 Site map

```
/                               Landing (hero + feature grid)
/getting-started/
  installation                  Rust + Python + from source
  quick-start                   First simulation in 10 lines
  concepts                      Q-tensor, BE, defects primer
/theory/
  beris-edwards                 Full BE on Riemannian manifolds
  connection-laplacian          Lichnerowicz, DEC, spin-2 transport
  stokes                        Stream-function, modified biharmonic
  cahn-hilliard                 BECH concentration coupling
  defect-detection              Holonomy, braiding classification
  boundary-conditions           Anchoring types, epitrochoid parametrisation
/benchmarks/
  passive-equilibrium           Tier 1
  self-consistency              Tier 2
  confined-active               Tier 3
  curved-manifolds              Tier 4
  performance                   Scaling, open-Qmin comparison
/api/
  volterra-core                 ActiveNematicParams, VError, Integrator
  volterra-fields               Field types
  volterra-solver               Runners, molecular field, Stokes, CH
  volterra-dec                  DecDomain, DEC operators, manifold solver
  volterra-mars                 MARS presets
  python                        Python API reference
/gallery/
  sphere-defects                S^2 active nematic (Three.js)
  torus-active                  Torus active turbulence (Three.js)
  braiding                      Defect braiding diagrams (D3 + canvas)
  epitrochoid                   Confined flow fields (canvas)
/changelog
```

### 4.3 New components

| Component | Purpose |
|-----------|---------|
| `<BenchmarkTable>` | Formatted comparison tables with pass/fail indicators |
| `<PerformancePlot>` | Embedded scaling/timing charts (static SVG from matplotlib, or D3) |
| `<DefectTrajectoryViewer>` | Interactive 2D defect worldlines + braid diagram |
| `<MeshViewer>` | Three.js triangulated domain with Q-tensor colour overlay |
| `<FeatureComparisonTable>` | volterra vs open-Qmin capability matrix |

### 4.4 Figure pipeline

Benchmark simulations (Rust) write to `~/.volterra-bench/cache/`. Python analysis scripts read cache, generate matplotlib figures, write SVG/PNG to `~/.volterra-bench/figures/`. A build script copies figures into `volterra-docs/public/benchmarks/` before `npm run build`. The docs site never invokes the solver.

---

## 5. Sub-project Decomposition

| Sub-project | Deliverables | Depends on | Estimated scope |
|-------------|-------------|------------|-----------------|
| **A** | 2D DEC Q-tensor solver: `QFieldDec`, connection Laplacian, molecular field, RK4, Tier 1a/1b/2a validation | None | Foundation |
| **B** | Stokes on 2D surfaces + confined domains: stream-function biharmonic, epitrochoid mesh, anchoring BCs, Tier 1d/3a-3c | A |  |
| **C** | Epitrochoid paper reproduction: full Tier 3 benchmarks, braid classification, figure generation | B |  |
| **D** | 3D DEC solver: `QFieldDec3D`, SO(3)->5x5 lift, Tier 2b | A |  |
| **E** | Curved manifold simulations: Tier 4 (S^2, torus, shell, cylinder), performance benchmarks P1-P4 | B, D |  |
| **F** | Documentation site: scaffold from cartan-docs, theory pages, benchmark pages, gallery, deploy | C, E (for content) |  |
| **G** | open-Qmin comparison: build open-Qmin, run Saturn ring, feature table, Tier 1e, P2 | Independent |  |

**Execution order:** A then B then C (delivers paper reproduction first), D in parallel with B-C, E after B+D, F consumes C+E output, G whenever convenient.

---

## 6. Open Questions and Risks

| Risk | Mitigation |
|------|-----------|
| Delaunay triangulation of epitrochoid domains may produce poor-quality elements near cusps | Adaptive refinement via cartan-remesh; no vertex at cusp points |
| Well-centeredness not guaranteed by Delaunay | Accept non-diagonal Hodge stars (denser but correct), or run mesh improvement passes |
| 3D Stokes on general manifolds is complex | Start with 2D surface Stokes (stream function); 3D Stokes on T^3 uses existing FFT |
| open-Qmin build may be non-trivial (CUDA dependency) | CPU-only build is supported; GPU comparison is a stretch goal |
| Mesh generation for S^2, torus | Use icosahedral subdivision (S^2) or rectangular-grid-based triangulation (torus) from cartan-dec |
