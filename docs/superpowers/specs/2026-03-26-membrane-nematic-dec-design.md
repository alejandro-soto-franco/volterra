# Membrane-Nematic DEC: Unified Design Spec

**Date**: 2026-03-26
**Scope**: cartan-dec, cartan-remesh (new), volterra-dec, pathwise-geo
**Motivation**: Port key ideas from Mem3DG (Zhu, Lee & Rangamani, Biophysical Reports 2022) into the volterra/pathwise/cartan ecosystem to enable sharp-interface Helfrich membrane dynamics coupled to Beris-Edwards nematohydrodynamics on triangulated 2-manifolds, with stochastic forcing via edge-based DPD.

**Supersedes**: Extends (does not replace) the Generic Covariant Beris-Edwards spec (2026-03-24). That spec introduced `DecDomain<M>` and `BerisEdwardsDomain` trait on a static mesh. This spec adds mesh deformation, Helfrich energy, adaptive remeshing, variational integration, and stochastic dynamics.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mesh data structure | Adjacency-augmented simplicial (extend `Mesh<M,K,B>`) | Preserves generic K-simplex design, avoids second mesh type, fixes O(V*E) advection |
| Sparse storage | `sprs` (`CsMat`/`CsVec`) | More mature sparse ecosystem than nalgebra-sparse; zero-copy interop via `as_slice()` |
| Helfrich approach | Sharp-interface (mesh = membrane) | LNP membrane deforms in response to nematic field; diffuse-interface answers a different question |
| DPD architecture | Split: pathwise provides noise primitives, volterra assembles forces | Clean separation of stochastic numerics from physics |
| Stochastic integrator | BAOAB with per-edge Shardlow O-step | Second-order configurational accuracy + exact FD per edge |
| Shape-Q coupling | Variational integrator (D') with forced dissipation | Symplectic core preserves energy, dissipation via O-step |
| Remesh trigger | Curvature-CFL spatial criterion + KE-minimum temporal trigger | Physics-aware refinement + minimal damage to symplectic structure |
| Q interpolation after remesh | Parallel transport (edge split) + Frechet mean (collapse) | Respects nematic geometry, dogfoods cartan infrastructure |

---

## Phase A: cartan-dec Upgrades (K-Generic DEC Core)

**Principle**: The DEC infrastructure (adjacency, exterior derivative, Hodge star, Laplacian, advection, divergence) must be generic over the simplex dimension K. Phases B-F specialize to K=3 (triangle meshes) for the membrane physics, but the core must not assume K=3.

### A1. K-Generic Adjacency Maps

Add precomputed adjacency to `Mesh<M, K, B>` at full generality:

```rust
pub struct Mesh<M: Manifold, const K: usize = 3, const B: usize = 2> {
    // existing (unchanged)
    pub vertices: Vec<M::Point>,
    pub simplices: Vec<[usize; K]>,
    pub boundaries: Vec<[usize; B]>,
    pub simplex_boundary_ids: Vec<[usize; K]>,
    pub boundary_signs: Vec<[f64; K]>,
    // new K-generic adjacency
    pub vertex_boundaries: Vec<Vec<usize>>,     // vertex -> incident (K-1)-simplices
    pub vertex_simplices: Vec<Vec<usize>>,       // vertex -> incident K-simplices
    pub boundary_simplices: Vec<Vec<usize>>,     // (K-1)-simplex -> adjacent K-simplices (variable count)
}
```

`boundary_simplices` is `Vec<Vec<usize>>` instead of `Vec<[Option<usize>; 2]>`. For triangles (K=3), an interior edge has exactly 2 co-faces; for tets (K=4), an interior face can have 2 co-tets; but for general K-complexes (or non-manifold meshes) the count is unbounded. `Vec<Vec<usize>>` handles all cases.

Built in `from_simplices` alongside existing edge deduction. A `rebuild_adjacency(&mut self)` method recomputes all three maps from the current `simplices`/`boundaries` arrays, for use after remeshing.

**Convenience accessors for K=3** (non-breaking):

```rust
impl<M: Manifold> Mesh<M, 3, 2> {
    /// For triangle meshes, an interior edge has at most 2 co-faces.
    /// Returns (face_a, Option<face_b>). Panics on non-manifold edges.
    pub fn edge_faces(&self, e: usize) -> (usize, Option<usize>);
}
```

### A2. K-Generic Sparse Operators via sprs

New dependency: `sprs = "0.11"` in cartan-dec's Cargo.toml.

**Exterior derivative** becomes a chain of sparse matrices, one per degree:

```rust
pub struct ExteriorDerivative {
    /// d[k] is the incidence matrix from k-simplices to (k+1)-simplices.
    /// For a K-simplex mesh: d[0] maps vertices->boundaries, d[1] maps boundaries->simplices.
    /// In general, d[k] has dimensions n_{k+1} x n_k.
    pub d: Vec<CsMat<f64>>,
}
```

For a triangle mesh (K=3), `d` has 2 entries: `d[0]` (n_edges x n_vertices) and `d[1]` (n_faces x n_edges). For a tet mesh (K=4), `d` would have 3 entries: d[0] (edges x verts), d[1] (faces x edges), d[2] (tets x faces). Construction via `sprs::TriMat` then `.to_csc()`.

The exactness property `d[k+1] * d[k] = 0` holds for all k and is checkable via `check_exactness()`.

Backward-compat accessors:

```rust
impl ExteriorDerivative {
    pub fn d0(&self) -> &CsMat<f64> { &self.d[0] }
    pub fn d1(&self) -> &CsMat<f64> { &self.d[1] }
    pub fn degree(&self) -> usize { self.d.len() }
}
```

The existing dense constructors remain as `from_mesh_generic_dense` (deprecated).

### A3. K-Generic Hodge Star

The Hodge star is a diagonal operator per degree, mapping k-forms to (n-k)-forms where n is the mesh dimension (K-1 for a K-simplex complex).

```rust
pub struct HodgeStar {
    /// star[k] is the diagonal Hodge star for k-forms.
    /// For a triangle mesh (n=2): star[0] (vertices), star[1] (edges), star[2] (faces).
    /// For a tet mesh (n=3): star[0..4].
    pub star: Vec<DVector<f64>>,
}

impl HodgeStar {
    pub fn star_k(&self, k: usize) -> &DVector<f64> { &self.star[k] }
    pub fn star_k_inv(&self, k: usize) -> DVector<f64>;  // element-wise reciprocal

    // backward compat
    pub fn star0(&self) -> &DVector<f64> { &self.star[0] }
    pub fn star1(&self) -> &DVector<f64> { &self.star[1] }
    pub fn star2(&self) -> &DVector<f64> { &self.star[2] }
}
```

**Generic construction** via circumcentric duality:

```rust
impl<M: Manifold, const K: usize, const B: usize> HodgeStar {
    pub fn from_mesh_generic(mesh: &Mesh<M, K, B>, manifold: &M) -> Result<Self, DecError>
}
```

The circumcentric Hodge star for a k-simplex sigma in an n-dimensional complex is:

```
star_k[sigma] = vol(dual(sigma)) / vol(sigma)
```

where `vol(dual(sigma))` is the (n-k)-dimensional volume of the dual cell and `vol(sigma)` is the k-dimensional volume of the primal simplex.

**Computing dual cell volumes generically**: For a k-simplex sigma, its dual cell is the union of (n-k)-dimensional polytope pieces, one per incident n-simplex. Each piece connects the circumcenters of the chain: sigma subset tau_{k+1} subset ... subset tau_n. The volume of each piece is computed in the tangent space at sigma's circumcenter via `manifold.log` of the circumcenter chain, using the standard simplex volume formula (1/m! * det of edge vectors).

For the concrete cases:
- **n=2 (triangle mesh)**: star0[v] = circumcentric dual cell area (polygon around v), star1[e] = dual edge length / primal edge length, star2[f] = 1 / face area. Same formulas as before, but derived from the general principle.
- **n=3 (tet mesh)**: star0[v] = dual cell volume (polyhedron around v), star1[e] = dual face area / primal edge length, star2[f] = dual edge length / primal face area, star3[t] = 1 / tet volume.

**Mesh geometric primitives** needed on `Mesh<M, K, B>`:

```rust
impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B> {
    /// Volume of a K-simplex (K=3: triangle area, K=4: tet volume).
    /// Computed via Gram determinant in the tangent space at the first vertex.
    pub fn simplex_volume(&self, manifold: &M, s: usize) -> f64;

    /// Volume of a B-simplex (boundary face).
    pub fn boundary_volume(&self, manifold: &M, b: usize) -> f64;

    /// Circumcenter of a K-simplex. Solves the equidistance system in
    /// the tangent space at vertex 0, then exp back to the manifold.
    pub fn simplex_circumcenter(&self, manifold: &M, s: usize) -> M::Point;

    /// Circumcenter of a B-simplex (boundary face).
    pub fn boundary_circumcenter(&self, manifold: &M, b: usize) -> M::Point;

    /// Midpoint of a B-simplex (geodesic barycenter).
    pub fn boundary_midpoint(&self, manifold: &M, b: usize) -> M::Point;
}
```

The existing `triangle_area`, `circumcenter`, `edge_length`, `boundary_midpoint` on `Mesh<M, 3, 2>` become thin wrappers around these generic methods. The flat-specific fast paths (`triangle_area_flat`, `circumcenter_flat`, etc.) remain on `Mesh<Euclidean<2>, 3, 2>`.

Well-centeredness check generalizes: for each K-simplex, verify that its circumcenter has non-negative barycentric coordinates (lies inside the simplex). Returns `Err(DecError::NotWellCentered)` if violated.

### A4. K-Generic Operators Assembly

```rust
pub struct Operators<M: Manifold, const K: usize = 3, const B: usize = 2> {
    pub laplace_beltrami: CsMat<f64>,    // n_vertices x n_vertices (sparse)
    pub mass: Vec<DVector<f64>>,          // mass[k] = star[k] diagonal, k = 0..K-1
    pub ext: ExteriorDerivative,          // d[0..K-2]
    pub hodge: HodgeStar,                 // star[0..K-1]
    _phantom: PhantomData<M>,
}
```

The scalar Laplace-Beltrami is always `star_0_inv * d0^T * diag(star_1) * d0` regardless of K, since it acts on 0-forms (vertex data). This is the same formula for triangles, tets, or any K-simplex mesh.

```rust
impl<M: Manifold, const K: usize, const B: usize> Operators<M, K, B> {
    pub fn from_mesh_generic(mesh: &Mesh<M, K, B>, manifold: &M) -> Result<Self, DecError>;
    pub fn apply_laplace_beltrami(&self, f: &DVector<f64>) -> DVector<f64>;
}
```

The Bochner and Lichnerowicz Laplacians remain specialized to K=3 (they operate on 2D vector/tensor fields and assume a 2-manifold tangent bundle):

```rust
impl<M: Manifold> Operators<M, 3, 2> {
    pub fn apply_bochner_laplacian(&self, u: &DVector<f64>, ricci: Option<...>) -> DVector<f64>;
    pub fn apply_lichnerowicz_laplacian(&self, q: &DVector<f64>, curv: Option<...>) -> DVector<f64>;
}
```

This is correct: the Bochner/Lichnerowicz operators depend on the tangent bundle dimension (which is the mesh dimension n = K-1), so their implementations are necessarily dimension-specific. The scalar Laplacian is universal.

Backward compat: `Operators<M>` (default K=3, B=2) works as before. The `from_mesh` method on `Operators<Euclidean<2>>` delegates to the flat fast path.

### A5. K-Generic Advection and Divergence

Advection and divergence generalize to any K, but the velocity representation changes with dimension:

```rust
/// Scalar advection: transport a 0-form by a velocity field.
/// Velocity is stored as one tangent vector per vertex.
pub fn apply_scalar_advection<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>, manifold: &M,
    f: &DVector<f64>,       // 0-form (scalar per vertex)
    u: &[M::Tangent],       // velocity per vertex
) -> DVector<f64>
```

The upwind scheme is dimension-agnostic: for each vertex v, iterate over incident (K-1)-simplices (edges for K=3, faces for K=4) via `vertex_boundaries`, project velocity onto the simplex direction via `manifold.inner`, and apply upwind flux. The adjacency maps make this O(V * avg_degree) regardless of K.

Divergence similarly generalizes:

```rust
pub fn apply_divergence<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>, manifold: &M,
    ext: &ExteriorDerivative, hodge: &HodgeStar,
    u: &[M::Tangent],
) -> DVector<f64>
```

The DEC divergence formula `star_0_inv * d0^T * star_1 * u_1form` is independent of K (it only uses the 0-form and 1-form operators). The velocity-to-1-form conversion (trapezoidal integration along edges) is also K-agnostic.

**Tensor divergence** remains K=3 specific (it assumes a 2D symmetric tensor layout `[T_xx, T_xy, T_yy]`). A K=4 version would need a 3D tensor layout, which is a future extension.

---

## Phase B: cartan-remesh (New Crate, K=3 Specialization)

### Crate Setup

```toml
[package]
name = "cartan-remesh"
version = "0.1.0"

[dependencies]
cartan-core = { path = "../cartan-core" }
cartan-manifolds = { path = "../cartan-manifolds" }
cartan-dec = { path = "../cartan-dec" }
thiserror = "2"
```

### RemeshLog

All operations return a log for downstream field interpolation:

```rust
pub struct RemeshLog {
    pub splits: Vec<EdgeSplit>,
    pub collapses: Vec<EdgeCollapse>,
    pub flips: Vec<EdgeFlip>,
    pub shifts: Vec<VertexShift>,
}

pub struct EdgeSplit {
    pub old_edge: usize,
    pub v_a: usize,
    pub v_b: usize,
    pub new_vertex: usize,
    pub new_edges: Vec<usize>,
}

pub struct EdgeCollapse {
    pub old_edge: usize,
    pub surviving_vertex: usize,
    pub removed_vertex: usize,
    pub removed_faces: Vec<usize>,
}

pub struct EdgeFlip {
    pub old_edge: usize,
    pub new_edge: [usize; 2],
    pub affected_faces: [usize; 2],
}

pub struct VertexShift {
    pub vertex: usize,
    pub old_pos_tangent: Vec<f64>,  // displacement in tangent space (for undo)
}
```

### Primitive Operations

All generic over `M: Manifold`, operate on `&mut Mesh<M, 3, 2>`:

- **`split_edge`**: Insert vertex at geodesic midpoint (`manifold.exp` from one endpoint along half the `manifold.log` to the other). Replace 2 adjacent triangles with 4. Rebuild local adjacency.

- **`collapse_edge`**: Move surviving vertex to geodesic midpoint. Remove 2 degenerate triangles. Foldover guard: for each face adjacent to the removed vertex, compute face normal before/after in the tangent space at the centroid; reject if angle exceeds `foldover_threshold`. Uses `manifold.log` for tangent-space computation.

- **`flip_edge`**: Replace the diagonal of the quad formed by two adjacent faces. Delaunay criterion: sum of opposite corner angles > pi. Corner angles via `manifold.inner` on edge tangent vectors from `manifold.log`.

- **`shift_vertex`**: Tangential Laplacian smoothing. Log-map all neighbors to the tangent space at v, compute barycenter, project out normal component, apply via `manifold.exp`. Boundary vertices: constrain to boundary edge tangent direction.

- **`smooth_biharmonic`**: Local biharmonic smoothing at a single vertex. Requires `Operators<M>` for the DEC Laplacian. Step direction: `-dual_area * Lap(H) * normal`. Adaptive step size with backtracking (halve if force norm increases).

### LCR Conformal Regularization

```rust
pub fn length_cross_ratio<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M, edge: usize,
) -> f64

pub fn capture_reference_lcrs<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M,
) -> Vec<f64>

pub fn lcr_spring_energy<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M, ref_lcrs: &[f64], kst: f64,
) -> f64

pub fn lcr_spring_gradient<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M, ref_lcrs: &[f64], kst: f64,
) -> Vec<M::Tangent>
```

LCR for an interior edge with diamond vertices {i, j, k, l}: `lcr = dist(i,l) * dist(j,k) / (dist(k,i) * dist(l,j))`. Boundary edges return 1.0. Spring energy: `0.5 * kst * sum((lcr - lcr_ref)^2 / lcr_ref^2)`.

### Adaptive Remeshing Driver

```rust
pub struct RemeshConfig {
    pub curvature_scale: f64,        // C in h < C / sqrt(k_max)
    pub min_edge_length: f64,
    pub max_edge_length: f64,
    pub min_face_area: f64,
    pub max_face_area: f64,
    pub foldover_threshold: f64,     // radians, default 0.5
    pub lcr_spring_stiffness: f64,
    pub smoothing_iterations: usize,
}

pub fn adaptive_remesh<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    operators: &Operators<M>,
    mean_curvatures: &[f64],
    gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> RemeshLog
```

Pipeline order: (1) flip non-Delaunay, (2) split edges violating curvature-CFL `h_e > config.curvature_scale / sqrt(k_max)` where `k_max = H + sqrt(H^2 - K)`, (3) collapse short/flat edges with foldover guard, (4) shift vertices, (5) biharmonic smooth at affected vertices. Rebuilds adjacency after each topology-changing pass.

### Remesh Predicate

```rust
pub fn needs_remesh<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M,
    mean_curvatures: &[f64], gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> bool
```

Returns true if any edge violates the curvature resolution criterion. The caller (volterra-dec) owns the kinetic-energy-minimum timing logic.

---

## Phase C: volterra-dec (DEC Domain, Helfrich, Beris-Edwards; K=3 Specialization)

### DecDomain

```rust
pub struct DecDomain<M: Manifold> {
    pub mesh: Mesh<M, 3, 2>,
    pub manifold: M,
    pub operators: Operators<M>,
    pub ref_lcrs: Vec<f64>,
    pub mean_curvatures: Vec<f64>,
    pub gaussian_curvatures: Vec<f64>,
    pub vertex_normals: Vec<M::Tangent>,
    pub face_areas: Vec<f64>,
    pub dual_areas: Vec<f64>,
}

impl<M: Manifold> DecDomain<M> {
    pub fn new(mesh: Mesh<M, 3, 2>, manifold: M) -> Result<Self, DecError>;
    pub fn reassemble(&mut self) -> Result<(), DecError>;
    pub fn n_vertices(&self) -> usize;
    pub fn n_edges(&self) -> usize;
    pub fn n_faces(&self) -> usize;
}
```

### Discrete Curvature (curvature.rs)

```rust
pub fn mean_curvature_cotan<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M, operators: &Operators<M>,
    normals: &[M::Tangent],
) -> Vec<f64>

pub fn gaussian_curvature_angle_deficit<M: Manifold>(
    mesh: &Mesh<M, 3, 2>, manifold: &M,
) -> Vec<f64>

pub fn principal_curvatures(h: &[f64], k: &[f64]) -> Vec<(f64, f64)>
```

Mean curvature: `H_v = inner(Lap(x)_v, n_v) / (2 * dual_area_v)` where `Lap(x)` is the cotan Laplacian applied to the position vector. Gaussian curvature: `K_v = (2*pi - sum(corner_angles)) / dual_area_v`.

Vertex normals: area-weighted average of incident face normals. Each face normal is computed in the tangent space at the face centroid via `manifold.log` of the three vertices, then transported to the vertex via `manifold.transport` (or projected if M does not implement ParallelTransport). Normalized to unit length.

### Helfrich Energy (helfrich.rs)

```rust
pub struct HelfrichParams {
    pub kb: f64,
    pub kg: f64,
    pub h0: Vec<f64>,   // per-vertex spontaneous curvature
}

pub struct HelfrichEnergy {
    pub total: f64,
    pub bending: f64,
    pub gaussian: f64,
}

pub fn helfrich_energy<M: Manifold>(
    domain: &DecDomain<M>, params: &HelfrichParams,
) -> HelfrichEnergy

pub fn helfrich_forces<M: Manifold>(
    domain: &DecDomain<M>, params: &HelfrichParams,
) -> Vec<M::Tangent>
```

Bending energy: `E_b = 2 * Kb * sum_v(dual_area_v * (H_v - H0_v)^2)`.
Gaussian energy: `E_g = Kg * sum_v(K_v * dual_area_v)` (topological by Gauss-Bonnet, but force contribution is nonzero when Kg varies spatially or the mesh deforms).

Forces via halfedge accumulation (Mem3DG pattern). For each vertex v, loop over outgoing edges to neighbor j, accumulate three components:

1. **Area gradient** (mean curvature vector per halfedge): `cross(face_normal, next_edge_vec) / 4`, weighted by `2*Kb*(H0^2 - H^2)` with 1/3 + 2/3 interpolation between v and j.
2. **Gauss vector**: `0.5 * dihedral_angle * edge_unit_vec`, weighted by `-2*Kb*(H - H0)`.
3. **Schlafli vector**: `edge_length * dihedral_angle_variation`, weighted by `-2*Kb*(H - H0)` for the Laplacian-of-curvature term.
4. **Gaussian curvature force**: corner angle variation vectors weighted by `-Kg`.

All geometric quantities (edge vectors, face normals, dihedral angles, corner angles) computed via `manifold.log` and `manifold.inner`.

### Nematic-Membrane Coupling (coupling.rs)

```rust
pub fn spontaneous_curvature_from_q<M: Manifold>(
    q_field: &[M::Tangent],
    coupling: f64,
) -> Vec<f64>
```

`H0(v) = coupling * S(v)` where S is the scalar order parameter extracted from the Q-tensor eigenvalues. Ordered nematic regions induce spontaneous curvature, driving membrane deformation.

### DEC Scalar Transport (transport.rs)

```rust
pub fn dec_flux_form<M: Manifold>(
    operators: &Operators<M>,
    phi: &DVector<f64>,
    chemical_potential: &DVector<f64>,
    mobility: f64,
) -> DVector<f64>
```

Conserved transport: `dphi/dt = mobility * star0_inv * d0^T * star1 * diag(phi_edge) * d0 * star0_inv * mu`. Edge-averaged phi: `phi_edge[e] = 0.5 * (phi[v_a] + phi[v_b])`.

### Beris-Edwards RHS on DEC (beris_edwards.rs)

```rust
pub fn beris_edwards_rhs_dec<M: Manifold>(
    domain: &DecDomain<M>,
    q: &[M::Tangent],
    velocity: Option<&[M::Tangent]>,
    params: &LandauDeGennesParams,
) -> Vec<M::Tangent>
```

`dQ/dt = -u.nabla(Q) + S(W,Q) + Gamma * H` where:
- Elastic: `K * Lich_Lap(Q)` via `operators.apply_lichnerowicz_laplacian`. The Lichnerowicz curvature correction callback is constructed from `M::riemann_curvature` when `M: Curvature` (e.g., for membranes embedded in curved ambient spaces). For `Euclidean<3>`, the callback returns zero and the Lichnerowicz reduces to the Bochner Laplacian.
- Advection: `apply_vector_advection` from cartan-dec (adjacency-accelerated upwind)
- Co-rotation: `S = xi*(D*Q + Q*D) - 2*xi*Tr(Q*D)*Q + Omega*Q - Q*Omega`, reused from volterra-solver as a free function
- Molecular field bulk terms: `-a_eff*Q - 2c*Tr(Q^2)*Q`, also reused

---

## Phase D: Variational Integrator + Forced Dissipation (volterra-dec)

### Discrete Lagrangian (variational.rs)

Configuration space: product of vertex positions on M and Q-tensor values on QTensor3.

Discrete kinetic energy: `T_d = 0.5 * sum_v(mass_v * |log_{x_n}(x_{n+1})|^2 / dt^2)`.
Discrete potential: `V_d = 0.5 * (V(x_n, Q_n) + V(x_{n+1}, Q_{n+1}))` (trapezoidal).
Total potential: `V = E_helfrich + E_ldg + E_lcr_spring`.

### BAOAB-Shardlow Integration

```rust
pub struct VariationalIntegrator<M: Manifold> {
    pub dt: f64,
    pub newton_max_iter: usize,   // 0 = explicit B steps, 1+ = semi-implicit
    pub newton_tol: f64,
    pub remesh_config: RemeshConfig,
    pub ke_tolerance: f64,        // for symplectic-aware remesh trigger
}
```

Per-step update (D' scheme):

```
B:  p_{n+1/4}  = p_n - (dt/4) * grad_x V(x_n, Q_n)
A:  x_{n+1/2}  = exp_{x_n}(p_{n+1/4} * dt / (2*mass))
O:  shardlow_sweep(edges, momenta, gamma, kBT, dt)     [per-edge DPD]
    Q_{n+1/2}  = Q_n + (dt/2) * BE_RHS(x_{n+1/2}, Q_n)  [half-step Q]
A:  x_{n+1}    = exp_{x_{n+1/2}}(p * dt / (2*mass))
B:  p_{n+1}    = p - (dt/4) * grad_x V(x_{n+1}, Q_{n+1/2})
    Q_{n+1}    = Q_{n+1/2} + (dt/2) * BE_RHS(x_{n+1}, Q_{n+1/2})  [half-step Q]
    reassemble DEC operators on x_{n+1}
```

B steps: explicit by default (newton_max_iter = 0). Forces = `helfrich_forces + lcr_spring_gradient + nematic_stress_force`.

A steps: on-manifold position update via exponential map.

O step: Shardlow sweep from pathwise-geo (see Phase E).

Q half-steps: explicit forward Euler on the DEC Beris-Edwards RHS. Strang-symmetric within the BAOAB frame (two half-steps).

Operator reassembly: once per step after final A. The trailing Q half-step uses operators assembled at `x_{n+1}`, so the only lag is within the A step itself (O(dt^2) error, consistent with integrator order).

### Adaptive Time Stepping

```
dt = min(
    dt_max,
    C_diff * h_min^2,                  // diffusive CFL
    C_force * h_min / max_force_norm,  // force CFL
)
```

### Remeshing Integration

After operator reassembly each step:

```rust
if self.kinetic_energy() < self.ke_running_min * (1.0 + self.ke_tolerance)
    && needs_remesh(&domain.mesh, &domain.manifold,
                    &domain.mean_curvatures, &domain.gaussian_curvatures,
                    &self.remesh_config)
{
    let log = adaptive_remesh(&mut domain.mesh, &domain.manifold,
                              &domain.operators,
                              &domain.mean_curvatures,
                              &domain.gaussian_curvatures,
                              &self.remesh_config);
    interpolate_fields_after_remesh(&log, &domain, &mut q_field, &mut momenta);
    domain.reassemble()?;
    self.ref_lcrs = capture_reference_lcrs(&domain.mesh, &domain.manifold);
}
self.ke_running_min = self.ke_running_min.min(self.kinetic_energy());
```

### Field Interpolation After Remesh (interpolation.rs)

```rust
pub fn interpolate_split<M: Manifold + ParallelTransport>(
    manifold: &M, mesh: &Mesh<M, 3, 2>,
    split: &EdgeSplit, field: &mut Vec<M::Tangent>,
)
```

Edge split: parallel-transport Q from each endpoint to the midpoint, average in the tangent space, apply via exp. Falls back to linear interpolation + projection if transport fails (cut locus).

```rust
pub fn interpolate_collapse<M: Manifold + ParallelTransport>(
    manifold: &M, mesh: &Mesh<M, 3, 2>,
    collapse: &EdgeCollapse, field: &mut Vec<M::Tangent>,
)
```

Edge collapse: Frechet mean of 1-ring neighbor Q values at the surviving vertex, using cartan-optim's `frechet_mean`.

---

## Phase E: Stochastic Primitives in pathwise-geo

### New Module: src/mesh/

Three submodules under `pathwise-geo/src/mesh/`:

#### noise.rs: Edge Noise Sampler

```rust
pub struct EdgeNoiseSampler {
    pub gamma: f64,
    pub kb_t: f64,
}

impl EdgeNoiseSampler {
    pub fn sigma(&self, dt: f64) -> f64 {
        (2.0 * self.gamma * self.kb_t / dt).sqrt()
    }

    pub fn sample<R: Rng>(&self, rng: &mut R, dt: f64) -> f64 {
        self.sigma(dt) * rng.sample::<f64, StandardNormal>(StandardNormal)
    }

    pub fn verify_fd(&self, dt: f64) -> f64 {
        let sigma = self.sigma(dt);
        (sigma * sigma * dt - 2.0 * self.gamma * self.kb_t).abs()
    }
}
```

Physics-agnostic: produces scalar noise increments satisfying the fluctuation-dissipation relation `sigma^2 * dt = 2 * gamma * kBT`. Does not know about meshes.

#### shardlow.rs: Per-Edge O-Step

```rust
pub fn edge_o_step<M: Manifold + ParallelTransport>(
    manifold: &M,
    x_i: &M::Point, x_j: &M::Point,
    p_i: &M::Tangent, p_j: &M::Tangent,
    mass_i: f64, mass_j: f64,
    gamma: f64, dt: f64, noise: f64,
) -> (M::Tangent, M::Tangent)
```

1. Edge unit vector: `r_hat = log_{x_i}(x_j) / |log_{x_i}(x_j)|`
2. Transport p_j to x_i: `p_j_at_i = manifold.transport(x_j, x_i, p_j)`
3. Relative velocity along edge: `dv_r = inner(p_i/mass_i - p_j_at_i/mass_j, r_hat)`
4. Total impulse: `dp = (-gamma * dv_r * dt + noise) * r_hat`
5. Return: `(dp, -manifold.transport(x_i, x_j, dp))` (Newton's third law on manifold)

Falls back to zero correction if log or transport fails (degenerate edge or cut locus).

#### baoab.rs: Sweep Driver

```rust
pub enum EdgeSweepOrder {
    Sequential,
    RandomPermutation,
    Checkerboard,
}

pub struct BAOABConfig {
    pub sampler: EdgeNoiseSampler,
    pub sweep_order: EdgeSweepOrder,
}

pub fn baoab_o_step<M: Manifold + ParallelTransport>(
    manifold: &M,
    mesh: &Mesh<M, 3, 2>,
    positions: &[M::Point],
    momenta: &mut [M::Tangent],
    masses: &[f64],
    config: &BAOABConfig,
    dt: f64,
    rng: &mut impl Rng,
)
```

Sweeps all edges according to `sweep_order`:
- `Sequential`: iterate edges in index order
- `RandomPermutation`: shuffle edge indices each call
- `Checkerboard`: 2-color the edges (greedy coloring from mesh adjacency), sweep each color as a batch. No two edges in the same color share a vertex, so updates within a color are independent (parallelizable with rayon later).

Applies `edge_o_step` to each edge pair, accumulating momentum corrections in-place.

**cartan-dec dependency**: pathwise-geo gains a dependency on cartan-dec (for `Mesh<M, 3, 2>`). This is acceptable since pathwise-geo already depends on cartan-core and cartan-manifolds.

---

## Phase F: Integration Driver + Observables (volterra-dec)

### Simulation Struct

```rust
pub struct MembraneNematicSim<M: Manifold + ParallelTransport> {
    pub domain: DecDomain<M>,
    pub q_field: Vec<M::Tangent>,
    pub momenta: Vec<M::Tangent>,
    pub masses: Vec<f64>,               // rho * dual_area per vertex
    pub integrator: VariationalIntegrator<M>,
    pub helfrich: HelfrichParams,
    pub ldg: LandauDeGennesParams,
    pub baoab: BAOABConfig,
    pub time: f64,
    pub step_count: usize,
    pub ke_running_min: f64,
}

pub struct LandauDeGennesParams {
    pub k_elastic: f64,    // Frank elastic constant
    pub a_eff: f64,        // effective Landau coefficient (a - zeta/2)
    pub c_landau: f64,     // cubic Landau coefficient
    pub gamma_r: f64,      // rotational viscosity
    pub lambda: f64,        // flow alignment parameter
    pub zeta: f64,          // activity
    pub h0_coupling: f64,  // H0 = h0_coupling * S(Q)
}

impl<M: Manifold + ParallelTransport + Curvature> MembraneNematicSim<M> {
    pub fn new(
        mesh: Mesh<M, 3, 2>, manifold: M,
        helfrich: HelfrichParams, ldg: LandauDeGennesParams,
        integrator: VariationalIntegrator<M>, baoab: BAOABConfig,
        initial_q: Vec<M::Tangent>,
    ) -> Result<Self, DecError>;

    pub fn step(&mut self, rng: &mut impl Rng) -> Result<StepResult, SimError>;
    pub fn energy(&self) -> EnergyBreakdown;
    pub fn needs_remesh(&self) -> bool;
}
```

The `M: Curvature` bound enables Lichnerowicz correction. For `Euclidean<3>` (flat ambient), the Riemann tensor vanishes and the correction is zero (Bochner = Lichnerowicz). The bound remains required for type-level correctness on curved ambient spaces.

### StepResult and EnergyBreakdown

```rust
pub struct StepResult {
    pub dt_used: f64,
    pub energy: EnergyBreakdown,
    pub max_force_norm: f64,
    pub remeshed: bool,
    pub remesh_log: Option<RemeshLog>,
}

pub struct EnergyBreakdown {
    pub kinetic: f64,
    pub helfrich_bending: f64,
    pub helfrich_gaussian: f64,
    pub landau_de_gennes: f64,
    pub lcr_spring: f64,
    pub total: f64,
    pub energy_drift: f64,  // |E_n - E_0| / |E_0|
}
```

`energy_drift` is the key variational integrator diagnostic. Should oscillate, not grow. Secular growth indicates dt too large, remeshing too aggressive, or a bug in the force computation.

### Observables (observables.rs)

```rust
pub fn defect_charge_density<M: Manifold + ParallelTransport>(
    domain: &DecDomain<M>, q: &[M::Tangent],
) -> Vec<f64>

pub fn nematic_stress_tensor<M: Manifold>(
    domain: &DecDomain<M>, q: &[M::Tangent], zeta: f64,
) -> Vec<[[f64; 3]; 3]>

pub fn euler_characteristic<M: Manifold>(domain: &DecDomain<M>) -> f64

pub fn total_mean_curvature<M: Manifold>(domain: &DecDomain<M>) -> f64
```

Defect charge density delegates to cartan-geo's holonomy-based disclination detection. Euler characteristic via `sum(K * dual_area) / (2*pi)` (Gauss-Bonnet sanity check). Nematic stress: `sigma_active = -zeta * Q`.

### Adaptive dt

```rust
pub fn compute_dt(
    h_min: f64, max_force: f64, dt_max: f64,
    c_diff: f64, c_force: f64,
) -> f64 {
    dt_max.min(c_diff * h_min * h_min).min(c_force * h_min / max_force)
}
```

---

## Testing Strategy

### Phase A Tests (cartan-dec, K-generic)
- Existing 17 tests continue to pass (backward compat)
- Sparse d[k] chain matches dense versions to machine epsilon
- Exactness: d[k+1] * d[k] = 0 for all k, tested on both K=3 and K=4 meshes
- Generic Hodge star on K=3 `Sphere<3>` mesh: icosahedron dual areas match known values
- Generic Hodge star on K=4 tet mesh: unit cube tets, star0 sums to cube volume
- Sparse Laplacian kills constants and is positive semidefinite (K=3 and K=4)
- Adjacency maps: vertex degree sum = (K-1) * n_boundaries (generalized handshaking)
- `boundary_simplices` counts: interior (K-1)-simplices have exactly 2 co-simplices for manifold meshes
- `simplex_volume` on K=4: unit regular tet volume = sqrt(2)/12
- `simplex_circumcenter` on K=4: regular tet circumcenter = centroid
- Generic advection: constant field advection vanishes (K=3 and K=4)

### Phase B Tests (cartan-remesh)
- Edge split preserves total area (to O(h^2) on curved meshes)
- Edge collapse with foldover guard rejects inversions
- Delaunay flip produces valid triangulation
- LCR of equilateral mesh = 1.0
- Adaptive remesh on a sphere with a Gaussian bump refines near the bump
- RemeshLog accurately records all mutations

### Phase C Tests (volterra-dec)
- Mean curvature of a sphere mesh = 1/R (within discretization error)
- Gaussian curvature of a sphere mesh sums to 4*pi (Gauss-Bonnet)
- Helfrich energy of a sphere with H0=0 matches analytic `8*pi*Kb`
- Helfrich forces on a sphere point radially inward (for H0 < 1/R)
- DEC flux form conserves total phi (sum = const)
- Beris-Edwards RHS: isotropic Q is a fixed point when a_eff > 0

### Phase D Tests (variational integrator)
- Energy oscillation (not drift) for a free membrane over 1000 steps
- BAOAB O-step: momentum distribution converges to Maxwell-Boltzmann
- Remesh + interpolation: total Q norm preserved (within interpolation error)
- Symplectic-aware trigger fires less frequently than naive per-step remeshing

### Phase E Tests (pathwise-geo mesh)
- FD relation: `sigma^2 * dt = 2 * gamma * kBT` to machine epsilon
- edge_o_step: Newton's third law `dp_i + PT(dp_j) = 0`
- Checkerboard coloring: no two same-color edges share a vertex
- Equilibrium test: DPD thermostat on a fixed mesh converges to kBT per DOF

---

## Dependency Summary

```
cartan-core  ─────────────────────────────────────────────┐
cartan-manifolds  ────────────────────────────────┐       │
cartan-dec (Phase A)  ──────────────────┐         │       │
cartan-remesh (Phase B)  ──────┐        │         │       │
                               │        │         │       │
pathwise-core  ────────────────│────────│─────────│───┐   │
pathwise-geo (Phase E)  ───────│────────│─────┐   │   │   │
                               │        │     │   │   │   │
volterra-core  ────────────────│────────│─────│───│───│───│
volterra-dec (Phase C/D/F)  ───┴────────┴─────┴───┘   │   │
                               depends on all above    │   │
```

No circular dependencies. volterra-dec is the sole integration point.
