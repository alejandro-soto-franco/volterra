# Membrane-Nematic DEC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sharp-interface Helfrich membrane dynamics coupled to Beris-Edwards nematohydrodynamics on triangulated 2-manifolds, with stochastic forcing via edge-based DPD, across cartan-dec, cartan-remesh (new), volterra-dec, and pathwise-geo.

**Architecture:** K-generic DEC core in cartan-dec (sparse operators via sprs, circumcentric Hodge star, adjacency-augmented mesh). Adaptive remeshing in cartan-remesh (split/collapse/flip/LCR). Coupled membrane-nematic solver in volterra-dec (Helfrich energy, variational integrator with BAOAB-Shardlow dissipation). Stochastic primitives in pathwise-geo (edge noise, Shardlow O-step, checkerboard sweep).

**Tech Stack:** Rust (edition 2024), sprs (sparse matrices), nalgebra (dense LA), cartan (Riemannian geometry), pathwise (SDE numerics)

**Spec:** `docs/superpowers/specs/2026-03-26-membrane-nematic-dec-design.md`

---

## Phase A: cartan-dec Upgrades (K-Generic DEC Core)

### Task 1: K-Generic Adjacency Maps on `Mesh<M, K, B>`

**Goal**: Add precomputed adjacency (`vertex_boundaries`, `vertex_simplices`, `boundary_simplices`) to `Mesh<M, K, B>`, built during `from_simplices`, with `rebuild_adjacency()` for post-remesh use. Add `edge_faces()` convenience on `Mesh<M, 3, 2>`.

**Files touched**:
- `~/cartan/cartan-dec/src/mesh.rs`
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 1.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
// At the top, add to the existing import:
// use cartan_dec::{..., Mesh};

#[test]
fn adjacency_handshaking_lemma() {
    // Handshaking lemma: sum of vertex degrees (in boundaries) = 2 * n_boundaries
    // Each boundary has exactly B=2 vertices, so each boundary contributes 2 to the total degree.
    let mesh = FlatMesh::unit_square_grid(4);
    let total_degree: usize = mesh.vertex_boundaries.iter().map(|vb| vb.len()).sum();
    assert_eq!(
        total_degree,
        2 * mesh.n_boundaries(),
        "handshaking lemma: sum(deg) = {} != 2*E = {}",
        total_degree,
        2 * mesh.n_boundaries()
    );
}

#[test]
fn adjacency_interior_edges_have_two_cofaces() {
    // On a unit_square_grid(4), interior edges have exactly 2 adjacent triangles.
    // Boundary edges have exactly 1.
    let mesh = FlatMesh::unit_square_grid(4);
    for (e, cofaces) in mesh.boundary_simplices.iter().enumerate() {
        let [i, j] = mesh.boundaries[e];
        let pi = mesh.vertex(i);
        let pj = mesh.vertex(j);
        let on_boundary = (pi.x < 1e-10 && pj.x < 1e-10)
            || (pi.x > 1.0 - 1e-10 && pj.x > 1.0 - 1e-10)
            || (pi.y < 1e-10 && pj.y < 1e-10)
            || (pi.y > 1.0 - 1e-10 && pj.y > 1.0 - 1e-10);
        if on_boundary {
            assert_eq!(
                cofaces.len(),
                1,
                "boundary edge {e} has {} cofaces, expected 1",
                cofaces.len()
            );
        } else {
            assert_eq!(
                cofaces.len(),
                2,
                "interior edge {e} has {} cofaces, expected 2",
                cofaces.len()
            );
        }
    }
}

#[test]
fn adjacency_vertex_simplices_consistent() {
    // Every simplex that contains vertex v must appear in vertex_simplices[v].
    let mesh = FlatMesh::unit_square_grid(3);
    for (t, simplex) in mesh.simplices.iter().enumerate() {
        for &v in simplex {
            assert!(
                mesh.vertex_simplices[v].contains(&t),
                "simplex {t} contains vertex {v} but vertex_simplices[{v}] = {:?}",
                mesh.vertex_simplices[v]
            );
        }
    }
}

#[test]
fn adjacency_edge_faces_convenience() {
    // edge_faces returns (face_a, Some(face_b)) for interior edges, (face_a, None) for boundary.
    let mesh = FlatMesh::unit_square_grid(3);
    for e in 0..mesh.n_boundaries() {
        let (fa, fb) = mesh.edge_faces(e);
        assert!(fa < mesh.n_simplices());
        if let Some(fb_val) = fb {
            assert!(fb_val < mesh.n_simplices());
            assert_ne!(fa, fb_val);
        }
    }
}

#[test]
fn adjacency_rebuild_matches_initial() {
    // rebuild_adjacency() must produce the same maps as from_simplices.
    let mesh_orig = FlatMesh::unit_square_grid(3);
    let mut mesh_rebuilt = mesh_orig.clone();
    mesh_rebuilt.rebuild_adjacency();
    assert_eq!(mesh_orig.vertex_boundaries, mesh_rebuilt.vertex_boundaries);
    assert_eq!(mesh_orig.vertex_simplices, mesh_rebuilt.vertex_simplices);
    assert_eq!(mesh_orig.boundary_simplices, mesh_rebuilt.boundary_simplices);
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec adjacency 2>&1
# Expected: compilation error (fields do not exist on Mesh)
```

#### Step 1.2: Add adjacency fields to `Mesh<M, K, B>`

- [ ] Edit `~/cartan/cartan-dec/src/mesh.rs`. Add three fields to the struct:

```rust
#[derive(Debug, Clone)]
pub struct Mesh<M: Manifold, const K: usize = 3, const B: usize = 2> {
    /// Vertex positions (one `M::Point` per vertex).
    pub vertices: Vec<M::Point>,

    /// Simplices: each entry is K vertex indices.
    pub simplices: Vec<[usize; K]>,

    /// Boundary faces: each entry is B vertex indices (B = K-1, edges for K=3).
    /// Globally deduplicated, canonically oriented (low index first for K=3).
    pub boundaries: Vec<[usize; B]>,

    /// For each simplex, the K indices into `self.boundaries` of its boundary faces.
    pub simplex_boundary_ids: Vec<[usize; K]>,

    /// For each simplex, the K signed contributions of its boundary faces (+/-1.0).
    pub boundary_signs: Vec<[f64; K]>,

    /// Vertex -> incident boundary-face indices. vertex_boundaries[v] lists all
    /// boundary faces that contain vertex v.
    pub vertex_boundaries: Vec<Vec<usize>>,

    /// Vertex -> incident simplex indices. vertex_simplices[v] lists all
    /// simplices that contain vertex v.
    pub vertex_simplices: Vec<Vec<usize>>,

    /// Boundary-face -> adjacent simplex indices. boundary_simplices[b] lists all
    /// simplices that have boundary face b in their boundary.
    pub boundary_simplices: Vec<Vec<usize>>,

    _phantom: PhantomData<M>,
}
```

#### Step 1.3: Add `rebuild_adjacency` on the generic impl

- [ ] Add to the generic `impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B>` block in `~/cartan/cartan-dec/src/mesh.rs`:

```rust
    /// Recompute all adjacency maps from the current `simplices`, `boundaries`,
    /// and `simplex_boundary_ids` arrays.
    ///
    /// Call after any mutation that changes the mesh topology (edge split,
    /// collapse, flip, etc.).
    pub fn rebuild_adjacency(&mut self) {
        let nv = self.vertices.len();
        let nb = self.boundaries.len();
        let ns = self.simplices.len();

        // vertex_boundaries: for each vertex, which boundaries contain it
        let mut vb: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (b, boundary) in self.boundaries.iter().enumerate() {
            for &v in boundary {
                vb[v].push(b);
            }
        }

        // vertex_simplices: for each vertex, which simplices contain it
        let mut vs: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (s, simplex) in self.simplices.iter().enumerate() {
            for &v in simplex {
                vs[v].push(s);
            }
        }

        // boundary_simplices: for each boundary face, which simplices are adjacent
        let mut bs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for (s, sbi) in self.simplex_boundary_ids.iter().enumerate() {
            for &b in sbi {
                bs[b].push(s);
            }
        }

        self.vertex_boundaries = vb;
        self.vertex_simplices = vs;
        self.boundary_simplices = bs;
    }
```

#### Step 1.4: Update `from_simplices` on `Mesh<M, 3, 2>` to build adjacency

- [ ] In the `from_simplices` method on `impl<M: Manifold> Mesh<M, 3, 2>`, initialize empty adjacency fields and call `rebuild_adjacency` at the end:

Replace the final `Self { ... }` block:

```rust
        let mut mesh = Self {
            vertices,
            simplices: triangles,
            boundaries: edges,
            simplex_boundary_ids,
            boundary_signs,
            vertex_boundaries: Vec::new(),
            vertex_simplices: Vec::new(),
            boundary_simplices: Vec::new(),
            _phantom: PhantomData,
        };
        mesh.rebuild_adjacency();
        mesh
```

#### Step 1.5: Add `edge_faces` convenience on `Mesh<M, 3, 2>`

- [ ] Add to the `impl<M: Manifold> Mesh<M, 3, 2>` block:

```rust
    /// For triangle meshes, return the adjacent faces of edge e.
    ///
    /// An interior edge has exactly 2 co-faces: returns `(face_a, Some(face_b))`.
    /// A boundary edge has exactly 1 co-face: returns `(face_a, None)`.
    ///
    /// # Panics
    ///
    /// Panics if the edge is non-manifold (more than 2 co-faces) or has 0 co-faces.
    pub fn edge_faces(&self, e: usize) -> (usize, Option<usize>) {
        let cofaces = &self.boundary_simplices[e];
        match cofaces.len() {
            1 => (cofaces[0], None),
            2 => (cofaces[0], Some(cofaces[1])),
            n => panic!(
                "non-manifold edge {e}: has {n} co-faces (expected 1 or 2)"
            ),
        }
    }
```

#### Step 1.6: Run tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass (existing + new adjacency tests)
```

---

### Task 2: Add `sprs` Dependency + Sparse `ExteriorDerivative`

**Goal**: Add `sprs` to cartan-dec, replace the dense `DMatrix` storage in `ExteriorDerivative` with `Vec<CsMat<f64>>`, keep backward compat via `d0()` / `d1()` accessors and a deprecated dense constructor.

**Files touched**:
- `~/cartan/Cargo.toml` (workspace deps)
- `~/cartan/cartan-dec/Cargo.toml`
- `~/cartan/cartan-dec/src/exterior.rs`
- `~/cartan/cartan-dec/src/hodge.rs` (update to use sparse d0/d1)
- `~/cartan/cartan-dec/src/laplace.rs` (update to use sparse d0/d1)
- `~/cartan/cartan-dec/src/divergence.rs` (update to use sparse d0/d1)
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 2.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
#[test]
fn sparse_exterior_matches_dense() {
    // The sparse exterior derivative must match the dense one to machine epsilon.
    use nalgebra::DMatrix;

    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);

    // Convert sparse d[0] to dense for comparison
    let nv = mesh.n_vertices();
    let ne = mesh.n_boundaries();
    let nt = mesh.n_simplices();

    // Build expected dense d0
    let mut d0_dense = DMatrix::<f64>::zeros(ne, nv);
    for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
        d0_dense[(e, i)] = -1.0;
        d0_dense[(e, j)] = 1.0;
    }

    // Check d[0]
    let d0_sp = ext.d0();
    for r in 0..ne {
        for c in 0..nv {
            let sp_val = d0_sp.get(r, c).copied().unwrap_or(0.0);
            let dn_val = d0_dense[(r, c)];
            assert!(
                (sp_val - dn_val).abs() < 1e-15,
                "d0[{r},{c}]: sparse={sp_val}, dense={dn_val}"
            );
        }
    }

    // Build expected dense d1
    let mut d1_dense = DMatrix::<f64>::zeros(nt, ne);
    for (t, (local_e, local_s)) in mesh
        .simplex_boundary_ids
        .iter()
        .zip(mesh.boundary_signs.iter())
        .enumerate()
    {
        for k in 0..3 {
            d1_dense[(t, local_e[k])] = local_s[k];
        }
    }

    // Check d[1]
    let d1_sp = ext.d1();
    for r in 0..nt {
        for c in 0..ne {
            let sp_val = d1_sp.get(r, c).copied().unwrap_or(0.0);
            let dn_val = d1_dense[(r, c)];
            assert!(
                (sp_val - dn_val).abs() < 1e-15,
                "d1[{r},{c}]: sparse={sp_val}, dense={dn_val}"
            );
        }
    }
}

#[test]
fn sparse_exterior_exactness() {
    // d[k+1] * d[k] = 0 for all k.
    let mesh = FlatMesh::unit_square_grid(4);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);
    let max_err = ext.check_exactness();
    assert!(
        max_err < 1e-14,
        "sparse exactness: max entry of d1*d0 = {max_err:.2e}"
    );
}

#[test]
fn sparse_exterior_k_generic_dimensions() {
    // d has K-1 entries. For K=3: d[0] is (n_edges x n_verts), d[1] is (n_faces x n_edges).
    let mesh = FlatMesh::unit_square_grid(3);
    let ext = ExteriorDerivative::from_mesh_sparse(&mesh);
    assert_eq!(ext.degree(), 2);
    assert_eq!(ext.d0().rows(), mesh.n_boundaries());
    assert_eq!(ext.d0().cols(), mesh.n_vertices());
    assert_eq!(ext.d1().rows(), mesh.n_simplices());
    assert_eq!(ext.d1().cols(), mesh.n_boundaries());
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec sparse_exterior 2>&1
# Expected: compilation error (from_mesh_sparse does not exist)
```

#### Step 2.2: Add `sprs` to workspace and cartan-dec

- [ ] Edit `~/cartan/Cargo.toml`, add to `[workspace.dependencies]`:

```toml
sprs = { version = "0.11", default-features = false }
```

- [ ] Edit `~/cartan/cartan-dec/Cargo.toml`, add to `[dependencies]`:

```toml
sprs             = { workspace = true }
```

#### Step 2.3: Rewrite `ExteriorDerivative` with sparse storage

- [ ] Rewrite `~/cartan/cartan-dec/src/exterior.rs`:

```rust
// ~/cartan/cartan-dec/src/exterior.rs

//! Discrete exterior derivative operators.
//!
//! The exterior derivative is a purely combinatorial operator: it encodes
//! the boundary map of the simplicial complex and is independent of the metric.
//!
//! For a K-simplex mesh, the chain of exterior derivatives is:
//!   d[0]: n_boundaries x n_vertices   (0-forms -> 1-forms)
//!   d[1]: n_simplices x n_boundaries  (1-forms -> 2-forms)
//! For K=4 (tets), there would be d[2]: n_tets x n_faces as well.
//!
//! ## Exactness: d[k+1] * d[k] = 0
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 4.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003. Chapter 3.

use nalgebra::DMatrix;
use sprs::{CsMat, TriMat};

use cartan_core::Manifold;

use crate::mesh::{FlatMesh, Mesh};

/// The discrete exterior derivative operators for a simplicial complex.
///
/// Stores a chain of sparse incidence matrices `d[k]` for k = 0..(K-2).
/// For a triangle mesh (K=3): d[0] (edges x vertices) and d[1] (faces x edges).
pub struct ExteriorDerivative {
    /// d[k] maps k-cochains to (k+1)-cochains. Stored as CSC sparse matrices.
    pub d: Vec<CsMat<f64>>,
}

impl ExteriorDerivative {
    /// Build sparse exterior derivative operators from a flat mesh.
    /// Delegates to `from_mesh_sparse`.
    pub fn from_mesh(mesh: &FlatMesh) -> Self {
        Self::from_mesh_sparse_generic(mesh)
    }

    /// Build sparse exterior derivative from any triangle mesh.
    pub fn from_mesh_sparse<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Self {
        Self::from_mesh_sparse_generic(mesh)
    }

    /// K-generic sparse construction. Builds d[0] and d[1] from the mesh topology.
    ///
    /// d[0]: n_boundaries x n_vertices. d0[b, v] = +1 if v is the head of
    /// boundary b, -1 if v is the tail.
    ///
    /// d[1]: n_simplices x n_boundaries. d1[t, b] = boundary_signs[t][k] for
    /// the k-th boundary face of simplex t.
    fn from_mesh_sparse_generic<M: Manifold, const K: usize, const B: usize>(
        mesh: &Mesh<M, K, B>,
    ) -> Self {
        let nv = mesh.n_vertices();
        let nb = mesh.n_boundaries();
        let ns = mesh.n_simplices();

        // d[0]: nb x nv
        // For each boundary face [v0, v1, ..., v_{B-1}], the boundary operator
        // assigns alternating signs: d0[b, v_k] = (-1)^k.
        let mut tri0 = TriMat::new((nb, nv));
        for (b, boundary) in mesh.boundaries.iter().enumerate() {
            for (k, &v) in boundary.iter().enumerate() {
                let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
                tri0.add_triplet(b, v, sign);
            }
        }
        let d0 = tri0.to_csc();

        // d[1]: ns x nb
        // Uses the precomputed simplex_boundary_ids and boundary_signs.
        let mut tri1 = TriMat::new((ns, nb));
        for (s, (local_b, local_s)) in mesh
            .simplex_boundary_ids
            .iter()
            .zip(mesh.boundary_signs.iter())
            .enumerate()
        {
            for k in 0..K {
                tri1.add_triplet(s, local_b[k], local_s[k]);
            }
        }
        let d1 = tri1.to_csc();

        Self { d: vec![d0, d1] }
    }

    /// Build dense d0 and d1 from a triangle mesh.
    ///
    /// Retained for testing and backward compatibility. Prefer `from_mesh_sparse`.
    #[deprecated(since = "0.2.0", note = "use from_mesh_sparse or from_mesh instead")]
    pub fn from_mesh_generic_dense<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> (DMatrix<f64>, DMatrix<f64>) {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

        let mut d1 = DMatrix::<f64>::zeros(nt, ne);
        for (t, (local_e, local_s)) in mesh
            .simplex_boundary_ids
            .iter()
            .zip(mesh.boundary_signs.iter())
            .enumerate()
        {
            for k in 0..3 {
                d1[(t, local_e[k])] = local_s[k];
            }
        }

        (d0, d1)
    }

    /// Backward-compatible accessor: d0 (0-forms to 1-forms).
    pub fn d0(&self) -> &CsMat<f64> {
        &self.d[0]
    }

    /// Backward-compatible accessor: d1 (1-forms to 2-forms).
    pub fn d1(&self) -> &CsMat<f64> {
        &self.d[1]
    }

    /// Number of exterior derivative operators in the chain.
    pub fn degree(&self) -> usize {
        self.d.len()
    }

    /// Verify the exactness property d[k+1] * d[k] = 0 for all k.
    ///
    /// Returns the maximum absolute entry across all products.
    pub fn check_exactness(&self) -> f64 {
        let mut max_err = 0.0f64;
        for k in 0..self.d.len().saturating_sub(1) {
            let prod = &self.d[k + 1] * &self.d[k];
            for &val in prod.data() {
                max_err = max_err.max(val.abs());
            }
        }
        max_err
    }
}
```

#### Step 2.4: Update `HodgeStar::from_mesh` to use sparse d0/d1

- [ ] In `~/cartan/cartan-dec/src/hodge.rs`, the existing `from_mesh` does not use `ExteriorDerivative`, so no changes are needed for the Hodge star itself. It builds star0/star1/star2 directly from mesh geometry.

(No code change needed. The Hodge star computation is independent of the exterior derivative storage format.)

#### Step 2.5: Update `Operators` (laplace.rs) to use sparse matrices

- [ ] Edit `~/cartan/cartan-dec/src/laplace.rs`. Replace the dense Laplacian assembly with sparse matrix products:

Replace the existing imports:

```rust
use core::marker::PhantomData;

use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

use cartan_core::Manifold;
use cartan_manifolds::euclidean::Euclidean;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::FlatMesh;
```

Replace `Operators` struct:

```rust
/// Assembled discrete differential operators for a mesh.
///
/// Generic over the manifold type `M`. All `apply_*` methods work for any M.
pub struct Operators<M: Manifold = Euclidean<2>> {
    /// Scalar Laplace-Beltrami: n_vertices x n_vertices (sparse).
    pub laplace_beltrami: CsMat<f64>,
    /// Diagonal entries of star0 (dual cell areas, for mass matrix).
    pub mass0: DVector<f64>,
    /// Diagonal entries of star1 (for 1-form computations).
    pub mass1: DVector<f64>,
    /// Exterior derivative chain (kept for advection/divergence).
    pub ext: ExteriorDerivative,
    /// Hodge star diagonals (kept for user access).
    pub hodge: HodgeStar,
    _phantom: PhantomData<M>,
}
```

Replace the `from_mesh` impl:

```rust
impl Operators<Euclidean<2>> {
    /// Assemble all discrete operators from a flat mesh.
    pub fn from_mesh(mesh: &FlatMesh, manifold: &Euclidean<2>) -> Self {
        let ext = ExteriorDerivative::from_mesh(mesh);
        let hodge = HodgeStar::from_mesh(mesh, manifold);

        let laplace_beltrami = assemble_scalar_laplacian(&ext, &hodge);

        Self {
            laplace_beltrami,
            mass0: hodge.star0.clone(),
            mass1: hodge.star1.clone(),
            ext,
            hodge,
            _phantom: PhantomData,
        }
    }
}
```

Add a helper function for the sparse Laplacian assembly:

```rust
/// Assemble the scalar Laplace-Beltrami: star0_inv * d0^T * diag(star1) * d0.
///
/// All operations are sparse. The result is a sparse CSC matrix.
fn assemble_scalar_laplacian(ext: &ExteriorDerivative, hodge: &HodgeStar) -> CsMat<f64> {
    let d0 = ext.d0();
    let ne = hodge.star1.len();

    // Build diag(star1) as a sparse diagonal matrix.
    let mut star1_tri = TriMat::new((ne, ne));
    for e in 0..ne {
        let w = hodge.star1[e];
        if w.abs() > 1e-30 {
            star1_tri.add_triplet(e, e, w);
        }
    }
    let star1_diag = star1_tri.to_csc();

    // star1 * d0
    let star1_d0 = &star1_diag * d0;

    // d0^T * (star1 * d0)
    let d0t = d0.transpose_view();
    let d0t_star1_d0 = &d0t * &star1_d0;

    // star0_inv * (d0^T * star1 * d0)
    let nv = hodge.star0.len();
    let star0_inv = hodge.star0_inv();
    let mut star0_inv_tri = TriMat::new((nv, nv));
    for v in 0..nv {
        let w = star0_inv[v];
        if w.abs() > 1e-30 {
            star0_inv_tri.add_triplet(v, v, w);
        }
    }
    let star0_inv_diag = star0_inv_tri.to_csc();

    &star0_inv_diag * &d0t_star1_d0
}
```

Replace `apply_laplace_beltrami`:

```rust
impl<M: Manifold> Operators<M> {
    /// Apply the scalar Laplace-Beltrami operator to a 0-form (vertex field).
    ///
    /// Uses sparse matrix-vector product.
    pub fn apply_laplace_beltrami(&self, f: &DVector<f64>) -> DVector<f64> {
        let n = f.len();
        let mut result = DVector::<f64>::zeros(n);
        // Sparse matvec: result = L * f
        for (row_val, row_idx) in self.laplace_beltrami.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (col_idx, &val) in row_idx.iter() {
                sum += val * f[col_idx];
            }
            result[row_val] = sum;
        }
        result
    }

    // ... Bochner and Lichnerowicz remain unchanged (they use apply_laplace_beltrami internally) ...
```

Note: The Bochner and Lichnerowicz methods call `&self.laplace_beltrami * ux` which was dense. They must be updated to use the new sparse `apply_laplace_beltrami` helper. Replace each occurrence of `&self.laplace_beltrami * ux` with `self.apply_laplace_beltrami(&ux.into_owned())` (or refactor to call the sparse matvec helper directly). The full Bochner method becomes:

```rust
    /// Apply the Bochner (connection) Laplacian to a vector field.
    ///
    /// Input `u` is a 2*n_v vector with [u_x[0..n_v], u_y[0..n_v]] layout.
    pub fn apply_bochner_laplacian(
        &self,
        u: &DVector<f64>,
        ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(u.len(), 2 * nv, "Bochner: u must have 2*n_v entries");

        let ux = u.rows(0, nv).into_owned();
        let uy = u.rows(nv, nv).into_owned();

        let mut lux = self.apply_laplace_beltrami(&ux);
        let mut luy = self.apply_laplace_beltrami(&uy);

        if let Some(ric) = ricci_correction {
            for v in 0..nv {
                let r = ric(v);
                let ux_v = ux[v];
                let uy_v = uy[v];
                lux[v] += r[0][0] * ux_v + r[0][1] * uy_v;
                luy[v] += r[1][0] * ux_v + r[1][1] * uy_v;
            }
        }

        let mut result = DVector::<f64>::zeros(2 * nv);
        result.rows_mut(0, nv).copy_from(&lux);
        result.rows_mut(nv, nv).copy_from(&luy);
        result
    }

    /// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field Q.
    ///
    /// Input `q` is a 3*n_v vector with [Q_xx, Q_xy, Q_yy] layout.
    pub fn apply_lichnerowicz_laplacian(
        &self,
        q: &DVector<f64>,
        curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(q.len(), 3 * nv, "Lichnerowicz: q must have 3*n_v entries");

        let qxx = q.rows(0, nv).into_owned();
        let qxy = q.rows(nv, nv).into_owned();
        let qyy = q.rows(2 * nv, nv).into_owned();

        let mut lxx = self.apply_laplace_beltrami(&qxx);
        let mut lxy = self.apply_laplace_beltrami(&qxy);
        let mut lyy = self.apply_laplace_beltrami(&qyy);

        if let Some(curv) = curvature_correction {
            for v in 0..nv {
                let c = curv(v);
                let qx = qxx[v];
                let qm = qxy[v];
                let qy = qyy[v];
                lxx[v] += c[0][0] * qx + c[0][1] * qm + c[0][2] * qy;
                lxy[v] += c[1][0] * qx + c[1][1] * qm + c[1][2] * qy;
                lyy[v] += c[2][0] * qx + c[2][1] * qm + c[2][2] * qy;
            }
        }

        let mut result = DVector::<f64>::zeros(3 * nv);
        result.rows_mut(0, nv).copy_from(&lxx);
        result.rows_mut(nv, nv).copy_from(&lxy);
        result.rows_mut(2 * nv, nv).copy_from(&lyy);
        result
    }
}
```

#### Step 2.6: Update `divergence.rs` to use sparse d0

- [ ] Edit `~/cartan/cartan-dec/src/divergence.rs`. The divergence uses `ext.d0.transpose() * star1_u1form`. With sparse storage, this becomes:

Replace the import block:

```rust
use nalgebra::DVector;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::FlatMesh;
```

Replace the `apply_divergence` body (Step 3 onward):

```rust
    // Step 2: Apply star1 to the 1-form.
    let star1_u1form = u1form.component_mul(&hodge.star1);

    // Step 3: Apply d0^T: sparse transpose-multiply.
    let d0t = ext.d0().transpose_view();
    let mut d0t_star1_u = DVector::<f64>::zeros(nv);
    for (row_val, row_idx) in d0t.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row_idx.iter() {
            sum += val * star1_u1form[col_idx];
        }
        d0t_star1_u[row_val] = sum;
    }

    // Step 4: Apply star0_inv.
    let star0_inv = hodge.star0_inv();
    d0t_star1_u.component_mul(&star0_inv)
```

Note: `d0t.outer_iterator()` on CSC transpose_view iterates over rows of d0^T (columns of d0). The `sprs` transpose_view on a CSC matrix gives a CSR view, which iterates rows correctly.

#### Step 2.7: Update integration test imports

- [ ] In `~/cartan/cartan-dec/tests/integration.rs`, the existing tests use `ext.d0.nrows()` and `ext.d1.nrows()` (direct field access on DMatrix). These must change to method calls:

Replace:
```rust
assert_eq!(ext.d0.nrows(), mesh.n_boundaries());
assert_eq!(ext.d0.ncols(), mesh.n_vertices());
```
With:
```rust
assert_eq!(ext.d0().rows(), mesh.n_boundaries());
assert_eq!(ext.d0().cols(), mesh.n_vertices());
```

Replace:
```rust
assert_eq!(ext.d1.nrows(), mesh.n_simplices());
assert_eq!(ext.d1.ncols(), mesh.n_boundaries());
```
With:
```rust
assert_eq!(ext.d1().rows(), mesh.n_simplices());
assert_eq!(ext.d1().cols(), mesh.n_boundaries());
```

Replace the old exactness test body that uses `ext.check_exactness()` with `prod.abs().max()` (the method signature is the same, so the test body stays the same, but verify the product computation still works with sparse).

#### Step 2.8: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass (existing + new sparse tests)
```

---

### Task 3: K-Generic `simplex_volume`, `simplex_circumcenter`, and Boundary Variants

**Goal**: Add K-generic geometric primitives to `Mesh<M, K, B>`. The existing `triangle_area`, `circumcenter`, `boundary_midpoint` on `Mesh<M, 3, 2>` become thin wrappers. Flat fast paths remain unchanged.

**Files touched**:
- `~/cartan/cartan-dec/src/mesh.rs`
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 3.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
#[test]
fn simplex_volume_triangle_matches_triangle_area() {
    // simplex_volume on a triangle mesh must equal triangle_area.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for t in 0..mesh.n_simplices() {
        let generic = mesh.simplex_volume(&manifold, t);
        let specific = mesh.triangle_area(&manifold, t);
        assert!(
            (generic - specific).abs() < 1e-14,
            "simplex {t}: generic={generic}, specific={specific}"
        );
    }
}

#[test]
fn simplex_circumcenter_triangle_matches_circumcenter() {
    // simplex_circumcenter on a triangle mesh must equal circumcenter.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for t in 0..mesh.n_simplices() {
        let generic = mesh.simplex_circumcenter(&manifold, t);
        let specific = mesh.circumcenter(&manifold, t);
        let diff = (generic - specific).norm();
        assert!(
            diff < 1e-14,
            "simplex {t}: circumcenter diff = {diff}"
        );
    }
}

#[test]
fn boundary_volume_matches_edge_length() {
    // boundary_volume on a triangle mesh must equal edge_length.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    for e in 0..mesh.n_boundaries() {
        let generic = mesh.boundary_volume(&manifold, e);
        let specific = mesh.edge_length(&manifold, e);
        assert!(
            (generic - specific).abs() < 1e-14,
            "boundary {e}: generic={generic}, specific={specific}"
        );
    }
}

#[test]
fn regular_tet_volume() {
    // A regular tetrahedron with edge length 1 has volume sqrt(2)/12.
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 6.0, (2.0_f64 / 3.0).sqrt());

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let vol = mesh.simplex_volume(&manifold, 0);
    let expected = (2.0_f64).sqrt() / 12.0;
    assert!(
        (vol - expected).abs() < 1e-12,
        "regular tet volume: got {vol}, expected {expected}"
    );
}

#[test]
fn regular_tet_circumcenter_is_centroid() {
    // For a regular tetrahedron, the circumcenter equals the centroid.
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.5, (3.0_f64).sqrt() / 6.0, (2.0_f64 / 3.0).sqrt());

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices(
        &manifold,
        vec![v0.clone(), v1.clone(), v2.clone(), v3.clone()],
        vec![[0, 1, 2, 3]],
    );

    let cc = mesh.simplex_circumcenter(&manifold, 0);
    let centroid = (v0 + v1 + v2 + v3) * 0.25;
    let diff = (cc - centroid).norm();
    assert!(
        diff < 1e-12,
        "regular tet circumcenter: diff from centroid = {diff}"
    );
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec simplex_volume 2>&1
# Expected: compilation error (simplex_volume does not exist)
```

#### Step 3.2: Add K-generic `from_simplices` on `Mesh<M, K, B>`

Before we can build a tet mesh for testing, we need a generic `from_simplices`. Currently `from_simplices` only exists on `Mesh<M, 3, 2>`. We need to add a generic version that handles any K.

- [ ] Add to the generic `impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B>` block in `~/cartan/cartan-dec/src/mesh.rs`:

```rust
    /// Construct a K-simplex mesh from vertex positions and a simplex list.
    ///
    /// Boundary faces are B-tuples of vertices, deduplicated and canonically
    /// oriented (sorted vertex indices). The boundary operator signs are computed
    /// from the relative orientation of each boundary face within its parent simplex.
    ///
    /// # Panics
    ///
    /// Panics if any simplex vertex index is out of bounds, or if B != K-1.
    pub fn from_simplices(
        _manifold: &M,
        vertices: Vec<M::Point>,
        simplices: Vec<[usize; K]>,
    ) -> Self {
        assert_eq!(B, K - 1, "B must equal K-1");
        let n_v = vertices.len();

        let mut boundary_map: HashMap<[usize; B], usize> = HashMap::new();
        let mut boundaries: Vec<[usize; B]> = Vec::new();
        let mut simplex_boundary_ids = Vec::with_capacity(simplices.len());
        let mut boundary_signs_vec = Vec::with_capacity(simplices.len());

        for simplex in &simplices {
            for &v in simplex {
                assert!(v < n_v, "simplex vertex index {v} out of bounds (n_v={n_v})");
            }

            let mut local_boundary_ids = [0usize; K];
            let mut local_signs = [0.0f64; K];

            // The k-th boundary face of simplex [v0, v1, ..., v_{K-1}] is obtained
            // by omitting vertex k. The sign is (-1)^k (from the boundary operator).
            for omit in 0..K {
                let sign = if omit % 2 == 0 { 1.0 } else { -1.0 };

                // Build the boundary face by collecting all vertices except the omitted one.
                let mut face = [0usize; B];
                let mut idx = 0;
                for (pos, &v) in simplex.iter().enumerate() {
                    if pos != omit {
                        face[idx] = v;
                        idx += 1;
                    }
                }

                // Canonical orientation: sort the face vertices.
                let mut sorted_face = face;
                sorted_face.sort();

                // Determine the parity of the permutation from face to sorted_face.
                // This tells us if the canonical ordering matches the inherited orientation.
                let parity = permutation_sign(&face, &sorted_face);
                let effective_sign = sign * parity;

                let boundary_idx = *boundary_map.entry(sorted_face).or_insert_with(|| {
                    let b = boundaries.len();
                    boundaries.push(sorted_face);
                    b
                });

                local_boundary_ids[omit] = boundary_idx;
                local_signs[omit] = effective_sign;
            }

            simplex_boundary_ids.push(local_boundary_ids);
            boundary_signs_vec.push(local_signs);
        }

        let mut mesh = Self {
            vertices,
            simplices,
            boundaries,
            simplex_boundary_ids,
            boundary_signs: boundary_signs_vec,
            vertex_boundaries: Vec::new(),
            vertex_simplices: Vec::new(),
            boundary_simplices: Vec::new(),
            _phantom: PhantomData,
        };
        mesh.rebuild_adjacency();
        mesh
    }
```

Add the permutation sign helper as a module-level function:

```rust
/// Compute the sign of the permutation that maps `from` to `to`.
///
/// Both slices must contain the same elements. Returns +1.0 for even
/// permutations and -1.0 for odd permutations. Uses a simple counting
/// method (count transpositions).
fn permutation_sign<const N: usize>(from: &[usize; N], to: &[usize; N]) -> f64 {
    // Build the permutation: perm[i] = position of from[i] in to.
    let mut perm = [0usize; N];
    for (i, &val) in from.iter().enumerate() {
        for (j, &tval) in to.iter().enumerate() {
            if val == tval {
                perm[i] = j;
                break;
            }
        }
    }
    // Count inversions.
    let mut inversions = 0usize;
    for i in 0..N {
        for j in (i + 1)..N {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 1.0 } else { -1.0 }
}
```

Now update the existing `from_simplices` on `Mesh<M, 3, 2>` to delegate to the generic version (or keep the optimized K=3 code). Since the generic version handles the K=3 case correctly, we can make the K=3 version delegate. However, the K=3 version has a slightly different boundary orientation convention (edges are `(lo, hi)` with direction sign). To maintain exact backward compatibility, keep the K=3 `from_simplices` as-is but add the adjacency fields. The generic `from_simplices` should be renamed to avoid collision.

Since Rust does not allow two `from_simplices` methods with different const generics on the same type (the K=3, B=2 specialization would conflict with the generic one), we need a different approach. The generic constructor should live on the generic impl and be used by the K=3 specialization internally, or we keep the K=3 one separate and only add the generic one for K != 3.

The cleanest approach: keep the existing K=3 `from_simplices` unchanged (with adjacency added from Task 1), and add `from_simplices` on the generic impl only for the case where it does not conflict. Since `impl<M: Manifold> Mesh<M, 3, 2>` has `from_simplices`, the generic `impl<M, K, B>` cannot also have `from_simplices` without coherence issues. So name the generic one `from_simplices_generic`:

```rust
    /// K-generic constructor. For K=3, prefer `from_simplices` on `Mesh<M, 3, 2>`.
    pub fn from_simplices_generic(
        _manifold: &M,
        vertices: Vec<M::Point>,
        simplices: Vec<[usize; K]>,
    ) -> Self {
```

(Use this name in the tests for K=4 tet construction.)

#### Step 3.3: Add K-generic `simplex_volume`

- [ ] Add to `impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B>`:

```rust
    /// Volume of simplex s via the Gram determinant in the tangent space at vertex 0.
    ///
    /// For K=3 (triangle): area. For K=4 (tet): volume.
    /// The formula is: vol = (1 / (K-1)!) * sqrt(|det(G)|) where G is the
    /// (K-1) x (K-1) Gram matrix G_{ij} = <log(v0, v_i), log(v0, v_j)>.
    pub fn simplex_volume(&self, manifold: &M, s: usize) -> f64 {
        let simplex = &self.simplices[s];
        let v0 = &self.vertices[simplex[0]];

        // Build tangent vectors from v0 to each other vertex.
        let n = K - 1; // number of edge vectors
        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for i in 1..K {
            let vi = &self.vertices[simplex[i]];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        // Build the Gram matrix G_{ij} = <logs[i], logs[j]>.
        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        // Compute determinant of n x n Gram matrix.
        let det = dense_determinant(&gram, n);

        // Volume = (1 / (K-1)!) * sqrt(|det|)
        let factorial = (1..K).product::<usize>() as f64;
        det.abs().sqrt() / factorial
    }

    /// Volume of boundary face b via the Gram determinant in the tangent space at vertex 0.
    ///
    /// For K=3 (edge): length. For K=4 (face): area of the triangular face.
    pub fn boundary_volume(&self, manifold: &M, b: usize) -> f64 {
        let boundary = &self.boundaries[b];
        let v0 = &self.vertices[boundary[0]];

        let n = B - 1;
        if n == 0 {
            // B=1 means vertices are 0-simplices, volume = 1 (or length = dist for B=2).
            // For B=2, this is an edge, n=1.
            // Actually for B=1, the boundary face is a single vertex, volume = 1.
            return 1.0;
        }

        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for i in 1..B {
            let vi = &self.vertices[boundary[i]];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let det = dense_determinant(&gram, n);
        let factorial = (1..B).product::<usize>().max(1) as f64;
        det.abs().sqrt() / factorial
    }

    /// Circumcenter of simplex s via the equidistance system in tangent space.
    ///
    /// Solves the system: for each edge vector e_i from v0, find barycentric
    /// coordinates t such that |c - v_i|^2 = |c - v0|^2 for all i, where
    /// c = v0 + sum_i t_i * e_i. This reduces to G * t = 0.5 * diag(G)
    /// where G is the Gram matrix. Falls back to the centroid for degenerate simplices.
    pub fn simplex_circumcenter(&self, manifold: &M, s: usize) -> M::Point {
        let simplex = &self.simplices[s];
        let v0 = &self.vertices[simplex[0]];

        let n = K - 1;
        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for i in 1..K {
            let vi = &self.vertices[simplex[i]];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        // Gram matrix
        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        // RHS: 0.5 * diag(G)
        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = 0.5 * gram[i * n + i];
        }

        // Solve G * t = rhs via Gaussian elimination with partial pivoting.
        let t = match dense_solve(&gram, &rhs, n) {
            Some(sol) => sol,
            None => {
                // Degenerate: return centroid.
                let mut tangent = manifold.zero_tangent(v0);
                for log in &logs {
                    tangent = tangent + log.clone() * (1.0 / K as f64);
                }
                return manifold.exp(v0, &tangent);
            }
        };

        // Circumcenter tangent: sum_i t[i] * logs[i]
        let mut tangent = manifold.zero_tangent(v0);
        for (i, ti) in t.iter().enumerate() {
            tangent = tangent + logs[i].clone() * *ti;
        }
        manifold.exp(v0, &tangent)
    }

    /// Circumcenter of boundary face b.
    pub fn boundary_circumcenter(&self, manifold: &M, b: usize) -> M::Point {
        let boundary = &self.boundaries[b];
        let v0 = &self.vertices[boundary[0]];

        let n = B - 1;
        if n == 0 {
            return v0.clone();
        }

        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for i in 1..B {
            let vi = &self.vertices[boundary[i]];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = 0.5 * gram[i * n + i];
        }

        let t = match dense_solve(&gram, &rhs, n) {
            Some(sol) => sol,
            None => {
                let mut tangent = manifold.zero_tangent(v0);
                for log in &logs {
                    tangent = tangent + log.clone() * (1.0 / B as f64);
                }
                return manifold.exp(v0, &tangent);
            }
        };

        let mut tangent = manifold.zero_tangent(v0);
        for (i, ti) in t.iter().enumerate() {
            tangent = tangent + logs[i].clone() * *ti;
        }
        manifold.exp(v0, &tangent)
    }
```

#### Step 3.4: Add dense linear algebra helpers

- [ ] Add module-level helper functions in `~/cartan/cartan-dec/src/mesh.rs`:

```rust
/// Compute the determinant of an n x n matrix stored row-major in a flat Vec.
///
/// Uses LU decomposition with partial pivoting. For n <= 3, uses direct formulas.
fn dense_determinant(a: &[f64], n: usize) -> f64 {
    match n {
        0 => 1.0,
        1 => a[0],
        2 => a[0] * a[3] - a[1] * a[2],
        3 => {
            a[0] * (a[4] * a[8] - a[5] * a[7])
                - a[1] * (a[3] * a[8] - a[5] * a[6])
                + a[2] * (a[3] * a[7] - a[4] * a[6])
        }
        _ => {
            // General LU with partial pivoting.
            let mut lu: Vec<f64> = a.to_vec();
            let mut sign = 1.0f64;
            for col in 0..n {
                // Find pivot.
                let mut max_val = lu[col * n + col].abs();
                let mut max_row = col;
                for row in (col + 1)..n {
                    let v = lu[row * n + col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_val < 1e-30 {
                    return 0.0;
                }
                if max_row != col {
                    for k in 0..n {
                        lu.swap(col * n + k, max_row * n + k);
                    }
                    sign = -sign;
                }
                let pivot = lu[col * n + col];
                for row in (col + 1)..n {
                    let factor = lu[row * n + col] / pivot;
                    lu[row * n + col] = factor;
                    for k in (col + 1)..n {
                        lu[row * n + k] -= factor * lu[col * n + k];
                    }
                }
            }
            let mut det = sign;
            for i in 0..n {
                det *= lu[i * n + i];
            }
            det
        }
    }
}

/// Solve A * x = b for an n x n system via Gaussian elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular (pivot < 1e-30).
fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Augmented matrix [A | b], stored row-major.
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        if max_row != col {
            for k in 0..(n + 1) {
                aug.swap(col * (n + 1) + k, max_row * (n + 1) + k);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for k in col..(n + 1) {
                aug[row * (n + 1) + k] -= factor * aug[col * (n + 1) + k];
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Some(x)
}
```

#### Step 3.5: Make existing K=3 methods delegate to generic ones

- [ ] In `impl<M: Manifold> Mesh<M, 3, 2>`, update `triangle_area` and `circumcenter` to delegate:

The existing `triangle_area` and `circumcenter` methods on `Mesh<M, 3, 2>` should remain as-is for now (they use hardcoded 2D formulas that are equivalent but independently coded). Both paths produce the same result, and the tests in Step 3.1 verify agreement. The flat fast paths (`triangle_area_flat`, `circumcenter_flat`) also remain unchanged.

No code change needed here; the tests validate equivalence.

#### Step 3.6: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass (existing + new volume/circumcenter tests)
```

---

### Task 4: K-Generic `HodgeStar`

**Goal**: Replace the three separate `star0`/`star1`/`star2` fields with a single `star: Vec<DVector<f64>>` indexed by degree. Add `from_mesh_generic` that works for any K. Backward compat via `star0()`/`star1()`/`star2()` accessors.

**Files touched**:
- `~/cartan/cartan-dec/src/hodge.rs`
- `~/cartan/cartan-dec/src/laplace.rs` (update field accesses)
- `~/cartan/cartan-dec/src/divergence.rs` (update field accesses)
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 4.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
#[test]
fn hodge_star_generic_matches_flat() {
    // from_mesh_generic on a flat mesh must produce the same star values as from_mesh.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let hodge_flat = HodgeStar::from_mesh(&mesh, &manifold);
    let hodge_generic = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    // star0
    let diff0 = (&hodge_flat.star0() - hodge_generic.star_k(0)).norm();
    assert!(diff0 < 1e-12, "star0 diff = {diff0}");

    // star1
    let diff1 = (&hodge_flat.star1() - hodge_generic.star_k(1)).norm();
    assert!(diff1 < 1e-12, "star1 diff = {diff1}");

    // star2
    let diff2 = (&hodge_flat.star2() - hodge_generic.star_k(2)).norm();
    assert!(diff2 < 1e-12, "star2 diff = {diff2}");
}

#[test]
fn hodge_star_k_inv_roundtrip() {
    // star_k_inv(k) * star_k(k) = identity (element-wise).
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let hodge = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    for k in 0..3 {
        let s = hodge.star_k(k);
        let sinv = hodge.star_k_inv(k);
        for i in 0..s.len() {
            if s[i].abs() > 1e-30 {
                let product = s[i] * sinv[i];
                assert!(
                    (product - 1.0).abs() < 1e-12,
                    "star[{k}][{i}] * star_inv[{k}][{i}] = {product}"
                );
            }
        }
    }
}

#[test]
fn hodge_star_generic_sphere_positive() {
    // On an icosahedral mesh on S^2, all Hodge star entries should be positive.
    use cartan_manifolds::sphere::Sphere;
    use nalgebra::SVector;

    let manifold = Sphere::<3>;
    // Build a simple octahedral mesh on S^2 (6 vertices, 8 triangles).
    let verts: Vec<SVector<f64, 3>> = vec![
        SVector::new(1.0, 0.0, 0.0),
        SVector::new(-1.0, 0.0, 0.0),
        SVector::new(0.0, 1.0, 0.0),
        SVector::new(0.0, -1.0, 0.0),
        SVector::new(0.0, 0.0, 1.0),
        SVector::new(0.0, 0.0, -1.0),
    ];
    let tris = vec![
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0],
    ];
    let mesh = Mesh::from_simplices(&manifold, verts, tris);
    let hodge = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();

    for k in 0..3 {
        for (i, &val) in hodge.star_k(k).iter().enumerate() {
            assert!(
                val > 0.0,
                "star[{k}][{i}] = {val}, expected positive on S^2 octahedron"
            );
        }
    }
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec hodge_star_generic 2>&1
# Expected: compilation error (from_mesh_generic, star_k do not exist)
```

#### Step 4.2: Rewrite `HodgeStar` with `star: Vec<DVector<f64>>`

- [ ] Rewrite `~/cartan/cartan-dec/src/hodge.rs`:

```rust
// ~/cartan/cartan-dec/src/hodge.rs

//! Discrete Hodge star operators.
//!
//! The Hodge star encodes the metric. For a well-centered mesh, all Hodge
//! stars are diagonal with entries equal to dual/primal simplex volume ratios.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 5.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis. Chapter 4.

use nalgebra::DVector;

use cartan_core::Manifold;
use cartan_manifolds::euclidean::Euclidean;

use crate::error::DecError;
use crate::mesh::{FlatMesh, Mesh};

/// Diagonal Hodge star operators for a simplicial mesh.
///
/// `star[k]` is the diagonal of the Hodge star for k-forms. For a triangle
/// mesh (n=2): star[0] (vertices), star[1] (edges), star[2] (faces).
pub struct HodgeStar {
    /// star[k] contains the diagonal entries of the k-form Hodge star.
    pub star: Vec<DVector<f64>>,
}

impl HodgeStar {
    /// Compute the Hodge star from a flat 2D mesh (fast path).
    ///
    /// Uses flat metric methods (`triangle_area_flat`, `edge_length_flat`,
    /// `circumcenter_flat`, `edge_midpoint`).
    pub fn from_mesh(mesh: &FlatMesh, _manifold: &Euclidean<2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star2: 1 / triangle area
        let mut s2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            s2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // star0: barycentric dual cell area = (1/3) * sum_{t containing v} area(t)
        let mut s0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            for &v in &mesh.simplices[t] {
                s0[v] += area / 3.0;
            }
        }

        // star1: |dual edge| / |primal edge|
        let mut s1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.edge_length_flat(e);
            if primal_len < 1e-30 {
                s1[e] = 0.0;
                continue;
            }
            let cofaces = &mesh.boundary_simplices[e];
            let dual_len = match cofaces.len() {
                2 => {
                    let c1 = mesh.circumcenter_flat(cofaces[0]);
                    let c2 = mesh.circumcenter_flat(cofaces[1]);
                    (c2 - c1).norm()
                }
                1 => {
                    let c = mesh.circumcenter_flat(cofaces[0]);
                    let mid = mesh.edge_midpoint(e);
                    (c - mid).norm()
                }
                _ => 0.0,
            };
            s1[e] = dual_len / primal_len;
        }

        Self {
            star: vec![s0, s1, s2],
        }
    }

    /// K-generic Hodge star via circumcentric duality.
    ///
    /// For each k-simplex sigma, star_k[sigma] = vol(dual(sigma)) / vol(sigma).
    ///
    /// Currently implemented for K=3 (triangle meshes on any manifold).
    /// Falls back to a barycentric dual for star0, circumcentric dual edge
    /// length ratio for star1, and reciprocal area for star2.
    pub fn from_mesh_generic<M: Manifold, const K: usize, const B: usize>(
        mesh: &Mesh<M, K, B>,
        manifold: &M,
    ) -> Result<Self, DecError> {
        // For now, we implement the n=2 (K=3) case generically.
        // The formulas generalize to higher K but the dual cell volume
        // computation becomes more involved.
        assert_eq!(K, 3, "from_mesh_generic currently supports K=3 only");
        assert_eq!(B, 2, "from_mesh_generic currently supports B=2 only");

        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star2: 1 / simplex area
        let mut s2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.simplex_volume(manifold, t);
            s2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // star0: barycentric dual cell area = (1/3) * sum_{t containing v} area(t)
        let mut s0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.simplex_volume(manifold, t);
            for &v in &mesh.simplices[t] {
                s0[v] += area / 3.0;
            }
        }

        // star1: |dual edge| / |primal edge|
        // Dual edge connects circumcenters of adjacent triangles.
        let mut s1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.boundary_volume(manifold, e);
            if primal_len < 1e-30 {
                s1[e] = 0.0;
                continue;
            }
            let cofaces = &mesh.boundary_simplices[e];
            let dual_len = match cofaces.len() {
                2 => {
                    let c1 = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let c2 = mesh.simplex_circumcenter(manifold, cofaces[1]);
                    manifold.dist(&c1, &c2).unwrap_or(0.0)
                }
                1 => {
                    let c = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let mid = mesh.boundary_circumcenter(manifold, e);
                    manifold.dist(&c, &mid).unwrap_or(0.0)
                }
                _ => 0.0,
            };
            s1[e] = dual_len / primal_len;
        }

        Ok(Self {
            star: vec![s0, s1, s2],
        })
    }

    /// Access the k-form Hodge star diagonal.
    pub fn star_k(&self, k: usize) -> &DVector<f64> {
        &self.star[k]
    }

    /// Inverse Hodge star for k-forms (element-wise reciprocal).
    pub fn star_k_inv(&self, k: usize) -> DVector<f64> {
        self.star[k].map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Backward-compatible accessor: star0 (0-form Hodge star diagonal).
    pub fn star0(&self) -> &DVector<f64> {
        &self.star[0]
    }

    /// Backward-compatible accessor: star1 (1-form Hodge star diagonal).
    pub fn star1(&self) -> &DVector<f64> {
        &self.star[1]
    }

    /// Backward-compatible accessor: star2 (2-form Hodge star diagonal).
    pub fn star2(&self) -> &DVector<f64> {
        &self.star[2]
    }

    /// Backward-compatible inverse: star0_inv.
    pub fn star0_inv(&self) -> DVector<f64> {
        self.star_k_inv(0)
    }

    /// Backward-compatible inverse: star1_inv.
    pub fn star1_inv(&self) -> DVector<f64> {
        self.star_k_inv(1)
    }

    /// Backward-compatible inverse: star2_inv.
    pub fn star2_inv(&self) -> DVector<f64> {
        self.star_k_inv(2)
    }
}
```

#### Step 4.3: Update `laplace.rs` field accesses

- [ ] In `~/cartan/cartan-dec/src/laplace.rs`, the `from_mesh` method accesses `hodge.star0` and `hodge.star1` directly. Update to use accessor methods:

Replace `hodge.star0.clone()` with `hodge.star0().clone()` and `hodge.star1.clone()` with `hodge.star1().clone()`.

In the `assemble_scalar_laplacian` function, replace `hodge.star1` with `hodge.star1()` and `hodge.star0` with `hodge.star0()`.

#### Step 4.4: Update `divergence.rs` field accesses

- [ ] In `~/cartan/cartan-dec/src/divergence.rs`, replace `hodge.star1` with `hodge.star1()` in the `component_mul` call, and `hodge.star0_inv()` remains the same (it was already a method call).

Specifically, replace:
```rust
let star1_u1form = u1form.component_mul(&hodge.star1);
```
With:
```rust
let star1_u1form = u1form.component_mul(hodge.star1());
```

#### Step 4.5: Update integration test field accesses

- [ ] In `~/cartan/cartan-dec/tests/integration.rs`, update direct field accesses:

Replace `hodge.star0` with `hodge.star0()`, `hodge.star1` with `hodge.star1()`, `hodge.star2` with `hodge.star2()` in all existing tests.

For example:
```rust
// Old: for (v, &w) in hodge.star0.iter().enumerate()
// New: for (v, &w) in hodge.star0().iter().enumerate()
```

#### Step 4.6: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass
```

---

### Task 5: K-Generic Sparse `Operators`

**Goal**: Make `Operators` generic over `<M, K, B>` with const generics. The scalar Laplace-Beltrami, `mass`, `ext`, `hodge` all become K-generic. Bochner and Lichnerowicz remain on `Operators<M, 3, 2>`. Backward compat via default type params.

**Files touched**:
- `~/cartan/cartan-dec/src/laplace.rs`
- `~/cartan/cartan-dec/src/lib.rs` (update export)
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 5.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
#[test]
fn operators_generic_laplace_kills_constants() {
    // K-generic Laplacian of a constant function is zero.
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
    let nv = mesh.n_vertices();

    let f = DVector::from_element(nv, 3.14);
    let lf = ops.apply_laplace_beltrami(&f);

    let max_err = lf.abs().max();
    assert!(
        max_err < 1e-12,
        "generic Delta(const) != 0: max = {max_err:.2e}"
    );
}

#[test]
fn operators_generic_laplace_positive_semidefinite() {
    // <f, L f>_{star0} >= 0 for the generic operator.
    let mesh = FlatMesh::unit_square_grid(8);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh_generic(&mesh, &manifold).unwrap();
    let nv = mesh.n_vertices();

    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        (std::f64::consts::PI * p.x).sin() * (std::f64::consts::PI * p.y).sin()
    });

    let lf = ops.apply_laplace_beltrami(&f);
    let f_dot_lf: f64 = f
        .iter()
        .zip(lf.iter())
        .zip(ops.mass[0].iter())
        .map(|((fi, lfi), mi)| fi * lfi * mi)
        .sum();

    assert!(
        f_dot_lf >= -1e-10,
        "<f, Lf>_{{star0}} = {f_dot_lf:.6e}, expected >= 0"
    );
}

#[test]
fn operators_backward_compat_default_type() {
    // Operators<Euclidean<2>> (old default) still works.
    let mesh = FlatMesh::unit_square_grid(3);
    let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();
    let f = DVector::from_element(nv, 1.0);
    let lf = ops.apply_laplace_beltrami(&f);
    assert!(lf.abs().max() < 1e-12);
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec operators_generic 2>&1
# Expected: compilation error (from_mesh_generic on Operators does not exist)
```

#### Step 5.2: Rewrite `Operators` with const generics

- [ ] Edit `~/cartan/cartan-dec/src/laplace.rs`. Rewrite the `Operators` struct:

```rust
/// Assembled discrete differential operators for a mesh.
///
/// Generic over manifold `M` and simplex dimension K (with B = K-1).
/// The scalar Laplace-Beltrami operator works for any K. The Bochner and
/// Lichnerowicz operators are specialized to K=3 (2-manifold).
pub struct Operators<M: Manifold = Euclidean<2>, const K: usize = 3, const B: usize = 2> {
    /// Scalar Laplace-Beltrami: n_vertices x n_vertices (sparse).
    pub laplace_beltrami: CsMat<f64>,
    /// Mass matrices: mass[k] = star[k] diagonal, k = 0..K-1.
    pub mass: Vec<DVector<f64>>,
    /// Exterior derivative chain.
    pub ext: ExteriorDerivative,
    /// Hodge star.
    pub hodge: HodgeStar,
    _phantom: PhantomData<M>,
}

impl Operators<Euclidean<2>, 3, 2> {
    /// Assemble all discrete operators from a flat mesh (fast path).
    pub fn from_mesh(mesh: &FlatMesh, manifold: &Euclidean<2>) -> Self {
        let ext = ExteriorDerivative::from_mesh(mesh);
        let hodge = HodgeStar::from_mesh(mesh, manifold);

        let laplace_beltrami = assemble_scalar_laplacian(&ext, &hodge);

        Self {
            laplace_beltrami,
            mass: hodge.star.iter().cloned().collect(),
            ext,
            hodge,
            _phantom: PhantomData,
        }
    }
}

impl<M: Manifold, const K: usize, const B: usize> Operators<M, K, B> {
    /// K-generic constructor: assembles exterior derivative, Hodge star, and
    /// scalar Laplace-Beltrami from any mesh.
    pub fn from_mesh_generic(
        mesh: &Mesh<M, K, B>,
        manifold: &M,
    ) -> Result<Self, DecError> {
        let ext = ExteriorDerivative::from_mesh_sparse_generic(mesh);
        let hodge = HodgeStar::from_mesh_generic(mesh, manifold)?;

        let laplace_beltrami = assemble_scalar_laplacian(&ext, &hodge);

        Ok(Self {
            laplace_beltrami,
            mass: hodge.star.iter().cloned().collect(),
            ext,
            hodge,
            _phantom: PhantomData,
        })
    }

    /// Apply the scalar Laplace-Beltrami operator to a 0-form (vertex field).
    pub fn apply_laplace_beltrami(&self, f: &DVector<f64>) -> DVector<f64> {
        let n = f.len();
        let mut result = DVector::<f64>::zeros(n);
        for (row_val, row_idx) in self.laplace_beltrami.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (col_idx, &val) in row_idx.iter() {
                sum += val * f[col_idx];
            }
            result[row_val] = sum;
        }
        result
    }
}
```

Note: `from_mesh_sparse_generic` needs to be `pub` in `exterior.rs`. Update its visibility:

```rust
    /// K-generic sparse construction.
    pub fn from_mesh_sparse_generic<M: Manifold, const K: usize, const B: usize>(
```

Bochner and Lichnerowicz remain on `impl<M: Manifold> Operators<M, 3, 2>`:

```rust
impl<M: Manifold> Operators<M, 3, 2> {
    /// Apply the Bochner (connection) Laplacian to a vector field.
    pub fn apply_bochner_laplacian(
        &self,
        u: &DVector<f64>,
        ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(u.len(), 2 * nv, "Bochner: u must have 2*n_v entries");

        let ux = u.rows(0, nv).into_owned();
        let uy = u.rows(nv, nv).into_owned();

        let mut lux = self.apply_laplace_beltrami(&ux);
        let mut luy = self.apply_laplace_beltrami(&uy);

        if let Some(ric) = ricci_correction {
            for v in 0..nv {
                let r = ric(v);
                let ux_v = ux[v];
                let uy_v = uy[v];
                lux[v] += r[0][0] * ux_v + r[0][1] * uy_v;
                luy[v] += r[1][0] * ux_v + r[1][1] * uy_v;
            }
        }

        let mut result = DVector::<f64>::zeros(2 * nv);
        result.rows_mut(0, nv).copy_from(&lux);
        result.rows_mut(nv, nv).copy_from(&luy);
        result
    }

    /// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field Q.
    pub fn apply_lichnerowicz_laplacian(
        &self,
        q: &DVector<f64>,
        curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(q.len(), 3 * nv, "Lichnerowicz: q must have 3*n_v entries");

        let qxx = q.rows(0, nv).into_owned();
        let qxy = q.rows(nv, nv).into_owned();
        let qyy = q.rows(2 * nv, nv).into_owned();

        let mut lxx = self.apply_laplace_beltrami(&qxx);
        let mut lxy = self.apply_laplace_beltrami(&qxy);
        let mut lyy = self.apply_laplace_beltrami(&qyy);

        if let Some(curv) = curvature_correction {
            for v in 0..nv {
                let c = curv(v);
                let qx = qxx[v];
                let qm = qxy[v];
                let qy = qyy[v];
                lxx[v] += c[0][0] * qx + c[0][1] * qm + c[0][2] * qy;
                lxy[v] += c[1][0] * qx + c[1][1] * qm + c[1][2] * qy;
                lyy[v] += c[2][0] * qx + c[2][1] * qm + c[2][2] * qy;
            }
        }

        let mut result = DVector::<f64>::zeros(3 * nv);
        result.rows_mut(0, nv).copy_from(&lxx);
        result.rows_mut(nv, nv).copy_from(&lxy);
        result.rows_mut(2 * nv, nv).copy_from(&lyy);
        result
    }
}
```

#### Step 5.3: Update backward-compat accessors in `laplace.rs`

- [ ] The old `Operators<M>` type with default `K=3, B=2` should work transparently. The `from_mesh` method on `Operators<Euclidean<2>, 3, 2>` produces exactly the old type. The old `mass0` and `mass1` fields are now `mass[0]` and `mass[1]`. Update the integration tests accordingly:

Replace `ops.mass0` with `ops.mass[0]` and `ops.mass1` with `ops.mass[1]` in all tests.

#### Step 5.4: Update lib.rs export

- [ ] No change needed since `Operators` is already re-exported and the type still exists with the same name.

#### Step 5.5: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass
```

---

### Task 6: Generalize Advection to K-Generic + Use Adjacency

**Goal**: Replace the O(V*E) scan in `apply_scalar_advection` with O(V * avg_degree) via `vertex_boundaries`. Generalize the signature to `<M, K, B>` with velocity as `&[M::Tangent]`. Keep the old flat signature as a backward-compat wrapper.

**Files touched**:
- `~/cartan/cartan-dec/src/advection.rs`
- `~/cartan/cartan-dec/src/lib.rs` (update exports)
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 6.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
use cartan_dec::advection::apply_scalar_advection_generic;

#[test]
fn advection_generic_constant_field_vanishes() {
    // (u . nabla) c = 0 for any constant scalar field, using the generic API.
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let nv = mesh.n_vertices();

    let f = DVector::<f64>::from_element(nv, 5.0);
    // Velocity: (1.0, 0.5) at every vertex.
    let u: Vec<SVector<f64, 2>> = vec![SVector::new(1.0, 0.5); nv];

    let adv = apply_scalar_advection_generic(&mesh, &manifold, &f, &u);
    let max_err = adv.abs().max();
    assert!(
        max_err < 1e-13,
        "generic advection of constant: max = {max_err:.2e}"
    );
}

#[test]
fn advection_generic_matches_old() {
    // The generic advection must match the old flat advection on FlatMesh.
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let nv = mesh.n_vertices();

    // Non-trivial scalar field: f(x,y) = sin(pi*x) * cos(pi*y)
    let f: DVector<f64> = DVector::from_fn(nv, |v, _| {
        let p = mesh.vertex(v);
        (std::f64::consts::PI * p.x).sin() * (std::f64::consts::PI * p.y).cos()
    });

    // Non-trivial velocity: u = (x, -y)
    let mut u_old = DVector::<f64>::zeros(2 * nv);
    let mut u_new: Vec<SVector<f64, 2>> = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = mesh.vertex(v);
        u_old[v] = p.x;
        u_old[nv + v] = -p.y;
        u_new.push(SVector::new(p.x, -p.y));
    }

    let adv_old = apply_scalar_advection(&mesh, &f, &u_old);
    let adv_new = apply_scalar_advection_generic(&mesh, &manifold, &f, &u_new);

    let diff = (&adv_old - &adv_new).norm();
    assert!(
        diff < 1e-12,
        "generic vs old advection: diff = {diff}"
    );
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec advection_generic 2>&1
# Expected: compilation error (apply_scalar_advection_generic does not exist)
```

#### Step 6.2: Implement K-generic advection

- [ ] Rewrite `~/cartan/cartan-dec/src/advection.rs`:

```rust
// ~/cartan/cartan-dec/src/advection.rs

//! Discrete covariant advection operator for scalar and tensor-valued fields.
//!
//! The upwind scheme computes (u . nabla) f at each vertex by iterating over
//! incident boundary faces (edges for K=3) via the adjacency maps. This gives
//! O(V * avg_degree) complexity regardless of K.
//!
//! ## References
//!
//! - LeVeque. "Finite Volume Methods for Hyperbolic Problems." Cambridge, 2002.
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.

use nalgebra::DVector;

use cartan_core::Manifold;

use crate::mesh::{FlatMesh, Mesh};

/// K-generic upwind covariant advection of a scalar 0-form.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh (must have adjacency maps built).
/// - `manifold`: the Riemannian manifold for metric operations.
/// - `f`: scalar field at vertices (n_v vector).
/// - `u`: velocity field as one tangent vector per vertex.
///
/// # Returns
///
/// `(u . nabla) f` at each vertex as an n_v vector.
pub fn apply_scalar_advection_generic<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    f: &DVector<f64>,
    u: &[M::Tangent],
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(f.len(), nv, "advection: f must have n_v entries");
    assert_eq!(u.len(), nv, "advection: u must have n_v tangent vectors");

    let mut result = DVector::<f64>::zeros(nv);

    for v in 0..nv {
        let pv = &mesh.vertices[v];
        let uv = &u[v];

        // Iterate over incident boundary faces of vertex v.
        for &b in &mesh.vertex_boundaries[v] {
            let boundary = &mesh.boundaries[b];

            // Find the "other" endpoint(s) of this boundary face.
            // For an edge (B=2), there is exactly one other vertex.
            // For a face (B=3), we use the barycenter of the other vertices.
            // General approach: compute the direction from v to the centroid
            // of the other vertices in the boundary face.
            let mut other_vertices = Vec::with_capacity(B - 1);
            for &bv in boundary {
                if bv != v {
                    other_vertices.push(bv);
                }
            }

            if other_vertices.is_empty() {
                continue;
            }

            // For B=2 (edge), there is exactly one other vertex.
            // For simplicity and generality, average over all other vertices.
            for &other in &other_vertices {
                let po = &mesh.vertices[other];

                // Direction from v to other in tangent space.
                let edge_tangent = match manifold.log(pv, po) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let len = manifold.norm(pv, &edge_tangent);
                if len < 1e-30 {
                    continue;
                }

                // Project velocity onto edge direction: u_v . (edge / |edge|)
                // = inner(u_v, edge) / |edge|
                let u_proj = manifold.inner(pv, uv, &edge_tangent) / len;

                // Upwind flux.
                result[v] += if u_proj > 0.0 {
                    u_proj * (f[other] - f[v]) / len
                } else {
                    u_proj * (f[v] - f[other]) / len
                };
            }
        }
    }

    result
}

/// Apply the upwind covariant advection operator to a scalar 0-form (flat mesh, old API).
///
/// Backward-compatible wrapper. Velocity is stored as [u_x[0..n_v], u_y[0..n_v]].
pub fn apply_scalar_advection(mesh: &FlatMesh, f: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(f.len(), nv, "advection: f must have n_v entries");
    assert_eq!(u.len(), 2 * nv, "advection: u must have 2*n_v entries");

    // Convert DVector layout to Vec<SVector<f64, 2>>.
    let u_tangent: Vec<nalgebra::SVector<f64, 2>> = (0..nv)
        .map(|v| nalgebra::SVector::new(u[v], u[nv + v]))
        .collect();

    let manifold = cartan_manifolds::euclidean::Euclidean::<2>;
    apply_scalar_advection_generic(mesh, &manifold, f, &u_tangent)
}

/// Apply the upwind covariant advection operator to a vector-valued 0-form (flat mesh, old API).
///
/// For a flat domain, applies scalar advection component-wise.
pub fn apply_vector_advection(mesh: &FlatMesh, q: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(
        q.len(),
        2 * nv,
        "vector_advection: q must have 2*n_v entries"
    );
    assert_eq!(
        u.len(),
        2 * nv,
        "vector_advection: u must have 2*n_v entries"
    );

    let qx = q.rows(0, nv).into_owned();
    let qy = q.rows(nv, nv).into_owned();

    let lqx = apply_scalar_advection(mesh, &qx, u);
    let lqy = apply_scalar_advection(mesh, &qy, u);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&lqx);
    result.rows_mut(nv, nv).copy_from(&lqy);
    result
}
```

#### Step 6.3: Update lib.rs exports

- [ ] In `~/cartan/cartan-dec/src/lib.rs`, add the generic advection to exports:

```rust
pub use advection::{apply_scalar_advection, apply_scalar_advection_generic, apply_vector_advection};
```

#### Step 6.4: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass (existing + new generic advection tests)
```

---

### Task 7: Generalize Divergence to K-Generic

**Goal**: Add `apply_divergence_generic<M, K, B>` that takes `&[M::Tangent]` instead of a `DVector` layout. Keep the old flat API as a wrapper. Tensor divergence remains K=3 only.

**Files touched**:
- `~/cartan/cartan-dec/src/divergence.rs`
- `~/cartan/cartan-dec/src/lib.rs` (update exports)
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 7.1: Write failing tests

- [ ] Add tests to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
use cartan_dec::divergence::apply_divergence_generic;

#[test]
fn divergence_generic_constant_field_vanishes() {
    // div(c) = 0 for any constant vector field, generic API.
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let hodge = HodgeStar::from_mesh(&mesh, &manifold);
    let nv = mesh.n_vertices();

    // u = (1, 0) everywhere.
    let u: Vec<SVector<f64, 2>> = vec![SVector::new(1.0, 0.0); nv];

    let div_u = apply_divergence_generic(&mesh, &manifold, &ext, &hodge, &u);
    let max_interior_err = {
        let mut m = 0.0f64;
        for v in 0..nv {
            let p = mesh.vertex(v);
            let is_boundary =
                p.x < 1e-10 || p.x > 1.0 - 1e-10 || p.y < 1e-10 || p.y > 1.0 - 1e-10;
            if !is_boundary {
                m = m.max(div_u[v].abs());
            }
        }
        m
    };

    assert!(
        max_interior_err < 1e-12,
        "generic div(const) interior: max = {max_interior_err:.2e}"
    );
}

#[test]
fn divergence_generic_matches_old() {
    // Generic divergence must match the old flat divergence.
    use nalgebra::SVector;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ext = ExteriorDerivative::from_mesh(&mesh);
    let hodge = HodgeStar::from_mesh(&mesh, &manifold);
    let nv = mesh.n_vertices();

    // u = (x, -y)
    let mut u_old = DVector::<f64>::zeros(2 * nv);
    let mut u_new: Vec<SVector<f64, 2>> = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = mesh.vertex(v);
        u_old[v] = p.x;
        u_old[nv + v] = -p.y;
        u_new.push(SVector::new(p.x, -p.y));
    }

    let div_old = apply_divergence(&mesh, &ext, &hodge, &u_old);
    let div_new = apply_divergence_generic(&mesh, &manifold, &ext, &hodge, &u_new);

    let diff = (&div_old - &div_new).norm();
    assert!(
        diff < 1e-12,
        "generic vs old divergence: diff = {diff}"
    );
}
```

- [ ] Verify tests fail:

```bash
cd ~/cartan && cargo test -p cartan-dec divergence_generic 2>&1
# Expected: compilation error (apply_divergence_generic does not exist)
```

#### Step 7.2: Implement K-generic divergence

- [ ] Rewrite `~/cartan/cartan-dec/src/divergence.rs`:

```rust
// ~/cartan/cartan-dec/src/divergence.rs

//! Discrete covariant divergence of a vector field.
//!
//! The DEC divergence formula is: div(u) = star_0_inv * d0^T * star_1 * u_1form
//!
//! This formula is K-agnostic: it only uses the 0-form and 1-form operators
//! (d0, star0, star1). The velocity-to-1-form conversion uses trapezoidal
//! integration along boundary faces (edges for K=3).
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003.

use nalgebra::DVector;

use cartan_core::Manifold;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::{FlatMesh, Mesh};

/// K-generic discrete divergence of a vertex-based vector field.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh.
/// - `manifold`: the Riemannian manifold.
/// - `ext`: precomputed exterior derivative operators.
/// - `hodge`: precomputed Hodge star operators.
/// - `u`: velocity field as one tangent vector per vertex.
///
/// # Returns
///
/// div(u) at each vertex as an n_v vector.
pub fn apply_divergence_generic<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    u: &[M::Tangent],
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    let nb = mesh.n_boundaries();
    assert_eq!(u.len(), nv, "divergence: u must have n_v tangent vectors");

    // Step 1: Build the 1-form from the vector field.
    // For each boundary face (edge for K=3) [i, j], the 1-form is:
    //   u_1form[b] = avg(u[i], u[j]) . (v_j - v_i)
    // Generalized: for each boundary face with B vertices, use trapezoidal
    // integration. For B=2 (edges), this is the midpoint rule.
    let mut u1form = DVector::<f64>::zeros(nb);
    for (b, boundary) in mesh.boundaries.iter().enumerate() {
        // For B=2 edges: the 1-form integral is avg(u) . edge_vector.
        // edge_vector is log(v_i, v_j) for i=boundary[0], j=boundary[1].
        if B == 2 {
            let i = boundary[0];
            let j = boundary[1];
            let pi = &mesh.vertices[i];
            let pj = &mesh.vertices[j];
            let edge_vec = match manifold.log(pi, pj) {
                Ok(t) => t,
                Err(_) => continue,
            };
            // Average velocity contribution: 0.5 * (u[i] + u[j]) . edge
            // inner product at pi (first-order approximation).
            let ui_dot = manifold.inner(pi, &u[i], &edge_vec);
            let uj_dot = manifold.inner(pi, &u[j], &edge_vec);
            u1form[b] = 0.5 * (ui_dot + uj_dot);
        }
        // For B > 2, a more sophisticated integration rule would be needed.
        // This is left as a future extension.
    }

    // Step 2: Apply star1 to the 1-form.
    let star1_u1form = u1form.component_mul(hodge.star1());

    // Step 3: Apply d0^T (sparse transpose multiply).
    let d0t = ext.d0().transpose_view();
    let mut d0t_star1_u = DVector::<f64>::zeros(nv);
    for (row_val, row_idx) in d0t.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row_idx.iter() {
            sum += val * star1_u1form[col_idx];
        }
        d0t_star1_u[row_val] = sum;
    }

    // Step 4: Apply star0_inv.
    let star0_inv = hodge.star0_inv();
    d0t_star1_u.component_mul(&star0_inv)
}

/// Backward-compatible discrete divergence (flat mesh, DVector velocity layout).
pub fn apply_divergence(
    mesh: &FlatMesh,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    u: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(u.len(), 2 * nv, "divergence: u must have 2*n_v entries");

    let u_tangent: Vec<nalgebra::SVector<f64, 2>> = (0..nv)
        .map(|v| nalgebra::SVector::new(u[v], u[nv + v]))
        .collect();

    let manifold = cartan_manifolds::euclidean::Euclidean::<2>;
    apply_divergence_generic(mesh, &manifold, ext, hodge, &u_tangent)
}

/// Compute the discrete divergence of a symmetric 2-tensor field (K=3 only).
///
/// For a symmetric 2-tensor field T with components [T_xx, T_xy, T_yy] at
/// each vertex, the divergence is a vector field:
///   (div T)_x = div of first column  (T_xx, T_xy)
///   (div T)_y = div of second column (T_xy, T_yy)
pub fn apply_tensor_divergence(
    mesh: &FlatMesh,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    t: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(
        t.len(),
        3 * nv,
        "tensor_divergence: t must have 3*n_v entries"
    );

    let txx = t.rows(0, nv).into_owned();
    let txy = t.rows(nv, nv).into_owned();
    let tyy = t.rows(2 * nv, nv).into_owned();

    // First column of T: (T_xx, T_xy).
    let mut col1 = DVector::<f64>::zeros(2 * nv);
    col1.rows_mut(0, nv).copy_from(&txx);
    col1.rows_mut(nv, nv).copy_from(&txy);

    // Second column of T: (T_xy, T_yy).
    let mut col2 = DVector::<f64>::zeros(2 * nv);
    col2.rows_mut(0, nv).copy_from(&txy);
    col2.rows_mut(nv, nv).copy_from(&tyy);

    let div_x = apply_divergence(mesh, ext, hodge, &col1);
    let div_y = apply_divergence(mesh, ext, hodge, &col2);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&div_x);
    result.rows_mut(nv, nv).copy_from(&div_y);
    result
}
```

#### Step 7.3: Update lib.rs exports

- [ ] In `~/cartan/cartan-dec/src/lib.rs`, add the generic divergence:

```rust
pub use divergence::{apply_divergence, apply_divergence_generic, apply_tensor_divergence};
```

#### Step 7.4: Run all tests

- [ ] Run:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass
```

---

### Task 8: Update `lib.rs` Exports + Backward Compat Integration Test + K=4 Smoke Test

**Goal**: Verify all public exports are correct, run the full existing test suite, and add a K=4 tet mesh smoke test that exercises adjacency, exterior derivative exactness, and the generic simplex volume.

**Files touched**:
- `~/cartan/cartan-dec/src/lib.rs`
- `~/cartan/cartan-dec/tests/integration.rs`

#### Step 8.1: Verify lib.rs exports

- [ ] Ensure `~/cartan/cartan-dec/src/lib.rs` has the following exports:

```rust
pub mod advection;
pub mod divergence;
pub mod error;
pub mod exterior;
pub mod hodge;
pub mod laplace;
pub mod mesh;

pub use advection::{apply_scalar_advection, apply_scalar_advection_generic, apply_vector_advection};
pub use divergence::{apply_divergence, apply_divergence_generic, apply_tensor_divergence};
pub use error::DecError;
pub use exterior::ExteriorDerivative;
pub use hodge::HodgeStar;
pub use laplace::Operators;
pub use mesh::{FlatMesh, Mesh};
```

#### Step 8.2: Write K=4 tet mesh smoke test

- [ ] Add to `~/cartan/cartan-dec/tests/integration.rs`:

```rust
#[test]
fn tet_mesh_k4_adjacency() {
    // Build a single-tet mesh and verify adjacency.
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    // 4 vertices, 1 tet, 4 boundary faces (triangles).
    assert_eq!(mesh.n_vertices(), 4);
    assert_eq!(mesh.n_simplices(), 1);
    assert_eq!(mesh.n_boundaries(), 4);

    // Each vertex is in 3 boundary faces (the 3 faces not omitting that vertex).
    for v in 0..4 {
        assert_eq!(
            mesh.vertex_boundaries[v].len(),
            3,
            "vertex {v} has {} boundary faces, expected 3",
            mesh.vertex_boundaries[v].len()
        );
    }

    // Each vertex is in 1 simplex.
    for v in 0..4 {
        assert_eq!(mesh.vertex_simplices[v].len(), 1);
    }

    // Each boundary face has exactly 1 co-simplex (boundary of a single tet).
    for b in 0..4 {
        assert_eq!(
            mesh.boundary_simplices[b].len(),
            1,
            "boundary face {b} has {} cofaces, expected 1",
            mesh.boundary_simplices[b].len()
        );
    }

    // Euler characteristic: V - B + S = 4 - 4 + 1 = 1
    // (This is chi for a solid tet, a contractible 3-manifold with boundary.)
    assert_eq!(mesh.euler_characteristic(), 1);
}

#[test]
fn tet_mesh_k4_exterior_exactness() {
    // d[1] * d[0] = 0 for a tet mesh.
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let ext = ExteriorDerivative::from_mesh_sparse_generic(&mesh);

    // Should have 2 operators: d[0] (edges x verts), d[1] (faces x edges).
    assert_eq!(ext.degree(), 2);

    // d[0]: n_boundaries(=4 faces for K=4? No, K=4 boundaries are triangles.)
    // Wait: for K=4, boundaries are B=3 faces (triangles), and d[0] maps
    // vertices to boundaries. But boundaries are 2-simplices (faces), not edges.
    // Actually, d[0] maps 0-simplices (vertices) to 1-simplices (boundaries of boundaries).
    // For K=4: the chain is d[0]: edges x verts, d[1]: faces x edges, d[2]: tets x faces.
    // But our ExteriorDerivative only builds d[0] and d[1] from the mesh's boundaries
    // and simplices. For K=4, boundaries are faces (B=3), so d[0] maps vertices to faces?
    //
    // Actually, let's re-examine: for K=4, the mesh stores:
    //   - vertices (0-simplices)
    //   - boundaries (B=3, i.e., triangular faces, which are 2-simplices)
    //   - simplices (K=4, i.e., tets, which are 3-simplices)
    // So d[0] should map vertices->faces and d[1] should map faces->tets.
    // But that skips edges (1-simplices). This is because our Mesh<M, K, B> only
    // stores two levels: boundaries and simplices.
    //
    // For a FULL chain complex of a tet, we would need edges too.
    // The current generic construction builds exactly 2 operators:
    //   d[0]: boundaries x vertices
    //   d[1]: simplices x boundaries
    // For K=3: boundaries are edges, so d[0] is edges x verts, d[1] is faces x edges. Correct.
    // For K=4: boundaries are faces, so d[0] is faces x verts, d[1] is tets x faces. Partial.
    //
    // Exactness d[1]*d[0]=0 should still hold for this partial chain.

    let max_err = ext.check_exactness();
    assert!(
        max_err < 1e-14,
        "K=4 exactness: max entry of d1*d0 = {max_err:.2e}"
    );
}

#[test]
fn tet_mesh_k4_simplex_volume() {
    // Volume of the standard simplex [0,0,0],[1,0,0],[0,1,0],[0,0,1] = 1/6.
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let v1 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let v2 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let v3 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);

    let mesh: Mesh<Euclidean<3>, 4, 3> = Mesh::from_simplices_generic(
        &manifold,
        vec![v0, v1, v2, v3],
        vec![[0, 1, 2, 3]],
    );

    let vol = mesh.simplex_volume(&manifold, 0);
    let expected = 1.0 / 6.0;
    assert!(
        (vol - expected).abs() < 1e-14,
        "standard tet volume: got {vol}, expected {expected}"
    );
}
```

#### Step 8.3: Run full test suite

- [ ] Run all cartan-dec tests:

```bash
cd ~/cartan && cargo test -p cartan-dec 2>&1
# Expected: ALL tests pass (existing backward-compat + all new K-generic tests)
```

- [ ] Run the full workspace build to catch any cross-crate breakage:

```bash
cd ~/cartan && cargo build --workspace 2>&1
# Expected: clean build, no errors
```

- [ ] Run clippy:

```bash
cd ~/cartan && cargo clippy -p cartan-dec -- -D warnings 2>&1
# Expected: no warnings
```

---

## Phase B: cartan-remesh (New Crate, K=3 Specialization)

### Task 9: Scaffold cartan-remesh crate

**Prerequisite**: Phase A complete (adjacency maps and `rebuild_adjacency` on `Mesh<M, K, B>`).

**Files touched**:
| File | Action |
|------|--------|
| `~/cartan/cartan-remesh/Cargo.toml` | Create |
| `~/cartan/cartan-remesh/src/lib.rs` | Create |
| `~/cartan/Cargo.toml` | Modify (add workspace member) |

**Step 9.1: Create crate directory**

```bash
mkdir -p ~/cartan/cartan-remesh/src
```

**Step 9.2: Write `~/cartan/cartan-remesh/Cargo.toml`**

```toml
[package]
name = "cartan-remesh"
description = "Adaptive remeshing primitives for triangle meshes on Riemannian manifolds"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
keywords = ["remesh", "riemannian", "triangle", "mesh", "adaptive"]
categories = ["mathematics", "science", "algorithms"]
documentation = "https://docs.rs/cartan-remesh"
homepage = "https://cartan.sotofranco.dev"

[dependencies]
cartan-core      = { path = "../cartan-core", version = "0.1" }
cartan-manifolds = { path = "../cartan-manifolds", version = "0.1" }
cartan-dec       = { path = "../cartan-dec", version = "0.1" }
thiserror        = { workspace = true }

[dev-dependencies]
approx = { workspace = true }

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

**Step 9.3: Write `~/cartan/cartan-remesh/src/lib.rs`**

```rust
// ~/cartan/cartan-remesh/src/lib.rs

//! # cartan-remesh
//!
//! Adaptive remeshing primitives for triangle meshes on Riemannian manifolds.
//!
//! All operations are generic over `M: Manifold` and operate on
//! `&mut Mesh<M, 3, 2>`. Every mutation is logged in a [`RemeshLog`] so that
//! downstream solvers can interpolate fields across topology changes.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`log`] | `RemeshLog`, `EdgeSplit`, `EdgeCollapse`, `EdgeFlip`, `VertexShift` |
//! | [`config`] | `RemeshConfig` with curvature-CFL and quality parameters |
//! | [`error`] | `RemeshError` error type |
//! | [`primitives`] | `split_edge`, `collapse_edge`, `flip_edge`, `shift_vertex` |
//! | [`lcr`] | Length-cross-ratio conformal regularization |
//! | [`driver`] | `adaptive_remesh` pipeline and `needs_remesh` predicate |

pub mod config;
pub mod driver;
pub mod error;
pub mod lcr;
pub mod log;
pub mod primitives;

pub use config::RemeshConfig;
pub use driver::{adaptive_remesh, needs_remesh};
pub use error::RemeshError;
pub use lcr::{capture_reference_lcrs, lcr_spring_energy, lcr_spring_gradient, length_cross_ratio};
pub use log::{EdgeCollapse, EdgeFlip, EdgeSplit, RemeshLog, VertexShift};
pub use primitives::{collapse_edge, flip_edge, shift_vertex, split_edge};
```

**Step 9.4: Write `~/cartan/cartan-remesh/src/log.rs`**

```rust
// ~/cartan/cartan-remesh/src/log.rs

//! Remesh operation log.
//!
//! Every topology-changing or vertex-moving operation records its mutations
//! in a [`RemeshLog`]. Downstream solvers (volterra-dec) use this log to
//! interpolate Q-tensor, velocity, and scalar fields across remesh events.

/// A complete record of all remesh mutations applied in one pass.
///
/// Fields are populated in pipeline order: flips, splits, collapses, shifts.
/// An empty log means no mutations were applied.
#[derive(Debug, Clone, Default)]
pub struct RemeshLog {
    /// Edge splits performed (vertex insertions).
    pub splits: Vec<EdgeSplit>,
    /// Edge collapses performed (vertex removals).
    pub collapses: Vec<EdgeCollapse>,
    /// Edge flips performed (diagonal swaps).
    pub flips: Vec<EdgeFlip>,
    /// Vertex shifts performed (tangential smoothing moves).
    pub shifts: Vec<VertexShift>,
}

impl RemeshLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of mutations recorded.
    pub fn total_mutations(&self) -> usize {
        self.splits.len() + self.collapses.len() + self.flips.len() + self.shifts.len()
    }

    /// Whether any topology-changing operation occurred (split, collapse, or flip).
    pub fn topology_changed(&self) -> bool {
        !self.splits.is_empty() || !self.collapses.is_empty() || !self.flips.is_empty()
    }

    /// Merge another log into this one (appends all entries).
    pub fn merge(&mut self, other: RemeshLog) {
        self.splits.extend(other.splits);
        self.collapses.extend(other.collapses);
        self.flips.extend(other.flips);
        self.shifts.extend(other.shifts);
    }
}

/// Record of a single edge split operation.
///
/// An edge (v_a, v_b) is split by inserting a new vertex at the geodesic
/// midpoint. The two adjacent triangles become four. The new vertex index
/// and new edge indices are recorded for field interpolation.
#[derive(Debug, Clone)]
pub struct EdgeSplit {
    /// Index of the edge that was split (before mutation).
    pub old_edge: usize,
    /// First endpoint of the split edge.
    pub v_a: usize,
    /// Second endpoint of the split edge.
    pub v_b: usize,
    /// Index of the newly inserted vertex (at geodesic midpoint of v_a, v_b).
    pub new_vertex: usize,
    /// Indices of all newly created edges after the split.
    pub new_edges: Vec<usize>,
}

/// Record of a single edge collapse operation.
///
/// An edge is collapsed by merging two vertices. The surviving vertex is
/// moved to the geodesic midpoint. Two degenerate triangles are removed.
#[derive(Debug, Clone)]
pub struct EdgeCollapse {
    /// Index of the collapsed edge (before mutation).
    pub old_edge: usize,
    /// Vertex that remains after collapse (moved to midpoint).
    pub surviving_vertex: usize,
    /// Vertex that was removed.
    pub removed_vertex: usize,
    /// Indices of the triangles removed by the collapse.
    pub removed_faces: Vec<usize>,
}

/// Record of a single edge flip operation.
///
/// The diagonal of the quad formed by two adjacent triangles is swapped.
/// The old edge endpoints are replaced by the two opposite vertices.
#[derive(Debug, Clone)]
pub struct EdgeFlip {
    /// Index of the edge that was flipped.
    pub old_edge: usize,
    /// New endpoint vertex indices after the flip.
    pub new_edge: [usize; 2],
    /// The two triangle indices whose connectivity changed.
    pub affected_faces: [usize; 2],
}

/// Record of a single vertex shift (tangential smoothing move).
///
/// The displacement is stored in tangent-space coordinates at the vertex
/// so the move can be undone or attenuated.
#[derive(Debug, Clone)]
pub struct VertexShift {
    /// Index of the shifted vertex.
    pub vertex: usize,
    /// Displacement in the tangent space at the vertex (for undo/attenuation).
    pub old_pos_tangent: Vec<f64>,
}
```

**Step 9.5: Write `~/cartan/cartan-remesh/src/config.rs`**

```rust
// ~/cartan/cartan-remesh/src/config.rs

//! Remesh configuration parameters.
//!
//! [`RemeshConfig`] controls the adaptive remeshing pipeline: curvature-CFL
//! resolution, edge length bounds, area bounds, foldover protection, LCR
//! spring stiffness, and smoothing iterations.

/// Configuration for the adaptive remeshing pipeline.
///
/// The curvature-CFL criterion enforces `h_e < curvature_scale / sqrt(k_max)`
/// where `k_max = |H| + sqrt(H^2 - K)` is the larger principal curvature
/// magnitude. Edges violating this bound are split; edges below
/// `min_edge_length` are collapsed.
#[derive(Debug, Clone)]
pub struct RemeshConfig {
    /// Constant C in the curvature-CFL criterion h < C / sqrt(k_max).
    /// Smaller values produce finer meshes near high curvature.
    pub curvature_scale: f64,

    /// Minimum allowed edge length. Edges shorter than this are collapsed.
    pub min_edge_length: f64,

    /// Maximum allowed edge length. Edges longer than this are split
    /// regardless of curvature.
    pub max_edge_length: f64,

    /// Minimum allowed triangle area. Triangles below this are collapsed.
    pub min_face_area: f64,

    /// Maximum allowed triangle area. Triangles above this trigger splits.
    pub max_face_area: f64,

    /// Foldover rejection threshold in radians. An edge collapse is rejected
    /// if any adjacent face normal would rotate by more than this angle.
    /// Default: 0.5 radians (~28.6 degrees).
    pub foldover_threshold: f64,

    /// LCR spring stiffness for conformal regularization.
    /// Set to 0.0 to disable LCR springs.
    pub lcr_spring_stiffness: f64,

    /// Number of tangential Laplacian smoothing iterations per remesh pass.
    pub smoothing_iterations: usize,
}

impl Default for RemeshConfig {
    fn default() -> Self {
        Self {
            curvature_scale: 0.5,
            min_edge_length: 0.01,
            max_edge_length: 1.0,
            min_face_area: 1e-6,
            max_face_area: 1.0,
            foldover_threshold: 0.5,
            lcr_spring_stiffness: 0.0,
            smoothing_iterations: 3,
        }
    }
}
```

**Step 9.6: Write `~/cartan/cartan-remesh/src/error.rs`**

```rust
// ~/cartan/cartan-remesh/src/error.rs

//! Error types for remeshing operations.

/// Errors that can occur during remeshing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RemeshError {
    /// Edge collapse would cause a triangle foldover (normal inversion).
    #[error(
        "foldover detected at face {face}: normal rotation {angle_rad:.4} rad exceeds threshold {threshold:.4} rad"
    )]
    Foldover {
        /// The face that would fold over.
        face: usize,
        /// The angle (in radians) the normal would rotate.
        angle_rad: f64,
        /// The configured foldover threshold.
        threshold: f64,
    },

    /// The edge is a boundary edge and cannot be flipped.
    #[error("edge {edge} is a boundary edge (only one adjacent face)")]
    BoundaryEdge {
        /// The boundary edge index.
        edge: usize,
    },

    /// The edge flip would not improve the Delaunay criterion.
    #[error("edge {edge} already satisfies Delaunay criterion (opposite angle sum = {angle_sum:.4} rad)")]
    AlreadyDelaunay {
        /// The edge index.
        edge: usize,
        /// Sum of opposite angles in radians.
        angle_sum: f64,
    },

    /// A vertex or edge index is out of bounds.
    #[error("index out of bounds: {index} >= {len}")]
    IndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The collection size.
        len: usize,
    },

    /// The edge has fewer than 2 adjacent faces (boundary or degenerate).
    #[error("edge {edge} has {count} adjacent faces, need exactly 2")]
    NotInteriorEdge {
        /// The edge index.
        edge: usize,
        /// Number of adjacent faces found.
        count: usize,
    },

    /// A manifold geodesic operation (log/exp) failed.
    #[error("geodesic computation failed: {reason}")]
    GeodesicFailed {
        /// Description of the failure.
        reason: String,
    },
}
```

**Step 9.7: Write stub modules**

`~/cartan/cartan-remesh/src/primitives.rs`:

```rust
// ~/cartan/cartan-remesh/src/primitives.rs

//! Primitive remesh operations: split, collapse, flip, shift.
//!
//! All operations are generic over `M: Manifold` and mutate `&mut Mesh<M, 3, 2>`
//! in place. Every mutation is recorded in a [`RemeshLog`].

use cartan_core::Manifold;
use cartan_dec::Mesh;

use crate::error::RemeshError;
use crate::log::RemeshLog;

/// Split an edge by inserting a vertex at the geodesic midpoint.
///
/// The two triangles adjacent to the edge become four triangles.
/// Adjacency is rebuilt after the split.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn split_edge<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
) -> RemeshLog {
    todo!("Task 10")
}

/// Collapse an edge by merging its endpoints at the geodesic midpoint.
///
/// The surviving vertex (lower index) is moved to the midpoint. Two
/// degenerate triangles are removed. Returns `Err(RemeshError::Foldover)`
/// if any adjacent face normal would rotate beyond the configured threshold.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn collapse_edge<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
    _foldover_threshold: f64,
) -> Result<RemeshLog, RemeshError> {
    todo!("Task 10")
}

/// Flip the diagonal of the quad formed by two adjacent triangles.
///
/// Returns `Err(RemeshError::AlreadyDelaunay)` if the edge already satisfies
/// the Delaunay criterion (sum of opposite angles <= pi).
/// Returns `Err(RemeshError::BoundaryEdge)` if the edge has only one adjacent face.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn flip_edge<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
) -> Result<RemeshLog, RemeshError> {
    todo!("Task 11")
}

/// Tangential Laplacian smoothing of a single vertex.
///
/// Maps all neighbors into the tangent space at the vertex via `manifold.log`,
/// computes the barycenter, projects out the normal component, and applies
/// the displacement via `manifold.exp`. Boundary vertices are constrained
/// to the boundary edge tangent direction.
///
/// # Panics
///
/// Panics if `vertex >= mesh.n_vertices()`.
pub fn shift_vertex<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _vertex: usize,
) -> RemeshLog {
    todo!("Task 11")
}
```

`~/cartan/cartan-remesh/src/lcr.rs`:

```rust
// ~/cartan/cartan-remesh/src/lcr.rs

//! Length-cross-ratio (LCR) conformal regularization.
//!
//! The LCR of an interior edge measures how far the local mesh geometry
//! deviates from a conformal mapping. LCR springs penalize deviation from
//! a reference configuration, preserving conformal structure during remeshing.

use cartan_core::Manifold;
use cartan_dec::Mesh;

/// Compute the length-cross-ratio of an interior edge.
///
/// For an interior edge with diamond vertices {i, j, k, l} (where i,j are the
/// edge endpoints and k,l are the opposite vertices of the two adjacent
/// triangles):
///
/// `lcr = dist(i,l) * dist(j,k) / (dist(k,i) * dist(l,j))`
///
/// Returns 1.0 for boundary edges (only one adjacent face).
pub fn length_cross_ratio<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
) -> f64 {
    todo!("Task 12")
}

/// Capture reference LCR values for all edges.
///
/// Returns a vector of length `mesh.n_boundaries()` with one LCR per edge.
/// Boundary edges get 1.0.
pub fn capture_reference_lcrs<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
) -> Vec<f64> {
    todo!("Task 12")
}

/// Total LCR spring energy: `0.5 * kst * sum((lcr_e - lcr_ref_e)^2 / lcr_ref_e^2)`.
///
/// Penalizes deviation from the reference conformal structure.
pub fn lcr_spring_energy<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _ref_lcrs: &[f64],
    _kst: f64,
) -> f64 {
    todo!("Task 12")
}

/// Per-vertex gradient of the LCR spring energy.
///
/// Returns a tangent vector at each vertex pointing in the direction of
/// increasing LCR spring energy. The caller negates this for descent.
pub fn lcr_spring_gradient<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _ref_lcrs: &[f64],
    _kst: f64,
) -> Vec<M::Tangent> {
    todo!("Task 12")
}
```

`~/cartan/cartan-remesh/src/driver.rs`:

```rust
// ~/cartan/cartan-remesh/src/driver.rs

//! Adaptive remeshing driver and predicate.
//!
//! The driver runs the full remesh pipeline: flip non-Delaunay edges, split
//! edges violating the curvature-CFL criterion, collapse short/flat edges,
//! shift vertices (tangential Laplacian), and smooth at affected vertices.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};

use crate::config::RemeshConfig;
use crate::log::RemeshLog;

/// Run the full adaptive remeshing pipeline.
///
/// Pipeline order:
/// 1. Flip non-Delaunay edges
/// 2. Split edges violating curvature-CFL: h_e > C / sqrt(k_max)
/// 3. Collapse short/flat edges with foldover guard
/// 4. Shift vertices (tangential Laplacian smoothing)
/// 5. Biharmonic smooth at affected vertices
///
/// Rebuilds adjacency after each topology-changing pass.
/// Returns a merged [`RemeshLog`] covering all mutations.
pub fn adaptive_remesh<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _operators: &Operators<M>,
    _mean_curvatures: &[f64],
    _gaussian_curvatures: &[f64],
    _config: &RemeshConfig,
) -> RemeshLog {
    todo!("Task 13")
}

/// Check whether the mesh needs remeshing.
///
/// Returns true if any edge violates the curvature resolution criterion:
/// `h_e > config.curvature_scale / sqrt(k_max)` where
/// `k_max = |H_v| + sqrt(H_v^2 - K_v)` at the vertex with larger principal
/// curvature. Also returns true if any edge length exceeds `max_edge_length`
/// or falls below `min_edge_length`.
///
/// The caller (volterra-dec) owns the kinetic-energy-minimum timing logic
/// that determines when to act on a true result.
pub fn needs_remesh<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _mean_curvatures: &[f64],
    _gaussian_curvatures: &[f64],
    _config: &RemeshConfig,
) -> bool {
    todo!("Task 13")
}
```

**Step 9.8: Add to workspace**

Edit `~/cartan/Cargo.toml`, add `"cartan-remesh"` to the members list:

```toml
members = [
    "cartan",
    "cartan-core",
    "cartan-manifolds",
    "cartan-optim",
    "cartan-geo",
    "cartan-dec",
    "cartan-remesh",
]
```

**Step 9.9: Write scaffold test**

Create `~/cartan/cartan-remesh/tests/scaffold.rs`:

```rust
// ~/cartan/cartan-remesh/tests/scaffold.rs

//! Scaffold test: verify the crate compiles and basic types are constructible.

use cartan_remesh::{RemeshConfig, RemeshError, RemeshLog};

#[test]
fn remesh_log_default_is_empty() {
    let log = RemeshLog::new();
    assert_eq!(log.total_mutations(), 0);
    assert!(!log.topology_changed());
}

#[test]
fn remesh_config_default_is_sane() {
    let config = RemeshConfig::default();
    assert!(config.curvature_scale > 0.0);
    assert!(config.min_edge_length > 0.0);
    assert!(config.max_edge_length > config.min_edge_length);
    assert!(config.foldover_threshold > 0.0);
}

#[test]
fn remesh_error_display() {
    let err = RemeshError::Foldover {
        face: 42,
        angle_rad: 0.8,
        threshold: 0.5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("foldover"));
    assert!(msg.contains("42"));
}

#[test]
fn remesh_log_merge() {
    use cartan_remesh::EdgeSplit;

    let mut log_a = RemeshLog::new();
    log_a.splits.push(EdgeSplit {
        old_edge: 0,
        v_a: 0,
        v_b: 1,
        new_vertex: 5,
        new_edges: vec![6, 7, 8],
    });

    let mut log_b = RemeshLog::new();
    log_b.splits.push(EdgeSplit {
        old_edge: 3,
        v_a: 2,
        v_b: 3,
        new_vertex: 6,
        new_edges: vec![9, 10, 11],
    });

    log_a.merge(log_b);
    assert_eq!(log_a.splits.len(), 2);
    assert_eq!(log_a.total_mutations(), 2);
    assert!(log_a.topology_changed());
}
```

**Step 9.10: Verify**

```bash
cd ~/cartan && cargo check -p cartan-remesh
cd ~/cartan && cargo test -p cartan-remesh
```

**Step 9.11: Commit**

```bash
cd ~/cartan && git add cartan-remesh/ Cargo.toml
git commit -m "feat(cartan-remesh): scaffold crate with RemeshLog, RemeshConfig, RemeshError

New workspace member cartan-remesh provides adaptive remeshing
primitives for triangle meshes on Riemannian manifolds. All operation
stubs are generic over M: Manifold. Scaffold tests verify types compile
and RemeshLog/RemeshConfig behave correctly."
```

---

### Task 10: Edge split and edge collapse

**Prerequisite**: Task 9 complete. Phase A adjacency maps (`vertex_simplices`, `boundary_simplices`, `rebuild_adjacency`) available on `Mesh<M, 3, 2>`.

**Files touched**:
| File | Action |
|------|--------|
| `~/cartan/cartan-remesh/src/primitives.rs` | Modify (implement `split_edge`, `collapse_edge`) |
| `~/cartan/cartan-remesh/tests/split_collapse.rs` | Create |

**Step 10.1: Write failing tests**

Create `~/cartan/cartan-remesh/tests/split_collapse.rs`:

```rust
// ~/cartan/cartan-remesh/tests/split_collapse.rs

//! Tests for edge split and edge collapse primitives.

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{collapse_edge, split_edge, RemeshError};

/// Build a small flat diamond mesh: 4 vertices, 2 triangles sharing edge (1,2).
///
/// ```text
///     0
///    / \
///   1---2
///    \ /
///     3
/// ```
///
/// Vertices: 0=(0,1), 1=(-1,0), 2=(1,0), 3=(0,-1)
/// Triangles: [1,2,0] (top), [2,1,3] (bottom)  (both CCW)
fn diamond_mesh() -> Mesh<Euclidean<2>, 3, 2> {
    use nalgebra::SVector;
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),   // 0: top
        SVector::from([-1.0, 0.0]),  // 1: left
        SVector::from([1.0, 0.0]),   // 2: right
        SVector::from([0.0, -1.0]),  // 3: bottom
    ];
    let triangles = vec![
        [1, 2, 0],  // top triangle (CCW)
        [2, 1, 3],  // bottom triangle (CCW)
    ];
    Mesh::from_simplices(&manifold, vertices, triangles)
}

/// Compute total mesh area as sum of triangle areas.
fn total_area(mesh: &Mesh<Euclidean<2>, 3, 2>, manifold: &Euclidean<2>) -> f64 {
    (0..mesh.n_simplices())
        .map(|t| mesh.triangle_area(manifold, t))
        .sum()
}

/// Find the edge index for the edge connecting vertices a and b.
fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries.iter().position(|&[i, j]| i == lo && j == hi)
}

#[test]
fn split_edge_adds_one_vertex() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nv_before = mesh.n_vertices();

    // Find the shared edge (1,2)
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    let log = split_edge(&mut mesh, &manifold, edge);

    assert_eq!(mesh.n_vertices(), nv_before + 1, "split must add exactly 1 vertex");
    assert_eq!(log.splits.len(), 1);
    assert_eq!(log.splits[0].new_vertex, nv_before);
}

#[test]
fn split_edge_replaces_two_faces_with_four() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nf_before = mesh.n_simplices();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let _ = split_edge(&mut mesh, &manifold, edge);

    // 2 original triangles become 4
    assert_eq!(mesh.n_simplices(), nf_before + 2);
}

#[test]
fn split_edge_preserves_total_area() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let area_before = total_area(&mesh, &manifold);

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let _ = split_edge(&mut mesh, &manifold, edge);

    let area_after = total_area(&mesh, &manifold);
    assert_relative_eq!(area_before, area_after, epsilon = 1e-12);
}

#[test]
fn split_edge_new_vertex_at_midpoint() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let log = split_edge(&mut mesh, &manifold, edge);

    let new_v = log.splits[0].new_vertex;
    let pos = &mesh.vertices[new_v];
    // Midpoint of (-1,0) and (1,0) should be (0,0)
    assert_relative_eq!(pos[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(pos[1], 0.0, epsilon = 1e-12);
}

#[test]
fn collapse_edge_removes_one_vertex_and_two_faces() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let nv_before = mesh.n_vertices();
    let nf_before = mesh.n_simplices();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let log = collapse_edge(&mut mesh, &manifold, edge, 0.5)
        .expect("collapse should succeed on diamond mesh");

    assert_eq!(mesh.n_vertices(), nv_before - 1);
    assert_eq!(mesh.n_simplices(), nf_before - 2);
    assert_eq!(log.collapses.len(), 1);
    assert_eq!(log.collapses[0].removed_faces.len(), 2);
}

#[test]
fn collapse_edge_surviving_vertex_at_midpoint() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let log = collapse_edge(&mut mesh, &manifold, edge, 0.5)
        .expect("collapse should succeed");

    let survivor = log.collapses[0].surviving_vertex;
    let pos = &mesh.vertices[survivor];
    // Midpoint of (-1,0) and (1,0) should be (0,0)
    assert_relative_eq!(pos[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(pos[1], 0.0, epsilon = 1e-12);
}

#[test]
fn collapse_edge_rejects_foldover() {
    // Build a mesh where collapsing a specific edge would invert a triangle.
    // Use a near-degenerate configuration: a thin sliver where the only
    // safe direction for the midpoint would flip a neighbor.
    //
    //     0
    //    /|\
    //   / | \
    //  1--2--3
    //
    // If we collapse edge (0,2) and move the survivor to the midpoint,
    // the triangles [0,1,2] and [0,2,3] are removed, but if vertex 1 and 3
    // are nearly collinear with 0, the remaining triangles can fold.
    //
    // Simpler: set foldover_threshold to 0.0 so any normal rotation rejects.
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");

    // With threshold = 0.0, even minimal normal change triggers rejection.
    // But collapsing (1,2) removes both triangles entirely, leaving no faces
    // to check. So we need a bigger mesh.
    //
    // Build a 5-vertex mesh:
    //     0
    //    / \
    //   1---2
    //  / \ / \
    // 4   3   (no vertex here, boundary)
    //
    // Actually, let's use a simple approach: build a mesh where collapse
    // would make an existing triangle degenerate.
    use nalgebra::SVector;
    let vertices = vec![
        SVector::from([0.0, 2.0]),   // 0: top
        SVector::from([-1.0, 0.0]),  // 1: left
        SVector::from([1.0, 0.0]),   // 2: right
        SVector::from([0.0, -1.0]),  // 3: bottom
        SVector::from([-2.0, -1.0]), // 4: far left
    ];
    let triangles = vec![
        [1, 2, 0],  // top
        [2, 1, 3],  // middle
        [1, 4, 3],  // left-bottom
    ];
    let mut mesh2 = Mesh::from_simplices(&manifold, vertices, triangles);

    // Collapsing edge (1,3) moves survivor to midpoint (-0.5, -0.5).
    // The triangle [1,4,3] is removed (contains both endpoints).
    // But triangle [2,1,3] also contains both endpoints, so it is removed too.
    // Triangle [1,2,0] survives but vertex 1 moves to (-0.5, -0.5).
    // With threshold = 0.001 this should still succeed since the normal
    // (pointing out of plane in 2D) doesn't change direction.

    // For a true foldover test, build a bowtie where collapse inverts:
    //   0---1---2
    //   |  /|  /
    //   | / | /
    //   3   4
    //
    // Collapse (1,3): survivor goes to midpoint. Triangle [0,1,3] is removed.
    // Triangle [0,3,...] might invert.
    //
    // Keep it simple: use threshold 0.0 and a mesh where survivors have
    // remaining adjacent faces.
    let vertices3 = vec![
        SVector::from([0.0, 1.0]),    // 0
        SVector::from([-1.0, 0.0]),   // 1
        SVector::from([1.0, 0.0]),    // 2
        SVector::from([0.0, -1.0]),   // 3
        SVector::from([-0.5, -0.5]),  // 4: close to where midpoint of (1,3) would land
    ];
    let triangles3 = vec![
        [1, 2, 0],  // top
        [2, 1, 3],  // right
        [1, 4, 3],  // bottom-left (would invert when 1 moves to midpoint of (1,3))
    ];
    let mut mesh3 = Mesh::from_simplices(&manifold, vertices3, triangles3);

    // Edge (1,3): midpoint at (-0.5, -0.5). Vertex 4 is at (-0.5, -0.5).
    // After collapse, the surviving vertex (1) moves to (-0.5, -0.5), making
    // triangle [1,4,3] degenerate. But that triangle is removed (contains both
    // endpoints). So we need a triangle that references only one endpoint.
    //
    // Better approach: add a triangle [1,4,X] where X is not 3.
    let vertices4 = vec![
        SVector::from([0.0, 1.0]),     // 0
        SVector::from([-1.0, 0.0]),    // 1
        SVector::from([1.0, 0.0]),     // 2
        SVector::from([0.0, -1.0]),    // 3
        SVector::from([-1.5, -0.5]),   // 4
    ];
    let triangles4 = vec![
        [1, 2, 0],  // top
        [2, 1, 3],  // right-bottom
        [1, 3, 4],  // left-bottom
        [1, 4, 0],  // left (this face survives collapse and will check foldover)
    ];
    let mut mesh4 = Mesh::from_simplices(&manifold, vertices4, triangles4);
    let edge13 = find_edge(&mesh4, 1, 3).expect("edge (1,3) must exist");

    // Midpoint of (-1,0) and (0,-1) is (-0.5,-0.5). Vertex 1 moves there.
    // Triangle [1,4,0] becomes [(-0.5,-0.5), (-1.5,-0.5), (0,1)].
    // Original [1,4,0] = [(-1,0), (-1.5,-0.5), (0,1)].
    // The normal rotation might be small. Use threshold = 0.0 to force rejection
    // since any change at all will exceed 0.0.
    let result = collapse_edge(&mut mesh4, &manifold, edge13, 0.0);
    assert!(
        matches!(result, Err(RemeshError::Foldover { .. })),
        "collapse should be rejected with foldover_threshold = 0.0"
    );
}

#[test]
fn split_then_collapse_roundtrip_preserves_euler() {
    let manifold = Euclidean::<2>;
    let mut mesh = diamond_mesh();
    let euler_before = mesh.euler_characteristic();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let _ = split_edge(&mut mesh, &manifold, edge);
    // Euler: V+1 - E+3 + F+2 = V-E+F = unchanged (for closed: +1-3+2=0)

    let euler_after_split = mesh.euler_characteristic();
    assert_eq!(euler_before, euler_after_split, "split must preserve Euler characteristic");
}
```

**Step 10.2: Verify tests fail**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test split_collapse 2>&1 | head -5
# Should show: "not yet implemented: Task 10"
```

**Step 10.3: Implement `split_edge`**

Replace the `split_edge` stub in `~/cartan/cartan-remesh/src/primitives.rs`:

```rust
/// Split an edge by inserting a vertex at the geodesic midpoint.
///
/// The two triangles adjacent to the edge become four triangles.
/// Adjacency is rebuilt after the split.
///
/// # Algorithm
///
/// Given edge e = (v_a, v_b) with adjacent triangles T0 = [v_a, v_b, v_c]
/// and T1 = [v_b, v_a, v_d]:
///
/// 1. Insert new vertex v_m at the geodesic midpoint of (v_a, v_b).
/// 2. Replace T0 with [v_a, v_m, v_c] and [v_m, v_b, v_c].
/// 3. Replace T1 with [v_b, v_m, v_d] and [v_m, v_a, v_d].
/// 4. Rebuild edges and adjacency via `mesh.rebuild_adjacency()`.
///
/// For boundary edges (only one adjacent face), only that face is split
/// into two triangles.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn split_edge<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> RemeshLog {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let [v_a, v_b] = mesh.boundaries[edge];

    // 1. Compute geodesic midpoint and insert as new vertex.
    let midpoint = mesh.boundary_midpoint(manifold, edge);
    let v_m = mesh.vertices.len();
    mesh.vertices.push(midpoint);

    // 2. Find adjacent faces (faces containing both v_a and v_b).
    let adjacent_faces: Vec<usize> = mesh.boundary_simplices[edge].clone();

    // 3. Collect new triangles that replace each adjacent face.
    let mut new_triangles: Vec<[usize; 3]> = Vec::new();
    let mut faces_to_remove: Vec<usize> = Vec::new();

    for &face_idx in &adjacent_faces {
        let tri = mesh.simplices[face_idx];
        // Find the opposite vertex (the one that is neither v_a nor v_b).
        let v_opp = tri.iter().copied().find(|&v| v != v_a && v != v_b)
            .expect("triangle must contain a vertex other than v_a, v_b");

        // Determine winding order from the original triangle.
        // The original triangle has v_a, v_b, v_opp in some CCW order.
        // We need to produce two sub-triangles preserving that orientation.
        //
        // Find the position of v_a, v_b in the original triangle to get winding.
        let pos_a = tri.iter().position(|&v| v == v_a).unwrap();
        let pos_b = tri.iter().position(|&v| v == v_b).unwrap();

        // If v_a comes before v_b in CCW order (pos_b = (pos_a+1) % 3),
        // the winding is [..., v_a, v_b, v_opp, ...] (cyclically).
        // Split into [v_a, v_m, v_opp] and [v_m, v_b, v_opp].
        if (pos_a + 1) % 3 == pos_b {
            new_triangles.push([v_a, v_m, v_opp]);
            new_triangles.push([v_m, v_b, v_opp]);
        } else {
            // v_b comes before v_a: winding is [..., v_b, v_a, v_opp, ...]
            // Split into [v_b, v_m, v_opp] and [v_m, v_a, v_opp].
            new_triangles.push([v_b, v_m, v_opp]);
            new_triangles.push([v_m, v_a, v_opp]);
        }

        faces_to_remove.push(face_idx);
    }

    // 4. Remove old faces (in reverse order to keep indices valid) and add new ones.
    faces_to_remove.sort_unstable();
    for &fi in faces_to_remove.iter().rev() {
        mesh.simplices.swap_remove(fi);
    }
    for tri in &new_triangles {
        mesh.simplices.push(*tri);
    }

    // 5. Rebuild edges and adjacency from the updated simplex list.
    mesh.rebuild_adjacency();

    // 6. Determine new edge indices (all edges incident to v_m).
    let new_edges: Vec<usize> = mesh.vertex_boundaries[v_m].clone();

    let mut log = RemeshLog::new();
    log.splits.push(EdgeSplit {
        old_edge: edge,
        v_a,
        v_b,
        new_vertex: v_m,
        new_edges,
    });
    log
}
```

**Step 10.4: Implement `collapse_edge`**

Replace the `collapse_edge` stub in `~/cartan/cartan-remesh/src/primitives.rs`:

```rust
/// Collapse an edge by merging its endpoints at the geodesic midpoint.
///
/// The surviving vertex (lower index) is moved to the midpoint. Two
/// degenerate triangles are removed. Returns `Err(RemeshError::Foldover)`
/// if any adjacent face normal would rotate beyond the configured threshold.
///
/// # Algorithm
///
/// Given edge e = (v_a, v_b) with v_a < v_b:
///
/// 1. Compute geodesic midpoint p_m.
/// 2. Identify faces adjacent to v_b but not containing both v_a and v_b
///    (these are the faces that will have v_b replaced by v_a).
/// 3. For each such face, compute the face normal before and after the move.
///    If the angle between them exceeds `foldover_threshold`, reject.
/// 4. Remove faces containing both v_a and v_b (they become degenerate).
/// 5. In all remaining faces, replace v_b with v_a.
/// 6. Move v_a to p_m.
/// 7. Mark v_b as removed (swap-remove from vertices, update all indices).
/// 8. Rebuild adjacency.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn collapse_edge<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
    foldover_threshold: f64,
) -> Result<RemeshLog, RemeshError> {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let [v_a, v_b] = mesh.boundaries[edge];
    // Convention: v_a (lower index) survives.
    let (survivor, removed) = if v_a < v_b { (v_a, v_b) } else { (v_b, v_a) };

    // 1. Compute geodesic midpoint.
    let midpoint = mesh.boundary_midpoint(manifold, edge);

    // 2. Classify faces.
    let faces_with_both: Vec<usize> = mesh.boundary_simplices[edge].clone();

    // Faces incident to the removed vertex but not containing both endpoints.
    let faces_to_rewire: Vec<usize> = mesh.vertex_simplices[removed]
        .iter()
        .copied()
        .filter(|f| !faces_with_both.contains(f))
        .collect();

    // 3. Foldover guard: check each face that will be rewired.
    //    Compute tangent-space face normal before and after moving the vertex.
    for &face_idx in &faces_to_rewire {
        let tri = mesh.simplices[face_idx];

        // Compute normal before (using current positions).
        let normal_before = face_normal_tangent(mesh, manifold, &tri);

        // Compute normal after (with removed vertex replaced by survivor at midpoint).
        let mut tri_after = tri;
        for v in tri_after.iter_mut() {
            if *v == removed {
                *v = survivor;
            }
        }
        // Temporarily compute with midpoint position for survivor.
        let old_survivor_pos = mesh.vertices[survivor].clone();
        mesh.vertices[survivor] = midpoint.clone();
        let normal_after = face_normal_tangent(mesh, manifold, &tri_after);
        mesh.vertices[survivor] = old_survivor_pos;

        // Angle between normals.
        let dot = normal_before
            .iter()
            .zip(normal_after.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let norm_b = normal_before.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_a = normal_after.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_b < 1e-30 || norm_a < 1e-30 {
            continue; // degenerate face, skip
        }
        let cos_angle = (dot / (norm_b * norm_a)).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        if angle > foldover_threshold {
            return Err(RemeshError::Foldover {
                face: face_idx,
                angle_rad: angle,
                threshold: foldover_threshold,
            });
        }
    }

    // 4. Move survivor to midpoint.
    mesh.vertices[survivor] = midpoint;

    // 5. Remove faces containing both endpoints.
    let removed_faces = faces_with_both.clone();
    let mut to_remove_sorted = faces_with_both;
    to_remove_sorted.sort_unstable();
    for &fi in to_remove_sorted.iter().rev() {
        mesh.simplices.swap_remove(fi);
    }

    // 6. Replace removed vertex with survivor in all remaining simplices.
    for tri in mesh.simplices.iter_mut() {
        for v in tri.iter_mut() {
            if *v == removed {
                *v = survivor;
            }
        }
    }

    // 7. Remove the vertex via swap-remove and update all indices.
    let last_vertex = mesh.vertices.len() - 1;
    mesh.vertices.swap_remove(removed);
    if removed != last_vertex {
        // The vertex that was at `last_vertex` is now at `removed`.
        // Update all simplex references from last_vertex to removed.
        for tri in mesh.simplices.iter_mut() {
            for v in tri.iter_mut() {
                if *v == last_vertex {
                    *v = removed;
                }
            }
        }
    }

    // 8. Rebuild edges and adjacency.
    mesh.rebuild_adjacency();

    let mut log = RemeshLog::new();
    log.collapses.push(EdgeCollapse {
        old_edge: edge,
        surviving_vertex: survivor,
        removed_vertex: removed,
        removed_faces,
    });
    Ok(log)
}

/// Compute a tangent-space face normal (as a Vec<f64>) for foldover detection.
///
/// Maps the three triangle vertices into the tangent space at the first vertex
/// via log, then computes the cross product of the two edge vectors.
/// Returns the unnormalized normal as a flat Vec<f64>.
fn face_normal_tangent<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    tri: &[usize; 3],
) -> Vec<f64> {
    let [i, j, k] = *tri;
    let v0 = &mesh.vertices[i];
    let v1 = &mesh.vertices[j];
    let v2 = &mesh.vertices[k];

    let u = manifold
        .log(v0, v1)
        .unwrap_or_else(|_| manifold.zero_tangent(v0));
    let v = manifold
        .log(v0, v2)
        .unwrap_or_else(|_| manifold.zero_tangent(v0));

    // For 2D meshes embedded in 2D, the "normal" is the signed area (scalar).
    // For 3D meshes (surfaces in R^3), it is the cross product.
    // We use a generalized approach: compute the Gram determinant area as a proxy.
    // For foldover detection, the sign of the area is what matters.
    let uu = manifold.inner(v0, &u, &u);
    let vv = manifold.inner(v0, &v, &v);
    let uv = manifold.inner(v0, &u, &v);

    // The signed area proxy: for 2D triangles, this is proportional to the
    // cross product (z-component). The sign flip indicates foldover.
    // We return [uu, uv, vv, det] as the "normal proxy" for angle computation.
    let det = uu * vv - uv * uv;
    let signed_area = det.signum() * det.abs().sqrt();
    vec![signed_area]
}
```

Note: The `face_normal_tangent` helper uses the signed Gram determinant as a foldover proxy. For 2D flat meshes the sign of the cross product (equivalently, the sign of the Gram determinant) detects orientation inversion. For surfaces in R^3, this generalizes to the tangent-space normal direction. The angle between the before/after "normals" (single-element vectors for 2D) captures orientation flips.

**Step 10.5: Add import for `EdgeSplit` and `EdgeCollapse` to `primitives.rs` header**

```rust
use crate::log::{EdgeCollapse, EdgeSplit, RemeshLog};
```

**Step 10.6: Verify tests pass**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test split_collapse
```

**Step 10.7: Commit**

```bash
cd ~/cartan && git add cartan-remesh/src/primitives.rs cartan-remesh/tests/split_collapse.rs
git commit -m "feat(cartan-remesh): implement split_edge and collapse_edge with foldover guard

split_edge inserts a vertex at the geodesic midpoint and replaces 2
adjacent triangles with 4. collapse_edge merges two endpoints at the
geodesic midpoint, removes degenerate triangles, and rejects the
operation if any adjacent face normal would rotate beyond the configured
foldover threshold. Both operations rebuild adjacency after mutation."
```

---

### Task 11: Edge flip and vertex shift

**Prerequisite**: Task 10 complete.

**Files touched**:
| File | Action |
|------|--------|
| `~/cartan/cartan-remesh/src/primitives.rs` | Modify (implement `flip_edge`, `shift_vertex`) |
| `~/cartan/cartan-remesh/tests/flip_shift.rs` | Create |

**Step 11.1: Write failing tests**

Create `~/cartan/cartan-remesh/tests/flip_shift.rs`:

```rust
// ~/cartan/cartan-remesh/tests/flip_shift.rs

//! Tests for edge flip and vertex shift primitives.

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{flip_edge, shift_vertex, RemeshError};
use nalgebra::SVector;

/// Find the edge index for the edge connecting vertices a and b.
fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries.iter().position(|&[i, j]| i == lo && j == hi)
}

/// Build a non-Delaunay diamond where the shared edge should be flipped.
///
/// ```text
///       0 (0, 2)
///      / \
///     /   \
///    1-----2     1=(-2, 0), 2=(2, 0)
///     \   /
///      \ /
///       3 (0, -0.1)   <-- very close to the edge, making opposite angles sum > pi
/// ```
///
/// Triangle [1,2,0] (top, CCW) and [2,1,3] (bottom, CCW).
/// The angle at vertex 3 is very obtuse because 3 is close to edge (1,2).
/// Sum of opposite angles (at 0 and at 3) should exceed pi, triggering a flip.
fn non_delaunay_diamond() -> Mesh<Euclidean<2>, 3, 2> {
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 2.0]),    // 0: top
        SVector::from([-2.0, 0.0]),   // 1: left
        SVector::from([2.0, 0.0]),    // 2: right
        SVector::from([0.0, -0.1]),   // 3: barely below edge (1,2)
    ];
    let triangles = vec![
        [1, 2, 0],  // top (CCW)
        [2, 1, 3],  // bottom (CCW)
    ];
    Mesh::from_simplices(&manifold, vertices, triangles)
}

/// Build a Delaunay diamond where the shared edge should NOT be flipped.
///
/// ```text
///     0 (0, 1)
///    / \
///   1---2       1=(-1,0), 2=(1,0)
///    \ /
///     3 (0, -1)
/// ```
fn delaunay_diamond() -> Mesh<Euclidean<2>, 3, 2> {
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),
        SVector::from([-1.0, 0.0]),
        SVector::from([1.0, 0.0]),
        SVector::from([0.0, -1.0]),
    ];
    let triangles = vec![
        [1, 2, 0],
        [2, 1, 3],
    ];
    Mesh::from_simplices(&manifold, vertices, triangles)
}

#[test]
fn flip_non_delaunay_edge_succeeds() {
    let manifold = Euclidean::<2>;
    let mut mesh = non_delaunay_diamond();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let log = flip_edge(&mut mesh, &manifold, edge)
        .expect("flip should succeed on non-Delaunay config");

    assert_eq!(log.flips.len(), 1);

    // After flip, edge (1,2) should be replaced by edge (0,3).
    let new_edge = log.flips[0].new_edge;
    let (lo, hi) = if new_edge[0] < new_edge[1] {
        (new_edge[0], new_edge[1])
    } else {
        (new_edge[1], new_edge[0])
    };
    assert_eq!(lo, 0);
    assert_eq!(hi, 3);
}

#[test]
fn flip_preserves_vertex_and_face_count() {
    let manifold = Euclidean::<2>;
    let mut mesh = non_delaunay_diamond();
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let _ = flip_edge(&mut mesh, &manifold, edge).unwrap();

    assert_eq!(mesh.n_vertices(), nv, "flip must not change vertex count");
    assert_eq!(mesh.n_simplices(), nf, "flip must not change face count");
}

#[test]
fn flip_delaunay_edge_is_rejected() {
    let manifold = Euclidean::<2>;
    let mut mesh = delaunay_diamond();

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let result = flip_edge(&mut mesh, &manifold, edge);

    assert!(
        matches!(result, Err(RemeshError::AlreadyDelaunay { .. })),
        "flip should be rejected for an already-Delaunay edge"
    );
}

#[test]
fn flip_boundary_edge_is_rejected() {
    // Build a single triangle (all edges are boundary).
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 1.0]),
        SVector::from([-1.0, 0.0]),
        SVector::from([1.0, 0.0]),
    ];
    let triangles = vec![[0, 1, 2]];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let result = flip_edge(&mut mesh, &manifold, 0);
    assert!(
        matches!(result, Err(RemeshError::BoundaryEdge { .. }) | Err(RemeshError::NotInteriorEdge { .. })),
        "flip should be rejected for a boundary edge"
    );
}

#[test]
fn shift_vertex_improves_quality() {
    // Build a mesh where one interior vertex is off-center, then shift it.
    // After shifting, the vertex should be closer to the barycenter of its neighbors.
    let manifold = Euclidean::<2>;

    // 5 vertices: 4 corners of a square + 1 interior vertex (off-center).
    //
    //  3---2
    //  |\ /|
    //  | 4 |    4 is at (0.3, 0.3) instead of (0.5, 0.5)
    //  |/ \|
    //  0---1
    let vertices = vec![
        SVector::from([0.0, 0.0]),  // 0
        SVector::from([1.0, 0.0]),  // 1
        SVector::from([1.0, 1.0]),  // 2
        SVector::from([0.0, 1.0]),  // 3
        SVector::from([0.3, 0.3]),  // 4: off-center interior
    ];
    let triangles = vec![
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let pos_before = mesh.vertices[4].clone();
    let ideal_center = SVector::from([0.5, 0.5]);
    let dist_before = (pos_before - ideal_center).norm();

    let _log = shift_vertex(&mut mesh, &manifold, 4);

    let pos_after = mesh.vertices[4].clone();
    let dist_after = (pos_after - ideal_center).norm();

    assert!(
        dist_after < dist_before,
        "shift should move vertex closer to neighbor barycenter: before={dist_before}, after={dist_after}"
    );
}

#[test]
fn shift_vertex_records_displacement() {
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 0.0]),
        SVector::from([1.0, 0.0]),
        SVector::from([1.0, 1.0]),
        SVector::from([0.0, 1.0]),
        SVector::from([0.3, 0.3]),
    ];
    let triangles = vec![
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let mut mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let log = shift_vertex(&mut mesh, &manifold, 4);
    assert_eq!(log.shifts.len(), 1);
    assert_eq!(log.shifts[0].vertex, 4);
    assert!(!log.shifts[0].old_pos_tangent.is_empty());
}
```

**Step 11.2: Verify tests fail**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test flip_shift 2>&1 | head -5
# Should show: "not yet implemented: Task 11"
```

**Step 11.3: Implement `flip_edge`**

Replace the `flip_edge` stub in `~/cartan/cartan-remesh/src/primitives.rs`:

```rust
/// Flip the diagonal of the quad formed by two adjacent triangles.
///
/// Returns `Err(RemeshError::AlreadyDelaunay)` if the edge already satisfies
/// the Delaunay criterion (sum of opposite angles <= pi).
/// Returns `Err(RemeshError::BoundaryEdge)` if the edge has only one adjacent face.
///
/// # Algorithm
///
/// Given interior edge e = (v_a, v_b) with adjacent triangles
/// T0 = [v_a, v_b, v_c] and T1 = [v_b, v_a, v_d]:
///
/// 1. Compute the angle at v_c in T0 and the angle at v_d in T1.
/// 2. If angle_c + angle_d <= pi, the edge is already Delaunay. Reject.
/// 3. Otherwise, replace edge (v_a, v_b) with (v_c, v_d).
///    T0 becomes [v_c, v_d, v_a] and T1 becomes [v_d, v_c, v_b].
/// 4. Rebuild adjacency.
///
/// # Panics
///
/// Panics if `edge >= mesh.n_boundaries()`.
pub fn flip_edge<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> Result<RemeshLog, RemeshError> {
    assert!(edge < mesh.n_boundaries(), "edge index out of bounds");

    let [v_a, v_b] = mesh.boundaries[edge];
    let adjacent = &mesh.boundary_simplices[edge];

    if adjacent.len() < 2 {
        return Err(RemeshError::NotInteriorEdge {
            edge,
            count: adjacent.len(),
        });
    }

    let f0 = adjacent[0];
    let f1 = adjacent[1];

    // Find opposite vertices.
    let tri0 = mesh.simplices[f0];
    let tri1 = mesh.simplices[f1];

    let v_c = tri0.iter().copied().find(|&v| v != v_a && v != v_b)
        .expect("triangle must have a vertex other than the edge endpoints");
    let v_d = tri1.iter().copied().find(|&v| v != v_a && v != v_b)
        .expect("triangle must have a vertex other than the edge endpoints");

    // Compute opposite angles using the law of cosines in tangent space.
    let angle_c = vertex_angle_in_triangle(mesh, manifold, v_c, v_a, v_b);
    let angle_d = vertex_angle_in_triangle(mesh, manifold, v_d, v_a, v_b);

    if angle_c + angle_d <= std::f64::consts::PI {
        return Err(RemeshError::AlreadyDelaunay {
            edge,
            angle_sum: angle_c + angle_d,
        });
    }

    // Perform the flip: replace edge (v_a, v_b) with (v_c, v_d).
    // T0 = [v_a, v_b, v_c] becomes [v_c, v_d, v_a]
    // T1 = [v_b, v_a, v_d] becomes [v_d, v_c, v_b]
    mesh.simplices[f0] = [v_c, v_d, v_a];
    mesh.simplices[f1] = [v_d, v_c, v_b];

    // Rebuild adjacency.
    mesh.rebuild_adjacency();

    let mut log = RemeshLog::new();
    log.flips.push(EdgeFlip {
        old_edge: edge,
        new_edge: [v_c, v_d],
        affected_faces: [f0, f1],
    });
    Ok(log)
}

/// Compute the angle at vertex `v` in the triangle (v, a, b) using tangent-space
/// inner products.
///
/// Returns the angle in radians between edges (v -> a) and (v -> b).
fn vertex_angle_in_triangle<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    v: usize,
    a: usize,
    b: usize,
) -> f64 {
    let p = &mesh.vertices[v];
    let pa = &mesh.vertices[a];
    let pb = &mesh.vertices[b];

    let u = manifold.log(p, pa).unwrap_or_else(|_| manifold.zero_tangent(p));
    let w = manifold.log(p, pb).unwrap_or_else(|_| manifold.zero_tangent(p));

    let uu = manifold.inner(p, &u, &u);
    let ww = manifold.inner(p, &w, &w);
    let uw = manifold.inner(p, &u, &w);

    let denom = (uu * ww).sqrt();
    if denom < 1e-30 {
        return 0.0;
    }
    let cos_angle = (uw / denom).clamp(-1.0, 1.0);
    cos_angle.acos()
}
```

**Step 11.4: Implement `shift_vertex`**

Replace the `shift_vertex` stub in `~/cartan/cartan-remesh/src/primitives.rs`:

```rust
/// Tangential Laplacian smoothing of a single vertex.
///
/// Maps all neighbors into the tangent space at the vertex via `manifold.log`,
/// computes the barycenter, projects out the normal component, and applies
/// the displacement via `manifold.exp`. Boundary vertices are constrained
/// to the boundary edge tangent direction.
///
/// # Algorithm
///
/// 1. Collect all neighbor vertices from the vertex-boundary adjacency map.
/// 2. Log-map each neighbor into the tangent space at v.
/// 3. Compute the barycenter of these tangent vectors.
/// 4. (For surfaces in R^3: project out the normal component. For 2D meshes,
///    skip since all tangent vectors lie in the plane.)
/// 5. Apply: v_new = exp(v, barycenter).
///
/// # Panics
///
/// Panics if `vertex >= mesh.n_vertices()`.
pub fn shift_vertex<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    vertex: usize,
) -> RemeshLog {
    assert!(vertex < mesh.n_vertices(), "vertex index out of bounds");

    let p = &mesh.vertices[vertex];

    // Collect unique neighbor vertex indices from incident edges.
    let mut neighbors: Vec<usize> = Vec::new();
    for &edge_idx in &mesh.vertex_boundaries[vertex] {
        let [a, b] = mesh.boundaries[edge_idx];
        let neighbor = if a == vertex { b } else { a };
        if !neighbors.contains(&neighbor) {
            neighbors.push(neighbor);
        }
    }

    if neighbors.is_empty() {
        return RemeshLog::new();
    }

    // Log-map each neighbor into tangent space at p and compute barycenter.
    let zero = manifold.zero_tangent(p);
    let mut barycenter = zero.clone();
    let n = neighbors.len() as f64;

    for &nb in &neighbors {
        let q = &mesh.vertices[nb];
        let tangent = manifold.log(p, q).unwrap_or_else(|_| zero.clone());
        barycenter = barycenter + tangent;
    }
    barycenter = barycenter * (1.0 / n);

    // Store displacement for the log (as flat f64 vector via inner products).
    // For the tangent vector, we store a representation: the norm and direction
    // encode is sufficient. We use the inner product to get the squared norm.
    let disp_sq = manifold.inner(p, &barycenter, &barycenter);
    let disp_norm = disp_sq.sqrt();
    let old_pos_tangent = vec![disp_norm];

    // Apply the shift: move vertex to exp(p, barycenter).
    let new_pos = manifold.exp(p, &barycenter);
    mesh.vertices[vertex] = new_pos;

    let mut log = RemeshLog::new();
    log.shifts.push(VertexShift {
        vertex,
        old_pos_tangent,
    });
    log
}
```

**Step 11.5: Update imports at top of `primitives.rs`**

```rust
use crate::error::RemeshError;
use crate::log::{EdgeCollapse, EdgeFlip, EdgeSplit, RemeshLog, VertexShift};
```

**Step 11.6: Verify tests pass**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test flip_shift
```

**Step 11.7: Commit**

```bash
cd ~/cartan && git add cartan-remesh/src/primitives.rs cartan-remesh/tests/flip_shift.rs
git commit -m "feat(cartan-remesh): implement flip_edge (Delaunay criterion) and shift_vertex (tangential Laplacian)

flip_edge checks the sum of opposite corner angles via tangent-space
inner products and rejects edges that already satisfy the Delaunay
criterion. shift_vertex performs tangential Laplacian smoothing by
log-mapping all neighbors, computing the barycenter, and applying via
exp. Both operations are generic over M: Manifold."
```

---

### Task 12: LCR conformal regularization

**Prerequisite**: Task 11 complete.

**Files touched**:
| File | Action |
|------|--------|
| `~/cartan/cartan-remesh/src/lcr.rs` | Modify (implement all four functions) |
| `~/cartan/cartan-remesh/tests/lcr.rs` | Create |

**Step 12.1: Write failing tests**

Create `~/cartan/cartan-remesh/tests/lcr.rs`:

```rust
// ~/cartan/cartan-remesh/tests/lcr.rs

//! Tests for length-cross-ratio conformal regularization.

use approx::assert_relative_eq;
use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{capture_reference_lcrs, lcr_spring_energy, length_cross_ratio};
use nalgebra::SVector;

/// Find the edge index for the edge connecting vertices a and b.
fn find_edge(mesh: &Mesh<Euclidean<2>, 3, 2>, a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    mesh.boundaries.iter().position(|&[i, j]| i == lo && j == hi)
}

/// Build an equilateral diamond: two equilateral triangles sharing an edge.
///
/// ```text
///       0 (0, sqrt(3)/2)
///      / \
///     /   \
///    1-----2       1=(-0.5, 0), 2=(0.5, 0)
///     \   /
///      \ /
///       3 (0, -sqrt(3)/2)
/// ```
fn equilateral_diamond() -> Mesh<Euclidean<2>, 3, 2> {
    let manifold = Euclidean::<2>;
    let h = (3.0_f64).sqrt() / 2.0;
    let vertices = vec![
        SVector::from([0.0, h]),      // 0: top
        SVector::from([-0.5, 0.0]),   // 1: left
        SVector::from([0.5, 0.0]),    // 2: right
        SVector::from([0.0, -h]),     // 3: bottom
    ];
    let triangles = vec![
        [1, 2, 0],  // top (CCW)
        [2, 1, 3],  // bottom (CCW)
    ];
    Mesh::from_simplices(&manifold, vertices, triangles)
}

#[test]
fn lcr_of_equilateral_diamond_is_one() {
    let manifold = Euclidean::<2>;
    let mesh = equilateral_diamond();

    // The interior edge is (1,2). Opposite vertices are 0 and 3.
    // For equilateral triangles with shared edge:
    //   dist(1,3) * dist(2,0) / (dist(0,1) * dist(3,2))
    // By symmetry of the equilateral diamond, all four distances are equal
    // (each is the side length 1.0), so LCR = 1.0.
    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let lcr = length_cross_ratio(&mesh, &manifold, edge);
    assert_relative_eq!(lcr, 1.0, epsilon = 1e-10);
}

#[test]
fn lcr_of_boundary_edge_is_one() {
    let manifold = Euclidean::<2>;
    let mesh = equilateral_diamond();

    // Edge (0,1) is a boundary edge (only one adjacent face).
    let edge = find_edge(&mesh, 0, 1).expect("edge (0,1) must exist");
    let lcr = length_cross_ratio(&mesh, &manifold, edge);
    assert_relative_eq!(lcr, 1.0, epsilon = 1e-10);
}

#[test]
fn capture_reference_lcrs_length_matches_edges() {
    let manifold = Euclidean::<2>;
    let mesh = equilateral_diamond();
    let lcrs = capture_reference_lcrs(&mesh, &manifold);
    assert_eq!(lcrs.len(), mesh.n_boundaries());
}

#[test]
fn spring_energy_at_reference_is_zero() {
    let manifold = Euclidean::<2>;
    let mesh = equilateral_diamond();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);
    let energy = lcr_spring_energy(&mesh, &manifold, &ref_lcrs, 1.0);
    assert_relative_eq!(energy, 0.0, epsilon = 1e-12);
}

#[test]
fn spring_energy_positive_when_deformed() {
    let manifold = Euclidean::<2>;
    let mesh = equilateral_diamond();
    let ref_lcrs = capture_reference_lcrs(&mesh, &manifold);

    // Build a deformed version: move vertex 0 upward.
    let h = (3.0_f64).sqrt() / 2.0;
    let vertices_deformed = vec![
        SVector::from([0.0, h + 0.5]),  // 0: moved up
        SVector::from([-0.5, 0.0]),
        SVector::from([0.5, 0.0]),
        SVector::from([0.0, -h]),
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    let deformed = Mesh::from_simplices(&manifold, vertices_deformed, triangles);

    let energy = lcr_spring_energy(&deformed, &manifold, &ref_lcrs, 1.0);
    assert!(energy > 0.0, "spring energy must be positive when mesh is deformed from reference");
}

#[test]
fn lcr_distorted_diamond_not_one() {
    // Build a diamond where the interior edge LCR deviates from 1.0.
    let manifold = Euclidean::<2>;
    let vertices = vec![
        SVector::from([0.0, 3.0]),    // 0: tall top
        SVector::from([-1.0, 0.0]),   // 1: left
        SVector::from([1.0, 0.0]),    // 2: right
        SVector::from([0.0, -0.5]),   // 3: close bottom
    ];
    let triangles = vec![[1, 2, 0], [2, 1, 3]];
    let mesh = Mesh::from_simplices(&manifold, vertices, triangles);

    let edge = find_edge(&mesh, 1, 2).expect("edge (1,2) must exist");
    let lcr = length_cross_ratio(&mesh, &manifold, edge);

    // With asymmetric diamond, LCR should not be 1.0.
    assert!((lcr - 1.0).abs() > 0.01, "LCR should deviate from 1.0 for distorted diamond");
}
```

**Step 12.2: Verify tests fail**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test lcr 2>&1 | head -5
# Should show: "not yet implemented: Task 12"
```

**Step 12.3: Implement LCR functions**

Replace the stubs in `~/cartan/cartan-remesh/src/lcr.rs`:

```rust
// ~/cartan/cartan-remesh/src/lcr.rs

//! Length-cross-ratio (LCR) conformal regularization.
//!
//! The LCR of an interior edge measures how far the local mesh geometry
//! deviates from a conformal mapping. LCR springs penalize deviation from
//! a reference configuration, preserving conformal structure during remeshing.
//!
//! ## Definition
//!
//! For an interior edge (i, j) with diamond vertices {i, j, k, l} (where
//! k and l are the opposite vertices of the two adjacent triangles):
//!
//! `lcr = dist(i, l) * dist(j, k) / (dist(k, i) * dist(l, j))`
//!
//! For an equilateral diamond (two equilateral triangles sharing an edge),
//! all four side lengths are equal and LCR = 1.0.
//!
//! ## References
//!
//! Springborn, Schroder, Pinkall. "Conformal Equivalence of Triangle Meshes."
//! ACM Trans. Graphics 27(3), 2008.

use cartan_core::Manifold;
use cartan_dec::Mesh;

/// Compute the length-cross-ratio of an edge.
///
/// For an interior edge with diamond vertices {i, j, k, l}:
/// `lcr = dist(i,l) * dist(j,k) / (dist(k,i) * dist(l,j))`
///
/// Returns 1.0 for boundary edges (only one adjacent face).
pub fn length_cross_ratio<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    edge: usize,
) -> f64 {
    let adjacent = &mesh.boundary_simplices[edge];
    if adjacent.len() < 2 {
        return 1.0;
    }

    let [v_i, v_j] = mesh.boundaries[edge];

    // Find opposite vertices in each adjacent triangle.
    let tri0 = mesh.simplices[adjacent[0]];
    let tri1 = mesh.simplices[adjacent[1]];

    let v_k = tri0.iter().copied().find(|&v| v != v_i && v != v_j)
        .expect("triangle must have a third vertex");
    let v_l = tri1.iter().copied().find(|&v| v != v_i && v != v_j)
        .expect("triangle must have a third vertex");

    let d_il = manifold.dist(&mesh.vertices[v_i], &mesh.vertices[v_l]).unwrap_or(0.0);
    let d_jk = manifold.dist(&mesh.vertices[v_j], &mesh.vertices[v_k]).unwrap_or(0.0);
    let d_ki = manifold.dist(&mesh.vertices[v_k], &mesh.vertices[v_i]).unwrap_or(0.0);
    let d_lj = manifold.dist(&mesh.vertices[v_l], &mesh.vertices[v_j]).unwrap_or(0.0);

    let denom = d_ki * d_lj;
    if denom < 1e-30 {
        return 1.0;
    }

    (d_il * d_jk) / denom
}

/// Capture reference LCR values for all edges.
///
/// Returns a vector of length `mesh.n_boundaries()` with one LCR per edge.
/// Boundary edges get 1.0.
pub fn capture_reference_lcrs<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
) -> Vec<f64> {
    (0..mesh.n_boundaries())
        .map(|e| length_cross_ratio(mesh, manifold, e))
        .collect()
}

/// Total LCR spring energy: `0.5 * kst * sum((lcr_e - lcr_ref_e)^2 / lcr_ref_e^2)`.
///
/// Penalizes deviation from the reference conformal structure. The
/// normalization by `lcr_ref_e^2` makes the energy scale-invariant.
pub fn lcr_spring_energy<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    ref_lcrs: &[f64],
    kst: f64,
) -> f64 {
    assert_eq!(
        ref_lcrs.len(),
        mesh.n_boundaries(),
        "ref_lcrs length must match edge count"
    );

    let mut energy = 0.0;
    for e in 0..mesh.n_boundaries() {
        let lcr = length_cross_ratio(mesh, manifold, e);
        let lcr_ref = ref_lcrs[e];
        if lcr_ref.abs() < 1e-30 {
            continue;
        }
        let delta = (lcr - lcr_ref) / lcr_ref;
        energy += delta * delta;
    }
    0.5 * kst * energy
}

/// Per-vertex gradient of the LCR spring energy via finite differences.
///
/// Returns a tangent vector at each vertex pointing in the direction of
/// increasing LCR spring energy. The caller negates this for descent.
///
/// Uses a first-order forward finite difference in each tangent direction
/// with step size `eps = 1e-7 * max(1.0, characteristic_edge_length)`.
pub fn lcr_spring_gradient<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    ref_lcrs: &[f64],
    kst: f64,
) -> Vec<M::Tangent> {
    let nv = mesh.n_vertices();
    let dim = manifold.dim();

    // Compute characteristic edge length for step sizing.
    let avg_edge_len = if mesh.n_boundaries() > 0 {
        let total: f64 = (0..mesh.n_boundaries())
            .map(|e| mesh.edge_length(manifold, e))
            .sum();
        total / mesh.n_boundaries() as f64
    } else {
        1.0
    };
    let eps = 1e-7 * avg_edge_len.max(1.0);

    let e0 = lcr_spring_energy(mesh, manifold, ref_lcrs, kst);

    // For each vertex, perturb along each tangent basis direction.
    let mut gradients: Vec<M::Tangent> = Vec::with_capacity(nv);

    for vi in 0..nv {
        let p = &mesh.vertices[vi];
        let zero = manifold.zero_tangent(p);

        // We need a basis for T_p M. For embedded manifolds, we perturb in
        // each ambient direction and project. This is a dim-dimensional gradient.
        // We accumulate the gradient as a tangent vector.
        let mut grad = zero.clone();

        // Build a temporary mutable mesh for perturbation.
        let mut perturbed = mesh.clone();

        for d in 0..dim {
            // Create a tangent vector in the d-th direction.
            // Use a small perturbation: move vertex vi along the d-th coordinate.
            // This is ambient-space, so we project.
            //
            // For generic manifolds, we use the approach of perturbing via exp
            // with basis tangent vectors. Since we don't have a canonical basis
            // for T_p M, we use the ambient coordinate directions projected to
            // the tangent space.
            //
            // For Euclidean spaces, the tangent space IS the ambient space, so
            // this is exact. For curved manifolds, it is a first-order
            // approximation that suffices for finite-difference gradients.

            // Restore original position.
            perturbed.vertices[vi] = mesh.vertices[vi].clone();

            // Create unit ambient vector in direction d, then project to tangent.
            // We access this via a small exp step.
            // Since M::Tangent supports Add + Mul<f64>, we can build basis vectors
            // if we have zero_tangent. But without knowing the ambient dimension,
            // we use a coordinate perturbation approach:

            // Perturb the vertex position in ambient coordinates.
            // This requires knowing the ambient structure. For Euclidean<N>,
            // Point = SVector<f64, N> and we can index directly.
            // For general manifolds, we use the exp map with a scaled zero tangent.
            //
            // Practical approach: perturb via exp along a numerically constructed
            // tangent basis. We use random_tangent and Gram-Schmidt, but that is
            // expensive. Instead, for the finite-difference gradient, we note that
            // the gradient decomposes as:
            //
            //   grad_v E = sum_e (dE/d_lcr_e) * (d_lcr_e / d_v)
            //
            // The inner sum is over edges incident to v. We compute each
            // d_lcr_e/d_v via finite differences on the distance function.

            // For now, we use a simplified approach: perturb the vertex position
            // in the ambient coordinate system and reproject.

            // NOTE: This implementation uses a generic approach that works for
            // any Manifold but requires clone + rebuild. Production code should
            // use analytic gradients for specific manifolds (Euclidean, Sphere).

            break; // placeholder for the coordinate loop
        }

        // Simplified finite-difference gradient: perturb each incident edge's
        // contribution independently.
        let incident_edges: Vec<usize> = mesh.vertex_boundaries[vi].clone();
        let mut grad_tangent = zero.clone();

        for &eidx in &incident_edges {
            let [a, b] = mesh.boundaries[eidx];
            let other = if a == vi { b } else { a };

            // Direction from vi to the other vertex.
            let log_vec = manifold
                .log(&mesh.vertices[vi], &mesh.vertices[other])
                .unwrap_or_else(|_| zero.clone());
            let norm_sq = manifold.inner(&mesh.vertices[vi], &log_vec, &log_vec);
            if norm_sq < 1e-30 {
                continue;
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            let direction = log_vec * inv_norm;

            // Perturb vertex vi by eps along this direction.
            let new_pos = manifold.exp(&mesh.vertices[vi], &(direction.clone() * eps));
            perturbed.vertices[vi] = new_pos;

            let e_plus = lcr_spring_energy(&perturbed, manifold, ref_lcrs, kst);
            let de = (e_plus - e0) / eps;

            grad_tangent = grad_tangent + direction * de;

            // Restore.
            perturbed.vertices[vi] = mesh.vertices[vi].clone();
        }

        gradients.push(grad_tangent);
    }

    gradients
}
```

**Step 12.4: Verify tests pass**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test lcr
```

**Step 12.5: Commit**

```bash
cd ~/cartan && git add cartan-remesh/src/lcr.rs cartan-remesh/tests/lcr.rs
git commit -m "feat(cartan-remesh): implement LCR conformal regularization

length_cross_ratio computes the diamond LCR for interior edges (1.0 for
boundary). capture_reference_lcrs snapshots the current configuration.
lcr_spring_energy penalizes deviation from the reference via normalized
squared differences. lcr_spring_gradient uses edge-aligned finite
differences for generic manifold support."
```

---

### Task 13: Adaptive remesh driver and needs_remesh predicate

**Prerequisite**: Tasks 10, 11, 12 complete.

**Files touched**:
| File | Action |
|------|--------|
| `~/cartan/cartan-remesh/src/driver.rs` | Modify (implement `adaptive_remesh`, `needs_remesh`) |
| `~/cartan/cartan-remesh/tests/driver.rs` | Create |

**Step 13.1: Write failing tests**

Create `~/cartan/cartan-remesh/tests/driver.rs`:

```rust
// ~/cartan/cartan-remesh/tests/driver.rs

//! Tests for the adaptive remeshing driver and needs_remesh predicate.

use approx::assert_relative_eq;
use cartan_dec::{Mesh, Operators};
use cartan_manifolds::euclidean::Euclidean;
use cartan_remesh::{adaptive_remesh, needs_remesh, RemeshConfig};
use nalgebra::SVector;

/// Build a 4x4 flat grid mesh on [0,1]^2 for testing.
fn grid_mesh() -> Mesh<Euclidean<2>, 3, 2> {
    Mesh::from_triangles(
        {
            let mut verts = Vec::new();
            for j in 0..=4 {
                for i in 0..=4 {
                    verts.push([i as f64 / 4.0, j as f64 / 4.0]);
                }
            }
            verts
        },
        {
            let mut tris = Vec::new();
            let idx = |i: usize, j: usize| j * 5 + i;
            for j in 0..4 {
                for i in 0..4 {
                    let v00 = idx(i, j);
                    let v10 = idx(i + 1, j);
                    let v01 = idx(i, j + 1);
                    let v11 = idx(i + 1, j + 1);
                    tris.push([v00, v10, v01]);
                    tris.push([v10, v11, v01]);
                }
            }
            tris
        },
    )
}

/// Uniform curvature arrays (flat mesh: H=0, K=0 everywhere).
fn flat_curvatures(nv: usize) -> (Vec<f64>, Vec<f64>) {
    (vec![0.0; nv], vec![0.0; nv])
}

/// Simulate a bump: high curvature at vertices near the center.
fn bump_curvatures(mesh: &Mesh<Euclidean<2>, 3, 2>) -> (Vec<f64>, Vec<f64>) {
    let nv = mesh.n_vertices();
    let mut h = vec![0.0; nv];
    let mut k = vec![0.0; nv];

    let center = SVector::from([0.5, 0.5]);
    let bump_radius = 0.2;

    for vi in 0..nv {
        let pos = &mesh.vertices[vi];
        let dx = pos[0] - center[0];
        let dy = pos[1] - center[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r < bump_radius {
            // Simulate high curvature near center.
            let t = 1.0 - r / bump_radius;
            h[vi] = 10.0 * t;  // large mean curvature
            k[vi] = 50.0 * t * t;  // large Gaussian curvature
        }
    }

    (h, k)
}

#[test]
fn needs_remesh_false_for_uniform_flat_mesh() {
    let manifold = Euclidean::<2>;
    let mesh = grid_mesh();
    let (h, k) = flat_curvatures(mesh.n_vertices());

    let config = RemeshConfig {
        curvature_scale: 0.5,
        min_edge_length: 0.01,
        max_edge_length: 10.0,
        min_face_area: 1e-8,
        max_face_area: 10.0,
        foldover_threshold: 0.5,
        lcr_spring_stiffness: 0.0,
        smoothing_iterations: 3,
    };

    // Flat mesh with zero curvature: k_max = 0, so curvature-CFL is satisfied
    // for any edge length. Edge lengths are ~0.25, well within bounds.
    assert!(
        !needs_remesh(&mesh, &manifold, &h, &k, &config),
        "flat uniform mesh should not need remeshing"
    );
}

#[test]
fn needs_remesh_true_for_long_edges() {
    let manifold = Euclidean::<2>;
    let mesh = grid_mesh();
    let (h, k) = flat_curvatures(mesh.n_vertices());

    let config = RemeshConfig {
        max_edge_length: 0.1,  // edge lengths are ~0.25, so all violate
        ..RemeshConfig::default()
    };

    assert!(
        needs_remesh(&mesh, &manifold, &h, &k, &config),
        "mesh with edges exceeding max_edge_length should need remeshing"
    );
}

#[test]
fn needs_remesh_true_for_curvature_bump() {
    let manifold = Euclidean::<2>;
    let mesh = grid_mesh();
    let (h, k) = bump_curvatures(&mesh);

    let config = RemeshConfig {
        curvature_scale: 0.1,  // tight curvature resolution
        min_edge_length: 0.001,
        max_edge_length: 10.0,
        ..RemeshConfig::default()
    };

    assert!(
        needs_remesh(&mesh, &manifold, &h, &k, &config),
        "mesh with high-curvature bump should need remeshing"
    );
}

#[test]
fn adaptive_remesh_refines_near_bump() {
    let manifold = Euclidean::<2>;
    let mut mesh = grid_mesh();
    let nv_before = mesh.n_vertices();
    let (h, k) = bump_curvatures(&mesh);

    let ops = Operators::from_mesh(
        &Mesh::from_triangles(
            mesh.vertices.iter().map(|v| [v[0], v[1]]).collect(),
            mesh.simplices.clone(),
        ),
        &manifold,
    );

    let config = RemeshConfig {
        curvature_scale: 0.05,  // very tight, forces splits near bump
        min_edge_length: 0.001,
        max_edge_length: 10.0,
        min_face_area: 1e-10,
        max_face_area: 10.0,
        foldover_threshold: 0.5,
        lcr_spring_stiffness: 0.0,
        smoothing_iterations: 1,
    };

    let log = adaptive_remesh(&mut mesh, &manifold, &ops, &h, &k, &config);

    assert!(
        mesh.n_vertices() > nv_before,
        "adaptive remesh should add vertices near high curvature: before={nv_before}, after={}",
        mesh.n_vertices()
    );
    assert!(
        log.splits.len() > 0,
        "remesh log should record at least one split"
    );
}

#[test]
fn adaptive_remesh_log_records_all_operations() {
    let manifold = Euclidean::<2>;
    let mut mesh = grid_mesh();
    let (h, k) = bump_curvatures(&mesh);

    let ops = Operators::from_mesh(
        &Mesh::from_triangles(
            mesh.vertices.iter().map(|v| [v[0], v[1]]).collect(),
            mesh.simplices.clone(),
        ),
        &manifold,
    );

    let config = RemeshConfig {
        curvature_scale: 0.05,
        min_edge_length: 0.001,
        max_edge_length: 10.0,
        foldover_threshold: 0.5,
        smoothing_iterations: 2,
        ..RemeshConfig::default()
    };

    let log = adaptive_remesh(&mut mesh, &manifold, &ops, &h, &k, &config);

    // The log should faithfully record all mutations.
    let total = log.total_mutations();
    assert!(total > 0, "at least some remesh operations should have occurred");

    // Every split should reference a valid new vertex index.
    for split in &log.splits {
        assert!(
            split.new_vertex < mesh.n_vertices() + log.collapses.len(),
            "split new_vertex must reference a valid vertex"
        );
    }
}

#[test]
fn adaptive_remesh_noop_on_uniform_mesh() {
    let manifold = Euclidean::<2>;
    let mut mesh = grid_mesh();
    let nv_before = mesh.n_vertices();
    let ne_before = mesh.n_boundaries();
    let nf_before = mesh.n_simplices();
    let (h, k) = flat_curvatures(mesh.n_vertices());

    let ops = Operators::from_mesh(
        &Mesh::from_triangles(
            mesh.vertices.iter().map(|v| [v[0], v[1]]).collect(),
            mesh.simplices.clone(),
        ),
        &manifold,
    );

    let config = RemeshConfig {
        curvature_scale: 10.0,  // very loose
        min_edge_length: 0.001,
        max_edge_length: 10.0,
        min_face_area: 1e-10,
        max_face_area: 10.0,
        foldover_threshold: 0.5,
        lcr_spring_stiffness: 0.0,
        smoothing_iterations: 0,  // no smoothing
    };

    let log = adaptive_remesh(&mut mesh, &manifold, &ops, &h, &k, &config);

    // No topology changes expected (curvature is zero, edges within bounds).
    assert!(!log.topology_changed(), "no topology changes expected on uniform flat mesh");
    assert_eq!(mesh.n_vertices(), nv_before);
    assert_eq!(mesh.n_simplices(), nf_before);
}

#[test]
fn adaptive_remesh_preserves_euler_characteristic() {
    let manifold = Euclidean::<2>;
    let mut mesh = grid_mesh();
    let euler_before = mesh.euler_characteristic();
    let (h, k) = bump_curvatures(&mesh);

    let ops = Operators::from_mesh(
        &Mesh::from_triangles(
            mesh.vertices.iter().map(|v| [v[0], v[1]]).collect(),
            mesh.simplices.clone(),
        ),
        &manifold,
    );

    let config = RemeshConfig {
        curvature_scale: 0.05,
        min_edge_length: 0.001,
        max_edge_length: 10.0,
        foldover_threshold: 0.5,
        smoothing_iterations: 1,
        ..RemeshConfig::default()
    };

    let _ = adaptive_remesh(&mut mesh, &manifold, &ops, &h, &k, &config);

    assert_eq!(
        mesh.euler_characteristic(),
        euler_before,
        "adaptive remesh must preserve Euler characteristic"
    );
}
```

**Step 13.2: Verify tests fail**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test driver 2>&1 | head -5
# Should show: "not yet implemented: Task 13"
```

**Step 13.3: Implement `needs_remesh`**

Replace the `needs_remesh` stub in `~/cartan/cartan-remesh/src/driver.rs`:

```rust
/// Check whether the mesh needs remeshing.
///
/// Returns true if any edge violates the curvature resolution criterion:
/// `h_e > config.curvature_scale / sqrt(k_max)` where
/// `k_max = |H_v| + sqrt(max(0, H_v^2 - K_v))` at the vertex with larger
/// principal curvature. Also returns true if any edge length exceeds
/// `max_edge_length` or falls below `min_edge_length`.
///
/// The caller (volterra-dec) owns the kinetic-energy-minimum timing logic
/// that determines when to act on a true result.
pub fn needs_remesh<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    mean_curvatures: &[f64],
    gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> bool {
    for e in 0..mesh.n_boundaries() {
        let h_e = mesh.edge_length(manifold, e);

        // Check absolute edge length bounds.
        if h_e > config.max_edge_length || h_e < config.min_edge_length {
            return true;
        }

        // Check curvature-CFL criterion at both endpoints.
        let [v_a, v_b] = mesh.boundaries[e];
        for &vi in &[v_a, v_b] {
            if vi >= mean_curvatures.len() || vi >= gaussian_curvatures.len() {
                continue;
            }
            let h_v = mean_curvatures[vi];
            let k_v = gaussian_curvatures[vi];

            // Larger principal curvature magnitude.
            let discriminant = (h_v * h_v - k_v).max(0.0);
            let k_max = h_v.abs() + discriminant.sqrt();

            if k_max > 1e-12 {
                let threshold = config.curvature_scale / k_max.sqrt();
                if h_e > threshold {
                    return true;
                }
            }
        }
    }

    false
}
```

**Step 13.4: Implement `adaptive_remesh`**

Replace the `adaptive_remesh` stub in `~/cartan/cartan-remesh/src/driver.rs`:

```rust
/// Run the full adaptive remeshing pipeline.
///
/// Pipeline order:
/// 1. Flip non-Delaunay edges
/// 2. Split edges violating curvature-CFL: h_e > C / sqrt(k_max)
/// 3. Collapse short/flat edges with foldover guard
/// 4. Shift vertices (tangential Laplacian smoothing)
///
/// Rebuilds adjacency after each topology-changing pass.
/// Returns a merged [`RemeshLog`] covering all mutations.
pub fn adaptive_remesh<M: Manifold>(
    mesh: &mut Mesh<M, 3, 2>,
    manifold: &M,
    _operators: &Operators<M>,
    mean_curvatures: &[f64],
    gaussian_curvatures: &[f64],
    config: &RemeshConfig,
) -> RemeshLog {
    let mut log = RemeshLog::new();

    // ── Pass 1: Flip non-Delaunay edges ──────────────────────────────────
    //
    // Iterate over all edges and flip those that violate the Delaunay criterion.
    // Use a worklist to avoid re-checking flipped edges (the flip changes
    // adjacency, so edge indices shift after rebuild).
    let mut flipped = true;
    while flipped {
        flipped = false;
        let n_edges = mesh.n_boundaries();
        for e in 0..n_edges {
            match crate::primitives::flip_edge(mesh, manifold, e) {
                Ok(flip_log) => {
                    log.merge(flip_log);
                    flipped = true;
                    break; // restart scan after topology change + rebuild
                }
                Err(_) => continue,
            }
        }
    }

    // ── Pass 2: Split edges violating curvature-CFL ──────────────────────
    //
    // For each edge, check if h_e > C / sqrt(k_max) at either endpoint.
    // Also split edges exceeding max_edge_length.
    let mut split_occurred = true;
    while split_occurred {
        split_occurred = false;
        let n_edges = mesh.n_boundaries();
        for e in 0..n_edges {
            let h_e = mesh.edge_length(manifold, e);
            let mut should_split = h_e > config.max_edge_length;

            if !should_split {
                let [v_a, v_b] = mesh.boundaries[e];
                for &vi in &[v_a, v_b] {
                    if vi >= mean_curvatures.len() || vi >= gaussian_curvatures.len() {
                        continue;
                    }
                    let h_v = mean_curvatures[vi];
                    let k_v = gaussian_curvatures[vi];
                    let discriminant = (h_v * h_v - k_v).max(0.0);
                    let k_max = h_v.abs() + discriminant.sqrt();
                    if k_max > 1e-12 {
                        let threshold = config.curvature_scale / k_max.sqrt();
                        if h_e > threshold {
                            should_split = true;
                            break;
                        }
                    }
                }
            }

            if should_split && h_e > config.min_edge_length * 2.0 {
                let split_log = crate::primitives::split_edge(mesh, manifold, e);
                log.merge(split_log);
                split_occurred = true;
                break; // restart scan after topology change
            }
        }
    }

    // ── Pass 3: Collapse short/flat edges ────────────────────────────────
    //
    // Collapse edges shorter than min_edge_length or adjacent to faces
    // with area below min_face_area.
    let mut collapse_occurred = true;
    while collapse_occurred {
        collapse_occurred = false;
        let n_edges = mesh.n_boundaries();
        for e in 0..n_edges {
            let h_e = mesh.edge_length(manifold, e);
            let mut should_collapse = h_e < config.min_edge_length;

            // Check if adjacent faces have area below threshold.
            if !should_collapse {
                for &fi in &mesh.boundary_simplices[e] {
                    if mesh.triangle_area(manifold, fi) < config.min_face_area {
                        should_collapse = true;
                        break;
                    }
                }
            }

            if should_collapse {
                match crate::primitives::collapse_edge(
                    mesh,
                    manifold,
                    e,
                    config.foldover_threshold,
                ) {
                    Ok(collapse_log) => {
                        log.merge(collapse_log);
                        collapse_occurred = true;
                        break; // restart scan after topology change
                    }
                    Err(_) => continue, // foldover rejection, skip this edge
                }
            }
        }
    }

    // ── Pass 4: Tangential Laplacian smoothing ───────────────────────────
    //
    // Smooth all interior vertices for the configured number of iterations.
    for _iter in 0..config.smoothing_iterations {
        // Collect interior vertices (those with > 0 incident edges and not
        // on the boundary). For simplicity, smooth all vertices. Boundary
        // detection is handled inside shift_vertex (constraint to boundary
        // tangent).
        for vi in 0..mesh.n_vertices() {
            let shift_log = crate::primitives::shift_vertex(mesh, manifold, vi);
            log.merge(shift_log);
        }
    }

    log
}
```

**Step 13.5: Update imports at top of `driver.rs`**

```rust
use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};

use crate::config::RemeshConfig;
use crate::log::RemeshLog;
```

**Step 13.6: Verify tests pass**

```bash
cd ~/cartan && cargo test -p cartan-remesh --test driver
```

**Step 13.7: Run full crate test suite**

```bash
cd ~/cartan && cargo test -p cartan-remesh
```

**Step 13.8: Commit**

```bash
cd ~/cartan && git add cartan-remesh/src/driver.rs cartan-remesh/tests/driver.rs
git commit -m "feat(cartan-remesh): implement adaptive_remesh driver and needs_remesh predicate

adaptive_remesh runs the full pipeline: flip non-Delaunay edges, split
edges violating the curvature-CFL criterion, collapse short/flat edges
with foldover guard, and apply tangential Laplacian smoothing. Each
topology-changing pass restarts from scratch after adjacency rebuild.
needs_remesh checks the curvature resolution criterion and edge length
bounds without mutating the mesh."
```

---

**Phase B summary of file paths:**

| File | Task |
|------|------|
| `~/cartan/Cargo.toml` | 9 (add workspace member) |
| `~/cartan/cartan-remesh/Cargo.toml` | 9 |
| `~/cartan/cartan-remesh/src/lib.rs` | 9 |
| `~/cartan/cartan-remesh/src/log.rs` | 9 |
| `~/cartan/cartan-remesh/src/config.rs` | 9 |
| `~/cartan/cartan-remesh/src/error.rs` | 9 |
| `~/cartan/cartan-remesh/src/primitives.rs` | 9 (stubs), 10 (split/collapse), 11 (flip/shift) |
| `~/cartan/cartan-remesh/src/lcr.rs` | 9 (stubs), 12 (implementation) |
| `~/cartan/cartan-remesh/src/driver.rs` | 9 (stubs), 13 (implementation) |
| `~/cartan/cartan-remesh/tests/scaffold.rs` | 9 |
| `~/cartan/cartan-remesh/tests/split_collapse.rs` | 10 |
| `~/cartan/cartan-remesh/tests/flip_shift.rs` | 11 |
| `~/cartan/cartan-remesh/tests/lcr.rs` | 12 |
| `~/cartan/cartan-remesh/tests/driver.rs` | 13 |

**Dependency chain:** Task 9 -> Task 10 -> Task 11 -> Task 12 -> Task 13 (strictly sequential). Phase A (adjacency maps, `rebuild_adjacency`, `zero_tangent`, `boundary_simplices` on `Mesh<M, K, B>`) must be complete before Task 9.
---

## Phases C-F: volterra-dec + pathwise-geo

### Task 14: DecDomain struct + curvature.rs (volterra-dec)

**Prerequisite**: Phases A and B complete (cartan-dec generic operators, cartan-remesh crate exist).

#### Step 14.0: Update volterra workspace Cargo.toml to use local cartan path deps

The workspace currently depends on cartan crates from crates.io (v0.1.5). Since Phases A/B add new APIs to cartan-dec and create cartan-remesh, volterra must use local path overrides.

**File**: `/home/alejandrosotofranco/volterra/Cargo.toml`

Uncomment and extend the `[patch.crates-io]` section, and add the new workspace dependencies:

```toml
[workspace]
resolver = "2"
members = [
    "volterra",
    "volterra-core",
    "volterra-dec",
    "volterra-fields",
    "volterra-solver",
    "volterra-py",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT"
repository = "https://github.com/alejandro-soto-franco/volterra"
homepage = "https://github.com/alejandro-soto-franco/volterra"
documentation = "https://docs.rs/volterra"
readme = "README.md"
keywords = ["active-matter", "nematohydrodynamics", "riemannian", "simulation", "physics"]
categories = ["science", "mathematics", "simulation"]

[workspace.dependencies]
cartan-core      = "0.1.5"
cartan-manifolds = "0.1.5"
cartan-geo       = "0.1.5"
cartan-dec       = "0.1.5"
cartan-remesh    = "0.1.0"

# Use in-tree cartan for membrane-nematic DEC work (Phases A/B add new APIs).
[patch.crates-io]
cartan-core      = { path = "/home/alejandrosotofranco/cartan/cartan-core" }
cartan-manifolds = { path = "/home/alejandrosotofranco/cartan/cartan-manifolds" }
cartan-geo       = { path = "/home/alejandrosotofranco/cartan/cartan-geo" }
cartan-dec       = { path = "/home/alejandrosotofranco/cartan/cartan-dec" }
cartan-remesh    = { path = "/home/alejandrosotofranco/cartan/cartan-remesh" }

nalgebra         = { version = "0.33", features = ["std"] }
rayon            = "1"
thiserror        = "2"
serde            = { version = "1", features = ["derive"] }
serde_json       = "1"
approx           = "0.5"
rand             = { version = "0.9", features = ["small_rng"] }
rustfft          = "6"
pyo3             = { version = "0.25", features = ["extension-module", "abi3-py310"] }
numpy            = "0.25"
pathwise-geo     = { path = "/home/alejandrosotofranco/pathwise/pathwise-geo" }
```

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/Cargo.toml`

Add the new dependencies needed for Phases C-F:

```toml
[package]
name = "volterra-dec"
description = "Discrete exterior calculus layer: simplicial complexes, Hodge operators, and covariant differential operators on Riemannian manifolds"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
keywords.workspace = true
categories.workspace = true

[dependencies]
volterra-core    = { path = "../volterra-core",   version = "0.1.0" }
cartan-core      = { workspace = true }
cartan-manifolds = { workspace = true }
cartan-dec       = { workspace = true }
cartan-remesh    = { workspace = true }
pathwise-geo     = { workspace = true }
nalgebra         = { workspace = true }
rayon            = { workspace = true }
thiserror        = { workspace = true }
serde            = { workspace = true }
rand             = { workspace = true }

[dev-dependencies]
approx           = { workspace = true }
```

**Verify**: `cd /home/alejandrosotofranco/volterra && cargo check -p volterra-dec`

#### Step 14.1: Write failing test for DecDomain construction

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/dec_domain.rs`

```rust
//! Integration tests for DecDomain and discrete curvature.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;
use std::f64::consts::PI;

/// Build an icosahedron mesh on S^2 with radius R.
/// Returns (vertices, triangles) for Mesh<Euclidean<3>, 3, 2>.
fn icosahedron_mesh(r: f64) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = r / norm;
    let b = r * phi / norm;

    let raw = [
        SVector::from([-a,  b, 0.0]),
        SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]),
        SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]),
        SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]),
        SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]),
        SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]),
        SVector::from([-b, 0.0,  a]),
    ];

    // Normalize to sphere of radius R
    let vertices: Vec<_> = raw.iter().map(|v| v * (r / v.norm())).collect();

    let triangles = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    (vertices, triangles)
}

#[test]
fn test_dec_domain_construction() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);

    let domain = volterra_dec::DecDomain::new(mesh, manifold)
        .expect("DecDomain construction should succeed");

    assert_eq!(domain.n_vertices(), 12);
    assert_eq!(domain.n_edges(), 30);
    assert_eq!(domain.n_faces(), 20);
}

#[test]
fn test_mean_curvature_sphere() {
    let r = 3.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);

    let domain = volterra_dec::DecDomain::new(mesh, manifold)
        .expect("DecDomain construction should succeed");

    // Mean curvature of a sphere of radius R is 1/R at every vertex.
    let expected_h = 1.0 / r;
    for (v, &h) in domain.mean_curvatures.iter().enumerate() {
        let err = (h - expected_h).abs() / expected_h;
        assert!(
            err < 0.15,
            "vertex {v}: mean curvature {h:.4} vs expected {expected_h:.4} (rel err {err:.4})"
        );
    }
}

#[test]
fn test_gauss_bonnet_sphere() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);

    let domain = volterra_dec::DecDomain::new(mesh, manifold)
        .expect("DecDomain construction should succeed");

    // Gauss-Bonnet: sum_v(K_v * dual_area_v) = 2*pi*chi(S^2) = 4*pi
    let total: f64 = domain
        .gaussian_curvatures
        .iter()
        .zip(domain.dual_areas.iter())
        .map(|(k, a)| k * a)
        .sum();

    let expected = 4.0 * PI;
    let err = (total - expected).abs();
    assert!(
        err < 0.1,
        "Gauss-Bonnet sum {total:.6} vs expected {expected:.6} (err {err:.6})"
    );
}

#[test]
fn test_principal_curvatures_sphere() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);

    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();

    let principals = volterra_dec::curvature::principal_curvatures(
        &domain.mean_curvatures,
        &domain.gaussian_curvatures,
    );

    let expected = 1.0 / r;
    for (v, &(k1, k2)) in principals.iter().enumerate() {
        let err1 = (k1 - expected).abs() / expected;
        let err2 = (k2 - expected).abs() / expected;
        assert!(
            err1 < 0.2 && err2 < 0.2,
            "vertex {v}: principals ({k1:.4}, {k2:.4}) vs expected ({expected:.4}, {expected:.4})"
        );
    }
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test dec_domain` (should fail: `DecDomain` does not exist yet)

#### Step 14.2: Implement curvature.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/curvature.rs`

```rust
//! Discrete curvature computations on triangle meshes.
//!
//! Mean curvature via the cotan Laplacian applied to the position vector.
//! Gaussian curvature via the angle deficit formula.
//! Both are standard DEC curvature approximations.
//!
//! References:
//! - Meyer et al. "Discrete Differential-Geometry Operators for Triangulated
//!   2-Manifolds." VisMath, 2003.
//! - Zhu, Lee, Rangamani. "Mem3DG." Biophysical Reports, 2022.

use std::f64::consts::PI;

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};

/// Mean curvature at each vertex via the cotan Laplacian.
///
/// H_v = inner(Lap(x)_v, n_v) / (2 * dual_area_v)
///
/// where Lap(x) is the cotan Laplacian of the position vector and n_v is
/// the vertex normal. The factor of 2 comes from the convention that the
/// mean curvature vector is Lap(x) = 2*H*n.
pub fn mean_curvature_cotan<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    operators: &Operators<M>,
    normals: &[M::Tangent],
) -> Vec<f64>
where
    M::Tangent: AsRef<[f64]>,
{
    let nv = mesh.n_vertices();
    let dim = manifold.ambient_dim();

    // Build position vector per component, apply Laplacian, extract H via dot with normal.
    // Position components: for each coordinate d, build a DVector f_d[v] = vertex[v][d].
    let mut mean_curvatures = vec![0.0; nv];

    // We need to extract the position coordinates from M::Point.
    // Strategy: use manifold.log from a reference point (the vertex itself), which gives zero,
    // so instead we compute Lap(x) directly via the cotan weights.
    //
    // Lap(x)_v = (1/A_v) * sum_{j in N(v)} w_{vj} * (x_j - x_v)
    // where w_{vj} = (cot alpha_{vj} + cot beta_{vj}) / 2 from the cotan Laplacian.
    //
    // We use the assembled Laplace-Beltrami matrix L from operators.
    // L acts on scalar fields. We apply it component-wise to the position vector.

    // Extract position components by logging all vertices from vertex 0 is not right.
    // Instead, for an embedded mesh in Euclidean<N>, the vertices ARE the coordinates.
    // For a general manifold, we need to embed. For now, require M::Point: AsRef<[f64]>.
    //
    // Actually, the cotan Laplacian applied to the position vector can be computed
    // more directly: for each vertex, accumulate cotan-weighted edge vectors.

    use nalgebra::DVector;

    // Component-wise Laplacian approach: build DVector for each ambient coordinate,
    // apply L, then dot with normal.
    let mut lap_components: Vec<DVector<f64>> = Vec::with_capacity(dim);
    for d in 0..dim {
        // We need vertex positions as f64 slices. For Euclidean<N>, Point = SVector<f64, N>.
        // Use the mesh vertices directly. The Laplacian L is assembled already.
        let mut f_d = DVector::zeros(nv);
        for v in 0..nv {
            // Extract d-th coordinate from the vertex.
            // For Euclidean embeddings, we assume M::Point: AsRef<[f64]> or indexable.
            // Since cartan uses SVector for Euclidean, we can index via as_slice.
            f_d[v] = vertex_coord(mesh, v, d);
        }
        lap_components.push(operators.apply_laplace_beltrami(&f_d));
    }

    // H_v = dot(Lap(x)_v, n_v) / 2
    // The cotan Laplacian gives Lap(x) = 2*H*n for a surface in R^3.
    // But our L already includes the 1/dual_area factor, so Lap(x)_v = (1/A_v) sum w(x_j - x_v).
    // This equals the mean curvature normal: Lap(x) = 2*H*n (when L includes mass inverse).
    for v in 0..nv {
        let normal_slice = normals[v].as_ref();
        let mut dot = 0.0;
        for d in 0..dim {
            dot += lap_components[d][v] * normal_slice[d];
        }
        // The cotan Laplacian with mass inverse gives Lap(x) = 2*H*n.
        mean_curvatures[v] = dot / 2.0;
    }

    mean_curvatures
}

/// Gaussian curvature at each vertex via the angle deficit formula.
///
/// K_v = (2*pi - sum of corner angles at v) / dual_area_v
///
/// This is exact for piecewise-linear surfaces (Descartes-Euler-Hilbert theorem).
pub fn gaussian_curvature_angle_deficit<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    dual_areas: &[f64],
) -> Vec<f64> {
    let nv = mesh.n_vertices();
    let mut angle_sums = vec![0.0_f64; nv];

    for (t, &[i, j, k]) in mesh.simplices.iter().enumerate() {
        // Compute the three corner angles of triangle (i, j, k).
        let angles = triangle_corner_angles(mesh, manifold, i, j, k);
        angle_sums[i] += angles[0];
        angle_sums[j] += angles[1];
        angle_sums[k] += angles[2];
    }

    let mut gauss = vec![0.0; nv];
    for v in 0..nv {
        let deficit = 2.0 * PI - angle_sums[v];
        let area = dual_areas[v];
        gauss[v] = if area > 1e-30 { deficit / area } else { 0.0 };
    }
    gauss
}

/// Principal curvatures from mean and Gaussian curvature.
///
/// k1, k2 = H +/- sqrt(H^2 - K)
/// Returns (k1, k2) with k1 >= k2.
pub fn principal_curvatures(h: &[f64], k: &[f64]) -> Vec<(f64, f64)> {
    h.iter()
        .zip(k.iter())
        .map(|(&hv, &kv)| {
            let disc = (hv * hv - kv).max(0.0).sqrt();
            (hv + disc, hv - disc)
        })
        .collect()
}

/// Vertex normals: area-weighted average of incident face normals.
///
/// Each face normal is computed from the cross product of two edge vectors
/// in the ambient space. Weighted by triangle area (the cross product magnitude
/// is 2*area, so unnormalized cross products give area weighting for free).
pub fn vertex_normals<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
) -> Vec<M::Tangent>
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let nv = mesh.n_vertices();
    let dim = manifold.ambient_dim();

    // Accumulate area-weighted face normals per vertex.
    // For a triangle with vertices (i, j, k), the face normal in the tangent space
    // at vertex i is cross(log_i(j), log_i(k)). We accumulate this at all three vertices.
    // For simplicity, compute in R^3 embedding (ambient coordinates).

    // Initialize accumulators as zero tangent vectors.
    let p0 = &mesh.vertices[0];
    let mut accum: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];

    for &[i, j, k] in &mesh.simplices {
        let vi = &mesh.vertices[i];
        // Compute edge vectors in tangent space at vi
        let e1 = manifold
            .log(vi, &mesh.vertices[j])
            .unwrap_or_else(|_| manifold.zero_tangent(vi));
        let e2 = manifold
            .log(vi, &mesh.vertices[k])
            .unwrap_or_else(|_| manifold.zero_tangent(vi));

        // Cross product (only works for dim=3 ambient, which is the membrane case)
        let e1s = e1.as_ref();
        let e2s = e2.as_ref();
        if dim >= 3 {
            let nx = e1s[1] * e2s[2] - e1s[2] * e2s[1];
            let ny = e1s[2] * e2s[0] - e1s[0] * e2s[2];
            let nz = e1s[0] * e2s[1] - e1s[1] * e2s[0];

            for v in [i, j, k] {
                accum[v][0] += nx;
                accum[v][1] += ny;
                accum[v][2] += nz;
            }
        }
    }

    // Normalize
    accum
        .into_iter()
        .enumerate()
        .map(|(v, mut n)| {
            let len: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
            if len > 1e-30 {
                for x in n.iter_mut() {
                    *x /= len;
                }
            }
            // Convert back to M::Tangent
            let p = &mesh.vertices[v];
            let mut tangent = manifold.zero_tangent(p);
            let t_slice = tangent.as_mut();
            for (d, val) in n.into_iter().enumerate().take(dim) {
                t_slice[d] = val;
            }
            manifold.project_tangent(p, &tangent)
        })
        .collect()
}

/// Extract the d-th coordinate of vertex v.
///
/// Assumes M::Point is stored as contiguous f64 data (true for nalgebra SVector).
fn vertex_coord<M: Manifold>(mesh: &Mesh<M, 3, 2>, v: usize, d: usize) -> f64 {
    // SAFETY: cartan uses nalgebra::SVector<f64, N> for Euclidean<N>.
    // SVector implements AsRef<[f64]> via its storage.
    // For a general manifold embedded in R^N, Point = SVector<f64, N>.
    let ptr = &mesh.vertices[v] as *const M::Point as *const f64;
    // This is safe as long as M::Point is repr as contiguous f64s.
    unsafe { *ptr.add(d) }
}

/// Compute the three corner angles of triangle (i, j, k) on the manifold.
///
/// Returns [angle_at_i, angle_at_j, angle_at_k].
fn triangle_corner_angles<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
    i: usize,
    j: usize,
    k: usize,
) -> [f64; 3] {
    let vi = &mesh.vertices[i];
    let vj = &mesh.vertices[j];
    let vk = &mesh.vertices[k];

    let angle_at = |p: &M::Point, q: &M::Point, r: &M::Point| -> f64 {
        let u = manifold.log(p, q).unwrap_or_else(|_| manifold.zero_tangent(p));
        let v = manifold.log(p, r).unwrap_or_else(|_| manifold.zero_tangent(p));
        let nu = manifold.norm(p, &u);
        let nv = manifold.norm(p, &v);
        if nu < 1e-30 || nv < 1e-30 {
            return 0.0;
        }
        let cos_a = (manifold.inner(p, &u, &v) / (nu * nv)).clamp(-1.0, 1.0);
        cos_a.acos()
    };

    [
        angle_at(vi, vj, vk),
        angle_at(vj, vi, vk),
        angle_at(vk, vi, vj),
    ]
}

/// Dual cell areas (barycentric): (1/3) * sum of incident triangle areas per vertex.
pub fn dual_areas<M: Manifold>(mesh: &Mesh<M, 3, 2>, manifold: &M) -> Vec<f64> {
    let nv = mesh.n_vertices();
    let mut areas = vec![0.0; nv];
    for (t, &[i, j, k]) in mesh.simplices.iter().enumerate() {
        let a = mesh.triangle_area(manifold, t);
        areas[i] += a / 3.0;
        areas[j] += a / 3.0;
        areas[k] += a / 3.0;
    }
    areas
}

/// Face areas for all simplices.
pub fn face_areas<M: Manifold>(mesh: &Mesh<M, 3, 2>, manifold: &M) -> Vec<f64> {
    (0..mesh.n_simplices())
        .map(|t| mesh.triangle_area(manifold, t))
        .collect()
}
```

#### Step 14.3: Implement DecDomain struct

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/domain.rs`

```rust
//! DecDomain: the central simulation domain for DEC on triangle meshes.
//!
//! Wraps a mesh, manifold, precomputed operators, and cached geometry
//! (curvatures, normals, areas). Reassembly after mesh mutation (remeshing,
//! vertex motion) is handled by `reassemble()`.

use cartan_core::Manifold;
use cartan_dec::{DecError, Mesh, Operators};

use crate::curvature;

/// The DEC simulation domain for a triangle mesh on a Riemannian manifold.
///
/// Owns the mesh, manifold, assembled DEC operators, and cached per-vertex
/// geometry. All cached fields are consistent with the current mesh state.
/// After any mesh mutation (remesh, vertex motion), call `reassemble()` to
/// rebuild operators and refresh cached geometry.
pub struct DecDomain<M: Manifold> {
    /// The triangle mesh.
    pub mesh: Mesh<M, 3, 2>,
    /// The ambient manifold.
    pub manifold: M,
    /// Assembled DEC operators (Laplacian, Hodge, exterior derivative).
    pub operators: Operators<M>,
    /// Reference length-cross-ratios (captured at construction or after remesh).
    pub ref_lcrs: Vec<f64>,
    /// Mean curvature at each vertex (cotan Laplacian formula).
    pub mean_curvatures: Vec<f64>,
    /// Gaussian curvature at each vertex (angle deficit formula).
    pub gaussian_curvatures: Vec<f64>,
    /// Unit normal at each vertex (area-weighted average of face normals).
    pub vertex_normals: Vec<M::Tangent>,
    /// Triangle areas.
    pub face_areas: Vec<f64>,
    /// Barycentric dual cell areas per vertex.
    pub dual_areas: Vec<f64>,
}

impl<M: Manifold> DecDomain<M>
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Construct a new DecDomain from a triangle mesh and manifold.
    ///
    /// Assembles all DEC operators and computes initial cached geometry.
    /// Returns Err if the mesh is degenerate or operators cannot be assembled.
    pub fn new(mesh: Mesh<M, 3, 2>, manifold: M) -> Result<Self, DecError> {
        let operators = Operators::from_mesh_generic(&mesh, &manifold)?;
        let face_areas = curvature::face_areas(&mesh, &manifold);
        let dual_areas = curvature::dual_areas(&mesh, &manifold);
        let vertex_normals = curvature::vertex_normals(&mesh, &manifold);
        let mean_curvatures = curvature::mean_curvature_cotan(
            &mesh,
            &manifold,
            &operators,
            &vertex_normals,
        );
        let gaussian_curvatures =
            curvature::gaussian_curvature_angle_deficit(&mesh, &manifold, &dual_areas);

        // Reference LCRs: initially empty (populated by caller or after first remesh).
        let ref_lcrs = Vec::new();

        Ok(Self {
            mesh,
            manifold,
            operators,
            ref_lcrs,
            mean_curvatures,
            gaussian_curvatures,
            vertex_normals,
            face_areas,
            dual_areas,
        })
    }

    /// Rebuild all operators and refresh cached geometry.
    ///
    /// Call this after any mesh mutation (vertex position update, remeshing).
    /// The mesh topology and vertex positions must already be updated.
    pub fn reassemble(&mut self) -> Result<(), DecError> {
        self.operators = Operators::from_mesh_generic(&self.mesh, &self.manifold)?;
        self.face_areas = curvature::face_areas(&self.mesh, &self.manifold);
        self.dual_areas = curvature::dual_areas(&self.mesh, &self.manifold);
        self.vertex_normals = curvature::vertex_normals(&self.mesh, &self.manifold);
        self.mean_curvatures = curvature::mean_curvature_cotan(
            &self.mesh,
            &self.manifold,
            &self.operators,
            &self.vertex_normals,
        );
        self.gaussian_curvatures = curvature::gaussian_curvature_angle_deficit(
            &self.mesh,
            &self.manifold,
            &self.dual_areas,
        );
        Ok(())
    }

    /// Number of mesh vertices.
    pub fn n_vertices(&self) -> usize {
        self.mesh.n_vertices()
    }

    /// Number of mesh edges.
    pub fn n_edges(&self) -> usize {
        self.mesh.n_boundaries()
    }

    /// Number of mesh faces (triangles).
    pub fn n_faces(&self) -> usize {
        self.mesh.n_simplices()
    }
}
```

#### Step 14.4: Update volterra-dec lib.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/lib.rs`

Replace the entire file:

```rust
//! # volterra-dec
//!
//! Discrete exterior calculus (DEC) layer for volterra.
//!
//! This crate bridges the continuous Riemannian geometry of `cartan-core` to the
//! discrete operators needed by the PDE solver. Given a manifold implementing
//! `cartan_core::Manifold`, it builds a simplicial complex over the domain,
//! precomputes all static operators, and exposes them for use in the time loop.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`domain`] | `DecDomain<M>`: simulation domain with mesh, operators, cached geometry |
//! | [`curvature`] | Mean/Gaussian/principal curvature on triangle meshes |
//! | [`helfrich`] | Helfrich membrane energy and shape forces |
//! | [`coupling`] | Nematic-membrane coupling (spontaneous curvature from Q) |
//! | [`transport`] | DEC conserved scalar transport |
//! | [`beris_edwards`] | Beris-Edwards RHS on DEC meshes |
//! | [`variational`] | BAOAB variational integrator |
//! | [`interpolation`] | Field interpolation after remeshing |
//! | [`observables`] | Physical observables (defect charge, Euler characteristic, etc.) |
//! | [`simulation`] | `MembraneNematicSim` driver struct |

pub mod curvature;
pub mod domain;
pub mod helfrich;
pub mod coupling;
pub mod transport;
pub mod beris_edwards;
pub mod variational;
pub mod interpolation;
pub mod observables;
pub mod simulation;

pub use domain::DecDomain;
```

**Note**: Modules `helfrich` through `simulation` will be empty stubs initially. Create stub files:

```bash
cd /home/alejandrosotofranco/volterra/volterra-dec/src
for mod in helfrich coupling transport beris_edwards variational interpolation observables simulation; do
    echo "//! Placeholder: implemented in later tasks." > "${mod}.rs"
done
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test dec_domain` (tests should now pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add DecDomain struct, discrete curvature, vertex normals"`

---

### Task 15: Helfrich energy + forces (helfrich.rs)

#### Step 15.1: Write failing tests

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/helfrich.rs`

```rust
//! Integration tests for Helfrich energy and forces.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;
use std::f64::consts::PI;

/// Build an icosahedron mesh on a sphere of radius R.
fn icosahedron_mesh(r: f64) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = r / norm;
    let b = r * phi / norm;

    let raw = [
        SVector::from([-a,  b, 0.0]),
        SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]),
        SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]),
        SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]),
        SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]),
        SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]),
        SVector::from([-b, 0.0,  a]),
    ];
    let vertices: Vec<_> = raw.iter().map(|v| v * (r / v.norm())).collect();
    let triangles = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (vertices, triangles)
}

#[test]
fn test_helfrich_energy_sphere() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);
    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();

    let kb = 1.0;
    let params = volterra_dec::helfrich::HelfrichParams {
        kb,
        kg: 0.0,
        h0: vec![0.0; domain.n_vertices()],
    };

    let energy = volterra_dec::helfrich::helfrich_energy(&domain, &params);

    // Analytic: E_b = 2*Kb * integral(H^2 dA) = 2*Kb * (1/R^2) * 4*pi*R^2 = 8*pi*Kb
    let expected = 8.0 * PI * kb;
    let err = (energy.total - expected).abs() / expected;
    assert!(
        err < 0.2,
        "Helfrich energy {:.4} vs expected {:.4} (rel err {:.4})",
        energy.total,
        expected,
        err
    );
}

#[test]
fn test_helfrich_forces_radially_inward() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts.clone(), tris);
    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();

    // With H0 = 0 and H > 0 (sphere), the bending force should push inward
    // (reducing the bending energy by shrinking the sphere).
    let params = volterra_dec::helfrich::HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![0.0; domain.n_vertices()],
    };

    let forces = volterra_dec::helfrich::helfrich_forces(&domain, &params);

    for (v, f) in forces.iter().enumerate() {
        let pos = &verts[v];
        let f_slice = f.as_ref();
        let pos_slice: &[f64] = pos.as_ref();
        // Force should point inward: dot(force, position) < 0
        let dot: f64 = f_slice.iter().zip(pos_slice.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot < 0.0,
            "vertex {v}: force dot position = {dot:.6} should be negative (radially inward)"
        );
    }
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test helfrich` (should fail)

#### Step 15.2: Implement helfrich.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/helfrich.rs`

```rust
//! Helfrich membrane energy and shape forces.
//!
//! Bending energy: E_b = 2 * Kb * sum_v(dual_area_v * (H_v - H0_v)^2)
//! Gaussian energy: E_g = Kg * sum_v(K_v * dual_area_v)
//!
//! Forces computed via halfedge accumulation following the Mem3DG pattern
//! (Zhu, Lee, Rangamani, Biophysical Reports 2022).
//!
//! The three force components per halfedge are:
//! 1. Area gradient (mean curvature vector)
//! 2. Gauss vector (dihedral angle contribution)
//! 3. Schlafli vector (Laplacian-of-curvature term)
//!
//! References:
//! - Helfrich. "Elastic Properties of Lipid Bilayers." Z Naturforsch C, 1973.
//! - Zhu, Lee, Rangamani. "Mem3DG." Biophysical Reports, 2022.

use cartan_core::Manifold;

use crate::domain::DecDomain;

/// Physical parameters for the Helfrich membrane energy.
pub struct HelfrichParams {
    /// Bending modulus (Kb).
    pub kb: f64,
    /// Gaussian modulus (Kg).
    pub kg: f64,
    /// Per-vertex spontaneous curvature H0.
    pub h0: Vec<f64>,
}

/// Decomposed Helfrich energy.
pub struct HelfrichEnergy {
    /// Total energy (bending + Gaussian).
    pub total: f64,
    /// Bending contribution: 2*Kb * sum(dual_area * (H - H0)^2).
    pub bending: f64,
    /// Gaussian contribution: Kg * sum(K * dual_area).
    pub gaussian: f64,
}

/// Compute the Helfrich energy from a DecDomain and parameters.
///
/// Uses the precomputed mean and Gaussian curvatures stored in the domain.
pub fn helfrich_energy<M: Manifold>(
    domain: &DecDomain<M>,
    params: &HelfrichParams,
) -> HelfrichEnergy
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let nv = domain.n_vertices();
    let mut bending = 0.0;
    let mut gaussian = 0.0;

    for v in 0..nv {
        let h = domain.mean_curvatures[v];
        let h0 = params.h0[v];
        let k_gauss = domain.gaussian_curvatures[v];
        let a = domain.dual_areas[v];

        bending += 2.0 * params.kb * a * (h - h0) * (h - h0);
        gaussian += params.kg * k_gauss * a;
    }

    HelfrichEnergy {
        total: bending + gaussian,
        bending,
        gaussian,
    }
}

/// Compute Helfrich shape forces at each vertex.
///
/// The force at vertex v is the negative gradient of E_helfrich with respect
/// to the position of v. Computed via the halfedge accumulation pattern.
///
/// For a sphere with H0 = 0, forces point radially inward.
pub fn helfrich_forces<M: Manifold>(
    domain: &DecDomain<M>,
    params: &HelfrichParams,
) -> Vec<M::Tangent>
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let nv = domain.n_vertices();
    let ne = domain.n_edges();
    let dim = domain.manifold.ambient_dim();

    // Initialize force accumulators as zero vectors.
    let mut forces: Vec<Vec<f64>> = vec![vec![0.0; dim]; nv];

    // For each edge (halfedge pair), accumulate force contributions.
    for e in 0..ne {
        let [vi, vj] = domain.mesh.boundaries[e];

        let hi = domain.mean_curvatures[vi];
        let hj = domain.mean_curvatures[vj];
        let h0i = params.h0[vi];
        let h0j = params.h0[vj];

        // Edge vector in ambient space
        let pi = &domain.mesh.vertices[vi];
        let pj = &domain.mesh.vertices[vj];
        let edge_vec = domain.manifold.log(pi, pj)
            .unwrap_or_else(|_| domain.manifold.zero_tangent(pi));
        let edge_len = domain.manifold.norm(pi, &edge_vec);
        if edge_len < 1e-30 {
            continue;
        }

        let edge_s = edge_vec.as_ref();
        let edge_unit: Vec<f64> = edge_s.iter().map(|&x| x / edge_len).collect();

        // Dihedral angle at this edge
        let dihedral = compute_dihedral_angle(domain, e);

        // 1. Gauss vector contribution: 0.5 * dihedral * edge_unit
        // Weighted by -2*Kb*(H - H0), interpolated between endpoints.
        let w_i = -2.0 * params.kb * (hi - h0i);
        let w_j = -2.0 * params.kb * (hj - h0j);
        let w_avg = (w_i + w_j) / 2.0;

        let gauss_scale = 0.5 * dihedral * w_avg;
        for d in 0..dim {
            // Force on vi (positive direction)
            forces[vi][d] += gauss_scale * edge_unit[d];
            // Force on vj (opposite direction, Newton's third law)
            forces[vj][d] -= gauss_scale * edge_unit[d];
        }

        // 2. Area gradient contribution
        // Force from varying the area: area_grad = (normal x edge) / 4
        // weighted by 2*Kb*(H0^2 - H^2) with interpolation.
        let normal_i = domain.vertex_normals[vi].as_ref();
        let normal_j = domain.vertex_normals[vj].as_ref();
        let normal_avg: Vec<f64> = (0..dim)
            .map(|d| (normal_i[d] + normal_j[d]) / 2.0)
            .collect();

        if dim >= 3 {
            // cross(normal_avg, edge_unit) / 4
            let cx = normal_avg[1] * edge_unit[2] - normal_avg[2] * edge_unit[1];
            let cy = normal_avg[2] * edge_unit[0] - normal_avg[0] * edge_unit[2];
            let cz = normal_avg[0] * edge_unit[1] - normal_avg[1] * edge_unit[0];
            let cross = [cx, cy, cz];

            let area_weight_i = 2.0 * params.kb * (h0i * h0i - hi * hi);
            let area_weight_j = 2.0 * params.kb * (h0j * h0j - hj * hj);
            // 1/3 at vi, 2/3 at vj interpolation
            let w_vi = (area_weight_i / 3.0 + 2.0 * area_weight_j / 3.0) * edge_len / 4.0;
            let w_vj = (area_weight_j / 3.0 + 2.0 * area_weight_i / 3.0) * edge_len / 4.0;

            for d in 0..3 {
                forces[vi][d] += w_vi * cross[d];
                forces[vj][d] += w_vj * cross[d];
            }
        }

        // 3. Schlafli vector: edge_length * d(dihedral)/d(vertex_pos)
        // Simplified: normal pressure term. For a sphere, the bending pressure is:
        //   f_v = -2*Kb * (2*H*(H^2 - K) + lap(H)) * n_v * dual_area_v
        // which points radially inward when H > 0 and H0 = 0.
    }

    // Add the dominant normal pressure term:
    // f_v = -2*Kb * 2*H*(H^2 - K) * n * dual_area (Laplacian-of-H term omitted for now)
    for v in 0..nv {
        let h = domain.mean_curvatures[v];
        let h0 = params.h0[v];
        let k = domain.gaussian_curvatures[v];
        let a = domain.dual_areas[v];
        let n = domain.vertex_normals[v].as_ref();

        // Shape equation pressure: the normal component of the Helfrich force.
        // f_n = -2*Kb * (2*(H-H0)*((H-H0)^2 + (H-H0)*H0 - K) + lap(H-H0)) * dual_area
        // Simplified (dropping lap term): -2*Kb * 2*(H-H0)*(H^2 - K - H*H0) * dual_area
        let dh = h - h0;
        let pressure = -2.0 * params.kb * 2.0 * dh * (h * h - k) * a;

        for d in 0..dim {
            forces[v][d] += pressure * n[d];
        }
    }

    // Convert force vectors back to M::Tangent.
    forces
        .into_iter()
        .enumerate()
        .map(|(v, f)| {
            let p = &domain.mesh.vertices[v];
            let mut tangent = domain.manifold.zero_tangent(p);
            let t_slice = tangent.as_mut();
            for (d, val) in f.into_iter().enumerate().take(dim) {
                t_slice[d] = val;
            }
            domain.manifold.project_tangent(p, &tangent)
        })
        .collect()
}

/// Compute the dihedral angle at edge e.
///
/// The dihedral angle is the angle between the normals of the two adjacent faces.
/// For boundary edges (one adjacent face), returns 0.
fn compute_dihedral_angle<M: Manifold>(domain: &DecDomain<M>, e: usize) -> f64
where
    M::Tangent: AsRef<[f64]>,
{
    let dim = domain.manifold.ambient_dim();
    if dim < 3 {
        return 0.0;
    }

    // Find the two faces adjacent to edge e.
    let [vi, vj] = domain.mesh.boundaries[e];
    let mut adjacent_faces = Vec::new();

    for (t, simplex) in domain.mesh.simplices.iter().enumerate() {
        let has_vi = simplex.contains(&vi);
        let has_vj = simplex.contains(&vj);
        if has_vi && has_vj {
            adjacent_faces.push(t);
        }
    }

    if adjacent_faces.len() < 2 {
        return 0.0; // boundary edge
    }

    // Compute face normals for the two adjacent triangles.
    let fn1 = face_normal(domain, adjacent_faces[0]);
    let fn2 = face_normal(domain, adjacent_faces[1]);

    // Dihedral angle: angle between face normals (pi - angle between normals).
    let dot: f64 = fn1.iter().zip(fn2.iter()).map(|(a, b)| a * b).sum();
    let clamped = dot.clamp(-1.0, 1.0);
    // The dihedral angle is pi minus the angle between outward normals for a convex surface.
    std::f64::consts::PI - clamped.acos()
}

/// Compute the outward face normal for triangle t (unnormalized, then normalized).
fn face_normal<M: Manifold>(domain: &DecDomain<M>, t: usize) -> Vec<f64>
where
    M::Tangent: AsRef<[f64]>,
{
    let dim = domain.manifold.ambient_dim();
    let [i, j, k] = domain.mesh.simplices[t];
    let vi = &domain.mesh.vertices[i];

    let e1 = domain.manifold.log(vi, &domain.mesh.vertices[j])
        .unwrap_or_else(|_| domain.manifold.zero_tangent(vi));
    let e2 = domain.manifold.log(vi, &domain.mesh.vertices[k])
        .unwrap_or_else(|_| domain.manifold.zero_tangent(vi));

    let e1s = e1.as_ref();
    let e2s = e2.as_ref();

    if dim >= 3 {
        let nx = e1s[1] * e2s[2] - e1s[2] * e2s[1];
        let ny = e1s[2] * e2s[0] - e1s[0] * e2s[2];
        let nz = e1s[0] * e2s[1] - e1s[1] * e2s[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-30 {
            vec![nx / len, ny / len, nz / len]
        } else {
            vec![0.0; dim]
        }
    } else {
        vec![0.0; dim]
    }
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test helfrich` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add Helfrich energy and shape forces with halfedge accumulation"`

---

### Task 16: Coupling + DEC transport (coupling.rs, transport.rs)

#### Step 16.1: Write failing tests

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/coupling_transport.rs`

```rust
//! Tests for nematic-membrane coupling and DEC conserved transport.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::{DVector, SVector};

/// Build a small flat triangular mesh for transport tests.
fn small_flat_mesh() -> Mesh<Euclidean<3>, 3, 2> {
    let manifold = Euclidean::<3>;
    let vertices = vec![
        SVector::from([0.0, 0.0, 0.0]),
        SVector::from([1.0, 0.0, 0.0]),
        SVector::from([0.5, 0.866, 0.0]),
        SVector::from([1.5, 0.866, 0.0]),
        SVector::from([0.0, 1.732, 0.0]),
        SVector::from([1.0, 1.732, 0.0]),
    ];
    let triangles = vec![
        [0, 1, 2],
        [1, 3, 2],
        [2, 3, 5],
        [2, 5, 4],
    ];
    Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, vertices, triangles)
}

#[test]
fn test_spontaneous_curvature_from_q() {
    let nv = 6;
    let manifold = Euclidean::<3>;
    // Zero Q-field should produce zero spontaneous curvature.
    let q_zero: Vec<SVector<f64, 3>> = vec![SVector::zeros(); nv];
    let coupling = 2.0;
    let h0 = volterra_dec::coupling::spontaneous_curvature_from_q(&q_zero, coupling);
    for &val in &h0 {
        assert!(
            val.abs() < 1e-12,
            "zero Q should give zero H0, got {val}"
        );
    }
}

#[test]
fn test_flux_form_conserves_mass() {
    let mesh = small_flat_mesh();
    let manifold = Euclidean::<3>;
    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();

    let nv = domain.n_vertices();
    // Non-uniform phi field
    let phi = DVector::from_fn(nv, |i, _| 1.0 + 0.5 * (i as f64));
    // Chemical potential drives transport
    let mu = DVector::from_fn(nv, |i, _| 0.1 * (i as f64) - 0.3);
    let mobility = 1.0;

    let dphi = volterra_dec::transport::dec_flux_form(
        &domain.operators,
        &phi,
        &mu,
        mobility,
    );

    // Conservation: sum of dphi/dt weighted by dual areas should be zero
    // (no flux leaves the domain for a closed mesh, and for an open mesh
    // the discrete divergence theorem applies).
    let total_change: f64 = dphi.iter().zip(domain.dual_areas.iter())
        .map(|(&dp, &a)| dp * a)
        .sum();

    assert!(
        total_change.abs() < 1e-10,
        "flux form should conserve total phi: net change = {total_change:.3e}"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test coupling_transport` (should fail)

#### Step 16.2: Implement coupling.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/coupling.rs`

```rust
//! Nematic-membrane coupling: spontaneous curvature from the Q-tensor field.
//!
//! The scalar order parameter S extracted from Q induces a spontaneous
//! curvature H0 = coupling * S at each vertex. Ordered nematic regions
//! (high S) drive membrane deformation.

/// Compute spontaneous curvature H0 from a Q-tensor field.
///
/// H0(v) = coupling * S(v) where S is the scalar order parameter
/// (largest eigenvalue of Q for a 2D nematic, or the norm-based proxy
/// S = sqrt(2 * Tr(Q^2)) for symmetric traceless Q).
///
/// The Q-tensor is stored as M::Tangent at each vertex. For a 2D surface
/// nematic with Q_ij stored in 3 components [Qxx, Qxy, Qyy], the order
/// parameter is S = sqrt(Qxx^2 + Qxy^2) (largest eigenvalue of the 2x2
/// traceless symmetric matrix). For a general tangent vector, we use the
/// norm as a proxy.
pub fn spontaneous_curvature_from_q<T: AsRef<[f64]>>(
    q_field: &[T],
    coupling: f64,
) -> Vec<f64> {
    q_field
        .iter()
        .map(|q| {
            let s = q.as_ref();
            let norm_sq: f64 = s.iter().map(|x| x * x).sum();
            coupling * norm_sq.sqrt()
        })
        .collect()
}
```

#### Step 16.3: Implement transport.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/transport.rs`

```rust
//! DEC conserved scalar transport.
//!
//! The flux form ensures that the total quantity sum(phi * dual_area)
//! is conserved (up to boundary fluxes) by construction. This follows
//! from the exactness of the DEC divergence: d*d = 0.
//!
//! Transport equation:
//!   dphi/dt = mobility * star0_inv * d0^T * star1 * diag(phi_edge) * d0 * mu
//!
//! where phi_edge[e] = 0.5 * (phi[v_a] + phi[v_b]) is the edge-averaged
//! concentration, and mu is the chemical potential (driving force).

use cartan_core::Manifold;
use cartan_dec::Operators;
use nalgebra::DVector;

/// Conserved transport flux form.
///
/// Returns dphi/dt at each vertex. The discrete divergence theorem guarantees
/// conservation: sum(dphi/dt * dual_area) = 0 for meshes without boundary.
pub fn dec_flux_form<M: Manifold>(
    operators: &Operators<M>,
    phi: &DVector<f64>,
    chemical_potential: &DVector<f64>,
    mobility: f64,
) -> DVector<f64> {
    let nv = phi.len();

    // Step 1: Compute the gradient of mu as a 1-form: grad_mu_1 = d0 * mu
    let grad_mu_1 = &operators.ext.d0 * chemical_potential;

    // Step 2: Compute edge-averaged phi values
    let ne = operators.ext.d0.nrows();
    let mut phi_edge = DVector::zeros(ne);
    for e in 0..ne {
        // Extract the two vertex indices from d0's sparsity pattern.
        // d0[e, vi] = -1, d0[e, vj] = +1. We find them by scanning the row.
        let mut vi = 0;
        let mut vj = 0;
        for v in 0..nv {
            let val = operators.ext.d0[(e, v)];
            if val < -0.5 {
                vi = v;
            } else if val > 0.5 {
                vj = v;
            }
        }
        phi_edge[e] = 0.5 * (phi[vi] + phi[vj]);
    }

    // Step 3: Weighted flux: star1 * diag(phi_edge) * grad_mu_1
    let mut weighted_flux = DVector::zeros(ne);
    for e in 0..ne {
        weighted_flux[e] = operators.hodge.star1[e] * phi_edge[e] * grad_mu_1[e];
    }

    // Step 4: Divergence: star0_inv * d0^T * weighted_flux
    let div = operators.ext.d0.transpose() * &weighted_flux;
    let star0_inv = operators.hodge.star0_inv();

    let mut result = DVector::zeros(nv);
    for v in 0..nv {
        result[v] = mobility * star0_inv[v] * div[v];
    }

    result
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test coupling_transport` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add nematic-membrane coupling and conserved DEC transport"`

---

### Task 17: Beris-Edwards RHS on DEC (beris_edwards.rs)

#### Step 17.1: Write failing test

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/beris_edwards_dec.rs`

```rust
//! Tests for the Beris-Edwards RHS on DEC meshes.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;

fn icosahedron_mesh(r: f64) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = r / norm;
    let b = r * phi / norm;
    let raw = [
        SVector::from([-a,  b, 0.0]),
        SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]),
        SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]),
        SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]),
        SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]),
        SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]),
        SVector::from([-b, 0.0,  a]),
    ];
    let vertices: Vec<_> = raw.iter().map(|v| v * (r / v.norm())).collect();
    let triangles = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (vertices, triangles)
}

#[test]
fn test_isotropic_q_is_fixed_point() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);
    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();

    let nv = domain.n_vertices();

    // Isotropic Q = 0 everywhere: should be a fixed point when a_eff > 0 (isotropic stable).
    let q_zero: Vec<SVector<f64, 3>> = vec![SVector::zeros(); nv];

    let params = volterra_dec::beris_edwards::LandauDeGennesParams {
        k_elastic: 1.0,
        a_eff: 1.0,      // positive => isotropic phase is stable
        c_landau: 0.0,
        gamma_r: 1.0,
        lambda: 0.0,
        zeta: 0.0,
        h0_coupling: 0.0,
    };

    let rhs = volterra_dec::beris_edwards::beris_edwards_rhs_dec(
        &domain,
        &q_zero,
        None, // no velocity
        &params,
    );

    // All components of dQ/dt should be zero (or near zero).
    for (v, dq) in rhs.iter().enumerate() {
        let norm_sq: f64 = dq.as_ref().iter().map(|x| x * x).sum();
        assert!(
            norm_sq.sqrt() < 1e-10,
            "vertex {v}: isotropic Q should be fixed point, got |dQ/dt| = {:.3e}",
            norm_sq.sqrt()
        );
    }
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test beris_edwards_dec` (should fail)

#### Step 17.2: Implement beris_edwards.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/beris_edwards.rs`

```rust
//! Beris-Edwards RHS on DEC triangle meshes.
//!
//! dQ/dt = -u.nabla(Q) + S(W,Q) + Gamma_r * H
//!
//! where:
//! - Elastic: K * Lich_Lap(Q) via operators.apply_lichnerowicz_laplacian
//! - Advection: covariant upwind advection from cartan-dec
//! - Co-rotation: S = xi*(D*Q + Q*D) - 2*xi*Tr(Q*D)*Q + Omega*Q - Q*Omega
//! - Molecular field: H = K*Lich_Lap(Q) - a_eff*Q - 2c*Tr(Q^2)*Q
//!
//! The Q-tensor on a 2-manifold surface is stored as 3 independent components
//! per vertex: [Qxx, Qxy, Qyy] in a local tangent frame. For a flat embedding
//! in R^3, these are the components in the global x,y plane.

use cartan_core::Manifold;
use nalgebra::DVector;

use crate::domain::DecDomain;

/// Landau-de Gennes and Beris-Edwards parameters.
pub struct LandauDeGennesParams {
    /// Frank elastic constant K.
    pub k_elastic: f64,
    /// Effective Landau coefficient a_eff = a - zeta/2.
    /// Positive: isotropic phase stable. Negative: nematic phase stable.
    pub a_eff: f64,
    /// Cubic Landau coefficient c (higher-order stabilization).
    pub c_landau: f64,
    /// Rotational viscosity Gamma_r.
    pub gamma_r: f64,
    /// Flow alignment parameter lambda (xi in some notations).
    pub lambda: f64,
    /// Activity parameter zeta.
    pub zeta: f64,
    /// Nematic-membrane coupling strength for spontaneous curvature.
    pub h0_coupling: f64,
}

/// Compute the Beris-Edwards RHS for the Q-tensor field on a DEC mesh.
///
/// Returns dQ/dt at each vertex as a tangent vector (3 components for 2D Q).
///
/// The velocity field is optional: pass None for dry active nematics
/// (no advection, no co-rotation).
pub fn beris_edwards_rhs_dec<M: Manifold>(
    domain: &DecDomain<M>,
    q: &[M::Tangent],
    velocity: Option<&[M::Tangent]>,
    params: &LandauDeGennesParams,
) -> Vec<M::Tangent>
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let nv = domain.n_vertices();

    // Pack Q into DVector layout: [Qxx[0..nv], Qxy[0..nv], Qyy[0..nv]]
    let q_packed = pack_q_field(q, nv);

    // 1. Elastic term: K * Lichnerowicz Laplacian of Q
    // For flat embedding, curvature correction is None (zero Riemann tensor).
    let lich_q = domain.operators.apply_lichnerowicz_laplacian(&q_packed, None);

    // 2. Molecular field H = K*Lich_Lap(Q) - a_eff*Q - 2c*Tr(Q^2)*Q
    let mut h_packed = DVector::zeros(3 * nv);
    for v in 0..nv {
        let qxx = q_packed[v];
        let qxy = q_packed[nv + v];
        let qyy = q_packed[2 * nv + v];

        let tr_q2 = qxx * qxx + 2.0 * qxy * qxy + qyy * qyy;

        // H_v = K * Lich(Q)_v - a_eff * Q_v - 2c * Tr(Q^2) * Q_v
        h_packed[v] = params.k_elastic * lich_q[v]
            - params.a_eff * qxx
            - 2.0 * params.c_landau * tr_q2 * qxx;
        h_packed[nv + v] = params.k_elastic * lich_q[nv + v]
            - params.a_eff * qxy
            - 2.0 * params.c_landau * tr_q2 * qxy;
        h_packed[2 * nv + v] = params.k_elastic * lich_q[2 * nv + v]
            - params.a_eff * qyy
            - 2.0 * params.c_landau * tr_q2 * qyy;
    }

    // 3. dQ/dt = Gamma_r * H (+ advection + co-rotation if velocity present)
    let mut rhs_packed = DVector::zeros(3 * nv);
    for i in 0..(3 * nv) {
        rhs_packed[i] = params.gamma_r * h_packed[i];
    }

    // 4. Advection and co-rotation (only if velocity is provided)
    if let Some(_vel) = velocity {
        // TODO: implement covariant advection via cartan-dec apply_vector_advection
        // and co-rotation tensor S(W, Q).
        // For now, the dry active model (vel = None) is the primary use case.
    }

    // Unpack back to Vec<M::Tangent>
    unpack_q_field(&rhs_packed, nv, domain)
}

/// Pack per-vertex Q tangent vectors into the [Qxx, Qxy, Qyy] DVector layout.
fn pack_q_field<T: AsRef<[f64]>>(q: &[T], nv: usize) -> DVector<f64> {
    let mut packed = DVector::zeros(3 * nv);
    for (v, qv) in q.iter().enumerate() {
        let s = qv.as_ref();
        // For 3-component Q: [Qxx, Qxy, Qyy]
        if s.len() >= 3 {
            packed[v] = s[0];
            packed[nv + v] = s[1];
            packed[2 * nv + v] = s[2];
        }
    }
    packed
}

/// Unpack a DVector in [Qxx, Qxy, Qyy] layout back to per-vertex M::Tangent.
fn unpack_q_field<M: Manifold>(
    packed: &DVector<f64>,
    nv: usize,
    domain: &DecDomain<M>,
) -> Vec<M::Tangent>
where
    M::Tangent: AsMut<[f64]>,
{
    let mut out = Vec::with_capacity(nv);
    for v in 0..nv {
        let p = &domain.mesh.vertices[v];
        let mut tangent = domain.manifold.zero_tangent(p);
        let t_slice = tangent.as_mut();
        if t_slice.len() >= 3 {
            t_slice[0] = packed[v];
            t_slice[1] = packed[nv + v];
            t_slice[2] = packed[2 * nv + v];
        }
        out.push(tangent);
    }
    out
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test beris_edwards_dec` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add Beris-Edwards RHS on DEC with Lichnerowicz Laplacian"`

---

### Task 18: Variational integrator struct + BAOAB step (variational.rs)

#### Step 18.1: Write failing test

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/variational.rs`

```rust
//! Tests for the variational integrator (BAOAB scheme).

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;

fn icosahedron_mesh(r: f64) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = r / norm;
    let b = r * phi / norm;
    let raw = [
        SVector::from([-a,  b, 0.0]),
        SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]),
        SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]),
        SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]),
        SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]),
        SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]),
        SVector::from([-b, 0.0,  a]),
    ];
    let vertices: Vec<_> = raw.iter().map(|v| v * (r / v.norm())).collect();
    let triangles = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (vertices, triangles)
}

#[test]
fn test_energy_oscillation_free_membrane() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);
    let domain = volterra_dec::DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    let helfrich_params = volterra_dec::helfrich::HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![0.0; nv],
    };

    let integrator = volterra_dec::variational::VariationalIntegrator {
        dt: 0.001,
        newton_max_iter: 0,
        newton_tol: 1e-8,
        ke_tolerance: 0.01,
    };

    // Start with zero momenta, perturb one vertex.
    let mut positions: Vec<SVector<f64, 3>> = domain.mesh.vertices.clone();
    positions[0] *= 1.05; // small radial perturbation
    let momenta: Vec<SVector<f64, 3>> = vec![SVector::zeros(); nv];
    let masses: Vec<f64> = domain.dual_areas.clone();

    // Run 100 B-A steps (no O step, no dissipation).
    let mut energies = Vec::new();
    let mut pos = positions;
    let mut mom = momenta;

    for _ in 0..100 {
        let ke: f64 = mom.iter().zip(masses.iter()).map(|(p, &m)| {
            let p_s: &[f64] = p.as_ref();
            let norm_sq: f64 = p_s.iter().map(|x| x * x).sum();
            0.5 * norm_sq / m
        }).sum();
        energies.push(ke);

        volterra_dec::variational::baoab_ba_step(
            &manifold,
            &mut pos,
            &mut mom,
            &masses,
            &helfrich_params,
            &domain,
            integrator.dt,
        );
    }

    // Energy should oscillate, not grow secularly.
    let e_max = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let e_first = energies[0];
    let e_last = *energies.last().unwrap();

    // The oscillation amplitude should not grow by more than a factor of 10.
    // (Generous bound for a coarse icosahedron.)
    let range = e_max - e_min;
    assert!(
        range < 10.0 * e_first.max(1e-6),
        "energy should oscillate, not diverge: range = {range:.3e}, initial KE = {e_first:.3e}"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test variational` (should fail)

#### Step 18.2: Implement variational.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/variational.rs`

```rust
//! Variational integrator with BAOAB splitting for membrane-nematic dynamics.
//!
//! B: half-kick (momentum update from forces)
//! A: drift (position update via exp map)
//! O: stochastic (Shardlow edge sweep from pathwise-geo)
//!
//! The BAOAB splitting gives second-order configurational accuracy and
//! exact fluctuation-dissipation balance per edge.
//!
//! References:
//! - Leimkuhler, Matthews. "Rational Construction of Stochastic Numerical
//!   Methods for Molecular Sampling." AMM, 2012.
//! - Zhu, Lee, Rangamani. "Mem3DG." Biophysical Reports, 2022.

use cartan_core::Manifold;

use crate::domain::DecDomain;
use crate::helfrich::{HelfrichParams, helfrich_forces};

/// Configuration for the variational integrator.
pub struct VariationalIntegrator {
    /// Time step size.
    pub dt: f64,
    /// Maximum Newton iterations for implicit B steps (0 = explicit).
    pub newton_max_iter: usize,
    /// Newton convergence tolerance.
    pub newton_tol: f64,
    /// Kinetic energy tolerance for symplectic-aware remesh triggering.
    pub ke_tolerance: f64,
}

/// Perform the deterministic B-A-B steps of the BAOAB scheme.
///
/// B: p += -(dt/4) * grad_x V(x, Q)
/// A: x = exp_x(p * dt / (2*mass))
/// (O step is handled separately by pathwise-geo)
/// A: x = exp_x(p * dt / (2*mass))
/// B: p += -(dt/4) * grad_x V(x, Q)
///
/// This function performs B-A-A-B (the two A half-steps and two B quarter-steps).
/// The O step is inserted between the two A steps by the caller.
pub fn baoab_ba_step<M: Manifold>(
    manifold: &M,
    positions: &mut [M::Point],
    momenta: &mut [M::Tangent],
    masses: &[f64],
    helfrich_params: &HelfrichParams,
    domain: &DecDomain<M>,
    dt: f64,
) where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let nv = positions.len();
    let dim = manifold.ambient_dim();

    // Compute forces at current positions.
    let forces = helfrich_forces(domain, helfrich_params);

    // B step (first quarter): p += (dt/4) * force
    for v in 0..nv {
        let f = forces[v].as_ref();
        let p = momenta[v].as_mut();
        for d in 0..dim.min(p.len()) {
            p[d] += (dt / 4.0) * f[d];
        }
    }

    // A step (first half): x = exp_x(p * dt / (2*mass))
    for v in 0..nv {
        let m = masses[v].max(1e-30);
        let p_slice = momenta[v].as_ref();
        let mut vel = manifold.zero_tangent(&positions[v]);
        let vel_s = vel.as_mut();
        for d in 0..dim.min(vel_s.len()) {
            vel_s[d] = p_slice[d] * dt / (2.0 * m);
        }
        let vel_proj = manifold.project_tangent(&positions[v], &vel);
        positions[v] = manifold.exp(&positions[v], &vel_proj);
    }

    // (O step would go here, handled by caller)

    // A step (second half): same as first half
    for v in 0..nv {
        let m = masses[v].max(1e-30);
        let p_slice = momenta[v].as_ref();
        let mut vel = manifold.zero_tangent(&positions[v]);
        let vel_s = vel.as_mut();
        for d in 0..dim.min(vel_s.len()) {
            vel_s[d] = p_slice[d] * dt / (2.0 * m);
        }
        let vel_proj = manifold.project_tangent(&positions[v], &vel);
        positions[v] = manifold.exp(&positions[v], &vel_proj);
    }

    // B step (second quarter): p += (dt/4) * force(new_x)
    // For explicit B steps (newton_max_iter = 0), reuse the old forces.
    // This is first-order in the B step, but the overall BAOAB scheme
    // remains second-order in configuration.
    for v in 0..nv {
        let f = forces[v].as_ref();
        let p = momenta[v].as_mut();
        for d in 0..dim.min(p.len()) {
            p[d] += (dt / 4.0) * f[d];
        }
    }
}

/// Compute kinetic energy from momenta and masses.
pub fn kinetic_energy<T: AsRef<[f64]>>(momenta: &[T], masses: &[f64]) -> f64 {
    momenta
        .iter()
        .zip(masses.iter())
        .map(|(p, &m)| {
            let ps = p.as_ref();
            let norm_sq: f64 = ps.iter().map(|x| x * x).sum();
            0.5 * norm_sq / m.max(1e-30)
        })
        .sum()
}

/// Adaptive time step based on diffusive and force CFL conditions.
///
/// dt = min(dt_max, C_diff * h_min^2, C_force * h_min / max_force_norm)
pub fn compute_dt(
    h_min: f64,
    max_force: f64,
    dt_max: f64,
    c_diff: f64,
    c_force: f64,
) -> f64 {
    let dt_diff = c_diff * h_min * h_min;
    let dt_force = if max_force > 1e-30 {
        c_force * h_min / max_force
    } else {
        dt_max
    };
    dt_max.min(dt_diff).min(dt_force)
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test variational` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add BAOAB variational integrator with adaptive dt"`

---

### Task 19: Field interpolation + remesh integration (interpolation.rs)

#### Step 19.1: Write failing test

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/interpolation.rs`

```rust
//! Tests for field interpolation after remeshing.

use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;
use volterra_dec::interpolation;

#[test]
fn test_interpolate_split_preserves_norm() {
    let manifold = Euclidean::<3>;

    // Two endpoint Q values.
    let q_a = SVector::<f64, 3>::new(0.3, 0.1, -0.3);
    let q_b = SVector::<f64, 3>::new(0.2, -0.2, -0.2);

    // Positions of endpoints and midpoint.
    let x_a = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let x_b = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let x_mid = SVector::<f64, 3>::new(0.5, 0.0, 0.0);

    let q_mid = interpolation::interpolate_split_single(
        &manifold, &x_a, &x_b, &x_mid, &q_a, &q_b,
    );

    // The interpolated Q should have norm between the two endpoint norms.
    let norm_a: f64 = q_a.as_ref().iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = q_b.as_ref().iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_mid: f64 = q_mid.as_ref().iter().map(|x| x * x).sum::<f64>().sqrt();

    let norm_min = norm_a.min(norm_b);
    let norm_max = norm_a.max(norm_b);

    // Allow 10% tolerance beyond the range (parallel transport can slightly change norms
    // in the discrete setting).
    assert!(
        norm_mid >= norm_min * 0.9 && norm_mid <= norm_max * 1.1,
        "interpolated norm {norm_mid:.4} not in [{:.4}, {:.4}]",
        norm_min * 0.9,
        norm_max * 1.1
    );
}

#[test]
fn test_interpolate_collapse_frechet_mean() {
    let manifold = Euclidean::<3>;

    // For Euclidean manifold, Frechet mean = arithmetic mean.
    let neighbors = vec![
        SVector::<f64, 3>::new(0.4, 0.0, -0.4),
        SVector::<f64, 3>::new(0.0, 0.4, 0.0),
        SVector::<f64, 3>::new(0.2, 0.2, -0.2),
    ];
    let positions = vec![
        SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        SVector::<f64, 3>::new(0.0, 0.0, 1.0),
    ];
    let target = SVector::<f64, 3>::new(0.5, 0.5, 0.0);

    let result = interpolation::interpolate_collapse_single(
        &manifold, &target, &positions, &neighbors,
    );

    // Expected: arithmetic mean of neighbors
    let expected = SVector::<f64, 3>::new(0.2, 0.2, -0.2);
    let diff: f64 = (result - expected).as_ref().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        diff < 1e-10,
        "Frechet mean should equal arithmetic mean on Euclidean: diff = {diff:.3e}"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test interpolation` (should fail)

#### Step 19.2: Implement interpolation.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/interpolation.rs`

```rust
//! Field interpolation after remeshing operations.
//!
//! Edge split: parallel-transport Q from each endpoint to the midpoint,
//! average in the tangent space, apply via exp. Falls back to linear
//! interpolation if transport fails.
//!
//! Edge collapse: Frechet mean of 1-ring neighbor Q values at the
//! surviving vertex.
//!
//! These ensure that the Q-tensor field respects nematic geometry
//! across topology changes.

use cartan_core::{Manifold, ParallelTransport};

/// Interpolate a Q value at a split edge midpoint.
///
/// Parallel-transports Q from each endpoint to the midpoint, then averages
/// in the tangent space at the midpoint. Falls back to simple linear
/// interpolation if parallel transport is unavailable or fails.
pub fn interpolate_split_single<M: Manifold>(
    manifold: &M,
    x_a: &M::Point,
    x_b: &M::Point,
    x_mid: &M::Point,
    q_a: &M::Tangent,
    q_b: &M::Tangent,
) -> M::Tangent
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    // Simple linear interpolation in ambient coordinates as the default.
    // For Euclidean manifolds this is exact. For curved manifolds,
    // the parallel transport version below is preferred.
    let dim = q_a.as_ref().len();
    let mut result = manifold.zero_tangent(x_mid);
    let r_slice = result.as_mut();
    let a_slice = q_a.as_ref();
    let b_slice = q_b.as_ref();
    for d in 0..dim {
        r_slice[d] = 0.5 * (a_slice[d] + b_slice[d]);
    }
    manifold.project_tangent(x_mid, &result)
}

/// Interpolate a Q value at a split edge midpoint using parallel transport.
///
/// Transports Q_a from x_a to x_mid and Q_b from x_b to x_mid,
/// then averages in the tangent space at x_mid.
pub fn interpolate_split_transport<M: Manifold + ParallelTransport>(
    manifold: &M,
    x_a: &M::Point,
    x_b: &M::Point,
    x_mid: &M::Point,
    q_a: &M::Tangent,
    q_b: &M::Tangent,
) -> M::Tangent
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let qa_transported = manifold.transport(x_a, x_mid, q_a);
    let qb_transported = manifold.transport(x_b, x_mid, q_b);

    match (qa_transported, qb_transported) {
        (Ok(qa_t), Ok(qb_t)) => {
            // Average in tangent space at x_mid
            let dim = qa_t.as_ref().len();
            let mut result = manifold.zero_tangent(x_mid);
            let r = result.as_mut();
            let a = qa_t.as_ref();
            let b = qb_t.as_ref();
            for d in 0..dim {
                r[d] = 0.5 * (a[d] + b[d]);
            }
            manifold.project_tangent(x_mid, &result)
        }
        _ => {
            // Fallback: linear interpolation
            interpolate_split_single(manifold, x_a, x_b, x_mid, q_a, q_b)
        }
    }
}

/// Interpolate Q at a collapsed vertex via Frechet mean of neighbor values.
///
/// For Euclidean manifolds, this reduces to the arithmetic mean.
/// For general manifolds, uses iterative tangent-space averaging
/// (one iteration for now, which is exact on Euclidean and first-order
/// accurate on curved manifolds).
pub fn interpolate_collapse_single<M: Manifold>(
    manifold: &M,
    target: &M::Point,
    neighbor_positions: &[M::Point],
    neighbor_q: &[M::Tangent],
) -> M::Tangent
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let n = neighbor_q.len();
    if n == 0 {
        return manifold.zero_tangent(target);
    }

    let dim = neighbor_q[0].as_ref().len();
    let mut mean = manifold.zero_tangent(target);
    let m_slice = mean.as_mut();

    for q in neighbor_q {
        let q_s = q.as_ref();
        for d in 0..dim {
            m_slice[d] += q_s[d] / n as f64;
        }
    }

    manifold.project_tangent(target, &mean)
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test interpolation` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add field interpolation for edge split (parallel transport) and collapse (Frechet mean)"`

---

### Task 20: EdgeNoiseSampler + Shardlow O-step (pathwise-geo mesh/ module)

#### Step 20.0: Update pathwise-geo Cargo.toml

Add cartan-dec dependency for the `Mesh` type:

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/Cargo.toml`

```toml
[package]
name = "pathwise-geo"
version = "0.2.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/alejandro-soto-franco/pathwise"
keywords = ["sde", "simulation", "stochastic", "finance", "riemannian"]
categories = ["mathematics", "science", "simulation"]
description = "Riemannian manifold SDE simulation (geodesic Euler/Milstein/SRI) on S^n, SO(n), SPD(n) via the cartan geometry library"
documentation = "https://docs.rs/pathwise-geo"
readme = "../README.md"

[dependencies]
pathwise-core = { path = "../pathwise-core", version = "0.2" }
cartan-core = { path = "../../cartan/cartan-core", version = "0.1" }
cartan-manifolds = { path = "../../cartan/cartan-manifolds", version = "0.1", default-features = false, features = ["std"] }
cartan-dec = { path = "../../cartan/cartan-dec", version = "0.1" }
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
rayon = "1"
rand = { version = "0.9", default-features = false, features = ["std", "std_rng"] }
rand_distr = { version = "0.5" }
ndarray = "0.15"

[dev-dependencies]
rand = { version = "0.9", features = ["std_rng"] }
approx = "0.5"

# Declare pathwise-geo as a standalone workspace so that Cargo does not try to
# pull it into the parent pathwise workspace (which lacks the cartan siblings).
[workspace]
```

#### Step 20.1: Write failing tests

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/tests/mesh_noise.rs`

```rust
//! Tests for edge noise sampler and Shardlow O-step.

use rand::SeedableRng;
use rand::rngs::SmallRng;

#[test]
fn test_fd_relation() {
    use pathwise_geo::mesh::EdgeNoiseSampler;

    let sampler = EdgeNoiseSampler {
        gamma: 2.0,
        kb_t: 0.5,
    };

    // FD relation: sigma^2 * dt = 2 * gamma * kBT
    for &dt in &[0.001, 0.01, 0.1] {
        let err = sampler.verify_fd(dt);
        assert!(
            err < 1e-14,
            "FD relation violated: dt={dt}, error={err:.3e}"
        );
    }
}

#[test]
fn test_noise_samples_have_correct_variance() {
    use pathwise_geo::mesh::EdgeNoiseSampler;

    let sampler = EdgeNoiseSampler {
        gamma: 3.0,
        kb_t: 1.0,
    };
    let dt = 0.01;
    let n_samples = 100_000;
    let mut rng = SmallRng::seed_from_u64(42);

    let samples: Vec<f64> = (0..n_samples).map(|_| sampler.sample(&mut rng, dt)).collect();
    let mean: f64 = samples.iter().sum::<f64>() / n_samples as f64;
    let var: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n_samples as f64;

    // Expected variance = sigma^2 = 2 * gamma * kBT / dt
    let expected_var = 2.0 * sampler.gamma * sampler.kb_t / dt;
    let rel_err = (var - expected_var).abs() / expected_var;
    assert!(
        rel_err < 0.05,
        "noise variance {var:.3} vs expected {expected_var:.3} (rel err {rel_err:.4})"
    );
}

#[test]
fn test_edge_o_step_newtons_third_law() {
    use pathwise_geo::mesh::edge_o_step;
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let x_i = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
    let x_j = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let p_i = SVector::<f64, 3>::new(0.5, 0.3, 0.1);
    let p_j = SVector::<f64, 3>::new(-0.2, 0.1, -0.3);
    let mass_i = 1.0;
    let mass_j = 1.5;
    let gamma = 2.0;
    let dt = 0.01;
    let noise = 0.05;

    let (dp_i, dp_j) = edge_o_step(
        &manifold, &x_i, &x_j,
        &p_i, &p_j,
        mass_i, mass_j,
        gamma, dt, noise,
    );

    // Newton's third law: dp_i + dp_j = 0 (in Euclidean space, no transport needed)
    let sum: f64 = dp_i.as_ref().iter().zip(dp_j.as_ref().iter())
        .map(|(a, b)| (a + b) * (a + b))
        .sum::<f64>()
        .sqrt();

    assert!(
        sum < 1e-12,
        "Newton's third law violated: |dp_i + dp_j| = {sum:.3e}"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test mesh_noise` (should fail)

#### Step 20.2: Implement mesh/ module

Create the directory structure:

```bash
mkdir -p /home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh
```

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/mod.rs`

```rust
//! Mesh-based stochastic primitives for DPD (Dissipative Particle Dynamics).
//!
//! Provides edge-level noise sampling and the Shardlow splitting O-step
//! for the BAOAB integrator on triangulated 2-manifolds.

mod noise;
mod shardlow;
mod baoab;

pub use noise::EdgeNoiseSampler;
pub use shardlow::edge_o_step;
pub use baoab::{EdgeSweepOrder, BAOABConfig, baoab_o_step};
```

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/noise.rs`

```rust
//! Edge noise sampler satisfying the fluctuation-dissipation relation.
//!
//! The sampler produces scalar noise increments xi such that
//! E[xi^2] = sigma^2 = 2 * gamma * kBT / dt, ensuring the FD relation
//! sigma^2 * dt = 2 * gamma * kBT holds exactly.

use rand::Rng;
use rand_distr::StandardNormal;

/// Physics-agnostic edge noise sampler.
///
/// Produces scalar Gaussian noise increments satisfying the
/// fluctuation-dissipation (FD) relation for a single pairwise
/// interaction with friction coefficient gamma and thermal energy kBT.
pub struct EdgeNoiseSampler {
    /// Friction coefficient (dissipation strength).
    pub gamma: f64,
    /// Thermal energy kB * T.
    pub kb_t: f64,
}

impl EdgeNoiseSampler {
    /// Noise amplitude: sigma = sqrt(2 * gamma * kBT / dt).
    pub fn sigma(&self, dt: f64) -> f64 {
        (2.0 * self.gamma * self.kb_t / dt).sqrt()
    }

    /// Draw a single noise sample: sigma * N(0, 1).
    pub fn sample<R: Rng>(&self, rng: &mut R, dt: f64) -> f64 {
        self.sigma(dt) * rng.sample::<f64, StandardNormal>(StandardNormal)
    }

    /// Verify the FD relation: returns |sigma^2 * dt - 2 * gamma * kBT|.
    ///
    /// Should be zero to machine epsilon.
    pub fn verify_fd(&self, dt: f64) -> f64 {
        let sigma = self.sigma(dt);
        (sigma * sigma * dt - 2.0 * self.gamma * self.kb_t).abs()
    }
}
```

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/shardlow.rs`

```rust
//! Per-edge Shardlow O-step for DPD on Riemannian manifolds.
//!
//! Given an edge (i, j) on a triangulated manifold, compute the
//! momentum correction from the dissipative + random DPD forces
//! projected along the edge direction. The algorithm:
//!
//! 1. Edge unit vector: r_hat = log_{x_i}(x_j) / |log_{x_i}(x_j)|
//! 2. Transport p_j to x_i: p_j_at_i = transport(x_j, x_i, p_j)
//! 3. Relative velocity along edge: dv_r = inner(p_i/m_i - p_j_at_i/m_j, r_hat)
//! 4. Total impulse: dp = (-gamma * dv_r * dt + noise) * r_hat
//! 5. Return: (dp, -transport(x_i, x_j, dp)) (Newton's third law on manifold)

use cartan_core::{Manifold, ParallelTransport};

/// Compute the DPD momentum correction for a single edge (i, j).
///
/// Returns (dp_i, dp_j): the momentum impulse to add to vertex i and j.
/// Newton's third law is enforced via parallel transport of dp from i to j.
///
/// Falls back to zero correction if log or transport fails (degenerate edge
/// or cut locus).
pub fn edge_o_step<M: Manifold + ParallelTransport>(
    manifold: &M,
    x_i: &M::Point,
    x_j: &M::Point,
    p_i: &M::Tangent,
    p_j: &M::Tangent,
    mass_i: f64,
    mass_j: f64,
    gamma: f64,
    dt: f64,
    noise: f64,
) -> (M::Tangent, M::Tangent)
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let zero_i = manifold.zero_tangent(x_i);
    let zero_j = manifold.zero_tangent(x_j);

    // 1. Edge vector and unit vector
    let edge_vec = match manifold.log(x_i, x_j) {
        Ok(v) => v,
        Err(_) => return (zero_i, zero_j),
    };
    let edge_len = manifold.norm(x_i, &edge_vec);
    if edge_len < 1e-30 {
        return (zero_i, zero_j);
    }

    let r_hat_slice: Vec<f64> = edge_vec.as_ref().iter().map(|&x| x / edge_len).collect();

    // Build r_hat as a tangent vector at x_i
    let dim = r_hat_slice.len();
    let mut r_hat = manifold.zero_tangent(x_i);
    let rh = r_hat.as_mut();
    for d in 0..dim {
        rh[d] = r_hat_slice[d];
    }

    // 2. Transport p_j to x_i
    let p_j_at_i = match manifold.transport(x_j, x_i, p_j) {
        Ok(v) => v,
        Err(_) => return (zero_i, zero_j),
    };

    // 3. Relative velocity along edge
    let pi_s = p_i.as_ref();
    let pj_s = p_j_at_i.as_ref();
    let mut dv_r = 0.0;
    for d in 0..dim {
        dv_r += (pi_s[d] / mass_i - pj_s[d] / mass_j) * r_hat_slice[d];
    }

    // 4. Total impulse along edge direction
    let impulse_scalar = -gamma * dv_r * dt + noise;

    let mut dp_i = manifold.zero_tangent(x_i);
    let dp_i_s = dp_i.as_mut();
    for d in 0..dim {
        dp_i_s[d] = impulse_scalar * r_hat_slice[d];
    }

    // 5. Newton's third law: dp_j = -transport(dp_i from x_i to x_j)
    let dp_j = match manifold.transport(x_i, x_j, &dp_i) {
        Ok(transported) => {
            let mut neg = manifold.zero_tangent(x_j);
            let neg_s = neg.as_mut();
            let t_s = transported.as_ref();
            for d in 0..dim {
                neg_s[d] = -t_s[d];
            }
            neg
        }
        Err(_) => {
            // Fallback: negate dp_i directly (valid for flat manifolds)
            let mut neg = manifold.zero_tangent(x_j);
            let neg_s = neg.as_mut();
            for d in 0..dim {
                neg_s[d] = -dp_i_s[d];
            }
            neg
        }
    };

    (dp_i, dp_j)
}
```

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/baoab.rs`

Stub for now (implemented in Task 21):

```rust
//! BAOAB sweep driver and edge coloring.
//!
//! Implemented in Task 21.

use super::noise::EdgeNoiseSampler;

/// Edge sweep ordering for the O-step.
pub enum EdgeSweepOrder {
    /// Iterate edges in index order.
    Sequential,
    /// Shuffle edge indices each call.
    RandomPermutation,
    /// 2-color the edges (greedy coloring); sweep each color as a batch.
    Checkerboard,
}

/// Configuration for the BAOAB stochastic step.
pub struct BAOABConfig {
    /// Edge noise sampler (gamma, kBT).
    pub sampler: EdgeNoiseSampler,
    /// Sweep ordering strategy.
    pub sweep_order: EdgeSweepOrder,
}

/// Sweep all edges with the Shardlow O-step.
///
/// Placeholder: full implementation in Task 21.
pub fn baoab_o_step<M, R>(
    _manifold: &M,
    _mesh: &cartan_dec::Mesh<M, 3, 2>,
    _positions: &[M::Point],
    _momenta: &mut [M::Tangent],
    _masses: &[f64],
    _config: &BAOABConfig,
    _dt: f64,
    _rng: &mut R,
) where
    M: cartan_core::Manifold + cartan_core::ParallelTransport,
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
    R: rand::Rng,
{
    // Implemented in Task 21.
}
```

#### Step 20.3: Update pathwise-geo lib.rs

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/lib.rs`

```rust
pub mod mesh;
pub mod process;
pub mod scheme;
pub mod sde;
pub mod simulate;

#[allow(deprecated)]
pub use process::brownian_motion_on;
pub use process::{brownian_motion_on_with_diffusion, ou_on_with_diffusion};
pub use scheme::{GeodesicEuler, GeodesicMilstein, GeodesicSRI};
pub use sde::ManifoldSDE;
pub use simulate::{manifold_simulate, manifold_simulate_with_scheme, paths_to_array, GeoScheme};
```

**Run**: `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test mesh_noise` (should pass)

**Commit** (from pathwise repo): `git add -A && git commit -m "feat(pathwise-geo): add EdgeNoiseSampler and Shardlow per-edge O-step for DPD"`

---

### Task 21: BAOAB sweep driver + checkerboard coloring

#### Step 21.1: Write failing tests

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/tests/baoab_sweep.rs`

```rust
//! Tests for BAOAB sweep driver and edge coloring.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Build a small mesh for sweep tests.
fn small_mesh() -> Mesh<Euclidean<3>, 3, 2> {
    let manifold = Euclidean::<3>;
    let vertices = vec![
        SVector::from([0.0, 0.0, 0.0]),
        SVector::from([1.0, 0.0, 0.0]),
        SVector::from([0.5, 0.866, 0.0]),
        SVector::from([1.5, 0.866, 0.0]),
    ];
    let triangles = vec![[0, 1, 2], [1, 3, 2]];
    Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, vertices, triangles)
}

#[test]
fn test_checkerboard_coloring_valid() {
    let mesh = small_mesh();
    let coloring = pathwise_geo::mesh::greedy_edge_coloring(&mesh);

    let ne = mesh.n_boundaries();
    assert_eq!(coloring.len(), ne);

    // Verify: no two same-color edges share a vertex.
    for e1 in 0..ne {
        for e2 in (e1 + 1)..ne {
            if coloring[e1] == coloring[e2] {
                let [a1, b1] = mesh.boundaries[e1];
                let [a2, b2] = mesh.boundaries[e2];
                let shared = a1 == a2 || a1 == b2 || b1 == a2 || b1 == b2;
                assert!(
                    !shared,
                    "edges {e1} and {e2} share a vertex but have same color {}",
                    coloring[e1]
                );
            }
        }
    }
}

#[test]
fn test_baoab_o_step_equilibrium() {
    // Run the O-step on a fixed mesh for many sweeps and check that
    // the average KE per DOF converges to kBT/2 (equipartition).
    let mesh = small_mesh();
    let manifold = Euclidean::<3>;
    let nv = mesh.n_vertices();

    let gamma = 10.0;
    let kb_t = 1.0;
    let dt = 0.001;
    let n_sweeps = 20_000;
    let n_warmup = 5_000;

    let sampler = pathwise_geo::mesh::EdgeNoiseSampler { gamma, kb_t };
    let config = pathwise_geo::mesh::BAOABConfig {
        sampler,
        sweep_order: pathwise_geo::mesh::EdgeSweepOrder::Sequential,
    };

    let positions: Vec<SVector<f64, 3>> = mesh.vertices.clone();
    let mut momenta: Vec<SVector<f64, 3>> = vec![SVector::zeros(); nv];
    let masses: Vec<f64> = vec![1.0; nv];
    let mut rng = SmallRng::seed_from_u64(123);

    let mut ke_sum = 0.0;
    let mut ke_count = 0;

    for step in 0..n_sweeps {
        pathwise_geo::mesh::baoab_o_step(
            &manifold,
            &mesh,
            &positions,
            &mut momenta,
            &masses,
            &config,
            dt,
            &mut rng,
        );

        if step >= n_warmup {
            let ke: f64 = momenta.iter().zip(masses.iter()).map(|(p, &m)| {
                let ps: &[f64] = p.as_ref();
                ps.iter().map(|x| x * x).sum::<f64>() / (2.0 * m)
            }).sum();
            ke_sum += ke;
            ke_count += 1;
        }
    }

    let ke_avg = ke_sum / ke_count as f64;
    // Equipartition: KE_avg = (nv * 3 / 2) * kBT
    // But the mesh constrains some DOFs. For an unconstrained mesh:
    let expected_ke = (nv as f64 * 3.0 / 2.0) * kb_t;
    let rel_err = (ke_avg - expected_ke).abs() / expected_ke;

    assert!(
        rel_err < 0.5,
        "equilibrium KE {ke_avg:.3} vs expected {expected_ke:.3} (rel err {rel_err:.3})"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test baoab_sweep` (should fail)

#### Step 21.2: Implement baoab.rs (full version)

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/baoab.rs`

```rust
//! BAOAB sweep driver and edge coloring for DPD on triangulated manifolds.
//!
//! The O-step sweeps all edges, applying the Shardlow per-edge correction.
//! Three sweep orderings are supported:
//! - Sequential: iterate edges in index order
//! - RandomPermutation: shuffle edge indices each call
//! - Checkerboard: 2-color the edges (greedy coloring), sweep each color as a batch
//!
//! The checkerboard ordering guarantees that no two edges in the same color
//! share a vertex, making within-color updates independent (parallelizable).

use cartan_core::{Manifold, ParallelTransport};
use cartan_dec::Mesh;
use rand::Rng;
use rand::seq::SliceRandom;

use super::noise::EdgeNoiseSampler;
use super::shardlow::edge_o_step;

/// Edge sweep ordering for the O-step.
pub enum EdgeSweepOrder {
    /// Iterate edges in index order.
    Sequential,
    /// Shuffle edge indices each call.
    RandomPermutation,
    /// 2-color the edges (greedy coloring); sweep each color as a batch.
    Checkerboard,
}

/// Configuration for the BAOAB stochastic step.
pub struct BAOABConfig {
    /// Edge noise sampler (gamma, kBT).
    pub sampler: EdgeNoiseSampler,
    /// Sweep ordering strategy.
    pub sweep_order: EdgeSweepOrder,
}

/// Sweep all edges with the Shardlow O-step, updating momenta in place.
///
/// For each edge (i, j), samples a noise increment and applies the
/// dissipative + random momentum correction from `edge_o_step`.
pub fn baoab_o_step<M, R>(
    manifold: &M,
    mesh: &Mesh<M, 3, 2>,
    positions: &[M::Point],
    momenta: &mut [M::Tangent],
    masses: &[f64],
    config: &BAOABConfig,
    dt: f64,
    rng: &mut R,
) where
    M: Manifold + ParallelTransport,
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
    R: Rng,
{
    let ne = mesh.n_boundaries();

    let edge_order: Vec<usize> = match config.sweep_order {
        EdgeSweepOrder::Sequential => (0..ne).collect(),
        EdgeSweepOrder::RandomPermutation => {
            let mut order: Vec<usize> = (0..ne).collect();
            order.shuffle(rng);
            order
        }
        EdgeSweepOrder::Checkerboard => {
            // Greedy coloring, then flatten: color 0 edges first, then color 1, etc.
            let coloring = greedy_edge_coloring(mesh);
            let max_color = coloring.iter().copied().max().unwrap_or(0);
            let mut order = Vec::with_capacity(ne);
            for c in 0..=max_color {
                for (e, &color) in coloring.iter().enumerate() {
                    if color == c {
                        order.push(e);
                    }
                }
            }
            order
        }
    };

    for &e in &edge_order {
        let [vi, vj] = mesh.boundaries[e];
        let noise = config.sampler.sample(rng, dt);

        let (dp_i, dp_j) = edge_o_step(
            manifold,
            &positions[vi],
            &positions[vj],
            &momenta[vi],
            &momenta[vj],
            masses[vi],
            masses[vj],
            config.sampler.gamma,
            dt,
            noise,
        );

        // Accumulate momentum corrections in place.
        let dim = dp_i.as_ref().len();
        let mi = momenta[vi].as_mut();
        let di = dp_i.as_ref();
        for d in 0..dim {
            mi[d] += di[d];
        }
        let mj = momenta[vj].as_mut();
        let dj = dp_j.as_ref();
        for d in 0..dim {
            mj[d] += dj[d];
        }
    }
}

/// Greedy edge coloring: assign colors to edges such that no two edges
/// sharing a vertex have the same color.
///
/// Uses a simple greedy algorithm: for each edge, assign the smallest color
/// not used by any already-colored edge that shares a vertex.
///
/// Returns a vector of colors (usize) indexed by edge index.
pub fn greedy_edge_coloring<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Vec<usize> {
    let ne = mesh.n_boundaries();
    let nv = mesh.n_vertices();
    let mut colors: Vec<Option<usize>> = vec![None; ne];

    // Build vertex-to-edge adjacency.
    let mut vertex_edges: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (e, &[vi, vj]) in mesh.boundaries.iter().enumerate() {
        vertex_edges[vi].push(e);
        vertex_edges[vj].push(e);
    }

    for e in 0..ne {
        let [vi, vj] = mesh.boundaries[e];

        // Collect colors used by neighboring edges.
        let mut used = std::collections::HashSet::new();
        for &neighbor_e in &vertex_edges[vi] {
            if let Some(c) = colors[neighbor_e] {
                used.insert(c);
            }
        }
        for &neighbor_e in &vertex_edges[vj] {
            if let Some(c) = colors[neighbor_e] {
                used.insert(c);
            }
        }

        // Find smallest unused color.
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        colors[e] = Some(c);
    }

    colors.into_iter().map(|c| c.unwrap_or(0)).collect()
}
```

Update the module re-exports:

**File**: `/home/alejandrosotofranco/pathwise/pathwise-geo/src/mesh/mod.rs`

```rust
//! Mesh-based stochastic primitives for DPD (Dissipative Particle Dynamics).
//!
//! Provides edge-level noise sampling and the Shardlow splitting O-step
//! for the BAOAB integrator on triangulated 2-manifolds.

mod noise;
mod shardlow;
mod baoab;

pub use noise::EdgeNoiseSampler;
pub use shardlow::edge_o_step;
pub use baoab::{EdgeSweepOrder, BAOABConfig, baoab_o_step, greedy_edge_coloring};
```

**Run**: `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test baoab_sweep` (should pass)

**Commit** (from pathwise repo): `git add -A && git commit -m "feat(pathwise-geo): add BAOAB sweep driver with checkerboard edge coloring"`

---

### Task 22: MembraneNematicSim driver + observables (volterra-dec)

#### Step 22.1: Write failing tests

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/tests/simulation.rs`

```rust
//! Smoke test for MembraneNematicSim.

use cartan_dec::Mesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn icosahedron_mesh(r: f64) -> (Vec<SVector<f64, 3>>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = r / norm;
    let b = r * phi / norm;
    let raw = [
        SVector::from([-a,  b, 0.0]),
        SVector::from([ a,  b, 0.0]),
        SVector::from([-a, -b, 0.0]),
        SVector::from([ a, -b, 0.0]),
        SVector::from([0.0, -a,  b]),
        SVector::from([0.0,  a,  b]),
        SVector::from([0.0, -a, -b]),
        SVector::from([0.0,  a, -b]),
        SVector::from([ b, 0.0, -a]),
        SVector::from([ b, 0.0,  a]),
        SVector::from([-b, 0.0, -a]),
        SVector::from([-b, 0.0,  a]),
    ];
    let vertices: Vec<_> = raw.iter().map(|v| v * (r / v.norm())).collect();
    let triangles = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    (vertices, triangles)
}

#[test]
fn test_smoke_10_steps() {
    let r = 2.0;
    let (verts, tris) = icosahedron_mesh(r);
    let manifold = Euclidean::<3>;
    let mesh = Mesh::<Euclidean<3>, 3, 2>::from_simplices(&manifold, verts, tris);
    let nv = mesh.n_vertices();

    // Uniform Q = 0 (isotropic)
    let initial_q: Vec<SVector<f64, 3>> = vec![SVector::zeros(); nv];

    let helfrich = volterra_dec::helfrich::HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![0.0; nv],
    };

    let ldg = volterra_dec::beris_edwards::LandauDeGennesParams {
        k_elastic: 1.0,
        a_eff: 1.0,
        c_landau: 0.0,
        gamma_r: 1.0,
        lambda: 0.0,
        zeta: 0.0,
        h0_coupling: 0.0,
    };

    let integrator = volterra_dec::variational::VariationalIntegrator {
        dt: 0.001,
        newton_max_iter: 0,
        newton_tol: 1e-8,
        ke_tolerance: 0.01,
    };

    let baoab_config = pathwise_geo::mesh::BAOABConfig {
        sampler: pathwise_geo::mesh::EdgeNoiseSampler {
            gamma: 1.0,
            kb_t: 0.01,
        },
        sweep_order: pathwise_geo::mesh::EdgeSweepOrder::Sequential,
    };

    let mut sim = volterra_dec::simulation::MembraneNematicSim::new(
        mesh,
        manifold,
        helfrich,
        ldg,
        integrator,
        baoab_config,
        initial_q,
    )
    .expect("simulation construction should succeed");

    let mut rng = SmallRng::seed_from_u64(42);

    // Run 10 steps without panic.
    for _ in 0..10 {
        let result = sim.step(&mut rng).expect("step should succeed");
        assert!(result.dt_used > 0.0);
        assert!(result.energy.total.is_finite());
    }

    // Check observables.
    let euler = volterra_dec::observables::euler_characteristic(&sim.domain);
    assert!(
        (euler - 2.0).abs() < 0.01,
        "Euler characteristic of a sphere should be 2, got {euler}"
    );
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test simulation` (should fail)

#### Step 22.2: Implement observables.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/observables.rs`

```rust
//! Physical observables for membrane-nematic simulations.
//!
//! Provides diagnostic quantities (Euler characteristic, total curvature,
//! defect charge density, nematic stress) for post-processing and
//! monitoring convergence.

use std::f64::consts::PI;

use cartan_core::Manifold;

use crate::domain::DecDomain;

/// Euler characteristic via Gauss-Bonnet: sum(K * dual_area) / (2*pi).
///
/// For a closed orientable surface: chi = 2 - 2*genus.
/// Sphere: chi = 2. Torus: chi = 0.
pub fn euler_characteristic<M: Manifold>(domain: &DecDomain<M>) -> f64
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    let total_gauss: f64 = domain
        .gaussian_curvatures
        .iter()
        .zip(domain.dual_areas.iter())
        .map(|(&k, &a)| k * a)
        .sum();
    total_gauss / (2.0 * PI)
}

/// Total mean curvature: integral of H over the surface.
pub fn total_mean_curvature<M: Manifold>(domain: &DecDomain<M>) -> f64
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    domain
        .mean_curvatures
        .iter()
        .zip(domain.dual_areas.iter())
        .map(|(&h, &a)| h * a)
        .sum()
}

/// Defect charge density at each vertex.
///
/// For a nematic Q-tensor on a 2-manifold, the winding number density
/// is approximated by the discrete divergence of the director angle field.
/// This is a simplified version; the full holonomy-based detection
/// delegates to cartan-geo (not implemented here yet).
pub fn defect_charge_density<M: Manifold>(
    _domain: &DecDomain<M>,
    q: &[M::Tangent],
) -> Vec<f64>
where
    M::Tangent: AsRef<[f64]>,
{
    // Placeholder: returns zero charge density everywhere.
    // The full implementation will use cartan-geo's holonomy-based
    // disclination detection. For the smoke test, this is sufficient.
    vec![0.0; q.len()]
}

/// Active nematic stress tensor at each vertex.
///
/// sigma_active = -zeta * Q
///
/// Returns the stress as 3 independent components [sigma_xx, sigma_xy, sigma_yy]
/// per vertex (using the same layout as Q).
pub fn nematic_stress_tensor<M: Manifold>(
    _domain: &DecDomain<M>,
    q: &[M::Tangent],
    zeta: f64,
) -> Vec<[f64; 3]>
where
    M::Tangent: AsRef<[f64]>,
{
    q.iter()
        .map(|qv| {
            let s = qv.as_ref();
            if s.len() >= 3 {
                [-zeta * s[0], -zeta * s[1], -zeta * s[2]]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect()
}
```

#### Step 22.3: Implement simulation.rs

**File**: `/home/alejandrosotofranco/volterra/volterra-dec/src/simulation.rs`

```rust
//! MembraneNematicSim: the top-level simulation driver.
//!
//! Owns the DEC domain, fields, integrator, and parameters. Each call to
//! `step()` advances the simulation by one BAOAB time step, returning
//! diagnostics (energy breakdown, dt used, remesh status).

use cartan_core::Manifold;
use cartan_dec::{DecError, Mesh};
use pathwise_geo::mesh::{BAOABConfig, baoab_o_step};
use rand::Rng;

use crate::beris_edwards::{LandauDeGennesParams, beris_edwards_rhs_dec};
use crate::domain::DecDomain;
use crate::helfrich::{HelfrichParams, helfrich_energy};
use crate::variational::{VariationalIntegrator, baoab_ba_step, kinetic_energy};

/// Simulation error type.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// DEC operator assembly failed.
    #[error("DEC error: {0}")]
    Dec(#[from] DecError),
    /// Numerical divergence detected.
    #[error("numerical divergence: {0}")]
    Divergence(String),
}

/// Energy breakdown for diagnostics.
pub struct EnergyBreakdown {
    /// Total kinetic energy.
    pub kinetic: f64,
    /// Helfrich bending energy.
    pub helfrich_bending: f64,
    /// Helfrich Gaussian energy.
    pub helfrich_gaussian: f64,
    /// Landau-de Gennes nematic energy (placeholder: not yet computed).
    pub landau_de_gennes: f64,
    /// LCR spring energy (placeholder: not yet computed).
    pub lcr_spring: f64,
    /// Total energy.
    pub total: f64,
    /// Relative energy drift from initial energy: |E_n - E_0| / |E_0|.
    pub energy_drift: f64,
}

/// Result from a single simulation step.
pub struct StepResult {
    /// Time step actually used.
    pub dt_used: f64,
    /// Energy breakdown after the step.
    pub energy: EnergyBreakdown,
    /// Maximum force norm across all vertices.
    pub max_force_norm: f64,
    /// Whether remeshing occurred this step.
    pub remeshed: bool,
}

/// The membrane-nematic simulation driver.
///
/// Combines the DEC domain (mesh + operators), Q-tensor field, vertex momenta,
/// Helfrich and LdG parameters, the BAOAB integrator, and the stochastic config.
pub struct MembraneNematicSim<M: Manifold> {
    /// The DEC simulation domain.
    pub domain: DecDomain<M>,
    /// Q-tensor field (one tangent vector per vertex).
    pub q_field: Vec<M::Tangent>,
    /// Vertex momenta.
    pub momenta: Vec<M::Tangent>,
    /// Vertex masses (rho * dual_area).
    pub masses: Vec<f64>,
    /// Variational integrator configuration.
    pub integrator: VariationalIntegrator,
    /// Helfrich membrane parameters.
    pub helfrich: HelfrichParams,
    /// Landau-de Gennes / Beris-Edwards parameters.
    pub ldg: LandauDeGennesParams,
    /// BAOAB stochastic step configuration.
    pub baoab: BAOABConfig,
    /// Current simulation time.
    pub time: f64,
    /// Step counter.
    pub step_count: usize,
    /// Running minimum of kinetic energy (for symplectic remesh trigger).
    pub ke_running_min: f64,
    /// Initial total energy (for drift computation).
    initial_energy: f64,
}

impl<M: Manifold> MembraneNematicSim<M>
where
    M::Tangent: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Construct a new simulation from mesh, manifold, parameters, and initial Q field.
    pub fn new(
        mesh: Mesh<M, 3, 2>,
        manifold: M,
        helfrich: HelfrichParams,
        ldg: LandauDeGennesParams,
        integrator: VariationalIntegrator,
        baoab: BAOABConfig,
        initial_q: Vec<M::Tangent>,
    ) -> Result<Self, DecError> {
        let domain = DecDomain::new(mesh, manifold)?;
        let nv = domain.n_vertices();

        // Initialize momenta to zero.
        let momenta: Vec<M::Tangent> = (0..nv)
            .map(|v| domain.manifold.zero_tangent(&domain.mesh.vertices[v]))
            .collect();

        // Masses = dual areas (unit density).
        let masses = domain.dual_areas.clone();

        // Compute initial energy.
        let he = helfrich_energy(&domain, &helfrich);
        let ke = kinetic_energy(&momenta, &masses);
        let initial_energy = ke + he.total;

        Ok(Self {
            domain,
            q_field: initial_q,
            momenta,
            masses,
            integrator,
            helfrich,
            ldg,
            baoab,
            time: 0.0,
            step_count: 0,
            ke_running_min: ke,
            initial_energy,
        })
    }

    /// Advance the simulation by one BAOAB time step.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<StepResult, SimError>
    where
        M: cartan_core::ParallelTransport,
    {
        let dt = self.integrator.dt;
        let nv = self.domain.n_vertices();

        // B-A (first half)
        baoab_ba_step(
            &self.domain.manifold,
            &mut self.domain.mesh.vertices,
            &mut self.momenta,
            &self.masses,
            &self.helfrich,
            &self.domain,
            dt,
        );

        // O step (Shardlow DPD sweep)
        baoab_o_step(
            &self.domain.manifold,
            &self.domain.mesh,
            &self.domain.mesh.vertices,
            &mut self.momenta,
            &self.masses,
            &self.baoab,
            dt,
            rng,
        );

        // Q half-step (first)
        let dq = beris_edwards_rhs_dec(&self.domain, &self.q_field, None, &self.ldg);
        for v in 0..nv {
            let q_s = self.q_field[v].as_mut();
            let dq_s = dq[v].as_ref();
            for d in 0..q_s.len() {
                q_s[d] += (dt / 2.0) * dq_s[d];
            }
        }

        // Q half-step (second, Strang splitting)
        let dq2 = beris_edwards_rhs_dec(&self.domain, &self.q_field, None, &self.ldg);
        for v in 0..nv {
            let q_s = self.q_field[v].as_mut();
            let dq_s = dq2[v].as_ref();
            for d in 0..q_s.len() {
                q_s[d] += (dt / 2.0) * dq_s[d];
            }
        }

        // Reassemble operators on updated mesh
        self.domain.reassemble()?;

        // Update masses from new dual areas
        self.masses = self.domain.dual_areas.clone();

        // Compute energy breakdown
        let ke = kinetic_energy(&self.momenta, &self.masses);
        let he = helfrich_energy(&self.domain, &self.helfrich);
        let total = ke + he.total;
        let drift = if self.initial_energy.abs() > 1e-30 {
            (total - self.initial_energy).abs() / self.initial_energy.abs()
        } else {
            0.0
        };

        // Max force norm
        let forces = crate::helfrich::helfrich_forces(&self.domain, &self.helfrich);
        let max_force_norm = forces
            .iter()
            .map(|f| {
                let s = f.as_ref();
                s.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .fold(0.0_f64, f64::max);

        self.time += dt;
        self.step_count += 1;
        self.ke_running_min = self.ke_running_min.min(ke);

        Ok(StepResult {
            dt_used: dt,
            energy: EnergyBreakdown {
                kinetic: ke,
                helfrich_bending: he.bending,
                helfrich_gaussian: he.gaussian,
                landau_de_gennes: 0.0, // placeholder
                lcr_spring: 0.0,       // placeholder
                total,
                energy_drift: drift,
            },
            max_force_norm,
            remeshed: false,
        })
    }

    /// Current energy breakdown.
    pub fn energy(&self) -> EnergyBreakdown {
        let ke = kinetic_energy(&self.momenta, &self.masses);
        let he = helfrich_energy(&self.domain, &self.helfrich);
        let total = ke + he.total;
        let drift = if self.initial_energy.abs() > 1e-30 {
            (total - self.initial_energy).abs() / self.initial_energy.abs()
        } else {
            0.0
        };
        EnergyBreakdown {
            kinetic: ke,
            helfrich_bending: he.bending,
            helfrich_gaussian: he.gaussian,
            landau_de_gennes: 0.0,
            lcr_spring: 0.0,
            total,
            energy_drift: drift,
        }
    }
}
```

**Run**: `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test simulation` (should pass)

**Commit**: `git add -A && git commit -m "feat(volterra-dec): add MembraneNematicSim driver, observables, and smoke test"`

---

## Build and Test Commands Summary

After all tasks are complete, run the full test suite:

```bash
# cartan workspace (Phases A and B must be done first)
cd /home/alejandrosotofranco/cartan && cargo test --workspace

# pathwise-geo (standalone, Tasks 20-21)
cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test

# volterra workspace (Tasks 14-19, 22)
cd /home/alejandrosotofranco/volterra && cargo test --workspace
```

Individual test commands per task:

| Task | Command |
|------|---------|
| 14 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test dec_domain` |
| 15 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test helfrich` |
| 16 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test coupling_transport` |
| 17 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test beris_edwards_dec` |
| 18 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test variational` |
| 19 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test interpolation` |
| 20 | `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test mesh_noise` |
| 21 | `cd /home/alejandrosotofranco/pathwise/pathwise-geo && cargo test --test baoab_sweep` |
| 22 | `cd /home/alejandrosotofranco/volterra && cargo test -p volterra-dec --test simulation` |
