//! A tetrahedral complex with its full chain of exterior derivatives.
//!
//! `cartan-dec` builds two incidence matrices from a `Mesh<M, K, B>`, which at
//! `K = 4` are the face-to-vertex incidence and the tet-to-face incidence.
//! Neither is `d1`, and edges never enter that type at all, so the complex a
//! three-dimensional Hodge Laplacian needs is built here.
//!
//! # Canonical ascending indexing
//!
//! Every tetrahedron is stored as an ascending quadruple of global vertex
//! indices, every face as an ascending triple, every edge as an ascending pair.
//! Omitting the `k`-th entry of an ascending list leaves an ascending list, so
//! the boundary coefficient is `(-1)^k` with no permutation parity anywhere,
//! `d1 d0 = 0` and `d2 d1 = 0` are combinatorial identities, and a local mimetic
//! star built on a tetrahedron's own vertex order needs no permutation to reach
//! its global rows.
//!
//! Geometry enters once. A tetrahedron ascending by index may be negatively
//! oriented, so `d2[t, f_k] = (-1)^k * orient(t)`. Scaling a row of `d2` by a
//! sign leaves `d2 d1 = 0` untouched and makes `(d2 u)_t` the net outflux with
//! the sign the divergence theorem asks for.

use std::collections::HashMap;

use sprs::{CsMat, TriMat};

/// A simplicial complex of tetrahedra, with every sub-simplex enumerated.
#[derive(Debug, Clone)]
pub struct TetComplex {
    /// Vertex positions.
    pub vertices: Vec<[f64; 3]>,
    /// Tetrahedra, each an ascending quadruple of vertex indices.
    pub tets: Vec<[usize; 4]>,
    /// Faces, each an ascending triple, deduplicated.
    pub faces: Vec<[usize; 3]>,
    /// Edges, each an ascending pair, deduplicated.
    pub edges: Vec<[usize; 2]>,
    /// `tet_faces[t][k]` is the face of tetrahedron `t` that omits its `k`-th
    /// vertex.
    pub tet_faces: Vec<[usize; 4]>,
    /// `face_edges[f][k]` is the edge of face `f` that omits its `k`-th vertex.
    pub face_edges: Vec<[usize; 3]>,
    /// `tet_edges[t][l]` is the global edge of tetrahedron `t` at lexicographic
    /// local position `l`, in the order
    /// `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`.
    pub tet_edges: Vec<[usize; 6]>,
    /// The sign of the determinant of each tetrahedron's edge matrix, taken in
    /// ascending vertex order.
    pub tet_orient: Vec<f64>,
    /// How many tetrahedra each face belongs to: one at the boundary, two
    /// inside.
    pub face_valence: Vec<u8>,
}

impl TetComplex {
    /// Build the complex from vertex positions and tetrahedra.
    ///
    /// Each tetrahedron is sorted ascending on entry, so a caller may supply any
    /// vertex order. A degenerate tetrahedron, one whose four vertices are
    /// coplanar or repeated, is rejected rather than assembled: its orientation
    /// sign is undefined and every mass matrix built on it is singular.
    pub fn new(vertices: Vec<[f64; 3]>, tets: Vec<[usize; 4]>) -> Result<Self, TetMeshError> {
        let nv = vertices.len();
        let mut sorted: Vec<[usize; 4]> = Vec::with_capacity(tets.len());
        for t in &tets {
            let mut s = *t;
            s.sort_unstable();
            for w in s.windows(2) {
                if w[0] == w[1] {
                    return Err(TetMeshError::RepeatedVertex(*t));
                }
            }
            if s[3] >= nv {
                return Err(TetMeshError::VertexOutOfRange { index: s[3], nv });
            }
            sorted.push(s);
        }

        let mut face_map: HashMap<[usize; 3], usize> = HashMap::new();
        let mut faces: Vec<[usize; 3]> = Vec::new();
        let mut tet_faces: Vec<[usize; 4]> = Vec::with_capacity(sorted.len());
        let mut face_valence: Vec<u8> = Vec::new();

        for t in &sorted {
            let mut local = [0usize; 4];
            for omit in 0..4 {
                let mut f = [0usize; 3];
                let mut i = 0;
                for (pos, &v) in t.iter().enumerate() {
                    if pos != omit {
                        f[i] = v;
                        i += 1;
                    }
                }
                let id = *face_map.entry(f).or_insert_with(|| {
                    faces.push(f);
                    face_valence.push(0);
                    faces.len() - 1
                });
                face_valence[id] += 1;
                if face_valence[id] > 2 {
                    return Err(TetMeshError::NonManifoldFace(f));
                }
                local[omit] = id;
            }
            tet_faces.push(local);
        }

        let mut edge_map: HashMap<[usize; 2], usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut face_edges: Vec<[usize; 3]> = Vec::with_capacity(faces.len());
        for f in &faces {
            let mut local = [0usize; 3];
            for omit in 0..3 {
                let mut e = [0usize; 2];
                let mut i = 0;
                for (pos, &v) in f.iter().enumerate() {
                    if pos != omit {
                        e[i] = v;
                        i += 1;
                    }
                }
                let id = *edge_map.entry(e).or_insert_with(|| {
                    edges.push(e);
                    edges.len() - 1
                });
                local[omit] = id;
            }
            face_edges.push(local);
        }

        let mut tet_edges: Vec<[usize; 6]> = Vec::with_capacity(sorted.len());
        for t in &sorted {
            let mut local = [0usize; 6];
            let mut l = 0;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    local[l] = edge_map[&[t[i], t[j]]];
                    l += 1;
                }
            }
            tet_edges.push(local);
        }

        let mut tet_orient = Vec::with_capacity(sorted.len());
        for t in &sorted {
            let d = signed_volume_six(&vertices, t);
            if d.abs() <= 1e-14 * scale_cube(&vertices, t) {
                return Err(TetMeshError::DegenerateTet(*t));
            }
            tet_orient.push(d.signum());
        }

        Ok(Self {
            vertices,
            tets: sorted,
            faces,
            edges,
            tet_faces,
            face_edges,
            tet_edges,
            tet_orient,
            face_valence,
        })
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }
    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
    /// Number of faces.
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
    /// Number of tetrahedra.
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }

    /// Whether a face lies on the boundary of the complex.
    pub fn is_boundary_face(&self, f: usize) -> bool {
        self.face_valence[f] == 1
    }

    /// Whether an edge lies on the boundary, meaning it belongs to at least one
    /// boundary face.
    pub fn boundary_edges(&self) -> Vec<bool> {
        let mut flag = vec![false; self.n_edges()];
        for f in 0..self.n_faces() {
            if self.is_boundary_face(f) {
                for &e in &self.face_edges[f] {
                    flag[e] = true;
                }
            }
        }
        flag
    }

    /// `d0`, of shape (edges x vertices). For edge `[a, b]` with `a < b`, the
    /// row has `-1` at `a` and `+1` at `b`.
    pub fn d0(&self) -> CsMat<f64> {
        let mut t = TriMat::new((self.n_edges(), self.n_vertices()));
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            t.add_triplet(e, a, -1.0);
            t.add_triplet(e, b, 1.0);
        }
        t.to_csr()
    }

    /// `d1`, of shape (faces x edges), with coefficient `(-1)^k` on the edge
    /// omitting the face's `k`-th vertex.
    pub fn d1(&self) -> CsMat<f64> {
        let mut t = TriMat::new((self.n_faces(), self.n_edges()));
        for (f, local) in self.face_edges.iter().enumerate() {
            for (k, &e) in local.iter().enumerate() {
                t.add_triplet(f, e, if k % 2 == 0 { 1.0 } else { -1.0 });
            }
        }
        t.to_csr()
    }

    /// `d2`, of shape (tets x faces), with coefficient `(-1)^k orient(t)` on the
    /// face omitting the tetrahedron's `k`-th vertex.
    ///
    /// The orientation factor is what makes `(d2 u)_t` the net outflux of the
    /// face fluxes, so a positive entry means the canonically oriented face
    /// normal points out of the cell.
    pub fn d2(&self) -> CsMat<f64> {
        let mut t = TriMat::new((self.n_tets(), self.n_faces()));
        for (c, local) in self.tet_faces.iter().enumerate() {
            let o = self.tet_orient[c];
            for (k, &f) in local.iter().enumerate() {
                t.add_triplet(c, f, if k % 2 == 0 { o } else { -o });
            }
        }
        t.to_csr()
    }

    /// Unsigned volume of a tetrahedron.
    pub fn tet_volume(&self, t: usize) -> f64 {
        signed_volume_six(&self.vertices, &self.tets[t]).abs() / 6.0
    }

    /// Twice the area vector of a face, in its canonical orientation:
    /// `(b - a) x (c - a)` for the ascending triple `[a, b, c]`.
    pub fn face_area_vector(&self, f: usize) -> [f64; 3] {
        let [a, b, c] = self.faces[f];
        let (pa, pb, pc) = (self.vertices[a], self.vertices[b], self.vertices[c]);
        let u = sub(pb, pa);
        let v = sub(pc, pa);
        let n = cross(u, v);
        [0.5 * n[0], 0.5 * n[1], 0.5 * n[2]]
    }

    /// Area of a face.
    pub fn face_area(&self, f: usize) -> f64 {
        norm(self.face_area_vector(f))
    }

    /// Unit normal of a face in its canonical orientation.
    pub fn face_normal(&self, f: usize) -> [f64; 3] {
        let a = self.face_area_vector(f);
        let n = norm(a);
        [a[0] / n, a[1] / n, a[2] / n]
    }

    /// Centroid of a face.
    pub fn face_centroid(&self, f: usize) -> [f64; 3] {
        let [a, b, c] = self.faces[f];
        let (pa, pb, pc) = (self.vertices[a], self.vertices[b], self.vertices[c]);
        [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ]
    }

    /// Centroid of a tetrahedron.
    pub fn tet_centroid(&self, t: usize) -> [f64; 3] {
        let mut c = [0.0; 3];
        for &v in &self.tets[t] {
            for i in 0..3 {
                c[i] += self.vertices[v][i] / 4.0;
            }
        }
        c
    }

    /// The tetrahedra incident on each vertex.
    pub fn vertex_tets(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n_vertices()];
        for (t, tet) in self.tets.iter().enumerate() {
            for &v in tet {
                out[v].push(t);
            }
        }
        out
    }

    /// Total volume of the complex.
    pub fn volume(&self) -> f64 {
        (0..self.n_tets()).map(|t| self.tet_volume(t)).sum()
    }
}

/// What can go wrong when a complex is built.
#[derive(Debug, Clone, PartialEq)]
pub enum TetMeshError {
    /// A tetrahedron names the same vertex twice.
    RepeatedVertex([usize; 4]),
    /// A tetrahedron names a vertex the position list does not have.
    VertexOutOfRange { index: usize, nv: usize },
    /// A tetrahedron's four vertices are coplanar.
    DegenerateTet([usize; 4]),
    /// A face belongs to three or more tetrahedra.
    NonManifoldFace([usize; 3]),
}

impl std::fmt::Display for TetMeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedVertex(t) => write!(f, "tetrahedron {t:?} repeats a vertex"),
            Self::VertexOutOfRange { index, nv } => {
                write!(
                    f,
                    "vertex index {index} is past the {nv} positions supplied"
                )
            }
            Self::DegenerateTet(t) => write!(f, "tetrahedron {t:?} is coplanar"),
            Self::NonManifoldFace(t) => write!(f, "face {t:?} belongs to three or more cells"),
        }
    }
}

impl std::error::Error for TetMeshError {}

/// Six times the signed volume of a tetrahedron.
fn signed_volume_six(vertices: &[[f64; 3]], t: &[usize; 4]) -> f64 {
    let p0 = vertices[t[0]];
    let a = sub(vertices[t[1]], p0);
    let b = sub(vertices[t[2]], p0);
    let c = sub(vertices[t[3]], p0);
    dot(a, cross(b, c))
}

/// The cube of the longest edge out of the first vertex, as a scale against
/// which a determinant counts as zero.
fn scale_cube(vertices: &[[f64; 3]], t: &[usize; 4]) -> f64 {
    let p0 = vertices[t[0]];
    let m = (1..4)
        .map(|i| norm(sub(vertices[t[i]], p0)))
        .fold(0.0_f64, f64::max);
    (m * m * m).max(f64::MIN_POSITIVE)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Extrude a planar triangulation to a prism layer stack and split each prism
/// into three tetrahedra.
///
/// With the base triangle's vertices ordered by global index as `v0 < v1 < v2`,
/// the split is
///
/// ```text
/// (v0_b, v1_b, v2_b, v2_t)   (v0_b, v1_b, v2_t, v1_t)   (v0_b, v1_t, v2_t, v0_t)
/// ```
///
/// Each of the three quadrilateral side faces then takes its diagonal from the
/// lower-indexed vertex at the bottom to the higher-indexed vertex at the top.
/// That rule reads only the two global indices, so the two prisms sharing a
/// quadrilateral choose the same diagonal and the result is conforming. A split
/// chosen per prism instead leaves a pair of triangles crossing each other on
/// the shared quadrilateral, which builds a face of valence one where the mesh
/// is interior and no assembled operator afterwards is meaningful.
///
/// Vertices are laid out layer by layer, so the copy of planar vertex `i` in
/// layer `j` is at `j * n + i`.
pub fn prism_extrude(
    planar: &[[f64; 2]],
    triangles: &[[usize; 3]],
    depth: f64,
    layers: usize,
) -> Result<TetComplex, TetMeshError> {
    assert!(depth > 0.0, "extrusion depth must be positive");
    assert!(layers >= 1, "extrusion needs at least one layer");
    let n = planar.len();
    let dz = depth / layers as f64;
    let mut vertices = Vec::with_capacity(n * (layers + 1));
    for j in 0..=layers {
        let z = j as f64 * dz;
        for p in planar {
            vertices.push([p[0], p[1], z]);
        }
    }

    let mut tets = Vec::with_capacity(3 * triangles.len() * layers);
    for tri in triangles {
        let mut v = *tri;
        v.sort_unstable();
        let [a, b, c] = v;
        for j in 0..layers {
            let lo = j * n;
            let hi = (j + 1) * n;
            let (ab, bb, cb) = (lo + a, lo + b, lo + c);
            let (at, bt, ct) = (hi + a, hi + b, hi + c);
            tets.push([ab, bb, cb, ct]);
            tets.push([ab, bb, ct, bt]);
            tets.push([ab, bt, ct, at]);
        }
    }
    TetComplex::new(vertices, tets)
}

/// A right-angled triangulation of the rectangle `[0, lx] x [0, ly]` on an
/// `nx by ny` grid, each cell split on its lower-left to upper-right diagonal.
///
/// Returned as planar vertices and triangles, ready for [`prism_extrude`].
pub fn rectangle_triangulation(
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let mut v = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            v.push([lx * i as f64 / nx as f64, ly * j as f64 / ny as f64]);
        }
    }
    let idx = |i: usize, j: usize| j * (nx + 1) + i;
    let mut t = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            t.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            t.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    (v, t)
}

/// A tetrahedral mesh of the box `[0, lx] x [0, ly] x [0, lz]`.
pub fn box_mesh(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
) -> Result<TetComplex, TetMeshError> {
    let (v, t) = rectangle_triangulation(nx, ny, lx, ly);
    prism_extrude(&v, &t, lz, nz)
}

impl TetComplex {
    /// Barycentric coordinates of a point with respect to a cell, in the cell's
    /// stored vertex order.
    ///
    /// All four are non-negative exactly when the point lies in the closed cell,
    /// so this is both the containment test and the linear interpolation weight.
    pub fn barycentric(&self, t: usize, x: [f64; 3]) -> [f64; 4] {
        let v = self.tets[t];
        let p0 = self.vertices[v[0]];
        let a = sub(self.vertices[v[1]], p0);
        let b = sub(self.vertices[v[2]], p0);
        let c = sub(self.vertices[v[3]], p0);
        let r = sub(x, p0);
        let det = dot(a, cross(b, c));
        if det == 0.0 {
            return [0.0; 4];
        }
        let l1 = dot(r, cross(b, c)) / det;
        let l2 = dot(a, cross(r, c)) / det;
        let l3 = dot(a, cross(b, r)) / det;
        [1.0 - l1 - l2 - l3, l1, l2, l3]
    }
}

/// A box meshed by [`box_mesh`], kept alongside its own dimensions so a point
/// can be located in constant time.
///
/// The mesh is structured, so the cell a point falls in is arithmetic rather
/// than a search: `box_mesh` lays the rectangle's triangles out as
/// `2 (j nx + i) + s` with `s` naming the lower or upper triangle, and
/// `prism_extrude` lays the tetrahedra out as `tri (3 layers) + layer * 3 + k`.
/// A tree over a hundred thousand cells buys nothing against that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredBox {
    /// Cell divisions along each axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Extent along each axis, with the box occupying `[0, lx] x [0, ly] x [0, lz]`.
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
}

impl StructuredBox {
    /// The divisions and extents of a box.
    pub fn new(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            lx,
            ly,
            lz,
        }
    }

    /// Build the complex this descriptor names.
    pub fn build(&self) -> Result<TetComplex, TetMeshError> {
        box_mesh(self.nx, self.ny, self.nz, self.lx, self.ly, self.lz)
    }

    /// The three tetrahedra of the prism over triangle `tri` in layer `layer`.
    fn prism_tets(&self, tri: usize, layer: usize) -> [usize; 3] {
        let base = tri * (3 * self.nz) + layer * 3;
        [base, base + 1, base + 2]
    }

    /// The cell a point falls in, or `None` when it lies outside the box by more
    /// than a rounding error.
    ///
    /// The arithmetic names one cell of the grid; the six tetrahedra over it are
    /// tested by barycentric coordinate, and a point on a shared face lands in
    /// whichever is tried first. A point that misses them all, which happens only
    /// when it sits within a rounding error of a cell wall, falls back to the
    /// twenty-six neighbouring cells before the search gives up.
    pub fn locate(&self, mesh: &TetComplex, x: [f64; 3]) -> Option<usize> {
        let tol = -1e-9;
        let fi = x[0] / self.lx * self.nx as f64;
        let fj = x[1] / self.ly * self.ny as f64;
        let fl = x[2] / self.lz * self.nz as f64;
        let clamp = |f: f64, n: usize| (f.floor().max(0.0) as usize).min(n - 1);
        let (ci, cj, cl) = (clamp(fi, self.nx), clamp(fj, self.ny), clamp(fl, self.nz));

        for radius in 0..2 {
            for di in -(radius as i64)..=(radius as i64) {
                for dj in -(radius as i64)..=(radius as i64) {
                    for dl in -(radius as i64)..=(radius as i64) {
                        if radius == 1 && di == 0 && dj == 0 && dl == 0 {
                            continue;
                        }
                        let i = ci as i64 + di;
                        let j = cj as i64 + dj;
                        let l = cl as i64 + dl;
                        if i < 0 || j < 0 || l < 0 {
                            continue;
                        }
                        let (i, j, l) = (i as usize, j as usize, l as usize);
                        if i >= self.nx || j >= self.ny || l >= self.nz {
                            continue;
                        }
                        let tri = 2 * (j * self.nx + i);
                        for t in self
                            .prism_tets(tri, l)
                            .into_iter()
                            .chain(self.prism_tets(tri + 1, l))
                        {
                            if mesh.barycentric(t, x).iter().all(|&b| b >= tol) {
                                return Some(t);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Extrude a `cartan-dec` planar triangulation into a chamber.
///
/// This is the path a chip footprint takes: `confined_mesh` triangulates an
/// arbitrary region bounded by a `PlaneCurve`, and the extrusion gives it a
/// uniform depth. Single-layer soft lithography moulds exactly that shape, a
/// planar footprint at one depth, so the prism restriction is the fabrication
/// process rather than a limitation of the mesher. A chamber whose depth varies
/// needs a constrained tetrahedraliser, and it also needs a second lithography
/// layer.
pub fn prism_extrude_flat(
    planar: &cartan_dec::mesh::FlatMesh,
    depth: f64,
    layers: usize,
) -> Result<TetComplex, TetMeshError> {
    let v: Vec<[f64; 2]> = planar.vertices.iter().map(|p| [p[0], p[1]]).collect();
    let t: Vec<[usize; 3]> = planar.simplices.clone();
    prism_extrude(&v, &t, depth, layers)
}

/// A structured triangulation of the annulus `r_inner <= r <= r_outer`, on an
/// `n_theta by n_r` polar grid.
///
/// Extruded, this is a chamber with a pillar through it, whose first Betti
/// number is one. That is the topology the bounded Stokes solver's pressure
/// pinning does not by itself account for, so it is what the harmonic
/// measurement runs on.
///
/// The grid closes in `theta`, so the ring of quadrilaterals is conforming and
/// the inner and outer circles are the only boundaries in the plane.
pub fn annulus_triangulation(
    n_theta: usize,
    n_r: usize,
    r_inner: f64,
    r_outer: f64,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    assert!(n_theta >= 3, "an annulus needs at least three sectors");
    assert!(n_r >= 1, "an annulus needs at least one radial band");
    assert!(
        0.0 < r_inner && r_inner < r_outer,
        "radii must be ordered and positive"
    );
    let mut v = Vec::with_capacity(n_theta * (n_r + 1));
    for j in 0..=n_r {
        let r = r_inner + (r_outer - r_inner) * j as f64 / n_r as f64;
        for i in 0..n_theta {
            let a = std::f64::consts::TAU * i as f64 / n_theta as f64;
            v.push([r * a.cos(), r * a.sin()]);
        }
    }
    let idx = |i: usize, j: usize| j * n_theta + (i % n_theta);
    let mut t = Vec::with_capacity(2 * n_theta * n_r);
    for j in 0..n_r {
        for i in 0..n_theta {
            t.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            t.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    (v, t)
}

/// A tetrahedral mesh of an annular chamber of depth `lz`: a slab with a pillar
/// through it.
pub fn pillar_mesh(
    n_theta: usize,
    n_r: usize,
    n_z: usize,
    r_inner: f64,
    r_outer: f64,
    lz: f64,
) -> Result<TetComplex, TetMeshError> {
    let (v, t) = annulus_triangulation(n_theta, n_r, r_inner, r_outer);
    prism_extrude(&v, &t, lz, n_z)
}

/// A degree-five, seven-point quadrature rule on the reference triangle, in
/// barycentric coordinates with weights summing to one.
///
/// The manufactured source of the validation ladder is a degree-eleven
/// polynomial, so the quadrature error on a cell of diameter `h` is `O(h^6)`
/// against a degree of freedom of size `O(h^2)`. That is six orders below the
/// first-order convergence the method itself has, so the rule never enters the
/// measured rate.
pub fn triangle_rule() -> [(f64, f64, f64, f64); 7] {
    let r = 15.0_f64.sqrt();
    let a1 = (6.0 - r) / 21.0;
    let b1 = (9.0 + 2.0 * r) / 21.0;
    let a2 = (6.0 + r) / 21.0;
    let b2 = (9.0 - 2.0 * r) / 21.0;
    let w1 = (155.0 - r) / 1200.0;
    let w2 = (155.0 + r) / 1200.0;
    [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 9.0 / 40.0),
        (b1, a1, a1, w1),
        (a1, b1, a1, w1),
        (a1, a1, b1, w1),
        (b2, a2, a2, w2),
        (a2, b2, a2, w2),
        (a2, a2, b2, w2),
    ]
}

impl TetComplex {
    /// The flux of a vector field through a face, in the face's canonical
    /// orientation, by the degree-five rule.
    ///
    /// This is the degree of freedom the three-dimensional Stokes solver takes
    /// for both its velocity and its body force.
    pub fn face_flux_of<F>(&self, f: usize, field: F) -> f64
    where
        F: Fn([f64; 3]) -> [f64; 3],
    {
        let [ia, ib, ic] = self.faces[f];
        let (pa, pb, pc) = (self.vertices[ia], self.vertices[ib], self.vertices[ic]);
        let a = self.face_area_vector(f);
        let mut acc = 0.0;
        for (l0, l1, l2, w) in triangle_rule() {
            let x = [
                l0 * pa[0] + l1 * pb[0] + l2 * pc[0],
                l0 * pa[1] + l1 * pb[1] + l2 * pc[1],
                l0 * pa[2] + l1 * pb[2] + l2 * pc[2],
            ];
            let v = field(x);
            acc += w * (v[0] * a[0] + v[1] * a[1] + v[2] * a[2]);
        }
        acc
    }

    /// The flux degrees of freedom of a vector field on every face.
    pub fn flux_dofs<F>(&self, field: F) -> Vec<f64>
    where
        F: Fn([f64; 3]) -> [f64; 3] + Copy,
    {
        (0..self.n_faces())
            .map(|f| self.face_flux_of(f, field))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spmul(a: &CsMat<f64>, b: &CsMat<f64>) -> CsMat<f64> {
        let a = a.to_csr();
        let b = b.to_csr();
        &a * &b
    }

    /// Dense mat-vec, so a test reads a plain slice rather than a sparse vector.
    fn matvec(m: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; m.rows()];
        for (v, (r, c)) in m.iter() {
            y[r] += v * x[c];
        }
        y
    }

    fn max_abs(m: &CsMat<f64>) -> f64 {
        m.iter().map(|(v, _)| v.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn the_chain_composes_to_zero_on_a_box() {
        let m = box_mesh(3, 2, 2, 1.0, 0.7, 0.4).unwrap();
        let (d0, d1, d2) = (m.d0(), m.d1(), m.d2());
        assert_eq!(
            max_abs(&spmul(&d1, &d0)),
            0.0,
            "d1 d0 must vanish entry by entry"
        );
        assert_eq!(
            max_abs(&spmul(&d2, &d1)),
            0.0,
            "d2 d1 must vanish entry by entry"
        );
    }

    #[test]
    fn the_euler_characteristic_of_a_filled_box_is_one() {
        let m = box_mesh(3, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let chi =
            m.n_vertices() as i64 - m.n_edges() as i64 + m.n_faces() as i64 - m.n_tets() as i64;
        assert_eq!(chi, 1, "a ball has Euler characteristic one");
    }

    #[test]
    fn the_split_is_conforming_so_the_boundary_area_is_the_box_surface() {
        // A per-prism split, chosen without reference to the global vertex
        // order, leaves the two prisms sharing a quadrilateral disagreeing on
        // its diagonal. Every such quadrilateral then contributes four
        // valence-one faces instead of none, and the boundary area overshoots.
        let (lx, ly, lz) = (1.0, 0.7, 0.4);
        let m = box_mesh(3, 2, 2, lx, ly, lz).unwrap();
        let area: f64 = (0..m.n_faces())
            .filter(|&f| m.is_boundary_face(f))
            .map(|f| m.face_area(f))
            .sum();
        let expect = 2.0 * (lx * ly + ly * lz + lz * lx);
        assert!(
            (area - expect).abs() < 1e-12 * expect,
            "boundary area {area} against surface {expect}"
        );
    }

    #[test]
    fn the_volume_of_the_box_mesh_is_the_box_volume() {
        let m = box_mesh(4, 3, 2, 1.3, 0.9, 0.5).unwrap();
        let expect = 1.3 * 0.9 * 0.5;
        assert!((m.volume() - expect).abs() < 1e-12 * expect);
    }

    /// Flux degrees of freedom of a constant field have zero divergence in
    /// every cell. This is the test with power over the `orient(t)` factor in
    /// `d2`: drop it and the negatively oriented tetrahedra, which are half of
    /// the prism split, report a divergence of twice the wrong flux.
    #[test]
    fn a_constant_field_is_divergence_free_cell_by_cell() {
        let m = box_mesh(3, 2, 2, 1.0, 0.7, 0.4).unwrap();
        let v = [0.31, -0.77, 0.52];
        let flux: Vec<f64> = (0..m.n_faces())
            .map(|f| {
                let a = m.face_area_vector(f);
                v[0] * a[0] + v[1] * a[1] + v[2] * a[2]
            })
            .collect();
        let d2 = m.d2();
        let div = matvec(&d2, &flux);
        let worst = div.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        let scale = flux.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        assert!(
            worst < 1e-13 * scale,
            "worst cell divergence {worst} against flux scale {scale}"
        );
    }

    /// The divergence theorem on every cell, for the linear field `v = x`.
    /// The centroid rule is exact for the flux of a linear field through a
    /// planar triangle, so the identity is exact up to rounding and it pins the
    /// sign of every entry of `d2` at once.
    #[test]
    fn the_divergence_theorem_is_exact_cell_by_cell_for_a_linear_field() {
        let m = box_mesh(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let flux: Vec<f64> = (0..m.n_faces())
            .map(|f| {
                let c = m.face_centroid(f);
                let a = m.face_area_vector(f);
                c[0] * a[0] + c[1] * a[1] + c[2] * a[2]
            })
            .collect();
        let d2 = m.d2();
        let div = matvec(&d2, &flux);
        for t in 0..m.n_tets() {
            let expect = 3.0 * m.tet_volume(t);
            assert!(
                (div[t] - expect).abs() < 1e-12,
                "cell {t}: divergence {} against 3 V = {expect}",
                div[t]
            );
        }
    }

    #[test]
    fn a_coplanar_tetrahedron_is_rejected() {
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let err = TetComplex::new(v, vec![[0, 1, 2, 3]]).unwrap_err();
        assert_eq!(err, TetMeshError::DegenerateTet([0, 1, 2, 3]));
    }

    #[test]
    fn a_repeated_vertex_is_rejected() {
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let err = TetComplex::new(v, vec![[0, 1, 1, 3]]).unwrap_err();
        assert_eq!(err, TetMeshError::RepeatedVertex([0, 1, 1, 3]));
    }

    #[test]
    fn boundary_edges_are_exactly_those_on_a_boundary_face() {
        let m = box_mesh(2, 2, 1, 1.0, 1.0, 1.0).unwrap();
        let flag = m.boundary_edges();
        for f in 0..m.n_faces() {
            if m.is_boundary_face(f) {
                for &e in &m.face_edges[f] {
                    assert!(flag[e]);
                }
            }
        }
        assert!(flag.iter().any(|&b| !b), "an interior edge must exist");
    }
}
