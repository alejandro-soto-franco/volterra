//! Mimetic Hodge stars on a tetrahedral complex.
//!
//! A diagonal Hodge star, the `cot / 2` of the two-dimensional lane, does not
//! survive in three dimensions. On a general tetrahedron no diagonal star at
//! `k = 2` reproduces the inner product of constant 2-forms, because four face
//! degrees of freedom carry six independent conditions, and at `k = 1` the
//! diagonal that does exist takes negative entries on ordinary meshes. Either
//! failure assembles an indefinite mass matrix, and a saddle point built on one
//! has no reason to be solvable. [`diagonal_star`] measures both on any given
//! cell rather than leaving the argument as a citation.
//!
//! The construction here is the consistency-plus-stabilisation star. Let `N` be
//! the matrix whose row `f`, column `I` is the integral of the constant basis
//! `k`-form `dx_I` over the `f`-th `k`-face of the cell, and let `G = |T| I` be
//! the Gram matrix of those basis forms over the cell. Consistency fixes the
//! star on the range of `N`,
//!
//! ```text
//! M_c = N (N^T N)^-1 G (N^T N)^-1 N^T,     so     N^T M_c N = G
//! ```
//!
//! and anything the transpose annihilates may be added without disturbing it.
//! The stabilisation is the projector onto that complement, scaled to the mean
//! diagonal of the consistent part so the two terms sit at one magnitude, which
//! is what keeps the star spectrally equivalent to the mass matrix of the
//! corresponding Whitney space.
//!
//! Upstream this construction lives in `cartan-mimetic`, which is unpublished
//! and sits at a workspace version this crate does not depend on. The property
//! that defines it, `N^T M N = G` together with positive definiteness, is
//! checkable here, so the port is pinned by its specification rather than by
//! agreement with another implementation.

use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

use crate::tet_mesh::TetComplex;

/// The `k`-faces of a tetrahedron, as ascending local vertex tuples in
/// lexicographic order.
///
/// At `k = 1` this is the six edges `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`, at
/// `k = 2` the four faces `(0,1,2) (0,1,3) (0,2,3) (1,2,3)`, and at `k = 3` the
/// cell itself. The ordering is the contract between this module and
/// [`TetComplex`]: a tetrahedron is stored ascending, so its local `k`-face at
/// lexicographic position `l` is the global one at `tet_edges[t][l]` for
/// `k = 1` and at `tet_faces[t][3 - l]` for `k = 2`, and both inherit the
/// ascending orientation the global degree of freedom uses.
pub fn local_k_faces(k: usize) -> Vec<Vec<usize>> {
    subsets(4, k + 1)
}

/// The basis `k`-forms on three-dimensional space, as ascending coordinate
/// tuples in lexicographic order.
pub fn form_indices(k: usize) -> Vec<Vec<usize>> {
    subsets(3, k)
}

/// Ascending `size`-subsets of `0..n`, lexicographic.
fn subsets(n: usize, size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(size);
    fn go(n: usize, size: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for v in start..n {
            cur.push(v);
            go(n, size, v + 1, cur, out);
            cur.pop();
        }
    }
    go(n, size, 0, &mut cur, &mut out);
    out
}

/// The consistency data of a tetrahedron at form degree `k`.
#[derive(Debug, Clone)]
pub struct Consistency {
    /// Row `f`, column `I`: the integral of `dx_I` over the `f`-th `k`-face.
    pub dofs: DMatrix<f64>,
    /// The Gram matrix of the basis `k`-forms over the cell, `|T| I`.
    pub gram: DMatrix<f64>,
}

/// The consistency data of the tetrahedron with the given vertex positions.
///
/// The integral of a constant `k`-form over a `k`-face is its value on the
/// face's `k`-vector divided by `k!`, and the `k`-vector's components are the
/// `k by k` minors of the face's edge matrix, so this is exact rather than
/// quadrature.
pub fn consistency(points: &[[f64; 3]; 4], k: usize) -> Consistency {
    assert!(k <= 3, "a tetrahedron has no faces above degree three");
    let local = DMatrix::from_fn(4, 3, |i, j| points[i][j] - points[0][j]);
    let faces = local_k_faces(k);
    let forms = form_indices(k);
    let scale: f64 = (1..=k).map(|i| i as f64).product::<f64>().max(1.0);

    let dofs = DMatrix::from_fn(faces.len(), forms.len(), |f, i| {
        if k == 0 {
            return 1.0;
        }
        let face = &faces[f];
        let index = &forms[i];
        let minor = DMatrix::from_fn(k, k, |r, c| {
            local[(face[c + 1], index[r])] - local[(face[0], index[r])]
        });
        minor.determinant() / scale
    });
    let vol = tet_volume(points);
    Consistency {
        dofs,
        gram: vol * DMatrix::identity(forms.len(), forms.len()),
    }
}

/// Unsigned volume of a tetrahedron.
fn tet_volume(p: &[[f64; 3]; 4]) -> f64 {
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    det.abs() / 6.0
}

/// The mimetic star of one tetrahedron at form degree `k`.
///
/// Returns `None` for a degenerate cell, where `N^T N` is singular and no inner
/// product is defined.
pub fn local_star(points: &[[f64; 3]; 4], k: usize) -> Option<DMatrix<f64>> {
    let c = consistency(points, k);
    let n = c.dofs.nrows();
    let ntn = (c.dofs.transpose() * &c.dofs).try_inverse()?;
    let consistent = &c.dofs * &ntn * &c.gram * &ntn * c.dofs.transpose();
    let projector = &c.dofs * &ntn * c.dofs.transpose();
    let gamma = consistent.trace() / c.gram.nrows() as f64;
    Some(consistent + gamma * (DMatrix::identity(n, n) - projector))
}

/// The best diagonal star on a tetrahedron, and what is wrong with it.
#[derive(Debug, Clone)]
pub struct DiagonalStar {
    /// One entry per `k`-face, in the lexicographic order of
    /// [`local_k_faces`].
    pub entries: DVector<f64>,
    /// How far `N^T diag(entries) N = G` is from satisfied, relative to the cell
    /// volume. A nonzero value means the conditions outnumber the faces and no
    /// diagonal star is consistent here.
    pub residual: f64,
}

impl DiagonalStar {
    /// Whether a consistent diagonal star exists on this cell at this degree.
    pub fn is_consistent(&self) -> bool {
        self.residual < 1e-10
    }
    /// Whether every entry is positive, which a stable star needs.
    pub fn is_positive(&self) -> bool {
        self.entries.iter().all(|&x| x > 0.0)
    }
    /// Usable as a Hodge star: consistent and positive at once. The two fail
    /// independently and for different reasons, so a caller deciding whether the
    /// cheap diagonal path is available wants this rather than either alone.
    pub fn is_usable(&self) -> bool {
        self.is_consistent() && self.is_positive()
    }
}

/// The least-squares diagonal star of a tetrahedron at degree `k`.
///
/// The conditions are the entries of the symmetric matrix `N^T D N = G`, which
/// number `m (m + 1) / 2` for `m` basis forms, against one unknown per `k`-face.
/// At `k = 2` that is six conditions against four faces, which is the counting
/// argument, and the returned residual is its size on this particular cell.
pub fn diagonal_star(points: &[[f64; 3]; 4], k: usize) -> Option<DiagonalStar> {
    let c = consistency(points, k);
    let nf = c.dofs.nrows();
    let m = c.gram.nrows();
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for i in 0..m {
        for j in i..m {
            rows.push((i, j));
        }
    }
    let a = DMatrix::from_fn(rows.len(), nf, |r, f| {
        let (i, j) = rows[r];
        c.dofs[(f, i)] * c.dofs[(f, j)]
    });
    let b = DVector::from_fn(rows.len(), |r, _| {
        let (i, j) = rows[r];
        c.gram[(i, j)]
    });
    let svd = a.clone().svd(true, true);
    let entries = svd.solve(&b, 1e-12).ok()?;
    let vol = tet_volume(points);
    let residual = (&a * &entries - &b).norm() / vol.max(f64::MIN_POSITIVE);
    Some(DiagonalStar { entries, residual })
}

/// Assemble the global mimetic star at degree `k` over a tetrahedral complex.
///
/// Every local star is placed by global index with no sign. That is a
/// consequence of the ascending convention: a tetrahedron's local `k`-face at
/// lexicographic position `l` is an ascending tuple of global indices, so it
/// agrees with the global `k`-face's own orientation and no relative sign
/// survives.
///
/// Returns `None` if any cell is degenerate.
pub fn assemble_star(mesh: &TetComplex, k: usize) -> Option<CsMat<f64>> {
    assert!((1..=3).contains(&k), "degrees one to three are what the solver uses");
    let n = match k {
        1 => mesh.n_edges(),
        2 => mesh.n_faces(),
        _ => mesh.n_tets(),
    };
    let mut trip = TriMat::new((n, n));
    for t in 0..mesh.n_tets() {
        let pts = tet_points(mesh, t);
        let local = local_star(&pts, k)?;
        let g = global_indices(mesh, t, k);
        for (i, &gi) in g.iter().enumerate() {
            for (j, &gj) in g.iter().enumerate() {
                let v = local[(i, j)];
                if v != 0.0 {
                    trip.add_triplet(gi, gj, v);
                }
            }
        }
    }
    Some(trip.to_csr())
}

/// The four vertex positions of a tetrahedron, in its stored ascending order.
pub fn tet_points(mesh: &TetComplex, t: usize) -> [[f64; 3]; 4] {
    let v = mesh.tets[t];
    [
        mesh.vertices[v[0]],
        mesh.vertices[v[1]],
        mesh.vertices[v[2]],
        mesh.vertices[v[3]],
    ]
}

/// Global indices of a tetrahedron's `k`-faces, in the lexicographic local order
/// [`local_k_faces`] uses.
pub fn global_indices(mesh: &TetComplex, t: usize, k: usize) -> Vec<usize> {
    match k {
        1 => mesh.tet_edges[t].to_vec(),
        2 => (0..4).map(|l| mesh.tet_faces[t][3 - l]).collect(),
        3 => vec![t],
        _ => panic!("degrees one to three are what the solver uses"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet_mesh::box_mesh;

    fn a_tetrahedron() -> [[f64; 3]; 4] {
        [
            [0.13, -0.21, 0.05],
            [1.07, 0.09, -0.13],
            [0.22, 0.94, 0.17],
            [-0.05, 0.31, 1.11],
        ]
    }

    #[test]
    fn the_local_star_is_consistent_at_every_degree() {
        let p = a_tetrahedron();
        for k in 1..=3 {
            let c = consistency(&p, k);
            let m = local_star(&p, k).unwrap();
            let lhs = c.dofs.transpose() * &m * &c.dofs;
            let err = (&lhs - &c.gram).norm() / c.gram.norm();
            assert!(err < 1e-12, "degree {k}: N^T M N against G is off by {err}");
        }
    }

    #[test]
    fn the_local_star_is_positive_definite_at_every_degree() {
        let p = a_tetrahedron();
        for k in 1..=3 {
            let m = local_star(&p, k).unwrap();
            let sym = (&m - &m.transpose()).norm();
            assert!(sym < 1e-14 * m.norm(), "degree {k} is not symmetric");
            let eig = m.symmetric_eigenvalues();
            let lo = eig.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(lo > 0.0, "degree {k}: smallest eigenvalue {lo}");
        }
    }

    #[test]
    fn the_top_degree_star_is_the_reciprocal_volume() {
        let p = a_tetrahedron();
        let m = local_star(&p, 3).unwrap();
        let vol = tet_volume(&p);
        assert_eq!(m.nrows(), 1);
        assert!((m[(0, 0)] - 1.0 / vol).abs() < 1e-12 / vol);
    }

    /// The counting argument at `k = 2`, as a number rather than a citation:
    /// six conditions against four faces, so the least-squares diagonal leaves a
    /// residual on a general tetrahedron.
    #[test]
    fn no_diagonal_star_is_consistent_at_degree_two() {
        let p = a_tetrahedron();
        let d = diagonal_star(&p, 2).unwrap();
        assert!(
            !d.is_consistent(),
            "residual {} was expected to be nonzero",
            d.residual
        );
        assert!(d.residual > 1e-3, "residual {} is too small to be real", d.residual);
    }

    /// The counting argument at `k = 1`: the diagonal exists, six conditions
    /// against six edges, and it takes negative entries on ordinary cells. A
    /// negative entry makes the assembled mass matrix indefinite, so this is the
    /// other half of why the stabilised star is here.
    #[test]
    fn the_diagonal_star_at_degree_one_goes_negative_on_a_box_mesh() {
        let m = box_mesh(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let mut worst = f64::INFINITY;
        let mut negatives = 0usize;
        for t in 0..m.n_tets() {
            let d = diagonal_star(&tet_points(&m, t), 1).unwrap();
            let lo = d.entries.iter().cloned().fold(f64::INFINITY, f64::min);
            if lo < 0.0 {
                negatives += 1;
            }
            worst = worst.min(lo);
        }
        assert!(
            negatives > 0,
            "no cell of {} had a negative diagonal entry; least was {worst}",
            m.n_tets()
        );
    }

    fn dense_from(m: &CsMat<f64>) -> DMatrix<f64> {
        let mut d = DMatrix::zeros(m.rows(), m.cols());
        for (v, (r, c)) in m.iter() {
            d[(r, c)] += v;
        }
        d
    }

    #[test]
    fn every_assembled_star_is_symmetric_positive_definite() {
        let mesh = box_mesh(2, 2, 1, 1.0, 0.8, 0.5).unwrap();
        for k in 1..=3 {
            let s = assemble_star(&mesh, k).unwrap();
            let d = dense_from(&s);
            let asym = (&d - &d.transpose()).norm();
            assert!(asym < 1e-13 * d.norm(), "degree {k} assembled asymmetric");
            let lo = d
                .symmetric_eigenvalues()
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            assert!(lo > 0.0, "degree {k}: smallest assembled eigenvalue {lo}");
        }
    }

    fn quad(m: &CsMat<f64>, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for (v, (r, c)) in m.iter() {
            s += x[r] * v * x[c];
        }
        s
    }

    /// The assembled star reproduces the L2 norm of a constant field exactly.
    ///
    /// This is the test with power over [`global_indices`]. The local star is
    /// consistent for constant forms cell by cell, so the assembled quadratic
    /// form on the degrees of freedom of a globally constant field must be the
    /// field's squared norm times the volume. Permute the local-to-global map at
    /// either degree and the cells no longer agree about which face is which,
    /// so the sum misses by a finite amount rather than by a rounding error.
    #[test]
    fn the_assembled_star_reproduces_a_constant_field_at_every_degree() {
        let mesh = box_mesh(3, 2, 2, 1.0, 0.8, 0.5).unwrap();
        let vol = mesh.volume();

        // Degree one: the 1-form `c . dx`, whose edge integral is `c . (b - a)`.
        let c = [0.41, -0.62, 0.23];
        let w: Vec<f64> = mesh
            .edges
            .iter()
            .map(|&[a, b]| {
                let (pa, pb) = (mesh.vertices[a], mesh.vertices[b]);
                (0..3).map(|i| c[i] * (pb[i] - pa[i])).sum()
            })
            .collect();
        let m1 = assemble_star(&mesh, 1).unwrap();
        let expect1 = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) * vol;
        let got1 = quad(&m1, &w);
        assert!(
            (got1 - expect1).abs() < 1e-11 * expect1,
            "degree one: {got1} against {expect1}"
        );

        // Degree two: the flux 2-form of a constant vector field.
        let v = [0.31, 0.77, -0.52];
        let u: Vec<f64> = (0..mesh.n_faces())
            .map(|f| {
                let a = mesh.face_area_vector(f);
                (0..3).map(|i| v[i] * a[i]).sum()
            })
            .collect();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let expect2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * vol;
        let got2 = quad(&m2, &u);
        assert!(
            (got2 - expect2).abs() < 1e-11 * expect2,
            "degree two: {got2} against {expect2}"
        );

        // Degree three: the constant scalar `s`, whose cell integral is
        // `s |T|`.
        let s = 1.37;
        let p: Vec<f64> = (0..mesh.n_tets()).map(|t| s * mesh.tet_volume(t)).collect();
        let m3 = assemble_star(&mesh, 3).unwrap();
        let expect3 = s * s * vol;
        let got3 = quad(&m3, &p);
        assert!(
            (got3 - expect3).abs() < 1e-11 * expect3,
            "degree three: {got3} against {expect3}"
        );
    }
}
