//! 3D Q-tensor field on a regular Cartesian grid with periodic BCs.
//!
//! ## Storage
//! 5 independent components [q11, q12, q13, q22, q23] per vertex.
//! q33 = -(q11 + q22) recovered on demand.
//! Linear index: k = (i * ny + j) * nz + l.
//!
//! ## Embedded matrix
//! Q_3D = [[q11, q12, q13],
//!         [q12, q22, q23],
//!         [q13, q23, -(q11+q22)]]

use nalgebra::SMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// 3D Q-tensor field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QField3D {
    /// Components [q11, q12, q13, q22, q23] at each vertex.
    pub q: Vec<[f64; 5]>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
}

impl QField3D {
    /// Create a zero Q-tensor field.
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { q: vec![[0.0; 5]; nx * ny * nz], nx, ny, nz, dx }
    }

    /// Create a uniform Q-tensor field with every vertex set to `q`.
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, q: [f64; 5]) -> Self {
        Self { q: vec![q; nx * ny * nz], nx, ny, nz, dx }
    }

    /// Create a small-amplitude random perturbation.
    ///
    /// Each component is drawn uniformly from `[-amplitude, amplitude]`.
    pub fn random_perturbation(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        amplitude: f64,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let q = (0..nx * ny * nz)
            .map(|_| {
                let mut v = [0.0f64; 5];
                for c in &mut v {
                    *c = rng.random_range(-amplitude..amplitude);
                }
                v
            })
            .collect();
        Self { q, nx, ny, nz, dx }
    }

    /// Number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Returns `true` if the field has no vertices.
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Linear index for vertex `(i, j, l)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        ((i % self.nx) * self.ny + (j % self.ny)) * self.nz + (l % self.nz)
    }

    /// Inverse of `idx`: returns (i, j, l) from linear index k.
    #[inline]
    pub fn ijk(&self, k: usize) -> (usize, usize, usize) {
        let l = k % self.nz;
        let ij = k / self.nz;
        let j = ij % self.ny;
        let i = ij / self.ny;
        (i, j, l)
    }

    /// Embed the 5-component Q at vertex k into a 3x3 nalgebra SMatrix.
    ///
    /// Returns the full 3x3 symmetric traceless matrix with q33 = -(q11+q22).
    pub fn embed_matrix3(&self, k: usize) -> SMatrix<f64, 3, 3> {
        let [q11, q12, q13, q22, q23] = self.q[k];
        let q33 = -(q11 + q22);
        SMatrix::<f64, 3, 3>::new(q11, q12, q13, q12, q22, q23, q13, q23, q33)
    }

    /// Component-wise 3D Laplacian with periodic BCs (6-point stencil).
    ///
    /// Uses the isotropic 6-point stencil:
    /// ∇²f_{i,j,l} ≈ (f_{i+1,j,l} + f_{i-1,j,l} + f_{i,j+1,l}
    ///                + f_{i,j-1,l} + f_{i,j,l+1} + f_{i,j,l-1} - 6 f_{i,j,l}) / dx²
    pub fn laplacian(&self) -> QField3D {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let mut out = QField3D::zeros(nx, ny, nz, self.dx);

        for i in 0..nx {
            for j in 0..ny {
                for l in 0..nz {
                    let k = self.idx(i, j, l);
                    let ip = self.idx((i + 1) % nx, j, l);
                    let im = self.idx((i + nx - 1) % nx, j, l);
                    let jp = self.idx(i, (j + 1) % ny, l);
                    let jm = self.idx(i, (j + ny - 1) % ny, l);
                    let lp = self.idx(i, j, (l + 1) % nz);
                    let lm = self.idx(i, j, (l + nz - 1) % nz);

                    for c in 0..5 {
                        out.q[k][c] = (self.q[ip][c]
                            + self.q[im][c]
                            + self.q[jp][c]
                            + self.q[jm][c]
                            + self.q[lp][c]
                            + self.q[lm][c]
                            - 6.0 * self.q[k][c])
                            * inv_dx2;
                    }
                }
            }
        }
        out
    }

    /// Scalar order parameter S = (3/2) * max eigenvalue at each vertex.
    pub fn scalar_order_s(&self) -> Vec<f64> {
        (0..self.len())
            .map(|k| {
                let m = self.embed_matrix3(k);
                let eig = m.symmetric_eigenvalues();
                let max_eig = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                1.5 * max_eig
            })
            .collect()
    }

    /// Biaxiality parameter P = lambda_mid - lambda_min (secondary observable).
    pub fn biaxiality_p(&self) -> Vec<f64> {
        (0..self.len())
            .map(|k| {
                let m = self.embed_matrix3(k);
                let mut eig: Vec<f64> = m.symmetric_eigenvalues().iter().cloned().collect();
                eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eig[1] - eig[0]
            })
            .collect()
    }

    /// Director field: eigenvector of the largest eigenvalue at each vertex.
    /// Returns a Vec of unit 3-vectors, one per vertex.
    pub fn director(&self) -> Vec<[f64; 3]> {
        (0..self.len()).map(|k| {
            let m = self.embed_matrix3(k);
            let eig = m.symmetric_eigen();
            // Find index of largest eigenvalue
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..3 {
                if eig.eigenvalues[i] > max_val {
                    max_val = eig.eigenvalues[i];
                    max_idx = i;
                }
            }
            let col = eig.eigenvectors.column(max_idx);
            [col[0], col[1], col[2]]
        }).collect()
    }

    /// Mean scalar order parameter over the whole field.
    pub fn mean_s(&self) -> f64 {
        let s = self.scalar_order_s();
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Maximum Frobenius norm over all vertices.
    pub fn max_norm(&self) -> f64 {
        self.q
            .iter()
            .map(|&[q11, q12, q13, q22, q23]| {
                let q33 = -(q11 + q22);
                (q11 * q11
                    + q12 * q12
                    + q13 * q13
                    + q22 * q22
                    + q23 * q23
                    + q33 * q33)
                    .sqrt()
            })
            .fold(0.0_f64, f64::max)
    }

    /// Point-wise addition: return `self + other`.
    ///
    /// # Panics
    /// Panics if grids have different dimensions.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.nx, other.nx);
        assert_eq!(self.ny, other.ny);
        assert_eq!(self.nz, other.nz);
        let q = self
            .q
            .iter()
            .zip(&other.q)
            .map(|(&[a0, a1, a2, a3, a4], &[b0, b1, b2, b3, b4])| {
                [a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4]
            })
            .collect();
        Self {
            q,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            dx: self.dx,
        }
    }

    /// Point-wise scalar multiply: return `s * self`.
    pub fn scale(&self, s: f64) -> Self {
        let q = self
            .q
            .iter()
            .map(|&[a, b, c, d, e]| [s * a, s * b, s * c, s * d, s * e])
            .collect();
        Self {
            q,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            dx: self.dx,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// cartan-geo integration
// ─────────────────────────────────────────────────────────────────────────────

impl cartan_geo::QTensorField3D for QField3D {
    fn nx(&self) -> usize { self.nx }
    fn ny(&self) -> usize { self.ny }
    fn nz(&self) -> usize { self.nz }
    fn dx(&self) -> f64 { self.dx }
    fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        ((i % self.nx) * self.ny + (j % self.ny)) * self.nz + (l % self.nz)
    }
    fn embed_matrix3(&self, k: usize) -> nalgebra::SMatrix<f64, 3, 3> {
        self.embed_matrix3(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qfield3d_zeros() {
        let q = QField3D::zeros(4, 4, 4, 1.0);
        assert_eq!(q.len(), 64);
        assert_eq!(q.q[0], [0.0; 5]);
    }

    #[test]
    fn test_qfield3d_embed_traceless() {
        let q = QField3D::zeros(4, 4, 4, 1.0);
        let m = q.embed_matrix3(0);
        let tr = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        assert!(tr.abs() < 1e-12);
    }

    #[test]
    fn test_qfield3d_laplacian_uniform_is_zero() {
        let q = QField3D::uniform(8, 8, 8, 1.0, [0.1, 0.05, 0.0, -0.05, 0.0]);
        let lap = q.laplacian();
        for k in 0..lap.len() {
            for c in 0..5 {
                assert!(
                    lap.q[k][c].abs() < 1e-10,
                    "laplacian of uniform field must be zero, got {}",
                    lap.q[k][c]
                );
            }
        }
    }

    #[test]
    fn test_qfield3d_index_roundtrip() {
        let q = QField3D::zeros(4, 6, 5, 1.0);
        let (i, j, l) = (2, 3, 4);
        let k = q.idx(i, j, l);
        assert_eq!(q.ijk(k), (i, j, l));
    }

    #[test]
    fn test_qfield3d_uniform() {
        let q = QField3D::uniform(3, 3, 3, 1.0, [0.1, 0.2, 0.3, -0.05, -0.1]);
        assert_eq!(q.len(), 27);
        for &comp in &q.q {
            assert_eq!(comp, [0.1, 0.2, 0.3, -0.05, -0.1]);
        }
    }

    #[test]
    fn test_qfield3d_embed_symmetry() {
        let q = QField3D::uniform(2, 2, 2, 1.0, [0.1, 0.05, 0.02, -0.08, 0.03]);
        let m = q.embed_matrix3(0);
        // Check symmetry: m = m^T
        assert_abs_diff_eq!((m - m.transpose()).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_qfield3d_scalar_order_s() {
        let q = QField3D::zeros(4, 4, 4, 1.0);
        let s = q.scalar_order_s();
        assert_eq!(s.len(), 64);
        // For zero Q-tensor, all eigenvalues are zero, so S = 0.
        for &val in &s {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_qfield3d_add_and_scale() {
        let a = QField3D::uniform(2, 2, 2, 1.0, [1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = QField3D::uniform(2, 2, 2, 1.0, [0.5, 1.0, 1.5, 2.0, 2.5]);
        let c = a.add(&b);
        for &[q11, q12, q13, q22, q23] in &c.q {
            assert_abs_diff_eq!(q11, 1.5, epsilon = 1e-14);
            assert_abs_diff_eq!(q12, 3.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q13, 4.5, epsilon = 1e-14);
            assert_abs_diff_eq!(q22, 6.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q23, 7.5, epsilon = 1e-14);
        }

        let d = a.scale(2.0);
        for &[q11, q12, q13, q22, q23] in &d.q {
            assert_abs_diff_eq!(q11, 2.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q12, 4.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q13, 6.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q22, 8.0, epsilon = 1e-14);
            assert_abs_diff_eq!(q23, 10.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_qfield3d_random_perturbation_bounded() {
        let q = QField3D::random_perturbation(4, 4, 4, 1.0, 0.01, 42);
        for &[q11, q12, q13, q22, q23] in &q.q {
            assert!(q11.abs() <= 0.01 + 1e-12);
            assert!(q12.abs() <= 0.01 + 1e-12);
            assert!(q13.abs() <= 0.01 + 1e-12);
            assert!(q22.abs() <= 0.01 + 1e-12);
            assert!(q23.abs() <= 0.01 + 1e-12);
        }
    }

    #[test]
    fn test_qfield3d_max_norm() {
        let q = QField3D::uniform(2, 2, 2, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_abs_diff_eq!(q.max_norm(), 0.0, epsilon = 1e-14);

        let q = QField3D::uniform(2, 2, 2, 1.0, [1.0, 0.0, 0.0, 0.0, 0.0]);
        // ||[1, 0, 0, 0, 0, -1]|| = sqrt(2)
        assert_abs_diff_eq!(q.max_norm(), 2.0_f64.sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn test_qfield3d_biaxiality() {
        let q = QField3D::zeros(2, 2, 2, 1.0);
        let p = q.biaxiality_p();
        assert_eq!(p.len(), 8);
        // For zero Q-tensor, all eigenvalues are zero.
        for &val in &p {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-14);
        }
    }
}
