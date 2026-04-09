//! Q-tensor field on a DEC mesh (2D surfaces).
//!
//! Stores a traceless symmetric 2-tensor Q per vertex in the representation
//! q1 = Q_xx, q2 = Q_xy. The full tensor is reconstructed as
//! Q = [[q1, q2], [q2, -q1]] (tracelessness: Q_yy = -Q_xx).
//!
//! For interfacing with cartan-dec's Lichnerowicz Laplacian (which expects
//! [Q_xx, Q_xy, Q_yy] layout), use [`QFieldDec::to_lichnerowicz_layout`] and
//! [`QFieldDec::from_lichnerowicz_layout`].

use nalgebra::DVector;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Q-tensor field on a 2D DEC mesh.
///
/// Two independent components per vertex: q1 = Q_xx and q2 = Q_xy.
/// The tracelessness constraint Q_yy = -Q_xx is automatic.
///
/// The scalar order parameter is S = 2 sqrt(q1^2 + q2^2), and Tr(Q^2) =
/// 2(q1^2 + q2^2).
#[derive(Debug, Clone)]
pub struct QFieldDec {
    /// Q_xx component at each vertex.
    pub q1: Vec<f64>,
    /// Q_xy component at each vertex.
    pub q2: Vec<f64>,
    /// Number of vertices.
    pub n_vertices: usize,
}

impl QFieldDec {
    /// All-zero Q-field on a mesh with `nv` vertices.
    pub fn zeros(nv: usize) -> Self {
        Self {
            q1: vec![0.0; nv],
            q2: vec![0.0; nv],
            n_vertices: nv,
        }
    }

    /// Uniform Q-field: same (q1, q2) at every vertex.
    pub fn uniform(nv: usize, q1_val: f64, q2_val: f64) -> Self {
        Self {
            q1: vec![q1_val; nv],
            q2: vec![q2_val; nv],
            n_vertices: nv,
        }
    }

    /// Small random perturbation around zero.
    pub fn random_perturbation(nv: usize, amplitude: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let q1: Vec<f64> = (0..nv)
            .map(|_| amplitude * (2.0 * rng.random::<f64>() - 1.0))
            .collect();
        let q2: Vec<f64> = (0..nv)
            .map(|_| amplitude * (2.0 * rng.random::<f64>() - 1.0))
            .collect();
        Self {
            q1,
            q2,
            n_vertices: nv,
        }
    }

    /// Scalar order parameter S = 2 sqrt(q1^2 + q2^2) at each vertex.
    pub fn scalar_order(&self) -> Vec<f64> {
        self.q1
            .iter()
            .zip(&self.q2)
            .map(|(a, b)| 2.0 * (a * a + b * b).sqrt())
            .collect()
    }

    /// Mean scalar order parameter over all vertices.
    pub fn mean_order_param(&self) -> f64 {
        let s = self.scalar_order();
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Tr(Q^2) = 2(q1^2 + q2^2) at each vertex.
    pub fn trace_q_squared(&self) -> Vec<f64> {
        self.q1
            .iter()
            .zip(&self.q2)
            .map(|(a, b)| 2.0 * (a * a + b * b))
            .collect()
    }

    /// Convert to the [Q_xx, Q_xy, Q_yy] layout expected by
    /// [`cartan_dec::Operators::apply_lichnerowicz_laplacian`].
    ///
    /// Returns a DVector of length 3 * n_vertices.
    pub fn to_lichnerowicz_layout(&self) -> DVector<f64> {
        let nv = self.n_vertices;
        let mut v = DVector::zeros(3 * nv);
        for i in 0..nv {
            v[i] = self.q1[i]; // Q_xx = q1
            v[nv + i] = self.q2[i]; // Q_xy = q2
            v[2 * nv + i] = -self.q1[i]; // Q_yy = -q1 (traceless)
        }
        v
    }

    /// Construct from the [Q_xx, Q_xy, Q_yy] layout returned by
    /// [`cartan_dec::Operators::apply_lichnerowicz_laplacian`].
    pub fn from_lichnerowicz_layout(v: &DVector<f64>) -> Self {
        let nv = v.len() / 3;
        assert_eq!(v.len(), 3 * nv);
        let q1: Vec<f64> = (0..nv).map(|i| v[i]).collect();
        let q2: Vec<f64> = (0..nv).map(|i| v[nv + i]).collect();
        Self {
            q1,
            q2,
            n_vertices: nv,
        }
    }

    /// Pointwise addition: self + other.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n_vertices, other.n_vertices);
        Self {
            q1: self
                .q1
                .iter()
                .zip(&other.q1)
                .map(|(a, b)| a + b)
                .collect(),
            q2: self
                .q2
                .iter()
                .zip(&other.q2)
                .map(|(a, b)| a + b)
                .collect(),
            n_vertices: self.n_vertices,
        }
    }

    /// Pointwise scalar multiplication.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            q1: self.q1.iter().map(|a| a * s).collect(),
            q2: self.q2.iter().map(|a| a * s).collect(),
            n_vertices: self.n_vertices,
        }
    }
}
