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
use rand::{RngExt, SeedableRng};

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

/// Generic integrator support: `QFieldDec` is a field vector-space element.
///
/// Implemented in terms of the existing [`QFieldDec::add`] and
/// [`QFieldDec::scale`] so that `volterra_core::sim::integrate::rk4` reproduces
/// the legacy hand-rolled DEC `rk4_step` op tree (`self + other * factor` with
/// the identical per-element multiply/add order) bit-for-bit.
impl volterra_core::sim::integrate::FieldVec for QFieldDec {
    fn add_scaled(&self, other: &Self, factor: f64) -> Self {
        self.add(&other.scale(factor))
    }
}

#[cfg(test)]
mod fieldvec_tests {
    use super::QFieldDec;
    use volterra_core::sim::integrate::rk4;

    /// The legacy hand-rolled DEC RK4 op tree (lifted verbatim from
    /// `volterra-solver/src/runner_dec.rs::rk4_step`), generic over the rhs.
    fn legacy_rk4_step<R: Fn(&QFieldDec) -> QFieldDec>(
        q: &QFieldDec,
        dt: f64,
        rhs: &R,
    ) -> QFieldDec {
        let k1 = rhs(q);
        let q2 = q.add(&k1.scale(0.5 * dt));
        let k2 = rhs(&q2);
        let q3 = q.add(&k2.scale(0.5 * dt));
        let k3 = rhs(&q3);
        let q4 = q.add(&k3.scale(dt));
        let k4 = rhs(&q4);
        let rhs_sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        q.add(&rhs_sum.scale(dt / 6.0))
    }

    #[test]
    fn rk4_matches_legacy_rk4_step_bit_for_bit() {
        // A small handcrafted field with non-trivial values.
        let q = QFieldDec {
            q1: vec![0.137, -0.92, 1.55, 0.0041, -3.7],
            q2: vec![-0.5, 0.333, -1.2, 2.65, 0.6],
            n_vertices: 5,
        };
        let dt = 0.0137_f64;
        // A representative linear rhs standing in for beris_edwards_rhs_dec; the
        // FieldVec == legacy-rk4_step equivalence is a property of the op tree,
        // independent of the specific (deterministic) rhs.
        let rhs = |qq: &QFieldDec| qq.scale(-0.83);

        let generic = rk4(&q, dt, rhs);
        let legacy = legacy_rk4_step(&q, dt, &rhs);

        assert_eq!(generic.n_vertices, legacy.n_vertices);
        for (a, b) in generic.q1.iter().zip(&legacy.q1) {
            assert_eq!(a.to_bits(), b.to_bits(), "q1 differs: {a} vs {b}");
        }
        for (a, b) in generic.q2.iter().zip(&legacy.q2) {
            assert_eq!(a.to_bits(), b.to_bits(), "q2 differs: {a} vs {b}");
        }
    }
}
