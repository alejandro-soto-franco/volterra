//! Complex nematic field on 2-manifolds.
//!
//! Wraps cartan-dec's `Section<2>` with physics convenience methods for
//! active nematic simulations: scalar order parameter, Tr(Q^2),
//! conversion to/from the old (q1, q2) layout.

use num_complex::Complex;
use cartan_dec::line_bundle::Section;

use crate::qfield_dec::QFieldDec;

/// Complex nematic field on a 2-manifold.
///
/// Stores one `Complex<f64>` per vertex. The complex number z = q1 + i*q2
/// encodes the traceless symmetric Q-tensor: Q = [[q1, q2], [q2, -q1]].
///
/// The scalar order parameter is S = 2|z| and the director angle is
/// theta = arg(z) / 2 (for K=2 nematic symmetry).
#[derive(Debug, Clone)]
pub struct NematicField2D {
    /// Underlying complex section of L_2.
    pub section: Section<2>,
}

impl NematicField2D {
    /// Zero nematic field on `nv` vertices.
    pub fn zeros(nv: usize) -> Self {
        Self {
            section: Section::<2>::zeros(nv),
        }
    }

    /// Uniform nematic field.
    pub fn uniform(nv: usize, q1: f64, q2: f64) -> Self {
        Self {
            section: Section::<2>::uniform(nv, Complex::new(q1, q2)),
        }
    }

    /// Construct from a complex section.
    pub fn from_section(section: Section<2>) -> Self {
        Self { section }
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.section.n_vertices()
    }

    /// Scalar order parameter S = 2|z| at each vertex.
    pub fn scalar_order(&self) -> Vec<f64> {
        self.section.scalar_order()
    }

    /// Mean scalar order parameter.
    pub fn mean_scalar_order(&self) -> f64 {
        self.section.mean_scalar_order()
    }

    /// Tr(Q^2) = 2(q1^2 + q2^2) = 2|z|^2 at each vertex.
    pub fn trace_q_squared(&self) -> Vec<f64> {
        self.section.values.iter().map(|z| 2.0 * z.norm_sqr()).collect()
    }

    /// Normalise to unit order parameter (|z| = 1) at each vertex.
    pub fn normalise(&mut self) {
        self.section.normalise(1e-15);
    }

    /// Convert to the old (q1, q2) real-component representation.
    pub fn to_qfield_dec(&self) -> QFieldDec {
        let (q1, q2) = self.section.to_real_components();
        QFieldDec {
            q1,
            q2,
            n_vertices: self.section.n_vertices(),
        }
    }

    /// Construct from the old (q1, q2) real-component representation.
    pub fn from_qfield_dec(q: &QFieldDec) -> Self {
        Self {
            section: Section::<2>::from_real_components(&q.q1, &q.q2),
        }
    }

    /// Access the underlying complex values.
    pub fn values(&self) -> &[Complex<f64>] {
        &self.section.values
    }

    /// Mutable access to the underlying complex values.
    pub fn values_mut(&mut self) -> &mut [Complex<f64>] {
        &mut self.section.values
    }
}
