// ~/volterra/volterra-fields/src/lib.rs

//! # volterra-fields
//!
//! Tensor field types for a 2D active nematic on a regular Cartesian grid.
//!
//! ## Coordinate conventions
//!
//! Vertices are indexed `(i, j)` with `i ∈ [0, nx)` (x direction) and
//! `j ∈ [0, ny)` (y direction). Linear index: `k = i * ny + j`.
//! All boundary conditions are periodic.
//!
//! ## Q-tensor representation
//!
//! In 2D the symmetric traceless Q-tensor is:
//!
//! ```text
//! Q = [[ q1,  q2 ],
//!      [ q2, -q1 ]]
//! ```
//!
//! with two independent components `(q1, q2)`. The scalar order parameter
//! is `S = 2 sqrt(q1² + q2²)` and the director angle is
//! `θ = atan2(q2, q1) / 2`.
//!
//! ## Embedding in 3D
//!
//! For holonomy-based defect detection the 2D Q-tensor is embedded in 3D:
//!
//! ```text
//! Q_3D = [[ q1,  q2,  0 ],
//!         [ q2, -q1,  0 ],
//!         [  0,   0,  0 ]]
//! ```
//!
//! This 3×3 matrix is then passed to `cartan_manifolds::frame_field::FrameField3D`.

use nalgebra::SMatrix;
use serde::{Deserialize, Serialize};

pub mod qfield3d;
pub use qfield3d::QField3D;

// ─────────────────────────────────────────────────────────────────────────────
// QField2D
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D Q-tensor field on a regular Cartesian grid with periodic BCs.
///
/// Each vertex stores two components `[q1, q2]` of the symmetric traceless
/// Q-tensor `Q = [[q1, q2],[q2,-q1]]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QField2D {
    /// Q-tensor components at each vertex, in row-major order (i*ny + j).
    pub q: Vec<[f64; 2]>,
    /// Number of vertices in x.
    pub nx: usize,
    /// Number of vertices in y.
    pub ny: usize,
    /// Grid spacing.
    pub dx: f64,
}

impl QField2D {
    /// Create a zero Q-tensor field.
    pub fn zeros(nx: usize, ny: usize, dx: f64) -> Self {
        Self { q: vec![[0.0, 0.0]; nx * ny], nx, ny, dx }
    }

    /// Create a uniform Q-tensor field with every vertex set to `q`.
    pub fn uniform(nx: usize, ny: usize, dx: f64, q: [f64; 2]) -> Self {
        Self { q: vec![q; nx * ny], nx, ny, dx }
    }

    /// Create a small-amplitude random perturbation (good initial condition for
    /// active turbulence simulations).
    ///
    /// Each component is drawn uniformly from `[-amplitude, amplitude]`.
    pub fn random_perturbation(
        nx: usize,
        ny: usize,
        dx: f64,
        amplitude: f64,
        seed: u64,
    ) -> Self {
        // Linear congruential generator for reproducibility without pulling in rand.
        let mut state: u64 = seed.wrapping_add(1);
        let mut next = move || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let q: Vec<[f64; 2]> = (0..nx * ny)
            .map(|_| [next() * amplitude, next() * amplitude])
            .collect();
        Self { q, nx, ny, dx }
    }

    /// Linear index for vertex `(i, j)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        (i % self.nx) * self.ny + (j % self.ny)
    }

    /// Linear index with signed (possibly negative) coordinates, using periodic BCs.
    #[inline]
    pub fn idx_i(&self, i: i64, j: i64) -> usize {
        let ii = i.rem_euclid(self.nx as i64) as usize;
        let jj = j.rem_euclid(self.ny as i64) as usize;
        ii * self.ny + jj
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns `true` if the field has no vertices.
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// 5-point finite-difference Laplacian ∇²Q at each vertex.
    ///
    /// Uses the isotropic 5-point stencil on a square grid with periodic BCs:
    ///
    /// ```text
    /// ∇²f_{i,j} ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4 f_{i,j}) / dx²
    /// ```
    pub fn laplacian(&self) -> Self {
        let dx2 = self.dx * self.dx;
        let mut out = Self::zeros(self.nx, self.ny, self.dx);
        for i in 0..self.nx {
            for j in 0..self.ny {
                let k = self.idx(i, j);
                let kp = self.idx_i(i as i64 + 1, j as i64);
                let km = self.idx_i(i as i64 - 1, j as i64);
                let kpj = self.idx_i(i as i64, j as i64 + 1);
                let kmj = self.idx_i(i as i64, j as i64 - 1);
                out.q[k][0] = (self.q[kp][0] + self.q[km][0] + self.q[kpj][0]
                    + self.q[kmj][0] - 4.0 * self.q[k][0])
                    / dx2;
                out.q[k][1] = (self.q[kp][1] + self.q[km][1] + self.q[kpj][1]
                    + self.q[kmj][1] - 4.0 * self.q[k][1])
                    / dx2;
            }
        }
        out
    }

    /// Scalar order parameter squared: |Q|² = q1² + q2² at each vertex.
    ///
    /// The scalar order parameter satisfies S = 2 sqrt(|Q|²).
    pub fn order_param_sq(&self) -> Vec<f64> {
        self.q.iter().map(|&[q1, q2]| q1 * q1 + q2 * q2).collect()
    }

    /// Director angle θ ∈ (-π/2, π/2] at each vertex.
    ///
    /// The 2D director is n = (cos θ, sin θ) and Q = S/2 (n⊗n - I/2),
    /// so θ = atan2(q2, q1) / 2.
    pub fn director_angle(&self) -> Vec<f64> {
        self.q
            .iter()
            .map(|&[q1, q2]| q2.atan2(q1) / 2.0)
            .collect()
    }

    /// Embed each 2D Q-tensor as a 3×3 sym-traceless matrix for holonomy computation.
    ///
    /// ```text
    /// Q_3D = [[ q1,  q2,  0 ],
    ///         [ q2, -q1,  0 ],
    ///         [  0,   0,  0 ]]
    /// ```
    pub fn to_q3d(&self) -> Vec<SMatrix<f64, 3, 3>> {
        self.q
            .iter()
            .map(|&[q1, q2]| {
                SMatrix::<f64, 3, 3>::from_row_slice(&[
                    q1, q2, 0.0, q2, -q1, 0.0, 0.0, 0.0, 0.0,
                ])
            })
            .collect()
    }

    /// Point-wise addition: return `self + other`.
    ///
    /// # Panics
    /// Panics if grids have different dimensions.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.nx, other.nx);
        assert_eq!(self.ny, other.ny);
        let q = self
            .q
            .iter()
            .zip(&other.q)
            .map(|(&[a0, a1], &[b0, b1])| [a0 + b0, a1 + b1])
            .collect();
        Self { q, nx: self.nx, ny: self.ny, dx: self.dx }
    }

    /// Point-wise scalar multiply: return `s * self`.
    pub fn scale(&self, s: f64) -> Self {
        let q = self.q.iter().map(|&[a, b]| [s * a, s * b]).collect();
        Self { q, nx: self.nx, ny: self.ny, dx: self.dx }
    }

    /// Max Frobenius norm over all vertices (useful for CFL monitoring).
    pub fn max_norm(&self) -> f64 {
        self.q
            .iter()
            .map(|&[q1, q2]| (q1 * q1 + q2 * q2).sqrt())
            .fold(0.0_f64, f64::max)
    }

    /// Mean scalar order parameter S = mean(2 sqrt(q1²+q2²)).
    pub fn mean_order_param(&self) -> f64 {
        let sum: f64 = self
            .q
            .iter()
            .map(|&[q1, q2]| 2.0 * (q1 * q1 + q2 * q2).sqrt())
            .sum();
        sum / self.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VelocityField2D
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D incompressible velocity field on a regular Cartesian grid.
///
/// Each vertex stores `[vx, vy]`. The field is assumed divergence-free
/// (enforced by the pressure solver in volterra-solver).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityField2D {
    /// Velocity components `[vx, vy]` at each vertex, in row-major order.
    pub v: Vec<[f64; 2]>,
    /// Number of vertices in x.
    pub nx: usize,
    /// Number of vertices in y.
    pub ny: usize,
    /// Grid spacing.
    pub dx: f64,
}

impl VelocityField2D {
    /// Create a zero velocity field.
    pub fn zeros(nx: usize, ny: usize, dx: f64) -> Self {
        Self { v: vec![[0.0, 0.0]; nx * ny], nx, ny, dx }
    }

    /// Linear index for vertex `(i, j)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        (i % self.nx) * self.ny + (j % self.ny)
    }

    /// Linear index with signed coordinates.
    #[inline]
    pub fn idx_i(&self, i: i64, j: i64) -> usize {
        let ii = i.rem_euclid(self.nx as i64) as usize;
        let jj = j.rem_euclid(self.ny as i64) as usize;
        ii * self.ny + jj
    }

    /// Compute u·∇Q at each vertex using upwind advection.
    ///
    /// For each component q_α of the Q-tensor, the upwind advection is:
    ///
    /// ```text
    /// (u·∇q)_{i,j} ≈ vx * ∂_x q + vy * ∂_y q
    /// ```
    ///
    /// with first-order upwind differences: if vx > 0, use backward difference
    /// in x; if vx < 0, use forward difference.
    pub fn advect(&self, q: &QField2D) -> QField2D {
        assert_eq!(self.nx, q.nx);
        assert_eq!(self.ny, q.ny);
        let dx = self.dx;
        let mut out = QField2D::zeros(self.nx, self.ny, dx);

        for i in 0..self.nx {
            for j in 0..self.ny {
                let k = self.idx(i, j);
                let [vx, vy] = self.v[k];

                // x-direction upwind.
                let dqx = if vx >= 0.0 {
                    let km = q.idx_i(i as i64 - 1, j as i64);
                    [(q.q[k][0] - q.q[km][0]) / dx, (q.q[k][1] - q.q[km][1]) / dx]
                } else {
                    let kp = q.idx_i(i as i64 + 1, j as i64);
                    [(q.q[kp][0] - q.q[k][0]) / dx, (q.q[kp][1] - q.q[k][1]) / dx]
                };

                // y-direction upwind.
                let dqy = if vy >= 0.0 {
                    let km = q.idx_i(i as i64, j as i64 - 1);
                    [(q.q[k][0] - q.q[km][0]) / dx, (q.q[k][1] - q.q[km][1]) / dx]
                } else {
                    let kp = q.idx_i(i as i64, j as i64 + 1);
                    [(q.q[kp][0] - q.q[k][0]) / dx, (q.q[kp][1] - q.q[k][1]) / dx]
                };

                out.q[k][0] = vx * dqx[0] + vy * dqy[0];
                out.q[k][1] = vx * dqx[1] + vy * dqy[1];
            }
        }
        out
    }

    /// Max velocity magnitude (useful for CFL check).
    pub fn max_speed(&self) -> f64 {
        self.v
            .iter()
            .map(|&[vx, vy]| (vx * vx + vy * vy).sqrt())
            .fold(0.0_f64, f64::max)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScalarField2D
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D scalar field φ(x,y) on a regular Cartesian grid with periodic BCs.
///
/// Used to represent the lipid volume fraction φ_l(x,t) in the Cahn-Hilliard
/// equation of the Beris-Edwards-Cahn-Hilliard (BECH) system.  Each vertex
/// stores a single `f64`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField2D {
    /// Scalar values at each vertex, in row-major order (i*ny + j).
    pub phi: Vec<f64>,
    /// Number of vertices in x.
    pub nx: usize,
    /// Number of vertices in y.
    pub ny: usize,
    /// Grid spacing (same in x and y; periodic BCs).
    pub dx: f64,
}

impl ScalarField2D {
    /// Create a zero scalar field.
    pub fn zeros(nx: usize, ny: usize, dx: f64) -> Self {
        Self { phi: vec![0.0; nx * ny], nx, ny, dx }
    }

    /// Create a uniform scalar field with every vertex set to `val`.
    pub fn uniform(nx: usize, ny: usize, dx: f64, val: f64) -> Self {
        Self { phi: vec![val; nx * ny], nx, ny, dx }
    }

    /// Linear index for vertex `(i, j)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        (i % self.nx) * self.ny + (j % self.ny)
    }

    /// Linear index with signed (possibly negative) coordinates, using periodic BCs.
    #[inline]
    pub fn idx_i(&self, i: i64, j: i64) -> usize {
        let ii = i.rem_euclid(self.nx as i64) as usize;
        let jj = j.rem_euclid(self.ny as i64) as usize;
        ii * self.ny + jj
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.phi.len()
    }

    /// Returns `true` if the field has no vertices.
    pub fn is_empty(&self) -> bool {
        self.phi.is_empty()
    }

    /// 5-point finite-difference Laplacian ∇²φ at each vertex.
    ///
    /// ```text
    /// ∇²φ_{i,j} ≈ (φ_{i+1,j} + φ_{i-1,j} + φ_{i,j+1} + φ_{i,j-1} - 4φ_{i,j}) / dx²
    /// ```
    pub fn laplacian(&self) -> Self {
        let dx2 = self.dx * self.dx;
        let mut out = Self::zeros(self.nx, self.ny, self.dx);
        for i in 0..self.nx {
            for j in 0..self.ny {
                let k   = self.idx(i, j);
                let kp  = self.idx_i(i as i64 + 1, j as i64);
                let km  = self.idx_i(i as i64 - 1, j as i64);
                let kpj = self.idx_i(i as i64, j as i64 + 1);
                let kmj = self.idx_i(i as i64, j as i64 - 1);
                out.phi[k] = (self.phi[kp] + self.phi[km]
                    + self.phi[kpj] + self.phi[kmj]
                    - 4.0 * self.phi[k]) / dx2;
            }
        }
        out
    }

    /// Point-wise addition: return `self + other`.
    ///
    /// # Panics
    /// Panics if grids have different dimensions.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.nx, other.nx);
        assert_eq!(self.ny, other.ny);
        let phi = self.phi.iter().zip(&other.phi).map(|(&a, &b)| a + b).collect();
        Self { phi, nx: self.nx, ny: self.ny, dx: self.dx }
    }

    /// Point-wise scalar multiply: return `s * self`.
    pub fn scale(&self, s: f64) -> Self {
        let phi = self.phi.iter().map(|&a| s * a).collect();
        Self { phi, nx: self.nx, ny: self.ny, dx: self.dx }
    }

    /// Mean value ⟨φ⟩ over all vertices.
    pub fn mean_value(&self) -> f64 {
        self.phi.iter().sum::<f64>() / self.len() as f64
    }

    /// Maximum value over all vertices.
    pub fn max_value(&self) -> f64 {
        self.phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum value over all vertices.
    pub fn min_value(&self) -> f64 {
        self.phi.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Variance Var[φ] = ⟨φ²⟩ - ⟨φ⟩².
    pub fn variance(&self) -> f64 {
        let mean = self.mean_value();
        let mean_sq: f64 = self.phi.iter().map(|&p| p * p).sum::<f64>() / self.len() as f64;
        (mean_sq - mean * mean).max(0.0)
    }

    /// Mean of |∇φ|² ≈ mean of [(Δ_x φ)² + (Δ_y φ)²] / (2 dx²).
    ///
    /// Approximated using centered differences.  Proportional to the CH
    /// interfacial energy ∫ κ_l |∇φ|² d²x.
    pub fn mean_gradient_sq(&self) -> f64 {
        let mut sum = 0.0_f64;
        let dx2 = self.dx * self.dx;
        for i in 0..self.nx {
            for j in 0..self.ny {
                let kp = self.idx_i(i as i64 + 1, j as i64);
                let km = self.idx_i(i as i64 - 1, j as i64);
                let kpj = self.idx_i(i as i64, j as i64 + 1);
                let kmj = self.idx_i(i as i64, j as i64 - 1);
                let gx = (self.phi[kp] - self.phi[km]) / (2.0 * self.dx);
                let gy = (self.phi[kpj] - self.phi[kmj]) / (2.0 * self.dx);
                sum += gx * gx + gy * gy;
            }
        }
        sum * dx2 / (self.len() as f64 * dx2) // = sum / len
    }
}

impl VelocityField2D {
    /// Upwind advection of a scalar field: returns u·∇φ at each vertex.
    ///
    /// Same upwind convention as [`VelocityField2D::advect`] for Q-tensors.
    pub fn advect_scalar(&self, phi: &ScalarField2D) -> ScalarField2D {
        assert_eq!(self.nx, phi.nx);
        assert_eq!(self.ny, phi.ny);
        let dx = self.dx;
        let mut out = ScalarField2D::zeros(self.nx, self.ny, dx);
        for i in 0..self.nx {
            for j in 0..self.ny {
                let k = self.idx(i, j);
                let [vx, vy] = self.v[k];

                let dpx = if vx >= 0.0 {
                    let km = phi.idx_i(i as i64 - 1, j as i64);
                    (phi.phi[k] - phi.phi[km]) / dx
                } else {
                    let kp = phi.idx_i(i as i64 + 1, j as i64);
                    (phi.phi[kp] - phi.phi[k]) / dx
                };

                let dpy = if vy >= 0.0 {
                    let km = phi.idx_i(i as i64, j as i64 - 1);
                    (phi.phi[k] - phi.phi[km]) / dx
                } else {
                    let kp = phi.idx_i(i as i64, j as i64 + 1);
                    (phi.phi[kp] - phi.phi[k]) / dx
                };

                out.phi[k] = vx * dpx + vy * dpy;
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zeros_is_zero() {
        let q = QField2D::zeros(4, 4, 1.0);
        assert_eq!(q.len(), 16);
        for &[a, b] in &q.q {
            assert_eq!(a, 0.0);
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn laplacian_of_constant_is_zero() {
        // ∇²(constant) = 0 everywhere.
        let q = QField2D::uniform(8, 8, 1.0, [0.3, -0.1]);
        let lap = q.laplacian();
        for &[a, b] in &lap.q {
            assert_abs_diff_eq!(a, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(b, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn laplacian_of_cosine_wave() {
        // For f = cos(2π k x / L) on a grid of size L with spacing dx=1:
        // ∇²f = -(2π k / L)² f at each point (up to FD error O(dx²)).
        let nx = 32;
        let ny = 32;
        let dx = 1.0;
        let mut q = QField2D::zeros(nx, ny, dx);
        let k: f64 = 2.0; // wave number
        let lx = (nx as f64) * dx;
        for i in 0..nx {
            let x = i as f64 * dx;
            let val = (2.0 * std::f64::consts::PI * k * x / lx).cos();
            for j in 0..ny {
                let idx = q.idx(i, j);
                q.q[idx][0] = val;
            }
        }
        let lap = q.laplacian();
        let expected_eigenvalue = -(2.0 * std::f64::consts::PI * k / lx).powi(2);
        // Allow 2% error (FD truncation on a 32-point grid, O(dx²) scheme).
        for i in 0..nx {
            for j in 0..ny {
                let k_idx = q.idx(i, j);
                let fd = lap.q[k_idx][0];
                let exact = expected_eigenvalue * q.q[k_idx][0];
                assert_abs_diff_eq!(fd, exact, epsilon = 0.02 * expected_eigenvalue.abs());
            }
        }
    }

    #[test]
    fn to_q3d_shape_and_symmetry() {
        let q = QField2D::uniform(2, 2, 1.0, [0.2, 0.1]);
        let q3d = q.to_q3d();
        assert_eq!(q3d.len(), 4);
        for m in &q3d {
            // Symmetric
            assert_abs_diff_eq!((m - m.transpose()).norm(), 0.0, epsilon = 1e-15);
            // Traceless
            assert_abs_diff_eq!(m.trace(), 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn add_and_scale() {
        let a = QField2D::uniform(4, 4, 1.0, [1.0, 2.0]);
        let b = QField2D::uniform(4, 4, 1.0, [3.0, 4.0]);
        let c = a.add(&b);
        for &[q1, q2] in &c.q {
            assert_abs_diff_eq!(q1, 4.0, epsilon = 1e-15);
            assert_abs_diff_eq!(q2, 6.0, epsilon = 1e-15);
        }
        let d = a.scale(2.0);
        for &[q1, q2] in &d.q {
            assert_abs_diff_eq!(q1, 2.0, epsilon = 1e-15);
            assert_abs_diff_eq!(q2, 4.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn advect_by_zero_velocity_is_zero() {
        let q = QField2D::uniform(8, 8, 1.0, [0.3, -0.2]);
        let v = VelocityField2D::zeros(8, 8, 1.0);
        let adv = v.advect(&q);
        for &[a, b] in &adv.q {
            assert_abs_diff_eq!(a, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(b, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn order_param_sq_correct() {
        let q = QField2D::uniform(2, 2, 1.0, [0.3, 0.4]);
        let sq = q.order_param_sq();
        for &s in &sq {
            assert_abs_diff_eq!(s, 0.09 + 0.16, epsilon = 1e-14);
        }
    }

    #[test]
    fn random_perturbation_bounded() {
        let q = QField2D::random_perturbation(8, 8, 1.0, 0.01, 42);
        for &[q1, q2] in &q.q {
            assert!(q1.abs() <= 0.01 + 1e-12);
            assert!(q2.abs() <= 0.01 + 1e-12);
        }
    }

    // ── ScalarField2D ────────────────────────────────────────────────────────

    #[test]
    fn scalar_laplacian_of_constant_is_zero() {
        let phi = ScalarField2D::uniform(8, 8, 1.0, 3.14);
        let lap = phi.laplacian();
        for &v in &lap.phi {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn scalar_mean_and_variance() {
        // Uniform field: mean = val, variance = 0.
        let phi = ScalarField2D::uniform(8, 8, 1.0, 2.5);
        assert_abs_diff_eq!(phi.mean_value(), 2.5, epsilon = 1e-14);
        assert_abs_diff_eq!(phi.variance(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn scalar_add_and_scale() {
        let a = ScalarField2D::uniform(4, 4, 1.0, 1.0);
        let b = ScalarField2D::uniform(4, 4, 1.0, 2.0);
        let c = a.add(&b);
        for &v in &c.phi {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-14);
        }
        let d = a.scale(5.0);
        for &v in &d.phi {
            assert_abs_diff_eq!(v, 5.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn advect_scalar_by_zero_velocity_is_zero() {
        let phi = ScalarField2D::uniform(8, 8, 1.0, 1.0);
        let v = VelocityField2D::zeros(8, 8, 1.0);
        let adv = v.advect_scalar(&phi);
        for &val in &adv.phi {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-15);
        }
    }
}
