//! 3D velocity, concentration, and pressure field types.

use serde::{Deserialize, Serialize};

/// 3-component velocity field on a 3D periodic grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityField3D {
    pub u: Vec<[f64; 3]>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
}

impl VelocityField3D {
    /// Create a zero velocity field.
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self {
            u: vec![[0.0; 3]; nx * ny * nz],
            nx,
            ny,
            nz,
            dx,
        }
    }

    /// Create a uniform velocity field with every vertex set to `u`.
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, u: [f64; 3]) -> Self {
        Self {
            u: vec![u; nx * ny * nz],
            nx,
            ny,
            nz,
            dx,
        }
    }

    /// Linear index for vertex `(i, j, l)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.ny + j) * self.nz + l
    }

    /// Scalar divergence via central differences with periodic BCs.
    pub fn divergence(&self) -> ScalarField3D {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let inv_2dx = 1.0 / (2.0 * self.dx);
        let mut out = ScalarField3D::zeros(nx, ny, nz, self.dx);

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

                    out.phi[k] = (self.u[ip][0] - self.u[im][0]
                        + self.u[jp][1] - self.u[jm][1]
                        + self.u[lp][2] - self.u[lm][2])
                        * inv_2dx;
                }
            }
        }
        out
    }

    /// Velocity gradient tensor at vertex k.
    /// Returns (D, Omega): symmetric strain rate and antisymmetric vorticity.
    pub fn velocity_gradient_at(&self, k: usize) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let ll = k % self.nz;
        let ij = k / self.nz;
        let jj = ij % self.ny;
        let ii = ij / self.ny;
        let (i, j, l) = (ii, jj, ll);
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let inv_2dx = 1.0 / (2.0 * self.dx);

        let grads = [
            {
                let kp = self.idx((i + 1) % nx, j, l);
                let km = self.idx((i + nx - 1) % nx, j, l);
                [
                    self.u[kp][0] - self.u[km][0],
                    self.u[kp][1] - self.u[km][1],
                    self.u[kp][2] - self.u[km][2],
                ]
            },
            {
                let kp = self.idx(i, (j + 1) % ny, l);
                let km = self.idx(i, (j + ny - 1) % ny, l);
                [
                    self.u[kp][0] - self.u[km][0],
                    self.u[kp][1] - self.u[km][1],
                    self.u[kp][2] - self.u[km][2],
                ]
            },
            {
                let kp = self.idx(i, j, (l + 1) % nz);
                let km = self.idx(i, j, (l + nz - 1) % nz);
                [
                    self.u[kp][0] - self.u[km][0],
                    self.u[kp][1] - self.u[km][1],
                    self.u[kp][2] - self.u[km][2],
                ]
            },
        ];

        let mut w = [[0.0f64; 3]; 3];
        for alpha in 0..3 {
            for beta in 0..3 {
                w[alpha][beta] = grads[alpha][beta] * inv_2dx;
            }
        }

        let mut d = [[0.0f64; 3]; 3];
        let mut omega = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                d[a][b] = 0.5 * (w[a][b] + w[b][a]);
                omega[a][b] = 0.5 * (w[a][b] - w[b][a]);
            }
        }
        (d, omega)
    }
}

/// Scalar field (used for lipid concentration and as base for PressureField3D).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField3D {
    pub phi: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
}

impl ScalarField3D {
    /// Create a zero scalar field.
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self {
            phi: vec![0.0; nx * ny * nz],
            nx,
            ny,
            nz,
            dx,
        }
    }

    /// Create a uniform scalar field with every vertex set to `v`.
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, v: f64) -> Self {
        Self {
            phi: vec![v; nx * ny * nz],
            nx,
            ny,
            nz,
            dx,
        }
    }

    /// Mean value over all vertices.
    pub fn mean(&self) -> f64 {
        self.phi.iter().sum::<f64>() / self.phi.len() as f64
    }

    /// Maximum value over all vertices.
    pub fn max(&self) -> f64 {
        self.phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Linear index for vertex `(i, j, l)` with periodic wrapping.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.ny + j) * self.nz + l
    }

    /// 7-point finite-difference Laplacian ∇²φ at each vertex with periodic BCs.
    pub fn laplacian(&self) -> ScalarField3D {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let mut out = ScalarField3D::zeros(nx, ny, nz, self.dx);

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

                    out.phi[k] = (self.phi[ip]
                        + self.phi[im]
                        + self.phi[jp]
                        + self.phi[jm]
                        + self.phi[lp]
                        + self.phi[lm]
                        - 6.0 * self.phi[k])
                        * inv_dx2;
                }
            }
        }
        out
    }

    /// Gradient at each vertex: returns `Vec<[f64; 3]>` of (d/dx, d/dy, d/dz).
    pub fn gradient(&self) -> Vec<[f64; 3]> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let inv_2dx = 1.0 / (2.0 * self.dx);

        (0..self.phi.len())
            .map(|k| {
                let ll = k % nz;
                let ij = k / nz;
                let jj = ij % ny;
                let ii = ij / ny;

                let ip = self.idx((ii + 1) % nx, jj, ll);
                let im = self.idx((ii + nx - 1) % nx, jj, ll);
                let jp = self.idx(ii, (jj + 1) % ny, ll);
                let jm = self.idx(ii, (jj + ny - 1) % ny, ll);
                let lp = self.idx(ii, jj, (ll + 1) % nz);
                let lm = self.idx(ii, jj, (ll + nz - 1) % nz);

                [
                    (self.phi[ip] - self.phi[im]) * inv_2dx,
                    (self.phi[jp] - self.phi[jm]) * inv_2dx,
                    (self.phi[lp] - self.phi[lm]) * inv_2dx,
                ]
            })
            .collect()
    }
}

/// Lipid concentration field (phi).
pub type ConcentrationField3D = ScalarField3D;

/// Pressure field (p) — same storage as ScalarField3D, semantically distinct.
pub struct PressureField3D(pub ScalarField3D);

impl PressureField3D {
    /// Create a zero pressure field.
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self(ScalarField3D::zeros(nx, ny, nz, dx))
    }

    /// Access the pressure data as a slice.
    pub fn phi(&self) -> &Vec<f64> {
        &self.0.phi
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity3d_divergence_uniform_zero() {
        let v = VelocityField3D::uniform(8, 8, 8, 1.0, [1.0, 0.5, 0.0]);
        let div = v.divergence();
        for d in &div.phi {
            assert!(
                d.abs() < 1e-10,
                "divergence of uniform field must be zero"
            );
        }
    }

    #[test]
    fn test_concentration3d_mean_conserved() {
        let phi = ConcentrationField3D::uniform(4, 4, 4, 1.0, 0.5);
        assert!((phi.mean() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_pressure_field3d_zeros() {
        let p = PressureField3D::zeros(4, 4, 4, 1.0);
        assert_eq!(p.phi().len(), 64);
        assert!(p.phi().iter().all(|&x| x == 0.0));
    }
}
