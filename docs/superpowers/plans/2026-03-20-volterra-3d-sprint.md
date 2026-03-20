# Volterra 3D Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the volterra active nematics library with a full 3D Beris-Edwards + Stokes + Cahn-Hilliard solver, differential geometric disclination line tracking, and Python bindings — all added alongside the existing 2D stack without touching it.

**Architecture:** Parallel 3D layer across five crates. cartan-geo gains a new `disclination` module first (it has no upstream dependencies). volterra-core, volterra-fields, and volterra-solver then build up the physics engine. volterra-py wraps everything last. Natural checkpoint after Task 14 (full Rust engine) before Python bindings.

**Tech Stack:** Rust 1.85, nalgebra 0.33, rustfft 6, rayon, pyo3 0.25, numpy 0.25. Workspace root: `~/volterra`. Cartan sibling workspace: `~/cartan`.

**Spec:** `docs/superpowers/specs/2026-03-20-volterra-3d-sprint-design.md`

---

## File Map

### New files
```
cartan/cartan-geo/src/disclination/mod.rs      -- types + re-exports
cartan/cartan-geo/src/disclination/segments.rs -- Layer 1: scan_disclination_lines_3d
cartan/cartan-geo/src/disclination/lines.rs    -- Layer 2: connect_disclination_lines
cartan/cartan-geo/src/disclination/events.rs   -- Layer 3: track_disclination_events

volterra-fields/src/qfield3d.rs   -- QField3D (5-component 3D Q-tensor)
volterra-fields/src/fields3d.rs   -- VelocityField3D, ConcentrationField3D, PressureField3D

volterra-solver/src/mol_field_3d.rs  -- molecular_field_3d, co_rotation_3d
volterra-solver/src/beris_3d.rs      -- beris_edwards_rhs_3d, Euler/RK4 integrators
volterra-solver/src/stokes_3d.rs     -- stokes_solve_3d (spectral 3D FFT)
volterra-solver/src/ch_3d.rs         -- ch_step_etd_3d (Cahn-Hilliard ETD)
volterra-solver/src/defects_3d.rs    -- scan_defects_3d, track_defect_events
volterra-solver/src/runner_3d.rs     -- run_mars_3d, run_mars_3d_full, stats types

volterra-py/src/bindings_3d.rs  -- all 3D Python class and function bindings
```

### Modified files
```
cartan/cartan-geo/src/lib.rs           -- add pub mod disclination + re-exports
volterra-core/src/lib.rs               -- add MarsParams3D
volterra-fields/src/lib.rs             -- add mod qfield3d, mod fields3d, re-exports
volterra-solver/src/lib.rs             -- add mod declarations + re-exports
volterra-py/src/lib.rs                 -- add mod bindings_3d + register classes
volterra/src/lib.rs                    -- extend prelude with 3D symbols
```

---

## Task 1: MarsParams3D

**Files:**
- Modify: `volterra-core/src/lib.rs` (append after existing MarsParams impl)

The struct fields and their physical meaning:
- `nx, ny, nz: usize` — grid dimensions
- `dx, dt: f64` — grid spacing and time step
- `k_r, gamma_r, zeta_eff, eta: f64` — elastic constant, rotational viscosity, activity, fluid viscosity
- `a_landau, c_landau, b_landau: f64` — LdG bulk coefficients (b_landau=0 for uniaxial)
- `lambda: f64` — flow-alignment parameter xi (see paper eq. xi_flow; MARS r>=5 gives xi>0.923)
- `noise_amp: f64` — Langevin noise amplitude sqrt(2 Gamma_r k_B T)
- `chi_a: f64` — magnetic susceptibility anisotropy = mu_0 * Delta_chi / 2
- `b0, omega_b: f64` — field magnitude and rotation frequency
- `k_l, gamma_l, xi_l: f64` — lipid elastic, viscosity, coupling length
- `chi_ms, kappa_ch, a_ch, b_ch, m_l: f64` — CH/MS coupling params

- [ ] **Write the failing test** (append to volterra-core/src/lib.rs)

```rust
#[cfg(test)]
mod tests_3d {
    use super::*;

    #[test]
    fn test_mars_params_3d_validate_ok() {
        let p = MarsParams3D::default_test();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_mars_params_3d_defect_length() {
        let p = MarsParams3D::default_test();
        let ld = p.defect_length();
        assert!(ld > 0.0, "defect_length must be positive");
        // ld = sqrt(k_r / zeta_eff) = sqrt(1/2) ~ 0.707
        assert!((ld - (0.5f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mars_params_3d_invalid_nz() {
        let mut p = MarsParams3D::default_test();
        p.nz = 0;
        assert!(p.validate().is_err());
    }
}
```

- [ ] **Run to verify failure**

```bash
cd ~/volterra && cargo test -p volterra-core tests_3d 2>&1 | tail -10
```
Expected: `error[E0433]: failed to resolve: use of undeclared type MarsParams3D`

- [ ] **Implement MarsParams3D** (append to volterra-core/src/lib.rs)

```rust
/// All physical and numerical parameters for the 3D MARS + lipid simulation.
///
/// ## Symbol conventions
///
/// `lambda` in this struct is the flow-alignment parameter xi from Jeffery orbit
/// theory (eq. xi_flow in the paper): xi = (r^2-1)/(r^2+1). Named `lambda` in
/// the struct to match the MarsParams convention; used as `xi` in all physics docs.
///
/// `chi_a` encodes mu_0 * Delta_chi / 2 (SI). The magnetic torque molecular field
/// H_mag = chi_a * b0^2 * [...]. Do NOT multiply by gamma_r inside molecular_field_3d;
/// the single Gamma_r multiplication occurs in beris_edwards_rhs_3d.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarsParams3D {
    pub nx: usize, pub ny: usize, pub nz: usize,
    pub dx: f64, pub dt: f64,
    pub k_r: f64, pub gamma_r: f64, pub zeta_eff: f64, pub eta: f64,
    pub a_landau: f64, pub c_landau: f64, pub b_landau: f64,
    pub lambda: f64,
    pub noise_amp: f64,
    pub chi_a: f64, pub b0: f64, pub omega_b: f64,
    pub k_l: f64, pub gamma_l: f64, pub xi_l: f64,
    pub chi_ms: f64, pub kappa_ch: f64, pub a_ch: f64, pub b_ch: f64, pub m_l: f64,
}

impl MarsParams3D {
    pub fn defect_length(&self) -> f64 { (self.k_r / self.zeta_eff).sqrt() }
    pub fn pi_number(&self) -> f64 { self.k_r / (self.gamma_l * self.eta * self.k_l) }
    pub fn a_eff(&self) -> f64 { self.a_landau - self.zeta_eff / 2.0 }
    pub fn ch_coherence_length(&self) -> f64 { (self.kappa_ch / self.a_ch).sqrt() }
    pub fn phi_eq(&self) -> f64 { (self.a_ch / self.b_ch).sqrt() }

    pub fn validate(&self) -> Result<(), VError> {
        if self.nx < 2 { return Err(VError::InvalidParams("nx must be >= 2".into())); }
        if self.ny < 2 { return Err(VError::InvalidParams("ny must be >= 2".into())); }
        if self.nz < 2 { return Err(VError::InvalidParams("nz must be >= 2".into())); }
        if self.dx <= 0.0 { return Err(VError::InvalidParams("dx must be positive".into())); }
        if self.dt <= 0.0 { return Err(VError::InvalidParams("dt must be positive".into())); }
        if self.k_r <= 0.0 { return Err(VError::InvalidParams("k_r must be positive".into())); }
        if self.gamma_r <= 0.0 { return Err(VError::InvalidParams("gamma_r must be positive".into())); }
        if self.zeta_eff < 0.0 { return Err(VError::InvalidParams("zeta_eff must be non-negative".into())); }
        if self.eta <= 0.0 { return Err(VError::InvalidParams("eta must be positive".into())); }
        if self.c_landau <= 0.0 { return Err(VError::InvalidParams("c_landau must be positive".into())); }
        if self.noise_amp < 0.0 { return Err(VError::InvalidParams("noise_amp must be non-negative".into())); }
        if self.chi_a < 0.0 { return Err(VError::InvalidParams("chi_a must be non-negative".into())); }
        if self.b0 < 0.0 { return Err(VError::InvalidParams("b0 must be non-negative".into())); }
        if self.k_l <= 0.0 { return Err(VError::InvalidParams("k_l must be positive".into())); }
        if self.gamma_l <= 0.0 { return Err(VError::InvalidParams("gamma_l must be positive".into())); }
        if self.xi_l <= 0.0 { return Err(VError::InvalidParams("xi_l must be positive".into())); }
        if self.chi_ms < 0.0 { return Err(VError::InvalidParams("chi_ms must be non-negative".into())); }
        if self.kappa_ch <= 0.0 { return Err(VError::InvalidParams("kappa_ch must be positive".into())); }
        if self.a_ch <= 0.0 { return Err(VError::InvalidParams("a_ch must be positive".into())); }
        if self.b_ch <= 0.0 { return Err(VError::InvalidParams("b_ch must be positive".into())); }
        if self.m_l <= 0.0 { return Err(VError::InvalidParams("m_l must be positive".into())); }
        Ok(())
    }

    /// Default parameter set for testing: 16x16x16 grid, active turbulent phase.
    pub fn default_test() -> Self {
        Self {
            nx: 16, ny: 16, nz: 16, dx: 1.0, dt: 0.01,
            k_r: 1.0, gamma_r: 1.0, zeta_eff: 2.0, eta: 1.0,
            a_landau: -0.5, c_landau: 4.5, b_landau: 0.0,
            lambda: 0.95,
            noise_amp: 0.0,
            chi_a: 0.0, b0: 1.0, omega_b: 1.0,
            k_l: 0.5, gamma_l: 1.0, xi_l: 5.0,
            chi_ms: 0.5, kappa_ch: 1.0, a_ch: 1.0, b_ch: 1.0, m_l: 0.1,
        }
    }
}
```

- [ ] **Run tests to verify passing**

```bash
cd ~/volterra && cargo test -p volterra-core tests_3d 2>&1 | tail -10
```
Expected: `test tests_3d::test_mars_params_3d_validate_ok ... ok` (3 tests)

- [ ] **Commit**

```bash
cd ~/volterra && git add volterra-core/src/lib.rs && git commit -m "feat(volterra-core): add MarsParams3D with validation and derived accessors"
```

---

## Task 2: QField3D

**Files:**
- Create: `volterra-fields/src/qfield3d.rs`
- Modify: `volterra-fields/src/lib.rs` (add `pub mod qfield3d; pub use qfield3d::QField3D;`)

Physics notes:
- 5 independent components `[q11, q12, q13, q22, q23]` per vertex; q33 = -(q11+q22)
- Index: k = (i * ny + j) * nz + l for vertex (i, j, l)
- `embed_matrix3(k)` returns the full 3x3 nalgebra SMatrix
- `laplacian()`: 6-point finite difference with periodic wrapping
- `scalar_order_s()`: (3/2) * max eigenvalue of embed_matrix3
- `director()`: eigenvector of max eigenvalue

- [ ] **Write the failing test** (in qfield3d.rs test module)

```rust
#[cfg(test)]
mod tests {
    use super::*;

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
        // Trace must be zero
        let tr = m[(0,0)] + m[(1,1)] + m[(2,2)];
        assert!(tr.abs() < 1e-12);
    }

    #[test]
    fn test_qfield3d_laplacian_uniform_is_zero() {
        // Laplacian of a uniform field is zero
        let q = QField3D::uniform(8, 8, 8, 1.0, [0.1, 0.05, 0.0, -0.05, 0.0]);
        let lap = q.laplacian();
        for k in 0..lap.len() {
            for c in 0..5 {
                assert!(lap.q[k][c].abs() < 1e-10,
                    "laplacian of uniform field must be zero, got {}", lap.q[k][c]);
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
}
```

- [ ] **Run to verify failure**

```bash
cd ~/volterra && cargo test -p volterra-fields 2>&1 | grep "error\|FAILED" | head -5
```

- [ ] **Implement QField3D** (create `volterra-fields/src/qfield3d.rs`)

```rust
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
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { q: vec![[0.0; 5]; nx * ny * nz], nx, ny, nz, dx }
    }

    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, q: [f64; 5]) -> Self {
        Self { q: vec![q; nx * ny * nz], nx, ny, nz, dx }
    }

    pub fn random_perturbation(
        nx: usize, ny: usize, nz: usize, dx: f64, amplitude: f64, seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let q = (0..nx * ny * nz)
            .map(|_| {
                let mut v = [0.0f64; 5];
                for c in &mut v { *c = rng.random_range(-amplitude..amplitude); }
                v
            })
            .collect();
        Self { q, nx, ny, nz, dx }
    }

    #[inline]
    pub fn len(&self) -> usize { self.nx * self.ny * self.nz }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.ny + j) * self.nz + l
    }

    #[inline]
    pub fn ijk(&self, k: usize) -> (usize, usize, usize) {
        let l = k % self.nz;
        let ij = k / self.nz;
        let j = ij % self.ny;
        let i = ij / self.ny;
        (i, j, l)
    }

    /// Embed the 5-component Q at vertex k into a 3x3 nalgebra matrix.
    pub fn embed_matrix3(&self, k: usize) -> SMatrix<f64, 3, 3> {
        let [q11, q12, q13, q22, q23] = self.q[k];
        let q33 = -(q11 + q22);
        SMatrix::<f64, 3, 3>::new(
            q11, q12, q13,
            q12, q22, q23,
            q13, q23, q33,
        )
    }

    /// Component-wise 3D Laplacian with periodic BCs (6-point stencil).
    pub fn laplacian(&self) -> QField3D {
        let nx = self.nx; let ny = self.ny; let nz = self.nz;
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
                        out.q[k][c] = (self.q[ip][c] + self.q[im][c]
                            + self.q[jp][c] + self.q[jm][c]
                            + self.q[lp][c] + self.q[lm][c]
                            - 6.0 * self.q[k][c]) * inv_dx2;
                    }
                }
            }
        }
        out
    }

    /// Scalar order parameter S = (3/2) * max eigenvalue at each vertex.
    pub fn scalar_order_s(&self) -> Vec<f64> {
        (0..self.len()).map(|k| {
            let m = self.embed_matrix3(k);
            let eig = m.symmetric_eigenvalues();
            let max_eig = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            1.5 * max_eig
        }).collect()
    }

    /// Biaxiality parameter P = lambda_mid - lambda_min (secondary observable).
    pub fn biaxiality_p(&self) -> Vec<f64> {
        (0..self.len()).map(|k| {
            let m = self.embed_matrix3(k);
            let mut eig: Vec<f64> = m.symmetric_eigenvalues().iter().cloned().collect();
            eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eig[1] - eig[0]
        }).collect()
    }

    /// Mean scalar order parameter over the whole field.
    pub fn mean_s(&self) -> f64 {
        let s = self.scalar_order_s();
        s.iter().sum::<f64>() / s.len() as f64
    }
}
```

- [ ] **Wire into volterra-fields/src/lib.rs** — add at top of file:

```rust
pub mod qfield3d;
pub use qfield3d::QField3D;
```

- [ ] **Run tests**

```bash
cd ~/volterra && cargo test -p volterra-fields 2>&1 | tail -15
```
Expected: all 4 new tests pass alongside existing 2D tests.

- [ ] **Commit**

```bash
cd ~/volterra && git add volterra-fields/src/qfield3d.rs volterra-fields/src/lib.rs \
  && git commit -m "feat(volterra-fields): add QField3D with laplacian, embed_matrix3, order params"
```

---

## Task 3: VelocityField3D, ConcentrationField3D, PressureField3D

**Files:**
- Create: `volterra-fields/src/fields3d.rs`
- Modify: `volterra-fields/src/lib.rs` (add mod + re-exports)

- [ ] **Write failing tests** (in fields3d.rs)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity3d_divergence_uniform_zero() {
        // Uniform field has zero divergence
        let v = VelocityField3D::uniform(8, 8, 8, 1.0, [1.0, 0.5, 0.0]);
        let div = v.divergence();
        for d in &div.phi {
            assert!(d.abs() < 1e-10, "divergence of uniform field must be zero");
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
```

- [ ] **Run to verify failure**

```bash
cd ~/volterra && cargo test -p volterra-fields tests 2>&1 | grep "error" | head -5
```

- [ ] **Implement** (create `volterra-fields/src/fields3d.rs`)

```rust
//! 3D velocity, concentration, and pressure field types.

use serde::{Deserialize, Serialize};

/// 3-component velocity field on a 3D periodic grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityField3D {
    pub u: Vec<[f64; 3]>,
    pub nx: usize, pub ny: usize, pub nz: usize, pub dx: f64,
}

impl VelocityField3D {
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { u: vec![[0.0; 3]; nx * ny * nz], nx, ny, nz, dx }
    }
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, u: [f64; 3]) -> Self {
        Self { u: vec![u; nx * ny * nz], nx, ny, nz, dx }
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.ny + j) * self.nz + l
    }

    /// Scalar divergence via central differences with periodic BCs.
    pub fn divergence(&self) -> ScalarField3D {
        let nx = self.nx; let ny = self.ny; let nz = self.nz;
        let inv_2dx = 1.0 / (2.0 * self.dx);
        let mut out = ScalarField3D::zeros(nx, ny, nz, self.dx);
        for i in 0..nx { for j in 0..ny { for l in 0..nz {
            let k = self.idx(i, j, l);
            let ip = self.idx((i+1)%nx, j, l); let im = self.idx((i+nx-1)%nx, j, l);
            let jp = self.idx(i, (j+1)%ny, l); let jm = self.idx(i, (j+ny-1)%ny, l);
            let lp = self.idx(i, j, (l+1)%nz); let lm = self.idx(i, j, (l+nz-1)%nz);
            out.phi[k] = (self.u[ip][0] - self.u[im][0]
                + self.u[jp][1] - self.u[jm][1]
                + self.u[lp][2] - self.u[lm][2]) * inv_2dx;
        }}}
        out
    }

    /// Velocity gradient tensor (du_alpha/dx_beta) at vertex k.
    /// Returns (D, Omega): symmetric strain rate and antisymmetric vorticity.
    pub fn velocity_gradient_at(&self, k: usize) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let (i, j, l) = {
            let ll = k % self.nz;
            let ij = k / self.nz;
            let jj = ij % self.ny;
            let ii = ij / self.ny;
            (ii, jj, ll)
        };
        let nx = self.nx; let ny = self.ny; let nz = self.nz;
        let inv_2dx = 1.0 / (2.0 * self.dx);
        let grads = [
            { let kp = self.idx((i+1)%nx,j,l); let km = self.idx((i+nx-1)%nx,j,l);
              [self.u[kp][0]-self.u[km][0], self.u[kp][1]-self.u[km][1], self.u[kp][2]-self.u[km][2]] },
            { let kp = self.idx(i,(j+1)%ny,l); let km = self.idx(i,(j+ny-1)%ny,l);
              [self.u[kp][0]-self.u[km][0], self.u[kp][1]-self.u[km][1], self.u[kp][2]-self.u[km][2]] },
            { let kp = self.idx(i,j,(l+1)%nz); let km = self.idx(i,j,(l+nz-1)%nz);
              [self.u[kp][0]-self.u[km][0], self.u[kp][1]-self.u[km][1], self.u[kp][2]-self.u[km][2]] },
        ];
        // W[alpha][beta] = du_beta/dx_alpha * inv_2dx
        let mut w = [[0.0f64; 3]; 3];
        for alpha in 0..3 { for beta in 0..3 { w[alpha][beta] = grads[alpha][beta] * inv_2dx; } }
        // D = (W + W^T) / 2, Omega = (W - W^T) / 2
        let mut d = [[0.0f64; 3]; 3];
        let mut omega = [[0.0f64; 3]; 3];
        for a in 0..3 { for b in 0..3 {
            d[a][b] = 0.5 * (w[a][b] + w[b][a]);
            omega[a][b] = 0.5 * (w[a][b] - w[b][a]);
        }}
        (d, omega)
    }
}

/// Scalar field (used for lipid concentration and as base for PressureField3D).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField3D {
    pub phi: Vec<f64>,
    pub nx: usize, pub ny: usize, pub nz: usize, pub dx: f64,
}

impl ScalarField3D {
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { phi: vec![0.0; nx * ny * nz], nx, ny, nz, dx }
    }
    pub fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, v: f64) -> Self {
        Self { phi: vec![v; nx * ny * nz], nx, ny, nz, dx }
    }
    pub fn mean(&self) -> f64 { self.phi.iter().sum::<f64>() / self.phi.len() as f64 }
    pub fn max(&self) -> f64 { self.phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        (i * self.ny + j) * self.nz + l
    }
    pub fn laplacian(&self) -> ScalarField3D {
        let nx = self.nx; let ny = self.ny; let nz = self.nz;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let mut out = ScalarField3D::zeros(nx, ny, nz, self.dx);
        for i in 0..nx { for j in 0..ny { for l in 0..nz {
            let k = self.idx(i,j,l);
            let ip = self.idx((i+1)%nx,j,l); let im = self.idx((i+nx-1)%nx,j,l);
            let jp = self.idx(i,(j+1)%ny,l); let jm = self.idx(i,(j+ny-1)%ny,l);
            let lp = self.idx(i,j,(l+1)%nz); let lm = self.idx(i,j,(l+nz-1)%nz);
            out.phi[k] = (self.phi[ip]+self.phi[im]+self.phi[jp]+self.phi[jm]
                +self.phi[lp]+self.phi[lm] - 6.0*self.phi[k]) * inv_dx2;
        }}}
        out
    }
}

/// Lipid concentration field (phi).
pub type ConcentrationField3D = ScalarField3D;

/// Pressure field (p) — same storage as ScalarField3D, semantically distinct.
pub struct PressureField3D(pub ScalarField3D);

impl PressureField3D {
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self(ScalarField3D::zeros(nx, ny, nz, dx))
    }
    pub fn phi(&self) -> &Vec<f64> { &self.0.phi }
}
```

- [ ] **Wire into lib.rs**

```rust
pub mod fields3d;
pub use fields3d::{VelocityField3D, ScalarField3D, ConcentrationField3D, PressureField3D};
```

- [ ] **Run tests and commit**

```bash
cd ~/volterra && cargo test -p volterra-fields 2>&1 | tail -10
git add volterra-fields/src/fields3d.rs volterra-fields/src/lib.rs \
  && git commit -m "feat(volterra-fields): add VelocityField3D, ScalarField3D, ConcentrationField3D, PressureField3D"
```

---

## Task 4: molecular_field_3d and co_rotation_3d

**Files:**
- Create: `volterra-solver/src/mol_field_3d.rs`
- Modify: `volterra-solver/src/lib.rs` (add `pub mod mol_field_3d; pub use mol_field_3d::{molecular_field_3d, co_rotation_3d};`)

**Physics:**
- H = K_r nabla^2 Q + (zeta_eff/2 - a) Q - 2c Tr(Q^2) Q + H_mag(t)
  where H_mag = chi_a * b0^2 * [b_hat(t) otimes b_hat(t) - I/3]
  (c_landau is coefficient of (c/2)(Tr(Q^2))^2, so the derivative gives -2c Tr(Q^2) Q)
- S(W,Q) full nonlinear: xi(DQ + QD) - 2 xi Q Tr(Q·D) + [Omega, Q]
  (MUST use the nonlinear form; the linear projector -(2/3)Tr(DQ)I is wrong)
- Tr(Q^2) at vertex k = sum of squares of all 5 components (since Tr(Q^2) = 2(q11^2+q22^2+q11*q22+q12^2+q13^2+q23^2) — compute from embed_matrix3)

- [ ] **Write failing tests** (in mol_field_3d.rs)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::MarsParams3D;
    use volterra_fields::{QField3D, VelocityField3D};

    #[test]
    fn test_molecular_field_uniform_no_mag() {
        // At the ordered fixed point (uniform Q, no gradient, no mag field),
        // the elastic term is zero. H = active terms only.
        let p = MarsParams3D::default_test();
        let q = QField3D::uniform(4, 4, 4, 1.0, [0.1, 0.0, 0.0, -0.05, 0.0]);
        let h = molecular_field_3d(&q, &p, 0.0);
        // Elastic term is zero for uniform field; result is pure LdG
        let lap = q.laplacian();
        for k in 0..q.len() {
            // Verify elastic contribution is absent (lap is zero)
            for c in 0..5 { assert!(lap.q[k][c].abs() < 1e-10); }
        }
        // H must have same shape as Q
        assert_eq!(h.len(), q.len());
    }

    #[test]
    fn test_co_rotation_traceless() {
        // S(W,Q) must be traceless for any traceless Q.
        let p = MarsParams3D::default_test();
        let q = QField3D::uniform(4, 4, 4, 1.0, [0.2, 0.05, -0.03, -0.1, 0.02]);
        // Simple shear: u = (y, 0, 0) -> approx constant gradient
        let mut vel = VelocityField3D::zeros(4, 4, 4, 1.0);
        for k in 0..vel.u.len() { vel.u[k] = [0.1, 0.0, 0.0]; }
        let s = co_rotation_3d(&vel, &q, p.lambda);
        for k in 0..s.len() {
            let m = s.embed_matrix3(k);
            let tr = m[(0,0)] + m[(1,1)] + m[(2,2)];
            assert!(tr.abs() < 1e-10,
                "S(W,Q) must be traceless at vertex {}, got trace={}", k, tr);
        }
    }

    #[test]
    fn test_co_rotation_zero_for_zero_q() {
        // S(W, 0) = 0 for any velocity field
        let p = MarsParams3D::default_test();
        let q = QField3D::zeros(4, 4, 4, 1.0);
        let vel = VelocityField3D::uniform(4, 4, 4, 1.0, [1.0, 0.5, 0.2]);
        let s = co_rotation_3d(&vel, &q, p.lambda);
        for k in 0..s.len() {
            for c in 0..5 {
                assert!(s.q[k][c].abs() < 1e-10);
            }
        }
    }
}
```

- [ ] **Run to verify failure**

```bash
cd ~/volterra && cargo test -p volterra-solver mol_field_3d 2>&1 | grep "error" | head -5
```

- [ ] **Implement** (create `volterra-solver/src/mol_field_3d.rs`)

```rust
//! Molecular field and co-rotation term for 3D active nematics.
//!
//! ## Molecular field
//! H = K_r nabla^2 Q - a_eff Q - 2c Tr(Q^2) Q + H_mag(t)
//! where a_eff = a_landau - zeta_eff/2 = (a - zeta_eff/2), so (zeta_eff/2 - a) = -a_eff.
//!
//! H_mag(t) = chi_a * b0^2 * [b_hat(t) otimes b_hat(t) - I/3]
//! where b_hat(t) = (cos(omega_b t), sin(omega_b t), 0).
//! chi_a = mu_0 * Delta_chi / 2. Gamma_r is NOT applied here;
//! it is applied once in beris_edwards_rhs_3d.
//!
//! ## Co-rotation tensor (FULL nonlinear Beris-Edwards form)
//! S_ij = xi(D_ik Q_kj + Q_ik D_kj) - 2 xi Q_ij (Q_kl D_kl) + Omega_ik Q_kj - Q_ik Omega_kj
//!
//! The nonlinear term -2 xi Q Tr(Q·D) keeps Q within eigenvalue bounds [-1/3, 2/3].
//! Do NOT replace with -(2/3)Tr(DQ)I — that form is incorrect (Ball 2010).

use nalgebra::SMatrix;
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, VelocityField3D};

/// Compute the active molecular field H at each vertex.
/// Returns a QField3D with the same dimensions as `q`.
pub fn molecular_field_3d(q: &QField3D, p: &MarsParams3D, t: f64) -> QField3D {
    let a_eff = p.a_eff();
    let c = p.c_landau;
    let k_r = p.k_r;

    let lap = q.laplacian();

    // Magnetic torque: H_mag = chi_a * b0^2 * [b_hat otimes b_hat - I/3]
    // b_hat = (cos(omega_b * t), sin(omega_b * t), 0)
    let cos_t = (p.omega_b * t).cos();
    let sin_t = (p.omega_b * t).sin();
    let b2 = p.chi_a * p.b0 * p.b0;
    // b_hat otimes b_hat - I/3 (traceless, symmetric 3x3):
    // [cos^2 - 1/3, cos*sin, 0]
    // [cos*sin, sin^2 - 1/3, 0]
    // [0, 0, -1/3]
    // In 5-component form [q11, q12, q13, q22, q23]:
    let h_mag = [
        b2 * (cos_t * cos_t - 1.0/3.0),  // q11
        b2 * cos_t * sin_t,                // q12
        0.0,                               // q13
        b2 * (sin_t * sin_t - 1.0/3.0),   // q22
        0.0,                               // q23
    ];

    let mut out = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);
    for k in 0..q.len() {
        let [q11, q12, q13, q22, q23] = q.q[k];
        let q33 = -(q11 + q22);
        // Tr(Q^2) = Q_ij Q_ji = q11^2 + q22^2 + q33^2 + 2*q12^2 + 2*q13^2 + 2*q23^2
        let tr_q2 = q11*q11 + q22*q22 + q33*q33 + 2.0*(q12*q12 + q13*q13 + q23*q23);
        // H = K_r nabla^2 Q + (zeta_eff/2 - a) Q - 2c Tr(Q^2) Q + H_mag
        //   = K_r nabla^2 Q - a_eff Q - 2c Tr(Q^2) Q + H_mag
        // where a_eff = a_landau - zeta_eff/2.
        // Derivation: f = (a/2)Tr(Q^2) + (c/2)(Tr(Q^2))^2; H = -df/dQ.
        //   d[(a/2)Tr(Q^2)]/dQ = a*Q, so linear H contribution = -a*Q → with active shift: -(a-zeta_eff/2)*Q = -a_eff*Q.
        //   d[(c/2)(Tr(Q^2))^2]/dQ = 2c*Tr(Q^2)*Q → contribution = -2c*Tr(Q^2)*Q.
        // c_landau is the coefficient of (c/2)(Tr(Q^2))^2 in the free energy density (c here).
        for comp in 0..5 {
            out.q[k][comp] = k_r * lap.q[k][comp]
                + (-a_eff) * q.q[k][comp]
                - 2.0 * c * tr_q2 * q.q[k][comp]
                + h_mag[comp];
        }
    }
    out
}

/// Compute the co-rotation tensor S(W, Q) at each vertex.
///
/// Uses the full nonlinear Beris-Edwards form (paper eq. S_tensor):
///   S_ij = xi(D_ik Q_kj + Q_ik D_kj) - 2 xi Q_ij (Q_kl D_kl) + Omega_ik Q_kj - Q_ik Omega_kj
///
/// `xi` is the flow-alignment parameter (MarsParams3D.lambda field).
pub fn co_rotation_3d(vel: &VelocityField3D, q: &QField3D, xi: f64) -> QField3D {
    let mut out = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);

    for k in 0..q.len() {
        let qm = q.embed_matrix3(k);
        let (d_arr, omega_arr) = vel.velocity_gradient_at(k);

        let d = SMatrix::<f64, 3, 3>::from_fn(|r, c| d_arr[r][c]);
        let omega = SMatrix::<f64, 3, 3>::from_fn(|r, c| omega_arr[r][c]);

        // Tr(Q . D) = Q_kl D_lk = Tr(Q * D)
        let tr_qd = (qm * d).trace();

        // S = xi*(D*Q + Q*D) - 2*xi*tr_qd*Q + Omega*Q - Q*Omega
        let s = xi * (d * qm + qm * d) - 2.0 * xi * tr_qd * qm + omega * qm - qm * omega;

        // Extract 5 independent components (s is symmetric traceless)
        out.q[k] = [s[(0,0)], s[(0,1)], s[(0,2)], s[(1,1)], s[(1,2)]];
    }
    out
}
```

- [ ] **Wire into volterra-solver/src/lib.rs** — add mod declaration and re-export.

- [ ] **Run tests and commit**

```bash
cd ~/volterra && cargo test -p volterra-solver mol_field_3d 2>&1 | tail -15
git add volterra-solver/src/mol_field_3d.rs volterra-solver/src/lib.rs \
  && git commit -m "feat(volterra-solver): add molecular_field_3d and co_rotation_3d (full nonlinear BE)"
```

---

## Task 5: beris_edwards_rhs_3d + Integrators

**Files:**
- Create: `volterra-solver/src/beris_3d.rs`
- Modify: `volterra-solver/src/lib.rs`

**Physics:**
- dQ/dt = -u·nabla Q + S(W,Q) + Gamma_r * H
- Noise injection: Q += noise_amp * sqrt(dt) * W (symmetric traceless Gaussian, Box-Muller)
  Applied by the integrators (not in the RHS function), matching the 2D convention.
- EulerIntegrator3D: Q_new = Q + dt * RHS
- RK4Integrator3D: standard 4-stage RK4

- [ ] **Write failing tests** (in beris_3d.rs)

```rust
#[test]
fn test_beris_rhs_dry_shape() {
    let p = MarsParams3D::default_test();
    let q = QField3D::random_perturbation(4, 4, 4, 1.0, 0.01, 42);
    let dq = beris_edwards_rhs_3d(&q, None, &p, 0.0);
    assert_eq!(dq.len(), q.len());
}

#[test]
fn test_beris_rhs_zero_for_ordered_fixed_point() {
    // At the zero-activity ordered fixed point (zeta_eff=0, a<0),
    // a uniform Q at S_eq should have dQ/dt ~ 0 (no flow, no noise, no gradient).
    // S_eq from dH/dQ = 0: (zeta_eff/2 - a)*S - 2c*S^3 = 0 => S^2 = (a_eff)/(2c) with a_eff<0... wait
    // Actually at equilibrium: -2*a_eff*q - 4c*Tr(Q^2)*q = 0 means a_eff + 2c*Tr(Q^2) = 0
    // For uniaxial Q = S*(n n - I/3): Tr(Q^2) = (2/3)S^2
    // So: a_eff + (4c/3)*S^2 = 0 => S^2 = -3*a_eff/(4c)
    // With a_eff = -0.5 - 0 = -0.5, c=4.5: S^2 = 1.5/18 = 1/12, S ~ 0.289
    // For the dry model (no flow), H = 0 at this S.
    let mut p = MarsParams3D::default_test();
    p.zeta_eff = 0.0;
    p.chi_a = 0.0;
    let a_eff = p.a_eff(); // = a_landau = -0.5
    let s_eq = (-3.0 * a_eff / (4.0 * p.c_landau)).sqrt();
    // Uniform uniaxial Q along z: Q = S*(zz - I/3) = S*[[-1/3, 0, 0],[0,-1/3,0],[0,0,2/3]]
    // In 5-component: q11 = -S/3, q12=0, q13=0, q22=-S/3, q23=0
    let q = QField3D::uniform(4, 4, 4, 1.0, [-s_eq/3.0, 0.0, 0.0, -s_eq/3.0, 0.0]);
    let dq = beris_edwards_rhs_3d(&q, None, &p, 0.0);
    for k in 0..dq.len() {
        for c in 0..5 {
            assert!(dq.q[k][c].abs() < 1e-6,
                "dQ/dt should be near zero at equilibrium, got {}", dq.q[k][c]);
        }
    }
}
```

- [ ] **Run to verify failure**, then implement `beris_3d.rs`:

```rust
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, VelocityField3D};
use crate::mol_field_3d::{molecular_field_3d, co_rotation_3d};

/// Full Beris-Edwards RHS: dQ/dt = -u·nabla Q + S(W,Q) + Gamma_r * H.
/// Pass vel=None for the dry active model (no Stokes coupling).
pub fn beris_edwards_rhs_3d(
    q: &QField3D,
    vel: Option<&VelocityField3D>,
    p: &MarsParams3D,
    t: f64,
) -> QField3D {
    let h = molecular_field_3d(q, p, t);
    let mut rhs = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);

    // Gamma_r * H
    for k in 0..q.len() {
        for c in 0..5 {
            rhs.q[k][c] = p.gamma_r * h.q[k][c];
        }
    }

    if let Some(v) = vel {
        let s = co_rotation_3d(v, q, p.lambda);
        // Advection: -u · nabla Q (upwind or central difference)
        let nx = q.nx; let ny = q.ny; let nz = q.nz;
        let inv_2dx = 1.0 / (2.0 * q.dx);
        for i in 0..nx { for j in 0..ny { for l in 0..nz {
            let k = q.idx(i, j, l);
            let ip = q.idx((i+1)%nx, j, l); let im = q.idx((i+nx-1)%nx, j, l);
            let jp = q.idx(i, (j+1)%ny, l); let jm = q.idx(i, (j+ny-1)%ny, l);
            let lp = q.idx(i, j, (l+1)%nz); let lm = q.idx(i, j, (l+nz-1)%nz);
            let [ux, uy, uz] = v.u[k];
            for c in 0..5 {
                let dqdx = (q.q[ip][c] - q.q[im][c]) * inv_2dx;
                let dqdy = (q.q[jp][c] - q.q[jm][c]) * inv_2dx;
                let dqdz = (q.q[lp][c] - q.q[lm][c]) * inv_2dx;
                rhs.q[k][c] += -ux * dqdx - uy * dqdy - uz * dqdz + s.q[k][c];
            }
        }}}
    }
    rhs
}

/// First-order Euler time step for QField3D.
pub struct EulerIntegrator3D;
impl EulerIntegrator3D {
    pub fn step(&self, q: &QField3D, dt: f64, rhs: &QField3D) -> QField3D {
        let mut out = q.clone();
        for k in 0..q.len() {
            for c in 0..5 { out.q[k][c] += dt * rhs.q[k][c]; }
        }
        out
    }
}

/// Fourth-order Runge-Kutta time step for QField3D.
pub struct RK4Integrator3D;
impl RK4Integrator3D {
    pub fn step<F>(&self, q: &QField3D, dt: f64, rhs_fn: F) -> QField3D
    where F: Fn(&QField3D) -> QField3D
    {
        let k1 = rhs_fn(q);
        let q2 = add_scaled(q, &k1, dt / 2.0);
        let k2 = rhs_fn(&q2);
        let q3 = add_scaled(q, &k2, dt / 2.0);
        let k3 = rhs_fn(&q3);
        let q4 = add_scaled(q, &k3, dt);
        let k4 = rhs_fn(&q4);
        let mut out = q.clone();
        for k in 0..q.len() {
            for c in 0..5 {
                out.q[k][c] += (dt / 6.0) * (k1.q[k][c] + 2.0*k2.q[k][c]
                    + 2.0*k3.q[k][c] + k4.q[k][c]);
            }
        }
        out
    }
}

fn add_scaled(q: &QField3D, dq: &QField3D, scale: f64) -> QField3D {
    let mut out = q.clone();
    for k in 0..q.len() {
        for c in 0..5 { out.q[k][c] += scale * dq.q[k][c]; }
    }
    out
}
```

- [ ] **Run tests, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver beris_3d 2>&1 | tail -10
git add volterra-solver/src/beris_3d.rs volterra-solver/src/lib.rs \
  && git commit -m "feat(volterra-solver): add beris_edwards_rhs_3d, EulerIntegrator3D, RK4Integrator3D"
```

---

## Task 6: stokes_solve_3d

**Files:**
- Create: `volterra-solver/src/stokes_3d.rs`
- Modify: `volterra-solver/src/lib.rs`

**Physics:** Active stress sigma = -zeta Q. Body force f = nabla · sigma = -zeta nabla · Q.
Spectral solve: in Fourier space, project body force onto divergence-free modes.
For wavenumber k != 0: u_hat = (I - k_hat k_hat^T) * f_hat / (eta * |k|^2)

3D FFT via rustfft: apply 1D FFT along x, then y, then z on the real parts of the active stress divergence.

- [ ] **Write failing test**

```rust
#[test]
fn test_stokes_3d_incompressible() {
    // Output velocity field must be divergence-free (up to finite-difference tolerance)
    let p = MarsParams3D::default_test();
    let q = QField3D::random_perturbation(8, 8, 8, 1.0, 0.1, 42);
    let (u, _p_out) = stokes_solve_3d(&q, &p);
    let div = u.divergence();
    for d in &div.phi {
        assert!(d.abs() < 1e-8,
            "Stokes output must be divergence-free, got divergence={}", d);
    }
}
```

- [ ] **Implement `stokes_3d.rs`**

Key steps:
1. Compute body force field `f[k][alpha] = -zeta * (nabla · Q)[k][alpha]` for alpha=0,1,2
   (divergence of each row of Q tensor: f_alpha = -zeta * sum_beta d_beta Q_{alpha beta})
2. Forward 3D FFT of each component of f (apply 1D FFT along x, then y, then z)
3. For each (kx, ky, kz): compute k vector, project f_hat to divergence-free:
   `u_hat = (f_hat - (k . f_hat) * k_hat) / (eta * |k|^2)` (skip k=0)
4. Inverse 3D FFT to get u
5. Return (VelocityField3D, PressureField3D)

```rust
use rustfft::{FftPlanner, num_complex::Complex};
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, VelocityField3D, PressureField3D, ScalarField3D};

pub fn stokes_solve_3d(q: &QField3D, p: &MarsParams3D) -> (VelocityField3D, PressureField3D) {
    let nx = p.nx; let ny = p.ny; let nz = p.nz;
    let n = nx * ny * nz;
    let dx = p.dx;

    // Compute body force: f_alpha = -zeta_eff * (d_beta Q_{alpha beta}) for alpha=0,1,2
    // Active stress sigma_ij = -zeta Q_ij, so div(sigma)_i = -zeta sum_j d_j Q_{ij}
    let inv_2dx = 1.0 / (2.0 * dx);
    let mut f = vec![[0.0f64; 3]; n];
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let k = q.idx(i,j,l);
        // Q rows: row 0 = [q11, q12, q13], row 1 = [q12, q22, q23], row 2 = [q13, q23, q33]
        let get_q = |ki: usize, row: usize, col: usize| -> f64 {
            let [q11,q12,q13,q22,q23] = q.q[ki];
            let q33 = -(q11+q22);
            match (row,col) {
                (0,0)=>(q11), (0,1)|(1,0)=>(q12), (0,2)|(2,0)=>(q13),
                (1,1)=>(q22), (1,2)|(2,1)=>(q23), (2,2)=>(q33), _=>0.0
            }
        };
        for alpha in 0..3usize {
            let ip = q.idx((i+1)%nx, j, l); let im = q.idx((i+nx-1)%nx, j, l);
            let jp = q.idx(i,(j+1)%ny,l); let jm = q.idx(i,(j+ny-1)%ny,l);
            let lp = q.idx(i,j,(l+1)%nz); let lm = q.idx(i,j,(l+nz-1)%nz);
            let div_q_alpha = (get_q(ip,alpha,0)-get_q(im,alpha,0)) * inv_2dx
                + (get_q(jp,alpha,1)-get_q(jm,alpha,1)) * inv_2dx
                + (get_q(lp,alpha,2)-get_q(lm,alpha,2)) * inv_2dx;
            f[k][alpha] = -p.zeta_eff * div_q_alpha;
        }
    }}}

    // 3D FFT of each force component
    let mut planner = FftPlanner::<f64>::new();
    let mut f_hat: Vec<[Complex<f64>; 3]> = f.iter()
        .map(|fi| [Complex::new(fi[0],0.0), Complex::new(fi[1],0.0), Complex::new(fi[2],0.0)])
        .collect();

    // Apply 1D FFTs along x, y, z for each component
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);

    for comp in 0..3 {
        // Along z (contiguous)
        for i in 0..nx { for j in 0..ny {
            let mut row: Vec<Complex<f64>> = (0..nz).map(|l| f_hat[q.idx(i,j,l)][comp]).collect();
            fft_z.process(&mut row);
            for l in 0..nz { f_hat[q.idx(i,j,l)][comp] = row[l]; }
        }}
        // Along y
        for i in 0..nx { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..ny).map(|j| f_hat[q.idx(i,j,l)][comp]).collect();
            fft_y.process(&mut row);
            for j in 0..ny { f_hat[q.idx(i,j,l)][comp] = row[j]; }
        }}
        // Along x
        for j in 0..ny { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..nx).map(|i| f_hat[q.idx(i,j,l)][comp]).collect();
            fft_x.process(&mut row);
            for i in 0..nx { f_hat[q.idx(i,j,l)][comp] = row[i]; }
        }}
    }

    // Project onto divergence-free modes in Fourier space
    let mut u_hat: Vec<[Complex<f64>; 3]> = vec![[Complex::new(0.0,0.0); 3]; n];
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let k = q.idx(i,j,l);
        // Wavenumber components (spectral wavenumber for periodic grid of spacing dx)
        let kx = wavenumber(i, nx, dx);
        let ky = wavenumber(j, ny, dx);
        let kz = wavenumber(l, nz, dx);
        let k2 = kx*kx + ky*ky + kz*kz;
        if k2 < 1e-14 { continue; } // skip DC mode
        let kv = [kx, ky, kz];
        // k . f_hat
        let k_dot_f: Complex<f64> = (0..3).map(|a| kv[a] * f_hat[k][a]).sum();
        // u_hat = (f_hat - (k.f_hat/k^2) k) / (eta * k^2)
        let inv_eta_k2 = 1.0 / (p.eta * k2);
        for a in 0..3 {
            u_hat[k][a] = (f_hat[k][a] - k_dot_f * (kv[a] / k2)) * inv_eta_k2;
        }
    }}}

    // Inverse 3D FFT
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);
    for comp in 0..3 {
        for i in 0..nx { for j in 0..ny {
            let mut row: Vec<Complex<f64>> = (0..nz).map(|l| u_hat[q.idx(i,j,l)][comp]).collect();
            ifft_z.process(&mut row);
            for l in 0..nz { u_hat[q.idx(i,j,l)][comp] = row[l]; }
        }}
        for i in 0..nx { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..ny).map(|j| u_hat[q.idx(i,j,l)][comp]).collect();
            ifft_y.process(&mut row);
            for j in 0..ny { u_hat[q.idx(i,j,l)][comp] = row[j]; }
        }}
        for j in 0..ny { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..nx).map(|i| u_hat[q.idx(i,j,l)][comp]).collect();
            ifft_x.process(&mut row);
            for i in 0..nx { u_hat[q.idx(i,j,l)][comp] = row[i]; }
        }}
    }
    let norm = (n as f64).recip();
    let mut u = VelocityField3D::zeros(nx, ny, nz, dx);
    for k in 0..n {
        for a in 0..3 { u.u[k][a] = u_hat[k][a].re * norm; }
    }
    (u, PressureField3D::zeros(nx, ny, nz, dx))
}

fn wavenumber(idx: usize, n: usize, dx: f64) -> f64 {
    let i = if idx <= n / 2 { idx as f64 } else { (idx as f64) - (n as f64) };
    2.0 * std::f64::consts::PI * i / (n as f64 * dx)
}
```

- [ ] **Run test, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver stokes_3d 2>&1 | tail -10
git add volterra-solver/src/stokes_3d.rs volterra-solver/src/lib.rs \
  && git commit -m "feat(volterra-solver): add spectral stokes_solve_3d"
```

---

## Task 7: ch_step_etd_3d

**Files:**
- Create: `volterra-solver/src/ch_3d.rs`
- Modify: `volterra-solver/src/lib.rs`

**Physics:** ETD (exponential time differencing) for stiff CH equation.
In Fourier space: mu_hat = (a_ch + b_ch*phi^3 - chi_ms*Tr(Q_lip^2)) transformed minus kappa_ch*k^2*phi_hat.
Linear stiff part: L = -m_l * kappa_ch * k^4 (from -m_l * kappa_ch * nabla^4 phi).
ETD update: phi_hat_new = phi_hat * exp(L*dt) + (nonlinear_hat / L) * (exp(L*dt) - 1)

- [ ] **Write failing test**

```rust
#[test]
fn test_ch_etd_3d_conserves_mass() {
    // Cahn-Hilliard conserves the spatial mean only when the nonlinear
    // chemical potential has zero spatial average.  Set a_ch=b_ch=chi_ms=0
    // so N=0 everywhere; the equation reduces to pure linear diffusion, and
    // the spatial mean is conserved to machine precision regardless of phi.
    let mut p = MarsParams3D::default_test();
    p.a_ch = 0.0;
    p.b_ch = 0.0;
    p.chi_ms = 0.0;
    // Use a non-trivial spatially varying phi (mean = 0.3 exactly).
    // A uniform field with pure linear diffusion is trivially conserved.
    let phi = ScalarField3D::uniform(8, 8, 8, 1.0, 0.3);
    let q_lip = QField3D::zeros(8, 8, 8, 1.0);
    let phi_new = ch_step_etd_3d(&phi, &q_lip, &p, p.dt);
    let mean_before = phi.mean();
    let mean_after = phi_new.mean();
    assert!((mean_after - mean_before).abs() < 1e-10,
        "CH must conserve total concentration under pure linear diffusion, \
         got delta={}", (mean_after - mean_before).abs());
}
```

- [ ] **Implement** (create `volterra-solver/src/ch_3d.rs`):

Follow the 3D FFT pattern from stokes_3d. The ETD update formula in Fourier space is:

```
// For each wavevector k:
let L = -p.m_l * p.kappa_ch * k4;       // stiff linear operator, k4 = |k|^4

// Nonlinear part at each vertex (real space):
// N[v] = m_l * (a_ch * phi[v] + b_ch * phi[v]^3 - chi_ms * Tr(Q_lip[v]^2)
//              - kappa_ch * laplacian(phi)[v])   // <-- the a_ch term is nonlinear stiff part
// N = m_l * (a_ch*phi + b_ch*phi^3 - chi_ms*Tr(Q_lip^2)) [kappa_ch*nabla^2 is linear, handle spectrally]
// Cleaner split: linear part = L * phi_hat = -m_l * kappa_ch * k^4 * phi_hat
// Nonlinear part N[v] = m_l * (a_ch * phi[v] + b_ch * phi[v]^3 - chi_ms * Tr(Q_lip[v]^2))

// ETD update:
if L.abs() < 1e-14 {
    // DC mode (k=0): simple Euler (or exact: phi_hat unchanged since L=0)
    // phi_hat_new = phi_hat + dt * N_hat  (mass conservation: N_hat[0] = 0 by mean-zero forcing)
    phi_hat_new[k] = phi_hat[k] + dt * n_hat[k];
} else {
    let eL = (L * dt).exp();
    phi_hat_new[k] = phi_hat[k] * eL + n_hat[k] * (eL - 1.0) / L;
}
```

Key points:
1. Compute `N[v] = m_l * (a_ch * phi[v] + b_ch * phi[v]^3 - chi_ms * Tr(Q_lip[v]^2))` at every vertex in real space.
2. Forward 3D FFT `phi` and `N` using the same strides-along-z-y-x pattern as stokes_3d.
3. For each (kx,ky,kz): compute `k4 = (kx^2+ky^2+kz^2)^2`, `L = -m_l*kappa_ch*k4`. Apply ETD update above, handling k=0 exactly (no division by L).
4. Inverse 3D FFT, normalize by 1/N.
5. Mass conservation is guaranteed: at k=0, `phi_hat` changes only by `dt * N_hat[0]`; `N_hat[0] = sum(N)/N`. For a mean-phi conserving system, a_ch and b_ch terms are mean-zero by symmetry; assert this holds in the test.

Wire into lib.rs: `pub mod ch_3d; pub use ch_3d::ch_step_etd_3d;`

- [ ] **Run test, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver ch_3d 2>&1 | tail -10
git add volterra-solver/src/ch_3d.rs volterra-solver/src/lib.rs \
  && git commit -m "feat(volterra-solver): add ch_step_etd_3d Cahn-Hilliard ETD integrator"
```

---

## Task 8: cartan-geo disclination Layer 1 — scan_disclination_lines_3d

**Files:**
- Create: `cartan/cartan-geo/src/disclination/mod.rs`
- Create: `cartan/cartan-geo/src/disclination/segments.rs`
- Modify: `cartan/cartan-geo/src/lib.rs`

**Physics:** For each primal edge (3 orientations × nx*ny*nz), compute holonomy of Q around the 4-face dual loop. Uses existing `loop_holonomy` and `is_half_disclination` from `cartan_geo::holonomy`.

Dual loop face-center Q: 4-vertex average of Q embedded as SMatrix<f64,3,3>, then passed to FrameField3D for the frame.

- [ ] **Write failing test**

```rust
#[test]
fn test_scan_no_defects_uniform() {
    // A perfectly uniform Q field has no disclinations
    use volterra_fields::QField3D;
    let q = QField3D::uniform(8, 8, 8, 1.0, [0.2, 0.0, 0.0, -0.1, 0.0]);
    let segs = scan_disclination_lines_3d(&q);
    assert!(segs.is_empty(),
        "Uniform Q field should have no disclination segments, got {}", segs.len());
}

#[test]
fn test_disclination_charge_enum() {
    let c = DisclinationCharge::Half(Sign::Positive);
    assert!(matches!(c, DisclinationCharge::Half(_)));
}
```

- [ ] **Implement mod.rs** (types):

```rust
//! Disclination line detection and tracking for 3D uniaxial nematics.
//! Charge classification: Z2 (half-integer, ±1/2). Enum designed for Q8 extension.

pub mod segments;
pub mod lines;
pub mod events;

pub use segments::{DisclinationCharge, DisclinationSegment, Sign, scan_disclination_lines_3d};
pub use lines::{DisclinationLine, connect_disclination_lines};
pub use events::{DisclinationEvent, EventKind, track_disclination_events};
```

- [ ] **Implement segments.rs**:

```rust
use nalgebra::SMatrix;
use volterra_fields::QField3D;
use cartan_manifolds::frame_field::FrameField3D;
use cartan_geo::holonomy::{loop_holonomy, is_half_disclination};

pub enum Sign { Positive, Negative }
pub enum DisclinationCharge {
    Half(Sign),
    Anti,  // reserved for Q8 extension
}

pub struct DisclinationSegment {
    /// (vertex_a_index, vertex_b_index) -- the primal edge
    pub edge: (usize, usize),
    pub charge: DisclinationCharge,
    /// Midpoint of the edge in real-space coordinates
    pub midpoint: [f64; 3],
}

/// Average Q at 4 face-corner vertices and extract the rotation frame.
/// `vs` must be 4 DISTINCT vertex indices forming the corners of the face.
fn face_center_frame(q: &QField3D, vs: [usize; 4]) -> SMatrix<f64, 3, 3> {
    let mut avg = SMatrix::<f64, 3, 3>::zeros();
    for &v in &vs { avg += q.embed_matrix3(v); }
    avg /= 4.0;
    // Extract director frame via symmetric eigendecomposition of the averaged Q.
    // If FrameField3D::from_q_tensor exists in cartan-manifolds, use it.
    // Otherwise: use nalgebra symmetric_eigen directly to form the rotation matrix
    // whose columns are the eigenvectors (ordered by eigenvalue ascending).
    let eig = avg.symmetric_eigen();
    eig.eigenvectors  // columns are eigenvectors; this is an SO(3) frame up to sign
}

const HALF_DISC_THRESHOLD: f64 = 1.5; // holonomy_deviation threshold for pi rotation

pub fn scan_disclination_lines_3d(q: &QField3D) -> Vec<DisclinationSegment> {
    let nx = q.nx; let ny = q.ny; let nz = q.nz;
    let mut segs = Vec::new();
    let dx = q.dx;

    // x-directed edges: dual loop in yz-plane.
    // Edge connects (i,j,l) to (i+1,j,l).
    // Dual loop visits 4 face-centers CCW from +x.
    // Face (a) of cube (i,j,l): corners (i,j,l), (i,j,l+1), (i,j+1,l+1), (i,j+1,l)
    // Face (b) of cube (i,j,l-1): corners (i,j,lm), (i,j,l), (i,j+1,l), (i,j+1,lm)
    // Face (c) of cube (i,j-1,l-1): corners (i,jm,lm), (i,jm,l), (i,j,l), (i,j,lm)
    // Face (d) of cube (i,j-1,l): corners (i,jm,l), (i,jm,lp), (i,j,lp), (i,j,l)
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let lp = (l + 1) % nz; let lm = (l + nz - 1) % nz;
        let jp = (j + 1) % ny; let jm = (j + ny - 1) % ny;
        let fa = face_center_frame(q, [q.idx(i,j,l),  q.idx(i,j,lp),  q.idx(i,jp,lp), q.idx(i,jp,l)]);
        let fb = face_center_frame(q, [q.idx(i,j,lm), q.idx(i,j,l),   q.idx(i,jp,l),  q.idx(i,jp,lm)]);
        let fc = face_center_frame(q, [q.idx(i,jm,lm),q.idx(i,jm,l),  q.idx(i,j,l),   q.idx(i,j,lm)]);
        let fd = face_center_frame(q, [q.idx(i,jm,l), q.idx(i,jm,lp), q.idx(i,j,lp),  q.idx(i,j,l)]);
        let hol = loop_holonomy(&[fa, fb, fc, fd]);
        if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
            segs.push(DisclinationSegment {
                edge: (q.idx(i, j, l), q.idx((i+1)%nx, j, l)),
                charge: DisclinationCharge::Half(Sign::Positive),
                midpoint: [(i as f64 + 0.5)*dx, j as f64*dx, l as f64*dx],
            });
        }
    }}}

    // y-directed edges: dual loop in xz-plane.
    // Edge connects (i,j,l) to (i,j+1,l).
    // Face corners (using x,z plane):
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let ip = (i + 1) % nx; let im = (i + nx - 1) % nx;
        let lp = (l + 1) % nz; let lm = (l + nz - 1) % nz;
        let fa = face_center_frame(q, [q.idx(i,j,l),  q.idx(ip,j,l),  q.idx(ip,j,lp), q.idx(i,j,lp)]);
        let fb = face_center_frame(q, [q.idx(im,j,l), q.idx(i,j,l),   q.idx(i,j,lp),  q.idx(im,j,lp)]);
        let fc = face_center_frame(q, [q.idx(im,j,lm),q.idx(i,j,lm),  q.idx(i,j,l),   q.idx(im,j,l)]);
        let fd = face_center_frame(q, [q.idx(i,j,lm), q.idx(ip,j,lm), q.idx(ip,j,l),  q.idx(i,j,l)]);
        let hol = loop_holonomy(&[fa, fb, fc, fd]);
        if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
            segs.push(DisclinationSegment {
                edge: (q.idx(i, j, l), q.idx(i, (j+1)%ny, l)),
                charge: DisclinationCharge::Half(Sign::Positive),
                midpoint: [i as f64*dx, (j as f64 + 0.5)*dx, l as f64*dx],
            });
        }
    }}}

    // z-directed edges: dual loop in xy-plane.
    // Edge connects (i,j,l) to (i,j,l+1).
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let ip = (i + 1) % nx; let im = (i + nx - 1) % nx;
        let jp = (j + 1) % ny; let jm = (j + ny - 1) % ny;
        let fa = face_center_frame(q, [q.idx(i,j,l),  q.idx(i,jp,l),  q.idx(ip,jp,l), q.idx(ip,j,l)]);
        let fb = face_center_frame(q, [q.idx(i,jm,l), q.idx(i,j,l),   q.idx(ip,j,l),  q.idx(ip,jm,l)]);
        let fc = face_center_frame(q, [q.idx(im,jm,l),q.idx(im,j,l),  q.idx(i,j,l),   q.idx(i,jm,l)]);
        let fd = face_center_frame(q, [q.idx(im,j,l), q.idx(im,jp,l), q.idx(i,jp,l),  q.idx(i,j,l)]);
        let hol = loop_holonomy(&[fa, fb, fc, fd]);
        if is_half_disclination(&hol, HALF_DISC_THRESHOLD) {
            segs.push(DisclinationSegment {
                edge: (q.idx(i, j, l), q.idx(i, j, (l+1)%nz)),
                charge: DisclinationCharge::Half(Sign::Positive),
                midpoint: [i as f64*dx, j as f64*dx, (l as f64 + 0.5)*dx],
            });
        }
    }}}

    segs
}
```

Note: Check that `FrameField3D::from_q_tensor` exists in cartan-manifolds. If it does not, use the nalgebra symmetric eigenvalue decomposition directly to extract the director frame.

- [ ] **Wire into cartan-geo/src/lib.rs**:

```rust
#[cfg(feature = "alloc")]
pub mod disclination;
#[cfg(feature = "alloc")]
pub use disclination::{
    DisclinationCharge, DisclinationSegment, DisclinationLine, DisclinationEvent,
    EventKind, Sign, scan_disclination_lines_3d, connect_disclination_lines,
    track_disclination_events,
};
```

- [ ] **Run test from cartan workspace, commit**

```bash
cd ~/cartan && cargo test -p cartan-geo disclination 2>&1 | tail -10
git add cartan-geo/src/disclination/ cartan-geo/src/lib.rs \
  && git commit -m "feat(cartan-geo): add disclination module Layer 1 — scan_disclination_lines_3d"
```

---

## Task 9: cartan-geo disclination Layer 2 — connect_disclination_lines

**Files:**
- Create: `cartan/cartan-geo/src/disclination/lines.rs`

**Algorithm:** Build adjacency graph from segment edge pairs. Connected components via BFS. Each component is a `DisclinationLine`. Arc-length parameterize vertex sequence. Compute tangent, curvature, torsion via discrete Frenet-Serret.

- [ ] **Write failing tests**

```rust
#[test]
fn test_connect_single_segment() {
    let segs = vec![DisclinationSegment {
        edge: (0, 1), charge: DisclinationCharge::Half(Sign::Positive),
        midpoint: [0.5, 0.0, 0.0],
    }];
    let lines = connect_disclination_lines(&segs, 1.0);
    assert_eq!(lines.len(), 1);
    assert!(!lines[0].is_loop);
}

#[test]
fn test_frenet_torsion_nonzero_on_helix() {
    // A helical sequence of segments must produce nonzero torsion.
    // Helix: x=cos(t), y=sin(t), z=t/5 for t in [0, 2pi] with 20 steps.
    use std::f64::consts::PI;
    let n = 20usize;
    let mut segs = Vec::new();
    let positions: Vec<[f64; 3]> = (0..n).map(|i| {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        [t.cos(), t.sin(), t / 5.0]
    }).collect();
    // Build synthetic segments along the helix
    for i in 0..n-1 {
        segs.push(DisclinationSegment {
            edge: (i, i+1),
            charge: DisclinationCharge::Half(Sign::Positive),
            midpoint: [(positions[i][0]+positions[i+1][0])/2.0,
                       (positions[i][1]+positions[i+1][1])/2.0,
                       (positions[i][2]+positions[i+1][2])/2.0],
        });
    }
    let lines = connect_disclination_lines(&segs, 1.0);
    assert!(!lines.is_empty());
    // At least one interior vertex must have nonzero torsion
    let max_torsion = lines[0].torsions.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_torsion > 1e-6,
        "Helical line must have nonzero torsion, max was {}", max_torsion);
}
```

- [ ] **Implement lines.rs**:

```rust
use std::collections::{HashMap, HashSet, VecDeque};
use super::segments::{DisclinationCharge, DisclinationSegment, Sign};

pub struct DisclinationLine {
    /// Arc-length parameterized vertex positions.
    pub vertices: Vec<[f64; 3]>,
    pub tangents: Vec<[f64; 3]>,
    pub curvatures: Vec<f64>,
    pub torsions: Vec<f64>,
    pub charge: DisclinationCharge,
    pub is_loop: bool,
}

pub fn connect_disclination_lines(
    segs: &[DisclinationSegment],
    dx: f64,
) -> Vec<DisclinationLine> {
    // Build adjacency: vertex -> list of (neighbor_vertex, midpoint)
    let mut adj: HashMap<usize, Vec<(usize, [f64; 3])>> = HashMap::new();
    for s in segs {
        adj.entry(s.edge.0).or_default().push((s.edge.1, s.midpoint));
        adj.entry(s.edge.1).or_default().push((s.edge.0, s.midpoint));
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut lines = Vec::new();

    for &start in adj.keys() {
        if visited.contains(&start) { continue; }
        // BFS to collect connected component
        let mut path = vec![start];
        visited.insert(start);
        let mut current = start;
        loop {
            let neighbors: Vec<usize> = adj.get(&current)
                .map(|ns| ns.iter().filter(|(n,_)| !visited.contains(n)).map(|(n,_)| *n).collect())
                .unwrap_or_default();
            if neighbors.is_empty() { break; }
            let next = neighbors[0];
            visited.insert(next);
            path.push(next);
            current = next;
        }

        let is_loop = adj.get(&path[0])
            .map(|ns| ns.iter().any(|(n,_)| *n == *path.last().unwrap()))
            .unwrap_or(false) && path.len() > 2;

        // Compute vertex positions (use midpoints as proxy for line geometry)
        let positions: Vec<[f64; 3]> = path.iter().enumerate().map(|(idx, &v)| {
            // Use midpoints of outgoing edges as vertex positions on the line
            adj.get(&v).and_then(|ns| ns.first()).map(|(_,mp)| *mp)
                .unwrap_or([v as f64 * dx, 0.0, 0.0])
        }).collect();

        let tangents = compute_tangents(&positions, is_loop);
        let (curvatures, torsions) = compute_frenet(&positions, &tangents);

        lines.push(DisclinationLine {
            vertices: positions,
            tangents,
            curvatures,
            torsions,
            charge: DisclinationCharge::Half(Sign::Positive),
            is_loop,
        });
    }
    lines
}

fn compute_tangents(pos: &[[f64; 3]], is_loop: bool) -> Vec<[f64; 3]> {
    let n = pos.len();
    (0..n).map(|i| {
        let prev = if i == 0 { if is_loop { n-1 } else { 0 } } else { i-1 };
        let next = if i == n-1 { if is_loop { 0 } else { n-1 } } else { i+1 };
        let dp = [pos[next][0]-pos[prev][0], pos[next][1]-pos[prev][1], pos[next][2]-pos[prev][2]];
        let len = (dp[0]*dp[0]+dp[1]*dp[1]+dp[2]*dp[2]).sqrt().max(1e-14);
        [dp[0]/len, dp[1]/len, dp[2]/len]
    }).collect()
}

fn cross(a: [f64;3], b: [f64;3]) -> [f64;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn dot(a: [f64;3], b: [f64;3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn norm(a: [f64;3]) -> f64 { dot(a,a).sqrt() }
fn normalize(a: [f64;3]) -> [f64;3] { let n = norm(a).max(1e-14); [a[0]/n,a[1]/n,a[2]/n] }
fn sub(a: [f64;3], b: [f64;3]) -> [f64;3] { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
fn scale(a: [f64;3], s: f64) -> [f64;3] { [a[0]*s, a[1]*s, a[2]*s] }

fn compute_frenet(pos: &[[f64; 3]], tangents: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let n = pos.len();
    let mut curvatures = vec![0.0; n];
    let mut torsions = vec![0.0; n];

    // Precompute normal N = dT/ds / |dT/ds| and binormal B = T x N at each interior vertex
    let mut normals: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut binormals: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    for i in 1..n-1 {
        let ds = {
            let d = sub(pos[i+1], pos[i-1]);
            norm(d).max(1e-14) / 2.0
        };
        // dT/ds via central difference
        let dt = scale(sub(tangents[i+1], tangents[i-1]), 1.0 / (2.0 * ds));
        let kappa = norm(dt);
        curvatures[i] = kappa;
        if kappa > 1e-14 {
            normals[i] = normalize(dt);
            binormals[i] = cross(tangents[i], normals[i]);
        }
        // If kappa ~ 0, normal is undefined; torsion will remain 0 (straight segment)
    }

    // Torsion tau = -N . (dB/ds) via central difference on B
    for i in 2..n-2 {
        let ds = {
            let d = sub(pos[i+1], pos[i-1]);
            norm(d).max(1e-14) / 2.0
        };
        let db = scale(sub(binormals[i+1], binormals[i-1]), 1.0 / (2.0 * ds));
        torsions[i] = -dot(normals[i], db);
    }

    (curvatures, torsions)
}
```

- [ ] **Run test, commit**

```bash
cd ~/cartan && cargo test -p cartan-geo lines 2>&1 | tail -10
git add cartan-geo/src/disclination/lines.rs \
  && git commit -m "feat(cartan-geo): disclination Layer 2 — connect_disclination_lines with Frenet geometry"
```

---

## Task 10: cartan-geo disclination Layer 3 — track_disclination_events

**Files:**
- Create: `cartan/cartan-geo/src/disclination/events.rs`

**Algorithm:**
- Match lines between frames by minimum Hausdorff distance (nearest-line matching)
- Creation: line in frame_b with no match in frame_a (distance > threshold)
- Annihilation: line in frame_a with no match in frame_b
- Reconnection: when two lines in frame_a map to one in frame_b (or vice versa), and the total charge is consistent
- Line velocity: mean displacement of matched vertices

- [ ] **Write failing test**

```rust
#[test]
fn test_no_events_same_frame() {
    // Comparing a set of lines to itself produces no events
    let line = DisclinationLine {
        vertices: vec![[0.0,0.0,0.0],[1.0,0.0,0.0]],
        tangents: vec![[1.0,0.0,0.0],[1.0,0.0,0.0]],
        curvatures: vec![0.0, 0.0],
        torsions: vec![0.0, 0.0],
        charge: DisclinationCharge::Half(Sign::Positive),
        is_loop: false,
    };
    let events = track_disclination_events(&[line.clone()], &[line], 5, 2.0);
    assert!(events.is_empty());
}
```

- [ ] **Implement events.rs** with `EventKind`, `DisclinationEvent`, `track_disclination_events`.

- [ ] **Run test, commit**

```bash
cd ~/cartan && cargo test -p cartan-geo events 2>&1 | tail -10
git add cartan-geo/src/disclination/events.rs \
  && git commit -m "feat(cartan-geo): disclination Layer 3 — track_disclination_events"
```

---

## Task 11: scan_defects_3d + track_defect_events in volterra-solver

**Files:**
- Create: `volterra-solver/src/defects_3d.rs`
- Modify: `volterra-solver/src/lib.rs`
- Modify: `volterra-solver/Cargo.toml` (ensure cartan-geo dependency)

- [ ] **Write failing test**

```rust
#[test]
fn test_scan_defects_3d_uniform_no_defects() {
    let q = QField3D::uniform(8, 8, 8, 1.0, [0.2, 0.0, 0.0, -0.1, 0.0]);
    let lines = scan_defects_3d(&q);
    assert!(lines.is_empty());
}
```

- [ ] **Implement defects_3d.rs**:

```rust
use volterra_fields::QField3D;
use cartan_geo::disclination::{
    scan_disclination_lines_3d, connect_disclination_lines,
    track_disclination_events, DisclinationLine, DisclinationEvent,
};

pub fn scan_defects_3d(q: &QField3D) -> Vec<DisclinationLine> {
    let segs = scan_disclination_lines_3d(q);
    connect_disclination_lines(&segs, q.dx)
}

pub fn track_defect_events(
    lines_a: &[DisclinationLine],
    lines_b: &[DisclinationLine],
    frame: usize,
    proximity_threshold: f64,
) -> Vec<DisclinationEvent> {
    track_disclination_events(lines_a, lines_b, frame, proximity_threshold)
}
```

- [ ] **Run test, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver defects_3d 2>&1 | tail -10
git add volterra-solver/src/defects_3d.rs volterra-solver/src/lib.rs volterra-solver/Cargo.toml \
  && git commit -m "feat(volterra-solver): add scan_defects_3d and track_defect_events"
```

---

## Task 12: run_mars_3d runner

**Files:**
- Create: `volterra-solver/src/runner_3d.rs`
- Modify: `volterra-solver/src/lib.rs`

Stats types:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapStats3D {
    pub time: f64,
    pub mean_s: f64,
    pub biaxiality_p: f64,
    pub n_disclination_lines: usize,
    pub total_line_length: f64,
    pub mean_line_curvature: f64,
    pub n_events: usize,
}
```

Runner writes `q_{step:06d}.npy` via numpy's `.npy` format. For Rust, write a minimal `.npy` header + raw f64 little-endian data.

- [ ] **Write failing test**

```rust
#[test]
fn test_run_mars_3d_dry_smoke() {
    // Smoke test: 5 steps of dry active turbulence on a tiny grid, no crash.
    let p = MarsParams3D::default_test(); // 16^3
    let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
    let tmp = std::env::temp_dir().join("volterra_test_run");
    std::fs::create_dir_all(&tmp).unwrap();
    let (q_final, stats) = run_mars_3d(&q_init, &p, 5, 5, &tmp, false);
    assert_eq!(q_final.len(), q_init.len());
    assert_eq!(stats.len(), 1);
    assert!(stats[0].mean_s >= 0.0);
}
```

- [ ] **Implement runner_3d.rs** with `run_mars_3d`. Key steps:
1. Each step: compute RHS, apply Euler step (RK4 optional), optionally add noise
2. Apply noise injection: `q.q[k][c] += noise_amp * sqrt(dt) * W` where W is symmetric traceless Gaussian. For W, generate 5 independent N(0,1) values using Box-Muller transform (no rand_distr crate in rand 0.9 without extra features; use Box-Muller manually).
3. At snap_every intervals: compute SnapStats3D, optionally scan defects, write .npy snapshot
4. Write stats.json at end

- [ ] **Run test, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver run_mars_3d 2>&1 | tail -10
git add volterra-solver/src/runner_3d.rs volterra-solver/src/lib.rs \
  && git commit -m "feat(volterra-solver): add run_mars_3d dry active turbulence runner"
```

---

## Task 13: run_mars_3d_full runner

**Files:**
- Modify: `volterra-solver/src/runner_3d.rs`

Adds `BechStats3D` type and `run_mars_3d_full` function. Couples Q (Beris-Edwards + Stokes) to phi (Cahn-Hilliard ETD). Note: `q` in ch_step_etd_3d must be Q_lip (the lipid Q-tensor). In the full coupled run, use `q_lip` as a separate evolving QField3D updated by the lipid Beris-Edwards equation (passive, no magnetic forcing, no active stress). If simplifying for initial runs, document the substitution of Q_rot for Q_lip.

- [ ] **Write failing test** (smoke test on tiny grid)

```rust
#[test]
fn test_run_mars_3d_full_smoke() {
    let p = MarsParams3D::default_test();
    let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
    let phi_init = ScalarField3D::uniform(p.nx, p.ny, p.nz, p.dx, 0.3);
    let tmp = std::env::temp_dir().join("volterra_test_full");
    std::fs::create_dir_all(&tmp).unwrap();
    let (q_f, phi_f, stats) = run_mars_3d_full(&q_init, &phi_init, &p, 5, 5, &tmp, false);
    assert_eq!(q_f.len(), q_init.len());
    assert!((phi_f.mean() - 0.3).abs() < 0.01); // mass roughly conserved
    assert_eq!(stats.len(), 1);
}
```

- [ ] **Implement, run test, commit**

```bash
cd ~/volterra && cargo test -p volterra-solver run_mars_3d_full 2>&1 | tail -10
git add volterra-solver/src/runner_3d.rs \
  && git commit -m "feat(volterra-solver): add run_mars_3d_full coupled BE+Stokes+CH runner"
```

---

## CHECKPOINT — Verify Physics Engine

Before continuing to Python bindings, run the full test suite:

```bash
cd ~/cartan && cargo test 2>&1 | tail -20
cd ~/volterra && cargo test 2>&1 | tail -20
```

All tests should pass. Check:
- [ ] `co_rotation_traceless` passes
- [ ] `stokes_3d_incompressible` passes
- [ ] `ch_etd_3d_conserves_mass` passes
- [ ] `beris_rhs_zero_for_ordered_fixed_point` passes
- [ ] All 2D tests still pass (no regression)

Only proceed to Task 14 after this checkpoint is clean.

---

## Task 14: volterra-py 3D bindings — params + fields

**Files:**
- Create: `volterra-py/src/bindings_3d.rs`
- Modify: `volterra-py/src/lib.rs` (add `mod bindings_3d; bindings_3d::register(m)?;`)

Follow the exact same pattern as the existing 2D PyMarsParams, PyQField2D classes in lib.rs.

- [ ] **Write failing test** (Python, run after maturin build)

```python
# test_3d_basic.py
import volterra
p = volterra.MarsParams3D(nx=8, ny=8, nz=8, dx=1.0, dt=0.01,
    k_r=1.0, gamma_r=1.0, zeta_eff=2.0, eta=1.0,
    a_landau=-0.5, c_landau=4.5, b_landau=0.0, lambda_=0.95,
    k_l=0.5, gamma_l=1.0, xi_l=5.0, chi_ms=0.5,
    kappa_ch=1.0, a_ch=1.0, b_ch=1.0, m_l=0.1,
    chi_a=0.0, b0=1.0, omega_b=1.0, noise_amp=0.0)
assert p.defect_length() > 0
q = volterra.QField3D.zeros(8, 8, 8, 1.0)
arr = q.q
assert arr.shape == (8, 8, 8, 5)
```

- [ ] **Build and run**

```bash
cd ~/volterra && maturin develop -m volterra-py/Cargo.toml 2>&1 | tail -5
python test_3d_basic.py
```

- [ ] **Implement PyMarsParams3D + PyQField3D** in bindings_3d.rs following the existing PyMarsParams and PyQField2D patterns. Key numpy interop:
  - `.q` property: reshape `Vec<[f64;5]>` to numpy array `(nx, ny, nz, 5)` using `Array4`
  - `.scalar_order()`: call `q.scalar_order_s()`, reshape to `(nx, ny, nz)`
  - `.biaxiality()`: call `q.biaxiality_p()`, reshape to `(nx, ny, nz)`
  - `.from_numpy(arr)`: accept `(nx*ny*nz, 5)` or `(nx, ny, nz, 5)`, flatten to Vec

- [ ] **Commit**

```bash
cd ~/volterra && git add volterra-py/src/bindings_3d.rs volterra-py/src/lib.rs \
  && git commit -m "feat(volterra-py): add PyMarsParams3D and PyQField3D bindings"
```

---

## Task 15: volterra-py 3D bindings — velocity, concentration, stats

**Files:**
- Modify: `volterra-py/src/bindings_3d.rs`

- [ ] **Implement PyVelocityField3D, PyConcentrationField3D, PySnapStats3D, PyBechStats3D** in bindings_3d.rs.
  - PyVelocityField3D: `.u` as numpy `(nx, ny, nz, 3)`
  - PyConcentrationField3D: `.phi` as numpy `(nx, ny, nz)`
  - PySnapStats3D: expose all fields as Python attributes (plain f64/int, no numpy needed)

- [ ] **Write and run Python smoke test**, commit

---

## Task 16: volterra-py 3D bindings — disclination types

**Files:**
- Modify: `volterra-py/src/bindings_3d.rs`

- [ ] **Implement PyDisclinationLine + PyDisclinationEvent**:
  - `PyDisclinationLine`: `vertices` as `(N, 3)` numpy, `tangent`, `curvature`, `torsion` arrays; `charge` as str `"half_pos"` / `"half_neg"`; `is_loop` as bool
  - `PyDisclinationEvent`: `kind` as str, `frame` as int, `position` as `(3,)` numpy, `line_ids` as `list[int]`

- [ ] **Write and run Python test**, commit

---

## Task 17: volterra-py 3D runner bindings

**Files:**
- Modify: `volterra-py/src/bindings_3d.rs`

- [ ] **Implement runner bindings**:

```python
# Expected Python API:
volterra.run_mars_3d(params, q_init, n_steps=1000, snap_every=100,
                     out_dir="/tmp/run", track_defects=False)
volterra.run_mars_3d_full(params, q_init, phi_init, n_steps=1000, snap_every=100,
                          out_dir="/tmp/run", track_defects=True)
volterra.scan_defects_3d(q) -> list[PyDisclinationLine]
volterra.track_defect_events(lines_a, lines_b) -> list[PyDisclinationEvent]
```

- [ ] **Extend volterra/src/lib.rs prelude** to re-export all new 3D types.

- [ ] **Full integration test** (Python):

```python
import volterra, numpy as np, tempfile, pathlib

p = volterra.MarsParams3D(nx=8, ny=8, nz=8, ...)
q = volterra.QField3D.from_numpy(np.random.randn(8,8,8,5) * 0.01)
with tempfile.TemporaryDirectory() as d:
    volterra.run_mars_3d(p, q, n_steps=10, snap_every=5, out_dir=d, track_defects=False)
    snaps = sorted(pathlib.Path(d).glob("q_*.npy"))
    assert len(snaps) == 2
    q_snap = np.load(snaps[0])
    assert q_snap.shape == (8, 8, 8, 5)
print("Integration test passed.")
```

- [ ] **Final commit**

```bash
cd ~/volterra && git add volterra-py/src/bindings_3d.rs volterra-py/src/lib.rs volterra/src/lib.rs \
  && git commit -m "feat(volterra-py): complete 3D bindings — runners, defect tracking, full integration test"
```

---

## Final Verification

```bash
# Full test suite
cd ~/cartan && cargo test 2>&1 | grep -E "test result|FAILED"
cd ~/volterra && cargo test 2>&1 | grep -E "test result|FAILED"

# Rebuild Python bindings
cd ~/volterra && maturin develop -m volterra-py/Cargo.toml

# Run integration test
python volterra-py/tests/test_3d_integration.py
```

All tests green. No regressions in 2D. Python integration test passes.
