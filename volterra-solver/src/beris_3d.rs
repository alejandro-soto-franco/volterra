//! Beris-Edwards RHS and time integrators for 3D active nematics.
//!
//! ## Governing equation
//! ```text
//! dQ/dt = -u · nabla Q + S(W, Q) + Gamma_r * H
//! ```
//!
//! where:
//! - `-u · nabla Q` is the material advection (central differences, periodic BCs)
//! - `S(W, Q)` is the co-rotation / strain-coupling tensor (from mol_field_3d)
//! - `H` is the molecular field (from mol_field_3d)
//! - `Gamma_r` is the rotational viscosity (from ActiveNematicParams3D)
//!
//! Noise injection (`Q += noise_amp * sqrt(dt) * W`) is applied by the
//! integrators, not in the RHS, matching the 2D convention.

use rayon::prelude::*;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::{QField3D, VelocityField3D};
use crate::mol_field_3d::{molecular_field_3d, molecular_field_3d_par, co_rotation_3d};

/// Full Beris-Edwards RHS: `dQ/dt = -u · nabla Q + S(W, Q) + Gamma_r * H`.
///
/// Pass `vel = None` for the dry active model (no Stokes coupling).
/// Advection uses central differences with periodic boundary conditions.
/// The co-rotation term `S(W, Q)` is only added when `vel` is `Some`.
///
/// `t` is the current time, used for the rotating magnetic field in `H`.
pub fn beris_edwards_rhs_3d(
    q: &QField3D,
    vel: Option<&VelocityField3D>,
    p: &ActiveNematicParams3D,
    t: f64,
) -> QField3D {
    let h = molecular_field_3d(q, p, t);
    let mut rhs = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);

    // Gamma_r * H contribution
    for k in 0..q.len() {
        for c in 0..5 {
            rhs.q[k][c] = p.gamma_r * h.q[k][c];
        }
    }

    if let Some(v) = vel {
        // Co-rotation / strain tensor S(W, Q) at every vertex
        let s = co_rotation_3d(v, q, p.lambda);

        // Advection: -u · nabla Q  (central differences, periodic BCs)
        let nx = q.nx;
        let ny = q.ny;
        let nz = q.nz;
        let inv_2dx = 1.0 / (2.0 * q.dx);

        for i in 0..nx {
            for j in 0..ny {
                for l in 0..nz {
                    let k = q.idx(i, j, l);
                    let ip = q.idx((i + 1) % nx, j, l);
                    let im = q.idx((i + nx - 1) % nx, j, l);
                    let jp = q.idx(i, (j + 1) % ny, l);
                    let jm = q.idx(i, (j + ny - 1) % ny, l);
                    let lp = q.idx(i, j, (l + 1) % nz);
                    let lm = q.idx(i, j, (l + nz - 1) % nz);
                    let [ux, uy, uz] = v.u[k];
                    for c in 0..5 {
                        let dqdx = (q.q[ip][c] - q.q[im][c]) * inv_2dx;
                        let dqdy = (q.q[jp][c] - q.q[jm][c]) * inv_2dx;
                        let dqdz = (q.q[lp][c] - q.q[lm][c]) * inv_2dx;
                        // S(W,Q) is purely vertex-local (no spatial stencil); s.q[k][c]
                        // is indexed by the same k as the advection stencil above.
                        rhs.q[k][c] += -ux * dqdx - uy * dqdy - uz * dqdz + s.q[k][c];
                    }
                }
            }
        }
    }
    rhs
}

/// First-order Euler time step for QField3D.
///
/// `Q_new = Q + dt * RHS`
///
/// Noise injection is NOT performed here; apply separately if needed.
pub struct EulerIntegrator3D;

impl EulerIntegrator3D {
    /// Advance `q` by one Euler step.
    ///
    /// Returns the updated field `Q + dt * rhs`.
    pub fn step(&self, q: &QField3D, dt: f64, rhs: &QField3D) -> QField3D {
        let mut out = q.clone();
        for k in 0..q.len() {
            for c in 0..5 {
                out.q[k][c] += dt * rhs.q[k][c];
            }
        }
        out
    }
}

/// Fourth-order Runge-Kutta time step for QField3D.
///
/// Standard 4-stage RK4:
/// ```text
/// k1 = RHS(Q)
/// k2 = RHS(Q + dt/2 * k1)
/// k3 = RHS(Q + dt/2 * k2)
/// k4 = RHS(Q + dt   * k3)
/// Q_new = Q + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
/// ```
///
/// **Frozen-flow approximation:** when the RHS closure captures a fixed
/// `VelocityField3D` (e.g. from a Stokes solve at the start of the time step),
/// the velocity is held constant across all four RK4 stages. This is an
/// operator-splitting approximation; the Stokes field should be updated between
/// outer time steps, not within a single RK4 call.
///
/// Noise injection is NOT performed here; apply separately if needed.
pub struct RK4Integrator3D;

impl RK4Integrator3D {
    /// Advance `q` by one RK4 step using the provided RHS closure.
    ///
    /// The closure `rhs_fn` takes a `&QField3D` and returns a `QField3D`
    /// containing `dQ/dt` at every vertex.
    pub fn step<F>(&self, q: &QField3D, dt: f64, rhs_fn: F) -> QField3D
    where
        F: Fn(&QField3D) -> QField3D,
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
                out.q[k][c] += (dt / 6.0)
                    * (k1.q[k][c] + 2.0 * k2.q[k][c] + 2.0 * k3.q[k][c] + k4.q[k][c]);
            }
        }
        out
    }
}

/// Add a scaled increment to a Q-field: returns `q + scale * dq`.
fn add_scaled(q: &QField3D, dq: &QField3D, scale: f64) -> QField3D {
    let mut out = q.clone();
    for k in 0..q.len() {
        for c in 0..5 {
            out.q[k][c] += scale * dq.q[k][c];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel versions (rayon)
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel Beris-Edwards RHS for the **dry** active nematic (no flow).
///
/// Uses the fused molecular field (`molecular_field_3d_par`) which computes
/// the Laplacian stencil and bulk LdG terms in a single parallel pass.
/// For the wet case (with flow), use [`beris_edwards_rhs_3d`] which handles
/// advection and co-rotation.
pub fn beris_edwards_rhs_3d_par_dry(
    q: &QField3D,
    p: &ActiveNematicParams3D,
    t: f64,
) -> QField3D {
    let h = molecular_field_3d_par(q, p, t);
    let gamma_r = p.gamma_r;
    let n = q.len();

    let out_data: Vec<[f64; 5]> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut r = [0.0_f64; 5];
            for c in 0..5 {
                r[c] = gamma_r * h.q[k][c];
            }
            r
        })
        .collect();

    QField3D {
        q: out_data,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        dx: q.dx,
    }
}

/// Parallel Euler step: Q_new = Q + dt * RHS.
pub fn euler_step_par(q: &QField3D, dt: f64, rhs: &QField3D) -> QField3D {
    let n = q.len();
    let out_data: Vec<[f64; 5]> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut r = [0.0_f64; 5];
            for c in 0..5 {
                r[c] = q.q[k][c] + dt * rhs.q[k][c];
            }
            r
        })
        .collect();

    QField3D {
        q: out_data,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        dx: q.dx,
    }
}

/// Parallel add_scaled: returns q + scale * dq.
fn add_scaled_par(q: &QField3D, dq: &QField3D, scale: f64) -> QField3D {
    let n = q.len();
    let out_data: Vec<[f64; 5]> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut r = [0.0_f64; 5];
            for c in 0..5 {
                r[c] = q.q[k][c] + scale * dq.q[k][c];
            }
            r
        })
        .collect();

    QField3D {
        q: out_data,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        dx: q.dx,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::ActiveNematicParams3D;
    use volterra_fields::QField3D;

    /// Verify that the RHS output has the same number of vertices as the input.
    #[test]
    fn test_beris_rhs_dry_shape() {
        let p = ActiveNematicParams3D::default_test();
        let q = QField3D::random_perturbation(4, 4, 4, 1.0, 0.01, 42);
        let dq = beris_edwards_rhs_3d(&q, None, &p, 0.0);
        assert_eq!(dq.len(), q.len());
    }

    /// At the dry (no-flow, zero-activity) ordered fixed point the RHS should
    /// vanish to numerical precision.
    ///
    /// Equilibrium condition for uniaxial Q along z:
    ///   Q = S * (zz - I/3)  =>  [q11, q12, q13, q22, q23] = [-S/3, 0, 0, -S/3, 0]
    ///
    /// The equilibrium order parameter satisfies H = 0:
    ///   -a_eff * Q - 2c * Tr(Q^2) * Q = 0
    ///   => a_eff + 2c * Tr(Q^2) = 0
    ///
    /// For uniaxial Q: Tr(Q^2) = (2/3) S^2
    ///   => S^2 = -3 a_eff / (4c)
    ///
    /// With a_eff = -0.5, c = 4.5:  S^2 = 1.5/18 = 1/12, S ~ 0.2887
    #[test]
    fn test_beris_rhs_zero_for_ordered_fixed_point() {
        let mut p = ActiveNematicParams3D::default_test();
        p.zeta_eff = 0.0;
        p.chi_a = 0.0;
        let a_eff = p.a_eff(); // = a_landau = -0.5 when zeta_eff = 0
        let s_eq = (-3.0 * a_eff / (4.0 * p.c_landau)).sqrt();
        // Uniform uniaxial Q along z: Q = S*(zz - I/3)
        // 5-component: [q11, q12, q13, q22, q23] = [-S/3, 0, 0, -S/3, 0]
        let q = QField3D::uniform(4, 4, 4, 1.0, [-s_eq / 3.0, 0.0, 0.0, -s_eq / 3.0, 0.0]);
        let dq = beris_edwards_rhs_3d(&q, None, &p, 0.0);
        for k in 0..dq.len() {
            for c in 0..5 {
                assert!(
                    dq.q[k][c].abs() < 1e-6,
                    "dQ/dt should be near zero at equilibrium, got {} at vertex {} component {}",
                    dq.q[k][c],
                    k,
                    c
                );
            }
        }
    }

    /// Verify that the parallel molecular field matches the sequential version.
    #[test]
    fn test_parallel_molecular_field_matches_sequential() {
        let p = ActiveNematicParams3D::default_test();
        let q = QField3D::random_perturbation(8, 8, 8, 1.0, 0.1, 42);

        let h_seq = molecular_field_3d(&q, &p, 0.5);
        let h_par = molecular_field_3d_par(&q, &p, 0.5);

        let max_diff: f64 = h_seq.q.iter().zip(&h_par.q)
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 1e-10,
            "parallel should match sequential: max_diff = {max_diff}"
        );
    }

    /// Verify that the parallel dry RHS matches the sequential version.
    #[test]
    fn test_parallel_rhs_matches_sequential() {
        let p = ActiveNematicParams3D::default_test();
        let q = QField3D::random_perturbation(8, 8, 8, 1.0, 0.1, 42);

        let rhs_seq = beris_edwards_rhs_3d(&q, None, &p, 0.3);
        let rhs_par = beris_edwards_rhs_3d_par_dry(&q, &p, 0.3);

        let max_diff: f64 = rhs_seq.q.iter().zip(&rhs_par.q)
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 1e-10,
            "parallel RHS should match sequential: max_diff = {max_diff}"
        );
    }

    /// Verify the parallel Euler step.
    #[test]
    fn test_parallel_euler_step() {
        let p = ActiveNematicParams3D::default_test();
        let q = QField3D::random_perturbation(8, 8, 8, 1.0, 0.1, 42);
        let rhs = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);
        let dt = 0.001;

        let q_seq = EulerIntegrator3D.step(&q, dt, &rhs);
        let q_par = euler_step_par(&q, dt, &rhs);

        let max_diff: f64 = q_seq.q.iter().zip(&q_par.q)
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 1e-14,
            "parallel Euler should match sequential: max_diff = {max_diff}"
        );
    }
}
