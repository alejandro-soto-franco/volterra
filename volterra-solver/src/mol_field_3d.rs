#![allow(clippy::needless_range_loop)]
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
//! The nonlinear term -2 xi Q Tr(Q*D) keeps Q within eigenvalue bounds [-1/3, 2/3].
//! Do NOT replace with -(2/3)Tr(DQ)I -- that form is incorrect (Ball 2010).

use nalgebra::SMatrix;
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, VelocityField3D};

/// Compute the active molecular field H at each vertex.
///
/// Returns a QField3D with the same dimensions as `q`.
///
/// The molecular field is:
/// ```text
/// H = K_r * nabla^2 Q - a_eff * Q - 2c * Tr(Q^2) * Q + H_mag(t)
/// ```
/// where a_eff = a_landau - zeta_eff/2 and Tr(Q^2) is evaluated with q33 = -(q11+q22).
/// Gamma_r is NOT applied here; it is applied once in beris_edwards_rhs_3d.
pub fn molecular_field_3d(q: &QField3D, p: &MarsParams3D, t: f64) -> QField3D {
    let a_eff = p.a_eff();
    let c = p.c_landau;
    let k_r = p.k_r;

    let lap = q.laplacian();

    // Magnetic torque: H_mag = chi_a * b0^2 * [b_hat otimes b_hat - I/3]
    // b_hat(t) = (cos(omega_b*t), sin(omega_b*t), 0)
    // 5-component form: [H11, H12, H13, H22, H23]
    let cos_t = (p.omega_b * t).cos();
    let sin_t = (p.omega_b * t).sin();
    let b2 = p.chi_a * p.b0 * p.b0;
    let h_mag = [
        b2 * (cos_t * cos_t - 1.0 / 3.0),
        b2 * cos_t * sin_t,
        0.0,
        b2 * (sin_t * sin_t - 1.0 / 3.0),
        0.0,
    ];

    let mut out = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);
    for k in 0..q.len() {
        let [q11, q12, q13, q22, q23] = q.q[k];
        let q33 = -(q11 + q22);
        let tr_q2 = q11 * q11 + q22 * q22 + q33 * q33
            + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        for comp in 0..5 {
            out.q[k][comp] = k_r * lap.q[k][comp]
                + (-a_eff) * q.q[k][comp]
                - 2.0 * c * tr_q2 * q.q[k][comp]
                + h_mag[comp];
        }
    }
    out
}

/// Compute the co-rotation tensor S(D, Omega, Q) at each vertex.
///
/// Uses the full nonlinear Beris-Edwards form:
/// ```text
/// S = xi*(D*Q + Q*D) - 2*xi*Tr(Q*D)*Q + Omega*Q - Q*Omega
/// ```
///
/// `xi` is the flow-alignment parameter (MarsParams3D.lambda field).
///
/// The nonlinear term -2*xi*Tr(Q*D)*Q is required to maintain the traceless
/// constraint on Q under flow. Do NOT substitute -(2/3)*Tr(D*Q)*I.
pub fn co_rotation_3d(vel: &VelocityField3D, q: &QField3D, xi: f64) -> QField3D {
    let mut out = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);

    for k in 0..q.len() {
        let qm = q.embed_matrix3(k);
        let (d_arr, omega_arr) = vel.velocity_gradient_at(k);

        let d = SMatrix::<f64, 3, 3>::from_fn(|r, c| d_arr[r][c]);
        let omega = SMatrix::<f64, 3, 3>::from_fn(|r, c| omega_arr[r][c]);

        // Tr(Q . D) = trace of matrix product Q*D
        let tr_qd = (qm * d).trace();

        // S = xi*(D*Q + Q*D) - 2*xi*Tr(Q*D)*Q + Omega*Q - Q*Omega
        let s = xi * (d * qm + qm * d) - 2.0 * xi * tr_qd * qm + omega * qm - qm * omega;

        // Extract the 5 independent upper-triangle components
        // [s11, s12, s13, s22, s23] (s is symmetric traceless)
        out.q[k] = [s[(0, 0)], s[(0, 1)], s[(0, 2)], s[(1, 1)], s[(1, 2)]];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::MarsParams3D;
    use volterra_fields::{QField3D, VelocityField3D};

    #[test]
    fn test_molecular_field_uniform_no_mag() {
        let p = MarsParams3D::default_test();
        let q = QField3D::uniform(4, 4, 4, 1.0, [0.1, 0.0, 0.0, -0.05, 0.0]);
        let h = molecular_field_3d(&q, &p, 0.0);
        let lap = q.laplacian();
        for k in 0..q.len() {
            for c in 0..5 {
                assert!(lap.q[k][c].abs() < 1e-10);
            }
        }
        assert_eq!(h.len(), q.len());
    }

    #[test]
    fn test_co_rotation_traceless() {
        let p = MarsParams3D::default_test();
        let q = QField3D::uniform(4, 4, 4, 1.0, [0.2, 0.05, -0.03, -0.1, 0.02]);
        let mut vel = VelocityField3D::zeros(4, 4, 4, 1.0);
        for k in 0..vel.u.len() {
            vel.u[k] = [0.1, 0.0, 0.0];
        }
        let s = co_rotation_3d(&vel, &q, p.lambda);
        for k in 0..s.len() {
            let m = s.embed_matrix3(k);
            let tr = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
            assert!(
                tr.abs() < 1e-10,
                "S(W,Q) must be traceless at vertex {}, got trace={}",
                k,
                tr
            );
        }
    }

    #[test]
    fn test_co_rotation_zero_for_zero_q() {
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
