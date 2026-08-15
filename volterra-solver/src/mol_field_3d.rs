#![allow(clippy::needless_range_loop)]
//! Molecular field and co-rotation term for 3D active nematics.
//!
//! ## Molecular field
//! H = K_r nabla^2 Q - a_eff Q - 2c Tr(Q^2) Q - 3b [Q^2 - Tr(Q^2)/3 I] + H_mag(t)
//! where a_eff = a_landau - zeta_eff/2 = (a - zeta_eff/2), so (zeta_eff/2 - a) = -a_eff.
//!
//! H_mag(t) = chi_a * b0^2 * [b_hat(t) otimes b_hat(t) - I/3]
//! where b_hat(t) = (cos(omega_b t), sin(omega_b t), 0).
//! chi_a = mu_0 * Delta_chi / 2. Gamma_r is NOT applied here;
//! it is applied once in beris_edwards_rhs_3d.
//!
//! ## Cubic bulk term, derivation
//!
//! The bulk free energy density is `f = (a_eff/2) Tr(Q^2) + b Tr(Q^3) +
//! (c/2) (Tr(Q^2))^2` (matching open-Qmin's raw convention `a Tr(Q^2) + b
//! Tr(Q^3) + c (Tr(Q^2))^2` term for term: `a_eff = 2a`, `c_landau = 2c`,
//! `b_landau = b`, verified by substitution -- both conventions give the
//! identical equilibrium condition below). `H = -delta f / delta Q`, the
//! variational derivative restricted to the traceless-symmetric subspace Q
//! lives in.
//!
//! For the quadratic and quartic terms this restriction is free: the
//! ordinary matrix-calculus gradient of any function of `Q` alone (`2Q` for
//! `Tr(Q^2)`, `4 Tr(Q^2) Q` for `(Tr(Q^2))^2`) is already traceless whenever
//! `Q` is, since both are proportional to `Q` itself.
//!
//! `Tr(Q^3)` does not have this property. The unconstrained matrix-calculus
//! gradient of `Tr(Q^3)` is `3 Q^2`, and `Q^2` is generally **not**
//! traceless even when `Q` is (`Tr(Q^2) != 0`). The physical variation `dQ`
//! is constrained to the traceless-symmetric subspace, so the molecular
//! field is the projection of the unconstrained gradient onto that
//! subspace: subtract the trace part,
//! `3 Q^2 - (1/3) Tr(3 Q^2) I = 3 Q^2 - Tr(Q^2) I`. With the free energy's
//! `b Tr(Q^3)` term this gives
//!
//! ```text
//! H_cubic = -b [3 Q^2 - Tr(Q^2) I] = -3b Q^2 + b Tr(Q^2) I
//! ```
//!
//! Restricted to the 5 independent components `[q11, q12, q13, q22, q23]`
//! (`q33 = -(q11+q22)`), with `Q^2` entries `qq11 = q11^2+q12^2+q13^2`,
//! `qq12 = q11 q12 + q12 q22 + q13 q23`, `qq13 = q11 q13 + q12 q23 + q13
//! q33`, `qq22 = q12^2+q22^2+q23^2`, `qq23 = q12 q13 + q22 q23 + q23 q33`:
//!
//! ```text
//! H_cubic[q11] = -3b*qq11 + b*Tr(Q^2)
//! H_cubic[q12] = -3b*qq12
//! H_cubic[q13] = -3b*qq13
//! H_cubic[q22] = -3b*qq22 + b*Tr(Q^2)
//! H_cubic[q23] = -3b*qq23
//! ```
//!
//! (`H_cubic[q33] = -3b*qq33 + b*Tr(Q^2)`, implied, sums to zero with the
//! two stored diagonal entries: tracelessness is exact by construction, not
//! approximate.)
//!
//! Setting `b_landau = 0` removes this term identically (every line above
//! is multiplied through by `b`), so every result measured before this term
//! existed is reproduced exactly, bit for bit.
//!
//! Validated against the closed-form uniaxial equilibrium
//! (`volterra-solver/tests/test_cubic_bulk_equilibrium.rs`): for `Q =
//! S(nn - I/3)`, `Tr(Q^2) = (2/3)S^2`, `Tr(Q^3) = (2/9)S^3`, so `dF/dS = 0`
//! gives `3 a_eff + 3 b_landau S + 4 c_landau S^2 = 0`, matched against
//! open-Qmin's own equilibrium condition (`inc/qTensorFunctions.h`'s
//! `derivativeTrQ3`, evaluated directly at the uniaxial point) under the
//! `a_eff=2a, b_landau=b, c_landau=2c` mapping above: both give the
//! identical quadratic `6a + 3bM + 8cM^2 = 0` in the standard order
//! parameter `M`, confirming the two codes' bulk equilibria agree exactly
//! once their `(a,b,c)` are matched this way.
//!
//! ## Co-rotation tensor (FULL nonlinear Beris-Edwards form)
//! S_ij = xi(D_ik Q_kj + Q_ik D_kj) - 2 xi Q_ij (Q_kl D_kl) + Omega_ik Q_kj - Q_ik Omega_kj
//!
//! The nonlinear term -2 xi Q Tr(Q*D) keeps Q within eigenvalue bounds [-1/3, 2/3].
//! Do NOT replace with -(2/3)Tr(DQ)I -- that form is incorrect (Ball 2010).

use nalgebra::SMatrix;
use rayon::prelude::*;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::{QField3D, VelocityField3D};

/// Cubic bulk contribution to the molecular field, `H_cubic = -3b Q^2 + b
/// Tr(Q^2) I`, restricted to the 5 independent components. See this module's
/// header for the derivation (the traceless-projected variation of `b
/// Tr(Q^3)`). `b_landau = 0.0` returns exactly `[0.0; 5]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn cubic_bulk_term(
    q11: f64,
    q12: f64,
    q13: f64,
    q22: f64,
    q23: f64,
    q33: f64,
    tr_q2: f64,
    b_landau: f64,
) -> [f64; 5] {
    if b_landau == 0.0 {
        return [0.0; 5];
    }
    let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
    let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
    let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
    let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
    let qq23 = q12 * q13 + q22 * q23 + q23 * q33;
    [
        -3.0 * b_landau * qq11 + b_landau * tr_q2,
        -3.0 * b_landau * qq12,
        -3.0 * b_landau * qq13,
        -3.0 * b_landau * qq22 + b_landau * tr_q2,
        -3.0 * b_landau * qq23,
    ]
}

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
pub fn molecular_field_3d(q: &QField3D, p: &ActiveNematicParams3D, t: f64) -> QField3D {
    let a_eff = p.a_eff();
    let c = p.c_landau;
    let b_landau = p.b_landau;
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
        let h_cubic = cubic_bulk_term(q11, q12, q13, q22, q23, q33, tr_q2, b_landau);
        for comp in 0..5 {
            out.q[k][comp] = k_r * lap.q[k][comp]
                + (-a_eff) * q.q[k][comp]
                - 2.0 * c * tr_q2 * q.q[k][comp]
                + h_cubic[comp]
                + h_mag[comp];
        }
    }
    out
}

/// Parallel molecular field computation with fused Laplacian stencil.
///
/// Computes the active molecular field H at each vertex using a single
/// parallel pass that inlines the 6-point Laplacian stencil with the
/// bulk Landau-de Gennes terms. Avoids allocating an intermediate
/// Laplacian field.
pub fn molecular_field_3d_par(q: &QField3D, p: &ActiveNematicParams3D, t: f64) -> QField3D {
    let a_eff = p.a_eff();
    let c = p.c_landau;
    let b_landau = p.b_landau;
    let k_r = p.k_r;
    let inv_dx2 = 1.0 / (q.dx * q.dx);
    let nx = q.nx;
    let ny = q.ny;
    let nz = q.nz;
    let n = nx * ny * nz;

    // Magnetic torque (precomputed, same for all vertices).
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

    // Parallel computation: each vertex computes its own Laplacian stencil
    // and combines with bulk terms in one shot.
    let q_data = &q.q;
    let nynz = ny * nz;

    let out_data: Vec<[f64; 5]> = (0..n)
        .into_par_iter()
        .map(|k| {
            // Decode (i, j, l) from flat index k = (i*ny + j)*nz + l.
            let l = k % nz;
            let ij = k / nz;
            let j = ij % ny;
            let i = ij / ny;

            // 6-point stencil neighbours (periodic).
            let ip = ((i + 1) % nx) * nynz + j * nz + l;
            let im = ((i + nx - 1) % nx) * nynz + j * nz + l;
            let jp = i * nynz + ((j + 1) % ny) * nz + l;
            let jm = i * nynz + ((j + ny - 1) % ny) * nz + l;
            let lp = i * nynz + j * nz + (l + 1) % nz;
            let lm = i * nynz + j * nz + (l + nz - 1) % nz;

            let qk = q_data[k];
            let [q11, q12, q13, q22, q23] = qk;
            let q33 = -(q11 + q22);
            let tr_q2 = q11 * q11 + q22 * q22 + q33 * q33
                + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
            let bulk = -a_eff - 2.0 * c * tr_q2;
            let h_cubic = cubic_bulk_term(q11, q12, q13, q22, q23, q33, tr_q2, b_landau);

            let mut h = [0.0_f64; 5];
            for comp in 0..5 {
                let lap = (q_data[ip][comp]
                    + q_data[im][comp]
                    + q_data[jp][comp]
                    + q_data[jm][comp]
                    + q_data[lp][comp]
                    + q_data[lm][comp]
                    - 6.0 * qk[comp])
                    * inv_dx2;
                h[comp] = k_r * lap + bulk * qk[comp] + h_cubic[comp] + h_mag[comp];
            }
            h
        })
        .collect();

    QField3D {
        q: out_data,
        nx,
        ny,
        nz,
        dx: q.dx,
    }
}

/// `scale * H` written into an existing buffer, allocating nothing.
///
/// [`molecular_field_3d_par`] returns a fresh `QField3D`, which is a 40 MB
/// allocation per call at `N = 100`. Called once per FIRE iteration, and
/// followed by a second full pass and a second allocation to apply `gamma_r`,
/// that dominated the CPU minimiser: the pages are new mappings every time, so
/// each iteration faults them in again, and page faults serialise where the
/// arithmetic does not. Passing `scale = gamma_r` here gives the dry
/// Beris-Edwards right-hand side in one pass with no allocation at all.
///
/// `out` must already have `q`'s dimensions.
pub fn molecular_field_3d_par_into(
    q: &QField3D,
    p: &ActiveNematicParams3D,
    t: f64,
    scale: f64,
    out: &mut QField3D,
) {
    assert_eq!(out.q.len(), q.q.len(), "out must match q's grid");
    let a_eff = p.a_eff();
    let c = p.c_landau;
    let b_landau = p.b_landau;
    let k_r = p.k_r;
    let inv_dx2 = 1.0 / (q.dx * q.dx);
    let (nx, ny, nz) = (q.nx, q.ny, q.nz);

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

    let q_data = &q.q;
    let nynz = ny * nz;

    out.q
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, out_k)| {
            let l = k % nz;
            let ij = k / nz;
            let j = ij % ny;
            let i = ij / ny;

            let ip = ((i + 1) % nx) * nynz + j * nz + l;
            let im = ((i + nx - 1) % nx) * nynz + j * nz + l;
            let jp = i * nynz + ((j + 1) % ny) * nz + l;
            let jm = i * nynz + ((j + ny - 1) % ny) * nz + l;
            let lp = i * nynz + j * nz + (l + 1) % nz;
            let lm = i * nynz + j * nz + (l + nz - 1) % nz;

            let qk = q_data[k];
            let [q11, q12, q13, q22, q23] = qk;
            let q33 = -(q11 + q22);
            let tr_q2 = q11 * q11 + q22 * q22 + q33 * q33
                + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
            let bulk = -a_eff - 2.0 * c * tr_q2;
            let h_cubic = cubic_bulk_term(q11, q12, q13, q22, q23, q33, tr_q2, b_landau);

            for comp in 0..5 {
                let lap = (q_data[ip][comp]
                    + q_data[im][comp]
                    + q_data[jp][comp]
                    + q_data[jm][comp]
                    + q_data[lp][comp]
                    + q_data[lm][comp]
                    - 6.0 * qk[comp])
                    * inv_dx2;
                out_k[comp] =
                    scale * (k_r * lap + bulk * qk[comp] + h_cubic[comp] + h_mag[comp]);
            }
        });
}

/// Fused Euler step: computes the molecular field and applies the Euler
/// update `Q <- Q + dt * gamma_r * H` in a single parallel pass.
///
/// Combines the Laplacian stencil, bulk Landau-de Gennes terms, magnetic
/// torque, and time integration into one kernel per vertex. Allocates a
/// single output vector (the updated Q) with no intermediate fields.
pub fn euler_step_fused_par(q: &mut QField3D, p: &ActiveNematicParams3D, t: f64) {
    let a_eff = p.a_eff();
    let c_ldg = p.c_landau;
    let b_landau = p.b_landau;
    let k_r = p.k_r;
    let gamma_r = p.gamma_r;
    let dt = p.dt;
    let inv_dx2 = 1.0 / (q.dx * q.dx);
    let nx = q.nx;
    let ny = q.ny;
    let nz = q.nz;
    let n = nx * ny * nz;

    // Precompute the combined coefficient: dt * gamma_r * k_r * inv_dx2
    let dt_gr = dt * gamma_r;
    let elastic_coeff = dt_gr * k_r * inv_dx2;

    // Magnetic torque (precomputed, same for all vertices).
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
    // Pre-scale magnetic torque by dt * gamma_r
    let dt_hmag = [
        dt_gr * h_mag[0],
        dt_gr * h_mag[1],
        dt_gr * h_mag[2],
        dt_gr * h_mag[3],
        dt_gr * h_mag[4],
    ];

    let nynz = ny * nz;
    let q_src = &q.q;

    // Parallel fused step: each vertex computes its own stencil, bulk LdG terms,
    // and Euler update in one map closure. The output is collected into a new Vec
    // (one allocation, replacing the two allocations of the old separate RHS + Euler).
    // The inner loop is manually unrolled for all 5 Q-components and uses pre-scaled
    // coefficients to minimise FLOPs per vertex.
    let out: Vec<[f64; 5]> = (0..n)
        .into_par_iter()
        .map(|k| {
            let l = k % nz;
            let ij = k / nz;
            let j = ij % ny;
            let i = ij / ny;

            // 6-point stencil neighbours (periodic).
            let ip = ((i + 1) % nx) * nynz + j * nz + l;
            let im = ((i + nx - 1) % nx) * nynz + j * nz + l;
            let jp = i * nynz + ((j + 1) % ny) * nz + l;
            let jm = i * nynz + ((j + ny - 1) % ny) * nz + l;
            let lp = i * nynz + j * nz + (l + 1) % nz;
            let lm = i * nynz + j * nz + (l + nz - 1) % nz;

            let qk = q_src[k];

            // Bulk LdG: dt * gamma_r * (-a_eff - 2c Tr(Q^2))
            let q11 = qk[0]; let q12 = qk[1]; let q13 = qk[2];
            let q22 = qk[3]; let q23 = qk[4];
            let q33 = -(q11 + q22);
            let tr_q2 = q11 * q11 + q22 * q22 + q33 * q33
                + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
            let dt_bulk = dt_gr * (-a_eff - 2.0 * c_ldg * tr_q2);

            // Cubic bulk term is not proportional to q_ij (unlike the a/c
            // terms above), so it cannot fold into the self-coefficient
            // `sc` below; it is added as its own dt*gamma_r-scaled term.
            // Zero cost when b_landau == 0.0 (`cubic_bulk_term` short-circuits).
            let h_cubic = cubic_bulk_term(q11, q12, q13, q22, q23, q33, tr_q2, b_landau);
            let dt_cubic = [
                dt_gr * h_cubic[0],
                dt_gr * h_cubic[1],
                dt_gr * h_cubic[2],
                dt_gr * h_cubic[3],
                dt_gr * h_cubic[4],
            ];

            // Combined self-coefficient:
            // q_new[c] = q[c] * (1 + dt_bulk - 6*elastic_coeff)
            //          + elastic_coeff * sum_neighbours + dt_cubic[c] + dt_hmag[c]
            let sc = 1.0 + dt_bulk - 6.0 * elastic_coeff;

            // Load 6 neighbours.
            let n0 = q_src[ip]; let n1 = q_src[im];
            let n2 = q_src[jp]; let n3 = q_src[jm];
            let n4 = q_src[lp]; let n5 = q_src[lm];

            // Fused stencil + bulk + cubic + magnetic + Euler, unrolled.
            [
                qk[0]*sc + elastic_coeff*(n0[0]+n1[0]+n2[0]+n3[0]+n4[0]+n5[0]) + dt_cubic[0] + dt_hmag[0],
                qk[1]*sc + elastic_coeff*(n0[1]+n1[1]+n2[1]+n3[1]+n4[1]+n5[1]) + dt_cubic[1] + dt_hmag[1],
                qk[2]*sc + elastic_coeff*(n0[2]+n1[2]+n2[2]+n3[2]+n4[2]+n5[2]) + dt_cubic[2] + dt_hmag[2],
                qk[3]*sc + elastic_coeff*(n0[3]+n1[3]+n2[3]+n3[3]+n4[3]+n5[3]) + dt_cubic[3] + dt_hmag[3],
                qk[4]*sc + elastic_coeff*(n0[4]+n1[4]+n2[4]+n3[4]+n4[4]+n5[4]) + dt_cubic[4] + dt_hmag[4],
            ]
        })
        .collect();

    q.q = out;
}

/// Compute the co-rotation tensor S(D, Omega, Q) at each vertex.
///
/// Uses the full nonlinear Beris-Edwards form:
/// ```text
/// S = xi*(D*Q + Q*D) - 2*xi*Tr(Q*D)*Q + Omega*Q - Q*Omega
/// ```
///
/// `xi` is the flow-alignment parameter (ActiveNematicParams3D.lambda field).
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
    use volterra_core::ActiveNematicParams3D;
    use volterra_fields::{QField3D, VelocityField3D};

    #[test]
    fn test_molecular_field_uniform_no_mag() {
        let p = ActiveNematicParams3D::default_test();
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
        let p = ActiveNematicParams3D::default_test();
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
        let p = ActiveNematicParams3D::default_test();
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
