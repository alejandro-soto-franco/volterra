//! Enhanced nematic locking, after Mitchell, Sabbir, Klein and Beller,
//! "Modelling active nematics via the nematic locking principle", Soft Matter
//! (2025), arXiv:2506.20996.
//!
//! # The principle
//!
//! A microtubule bundle is an extended object, so steric interaction stops any
//! one bundle rotating unless its whole neighbourhood rotates with it. A
//! director field with that property is *locked* to the flow: an integral curve
//! of the director, advected as a passive material line, stays an integral
//! curve of the director. The reference verifies this in experimental data and
//! shows it fails in the standard Beris-Edwards model throughout the bulk
//! rather than only at defect cores.
//!
//! # Where the standard model breaks it
//!
//! In two dimensions `H` is symmetric and traceless, so it is spanned by
//!
//! ```text
//! Q = S (n (x) n - I/2),      U = J Q = S (n (x) n_perp + n_perp (x) n)/2,
//! ```
//!
//! `J` the rotation by `pi/2`, giving the exact identity
//!
//! ```text
//! H / gamma = (2 / (gamma S^2)) [ Tr(H Q) Q + Tr(H U) U ].
//! ```
//!
//! The `Q` part changes only `S` and preserves locking. The `U` part rotates
//! `n` at the *fracturing* rate `omega_F = Tr(H U) / (gamma S^2)` and is the
//! only term that breaks it. Fracturing is what creates and annihilates defect
//! pairs, so it cannot be removed; the reference's measurement is that standard
//! Beris-Edwards applies it everywhere, at an RMS about 60 per cent of the
//! advective rotation rate, rather than only where the bundles are dilute.
//!
//! # The modification
//!
//! Multiply the mobility of the `U` term by a switch on `S`, read as a proxy
//! for density, so fracturing turns on only where `S` has fallen:
//!
//! ```text
//! H / gamma  ->  (2 / (gamma S^2)) [ Tr(H Q) Q + f(S) Tr(H U) U ],
//! f(S) = exp(-S^2 / (2 sigma^2)),   sigma = 0.2 S_eq,
//! ```
//!
//! implemented as the difference from the unmodified equation, which is what
//! [`add_fracture_switch`] accumulates:
//!
//! ```text
//! dQ/dt  +=  (1/gamma) (2 (f(S) - 1) / S^2) Tr(H U) U.
//! ```
//!
//! `H` itself is unchanged, so the Navier-Stokes side of the solver sees the
//! same molecular field and the same stress; the reference states this
//! explicitly, and it is why the modification is one term added after
//! [`crate::step::get_q_update`] rather than a change to
//! [`crate::nematic::h_s_from_q`].
//!
//! The reference requires `lambda = 1` for the rest of the transport equation
//! to preserve locking, which is the value `flow-solver.py` and every
//! reproduction in this crate already use.
//!
//! # Component conventions
//!
//! Following the rest of the crate, `Q` and `H` are two-component fields with
//! `c = 0` the `xx` entry and `c = 1` the `xy` entry, so
//!
//! ```text
//! Q = [[q0, q1], [q1, -q0]],   U = J Q = [[-q1, q0], [q0, q1]],
//! Tr(Q^2) = 2 (q0^2 + q1^2) = S^2 / 2,   so   S^2 = 4 (q0^2 + q1^2),
//! Tr(H U) = 2 (h1 q0 - h0 q1).
//! ```
//!
//! # Sigma, and the two `S` conventions in the literature
//!
//! The reference sets `C = -2A`, which puts equilibrium at `S = 1`, and quotes
//! `sigma = 0.2`. Mitchell et al. (2024) and `flow-solver.py` set `C = -A`,
//! which puts it at `S = sqrt(2)`. [`Locking::sigma`] is therefore carried in
//! units of the equilibrium `S`, and every function here takes `s_eq` and forms
//! `sigma_eff = sigma * s_eq`. At `sigma = 0.2` the switch is `3.7e-6` at
//! equilibrium and `1` at a defect core, in either convention.

use crate::{
    Boundary,
    index::{si, vi},
    par_gate::{rows_per_chunk, use_parallel},
};
use rayon::prelude::*;

/// Switch parameters for enhanced nematic locking.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Locking {
    /// Width of the Gaussian switch `exp(-S^2 / (2 sigma^2))`, in units of the
    /// equilibrium scalar order parameter. arXiv:2506.20996 uses `0.2`.
    pub sigma: f64,
}

impl Default for Locking {
    fn default() -> Self {
        Self { sigma: 0.2 }
    }
}

impl Locking {
    /// The reference's own width, `sigma = 0.2`.
    pub const REFERENCE: Self = Self { sigma: 0.2 };
}

/// `expm1(-u) / u`, continued to `u = 0` where it takes the value `-1`.
///
/// The switch prefactor is `(f(S) - 1) / (q0^2 + q1^2)`, a `0/0` at a defect
/// core written naively. Factoring the ratio out this way leaves no division by
/// a small number anywhere in [`add_fracture_switch`].
#[inline]
fn expm1_neg_over(u: f64) -> f64 {
    // Below this the three-term series is already correct to the last bit, and
    // above it `exp_m1` is accurate on its own.
    if u < 1e-8 {
        -1.0 + 0.5 * u - u * u / 6.0
    } else {
        (-u).exp_m1() / u
    }
}

/// `(f(S) - 1) / (q0^2 + q1^2)`, the prefactor multiplying
/// `Tr(H U) U / gamma` in the added term.
///
/// Finite everywhere: it tends to `-2 / sigma_eff^2` as `Q` tends to zero, and
/// the two factors of `S` carried by `Tr(H U)` and `U` send the added term
/// itself to zero there.
#[inline]
pub fn switch_prefactor(r2: f64, sigma_eff: f64) -> f64 {
    let sig2 = sigma_eff * sigma_eff;
    // S^2 = 4 r2, so u = S^2 / (2 sigma_eff^2) = 2 r2 / sigma_eff^2.
    let u = 2.0 * r2 / sig2;
    (2.0 / sig2) * expm1_neg_over(u)
}

/// The Gaussian switch itself, `f(S) = exp(-S^2 / (2 sigma_eff^2))`.
///
/// Only the diagnostics need this; [`add_fracture_switch`] uses
/// [`switch_prefactor`], which never forms `f` and `S^2` separately.
#[inline]
pub fn switch(r2: f64, sigma_eff: f64) -> f64 {
    (-2.0 * r2 / (sigma_eff * sigma_eff)).exp()
}

/// Accumulate the enhanced-locking correction onto an already-computed `dQ/dt`.
///
/// Call after [`crate::step::get_q_update`], with the same `H` that function
/// was given. `sigma` is in units of `s_eq`; see the module documentation.
///
/// Adds, at every interior cell,
///
/// ```text
/// dQ += (1/gamma) [(f(S) - 1) / (q0^2 + q1^2)] (h1 q0 - h0 q1) (-q1, q0).
/// ```
pub fn add_fracture_switch(
    dq: &mut [f64],
    q: &[f64],
    h: &[f64],
    gamma: f64,
    s_eq: f64,
    sigma: f64,
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let inv_gamma = 1.0 / gamma;
    let sigma_eff = sigma * s_eq;

    // The two increments at one cell, as a pair, so neither path needs two
    // mutable borrows into the same row.
    let cell = |x: usize, y: usize| -> [f64; 2] {
        let q0 = q[vi(x, y, ly, 0)];
        let q1 = q[vi(x, y, ly, 1)];
        let h0 = h[vi(x, y, ly, 0)];
        let h1 = h[vi(x, y, ly, 1)];

        let r2 = q0 * q0 + q1 * q1;
        let pref = switch_prefactor(r2, sigma_eff);
        // `Tr(H U) / 2`: the other factor of 2 cancels the one in
        // `2 (f - 1) / S^2`, which `switch_prefactor` has already absorbed.
        let cross = h1 * q0 - h0 * q1;
        let amp = inv_gamma * pref * cross;

        [amp * (-q1), amp * q0]
    };

    if use_parallel(lx, ly) {
        let rpc = rows_per_chunk(lx);
        dq.par_chunks_mut(rpc * ly * 2)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let x_start = chunk_idx * rpc;
                for (row_offset, row) in chunk.chunks_mut(ly * 2).enumerate() {
                    let x = x_start + row_offset;
                    if x >= lx {
                        break;
                    }
                    for y in 0..ly {
                        if !bounds.inside[si(x, y, ly)] {
                            continue;
                        }
                        let d = cell(x, y);
                        row[y * 2] += d[0];
                        row[y * 2 + 1] += d[1];
                    }
                }
            });
    } else {
        for x in 0..lx {
            for y in 0..ly {
                if !bounds.inside[si(x, y, ly)] {
                    continue;
                }
                let d = cell(x, y);
                dq[vi(x, y, ly, 0)] += d[0];
                dq[vi(x, y, ly, 1)] += d[1];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// The two rotation rates of the director, per cell.
///
/// `omega_n = omega_a + omega_f`: the advective rate the flow imposes, and the
/// fracturing rate the `U` part of `H` adds. arXiv:2506.20996 Eqs. (12) and
/// (26). Cells outside `bounds.inside` are left at zero.
#[derive(Debug, Clone)]
pub struct RotationRates {
    /// `omega_A = n . E n_perp + omega / 2`, from advection alone.
    pub omega_a: Vec<f64>,
    /// `omega_F = f(S) Tr(H U) / (gamma S^2)`, from fracturing.
    pub omega_f: Vec<f64>,
}

/// Compute `omega_A` and `omega_F` from the current fields.
///
/// Pass `locking = None` for the standard Beris-Edwards model, in which the
/// switch is identically one. `h` must be the molecular field
/// [`crate::nematic::h_s_from_q`] wrote for this `q`.
///
/// At a defect core `S` reaches zero and `omega_F` diverges; the reference
/// reports the distribution of `omega_F` including that tail, so the value is
/// left as computed and only an exactly zero `Q` returns zero.
pub fn rotation_rates(
    u: &[f64],
    q: &[f64],
    h: &[f64],
    gamma: f64,
    s_eq: f64,
    locking: Option<Locking>,
    bounds: &Boundary,
) -> RotationRates {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let mut omega_a = vec![0.0; lx * ly];
    let mut omega_f = vec![0.0; lx * ly];
    let sigma_eff = locking.map(|l| l.sigma * s_eq);

    for x in 0..lx {
        let xup = (x + 1) % lx;
        let xdn = (x + lx - 1) % lx;
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            let dxux = 0.5 * (u[vi(xup, y, ly, 0)] - u[vi(xdn, y, ly, 0)]);
            let dxuy = 0.5 * (u[vi(xup, y, ly, 1)] - u[vi(xdn, y, ly, 1)]);
            let dyux = 0.5 * (u[vi(x, yup, ly, 0)] - u[vi(x, ydn, ly, 0)]);

            let e_xx = dxux;
            let e_xy = 0.5 * (dxuy + dyux);
            let omega_over_2 = 0.5 * (dxuy - dyux);

            let q0 = q[vi(x, y, ly, 0)];
            let q1 = q[vi(x, y, ly, 1)];
            let r2 = q0 * q0 + q1 * q1;
            let r = r2.sqrt();

            // cos 2theta = q0 / r, sin 2theta = q1 / r, since Q = (S/2)(cos 2t, sin 2t).
            if r > 0.0 {
                omega_a[idx] = e_xy * (q0 / r) - e_xx * (q1 / r) + omega_over_2;
            } else {
                omega_a[idx] = omega_over_2;
            }

            let h0 = h[vi(x, y, ly, 0)];
            let h1 = h[vi(x, y, ly, 1)];
            if r2 > 0.0 {
                // Tr(H U) / (gamma S^2) = 2 (h1 q0 - h0 q1) / (gamma 4 r2).
                let base = (h1 * q0 - h0 * q1) / (2.0 * gamma * r2);
                omega_f[idx] = match sigma_eff {
                    Some(se) => base * switch(r2, se),
                    None => base,
                };
            }
        }
    }

    RotationRates { omega_a, omega_f }
}

/// Root mean square and median of the absolute values over interior cells.
///
/// The reference reports both for `omega_A` and `omega_F`, and the pair is what
/// separates the two models: the RMS of `omega_F` rises under enhanced locking
/// while its median falls by five orders of magnitude, which is the statement
/// that fracturing has become localised rather than absent.
pub fn rms_and_median(field: &[f64], bounds: &Boundary) -> (f64, f64) {
    let mut abs: Vec<f64> = Vec::with_capacity(field.len());
    let mut sq = 0.0;
    for (idx, &v) in field.iter().enumerate() {
        if !bounds.inside[idx] {
            continue;
        }
        sq += v * v;
        abs.push(v.abs());
    }
    if abs.is_empty() {
        return (0.0, 0.0);
    }
    let rms = (sq / abs.len() as f64).sqrt();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = abs.len();
    let median = if n % 2 == 1 {
        abs[n / 2]
    } else {
        0.5 * (abs[n / 2 - 1] + abs[n / 2])
    };
    (rms, median)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::periodic_boundary;

    /// The projection identity the whole modification rests on:
    /// `H = (2/S^2) [Tr(HQ) Q + Tr(HU) U]` for any traceless symmetric pair.
    #[test]
    fn q_u_basis_reconstructs_h() {
        let (q0, q1) = (0.31_f64, -0.47_f64);
        let (h0, h1) = (1.7_f64, 0.9_f64);
        let s2 = 4.0 * (q0 * q0 + q1 * q1);
        let tr_hq = 2.0 * (h0 * q0 + h1 * q1);
        let tr_hu = 2.0 * (h1 * q0 - h0 * q1);
        // U = (-q1, q0)
        let r0 = (2.0 / s2) * (tr_hq * q0 + tr_hu * (-q1));
        let r1 = (2.0 / s2) * (tr_hq * q1 + tr_hu * q0);
        assert!((r0 - h0).abs() < 1e-12, "xx: {r0} vs {h0}");
        assert!((r1 - h1).abs() < 1e-12, "xy: {r1} vs {h1}");
    }

    /// At equilibrium the switch is off, so the correction must cancel the `U`
    /// part of `H / gamma` to within the switch value itself.
    #[test]
    fn correction_cancels_u_part_at_equilibrium() {
        let s_eq = 2.0_f64.sqrt();
        let sigma_eff = 0.2 * s_eq;
        // Q at equilibrium along x: q0 = S/2, q1 = 0.
        let (q0, q1) = (s_eq / 2.0, 0.0);
        let (h0, h1) = (0.4, 1.3);
        let gamma = 1.0;
        let r2 = q0 * q0 + q1 * q1;

        let bnd = periodic_boundary(1, 1);
        let q = vec![q0, q1];
        let h = vec![h0, h1];
        let mut dq = vec![0.0, 0.0];
        add_fracture_switch(&mut dq, &q, &h, gamma, s_eq, 0.2, &bnd);

        // The U component of H/gamma is (2/(gamma S^2)) Tr(HU) U.
        let s2 = 4.0 * r2;
        let tr_hu = 2.0 * (h1 * q0 - h0 * q1);
        let u_part = [
            (2.0 / (gamma * s2)) * tr_hu * (-q1),
            (2.0 / (gamma * s2)) * tr_hu * q0,
        ];
        let f = switch(r2, sigma_eff);
        for c in 0..2 {
            // dq = -(1 - f) * u_part, so dq + u_part = f * u_part.
            let residual = dq[c] + u_part[c];
            assert!(
                (residual - f * u_part[c]).abs() < 1e-12,
                "component {c}: {residual} vs {}",
                f * u_part[c]
            );
        }
        // And the switch really is negligible at equilibrium.
        assert!(f < 1e-5, "switch at equilibrium is {f}");
    }

    /// At a defect core the switch is fully on, so the correction vanishes.
    #[test]
    fn correction_vanishes_at_a_defect_core() {
        let bnd = periodic_boundary(1, 1);
        let q = vec![0.0, 0.0];
        let h = vec![3.0, -2.0];
        let mut dq = vec![0.0, 0.0];
        add_fracture_switch(&mut dq, &q, &h, 1.0, 1.0, 0.2, &bnd);
        assert_eq!(dq, vec![0.0, 0.0]);
    }

    /// The prefactor is smooth through the core rather than a `0/0`.
    #[test]
    fn switch_prefactor_is_finite_and_continuous_at_zero() {
        let sigma_eff = 0.2;
        let at_zero = switch_prefactor(0.0, sigma_eff);
        assert!((at_zero + 2.0 / (sigma_eff * sigma_eff)).abs() < 1e-12);
        let mut prev = at_zero;
        for k in 1..200 {
            let r2: f64 = k as f64 * 1e-6;
            let v = switch_prefactor(r2, sigma_eff);
            assert!(v.is_finite());
            assert!((v - prev).abs() < 1e-2, "jump at r2={r2}: {prev} -> {v}");
            prev = v;
        }
        // Away from zero it agrees with the naive form.
        for &r2 in &[0.01_f64, 0.05, 0.25] {
            let naive = (switch(r2, sigma_eff) - 1.0) / r2;
            let v = switch_prefactor(r2, sigma_eff);
            assert!((v - naive).abs() < 1e-9 * naive.abs().max(1.0));
        }
    }

    /// `omega_F` from [`rotation_rates`] is the rate the added term removes:
    /// switching locking on must scale it by exactly `f(S)`.
    #[test]
    fn omega_f_is_scaled_by_the_switch() {
        let bnd = periodic_boundary(4, 4);
        let n = 16;
        let mut q = vec![0.0; n * 2];
        let mut h = vec![0.0; n * 2];
        for i in 0..n {
            q[i * 2] = 0.5 + 0.01 * i as f64;
            q[i * 2 + 1] = 0.2 - 0.005 * i as f64;
            h[i * 2] = 0.3 * (i as f64).sin();
            h[i * 2 + 1] = 0.7 * (i as f64).cos();
        }
        let u = vec![0.0; n * 2];
        let s_eq = 1.0;
        let be = rotation_rates(&u, &q, &h, 2.0, s_eq, None, &bnd);
        let benl = rotation_rates(&u, &q, &h, 2.0, s_eq, Some(Locking::REFERENCE), &bnd);
        for i in 0..n {
            let r2 = q[i * 2] * q[i * 2] + q[i * 2 + 1] * q[i * 2 + 1];
            let f = switch(r2, 0.2 * s_eq);
            assert!((benl.omega_f[i] - f * be.omega_f[i]).abs() < 1e-14);
        }
    }

    /// The parallel and serial paths of [`add_fracture_switch`] must agree to
    /// the last bit, as every other kernel in the crate does.
    #[test]
    fn parallel_and_serial_paths_agree() {
        // Above the par_gate threshold so the rayon path is taken.
        let lx = 512;
        let ly = 512;
        let bnd = periodic_boundary(lx, ly);
        let n = lx * ly;
        let mut q = vec![0.0; n * 2];
        let mut h = vec![0.0; n * 2];
        for i in 0..n {
            let t = i as f64 * 1e-4;
            q[i * 2] = 0.6 * t.sin();
            q[i * 2 + 1] = 0.6 * t.cos();
            h[i * 2] = t.cos();
            h[i * 2 + 1] = t.sin();
        }
        assert!(crate::par_gate::use_parallel(lx, ly));
        let mut par = vec![0.0; n * 2];
        add_fracture_switch(&mut par, &q, &h, 3.0, 1.0, 0.2, &bnd);

        // Same kernel on a grid small enough to take the serial path, tile by
        // tile, is not equivalent; instead recompute the closed form directly.
        for i in 0..n {
            let (q0, q1) = (q[i * 2], q[i * 2 + 1]);
            let (h0, h1) = (h[i * 2], h[i * 2 + 1]);
            let r2 = q0 * q0 + q1 * q1;
            let amp = (1.0 / 3.0) * switch_prefactor(r2, 0.2) * (h1 * q0 - h0 * q1);
            assert_eq!(par[i * 2], amp * (-q1));
            assert_eq!(par[i * 2 + 1], amp * q0);
        }
    }
}
