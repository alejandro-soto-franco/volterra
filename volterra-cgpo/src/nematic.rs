//! Beris–Edwards molecular field and stress kernels.
//!
//! Ports the following numba functions from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`:
//!
//! - `H_S_from_Q`          — molecular field H_ij (LdG bulk + K∇²Q) and co-rotation S_ij
//! - `get_Erickson_stress`  — Ericksen elastic stress contribution to Π_S
//! - `get_TrQH_term`        — 2 Tr[QH] Q term added to Π_S
//! - `calculate_Pi`         — assembles full symmetric Π_S and antisymmetric Π_A
//!
//! # Field layout
//!
//! All layouts follow the conventions in `ops.rs`:
//!
//! - **Q, H, S, Π_S**: 2-component fields, `&[f64]` length `lx*ly*2`,
//!   flat index `(x*ly + y)*2 + c` where `c ∈ {0=xx, 1=xy}`.
//! - **Π_A**: scalar field, `&[f64]` length `lx*ly`,
//!   flat index `x*ly + y`.  Only the `xy` (antisymmetric) component is stored.
//! - **u**: 2-component velocity field, same layout as Q.
//!
//! # Index wrapping
//!
//! Negative neighbour indices wrap as `(i + n) % n`, replicating Python's
//! bare `-1` which NumPy wraps to `n-1`.
//!
//! # Sign conventions
//!
//! - `A` enters as the raw LdG coefficient (expected negative for ordering).
//! - `C` is the positive quartic coefficient.
//! - `trQsq = 2*(Q0² + Q1²)` — the 2D trace of Q².
//! - Bulk contribution to H: `-(A + C*trQsq)*Q`  (subtracted from the Laplacian result).
//! - `Π_S` initialised to `−λH − ζQ`, then Ericksen term subtracted, then `2Tr[QH]Q` added.
//! - `Π_A = 2*(Q0*H1 − H0*Q1)` — one scalar.

use crate::{
    ops::laplacian_vector,
    par_gate::{rows_per_chunk, use_parallel},
    Boundary,
};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// index helpers (module-local, matching ops.rs conventions)
// ---------------------------------------------------------------------------

#[inline(always)]
fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}

#[inline(always)]
fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}

// ---------------------------------------------------------------------------
// Public kernels
// ---------------------------------------------------------------------------

/// Compute the molecular field H and co-rotation tensor S from Q.
///
/// Ports `H_S_from_Q(u, Q, H, S, A, C, K, λ, bounds)`.
///
/// On entry `h` need not be zeroed — it is fully overwritten.
/// `s` is also fully overwritten at every interior cell.
///
/// # Steps
///
/// 1. `H = K * ∇²Q`  (via the 9-point isotropic Laplacian).
/// 2. For every interior cell:
///    - `trQsq = 2*(Q0² + Q1²)`
///    - `H -= (A + C*trQsq) * Q`
///    - Compute velocity gradients (centred differences with wrap):
///      `dxux, dxuy, dyux, dyuy`
///    - `ωxy = 0.5*(dxuy − dyux)`
///    - `λS = λ * sqrt(2*trQsq)`
///    - `TrQE = 2*Q0*dxux + Q1*(dyux + dxuy)`
///    - `S0 = λS*dxux − 2*ωxy*Q1 − 2*TrQE*Q0`
///    - `S1 = λS*0.5*(dxuy + dyux) + 2*ωxy*Q0 − 2*TrQE*Q1`
pub fn h_s_from_q(
    u: &[f64],
    q: &[f64],
    h: &mut [f64],
    s: &mut [f64],
    a: f64,
    c_coeff: f64,
    k: f64,
    lambda: f64,
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;

    // Step 1: H = K * ∇²Q
    laplacian_vector(q, h, bounds, k);

    // Step 2: bulk LdG correction + co-rotation S.
    if use_parallel(lx, ly) {
        let rpc = rows_per_chunk(lx);
        h.par_chunks_mut(rpc * ly * 2)
            .zip(s.par_chunks_mut(rpc * ly * 2))
            .enumerate()
            .for_each(|(chunk_idx, (h_chunk, s_chunk))| {
                let x_start = chunk_idx * rpc;
                for (row_offset, (h_row, s_row)) in h_chunk
                    .chunks_mut(ly * 2)
                    .zip(s_chunk.chunks_mut(ly * 2))
                    .enumerate()
                {
                    let x = x_start + row_offset;
                    if x >= lx { break; }
                    let xup = (x + 1) % lx;
                    let xdn = (x + lx - 1) % lx;
                    for y in 0..ly {
                        let idx = si(x, y, ly);
                        if !bounds.inside[idx] { continue; }
                        let yup = (y + 1) % ly;
                        let ydn = (y + ly - 1) % ly;

                        let q0 = q[vi(x, y, ly, 0)];
                        let q1 = q[vi(x, y, ly, 1)];
                        let trqsq = 2.0 * (q0 * q0 + q1 * q1);

                        h_row[y * 2]     -= (a + c_coeff * trqsq) * q0;
                        h_row[y * 2 + 1] -= (a + c_coeff * trqsq) * q1;

                        let dxux = 0.5 * (u[vi(xup, y, ly, 0)] - u[vi(xdn, y, ly, 0)]);
                        let dxuy = 0.5 * (u[vi(xup, y, ly, 1)] - u[vi(xdn, y, ly, 1)]);
                        let dyux = 0.5 * (u[vi(x, yup, ly, 0)] - u[vi(x, ydn, ly, 0)]);
                        let dyuy = 0.5 * (u[vi(x, yup, ly, 1)] - u[vi(x, ydn, ly, 1)]);
                        let _ = dyuy;

                        let omega_xy = 0.5 * (dxuy - dyux);
                        let lambda_s = lambda * (2.0 * trqsq).sqrt();
                        let tr_qe = 2.0 * q0 * dxux + q1 * (dyux + dxuy);

                        s_row[y * 2]     = lambda_s * dxux
                            - 2.0 * omega_xy * q1
                            - 2.0 * tr_qe * q0;
                        s_row[y * 2 + 1] = lambda_s * 0.5 * (dxuy + dyux)
                            + 2.0 * omega_xy * q0
                            - 2.0 * tr_qe * q1;
                    }
                }
            });
    } else {
        for x in 0..lx {
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            for y in 0..ly {
                let idx = si(x, y, ly);
                if !bounds.inside[idx] { continue; }
                let yup = (y + 1) % ly;
                let ydn = (y + ly - 1) % ly;

                let q0 = q[vi(x, y, ly, 0)];
                let q1 = q[vi(x, y, ly, 1)];
                let trqsq = 2.0 * (q0 * q0 + q1 * q1);

                h[vi(x, y, ly, 0)] -= (a + c_coeff * trqsq) * q0;
                h[vi(x, y, ly, 1)] -= (a + c_coeff * trqsq) * q1;

                let dxux = 0.5 * (u[vi(xup, y, ly, 0)] - u[vi(xdn, y, ly, 0)]);
                let dxuy = 0.5 * (u[vi(xup, y, ly, 1)] - u[vi(xdn, y, ly, 1)]);
                let dyux = 0.5 * (u[vi(x, yup, ly, 0)] - u[vi(x, ydn, ly, 0)]);
                let dyuy = 0.5 * (u[vi(x, yup, ly, 1)] - u[vi(x, ydn, ly, 1)]);
                let _ = dyuy;

                let omega_xy = 0.5 * (dxuy - dyux);
                let lambda_s = lambda * (2.0 * trqsq).sqrt();
                let tr_qe = 2.0 * q0 * dxux + q1 * (dyux + dxuy);

                s[vi(x, y, ly, 0)] = lambda_s * dxux
                    - 2.0 * omega_xy * q1
                    - 2.0 * tr_qe * q0;
                s[vi(x, y, ly, 1)] = lambda_s * 0.5 * (dxuy + dyux)
                    + 2.0 * omega_xy * q0
                    - 2.0 * tr_qe * q1;
            }
        }
    }
}

/// Add the Ericksen (elastic gradient) stress contribution to Π_S.
///
/// Ports `get_Erickson_stress(Q, K, Π_S, bounds)`.
///
/// **Accumulates** into `pi_s` — caller must initialise it before calling.
///
/// ```text
/// dxQ = 0.5*(Q[x+1,y] - Q[x-1,y])
/// dyQ = 0.5*(Q[x,y+1] - Q[x,y-1])
/// pi_s[x,y,0] -= K * (dxQ0² + dxQ1² - dyQ0² - dyQ1²)
/// pi_s[x,y,1] -= 2*K * (dxQ1*dyQ1 + dxQ0*dyQ0)
/// ```
pub fn get_ericksen_stress(q: &[f64], k: f64, pi_s: &mut [f64], bounds: &Boundary) {
    let lx = bounds.lx;
    let ly = bounds.ly;

    if use_parallel(lx, ly) {
        let rpc = rows_per_chunk(lx);
        pi_s.par_chunks_mut(rpc * ly * 2)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let x_start = chunk_idx * rpc;
                for (row_offset, row) in chunk.chunks_mut(ly * 2).enumerate() {
                    let x = x_start + row_offset;
                    if x >= lx { break; }
                    let xup = (x + 1) % lx;
                    let xdn = (x + lx - 1) % lx;
                    for y in 0..ly {
                        let idx = si(x, y, ly);
                        if !bounds.inside[idx] { continue; }
                        let yup = (y + 1) % ly;
                        let ydn = (y + ly - 1) % ly;

                        let dxq0 = 0.5 * (q[vi(xup, y, ly, 0)] - q[vi(xdn, y, ly, 0)]);
                        let dxq1 = 0.5 * (q[vi(xup, y, ly, 1)] - q[vi(xdn, y, ly, 1)]);
                        let dyq0 = 0.5 * (q[vi(x, yup, ly, 0)] - q[vi(x, ydn, ly, 0)]);
                        let dyq1 = 0.5 * (q[vi(x, yup, ly, 1)] - q[vi(x, ydn, ly, 1)]);

                        row[y * 2]     -= k * (dxq0 * dxq0 + dxq1 * dxq1 - dyq0 * dyq0 - dyq1 * dyq1);
                        row[y * 2 + 1] -= 2.0 * k * (dxq1 * dyq1 + dxq0 * dyq0);
                    }
                }
            });
    } else {
        for x in 0..lx {
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            for y in 0..ly {
                let idx = si(x, y, ly);
                if !bounds.inside[idx] { continue; }
                let yup = (y + 1) % ly;
                let ydn = (y + ly - 1) % ly;

                let dxq0 = 0.5 * (q[vi(xup, y, ly, 0)] - q[vi(xdn, y, ly, 0)]);
                let dxq1 = 0.5 * (q[vi(xup, y, ly, 1)] - q[vi(xdn, y, ly, 1)]);
                let dyq0 = 0.5 * (q[vi(x, yup, ly, 0)] - q[vi(x, ydn, ly, 0)]);
                let dyq1 = 0.5 * (q[vi(x, yup, ly, 1)] - q[vi(x, ydn, ly, 1)]);

                pi_s[vi(x, y, ly, 0)] -= k * (dxq0 * dxq0 + dxq1 * dxq1 - dyq0 * dyq0 - dyq1 * dyq1);
                pi_s[vi(x, y, ly, 1)] -= 2.0 * k * (dxq1 * dyq1 + dxq0 * dyq0);
            }
        }
    }
}

/// Add the `2 Tr[QH] Q` term to Π_S at every grid cell.
///
/// Ports `get_TrQH_term(Q, H, Π_S)`.
///
/// **No boundary mask** — applied globally (matching the Python broadcast).
/// `Π_S` must be pre-initialised before calling.
///
/// ```text
/// TrQH = 2*(Q0*H0 + Q1*H1)
/// pi_s[x,y,0] += TrQH * Q0
/// pi_s[x,y,1] += TrQH * Q1
/// ```
pub fn get_trqh_term(q: &[f64], h: &[f64], pi_s: &mut [f64], lx: usize, ly: usize) {
    let n2 = lx * ly * 2;

    if use_parallel(lx, ly) {
        let rpc = rows_per_chunk(lx);
        pi_s[..n2].par_chunks_mut(rpc * ly * 2)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let x_start = chunk_idx * rpc;
                for (row_offset, row) in chunk.chunks_mut(ly * 2).enumerate() {
                    let x = x_start + row_offset;
                    if x >= lx { break; }
                    for y in 0..ly {
                        let q0 = q[vi(x, y, ly, 0)];
                        let q1 = q[vi(x, y, ly, 1)];
                        let h0 = h[vi(x, y, ly, 0)];
                        let h1 = h[vi(x, y, ly, 1)];
                        let trqh = 2.0 * (q0 * h0 + q1 * h1);
                        row[y * 2]     += trqh * q0;
                        row[y * 2 + 1] += trqh * q1;
                    }
                }
            });
    } else {
        for x in 0..lx {
            for y in 0..ly {
                let q0 = q[vi(x, y, ly, 0)];
                let q1 = q[vi(x, y, ly, 1)];
                let h0 = h[vi(x, y, ly, 0)];
                let h1 = h[vi(x, y, ly, 1)];
                let trqh = 2.0 * (q0 * h0 + q1 * h1);
                pi_s[vi(x, y, ly, 0)] += trqh * q0;
                pi_s[vi(x, y, ly, 1)] += trqh * q1;
            }
        }
    }
}

/// Assemble the symmetric (Π_S) and antisymmetric (Π_A) stress tensors.
///
/// Ports `calculate_Π(Π_S, Π_A, H, Q, λ, ζ, K, bounds)`.
///
/// # Algorithm
///
/// 1. `Π_S = −λ*H − ζ*Q`  (broadcast over all cells)
/// 2. Add Ericksen stress to Π_S (interior cells only, via [`get_ericksen_stress`])
/// 3. Add `2 Tr[QH] Q` to Π_S (all cells, via [`get_trqh_term`])
/// 4. `Π_A[x,y] = 2*(Q0*H1 − H0*Q1)`  (broadcast over all cells)
///
/// # Arguments
///
/// - `pi_s` : output symmetric stress, 2-component, length `lx*ly*2` — overwritten
/// - `pi_a` : output antisymmetric stress, scalar, length `lx*ly` — overwritten
/// - `h`    : molecular field (from [`h_s_from_q`])
/// - `q`    : Q-tensor field
/// - `lambda`, `zeta`, `k` : model constants
/// - `bounds` : boundary mask
pub fn calculate_pi(
    pi_s: &mut [f64],
    pi_a: &mut [f64],
    h: &[f64],
    q: &[f64],
    lambda: f64,
    zeta: f64,
    k: f64,
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let n2 = lx * ly * 2;

    // Step 1: Π_S = −λ*H − ζ*Q
    if use_parallel(lx, ly) {
        pi_s[..n2].par_iter_mut()
            .zip(h[..n2].par_iter())
            .zip(q[..n2].par_iter())
            .for_each(|((ps, hi), qi)| {
                *ps = -lambda * hi - zeta * qi;
            });
    } else {
        for i in 0..n2 {
            pi_s[i] = -lambda * h[i] - zeta * q[i];
        }
    }

    // Step 2: add Ericksen stress (interior cells)
    get_ericksen_stress(q, k, pi_s, bounds);

    // Step 3: add 2 Tr[QH] Q (all cells)
    get_trqh_term(q, h, pi_s, lx, ly);

    // Step 4: Π_A = 2*(Q0*H1 − H0*Q1)
    if use_parallel(lx, ly) {
        let rpc = rows_per_chunk(lx);
        pi_a.par_chunks_mut(rpc * ly)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let x_start = chunk_idx * rpc;
                for (row_offset, row) in chunk.chunks_mut(ly).enumerate() {
                    let x = x_start + row_offset;
                    if x >= lx { break; }
                    for y in 0..ly {
                        let q0 = q[vi(x, y, ly, 0)];
                        let q1 = q[vi(x, y, ly, 1)];
                        let h0 = h[vi(x, y, ly, 0)];
                        let h1 = h[vi(x, y, ly, 1)];
                        row[y] = 2.0 * (q0 * h1 - h0 * q1);
                    }
                }
            });
    } else {
        for x in 0..lx {
            for y in 0..ly {
                let q0 = q[vi(x, y, ly, 0)];
                let q1 = q[vi(x, y, ly, 1)];
                let h0 = h[vi(x, y, ly, 0)];
                let h1 = h[vi(x, y, ly, 1)];
                pi_a[si(x, y, ly)] = 2.0 * (q0 * h1 - h0 * q1);
            }
        }
    }
}
