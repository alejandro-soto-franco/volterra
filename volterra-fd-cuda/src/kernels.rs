//! Double-precision CUDA kernels for volterra's 2D confined active-nematic
//! solver, compiled to PTX by `rustc-codegen-cuda` (the pattern `cartan-cuda`
//! and `volterra-cuda` establish).
//!
//! ## Layout
//!
//! The CPU side stores a scalar field as `s[x * ly + y]` and a 2-vector field
//! as `v[(x * ly + y) * 2 + c]`, `c` in `{0, 1}` (`volterra_fd::index`). The
//! kernels below take the same flat buffers with the same indexing, so a host
//! buffer transfers with no repacking and a kernel can be checked against its
//! CPU counterpart element for element.
//!
//! ## The interior mask
//!
//! Every operator writes only where `inside[idx]` is true and leaves the rest
//! of `out` as it found it. The CPU does the same, with `continue`, and callers
//! rely on it: several of these operators accumulate, and the confining
//! boundary owns the cells they skip. `inside` arrives as `u8` because `bool`
//! is not a device-copyable element type.
//!
//! ## Threading
//!
//! One thread per grid cell, indexed `idx = x * ly + y`, so `x = idx / ly` and
//! `y = idx % ly`. The 2D grid is 100x100 in the runs this exists for, which is
//! 10,000 cells and does not fill a modern device on its own; the batched entry
//! points take a run index as the slowest dimension so a parameter sweep fills
//! it instead.

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// 9-point isotropic Laplacian of a scalar field, `coeff` scaled.
    ///
    /// Ports `volterra_fd::ops::laplacian`, including its `/6` normalisation
    /// and its `-20 / 4 / 1` stencil weights, and including that it writes
    /// nothing outside the mask.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn laplacian_scalar(
        arr: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        coeff: f64,
        mut out: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let s = |a: usize, b: usize| a * lyu + b;

        let c = coeff / 6.0;
        let v = c
            * (-20.0 * arr[idx]
                + 4.0
                    * (arr[s(xup, y)] + arr[s(xdn, y)] + arr[s(x, yup)] + arr[s(x, ydn)])
                + arr[s(xup, yup)]
                + arr[s(xup, ydn)]
                + arr[s(xdn, yup)]
                + arr[s(xdn, ydn)]);

        if let Some(slot) = out.get_mut(tid) {
            *slot = v;
        }
    }

    /// The same stencil applied to each component of a 2-vector field.
    ///
    /// Ports `volterra_fd::ops::laplacian_vector`. One thread per cell writes
    /// both components, so the neighbour index arithmetic is shared rather than
    /// repeated per component.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn laplacian_vector(
        arr: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        coeff: f64,
        mut out: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        let k = coeff / 6.0;
        let mut result = [0.0_f64; 2];
        for c in 0..2 {
            result[c] = k
                * (-20.0 * arr[v(x, y, c)]
                    + 4.0
                        * (arr[v(xup, y, c)]
                            + arr[v(xdn, y, c)]
                            + arr[v(x, yup, c)]
                            + arr[v(x, ydn, c)])
                    + arr[v(xup, yup, c)]
                    + arr[v(xup, ydn, c)]
                    + arr[v(xdn, yup, c)]
                    + arr[v(xdn, ydn, c)]);
        }

        if let Some(slot) = out.get_mut(tid) {
            *slot = result;
        }
    }

    /// Central-difference divergence of a 2-vector field.
    ///
    /// Ports `volterra_fd::ops::div_vector`.
    #[kernel]
    pub fn div_vector(
        arr: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        let d = 0.5
            * ((arr[v(xup, y, 0)] - arr[v(xdn, y, 0)])
                + (arr[v(x, yup, 1)] - arr[v(x, ydn, 1)]));

        if let Some(slot) = out.get_mut(tid) {
            *slot = d;
        }
    }

    /// Second-order upwind advection, accumulated into `out`.
    ///
    /// Ports `volterra_fd::ops::upwind_advective_term`: adds
    /// `coeff * (u . grad) arr` to `out`, choosing the one-sided three-point
    /// stencil on each axis by the sign of that axis's velocity component. The
    /// branch is on `u > 0` exactly as the CPU writes it, so a zero velocity
    /// takes the same side on both.
    ///
    /// Accumulates, so `out` carries whatever the caller left in it.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn upwind_advective(
        u: &[f64],
        arr: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        coeff: f64,
        mut out: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let xupup = (x + 2) % lxu;
        let xdndn = (x + lxu - 2) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let yupup = (y + 2) % lyu;
        let ydndn = (y + lyu - 2) % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        let half = coeff * 0.5;
        let ux = u[v(x, y, 0)];
        let uy = u[v(x, y, 1)];
        let tmp_x = half * ux;
        let tmp_y = half * uy;

        // The x contribution lands first and the y contribution second, each
        // added to the running value, which is the order the CPU writes as two
        // separate `out[...] +=` statements. Summing the two terms before
        // adding them associates differently and lands one ulp away.
        if let Some(slot) = out.get_mut(tid) {
            for c in 0..2 {
                slot[c] += if ux > 0.0 {
                    tmp_x * (3.0 * arr[v(x, y, c)] - 4.0 * arr[v(xdn, y, c)] + arr[v(xdndn, y, c)])
                } else {
                    tmp_x * (-3.0 * arr[v(x, y, c)] + 4.0 * arr[v(xup, y, c)] - arr[v(xupup, y, c)])
                };
            }
            for c in 0..2 {
                slot[c] += if uy > 0.0 {
                    tmp_y * (3.0 * arr[v(x, y, c)] - 4.0 * arr[v(x, ydn, c)] + arr[v(x, ydndn, c)])
                } else {
                    tmp_y * (-3.0 * arr[v(x, y, c)] + 4.0 * arr[v(x, yup, c)] - arr[v(x, yupup, c)])
                };
            }
        }
    }

    /// The molecular field `H` and the co-rotation tensor `S`, in one pass.
    ///
    /// Ports `volterra_fd::nematic::h_s_from_q`, which the CPU runs as two
    /// passes: `laplacian_vector(q, h, K)` and then a per-cell correction that
    /// subtracts the bulk Landau-de Gennes term and writes `S`. Both touch only
    /// interior cells and neither reads what the other writes, so one thread
    /// per cell does both and reads `q` once instead of twice.
    ///
    /// `h` is written only inside the mask, so its exterior keeps whatever the
    /// caller left there, matching the CPU.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn h_s_from_q(
        u: &[f64],
        q: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        a: f64,
        c_coeff: f64,
        k: f64,
        lambda: f64,
        mut h: DisjointSlice<[f64; 2]>,
        mut s: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        // H = K * lap Q, the 9-point stencil `laplacian_vector` applies.
        let kk = k / 6.0;
        let mut hv = [0.0_f64; 2];
        for c in 0..2 {
            hv[c] = kk
                * (-20.0 * q[v(x, y, c)]
                    + 4.0
                        * (q[v(xup, y, c)]
                            + q[v(xdn, y, c)]
                            + q[v(x, yup, c)]
                            + q[v(x, ydn, c)])
                    + q[v(xup, yup, c)]
                    + q[v(xup, ydn, c)]
                    + q[v(xdn, yup, c)]
                    + q[v(xdn, ydn, c)]);
        }

        let q0 = q[v(x, y, 0)];
        let q1 = q[v(x, y, 1)];
        let trqsq = 2.0 * (q0 * q0 + q1 * q1);
        hv[0] -= (a + c_coeff * trqsq) * q0;
        hv[1] -= (a + c_coeff * trqsq) * q1;

        let dxux = 0.5 * (u[v(xup, y, 0)] - u[v(xdn, y, 0)]);
        let dxuy = 0.5 * (u[v(xup, y, 1)] - u[v(xdn, y, 1)]);
        let dyux = 0.5 * (u[v(x, yup, 0)] - u[v(x, ydn, 0)]);

        let omega_xy = 0.5 * (dxuy - dyux);
        let lambda_s = lambda * (2.0 * trqsq).sqrt();
        let tr_qe = 2.0 * q0 * dxux + q1 * (dyux + dxuy);

        let sv = [
            lambda_s * dxux - 2.0 * omega_xy * q1 - 2.0 * tr_qe * q0,
            lambda_s * 0.5 * (dxuy + dyux) + 2.0 * omega_xy * q0 - 2.0 * tr_qe * q1,
        ];

        if let Some(slot) = h.get_mut(tid) {
            *slot = hv;
        }
        let tid_s = thread::index_1d();
        if let Some(slot) = s.get_mut(tid_s) {
            *slot = sv;
        }
    }

    /// The symmetric and antisymmetric stresses, in one pass.
    ///
    /// Ports `volterra_fd::nematic::calculate_pi`, which the CPU runs as four
    /// passes over the field: `Pi_S = -lambda H - zeta Q` everywhere, minus the
    /// Ericksen stress inside the mask, plus `2 Tr[QH] Q` everywhere, and
    /// `Pi_A = 2 (Q0 H1 - H0 Q1)` everywhere. Each cell's value depends only on
    /// that cell and its neighbours, and the three writes to `Pi_S` happen in
    /// that order, so one thread per cell applies all four in the same order.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_pi(
        h: &[f64],
        q: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        lambda: f64,
        zeta: f64,
        k: f64,
        mut pi_s: DisjointSlice<[f64; 2]>,
        mut pi_a: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        let q0 = q[v(x, y, 0)];
        let q1 = q[v(x, y, 1)];
        let h0 = h[v(x, y, 0)];
        let h1 = h[v(x, y, 1)];

        // Every cell: Pi_S = -lambda H - zeta Q.
        let mut ps = [-lambda * h0 - zeta * q0, -lambda * h1 - zeta * q1];

        // Interior only: subtract the Ericksen elastic stress.
        if inside[idx] != 0 {
            let xup = (x + 1) % lxu;
            let xdn = (x + lxu - 1) % lxu;
            let yup = (y + 1) % lyu;
            let ydn = (y + lyu - 1) % lyu;
            let dxq0 = 0.5 * (q[v(xup, y, 0)] - q[v(xdn, y, 0)]);
            let dxq1 = 0.5 * (q[v(xup, y, 1)] - q[v(xdn, y, 1)]);
            let dyq0 = 0.5 * (q[v(x, yup, 0)] - q[v(x, ydn, 0)]);
            let dyq1 = 0.5 * (q[v(x, yup, 1)] - q[v(x, ydn, 1)]);
            ps[0] -= k * (dxq0 * dxq0 + dxq1 * dxq1 - dyq0 * dyq0 - dyq1 * dyq1);
            ps[1] -= 2.0 * k * (dxq1 * dyq1 + dxq0 * dyq0);
        }

        // Every cell: add 2 Tr[QH] Q.
        let trqh = 2.0 * (q0 * h0 + q1 * h1);
        ps[0] += trqh * q0;
        ps[1] += trqh * q1;

        if let Some(slot) = pi_s.get_mut(tid) {
            *slot = ps;
        }
        let tid_a = thread::index_1d();
        if let Some(slot) = pi_a.get_mut(tid_a) {
            *slot = 2.0 * (q0 * h1 - h0 * q1);
        }
    }

    /// One Jacobi sweep of the 9-point pressure Poisson stencil.
    ///
    /// Ports `volterra_fd::stokes::relax_pressure_inner_loop`. Reads only
    /// `p_aux` and writes only `p`, which is what makes the sweep Jacobi and
    /// what makes one thread per cell safe with no ordering between them.
    #[kernel]
    pub fn jacobi_sweep(
        p_aux: &[f64],
        rhs: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        mut p: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let s = |a: usize, b: usize| a * lyu + b;

        let v = 0.05
            * (-6.0 * rhs[idx]
                + 4.0 * (p_aux[s(xup, y)] + p_aux[s(x, yup)] + p_aux[s(x, ydn)] + p_aux[s(xdn, y)])
                + p_aux[s(xup, yup)]
                + p_aux[s(xup, ydn)]
                + p_aux[s(xdn, yup)]
                + p_aux[s(xdn, ydn)]);

        if let Some(slot) = p.get_mut(tid) {
            *slot = v;
        }
    }

    /// The non-divergence part of the pressure Poisson right-hand side.
    ///
    /// Ports `volterra_fd::stokes::calculate_pressure_terms`, which
    /// accumulates `div F - rho * (d_i u_j)(d_j u_i)` onto `rhs`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn pressure_terms(
        u: &[f64],
        pi_s: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        rho: f64,
        mut rhs: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        let dudx = 0.5 * (u[v(xup, y, 0)] - u[v(xdn, y, 0)]);
        let dvdy = 0.5 * (u[v(x, yup, 1)] - u[v(x, ydn, 1)]);
        let dyux = 0.5 * (u[v(x, yup, 0)] - u[v(x, ydn, 0)]);
        let dxuy = 0.5 * (u[v(xup, y, 1)] - u[v(xdn, y, 1)]);

        let div_f = (pi_s[v(xup, y, 0)] + pi_s[v(xdn, y, 0)]
            - pi_s[v(x, yup, 0)]
            - pi_s[v(x, ydn, 0)])
            + 0.5
                * (pi_s[v(xup, yup, 1)] - pi_s[v(xup, ydn, 1)] - pi_s[v(xdn, yup, 1)]
                    + pi_s[v(xdn, ydn, 1)]);

        let conv = rho * (dudx * dudx + dvdy * dvdy + dyux * 2.0 * dxuy);

        if let Some(slot) = rhs.get_mut(tid) {
            *slot += div_f - conv;
        }
    }

    /// The velocity time derivative, in one pass.
    ///
    /// Ports `volterra_fd::stokes::get_u_update`, which the CPU runs as three
    /// passes accumulating into `dudt`: the viscous term `nu lap u`, the
    /// convective term by upwind advection, and the pressure and stress
    /// gradients. They accumulate in that order, which this keeps.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn u_update(
        u: &[f64],
        p: &[f64],
        pi_s: &[f64],
        pi_a: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        rho: f64,
        nu: f64,
        mut dudt: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        if inside[idx] == 0 {
            return;
        }
        let lxu = lx as usize;
        let lyu = ly as usize;
        let x = idx / lyu;
        let y = idx % lyu;
        let xup = (x + 1) % lxu;
        let xdn = (x + lxu - 1) % lxu;
        let xupup = (x + 2) % lxu;
        let xdndn = (x + lxu - 2) % lxu;
        let yup = (y + 1) % lyu;
        let ydn = (y + lyu - 1) % lyu;
        let yupup = (y + 2) % lyu;
        let ydndn = (y + lyu - 2) % lyu;
        let s = |a: usize, b: usize| a * lyu + b;
        let v = |a: usize, b: usize, c: usize| (a * lyu + b) * 2 + c;

        // 1. viscous, the same 9-point stencil `laplacian_vector` applies.
        let kk = nu / 6.0;
        let mut d = [0.0_f64; 2];
        for c in 0..2 {
            d[c] = kk
                * (-20.0 * u[v(x, y, c)]
                    + 4.0
                        * (u[v(xup, y, c)] + u[v(xdn, y, c)] + u[v(x, yup, c)] + u[v(x, ydn, c)])
                    + u[v(xup, yup, c)]
                    + u[v(xup, ydn, c)]
                    + u[v(xdn, yup, c)]
                    + u[v(xdn, ydn, c)]);
        }

        // 2. convective, upwind with coeff = -1, advecting u by itself. The x
        // contribution lands before the y one, as on the CPU.
        let ux = u[v(x, y, 0)];
        let uy = u[v(x, y, 1)];
        let tmp_x = -0.5 * ux;
        let tmp_y = -0.5 * uy;
        for c in 0..2 {
            d[c] += if ux > 0.0 {
                tmp_x * (3.0 * u[v(x, y, c)] - 4.0 * u[v(xdn, y, c)] + u[v(xdndn, y, c)])
            } else {
                tmp_x * (-3.0 * u[v(x, y, c)] + 4.0 * u[v(xup, y, c)] - u[v(xupup, y, c)])
            };
        }
        for c in 0..2 {
            d[c] += if uy > 0.0 {
                tmp_y * (3.0 * u[v(x, y, c)] - 4.0 * u[v(x, ydn, c)] + u[v(x, ydndn, c)])
            } else {
                tmp_y * (-3.0 * u[v(x, y, c)] + 4.0 * u[v(x, yup, c)] - u[v(x, yupup, c)])
            };
        }

        // 3. pressure and stress gradients.
        let inv_rho = 0.5 / rho;
        d[0] += inv_rho
            * (-(p[s(xup, y)] - p[s(xdn, y)])
                + (pi_s[v(xup, y, 0)] - pi_s[v(xdn, y, 0)])
                + ((pi_s[v(x, yup, 1)] + pi_a[s(x, yup)])
                    - (pi_s[v(x, ydn, 1)] + pi_a[s(x, ydn)])));
        d[1] += inv_rho
            * (-(p[s(x, yup)] - p[s(x, ydn)])
                + ((pi_s[v(xup, y, 1)] - pi_a[s(xup, y)])
                    - (pi_s[v(xdn, y, 1)] - pi_a[s(xdn, y)]))
                - (pi_s[v(x, yup, 0)] - pi_s[v(x, ydn, 0)]));

        if let Some(slot) = dudt.get_mut(tid) {
            *slot = d;
        }
    }

    /// No-slip: zero the velocity on every boundary cell.
    ///
    /// Ports `volterra_fd::bc::apply_u_boundary_conditions`, which sets both
    /// components to zero at any cell carrying a normal in either layer.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_u_bc(
        is_inner: &[u8],
        is_outer: &[u8],
        inner_normals: &[f64],
        outer_normals: &[f64],
        lx: u32,
        ly: u32,
        mut u: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        let inner = is_inner[idx] != 0
            && (inner_normals[idx * 2] != 0.0 || inner_normals[idx * 2 + 1] != 0.0);
        let outer = is_outer[idx] != 0
            && (outer_normals[idx * 2] != 0.0 || outer_normals[idx * 2 + 1] != 0.0);
        if !inner && !outer {
            return;
        }
        if let Some(slot) = u.get_mut(tid) {
            *slot = [0.0, 0.0];
        }
    }

    /// Dirichlet anchoring of Q to the winding tangent.
    ///
    /// Ports `volterra_fd::bc::apply_q_boundary_conditions`. The director
    /// angle is `net_charge` times the polar angle of the outward normal, and
    /// `Q` is the traceless-symmetric form of the tangent to it.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_q_bc(
        is_inner: &[u8],
        is_outer: &[u8],
        inner_normals: &[f64],
        outer_normals: &[f64],
        lx: u32,
        ly: u32,
        s0: f64,
        net_charge: f64,
        mut q: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let n = (lx as usize) * (ly as usize);
        if idx >= n {
            return;
        }
        // The outer layer is tested first so that it wins, matching the CPU's
        // second pass overwriting the first.
        let (nx, ny) = if is_outer[idx] != 0
            && (outer_normals[idx * 2] != 0.0 || outer_normals[idx * 2 + 1] != 0.0)
        {
            (outer_normals[idx * 2], outer_normals[idx * 2 + 1])
        } else if is_inner[idx] != 0
            && (inner_normals[idx * 2] != 0.0 || inner_normals[idx * 2 + 1] != 0.0)
        {
            (inner_normals[idx * 2], inner_normals[idx * 2 + 1])
        } else {
            return;
        };

        // Written out rather than as `nx.clamp(-1.0, 1.0)`: clamp asserts that
        // its bounds are ordered, and a panic path in device code is worth
        // avoiding for a comparison the compiler would fold anyway.
        #[allow(clippy::manual_clamp)]
        let clamped = if nx > 1.0 {
            1.0
        } else if nx < -1.0 {
            -1.0
        } else {
            nx
        };
        let mut theta = clamped.acos();
        if ny < 0.0 {
            theta = 2.0 * core::f64::consts::PI - theta;
        }
        let nnx = (theta * net_charge).cos();
        let nny = (theta * net_charge).sin();

        if let Some(slot) = q.get_mut(tid) {
            *slot = [s0 * (nny * nny - 0.5), s0 * (-nnx * nny)];
        }
    }

    /// The molecular-field boundary condition that holds `dQ/dt` at zero on the
    /// wall.
    ///
    /// Ports `volterra_fd::bc::apply_h_boundary_conditions`. The one-sided
    /// difference runs inward, against the sign of each normal component.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_h_bc(
        q: &[f64],
        u: &[f64],
        s: &[f64],
        is_inner: &[u8],
        is_outer: &[u8],
        inner_normals: &[f64],
        outer_normals: &[f64],
        lx: u32,
        ly: u32,
        gamma: f64,
        mut h: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let lxu = lx as usize;
        let lyu = ly as usize;
        let n = lxu * lyu;
        if idx >= n {
            return;
        }
        let (nx, ny) = if is_outer[idx] != 0
            && (outer_normals[idx * 2] != 0.0 || outer_normals[idx * 2 + 1] != 0.0)
        {
            (outer_normals[idx * 2], outer_normals[idx * 2 + 1])
        } else if is_inner[idx] != 0
            && (inner_normals[idx * 2] != 0.0 || inner_normals[idx * 2 + 1] != 0.0)
        {
            (inner_normals[idx * 2], inner_normals[idx * 2 + 1])
        } else {
            return;
        };

        let x = idx / lyu;
        let y = idx % lyu;
        let sign = |v: f64| -> i64 {
            if v > 0.0 {
                1
            } else if v < 0.0 {
                -1
            } else {
                0
            }
        };
        let wrap = |i: usize, delta: i64, m: usize| -> usize {
            let t = i as i64 + m as i64 + delta;
            (t.rem_euclid(m as i64)) as usize
        };
        let a = sign(nx);
        let b = sign(ny);
        let xa = wrap(x, -a, lxu);
        let yb = wrap(y, -b, lyu);
        let v = |p: usize, r: usize, c: usize| (p * lyu + r) * 2 + c;

        let ux = u[v(x, y, 0)];
        let uy = u[v(x, y, 1)];
        let mut out = [0.0_f64; 2];
        for c in 0..2 {
            let dq_x = q[v(x, y, c)] - q[v(xa, y, c)];
            let dq_y = q[v(x, y, c)] - q[v(x, yb, c)];
            out[c] = gamma * (a as f64 * ux * dq_x + b as f64 * uy * dq_y - s[v(x, y, c)]);
        }

        if let Some(slot) = h.get_mut(tid) {
            *slot = out;
        }
    }

    /// The Neumann pressure boundary condition.
    ///
    /// Ports `volterra_fd::bc::apply_p_boundary_conditions`, including the
    /// degenerate-normal test that leaves a cell untouched when the inward
    /// direction projects to nothing.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_p_bc(
        p_aux: &[f64],
        u: &[f64],
        pi_s: &[f64],
        pi_a: &[f64],
        is_inner: &[u8],
        is_outer: &[u8],
        inner_normals: &[f64],
        outer_normals: &[f64],
        lx: u32,
        ly: u32,
        rho: f64,
        nu: f64,
        mut p: DisjointSlice<f64>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let lxu = lx as usize;
        let lyu = ly as usize;
        let n = lxu * lyu;
        if idx >= n {
            return;
        }
        let (nx, ny) = if is_outer[idx] != 0
            && (outer_normals[idx * 2] != 0.0 || outer_normals[idx * 2 + 1] != 0.0)
        {
            (outer_normals[idx * 2], outer_normals[idx * 2 + 1])
        } else if is_inner[idx] != 0
            && (inner_normals[idx * 2] != 0.0 || inner_normals[idx * 2 + 1] != 0.0)
        {
            (inner_normals[idx * 2], inner_normals[idx * 2 + 1])
        } else {
            return;
        };

        let sign = |v: f64| -> i64 {
            if v > 0.0 {
                1
            } else if v < 0.0 {
                -1
            } else {
                0
            }
        };
        let a = sign(nx);
        let b = sign(ny);
        let denom = a as f64 * nx + b as f64 * ny;
        if denom < 1e-15 && denom > -1e-15 {
            return;
        }

        let x = idx / lyu;
        let y = idx % lyu;
        let wrap = |i: usize, delta: i64, m: usize| -> usize {
            let t = i as i64 + 2 * m as i64 + delta;
            (t.rem_euclid(m as i64)) as usize
        };
        let xa = wrap(x, -a, lxu);
        let xaa = wrap(x, -2 * a, lxu);
        let yb = wrap(y, -b, lyu);
        let ybb = wrap(y, -2 * b, lyu);
        let s = |q: usize, r: usize| q * lyu + r;
        let v = |q: usize, r: usize, c: usize| (q * lyu + r) * 2 + c;

        let fx = a as f64 * (pi_s[v(x, y, 0)] - pi_s[v(xa, y, 0)])
            + b as f64 * (pi_s[v(x, y, 1)] + pi_a[s(x, y)] - pi_s[v(x, yb, 1)] - pi_a[s(x, yb)]);
        let fy = a as f64
            * (pi_s[v(x, y, 1)] - pi_a[s(x, y)] - pi_s[v(xa, y, 1)] + pi_a[s(xa, y)])
            - b as f64 * (pi_s[v(x, y, 0)] - pi_s[v(x, yb, 0)]);

        let lapu0 = 2.0 * u[v(x, y, 0)] - 2.0 * (u[v(xa, y, 0)] + u[v(x, yb, 0)])
            + u[v(xaa, y, 0)]
            + u[v(x, ybb, 0)];
        let lapu1 = 2.0 * u[v(x, y, 1)] - 2.0 * (u[v(xa, y, 1)] + u[v(x, yb, 1)])
            + u[v(xaa, y, 1)]
            + u[v(x, ybb, 1)];

        let n_dot = nx * (fx + rho * nu * lapu0) + ny * (fy + rho * nu * lapu1);
        let p_neighbours = a as f64 * nx * p_aux[s(xa, y)] + b as f64 * ny * p_aux[s(x, yb)];

        if let Some(slot) = p.get_mut(tid) {
            *slot = (n_dot + p_neighbours) / denom;
        }
    }

    /// The Q time derivative, in one pass.
    ///
    /// Ports `volterra_fd::step::get_q_update`: `dQ = H/gamma + S` at every
    /// cell, then minus the upwind advection inside the mask. The first part
    /// carries no mask and the second does, which is why the mask test sits
    /// where it does rather than at the top.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn q_update(
        q: &[f64],
        h: &[f64],
        s: &[f64],
        u: &[f64],
        inside: &[u8],
        lx: u32,
        ly: u32,
        gamma: f64,
        mut dq: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let idx = tid.get();
        let lxu = lx as usize;
        let lyu = ly as usize;
        let n = lxu * lyu;
        if idx >= n {
            return;
        }
        let x = idx / lyu;
        let y = idx % lyu;
        let v = |p: usize, r: usize, c: usize| (p * lyu + r) * 2 + c;

        let inv_gamma = 1.0 / gamma;
        let mut d = [
            inv_gamma * h[v(x, y, 0)] + s[v(x, y, 0)],
            inv_gamma * h[v(x, y, 1)] + s[v(x, y, 1)],
        ];

        if inside[idx] != 0 {
            let xup = (x + 1) % lxu;
            let xdn = (x + lxu - 1) % lxu;
            let xupup = (x + 2) % lxu;
            let xdndn = (x + lxu - 2) % lxu;
            let yup = (y + 1) % lyu;
            let ydn = (y + lyu - 1) % lyu;
            let yupup = (y + 2) % lyu;
            let ydndn = (y + lyu - 2) % lyu;
            let ux = u[v(x, y, 0)];
            let uy = u[v(x, y, 1)];
            let tmp_x = -0.5 * ux;
            let tmp_y = -0.5 * uy;
            for c in 0..2 {
                d[c] += if ux > 0.0 {
                    tmp_x * (3.0 * q[v(x, y, c)] - 4.0 * q[v(xdn, y, c)] + q[v(xdndn, y, c)])
                } else {
                    tmp_x * (-3.0 * q[v(x, y, c)] + 4.0 * q[v(xup, y, c)] - q[v(xupup, y, c)])
                };
            }
            for c in 0..2 {
                d[c] += if uy > 0.0 {
                    tmp_y * (3.0 * q[v(x, y, c)] - 4.0 * q[v(x, ydn, c)] + q[v(x, ydndn, c)])
                } else {
                    tmp_y * (-3.0 * q[v(x, y, c)] + 4.0 * q[v(x, yup, c)] - q[v(x, yupup, c)])
                };
            }
        }

        if let Some(slot) = dq.get_mut(tid) {
            *slot = d;
        }
    }

    /// `field *= scale`, over a scalar field with no mask.
    ///
    /// The pressure right-hand side starts as `div u` and is scaled by
    /// `rho / dt` before the other terms accumulate onto it.
    #[kernel]
    pub fn scale_scalar(len: u32, scale: f64, mut field: DisjointSlice<f64>) {
        let tid = thread::index_1d();
        let j = tid.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = field.get_mut(tid) {
            *slot *= scale;
        }
    }

    /// Copy, for the Jacobi double buffer.
    #[kernel]
    pub fn copy_scalar(src: &[f64], len: u32, mut dst: DisjointSlice<f64>) {
        let tid = thread::index_1d();
        let j = tid.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = dst.get_mut(tid) {
            *slot = src[j];
        }
    }

    /// Partial sums of `|p_aux - p|` and of `p_aux`, one pair per stride block.
    ///
    /// The pressure loop stops when `sum|p_aux - p| / (1e-7 + sum p_aux)` falls
    /// under its target, so the sweep count depends on this sum. Accumulating
    /// it through device atomics would make the order of accumulation, and so
    /// the sweep count, vary between runs of the same binary. Each thread
    /// instead owns a fixed, contiguous span of the grid and sums it in index
    /// order into its own slot, and the caller adds those slots up in index
    /// order too, so the result is the same every time.
    ///
    /// The convergence measure's two sums, one pair per span.
    ///
    /// The pressure loop stops when `sum|p_aux - p| / (1e-7 + sum p_aux)` falls
    /// under its target, so the sweep count depends on this sum. Accumulating
    /// it through device atomics would make the order of accumulation, and so
    /// the sweep count, vary between runs of the same binary. Each thread
    /// instead owns a fixed, contiguous span of the grid and sums it in index
    /// order into its own slot, and the caller adds those slots up in index
    /// order too, so the result is the same every time. It is a different
    /// association from the CPU's element-by-element sum, not a random one.
    ///
    /// Both sums go in one 2-wide slot so the host reads them back in a single
    /// transfer. That transfer is the only synchronisation inside a step, and
    /// two of them cost about as much as everything else the step does.
    #[kernel]
    pub fn pressure_partials(
        p: &[f64],
        p_aux: &[f64],
        len: u32,
        span: u32,
        n_blocks: u32,
        mut partials: DisjointSlice<[f64; 2]>,
    ) {
        let tid = thread::index_1d();
        let b = tid.get();
        if b >= n_blocks as usize {
            return;
        }
        let start = b * (span as usize);
        let mut end = start + (span as usize);
        if end > len as usize {
            end = len as usize;
        }
        let mut diff = 0.0_f64;
        let mut old = 0.0_f64;
        let mut i = start;
        while i < end {
            diff += (p_aux[i] - p[i]).abs();
            old += p_aux[i];
            i += 1;
        }
        if let Some(slot) = partials.get_mut(tid) {
            *slot = [diff, old];
        }
    }

    /// `field += dt * rate`, over a 2-vector field with no mask.
    ///
    /// Ports both integrate steps of `update_step_inner`, which advance `Q` by
    /// `dQ` and `u` by `du/dt` over every cell.
    #[kernel]
    pub fn integrate(rate: &[f64], len: u32, dt: f64, mut field: DisjointSlice<f64>) {
        let tid = thread::index_1d();
        let j = tid.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = field.get_mut(tid) {
            *slot += dt * rate[j];
        }
    }
}

// `#[cuda_module]` generates its `load`/`LoadedModule` inside the `kernels`
// module above, one level deeper than this file's own `kernels` module.
// Flatten that one level so `crate::kernels::{load, LoadedModule}` is the path
// callers use, matching `volterra-cuda`.
pub use kernels::{LoadedModule, load};
