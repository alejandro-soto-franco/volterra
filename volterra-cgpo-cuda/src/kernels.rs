//! Double-precision CUDA kernels for volterra's 2D confined active-nematic
//! solver, compiled to PTX by `rustc-codegen-cuda` (the pattern `cartan-cuda`
//! and `volterra-cuda` establish).
//!
//! ## Layout
//!
//! The CPU side stores a scalar field as `s[x * ly + y]` and a 2-vector field
//! as `v[(x * ly + y) * 2 + c]`, `c` in `{0, 1}` (`volterra_cgpo::index`). The
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
    /// Ports `volterra_cgpo::ops::laplacian`, including its `/6` normalisation
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
    /// Ports `volterra_cgpo::ops::laplacian_vector`. One thread per cell writes
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
    /// Ports `volterra_cgpo::ops::div_vector`.
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
    /// Ports `volterra_cgpo::ops::upwind_advective_term`: adds
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
}

// `#[cuda_module]` generates its `load`/`LoadedModule` inside the `kernels`
// module above, one level deeper than this file's own `kernels` module.
// Flatten that one level so `crate::kernels::{load, LoadedModule}` is the path
// callers use, matching `volterra-cuda`.
pub use kernels::{LoadedModule, load};
