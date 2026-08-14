//! Double-precision CUDA kernels for the passive Landau-de Gennes molecular
//! field and the FIRE velocity-Verlet update, compiled to PTX by
//! `rustc-codegen-cuda` (see `cartan-cuda` for the pattern this follows).
//!
//! ## Layout
//!
//! Q, V (FIRE velocity) and F (force = `gamma_r * H`, the same quantity
//! `volterra_solver::beris_edwards_rhs_3d_par_dry` computes on the CPU) are
//! all flat `f64` buffers of length `n_sites * 5`, component-fastest:
//! `buf[site * 5 + c]`, `c` in `[q11, q12, q13, q22, q23]` -- the same layout
//! `QField3D::q` uses once flattened. `trq2` is one `f64` per site.
//!
//! ## Kernel split
//!
//! Two kernels compute the force, following the same reduction-then-apply
//! split `cartan-cuda` uses for `sphere_exp`: `trq2` is an O(1)-per-site
//! reduction (5 reads, no stencil), kept out of the per-component kernel so
//! it runs once per site rather than once per component; `force` then reads
//! it back alongside the 6-neighbour stencil. Fusing them would repeat the
//! `Tr(Q^2)` computation 5 times per site for no benefit, since it does not
//! depend on which component is being written.
//!
//! `reduce_fire` is the FIRE reduction proper (`sum|f|^2`, `sum|v|^2`,
//! `sum f.v`): one thread per site computes its local 5-component
//! contribution, a full-warp tree reduction (`warp::reduce_sum_f64`) sums
//! within the warp, and lane 0 folds the warp's result into three
//! device-scope `f64` atomics. This is the piece a first GPU port most often
//! treats as an afterthought; here it is a first-class kernel for exactly
//! that reason.
//!
//! ## Bounds
//!
//! `LaunchConfig::for_num_elems` rounds the grid up to a whole number of
//! 256-thread blocks, so a launch generally has more threads than elements
//! whenever the element count is not itself a multiple of 256 (true for
//! every grid size used here). Writes go through `DisjointSlice::get_mut`,
//! which self-guards; every plain `&[f64]` **read** is guarded explicitly
//! against the passed-in element count before indexing, since a plain slice
//! index has no such protection in device code.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `Tr(Q^2)` at every site, one thread per site.
    #[kernel]
    pub fn trq2(q: &[f64], n_sites: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let s = idx.get();
        if s >= n_sites as usize {
            return;
        }
        let b = s * 5;
        let q11 = q[b];
        let q12 = q[b + 1];
        let q13 = q[b + 2];
        let q22 = q[b + 3];
        let q23 = q[b + 4];
        let q33 = -(q11 + q22);
        let tr = q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        if let Some(slot) = out.get_mut(idx) {
            *slot = tr;
        }
    }

    /// The fused 6-point Laplacian stencil plus bulk Landau-de Gennes terms,
    /// scaled by `gamma_r`: `F = gamma_r * (k_r * lap(Q) + bulk * Q)`, `bulk
    /// = -a_eff - 2 c Tr(Q^2)`. One thread per (site, component), matching
    /// `volterra_solver::mol_field_3d::molecular_field_3d_par`'s CPU kernel
    /// exactly (same periodic 6-neighbour stencil, same bulk formula).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn force(
        q: &[f64],
        trq2: &[f64],
        nx: u32,
        ny: u32,
        nz: u32,
        a_eff: f64,
        c_landau: f64,
        k_r: f64,
        gamma_r: f64,
        inv_dx2: f64,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let n_sites = (nx as usize) * (ny as usize) * (nz as usize);
        if j >= n_sites * 5 {
            return;
        }
        let s = j / 5;
        let c = j % 5;

        let nxu = nx as usize;
        let nyu = ny as usize;
        let nzu = nz as usize;
        let nynz = nyu * nzu;

        let l = s % nzu;
        let ij = s / nzu;
        let jc = ij % nyu;
        let i = ij / nyu;

        let ip = ((i + 1) % nxu) * nynz + jc * nzu + l;
        let im = ((i + nxu - 1) % nxu) * nynz + jc * nzu + l;
        let jp = i * nynz + ((jc + 1) % nyu) * nzu + l;
        let jm = i * nynz + ((jc + nyu - 1) % nyu) * nzu + l;
        let lp = i * nynz + jc * nzu + (l + 1) % nzu;
        let lm = i * nynz + jc * nzu + (l + nzu - 1) % nzu;

        let qk = q[j];
        let lap = (q[ip * 5 + c]
            + q[im * 5 + c]
            + q[jp * 5 + c]
            + q[jm * 5 + c]
            + q[lp * 5 + c]
            + q[lm * 5 + c]
            - 6.0 * qk)
            * inv_dx2;
        let bulk = -a_eff - 2.0 * c_landau * trq2[s];
        let val = gamma_r * (k_r * lap + bulk * qk);

        if let Some(slot) = out.get_mut(idx) {
            *slot = val;
        }
    }

    /// In-place `y += alpha * x` over `len` elements. Used for both FIRE
    /// half-kicks (`v += 0.5 dt f`) with a plain scalar `alpha`.
    #[kernel]
    pub fn axpy_inplace(x: &[f64], alpha: f64, len: u32, mut y: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = y.get_mut(idx) {
            *slot += alpha * x[j];
        }
    }

    /// In-place velocity-Verlet position update:
    /// `q += dt * v_old + 0.5 * dt^2 * f_old`.
    #[kernel]
    pub fn position_update(v_old: &[f64], f_old: &[f64], dt: f64, len: u32, mut q: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = q.get_mut(idx) {
            *slot += dt * v_old[j] + 0.5 * dt * dt * f_old[j];
        }
    }

    /// In-place FIRE velocity mix: `v = (1 - alpha) v + alpha * scaling * f`.
    #[kernel]
    pub fn fire_mix(f: &[f64], scaling: f64, alpha: f64, len: u32, mut v: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = v.get_mut(idx) {
            *slot = (1.0 - alpha) * (*slot) + alpha * scaling * f[j];
        }
    }

    /// In-place zero, used on a FIRE reset (`v = 0`).
    #[kernel]
    pub fn zero_field(len: u32, mut v: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = v.get_mut(idx) {
            *slot = 0.0;
        }
    }

    /// The FIRE reduction: `sum|f|^2`, `sum|v|^2`, `sum f.v`, one thread per
    /// site (a site's 5 components are summed locally, then a full-warp tree
    /// reduction, then lane 0 folds into the three device-scope accumulators).
    ///
    /// Callers must zero the three accumulators before each launch.
    #[kernel]
    pub fn reduce_fire(
        f: &[f64],
        v: &[f64],
        n_sites: u32,
        force_acc: &[DeviceAtomicF64],
        vel_acc: &[DeviceAtomicF64],
        power_acc: &[DeviceAtomicF64],
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        let n = n_sites as usize;

        let mut fs = 0.0f64;
        let mut vs = 0.0f64;
        let mut ps = 0.0f64;
        if s < n {
            let b = s * 5;
            for c in 0..5 {
                let fc = f[b + c];
                let vc = v[b + c];
                fs += fc * fc;
                vs += vc * vc;
                ps += fc * vc;
            }
        }

        let fs_w = warp::reduce_sum_f64(fs);
        let vs_w = warp::reduce_sum_f64(vs);
        let ps_w = warp::reduce_sum_f64(ps);

        if warp::lane_id() == 0 {
            force_acc[0].fetch_add(fs_w, AtomicOrdering::Relaxed);
            vel_acc[0].fetch_add(vs_w, AtomicOrdering::Relaxed);
            power_acc[0].fetch_add(ps_w, AtomicOrdering::Relaxed);
        }
    }
}

// `#[cuda_module]` generates its `load`/`LoadedModule` inside the `kernels`
// module above, one level deeper than this file's own `kernels` module
// (declared via `mod kernels;` in `lib.rs`). Flatten that one level so
// `crate::kernels::{load, LoadedModule}` is the path callers use, matching
// `cartan-cuda`'s (which declares its module inline in `lib.rs` and so has
// no extra nesting to flatten).
pub use kernels::{load, LoadedModule};
