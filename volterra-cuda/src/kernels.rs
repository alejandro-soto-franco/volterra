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
//! ## Kernel split, and the fused alternative
//!
//! `Device::fire_minimize` uses two kernels for the force, following the
//! same reduction-then-apply split `cartan-cuda` uses for `sphere_exp`:
//! `trq2` is an O(1)-per-site reduction (5 reads, no stencil), kept out of
//! the per-component kernel so it runs once per site rather than once per
//! component; `force` then reads it back alongside the 6-neighbour stencil,
//! one thread per (site, component).
//!
//! `force_fused_aos` folds both into one thread per **site** instead,
//! writing all 5 components through `DisjointSlice<[f64; 5]>`. This is
//! `BENCHMARKS.md`'s answer to "fuse the two passes": a *naive* fusion that
//! keeps one thread per (site, component) and has each of the 5 threads
//! independently re-read all 5 of a site's own components to compute
//! `Tr(Q^2)` moves *more* traffic than the split design (`480n` bytes/site
//! against `408n`); `force_fused_aos` shares that read once per site the way
//! the split design already does, and additionally shares the six
//! neighbours' index arithmetic across a site's 5 components instead of
//! repeating it 5 times. `force_fused_soa` is the same computation again
//! over 5 separate component planes, to answer the memory-layout question
//! independently of the fusion question. Neither fused kernel is wired into
//! `Device::fire_minimize` yet; both compile, and their arithmetic is
//! checked against the CPU reference (`volterra_solver::mol_field_3d`) in
//! this crate's tests, ready for a GPU timing run to decide whether either
//! earns the switch.
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
//! which bounds-checks itself; every plain `&[f64]` **read** is bounds-checked
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
        b_landau: f64,
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

        // Cubic bulk term (see `volterra_solver::mol_field_3d`'s module
        // header for the derivation): not proportional to `qk` like the a/c
        // terms above, so this thread reads its own site's other 4
        // components to form it, same as `trq2` already does once per site.
        let b = s * 5;
        let q11 = q[b];
        let q12 = q[b + 1];
        let q13 = q[b + 2];
        let q22 = q[b + 3];
        let q23 = q[b + 4];
        let q33 = -(q11 + q22);
        let qq = [
            q11 * q11 + q12 * q12 + q13 * q13,
            q11 * q12 + q12 * q22 + q13 * q23,
            q11 * q13 + q12 * q23 + q13 * q33,
            q12 * q12 + q22 * q22 + q23 * q23,
            q12 * q13 + q22 * q23 + q23 * q33,
        ];
        let trace_add = if c == 0 || c == 3 { trq2[s] } else { 0.0 };
        let h_cubic = -3.0 * b_landau * qq[c] + b_landau * trace_add;

        let val = gamma_r * (k_r * lap + bulk * qk + h_cubic);

        if let Some(slot) = out.get_mut(idx) {
            *slot = val;
        }
    }

    /// `trq2` and `force` fused into one thread per **site** (not per
    /// component), writing all 5 output components through
    /// `DisjointSlice<[f64; 5]>` -- no `unsafe`, since each thread's 5-wide
    /// slot is exactly one `DisjointSlice` element. This is the "properly
    /// fused" design characterised in `BENCHMARKS.md` (the trivial fusion,
    /// one thread per (site, component) each recomputing `Tr(Q^2)` from a
    /// redundant 5-read, moves *more* traffic than the split design, not
    /// less; this one shares the read once per site the way the split
    /// design already does, and additionally shares the six neighbour
    /// reads' index arithmetic across all 5 components of a thread instead
    /// of repeating it 5 times across 5 threads).
    ///
    /// AoS input, matching `QField3D`'s own layout (`q[site*5+c]`). See
    /// `force_fused_soa` for the same computation over 5 separate planes.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn force_fused_aos(
        q: &[f64],
        nx: u32,
        ny: u32,
        nz: u32,
        a_eff: f64,
        b_landau: f64,
        c_landau: f64,
        k_r: f64,
        gamma_r: f64,
        inv_dx2: f64,
        mut out: DisjointSlice<[f64; 5]>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        let n_sites = (nx as usize) * (ny as usize) * (nz as usize);
        if s >= n_sites {
            return;
        }

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

        let b = s * 5;
        let q11 = q[b];
        let q12 = q[b + 1];
        let q13 = q[b + 2];
        let q22 = q[b + 3];
        let q23 = q[b + 4];
        let q33 = -(q11 + q22);
        let trq2 = q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        let bulk = -a_eff - 2.0 * c_landau * trq2;

        // Cubic bulk term (see `volterra_solver::mol_field_3d`'s module
        // header for the derivation): `H_cubic = -3b Q^2 + b Tr(Q^2) I`.
        let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
        let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
        let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
        let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
        let qq23 = q12 * q13 + q22 * q23 + q23 * q33;
        let h_cubic = [
            -3.0 * b_landau * qq11 + b_landau * trq2,
            -3.0 * b_landau * qq12,
            -3.0 * b_landau * qq13,
            -3.0 * b_landau * qq22 + b_landau * trq2,
            -3.0 * b_landau * qq23,
        ];

        let mut result = [0.0_f64; 5];
        for c in 0..5 {
            let lap = (q[ip * 5 + c]
                + q[im * 5 + c]
                + q[jp * 5 + c]
                + q[jm * 5 + c]
                + q[lp * 5 + c]
                + q[lm * 5 + c]
                - 6.0 * q[b + c])
                * inv_dx2;
            result[c] = gamma_r * (k_r * lap + bulk * q[b + c] + h_cubic[c]);
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = result;
        }
    }

    /// Same computation as `force_fused_aos`, over 5 separate component
    /// planes (structure-of-arrays) instead of one interleaved buffer. Exists
    /// to answer the memory-layout question the split/fused-AoS kernels
    /// cannot: whether AoS's interleaving costs coalescing efficiency in the
    /// neighbour-stencil reads. Not wired into `Device::fire_minimize`
    /// (which is AoS end to end, matching `QField3D`); reachable via
    /// `Device::force_soa` for a direct, controlled, same-formula
    /// measurement once the GPU is free (see
    /// `BENCHMARKS.md`'s "Memory layout" section).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn force_fused_soa(
        q11p: &[f64],
        q12p: &[f64],
        q13p: &[f64],
        q22p: &[f64],
        q23p: &[f64],
        nx: u32,
        ny: u32,
        nz: u32,
        a_eff: f64,
        b_landau: f64,
        c_landau: f64,
        k_r: f64,
        gamma_r: f64,
        inv_dx2: f64,
        mut out11: DisjointSlice<f64>,
        mut out12: DisjointSlice<f64>,
        mut out13: DisjointSlice<f64>,
        mut out22: DisjointSlice<f64>,
        mut out23: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        let n_sites = (nx as usize) * (ny as usize) * (nz as usize);
        if s >= n_sites {
            return;
        }

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

        let q11 = q11p[s];
        let q12 = q12p[s];
        let q13 = q13p[s];
        let q22 = q22p[s];
        let q23 = q23p[s];
        let q33 = -(q11 + q22);
        let trq2 = q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        let bulk = -a_eff - 2.0 * c_landau * trq2;

        // Cubic bulk term, same formula as `force_fused_aos`.
        let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
        let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
        let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
        let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
        let qq23 = q12 * q13 + q22 * q23 + q23 * q33;
        let h11 = -3.0 * b_landau * qq11 + b_landau * trq2;
        let h12 = -3.0 * b_landau * qq12;
        let h13 = -3.0 * b_landau * qq13;
        let h22 = -3.0 * b_landau * qq22 + b_landau * trq2;
        let h23 = -3.0 * b_landau * qq23;

        #[inline(always)]
        fn stencil(
            plane: &[f64],
            ip: usize,
            im: usize,
            jp: usize,
            jm: usize,
            lp: usize,
            lm: usize,
            centre: f64,
            inv_dx2: f64,
        ) -> f64 {
            (plane[ip] + plane[im] + plane[jp] + plane[jm] + plane[lp] + plane[lm] - 6.0 * centre)
                * inv_dx2
        }

        let lap11 = stencil(q11p, ip, im, jp, jm, lp, lm, q11, inv_dx2);
        let lap12 = stencil(q12p, ip, im, jp, jm, lp, lm, q12, inv_dx2);
        let lap13 = stencil(q13p, ip, im, jp, jm, lp, lm, q13, inv_dx2);
        let lap22 = stencil(q22p, ip, im, jp, jm, lp, lm, q22, inv_dx2);
        let lap23 = stencil(q23p, ip, im, jp, jm, lp, lm, q23, inv_dx2);

        // `ThreadIndex` is deliberately `!Copy` (it is a per-access witness,
        // not a value to be squirrelled away), so each of the 5 output
        // planes mints its own from `thread::index_1d()` rather than reusing
        // `idx` above. The hardware registers it reads
        // (`threadIdx`/`blockIdx`/`blockDim`) do not change during a kernel
        // invocation, so every mint resolves to the same site `s`.
        if let Some(slot) = out11.get_mut(thread::index_1d()) {
            *slot = gamma_r * (k_r * lap11 + bulk * q11 + h11);
        }
        if let Some(slot) = out12.get_mut(thread::index_1d()) {
            *slot = gamma_r * (k_r * lap12 + bulk * q12 + h12);
        }
        if let Some(slot) = out13.get_mut(thread::index_1d()) {
            *slot = gamma_r * (k_r * lap13 + bulk * q13 + h13);
        }
        if let Some(slot) = out22.get_mut(thread::index_1d()) {
            *slot = gamma_r * (k_r * lap22 + bulk * q22 + h22);
        }
        if let Some(slot) = out23.get_mut(thread::index_1d()) {
            *slot = gamma_r * (k_r * lap23 + bulk * q23 + h23);
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

    /// The FIRE velocity mix, the position update and the first half-kick in
    /// one pass, one thread per site.
    ///
    /// `fire_mix`, `position_update` and `axpy_inplace` run back to back with
    /// nothing between them that touches `q`, `v` or `f`, so they read the same
    /// three fields three times over. Together they move `400n` bytes; this
    /// moves `200n`. The mix belongs to the previous iteration and the other two
    /// to this one, so the fusion crosses the loop boundary: the caller carries
    /// the mix coefficients forward.
    ///
    /// `mix_v` and `mix_f` are `1 - alpha` and `alpha * scaling`, and both zero
    /// on a reset iteration, where the mix used to be followed by `zero_field`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fire_advance(
        f: &[f64],
        mix_v: f64,
        mix_f: f64,
        dt: f64,
        len: u32,
        mut q: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        let half_dt = 0.5 * dt;
        let half_dt2 = half_dt * dt;

        // One index token per slice: a token is consumed by the slice it
        // indexes, and each thread owns element `j` of `q` and of `v`
        // independently.
        let idx_v = thread::index_1d();
        if let (Some(qj), Some(vj)) = (q.get_mut(idx), v.get_mut(idx_v)) {
            let fj = f[j];
            let vm = mix_v * (*vj) + mix_f * fj;
            *qj += dt * vm + half_dt2 * fj;
            *vj = vm + half_dt * fj;
        }
    }

    /// The second half-kick and the FIRE reduction in one pass, one thread per
    /// site.
    ///
    /// The reduction reads exactly the `f` and `v` the half-kick just touched,
    /// so running them separately reads both fields twice: `200n` bytes against
    /// this kernel's `120n`. The reduction itself is unchanged from
    /// [`reduce_fire`], including the warp tree and the three device-scope
    /// accumulators, which the caller must still zero before each launch.
    #[kernel]
    pub fn fire_kick_reduce(
        f: &[f64],
        half_dt: f64,
        n_sites: u32,
        mut v: DisjointSlice<[f64; 5]>,
        acc: &[DeviceAtomicF64],
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        let n = n_sites as usize;

        let mut fs = 0.0f64;
        let mut vs = 0.0f64;
        let mut ps = 0.0f64;
        if s < n {
            let b = s * 5;
            if let Some(vv) = v.get_mut(idx) {
                for c in 0..5 {
                    let fc = f[b + c];
                    let vc = vv[c] + half_dt * fc;
                    vv[c] = vc;
                    fs += fc * fc;
                    vs += vc * vc;
                    ps += fc * vc;
                }
            }
        }

        // Every lane reaches the warp reduction, including the out-of-range
        // ones, which contribute zero.
        let fs_w = warp::reduce_sum_f64(fs);
        let vs_w = warp::reduce_sum_f64(vs);
        let ps_w = warp::reduce_sum_f64(ps);

        if warp::lane_id() == 0 {
            acc[0].fetch_add(fs_w, AtomicOrdering::Relaxed);
            acc[1].fetch_add(vs_w, AtomicOrdering::Relaxed);
            acc[2].fetch_add(ps_w, AtomicOrdering::Relaxed);
        }
    }

    /// The FIRE velocity mix in the coefficient form `v = mix_v * v + mix_f * f`.
    ///
    /// Same arithmetic as [`fire_mix`] with `mix_v = 1 - alpha` and
    /// `mix_f = alpha * scaling`, and `v = 0` at both coefficients zero, which
    /// is what a reset does. The split bookkeeping path uses it so that it and
    /// the fused path defer the mix by the same amount and the fusion is the
    /// only difference between them.
    #[kernel]
    pub fn fire_mix_coeffs(f: &[f64], mix_v: f64, mix_f: f64, len: u32, mut v: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = v.get_mut(idx) {
            *slot = mix_v * (*slot) + mix_f * f[j];
        }
    }

    /// Pure copy, `y[i] = x[i]`, one read and one write per element and
    /// nothing else: the roofline probe. Used to measure this device's
    /// *achieved* bandwidth (not its spec-sheet peak) so the FIRE kernels'
    /// data volume can be turned into a floor on wall-clock time, rather
    /// than optimised against a number nobody measured.
    #[kernel]
    pub fn bandwidth_copy(x: &[f64], len: u32, mut y: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= len as usize {
            return;
        }
        if let Some(slot) = y.get_mut(idx) {
            *slot = x[j];
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
    /// The three accumulators are one 3-element buffer (`acc[0]=sum|f|^2`,
    /// `acc[1]=sum|v|^2`, `acc[2]=sum f.v`), not three separate one-element
    /// buffers: three independent `to_host_vec` calls per FIRE iteration
    /// (three small D2H copies, each with its own implicit stream sync) cost
    /// more in host round-trip latency than the reduction itself, measured
    /// directly with `nsys` (see `BENCHMARKS.md`'s "Optimisations tried"). One
    /// buffer, one launch, one read-back removes that.
    ///
    /// Callers must zero the accumulator before each launch.
    #[kernel]
    pub fn reduce_fire(f: &[f64], v: &[f64], n_sites: u32, acc: &[DeviceAtomicF64]) {
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
            acc[0].fetch_add(fs_w, AtomicOrdering::Relaxed);
            acc[1].fetch_add(vs_w, AtomicOrdering::Relaxed);
            acc[2].fetch_add(ps_w, AtomicOrdering::Relaxed);
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
