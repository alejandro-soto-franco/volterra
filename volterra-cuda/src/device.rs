//! `Device`: an open CUDA context holding Q, V (FIRE velocity), F (force)
//! and the reduction accumulators, running the same FIRE algorithm as
//! `volterra_solver::fire` (see that module's doc comment for the algorithm
//! and the open-Qmin correspondence) with the elementwise and stencil work
//! on the GPU and the small adaptive-timestep bookkeeping on the host,
//! exactly mirroring `volterra_solver::fire::FireState`.

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::DeviceAtomicF64;

use crate::error::CudaError;
use crate::kernels;

/// FIRE tuning parameters. Field-for-field identical to
/// `volterra_solver::fire::FireParams`, duplicated here so this crate does
/// not need to compile `volterra-solver` through the CUDA host toolchain for
/// its own sake beyond the CPU cross-check binary, which already depends on
/// it directly.
#[derive(Debug, Clone, Copy)]
pub struct FireParams {
    pub delta_t: f64,
    pub delta_t_min: f64,
    pub delta_t_max: f64,
    pub delta_t_inc: f64,
    pub delta_t_dec: f64,
    pub alpha_start: f64,
    pub alpha_dec: f64,
    pub alpha_min: f64,
    pub n_min: i32,
    pub force_cutoff: f64,
    pub max_iterations: usize,
}

impl FireParams {
    pub fn open_qmin_defaults(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t,
            delta_t_min: delta_t * 0.01,
            delta_t_max: 100.0 * delta_t,
            delta_t_inc: 1.1,
            delta_t_dec: 0.95,
            alpha_start: 0.99,
            alpha_dec: 0.9,
            alpha_min: 0.0,
            n_min: 4,
            force_cutoff,
            max_iterations,
        }
    }

    /// Retuned for volterra's own energy landscape, matching
    /// `volterra_solver::fire::FireParams::volterra_tuned` exactly (see that
    /// constructor's doc comment for the sweep it comes from).
    pub fn volterra_tuned(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t_inc: 1.6,
            alpha_dec: 0.7,
            n_min: 0,
            ..Self::open_qmin_defaults(delta_t, force_cutoff, max_iterations)
        }
    }

    /// Retuned for the matched-physics landscape, matching
    /// `volterra_solver::fire::FireParams::matched_tuned` exactly (see that
    /// constructor's doc comment for the sweep it comes from).
    pub fn matched_tuned(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t_inc: 2.5,
            alpha_dec: 0.7,
            n_min: 0,
            ..Self::open_qmin_defaults(delta_t, force_cutoff, max_iterations)
        }
    }
}

/// Physical parameters the force kernel needs (the passive-dry subset of
/// `ActiveNematicParams3D`).
#[derive(Debug, Clone, Copy)]
pub struct LdgParams {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dx: f64,
    pub a_eff: f64,
    /// Cubic bulk coefficient. `0.0` reproduces every result measured before
    /// this term existed exactly, bit for bit (see
    /// `volterra_solver::mol_field_3d`'s module header for the derivation).
    pub b_landau: f64,
    pub c_landau: f64,
    pub k_r: f64,
    pub gamma_r: f64,
}

pub struct FireResult {
    pub q: Vec<f64>,
    pub iterations: usize,
    pub force_max: f64,
    pub converged: bool,
}

pub struct Device {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Self {
            _ctx: ctx,
            stream,
            module,
        })
    }

    /// Run FIRE to convergence (or `params.max_iterations`), starting from
    /// `q0` (length `n_sites * 5`). Uploads `q0` once; every iteration stays
    /// device-resident except the three reduced scalars, copied back each
    /// step because the adaptive rule needs them on the host (exactly as
    /// open-Qmin's own `sim->sumUpdaterData` does across MPI ranks).
    ///
    /// The force computation is `force_fused_aos` (one thread per site, no
    /// separate `trq2` pass), not the split `trq2`+`force` pipeline: fused
    /// measured 17.6% faster as a kernel (`BENCHMARKS.md`'s "fuse the two
    /// passes"), and this loop is where that saving is realised. The split
    /// path stays available via `compute_force`/`time_split_force` for the
    /// side-by-side kernel comparison.
    pub fn fire_minimize(
        &self,
        q0: &[f64],
        ldg: &LdgParams,
        params: &FireParams,
    ) -> Result<FireResult, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        let len5 = n_sites * 5;
        assert_eq!(q0.len(), len5, "q0 length must be n_sites * 5");

        let stream = &self.stream;
        let mut q = DeviceBuffer::from_host(stream, q0)?;
        let mut v = DeviceBuffer::<f64>::zeroed(stream, len5)?;
        let mut f = DeviceBuffer::<f64>::zeroed(stream, len5)?;
        let mut f_new = DeviceBuffer::<f64>::zeroed(stream, len5)?;
        // One 3-element accumulator (`[sum|f|^2, sum|v|^2, sum f.v]`), not
        // three separate one-element buffers: three independent
        // `to_host_vec` calls per iteration each carry their own implicit
        // stream sync, which measurably costs more in host round-trip
        // latency than the reduction itself (see `BENCHMARKS.md`'s
        // "Optimisations tried"). Kept as a plain `f64` buffer between
        // iterations (`DeviceAtomicF64` is not `DeviceCopy`, so it cannot be
        // read back with `to_host_vec` directly); each reduction ping-pongs
        // one `cast_elem` out to the atomic view for the kernel launch and
        // one back for the read-back, a same-allocation reinterpret that
        // costs nothing beyond the type change.
        let mut acc = DeviceBuffer::<f64>::zeroed(stream, 3)?;

        let cfg_sites = LaunchConfig::for_num_elems(n_sites as u32);
        let cfg_len5 = LaunchConfig::for_num_elems(len5 as u32);

        // Initial force at q0.
        f = self.compute_force_fused(&q, f, ldg, cfg_sites)?;

        let mut delta_t = params.delta_t;
        let mut alpha = params.alpha_start;
        let mut n_since_negative_power: i32 = 0;
        let mut iterations: usize = 0;
        let mut force_max: f64;

        loop {
            let dt = delta_t;

            // Velocity-Verlet, against the OLD force `f`:
            //   q += dt*v + 0.5*dt^2*f   (position_update reads v, f both old)
            //   v += 0.5*dt*f            (half-kick #1, f still old)
            // SAFETY: every buffer above has length `len5`, matching
            // `cfg_len5`'s element count, and `position_update`/`axpy_inplace`
            // bounds-check every read against the `len` argument passed
            // alongside them.
            unsafe {
                self.module.position_update(
                    stream, cfg_len5, &v, &f, dt, len5 as u32, &mut q,
                )?;
                self.module.axpy_inplace(
                    stream, cfg_len5, &f, 0.5 * dt, len5 as u32, &mut v,
                )?;
            }

            // Recompute the force at the new q.
            f_new = self.compute_force_fused(&q, f_new, ldg, cfg_sites)?;
            std::mem::swap(&mut f, &mut f_new);

            // half-kick #2, against the NEW force.
            unsafe {
                self.module
                    .axpy_inplace(stream, cfg_len5, &f, 0.5 * dt, len5 as u32, &mut v)?;
            }

            // FIRE reduction: zero the accumulator while still typed as
            // plain `f64` (`DeviceAtomicF64` is not `DeviceCopy`, so neither
            // `zero_async` nor `to_host_vec` accept that view), cast to the
            // atomic view for the launch, cast back, and read all three
            // reduced scalars in one `to_host_vec` call.
            acc.zero_async(stream)?;
            let acc_atomic = acc.cast_elem::<DeviceAtomicF64>();
            // SAFETY: `f` and `v` both have length `len5 = n_sites * 5`, the
            // reduction reads exactly that extent under `n_sites`, and `acc`
            // holds the three atomic accumulators the kernel writes.
            unsafe {
                self.module
                    .reduce_fire(stream, cfg_sites, &f, &v, n_sites as u32, &acc_atomic)?;
            }
            acc = acc_atomic.cast_elem::<f64>();
            let reduced = acc.to_host_vec(stream)?;
            let (force_norm, velocity_norm, power) = (reduced[0], reduced[1], reduced[2]);

            force_max = force_norm.sqrt() / (n_sites as f64);
            let scaling = if force_norm > 0.0 {
                (velocity_norm / force_norm).sqrt()
            } else {
                0.0
            };

            // FIRE velocity mix.
            unsafe {
                self.module
                    .fire_mix(stream, cfg_len5, &f, scaling, alpha, len5 as u32, &mut v)?;
            }

            iterations += 1;
            if power > 0.0 && !iterations.is_multiple_of(500) {
                if n_since_negative_power > params.n_min {
                    delta_t = (delta_t * params.delta_t_inc).min(params.delta_t_max);
                    alpha = (alpha * params.alpha_dec).max(params.alpha_min);
                }
                n_since_negative_power += 1;
            } else {
                n_since_negative_power = 0;
                delta_t = (delta_t * params.delta_t_dec).max(params.delta_t_min);
                alpha = params.alpha_start;
                unsafe {
                    self.module.zero_field(stream, cfg_len5, len5 as u32, &mut v)?;
                }
            }

            if iterations >= params.max_iterations || force_max <= params.force_cutoff {
                break;
            }
        }

        let q_host = q.to_host_vec(stream)?;
        Ok(FireResult {
            q: q_host,
            iterations,
            force_max,
            converged: force_max <= params.force_cutoff,
        })
    }

    /// Achieved bandwidth (GB/s) of a pure copy kernel (one read, one write,
    /// nothing else), measured, not read off a spec sheet. `elements` should
    /// be large enough that the kernel is bandwidth-bound rather than
    /// latency-bound; `reps` timed copies, one untimed warm-up first.
    pub fn measure_bandwidth(&self, elements: usize, reps: usize) -> Result<f64, CudaError> {
        let stream = &self.stream;
        let host = vec![1.0_f64; elements];
        let x = DeviceBuffer::from_host(stream, &host)?;
        let mut y = DeviceBuffer::<f64>::zeroed(stream, elements)?;
        let cfg = LaunchConfig::for_num_elems(elements as u32);

        // SAFETY: `x` and `y` both have length `elements`, matching `cfg`.
        unsafe {
            self.module
                .bandwidth_copy(stream, cfg, &x, elements as u32, &mut y)?;
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            // SAFETY: as above.
            unsafe {
                self.module
                    .bandwidth_copy(stream, cfg, &x, elements as u32, &mut y)?;
            }
        }
        stream.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let bytes_per_rep = 2 * elements * std::mem::size_of::<f64>(); // one read + one write
        Ok((bytes_per_rep as f64 * reps as f64) / elapsed / 1e9)
    }

    /// Mean wall-clock per kernel launch (seconds), from a run of tiny
    /// (1-element) launches where the kernel body itself is negligible, so
    /// the measured time is overwhelmingly launch/dispatch overhead.
    pub fn measure_launch_overhead(&self, reps: usize) -> Result<f64, CudaError> {
        let stream = &self.stream;
        let x = DeviceBuffer::from_host(stream, &[1.0_f64])?;
        let mut y = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let cfg = LaunchConfig::for_num_elems(1);

        // SAFETY: `x` and `y` both have length 1, matching `cfg`.
        unsafe {
            self.module.bandwidth_copy(stream, cfg, &x, 1, &mut y)?;
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            // SAFETY: as above.
            unsafe {
                self.module.bandwidth_copy(stream, cfg, &x, 1, &mut y)?;
            }
        }
        stream.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();
        Ok(elapsed / reps as f64)
    }

    /// The fused, one-thread-per-site force kernel (`force_fused_aos`), AoS
    /// layout. Not used by `fire_minimize`; exposed so it can be timed and
    /// checked against the split `trq2`+`force` path once the GPU is free
    /// (see `BENCHMARKS.md`'s "fuse the two passes"). `q` has length
    /// `n_sites * 5`; returns the same length, gamma_r * H per site.
    pub fn force_fused_aos(&self, q: &[f64], ldg: &LdgParams) -> Result<Vec<f64>, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        assert_eq!(q.len(), n_sites * 5, "q length must be n_sites * 5");
        let stream = &self.stream;
        let q_dev = DeviceBuffer::from_host(stream, q)?;
        let mut out = DeviceBuffer::<[f64; 5]>::zeroed(stream, n_sites)?;
        let cfg = LaunchConfig::for_num_elems(n_sites as u32);
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);

        // SAFETY: `q_dev` has length `n_sites * 5`, the extent the kernel
        // reads under `nx,ny,nz`; `out` has `n_sites` elements, one
        // `[f64; 5]` slot per thread, matching `cfg`.
        unsafe {
            self.module.force_fused_aos(
                stream,
                cfg,
                &q_dev,
                ldg.nx,
                ldg.ny,
                ldg.nz,
                ldg.a_eff,
                ldg.b_landau,
                ldg.c_landau,
                ldg.k_r,
                ldg.gamma_r,
                inv_dx2,
                &mut out,
            )?;
        }
        let out_host = out.to_host_vec(stream)?;
        Ok(out_host.into_iter().flatten().collect())
    }

    /// The fused force kernel over 5 separate component planes
    /// (structure-of-arrays), `force_fused_soa`. Takes and returns AoS
    /// (`n_sites * 5`, `QField3D`'s own layout) and does the plane
    /// conversion on the host, so callers do not need to carry two layouts
    /// themselves; the point of this method is measuring whether the
    /// device-side layout change helps, not exposing SoA as a new format.
    pub fn force_soa(&self, q_aos: &[f64], ldg: &LdgParams) -> Result<Vec<f64>, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        assert_eq!(q_aos.len(), n_sites * 5, "q_aos length must be n_sites * 5");
        let stream = &self.stream;

        let mut planes: [Vec<f64>; 5] = Default::default();
        for plane in &mut planes {
            *plane = vec![0.0; n_sites];
        }
        for s in 0..n_sites {
            for c in 0..5 {
                planes[c][s] = q_aos[s * 5 + c];
            }
        }
        let q11 = DeviceBuffer::from_host(stream, &planes[0])?;
        let q12 = DeviceBuffer::from_host(stream, &planes[1])?;
        let q13 = DeviceBuffer::from_host(stream, &planes[2])?;
        let q22 = DeviceBuffer::from_host(stream, &planes[3])?;
        let q23 = DeviceBuffer::from_host(stream, &planes[4])?;
        let mut o11 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o12 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o13 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o22 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o23 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;

        let cfg = LaunchConfig::for_num_elems(n_sites as u32);
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);

        // SAFETY: every plane (input and output) has length `n_sites`,
        // matching `cfg`'s element count; the kernel bounds-checks its
        // neighbour reads against `nx,ny,nz`.
        unsafe {
            self.module.force_fused_soa(
                stream,
                cfg,
                &q11,
                &q12,
                &q13,
                &q22,
                &q23,
                ldg.nx,
                ldg.ny,
                ldg.nz,
                ldg.a_eff,
                ldg.b_landau,
                ldg.c_landau,
                ldg.k_r,
                ldg.gamma_r,
                inv_dx2,
                &mut o11,
                &mut o12,
                &mut o13,
                &mut o22,
                &mut o23,
            )?;
        }

        let p11 = o11.to_host_vec(stream)?;
        let p12 = o12.to_host_vec(stream)?;
        let p13 = o13.to_host_vec(stream)?;
        let p22 = o22.to_host_vec(stream)?;
        let p23 = o23.to_host_vec(stream)?;

        let mut out = vec![0.0; n_sites * 5];
        for s in 0..n_sites {
            out[s * 5] = p11[s];
            out[s * 5 + 1] = p12[s];
            out[s * 5 + 2] = p13[s];
            out[s * 5 + 3] = p22[s];
            out[s * 5 + 4] = p23[s];
        }
        Ok(out)
    }

    /// Mean host-to-device upload time (seconds) for a buffer of `elements`
    /// `f64`s, via `DeviceBuffer::from_host` -- the same call
    /// `fire_minimize`'s first statement makes for the initial condition.
    /// One untimed warm-up (first-touch allocator effects), then `reps`
    /// timed uploads, each a fresh allocation (`from_host` both allocates
    /// and copies, matching what a real caller pays).
    pub fn measure_h2d_upload(&self, elements: usize, reps: usize) -> Result<f64, CudaError> {
        let stream = &self.stream;
        let host = vec![1.0_f64; elements];
        let _warm = DeviceBuffer::from_host(stream, &host)?;
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            let _buf = DeviceBuffer::from_host(stream, &host)?;
        }
        stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    /// Pure kernel timing for the split `trq2`+`force` pipeline
    /// `fire_minimize` actually uses: upload `q` once, launch `reps` times
    /// back-to-back with no host round trip between launches, sync once at
    /// the end. Comparable, same-conditions basis for
    /// `time_fused_aos_force`/`time_fused_soa_force` below -- none of the
    /// three read anything back from the device mid-loop.
    pub fn time_split_force(&self, q0: &[f64], ldg: &LdgParams, reps: usize) -> Result<f64, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        let stream = &self.stream;
        let q = DeviceBuffer::from_host(stream, q0)?;
        let mut trq2 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut out = DeviceBuffer::<f64>::zeroed(stream, n_sites * 5)?;
        let cfg_sites = LaunchConfig::for_num_elems(n_sites as u32);
        let cfg_len5 = LaunchConfig::for_num_elems((n_sites * 5) as u32);

        self.compute_force(&q, &mut trq2, &mut out, ldg, cfg_sites, cfg_len5)?;
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            self.compute_force(&q, &mut trq2, &mut out, ldg, cfg_sites, cfg_len5)?;
        }
        stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    /// Pure kernel timing for `force_fused_aos`, same protocol as
    /// `time_split_force`: one untimed warm-up launch, `reps` timed launches
    /// with no host round trip, one final sync.
    pub fn time_fused_aos_force(&self, q0: &[f64], ldg: &LdgParams, reps: usize) -> Result<f64, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        let stream = &self.stream;
        let q = DeviceBuffer::from_host(stream, q0)?;
        let mut out = DeviceBuffer::<[f64; 5]>::zeroed(stream, n_sites)?;
        let cfg = LaunchConfig::for_num_elems(n_sites as u32);
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);

        // SAFETY: as in `force_fused_aos` above -- `q` has length
        // `n_sites * 5`, `out` has `n_sites` `[f64; 5]` slots, matching `cfg`.
        unsafe {
            self.module.force_fused_aos(
                stream, cfg, &q, ldg.nx, ldg.ny, ldg.nz, ldg.a_eff, ldg.b_landau, ldg.c_landau, ldg.k_r,
                ldg.gamma_r, inv_dx2, &mut out,
            )?;
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe {
                self.module.force_fused_aos(
                    stream, cfg, &q, ldg.nx, ldg.ny, ldg.nz, ldg.a_eff, ldg.b_landau, ldg.c_landau, ldg.k_r,
                    ldg.gamma_r, inv_dx2, &mut out,
                )?;
            }
        }
        stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    /// Pure kernel timing for `force_fused_soa`, same protocol. The AoS->SoA
    /// plane conversion happens once, untimed, before the warm-up launch
    /// (matching `force_soa`'s own boundary conversion, but paid once here
    /// rather than once per call, since the point is timing the kernel, not
    /// the conversion).
    pub fn time_fused_soa_force(&self, q0_aos: &[f64], ldg: &LdgParams, reps: usize) -> Result<f64, CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        assert_eq!(q0_aos.len(), n_sites * 5, "q0_aos length must be n_sites * 5");
        let stream = &self.stream;

        let mut planes: [Vec<f64>; 5] = Default::default();
        for plane in &mut planes {
            *plane = vec![0.0; n_sites];
        }
        for s in 0..n_sites {
            for c in 0..5 {
                planes[c][s] = q0_aos[s * 5 + c];
            }
        }
        let q11 = DeviceBuffer::from_host(stream, &planes[0])?;
        let q12 = DeviceBuffer::from_host(stream, &planes[1])?;
        let q13 = DeviceBuffer::from_host(stream, &planes[2])?;
        let q22 = DeviceBuffer::from_host(stream, &planes[3])?;
        let q23 = DeviceBuffer::from_host(stream, &planes[4])?;
        let mut o11 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o12 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o13 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o22 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        let mut o23 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;

        let cfg = LaunchConfig::for_num_elems(n_sites as u32);
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);

        // SAFETY: as in `force_soa` above -- every plane has length
        // `n_sites`, matching `cfg`.
        unsafe {
            self.module.force_fused_soa(
                stream, cfg, &q11, &q12, &q13, &q22, &q23, ldg.nx, ldg.ny, ldg.nz, ldg.a_eff,
                ldg.b_landau, ldg.c_landau, ldg.k_r, ldg.gamma_r, inv_dx2, &mut o11, &mut o12, &mut o13,
                &mut o22, &mut o23,
            )?;
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..reps {
            unsafe {
                self.module.force_fused_soa(
                    stream, cfg, &q11, &q12, &q13, &q22, &q23, ldg.nx, ldg.ny, ldg.nz, ldg.a_eff,
                    ldg.b_landau, ldg.c_landau, ldg.k_r, ldg.gamma_r, inv_dx2, &mut o11, &mut o12, &mut o13,
                    &mut o22, &mut o23,
                )?;
            }
        }
        stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_force(
        &self,
        q: &DeviceBuffer<f64>,
        trq2: &mut DeviceBuffer<f64>,
        out: &mut DeviceBuffer<f64>,
        ldg: &LdgParams,
        cfg_sites: LaunchConfig,
        cfg_len5: LaunchConfig,
    ) -> Result<(), CudaError> {
        let n_sites = (ldg.nx as usize) * (ldg.ny as usize) * (ldg.nz as usize);
        // SAFETY: `q` has length `n_sites * 5`, `trq2` has length `n_sites`
        // matching `cfg_sites`'s element count, and `trq2` bounds-checks its
        // read of `q` against the `n_sites` argument.
        unsafe {
            self.module
                .trq2(&self.stream, cfg_sites, q, n_sites as u32, trq2)?;
        }
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);
        // SAFETY: `q` and `trq2` both cover the extents `force` reads under
        // `nx,ny,nz`, and `out` has length `n_sites * 5` matching `cfg_len5`.
        unsafe {
            self.module.force(
                &self.stream,
                cfg_len5,
                q,
                trq2,
                ldg.nx,
                ldg.ny,
                ldg.nz,
                ldg.a_eff,
                ldg.b_landau,
                ldg.c_landau,
                ldg.k_r,
                ldg.gamma_r,
                inv_dx2,
                out,
            )?;
        }
        Ok(())
    }

    /// The fused, one-thread-per-site force kernel (`force_fused_aos`),
    /// wired into `fire_minimize`'s hot loop (`BENCHMARKS.md`'s "the cheap
    /// volterra gains": measured 17.6% faster than the split `trq2`+`force`
    /// pair as a kernel). Takes `out` by value and returns it: the kernel
    /// writes through `DisjointSlice<[f64; 5]>`, so `out` (allocated flat,
    /// `n_sites * 5` `f64`s, to match every other kernel in the loop) is
    /// reinterpreted via `cast_chunks` for the launch and back to flat `f64`
    /// on return -- a pointer/length reinterpretation of the same
    /// allocation, no device copy, exactly the `acc`/`acc_atomic` pattern
    /// `fire_minimize`'s reduction already uses for the same reason.
    fn compute_force_fused(
        &self,
        q: &DeviceBuffer<f64>,
        out: DeviceBuffer<f64>,
        ldg: &LdgParams,
        cfg_sites: LaunchConfig,
    ) -> Result<DeviceBuffer<f64>, CudaError> {
        let inv_dx2 = 1.0 / (ldg.dx * ldg.dx);
        let mut out5 = out.cast_chunks::<[f64; 5]>().unwrap_or_else(|_| {
            panic!("n_sites*5 f64 buffer must reinterpret as n_sites [f64;5] (same alignment)")
        });
        // SAFETY: `q` has length `n_sites * 5`, the extent the kernel reads
        // under `nx,ny,nz`; `out5` has `n_sites` `[f64; 5]` slots, matching
        // `cfg_sites`.
        unsafe {
            self.module.force_fused_aos(
                &self.stream,
                cfg_sites,
                q,
                ldg.nx,
                ldg.ny,
                ldg.nz,
                ldg.a_eff,
                ldg.b_landau,
                ldg.c_landau,
                ldg.k_r,
                ldg.gamma_r,
                inv_dx2,
                &mut out5,
            )?;
        }
        Ok(out5.cast_chunks::<f64>().unwrap_or_else(|_| {
            panic!("n_sites [f64;5] buffer must reinterpret back to n_sites*5 f64")
        }))
    }
}
