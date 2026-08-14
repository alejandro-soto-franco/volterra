//! `Device`: an open CUDA context holding Q, V (FIRE velocity), F (force)
//! and the reduction accumulators, running the same FIRE algorithm as
//! `volterra_solver::fire` (see that module's doc comment for the algorithm
//! and the open-Qmin correspondence) with the elementwise and stencil work
//! on the GPU and the small adaptive-timestep bookkeeping on the host,
//! exactly mirroring `volterra_solver::fire::FireState`.

use std::sync::Arc;

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
        let mut trq2 = DeviceBuffer::<f64>::zeroed(stream, n_sites)?;
        // Kept as plain `f64` buffers between iterations (`DeviceAtomicF64`
        // is not `DeviceCopy`, so it cannot be read back with `to_host_vec`
        // directly); each reduction ping-pongs one `cast_elem` out to the
        // atomic view for the kernel launch and one back for the read-back.
        let mut force_acc = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut vel_acc = DeviceBuffer::<f64>::zeroed(stream, 1)?;
        let mut power_acc = DeviceBuffer::<f64>::zeroed(stream, 1)?;

        let cfg_sites = LaunchConfig::for_num_elems(n_sites as u32);
        let cfg_len5 = LaunchConfig::for_num_elems(len5 as u32);

        // Initial force at q0.
        self.compute_force(&q, &mut trq2, &mut f, ldg, cfg_sites, cfg_len5)?;

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
            self.compute_force(&q, &mut trq2, &mut f_new, ldg, cfg_sites, cfg_len5)?;
            std::mem::swap(&mut f, &mut f_new);

            // half-kick #2, against the NEW force.
            unsafe {
                self.module
                    .axpy_inplace(stream, cfg_len5, &f, 0.5 * dt, len5 as u32, &mut v)?;
            }

            // FIRE reduction: zero the three accumulators while still typed
            // as plain `f64` (`DeviceAtomicF64` is not `DeviceCopy`, so
            // neither `zero_async` nor `to_host_vec` accept that view), then
            // cast to the atomic view for the launch and back to `f64` to
            // read the result. `cast_elem` is a same-allocation reinterpret,
            // not a copy, so this ping-pong costs nothing beyond the type change.
            force_acc.zero_async(stream)?;
            vel_acc.zero_async(stream)?;
            power_acc.zero_async(stream)?;
            let force_acc_atomic = force_acc.cast_elem::<DeviceAtomicF64>();
            let vel_acc_atomic = vel_acc.cast_elem::<DeviceAtomicF64>();
            let power_acc_atomic = power_acc.cast_elem::<DeviceAtomicF64>();
            // SAFETY: `f` and `v` both have length `len5 = n_sites * 5`, the
            // reduction reads exactly that extent under `n_sites`, and the
            // three accumulators are single-element atomic buffers.
            unsafe {
                self.module.reduce_fire(
                    stream,
                    cfg_sites,
                    &f,
                    &v,
                    n_sites as u32,
                    &force_acc_atomic,
                    &vel_acc_atomic,
                    &power_acc_atomic,
                )?;
            }
            force_acc = force_acc_atomic.cast_elem::<f64>();
            vel_acc = vel_acc_atomic.cast_elem::<f64>();
            power_acc = power_acc_atomic.cast_elem::<f64>();
            let force_norm = force_acc.to_host_vec(stream)?[0];
            let velocity_norm = vel_acc.to_host_vec(stream)?[0];
            let power = power_acc.to_host_vec(stream)?[0];

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
                ldg.c_landau,
                ldg.k_r,
                ldg.gamma_r,
                inv_dx2,
                out,
            )?;
        }
        Ok(())
    }
}
