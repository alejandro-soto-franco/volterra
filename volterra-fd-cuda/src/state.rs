//! A whole solver step, device resident.
//!
//! [`DeviceState`] holds every field `volterra_fd::step::State` holds, on the
//! device, and [`DeviceState::step`] runs one advance of
//! `volterra_fd::step::update_step_inner` across them. Nothing crosses the
//! bus during a step except the pressure loop's convergence measure, which the
//! adaptive stopping rule needs on the host, exactly as `volterra-cuda`'s FIRE
//! reads back its three reduced scalars.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};

use crate::device::{Device, DeviceBoundary};
use crate::error::CudaError;

/// The physical parameters a step needs, mirroring `volterra_fd::Params`.
#[derive(Debug, Clone, Copy)]
pub struct StepParams {
    pub a_landau: f64,
    pub c_landau: f64,
    pub k_elastic: f64,
    pub lambda: f64,
    pub zeta: f64,
    pub gamma: f64,
    pub eta: f64,
    pub rho: f64,
    pub dt: f64,
    pub s0: f64,
    pub net_charge: f64,
    pub max_p_iters: i64,
    /// Run exactly this many Jacobi sweeps and never read the convergence
    /// measure back.
    ///
    /// The measure is the only synchronisation inside a step, and it costs
    /// about as much as everything else the step does. The census in
    /// `volterra-fd/examples/jacobi_census.rs` measures the golden
    /// configuration at 1.45 sweeps per step, with every one of the 41 steps
    /// that reach the cap inside the first hundred: after the transient the
    /// solve converges in a single sweep, for the rest of a 750,000-step run.
    ///
    /// `None` keeps the adaptive rule and the readback, and is what the
    /// comparisons against the CPU use. `Some(k)` is a different algorithm on
    /// the transient and the same one after it.
    pub fixed_sweeps: Option<usize>,
}

impl StepParams {
    /// Read the parameters off a CPU `Params`, so both sides run the same run.
    pub fn from_cpu(p: &volterra_fd::Params) -> Self {
        Self {
            a_landau: p.a_landau,
            c_landau: p.c_landau,
            k_elastic: p.k_elastic,
            lambda: p.lambda,
            zeta: p.zeta,
            gamma: p.gamma,
            eta: p.eta,
            rho: p.rho,
            dt: p.dt,
            s0: p.s0,
            net_charge: p.net_charge,
            max_p_iters: p.max_p_iters,
            fixed_sweeps: None,
        }
    }
}

/// Every field of one simulation, on the device.
pub struct DeviceState {
    pub u: DeviceBuffer<f64>,
    pub q: DeviceBuffer<f64>,
    pub p: DeviceBuffer<f64>,
    pub p_aux: DeviceBuffer<f64>,
    pub rhs: DeviceBuffer<f64>,
    pub h: DeviceBuffer<f64>,
    pub s: DeviceBuffer<f64>,
    pub dq: DeviceBuffer<f64>,
    pub dudt: DeviceBuffer<f64>,
    pub pi_s: DeviceBuffer<f64>,
    pub pi_a: DeviceBuffer<f64>,
    /// Per-span partial sums for the pressure convergence measure, each span's
    /// two sums in one 2-wide slot so the host reads them in one transfer.
    partials: DeviceBuffer<f64>,
    /// The stream this simulation's work is queued on. Separate states get
    /// separate streams so their kernels may overlap, which is what fills a
    /// device that one 10,000-cell grid cannot.
    pub stream: Arc<CudaStream>,
    n: usize,
    /// How many spans the convergence measure is cut into, and how long each
    /// is. Fixed at construction so the summation order never varies.
    n_blocks: usize,
    span: usize,
}

impl DeviceState {
    /// Allocate every field zeroed, for a grid of `lx * ly` cells.
    pub fn zeroed(stream: &Arc<CudaStream>, lx: usize, ly: usize) -> Result<Self, CudaError> {
        let own = stream.clone();
        let n = lx * ly;
        let n2 = n * 2;
        // 256 spans over the grid: enough threads to occupy the device on the
        // reduction, few enough that the host-side sum stays short.
        let n_blocks = 256.min(n);
        let span = n.div_ceil(n_blocks);
        Ok(Self {
            u: DeviceBuffer::zeroed(stream, n2)?,
            q: DeviceBuffer::zeroed(stream, n2)?,
            p: DeviceBuffer::zeroed(stream, n)?,
            p_aux: DeviceBuffer::zeroed(stream, n)?,
            rhs: DeviceBuffer::zeroed(stream, n)?,
            h: DeviceBuffer::zeroed(stream, n2)?,
            s: DeviceBuffer::zeroed(stream, n2)?,
            dq: DeviceBuffer::zeroed(stream, n2)?,
            dudt: DeviceBuffer::zeroed(stream, n2)?,
            pi_s: DeviceBuffer::zeroed(stream, n2)?,
            pi_a: DeviceBuffer::zeroed(stream, n)?,
            partials: DeviceBuffer::zeroed(stream, n_blocks * 2)?,
            stream: own,
            n,
            n_blocks,
            span,
        })
    }

    /// Copy a CPU state's `q`, `u` and `p` up, leaving the scratch fields as
    /// they are. The CPU's scratch carries nothing across a step boundary.
    pub fn upload_from(
        &mut self,
        stream: &Arc<CudaStream>,
        q: &[f64],
        u: &[f64],
        p: &[f64],
    ) -> Result<(), CudaError> {
        self.q = DeviceBuffer::from_host(stream, q)?;
        self.u = DeviceBuffer::from_host(stream, u)?;
        self.p = DeviceBuffer::from_host(stream, p)?;
        Ok(())
    }

    /// Read `q`, `u` and `p` back.
    pub fn download(&self, stream: &Arc<CudaStream>) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), CudaError> {
        Ok((
            self.q.to_host_vec(stream)?,
            self.u.to_host_vec(stream)?,
            self.p.to_host_vec(stream)?,
        ))
    }
}

impl Device {
    /// One advance of the whole step, device resident.
    ///
    /// Follows `volterra_fd::step::update_step_inner` stage for stage, and
    /// returns the number of Jacobi sweeps the pressure solve took, which is
    /// what that function returns.
    pub fn step(
        &self,
        st: &mut DeviceState,
        bnd: &DeviceBoundary,
        p: &StepParams,
        target_rel_change: f64,
    ) -> Result<usize, CudaError> {
        let stream = &st.stream.clone();
        let n = st.n;
        let n2 = n * 2;
        let cells = LaunchConfig::for_num_elems(n as u32);
        let elems2 = LaunchConfig::for_num_elems(n2 as u32);
        let blocks = LaunchConfig::for_num_elems(st.n_blocks as u32);
        let (lx, ly) = (bnd.lx as u32, bnd.ly as u32);
        let m = &self.module_ref();

        // 1. H and S from Q.
        {
            let mut h2 = std::mem::replace(&mut st.h, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("h as [f64;2]"));
            let mut s2 = std::mem::replace(&mut st.s, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("s as [f64;2]"));
            // SAFETY: every buffer is the grid's own extent, which the kernel
            // bounds-checks against `lx * ly`; the two outputs hold `n` 2-wide
            // slots matching `cells`.
            let r = unsafe {
                m.h_s_from_q(
                    stream, cells, &st.u, &st.q, &bnd.inside, lx, ly,
                    p.a_landau, p.c_landau, p.k_elastic, p.lambda, &mut h2, &mut s2,
                )
            };
            st.h = h2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("h back"));
            st.s = s2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("s back"));
            r?;
        }

        // 2. H boundary condition.
        {
            let mut h2 = std::mem::replace(&mut st.h, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("h as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.apply_h_bc(
                    stream, cells, &st.q, &st.u, &st.s,
                    &bnd.is_inner, &bnd.is_outer, &bnd.inner_normals, &bnd.outer_normals,
                    lx, ly, p.gamma, &mut h2,
                )
            };
            st.h = h2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("h back"));
            r?;
        }

        // 3. The stresses.
        {
            let mut pi_s2 = std::mem::replace(&mut st.pi_s, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("pi_s as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.calculate_pi(
                    stream, cells, &st.h, &st.q, &bnd.inside, lx, ly,
                    p.lambda, p.zeta, p.k_elastic, &mut pi_s2, &mut st.pi_a,
                )
            };
            st.pi_s = pi_s2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("pi_s back"));
            r?;
        }

        // 4. Pressure relaxation.
        let p_iters = self.relax_pressure(st, bnd, p, target_rel_change)?;

        // 5. The Q update.
        {
            let mut dq2 = std::mem::replace(&mut st.dq, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("dq as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.q_update(
                    stream, cells, &st.q, &st.h, &st.s, &st.u, &bnd.inside, lx, ly,
                    p.gamma, &mut dq2,
                )
            };
            st.dq = dq2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("dq back"));
            r?;
        }

        // 6. The velocity update.
        {
            let mut dudt2 = std::mem::replace(&mut st.dudt, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("dudt as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.u_update(
                    stream, cells, &st.u, &st.p, &st.pi_s, &st.pi_a, &bnd.inside, lx, ly,
                    p.rho, p.eta, &mut dudt2,
                )
            };
            st.dudt = dudt2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("dudt back"));
            r?;
        }

        // 7 and 8. Integrate both fields.
        // SAFETY: `dq` and `q` are the same length, as are `dudt` and `u`, and
        // the kernel bounds-checks both against `len`.
        unsafe {
            m.integrate(stream, elems2, &st.dq, n2 as u32, p.dt, &mut st.q)?;
            m.integrate(stream, elems2, &st.dudt, n2 as u32, p.dt, &mut st.u)?;
        }

        // 9. Q anchoring.
        {
            let mut q2 = std::mem::replace(&mut st.q, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("q as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.apply_q_bc(
                    stream, cells,
                    &bnd.is_inner, &bnd.is_outer, &bnd.inner_normals, &bnd.outer_normals,
                    lx, ly, p.s0, p.net_charge, &mut q2,
                )
            };
            st.q = q2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("q back"));
            r?;
        }

        // 10. No-slip.
        {
            let mut u2 = std::mem::replace(&mut st.u, DeviceBuffer::zeroed(stream, 0)?)
                .cast_chunks::<[f64; 2]>()
                .unwrap_or_else(|_| panic!("u as [f64;2]"));
            // SAFETY: as above.
            let r = unsafe {
                m.apply_u_bc(
                    stream, cells,
                    &bnd.is_inner, &bnd.is_outer, &bnd.inner_normals, &bnd.outer_normals,
                    lx, ly, &mut u2,
                )
            };
            st.u = u2.cast_chunks::<f64>().unwrap_or_else(|_| panic!("u back"));
            r?;
        }

        let _ = blocks;
        Ok(p_iters)
    }

    /// The pressure solve, matching `relax_pressure_with_bc`.
    fn relax_pressure(
        &self,
        st: &mut DeviceState,
        bnd: &DeviceBoundary,
        p: &StepParams,
        target_rel_change: f64,
    ) -> Result<usize, CudaError> {
        let stream = &st.stream.clone();
        let n = st.n;
        let cells = LaunchConfig::for_num_elems(n as u32);
        let blocks = LaunchConfig::for_num_elems(st.n_blocks as u32);
        let (lx, ly) = (bnd.lx as u32, bnd.ly as u32);
        let m = &self.module_ref();

        // rhs = (rho/dt) div u, then the stress and convective terms.
        // SAFETY: `u` holds `2n` and `rhs` `n`, both indexed under `lx * ly`.
        unsafe {
            m.div_vector(stream, cells, &st.u, &bnd.inside, lx, ly, &mut st.rhs)?;
            m.scale_scalar(stream, cells, n as u32, p.rho / p.dt, &mut st.rhs)?;
            m.pressure_terms(
                stream, cells, &st.u, &st.pi_s, &bnd.inside, lx, ly, p.rho, &mut st.rhs,
            )?;
        }

        // A fixed sweep count needs no convergence measure, so the step runs
        // with no host interaction at all.
        if let Some(k) = p.fixed_sweeps {
            for _ in 0..k {
                // SAFETY: as below.
                unsafe {
                    m.copy_scalar(stream, cells, &st.p, n as u32, &mut st.p_aux)?;
                    m.jacobi_sweep(
                        stream, cells, &st.p_aux, &st.rhs, &bnd.inside, lx, ly, &mut st.p,
                    )?;
                    m.apply_p_bc(
                        stream, cells, &st.p_aux, &st.u, &st.pi_s, &st.pi_a,
                        &bnd.is_inner, &bnd.is_outer, &bnd.inner_normals, &bnd.outer_normals,
                        lx, ly, p.rho, p.eta, &mut st.p,
                    )?;
                }
            }
            let _ = blocks;
            return Ok(k);
        }

        let mut p_iters = 0usize;
        let mut rel_change = target_rel_change + 1.0;
        loop {
            if p.max_p_iters >= 0 && p_iters >= p.max_p_iters as usize {
                break;
            }
            if p_iters > 0 && rel_change <= target_rel_change {
                break;
            }

            // SAFETY: `p` and `p_aux` are both `n` long, bounds-checked against
            // `len` and `lx * ly` respectively.
            unsafe {
                m.copy_scalar(stream, cells, &st.p, n as u32, &mut st.p_aux)?;
                m.jacobi_sweep(
                    stream, cells, &st.p_aux, &st.rhs, &bnd.inside, lx, ly, &mut st.p,
                )?;
                m.apply_p_bc(
                    stream, cells, &st.p_aux, &st.u, &st.pi_s, &st.pi_a,
                    &bnd.is_inner, &bnd.is_outer, &bnd.inner_normals, &bnd.outer_normals,
                    lx, ly, p.rho, p.eta, &mut st.p,
                )?;
            }
            {
                let mut pr = std::mem::replace(&mut st.partials, DeviceBuffer::zeroed(stream, 0)?)
                    .cast_chunks::<[f64; 2]>()
                    .unwrap_or_else(|_| panic!("partials as [f64;2]"));
                // SAFETY: `p` and `p_aux` hold `n`, bounds-checked against
                // `len`, and `pr` holds `n_blocks` 2-wide slots matching
                // `blocks`.
                let r = unsafe {
                    m.pressure_partials(
                        stream, blocks, &st.p, &st.p_aux, n as u32, st.span as u32,
                        st.n_blocks as u32, &mut pr,
                    )
                };
                st.partials = pr.cast_chunks::<f64>().unwrap_or_else(|_| panic!("partials back"));
                r?;
            }

            // The only host round trip in a step, and one transfer rather than
            // two. Both sums are added in index order, so the measure is the
            // same on every run.
            let flat = st.partials.to_host_vec(stream)?;
            let mut sum_diff = 0.0_f64;
            let mut sum_old = 0.0_f64;
            for b in 0..st.n_blocks {
                sum_diff += flat[b * 2];
            }
            for b in 0..st.n_blocks {
                sum_old += flat[b * 2 + 1];
            }
            rel_change = sum_diff / (1e-7 + sum_old);

            p_iters += 1;
        }
        Ok(p_iters)
    }
}
