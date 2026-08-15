//! `Device`: an open CUDA context and the field buffers one 2D CGPO grid
//! needs, with one method per operator so each can be checked against its CPU
//! counterpart on the same input before anything is composed out of them.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

use crate::error::CudaError;
use crate::kernels;

/// The confining geometry, as the device holds it.
///
/// `inside` is `u8` rather than `bool`: `bool` is not a device-copyable element
/// type, and the kernels test it against zero.
pub struct DeviceBoundary {
    pub lx: usize,
    pub ly: usize,
    pub inside: DeviceBuffer<u8>,
}

impl DeviceBoundary {
    /// Upload a CPU boundary's interior mask.
    pub fn upload(
        stream: &Arc<CudaStream>,
        lx: usize,
        ly: usize,
        inside: &[bool],
    ) -> Result<Self, CudaError> {
        assert_eq!(inside.len(), lx * ly, "mask must be lx * ly");
        let bytes: Vec<u8> = inside.iter().map(|&b| u8::from(b)).collect();
        Ok(Self {
            lx,
            ly,
            inside: DeviceBuffer::from_host(stream, &bytes)?,
        })
    }

    /// Cell count.
    pub fn cells(&self) -> usize {
        self.lx * self.ly
    }
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

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// 9-point Laplacian of a scalar field.
    ///
    /// `out` is read back with the cells outside the mask left as they were on
    /// the device, matching the CPU, so the caller seeds it with whatever the
    /// CPU would have had there.
    pub fn laplacian_scalar(
        &self,
        arr: &[f64],
        bnd: &DeviceBoundary,
        coeff: f64,
        out_seed: &[f64],
    ) -> Result<Vec<f64>, CudaError> {
        let n = bnd.cells();
        assert_eq!(arr.len(), n, "scalar field must be lx * ly");
        assert_eq!(out_seed.len(), n, "seed must be lx * ly");
        let stream = &self.stream;
        let d_arr = DeviceBuffer::from_host(stream, arr)?;
        let mut d_out = DeviceBuffer::from_host(stream, out_seed)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: `d_arr` and `bnd.inside` both hold `n` elements, which the
        // kernel bounds-checks against `lx * ly` before indexing, and `d_out`
        // holds `n` slots matching `cfg`.
        unsafe {
            self.module.laplacian_scalar(
                stream,
                cfg,
                &d_arr,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                coeff,
                &mut d_out,
            )?;
        }
        Ok(d_out.to_host_vec(stream)?)
    }

    /// 9-point Laplacian of a 2-vector field.
    pub fn laplacian_vector(
        &self,
        arr: &[f64],
        bnd: &DeviceBoundary,
        coeff: f64,
        out_seed: &[f64],
    ) -> Result<Vec<f64>, CudaError> {
        let n = bnd.cells();
        assert_eq!(arr.len(), n * 2, "vector field must be lx * ly * 2");
        assert_eq!(out_seed.len(), n * 2, "seed must be lx * ly * 2");
        let stream = &self.stream;
        let d_arr = DeviceBuffer::from_host(stream, arr)?;
        let d_out = DeviceBuffer::from_host(stream, out_seed)?;
        let mut d_out2 = d_out.cast_chunks::<[f64; 2]>().unwrap_or_else(|_| {
            panic!("lx*ly*2 f64 buffer must reinterpret as lx*ly [f64;2]")
        });
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: as above; `d_out2` holds `n` 2-wide slots matching `cfg`.
        let launched = unsafe {
            self.module.laplacian_vector(
                stream,
                cfg,
                &d_arr,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                coeff,
                &mut d_out2,
            )
        };
        let d_out = d_out2
            .cast_chunks::<f64>()
            .unwrap_or_else(|_| panic!("buffer must reinterpret back to f64"));
        launched?;
        Ok(d_out.to_host_vec(stream)?)
    }

    /// Divergence of a 2-vector field.
    pub fn div_vector(
        &self,
        arr: &[f64],
        bnd: &DeviceBoundary,
        out_seed: &[f64],
    ) -> Result<Vec<f64>, CudaError> {
        let n = bnd.cells();
        assert_eq!(arr.len(), n * 2, "vector field must be lx * ly * 2");
        assert_eq!(out_seed.len(), n, "seed must be lx * ly");
        let stream = &self.stream;
        let d_arr = DeviceBuffer::from_host(stream, arr)?;
        let mut d_out = DeviceBuffer::from_host(stream, out_seed)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: as above.
        unsafe {
            self.module.div_vector(
                stream,
                cfg,
                &d_arr,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                &mut d_out,
            )?;
        }
        Ok(d_out.to_host_vec(stream)?)
    }

    /// The molecular field and the co-rotation tensor.
    ///
    /// Returns `(h, s)`. `h_seed` supplies what the exterior of `h` should
    /// hold, since the kernel writes only inside the mask, as the CPU does.
    #[allow(clippy::too_many_arguments)]
    pub fn h_s_from_q(
        &self,
        u: &[f64],
        q: &[f64],
        bnd: &DeviceBoundary,
        a: f64,
        c_coeff: f64,
        k: f64,
        lambda: f64,
        h_seed: &[f64],
        s_seed: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), CudaError> {
        let n = bnd.cells();
        assert_eq!(u.len(), n * 2, "velocity must be lx * ly * 2");
        assert_eq!(q.len(), n * 2, "Q must be lx * ly * 2");
        let stream = &self.stream;
        let d_u = DeviceBuffer::from_host(stream, u)?;
        let d_q = DeviceBuffer::from_host(stream, q)?;
        let mut d_h = DeviceBuffer::from_host(stream, h_seed)?
            .cast_chunks::<[f64; 2]>()
            .unwrap_or_else(|_| panic!("h must reinterpret as [f64;2]"));
        let mut d_s = DeviceBuffer::from_host(stream, s_seed)?
            .cast_chunks::<[f64; 2]>()
            .unwrap_or_else(|_| panic!("s must reinterpret as [f64;2]"));
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: every input holds the extent the kernel bounds-checks against
        // `lx * ly`, and both outputs hold `n` 2-wide slots matching `cfg`.
        let launched = unsafe {
            self.module.h_s_from_q(
                stream,
                cfg,
                &d_u,
                &d_q,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                a,
                c_coeff,
                k,
                lambda,
                &mut d_h,
                &mut d_s,
            )
        };
        let d_h = d_h.cast_chunks::<f64>().unwrap_or_else(|_| panic!("h back to f64"));
        let d_s = d_s.cast_chunks::<f64>().unwrap_or_else(|_| panic!("s back to f64"));
        launched?;
        Ok((d_h.to_host_vec(stream)?, d_s.to_host_vec(stream)?))
    }

    /// The symmetric and antisymmetric stresses. Returns `(pi_s, pi_a)`.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_pi(
        &self,
        h: &[f64],
        q: &[f64],
        bnd: &DeviceBoundary,
        lambda: f64,
        zeta: f64,
        k: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), CudaError> {
        let n = bnd.cells();
        assert_eq!(h.len(), n * 2, "H must be lx * ly * 2");
        assert_eq!(q.len(), n * 2, "Q must be lx * ly * 2");
        let stream = &self.stream;
        let d_h = DeviceBuffer::from_host(stream, h)?;
        let d_q = DeviceBuffer::from_host(stream, q)?;
        // Both outputs are written at every cell, so neither needs a seed.
        let mut d_pi_s = DeviceBuffer::<f64>::zeroed(stream, n * 2)?
            .cast_chunks::<[f64; 2]>()
            .unwrap_or_else(|_| panic!("pi_s must reinterpret as [f64;2]"));
        let mut d_pi_a = DeviceBuffer::<f64>::zeroed(stream, n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: as above.
        let launched = unsafe {
            self.module.calculate_pi(
                stream,
                cfg,
                &d_h,
                &d_q,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                lambda,
                zeta,
                k,
                &mut d_pi_s,
                &mut d_pi_a,
            )
        };
        let d_pi_s = d_pi_s
            .cast_chunks::<f64>()
            .unwrap_or_else(|_| panic!("pi_s back to f64"));
        launched?;
        Ok((d_pi_s.to_host_vec(stream)?, d_pi_a.to_host_vec(stream)?))
    }

    /// Second-order upwind advection, accumulated into `out_seed`.
    pub fn upwind_advective(
        &self,
        u: &[f64],
        arr: &[f64],
        bnd: &DeviceBoundary,
        coeff: f64,
        out_seed: &[f64],
    ) -> Result<Vec<f64>, CudaError> {
        let n = bnd.cells();
        assert_eq!(u.len(), n * 2, "velocity must be lx * ly * 2");
        assert_eq!(arr.len(), n * 2, "field must be lx * ly * 2");
        assert_eq!(out_seed.len(), n * 2, "seed must be lx * ly * 2");
        let stream = &self.stream;
        let d_u = DeviceBuffer::from_host(stream, u)?;
        let d_arr = DeviceBuffer::from_host(stream, arr)?;
        let d_out = DeviceBuffer::from_host(stream, out_seed)?;
        let mut d_out2 = d_out.cast_chunks::<[f64; 2]>().unwrap_or_else(|_| {
            panic!("lx*ly*2 f64 buffer must reinterpret as lx*ly [f64;2]")
        });
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: as above.
        let launched = unsafe {
            self.module.upwind_advective(
                stream,
                cfg,
                &d_u,
                &d_arr,
                &bnd.inside,
                bnd.lx as u32,
                bnd.ly as u32,
                coeff,
                &mut d_out2,
            )
        };
        let d_out = d_out2
            .cast_chunks::<f64>()
            .unwrap_or_else(|_| panic!("buffer must reinterpret back to f64"));
        launched?;
        Ok(d_out.to_host_vec(stream)?)
    }
}
