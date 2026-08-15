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
