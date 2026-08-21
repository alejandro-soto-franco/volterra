//! Double-precision CUDA FIRE minimiser for volterra's passive Landau-de
//! Gennes relaxation, via cuda-oxide (the pattern `cartan-cuda` establishes:
//! kernels are ordinary Rust, compiled to PTX by `rustc-codegen-cuda`).
//!
//! See `volterra_fd::fire` for the algorithm this ports (it is the same
//! FIRE, ported step for step from open-Qmin's reference implementation) and
//! `device.rs` for how it is split across kernel launches.

mod device;
mod error;
mod kernels;

pub use device::{Bookkeeping, Device, FireParams, FireResult, LdgParams};
pub use error::CudaError;
