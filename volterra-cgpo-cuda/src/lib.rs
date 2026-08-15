//! Double-precision CUDA path for volterra's 2D confined active-nematic
//! solver, via cuda-oxide (the pattern `cartan-cuda` and `volterra-cuda`
//! establish: kernels are ordinary Rust, compiled to PTX by
//! `rustc-codegen-cuda`).
//!
//! `volterra_cgpo` is the CPU reference. Every kernel here ports one of its
//! functions and is checked against it on the same input by this crate's
//! `validate` phase, before anything is composed out of them.
//!
//! # Why this exists
//!
//! The golden and silver braid runs are 100x100 grids stepped 750,000 and
//! 500,000 times, and `volterra_cgpo` takes the serial path at that size
//! (`par_gate::PAR_THRESHOLD` is 250,000 cells), so they run on one core in
//! 484 s and 379 s. A 10,000-cell grid does not fill a modern device on its
//! own, so the design that pays is a batch: a parameter sweep over `q` and
//! seeds shares one set of launches across every run in it.

mod device;
mod error;
mod kernels;

pub use device::{Device, DeviceBoundary};
pub use error::CudaError;
