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
//!
//! # State
//!
//! Every stage of `volterra_cgpo::step::update_step_inner` has a kernel here,
//! and each is checked against its CPU counterpart by the `validate` phase:
//! the molecular field and co-rotation, the four boundary conditions, the two
//! stresses, the pressure right-hand side and its Jacobi sweep, the Q and
//! velocity updates, and the integrator. The three differential operators are
//! exposed separately as well, since several stages are built from them.
//!
//! Two things stand between that and a device-resident run. The pressure
//! loop's convergence measure is a reduction over the whole grid, and running
//! it as device atomics makes the sweep count depend on accumulation order,
//! which the measured behaviour makes cheap to avoid: the solve converges in a
//! single sweep for all but the first hundred steps of a run. And the step
//! itself still round-trips each field to the host between kernels, which is
//! what the assembly stage removes.
//!
//! # Agreement
//!
//! Bit for bit where the arithmetic allows it: the three operators, the Jacobi
//! sweep, and the no-slip condition. Elsewhere the device contracts multiplies
//! and adds into single instructions, which rounds once where the host rounds
//! twice, and its transcendentals are not the host libm's. Both show up as a
//! difference in the last bits, and the harness measures in units in the last
//! place so that a real defect, which is not small, cannot hide behind them.
//!
//! Two inputs mislead badly, and both are named here. A random field is the
//! worst case for a 9-point stencil, whose weights nearly cancel, so at
//! `K = 16384` a last-bit difference in the terms reads as a hundred ulp in
//! the result; the physics kernels are checked on smooth fields for that
//! reason. And `Pi_A` cancels by construction, so it is measured against the
//! size of the two products it differences rather than against its own value.

mod device;
mod error;
mod kernels;

pub use device::{Device, DeviceBoundary};
pub use error::CudaError;
