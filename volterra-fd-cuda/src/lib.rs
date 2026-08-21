//! Double-precision CUDA path for volterra's 2D confined active-nematic
//! solver, via cuda-oxide (the pattern `cartan-cuda` and `volterra-cuda`
//! establish: kernels are ordinary Rust, compiled to PTX by
//! `rustc-codegen-cuda`).
//!
//! `volterra_fd` is the CPU reference. Every kernel here ports one of its
//! functions and is checked against it on the same input by this crate's
//! `validate` phase, before anything is composed out of them.
//!
//! # Why this exists
//!
//! The golden and silver braid runs are 100x100 grids stepped 750,000 and
//! 500,000 times, and `volterra_fd` takes the serial path at that size
//! (`par_gate::PAR_THRESHOLD` is 250,000 cells), so they run on one core in
//! 484 s and 379 s. A 10,000-cell grid does not fill a modern device on its
//! own, so the design that pays is a batch: a parameter sweep over `q` and
//! seeds shares one set of launches across every run in it.
//!
//! # State
//!
//! Every stage of `volterra_fd::step::update_step_inner` has a kernel here,
//! and each is checked against its CPU counterpart by the `validate` phase:
//! the molecular field and co-rotation, the four boundary conditions, the two
//! stresses, the pressure right-hand side and its Jacobi sweep, the Q and
//! velocity updates, and the integrator. The three differential operators are
//! exposed separately as well, since several stages are built from them.
//!
//! [`DeviceState`] holds every field on the device and runs a whole step
//! across them, stage for stage against `update_step_inner`. Nothing crosses
//! the bus during a step except the pressure loop's convergence measure, which
//! its stopping rule needs on the host. That measure is summed over fixed
//! spans and added up in index order rather than through device atomics, so
//! the sweep count cannot vary between runs of the same binary.
//!
//! Over 5,000 steps from the golden run's own initial condition, the device
//! and the CPU agree to around `1e-15` of each field's own range, flat rather
//! than growing, and the sweep count matches on every step.
//!
//! What remains is the batch. One 100x100 grid is 10,000 cells and will not
//! fill this device; a sweep over `q` and seeds carried as a batch dimension
//! shares one set of launches across every run in it.
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
mod state;

pub use device::{Device, DeviceBoundary};
pub use error::CudaError;
pub use state::{DeviceState, StepParams};
