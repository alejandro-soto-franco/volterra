//! `PhysicsStep` implementations for the volterra-solver runners.
//!
//! Each submodule provides a concrete `PhysicsStep` that encapsulates one
//! runner's physics and noise, allowing `SimulationRunner` to drive the loop
//! while the caller's sink handles snapshot collection.

pub mod cartesian2d;
pub mod cartesian3d;
