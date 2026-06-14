//! Generic simulation runner core.
//!
//! One canonical time-stepping loop (`SimulationRunner::run`) plus the reusable
//! building blocks every runner shares: seeded Langevin noise, generic
//! integrators, a unified per-step statistics currency, and snapshot sinks.
//! This module is dependency-free so it can live in the lowest crate.

pub mod noise;
pub mod integrate;
pub mod stats;
pub mod snapshot;

mod runner;
pub use runner::{Observer, PhysicsStep, RunConfig, SimulationRunner};
