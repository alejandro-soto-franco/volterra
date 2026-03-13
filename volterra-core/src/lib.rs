//! # volterra-core
//!
//! Trait definitions, error types, and shared primitives for the volterra framework.
//!
//! This crate defines the interfaces that all other volterra crates program against.
//! It has minimal dependencies and is the only crate that downstream users need to
//! import when writing code that is generic over manifold type or field type.
//!
//! ## Traits
//!
//! - [`NematicField`] -- a symmetric traceless rank-2 tensor field on a manifold
//! - [`VelocityField`] -- an incompressible vector field on a manifold
//! - [`Integrator`] -- a time integration scheme (Euler, RK4, ...)
//! - [`DefectTracker`] -- detects and tracks topological defect positions and charges
//!
//! ## Error types
//!
//! [`VolterraError`] is the unified error type used across all volterra crates.
