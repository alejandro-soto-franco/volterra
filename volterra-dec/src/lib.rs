//! # volterra-dec
//!
//! Discrete exterior calculus (DEC) layer for volterra.
//!
//! Bridges cartan-dec geometry to the physics solver. Provides the `DecDomain`
//! bundle, Helfrich membrane energy, and BAOAB variational integrator.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`domain`] | `DecDomain` -- mesh + precomputed DEC operators |
//! | [`helfrich`] | Helfrich bending energy and forces |
//! | [`variational`] | BAOAB splitting integrator for membrane dynamics |

pub mod boundary_conditions;
pub mod snapshot;
pub mod curvature_correction;
pub mod domain;
pub mod epitrochoid;
pub mod poisson;
pub mod stokes_dec;
pub mod helfrich;
pub mod molecular_field_dec;
pub mod qfield_dec;
pub mod variational;

pub use domain::DecDomain;
pub use molecular_field_dec::molecular_field_dec;
pub use qfield_dec::QFieldDec;
