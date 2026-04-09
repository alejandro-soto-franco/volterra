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

pub mod domain;
pub mod helfrich;
pub mod qfield_dec;
pub mod variational;

pub use domain::DecDomain;
pub use qfield_dec::QFieldDec;
