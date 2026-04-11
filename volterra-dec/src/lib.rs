//! # volterra-dec
//!
//! Discrete exterior calculus (DEC) layer for volterra.
//!
//! Bridges cartan-dec geometry to the physics solver. Provides the `DecDomain`
//! bundle, Stokes solvers (stream function and flat), semi-Lagrangian advection,
//! Q-tensor field types, Helfrich membrane energy, and variational integrators.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`domain`] | `DecDomain`: mesh + precomputed DEC operators + curvature |
//! | [`qfield_dec`] | `QFieldDec`: Q-tensor field (q1, q2 real components) |
//! | [`curved_stokes`] | `CurvedStokesSolver`: stream-function biharmonic on curved 2-manifolds |
//! | [`stokes_dec`] | `StokesSolverDec`, `VelocityFieldDec`, vorticity source, velocity from psi |
//! | [`semi_lagrangian`] | `SemiLagrangian`: BVH-accelerated advection with RK4 + deformation gradient |
//! | [`connection_laplacian`] | Covariant Laplacian for spin-2 fields |
//! | [`molecular_field_dec`] | Landau-de Gennes molecular field on DEC meshes |
//! | [`helfrich`] | Helfrich bending energy and forces |
//! | [`variational`] | BAOAB splitting integrator for membrane dynamics |
//! | [`mesh_gen`] | Icosphere, torus, and epitrochoid mesh generators |
//! | [`poisson`] | Precomputed LDL^T Poisson solver |
//! | [`boundary_conditions`] | Boundary condition handling |
//! | [`curvature_correction`] | Curvature corrections for DEC operators |
//! | [`snapshot`] | `.npy` field snapshot export |

pub mod boundary_conditions;
pub mod connection_laplacian;
pub mod curved_stokes;
pub mod mesh_gen;
pub mod semi_lagrangian;
pub mod snapshot;
pub mod curvature_correction;
pub mod domain;
pub mod epitrochoid;
pub mod evolving_domain;
pub mod poisson;
pub mod stokes_dec;
pub mod helfrich;
pub mod molecular_field_dec;
pub mod qfield_dec;
pub mod variational;

pub use domain::DecDomain;
pub use evolving_domain::EvolvingDomain;
pub use molecular_field_dec::molecular_field_dec;
pub use qfield_dec::QFieldDec;
