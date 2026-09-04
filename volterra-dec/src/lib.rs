#![allow(clippy::needless_range_loop)]
// Indexed loops are the clearer form in stencil code, where one index reads
// several arrays at once. `volterra-solver` carried this allow at its crate
// root, and the physics that moved here came with it.

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
//! | [`qfield`] | `QField`: Q-tensor field (q1, q2 real components) |
//! | [`curved_stokes`] | `CurvedStokesSolver`: stream-function biharmonic on curved 2-manifolds |
//! | [`stokes`] | `SurfaceStokes`, `VelocityField`, vorticity source, velocity from psi |
//! | [`semi_lagrangian`] | `SemiLagrangian`: BVH-accelerated advection with RK4 + deformation gradient |
//! | [`connection_laplacian`] | Covariant Laplacian for spin-2 fields |
//! | [`molecular_field_dec`] | Landau-de Gennes molecular field on DEC meshes |
//! | [`bending`] | Discrete Helfrich energy and its exact vertex gradient |
//! | [`flow`] | Semi-implicit time stepping for overdamped Helfrich flow |
//! | [`helfrich`] | Superseded shape-equation forces; see [`bending`] |
//! | [`variational`] | BAOAB splitting integrator for membrane dynamics |
//! | [`curve`] | `PlaneCurve`: the boundary geometry a confined mesh conforms to |
//! | [`mesh_gen`] | Icosphere, torus, and epitrochoid mesh generators |
//! | [`poisson`] | Precomputed LDL^T Poisson solver |
//! | [`boundary_conditions`] | Boundary condition handling |
//! | [`curvature_correction`] | Curvature corrections for DEC operators |
//! | [`snapshot`] | `.npy` field snapshot export |

pub mod bending;
pub mod boundary_conditions;
pub mod flow;
pub mod connection_laplacian;
pub mod mesh_gen;
pub mod semi_lagrangian;
pub mod snapshot;
pub mod surface_defects;
pub mod curvature_correction;
pub mod domain;
pub mod implicit;
pub mod confined;
pub mod curve;
pub mod confined_ldg;
pub mod nematic_params;
pub mod epitrochoid;
pub mod evolving_domain;
pub mod ichol;
pub mod poisson;
pub mod stokes;
pub mod tracers;
pub mod helfrich;
pub mod molecular_field_dec;
pub mod qfield;
pub mod variational;

// The engine layer and the runners that drive it, moved here from
// volterra-solver. Each is written against DEC types, so this is where they
// belong: `engine` reaches for the connection Laplacian, the curved Stokes
// solver and semi-Lagrangian advection, and the runners drive those.
pub mod engine;
pub use engine::{NematicEngine, EngineStats};

pub mod active_nematic_engine;
pub use active_nematic_engine::{ActiveNematicEngine, EngineParams, StepDiagnostics};

pub mod stokes_trait;
pub use stokes_trait::{
    StokesSolver, StokesBackend, FlowField, KillingOperatorSolver, StreamFunctionStokes,
};

pub mod nematic_field_2d;
pub use nematic_field_2d::NematicField2D;

pub mod runner_dec;
pub use runner_dec::{run_dry_active_nematic_dec, run_dry_active_nematic_dec_smoke, SnapStatsDec};

pub mod runner_dec_wet;
pub use runner_dec_wet::{run_wet_active_nematic_dec, run_wet_active_nematic_dec_confined};

pub mod sim_impls;

pub use domain::DecDomain;
pub use evolving_domain::EvolvingDomain;
pub use molecular_field_dec::molecular_field_dec;
pub use curve::{PlaneCurve, PolyCurve};
pub use qfield::QField;
