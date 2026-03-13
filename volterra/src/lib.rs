//! # volterra
//!
//! Covariant active nematics simulation on Riemannian manifolds.
//!
//! volterra solves the Beris-Edwards nematohydrodynamics equations for active
//! liquid crystals on curved spaces. It is built on `cartan`, a Riemannian
//! geometry library, and uses discrete exterior calculus (DEC) to discretize
//! covariant differential operators on arbitrary manifolds.
//!
//! ## Crate structure
//!
//! ```text
//! volterra          -- facade crate (use this)
//! volterra-core     -- trait definitions and error types
//! volterra-dec      -- DEC mesh, Hodge operators, covariant derivatives
//! volterra-fields   -- Q-tensor, velocity, pressure, and stress field types
//! volterra-solver   -- equations of motion, time integration, defect tracking
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use volterra::prelude::*;
//!
//! // Define a 2D domain and build a well-centered Delaunay mesh.
//! // Specify physical parameters and run the solver.
//! // Extract defect world-lines for topological analysis.
//! ```
//!
//! ## Physics
//!
//! Active nematics are materials composed of self-driven elongated units -- such
//! as collections of cytoskeletal filaments driven by molecular motors -- that
//! spontaneously develop orientational order and generate internal stresses. The
//! nematic order parameter Q is a symmetric traceless rank-2 tensor field; its
//! topological defects (points in 2D, lines in 3D) are the primary objects of
//! physical and mathematical interest.
//!
//! In confined geometries the interplay between domain curvature, boundary
//! conditions, and active stress drives rich dynamical behavior including
//! spontaneous flows, defect nucleation and annihilation, and long-lived periodic
//! orbits in the defect configuration space.
//!
//! volterra is designed to study this behavior across a family of domain geometries
//! and in arbitrary spatial dimensions, with the long-term goal of characterizing
//! the topological and cohomological structure of the defect phase space.
//!
//! ## Substrate
//!
//! volterra depends on [`cartan`](https://github.com/alejandro-soto-franco/cartan)
//! for Riemannian geometry. Any manifold implementing `cartan_core::Manifold` can
//! serve as the simulation domain.

pub use volterra_core as core;
pub use volterra_dec as dec;
pub use volterra_fields as fields;
pub use volterra_solver as solver;

pub mod prelude {
    pub use volterra_core::*;
    pub use volterra_fields::*;
    pub use volterra_solver::*;
}
