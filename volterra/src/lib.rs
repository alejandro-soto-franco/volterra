// ~/volterra/volterra/src/lib.rs

//! # volterra
//!
//! Covariant active nematics simulation on Riemannian manifolds.
//!
//! volterra solves the Beris-Edwards nematohydrodynamics equations for active
//! liquid crystals. It is built on `cartan`, a Riemannian geometry library,
//! and uses discrete exterior calculus (DEC) to discretise covariant
//! differential operators on arbitrary manifolds.
//!
//! ## Crate structure
//!
//! ```text
//! volterra          -- facade crate (use this)
//! volterra-core     -- trait definitions, parameters and error types
//! volterra-fd       -- finite-difference physics on a lattice, 2D and 3D
//! volterra-dec      -- DEC mesh physics: Hodge operators, confined domains, Stokes
//! volterra-braid    -- braid words, tracking and topological entropy
//! volterra-py       -- Python bindings, published as a wheel rather than to crates.io
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use volterra::prelude::*;
//!
//! // Build parameters for a 128x128 active turbulent simulation.
//! let mut params = ActiveNematicParams::default_test();
//! params.nx = 128;
//! params.ny = 128;
//! params.zeta_eff = 3.0;
//! params.validate().expect("valid params");
//!
//! // Start from a small random perturbation.
//! let q_init = QField2D::random_perturbation(params.nx, params.ny, params.dx, 0.001, 42);
//!
//! // Run 1000 steps, snapshot every 100.
//! let (_q_final, stats) = run_dry_active_nematic(&q_init, &params, 1000, 100);
//! for s in &stats {
//!     println!("t={:.2}  S={:.4}  defects={}", s.time, s.mean_s, s.n_defects);
//! }
//! ```

pub mod cli;

pub use volterra_core as core;
pub use volterra_core::fields;
pub use volterra_fd as fd;

pub mod prelude {
    pub use volterra_core::{ActiveNematicParams, VError};
    pub use volterra_core::{QField2D, VelocityField2D};
    pub use volterra_fd::{
        DefectInfo, SnapStats, beris_edwards_rhs, defect_count, k0_convolution,
        molecular_field, run_dry_active_nematic, scan_defects,
        EulerIntegrator, RK4Integrator,
    };
}
