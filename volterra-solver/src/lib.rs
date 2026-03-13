//! # volterra-solver
//!
//! Beris-Edwards nematohydrodynamics solver for active nematics on Riemannian manifolds.
//!
//! This crate implements the equations of motion, time integration, and defect
//! tracking for a covariant active nematic. All differential operators are provided
//! by `volterra-dec` and all field types by `volterra-fields`.
//!
//! ## Equations of motion
//!
//! The solver integrates the covariant Beris-Edwards equations:
//!
//! ```text
//! (d/dt + u . nabla) Q - S(W, Q) = Gamma * H
//! ```
//!
//! where `S(W, Q)` encodes co-rotation and strain alignment with flow parameter
//! lambda, and `H = -delta F / delta Q` is the molecular field derived from the
//! Landau-de Gennes free energy `F[Q]`.
//!
//! Hydrodynamics are governed by the incompressible Navier-Stokes equation on the
//! manifold with nematic body force:
//!
//! ```text
//! rho (d/dt + u . nabla) u = -nabla p + eta Delta u + nabla . sigma
//! nabla . u = 0
//! ```
//!
//! The stress tensor decomposes as:
//!
//! ```text
//! sigma = sigma_viscous + sigma_elastic + sigma_antisym + sigma_active
//! sigma_active = -zeta * Q
//! ```
//!
//! Active extensile systems have zeta > 0; contractile systems have zeta < 0.
//!
//! All derivatives are covariant with respect to the Levi-Civita connection of the
//! background Riemannian manifold. The Laplacian on Q is the Lichnerowicz operator:
//! the Bochner (connection) Laplacian plus a curvature correction quadratic in the
//! Riemann tensor.
//!
//! ## Defect tracking
//!
//! `DefectTracker` detects topological defects as zeros of the Q eigenvalue field
//! and tracks their positions, charges, and world-lines across time steps. Defect
//! world-lines can be exported as braid words in the mapping class group of the
//! punctured domain.
//!
//! ## Time integration
//!
//! - `EulerIntegrator` -- first-order forward Euler (fast, for exploratory runs)
//! - `RK4Integrator` -- fourth-order Runge-Kutta (recommended for production runs)
//!
//! ## Planned components
//!
//! - `SimulationParams` -- all physical parameters (K, zeta, Gamma, eta, lambda, ...)
//! - `EulerIntegrator`, `RK4Integrator`
//! - `BerisEdwardsRhs` -- right-hand side evaluation
//! - `PressureSolver` -- discrete Poisson solve for the incompressibility constraint
//! - `DefectTracker` -- real-time defect detection and world-line extraction
