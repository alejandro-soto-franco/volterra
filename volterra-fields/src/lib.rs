//! # volterra-fields
//!
//! Tensor field types for active nematics, discretized over a DEC mesh.
//!
//! Fields are stored in structure-of-arrays layout: each independent tensor
//! component occupies a contiguous `Vec<f64>` indexed by mesh vertex (or edge,
//! face, depending on the form degree). This layout enables SIMD vectorization
//! and minimizes cache pressure during stencil evaluation.
//!
//! ## Fields
//!
//! - [`QField`] -- the nematic order parameter: a symmetric traceless rank-2
//!   tensor field on mesh vertices. In d dimensions, Q has d(d+1)/2 - 1
//!   independent components.
//!
//! - [`VelocityField`] -- incompressible velocity: a vector field on mesh vertices,
//!   constrained to satisfy discrete divergence-free condition.
//!
//! - [`PressureField`] -- scalar pressure field on mesh vertices.
//!
//! - [`MolecularField`] -- H = -delta F / delta Q, derived from Q via the
//!   Landau-de Gennes free energy functional. Same type as QField.
//!
//! ## Derived quantities
//!
//! - [`DefectCharge`] -- topological charge density, computed from the winding
//!   of the Q eigenvector field around mesh faces.
//!
//! ## Planned components
//!
//! - `QField`, `VelocityField`, `PressureField`, `MolecularField`
//! - `ActiveStress` -- sigma^active = -zeta * Q
//! - `EricksenStress` -- elastic stress derived from F[Q]
//! - `DefectCharge` -- topological charge density per face
