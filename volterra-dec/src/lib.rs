//! # volterra-dec
//!
//! Discrete exterior calculus (DEC) layer for volterra.
//!
//! This crate bridges the continuous Riemannian geometry of `cartan-core` to the
//! discrete operators needed by the PDE solver. Given a manifold implementing
//! `cartan_core::Manifold`, it builds a simplicial complex over the domain,
//! precomputes all static operators, and exposes them for use in the time loop.
//!
//! ## Mathematical structure
//!
//! DEC represents smooth differential operators via their discrete analogues on a
//! simplicial complex. The key factorization is:
//!
//! ```text
//! Laplace-Beltrami = d * star * d * star
//! ```
//!
//! where `d` is the exterior derivative (a sparse {0, +1, -1} matrix, metric-free
//! and fixed for a given mesh) and `star` is the Hodge star (diagonal for
//! well-centered meshes, encoding the metric).
//!
//! This factorization gives cache-friendly computation: the expensive metric
//! information lives in a diagonal vector, and all structural traversal is over a
//! fixed sparse {0, +1, -1} incidence matrix.
//!
//! ## Mesh requirements
//!
//! Operators in this crate assume a **well-centered mesh**: every simplex's
//! circumcenter lies in the interior of that simplex. This guarantees positive
//! Hodge weights and a diagonal Hodge star. Mesh generation utilities enforce this
//! constraint via constrained Delaunay triangulation.
//!
//! ## Cache layout
//!
//! Simplices are reordered by Hilbert space-filling curve index so that spatially
//! local simplices are adjacent in memory. Field arrays use structure-of-arrays
//! layout so that each tensor component is a contiguous `Vec<f64>`, enabling SIMD
//! operations across the mesh.
//!
//! ## Planned components
//!
//! - `Mesh` -- simplicial complex with Hilbert-ordered simplices and circumcentric dual
//! - `ExteriorDerivative` -- sparse {0, +1, -1} incidence matrix for each degree k
//! - `HodgeStar` -- diagonal metric weights for each degree k
//! - `BochnerLaplacian` -- connection Laplacian on tensor fields
//! - `LichnerowiczLaplacian` -- Bochner Laplacian plus Riemann curvature correction, for Q
//! - `CovariantAdvection` -- upwind covariant advection operator
