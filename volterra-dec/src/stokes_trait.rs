//! Stokes solver trait and backends for active nematic simulations.
//!
//! Two backends:
//! - `StreamFunctionSolver`: existing modified biharmonic (2-manifolds only, fast).
//! - `KillingOperatorSolver`: augmented Lagrangian from cartan-dec (any dimension).

use nalgebra::SVector;

use cartan_core::Manifold;
use cartan_dec::mesh::Mesh;
use cartan_dec::stokes::{StokesSolverAL, StokesResult};

use crate::curved_stokes::CurvedStokesSolver;

/// Flow field result from a Stokes solve.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Velocity: R^3-valued per vertex (length 3*nv for extrinsic,
    /// or tangent-valued per vertex for stream function).
    pub velocity_3d: Vec<[f64; 3]>,
    /// Divergence residual (only meaningful for Killing backend).
    pub div_residual: f64,
}

/// Which Stokes solver backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StokesBackend {
    /// Stream function formulation (2-manifolds only, direct solve).
    StreamFunction,
    /// Killing operator + augmented Lagrangian (any dimension, iterative).
    KillingOperator,
}

/// Trait for Stokes solver backends.
pub trait StokesSolver {
    /// Solve for the velocity field given a force per vertex.
    ///
    /// `force_3d` is R^3-valued per vertex: force_3d[v] = [fx, fy, fz].
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField;
}

/// Killing operator Stokes solver (wraps cartan-dec's StokesSolverAL).
pub struct KillingOperatorSolver {
    inner: StokesSolverAL,
    n_vertices: usize,
}

impl KillingOperatorSolver {
    /// Create a new Killing operator solver for a mesh.
    pub fn new<M: Manifold<Point = SVector<f64, 3>>>(
        mesh: &Mesh<M, 3, 2>,
        penalty: f64,
        tolerance: f64,
    ) -> Self {
        let nv = mesh.n_vertices();
        let inner = StokesSolverAL::new(mesh, penalty, tolerance, 100, 1000);
        Self { inner, n_vertices: nv }
    }
}

impl StokesSolver for KillingOperatorSolver {
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField {
        let nv = self.n_vertices;
        assert_eq!(force_3d.len(), nv);

        // Flatten to 3*nv vector.
        let mut force_flat = vec![0.0; 3 * nv];
        for v in 0..nv {
            force_flat[v * 3] = force_3d[v][0];
            force_flat[v * 3 + 1] = force_3d[v][1];
            force_flat[v * 3 + 2] = force_3d[v][2];
        }

        let result = self.inner.solve(&force_flat);

        // Unflatten velocity.
        let mut velocity_3d = vec![[0.0; 3]; nv];
        for v in 0..nv {
            velocity_3d[v][0] = result.velocity[v * 3];
            velocity_3d[v][1] = result.velocity[v * 3 + 1];
            velocity_3d[v][2] = result.velocity[v * 3 + 2];
        }

        FlowField {
            velocity_3d,
            div_residual: result.div_residual,
        }
    }
}

/// Stream function Stokes solver (wraps existing CurvedStokesSolver).
///
/// Only works on 2-manifolds. Uses the modified biharmonic factorisation
/// with direct LDL^T solve.
pub struct StreamFunctionStokes {
    inner: CurvedStokesSolver,
    n_vertices: usize,
}

impl StreamFunctionStokes {
    /// Create from an existing CurvedStokesSolver.
    pub fn new(solver: CurvedStokesSolver, n_vertices: usize) -> Self {
        Self {
            inner: solver,
            n_vertices,
        }
    }
}

impl StokesSolver for StreamFunctionStokes {
    fn solve(&mut self, force_3d: &[[f64; 3]]) -> FlowField {
        // The stream function solver works with vorticity (scalar).
        // For now, pass through as-is. The caller is responsible for
        // converting the active stress to the appropriate vorticity source.
        // This is a placeholder that returns the force as "velocity"
        // to maintain the trait interface. The full integration with the
        // existing CurvedStokesSolver's vorticity-based API will be done
        // when the engine is wired up.
        FlowField {
            velocity_3d: force_3d.to_vec(),
            div_residual: 0.0,
        }
    }
}
