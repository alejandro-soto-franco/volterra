//! `PhysicsStep` adapter routing the 2D solver through the shared simulation loop.

use volterra_core::sim::PhysicsStep;
use volterra_core::sim::stats::StepStats;
use crate::{Params, boundary::Boundary, step::{update_step_inner, State}};

/// Wraps one finite-difference advance as a `PhysicsStep`.
pub struct FdStep {
    /// Solver parameters.
    pub params: Params,
    /// Nephroid boundary geometry.
    pub boundary: Boundary,
    /// Pressure-Poisson relative-change relaxation target.
    pub target_rel_change: f64,
}

impl PhysicsStep for FdStep {
    type Field = State;

    fn step(&mut self, state: &mut State, _t: f64) -> StepStats {
        let _p_iters = update_step_inner(state, &self.params, &self.boundary, self.target_rel_change);
        StepStats::default()
    }
}
