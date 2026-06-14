//! The single canonical simulation loop and its extension points.

use super::stats::StepStats;

/// One operator-split advance of a field by `dt`, returning per-step diagnostics.
pub trait PhysicsStep {
    /// The mutable simulation state this physics advances.
    type Field;
    /// Advance `field` by one step ending at time `t`; return diagnostics.
    fn step(&mut self, field: &mut Self::Field, t: f64) -> StepStats;
}

/// Receives snapshots; concrete sinks accumulate stats or write to disk.
pub trait Observer<F> {
    /// Called at each snapshot point with the current step, time, field, and the
    /// stats of the most recent step (default `StepStats` at step 0).
    fn observe(&mut self, step: usize, t: f64, field: &F, stats: &StepStats);
}

/// Loop configuration shared by every runner.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Number of physics steps to advance.
    pub steps: usize,
    /// Snapshot every `snap_every` steps (and always at step 0).
    pub snap_every: usize,
    /// Time step.
    pub dt: f64,
    /// Base RNG seed (for physics that consume it; legacy runners derive their own).
    pub seed: u64,
}

/// Drives a `PhysicsStep` through `RunConfig` against an `Observer`.
pub struct SimulationRunner {
    /// Loop configuration.
    pub config: RunConfig,
}

impl SimulationRunner {
    /// The single canonical loop:
    /// `for step in 0..=steps { if step % snap_every == 0 { observe }; if step < steps { physics.step() } }`
    pub fn run<P: PhysicsStep>(
        &self,
        field: &mut P::Field,
        physics: &mut P,
        observer: &mut dyn Observer<P::Field>,
    ) {
        let RunConfig { steps, snap_every, dt, .. } = self.config;
        let mut last = StepStats::default();
        for step in 0..=steps {
            if snap_every != 0 && step % snap_every == 0 {
                observer.observe(step, step as f64 * dt, field, &last);
            }
            if step < steps {
                last = physics.step(field, (step + 1) as f64 * dt);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::stats::StepStats;
    use crate::sim::snapshot::StatsSink;

    // A trivial physics: state is a counter, each step increments and reports it.
    struct Counter;
    impl PhysicsStep for Counter {
        type Field = u64;
        fn step(&mut self, field: &mut u64, t: f64) -> StepStats {
            *field += 1;
            StepStats::default().with_time(t).with_order_param(*field as f64)
        }
    }

    #[test]
    fn loop_snapshots_at_cadence_and_steps_exactly_n() {
        let cfg = RunConfig { steps: 4, snap_every: 2, dt: 0.5, seed: 0 };
        let runner = SimulationRunner { config: cfg };
        let mut field = 0u64;
        let mut sink = StatsSink::default();
        runner.run(&mut field, &mut Counter, &mut sink);
        // Snapshots at steps 0,2,4 -> 3 snapshots; field advanced exactly 4 times.
        assert_eq!(field, 4);
        assert_eq!(sink.snapshots.len(), 3);
        // Step 0 snapshot uses last = StepStats::default(), so order_param is None
        // (Counter has not stepped yet; the field has not been observed by physics).
        assert_eq!(sink.snapshots[0].order_param, None);
        // Step 2 snapshot: Counter has stepped twice, field==2, reported at step 2.
        assert_eq!(sink.snapshots[1].order_param, Some(2.0));
        // Step 4 snapshot: field==4.
        assert_eq!(sink.snapshots[2].order_param, Some(4.0));
    }
}
