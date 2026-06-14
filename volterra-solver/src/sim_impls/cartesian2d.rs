//! `PhysicsStep` implementations for the flat Cartesian 2D runners.

use volterra_core::sim::noise::LangevinNoise;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::PhysicsStep;
use volterra_core::{ActiveNematicParams, Integrator};
use volterra_fields::QField2D;
use crate::{beris_edwards_rhs, RK4Integrator};

/// Dry (no-flow) active nematic on a flat 2D grid: RK4 Beris-Edwards + Langevin.
///
/// Reproduces the legacy `run_dry_active_nematic` loop bit-for-bit:
/// - RK4 step via `RK4Integrator` with a per-step cloned params (matching the
///   legacy `let p = params.clone(); move |q_| ...` closure capture).
/// - Langevin noise via `LangevinNoise::fill_pairs` on the flat field buffer,
///   drawn from the same single per-run RNG stream as the legacy inline loop.
pub struct Cartesian2DDry {
    /// Physics parameters.
    pub params: ActiveNematicParams,
    /// Per-run noise source (seeded once, drawn across all steps).
    pub noise: Option<LangevinNoise>,
}

impl PhysicsStep for Cartesian2DDry {
    type Field = QField2D;

    fn step(&mut self, q: &mut QField2D, _t: f64) -> StepStats {
        // RK4 step: clone params once per step into the closure, exactly as
        // the legacy `let p = params.clone(); move |q_| beris_edwards_rhs(q_, None, &p)`.
        let p = self.params.clone();
        let integrator = RK4Integrator;
        *q = integrator.step(q, self.params.dt, move |q_| beris_edwards_rhs(q_, None, &p));

        // Langevin noise injection: fill a flat buffer of pairs then add back,
        // reproducing the legacy `for [q1, q2] in iter.by_ref()` loop order.
        if let Some(noise) = self.noise.as_mut() {
            // q.q is Vec<[f64; 2]>; one pair per cell -> buf len = n_cells * 2.
            let n_cells = q.q.len();
            let mut buf = vec![0.0f64; n_cells * 2];
            noise.fill_pairs(&mut buf, self.params.noise_amp, self.params.dt);
            for (i, [q1, q2]) in q.q.iter_mut().enumerate() {
                *q1 += buf[2 * i];
                *q2 += buf[2 * i + 1];
            }
        }

        // Return only order_param; the wrapper reconstructs the full SnapStats
        // including defect breakdown from the post-step field.
        StepStats::default().with_order_param(q.mean_order_param())
    }
}
