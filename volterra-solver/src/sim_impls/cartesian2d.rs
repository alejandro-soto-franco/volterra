//! `PhysicsStep` implementations for the flat Cartesian 2D runners.

use volterra_core::sim::noise::LangevinNoise;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::PhysicsStep;
use volterra_core::{ActiveNematicParams, Integrator};
use volterra_fields::{QField2D, ScalarField2D};
use crate::{beris_edwards_rhs, RK4Integrator, stokes_solve, k0_convolution, ch_step_etd};

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

/// Wet active nematic on a flat 2D grid: Stokes solve + Euler Beris-Edwards + Langevin.
///
/// Reproduces the legacy `run_active_nematic_hydro` loop bit-for-bit:
/// - Stokes solve each step, then Euler advance `q = q.add(&dq.scale(dt))`.
/// - `beris_edwards_rhs` called with a per-step cloned params (matching the
///   legacy `let p = params.clone(); let dq = beris_edwards_rhs(&q, Some(&v), &p);`).
/// - Langevin noise via `LangevinNoise::fill_pairs` on the flat field buffer,
///   drawn from the same single per-run RNG stream as the legacy inline loop.
pub struct Cartesian2DWet {
    /// Physics parameters.
    pub params: ActiveNematicParams,
    /// Per-run noise source (seeded once, drawn across all steps).
    pub noise: Option<LangevinNoise>,
}

impl PhysicsStep for Cartesian2DWet {
    type Field = QField2D;

    fn step(&mut self, q: &mut QField2D, _t: f64) -> StepStats {
        // Stokes solve: compute velocity from active stress.
        let v = stokes_solve(q, &self.params);
        // Euler step: clone params once per step into `p`, exactly as the legacy
        // `let p = params.clone(); let dq = beris_edwards_rhs(&q, Some(&v), &p);`.
        let p = self.params.clone();
        let dq = beris_edwards_rhs(q, Some(&v), &p);
        *q = q.add(&dq.scale(self.params.dt));

        // Langevin noise injection: fill a flat buffer of pairs then add back,
        // reproducing the legacy `for [q1, q2] in q.q.iter_mut()` loop order.
        if let Some(noise) = self.noise.as_mut() {
            let n_cells = q.q.len();
            let mut buf = vec![0.0f64; n_cells * 2];
            noise.fill_pairs(&mut buf, self.params.noise_amp, self.params.dt);
            for (i, [q1, q2]) in q.q.iter_mut().enumerate() {
                *q1 += buf[2 * i];
                *q2 += buf[2 * i + 1];
            }
        }

        StepStats::default().with_order_param(q.mean_order_param())
    }
}

/// Coupled rotor-Q + lipid-fraction state for the BECH runner.
pub struct BechState {
    /// Rotor Q-tensor field.
    pub q: QField2D,
    /// Lipid volume-fraction field.
    pub phi: ScalarField2D,
}

/// BECH runner: Stokes + Euler Beris-Edwards + Langevin + K0 convolution + CH-ETD.
///
/// Reproduces the legacy `run_bech` loop bit-for-bit:
/// - Stokes solve, then Euler step with `beris_edwards_rhs(&q, Some(&v), params)`
///   (params passed directly, not cloned, matching the legacy call site).
/// - Langevin noise via `LangevinNoise::fill_pairs`.
/// - K0 convolution then CH-ETD step.
pub struct Cartesian2DBech {
    /// Physics parameters.
    pub params: ActiveNematicParams,
    /// Per-run noise source (seeded once, drawn across all steps).
    pub noise: Option<LangevinNoise>,
}

impl PhysicsStep for Cartesian2DBech {
    type Field = BechState;

    fn step(&mut self, st: &mut BechState, _t: f64) -> StepStats {
        // 1. Stokes: velocity from active stress.
        let v = stokes_solve(&st.q, &self.params);

        // 2. Beris-Edwards Euler step (rotor field).
        // Legacy passes `params` directly: `beris_edwards_rhs(&q, Some(&v), params)`.
        let dq = beris_edwards_rhs(&st.q, Some(&v), &self.params);
        st.q = st.q.add(&dq.scale(self.params.dt));

        // 3. Langevin noise injection into Q^rot.
        if let Some(noise) = self.noise.as_mut() {
            let n_cells = st.q.q.len();
            let mut buf = vec![0.0f64; n_cells * 2];
            noise.fill_pairs(&mut buf, self.params.noise_amp, self.params.dt);
            for (i, [q1, q2]) in st.q.q.iter_mut().enumerate() {
                *q1 += buf[2 * i];
                *q2 += buf[2 * i + 1];
            }
        }

        // 4. Transfer map: Q^lip = K0 * Q^rot.
        let q_lip = k0_convolution(&st.q, &self.params);

        // 5. CH-ETD1 step (lipid concentration field).
        st.phi = ch_step_etd(&st.phi, &q_lip, &v, &self.params);

        StepStats::default().with_order_param(st.q.mean_order_param())
    }
}
