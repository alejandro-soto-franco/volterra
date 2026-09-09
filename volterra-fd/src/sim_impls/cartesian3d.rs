//! `PhysicsStep` implementations for the 3D Cartesian runners.
//!
//! The 3D runners use a per-step-seeded RNG convention (unlike the 2D
//! per-run convention). Each step creates a fresh `LangevinNoise` from
//! the step index and a runner-specific tag, then draws 5 independent
//! N(0,1) samples per vertex from that single per-step RNG.

use volterra_core::sim::noise::LangevinNoise;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::PhysicsStep;
use volterra_core::ActiveNematicParams3D;
use volterra_core::{QField3D, ScalarField3D, VelocityField3D};

use crate::beris_3d::{beris_edwards_rhs_3d, EulerIntegrator3D};
use crate::ch_3d::ch_step_etd_3d;
use crate::stokes_3d::stokes_solve_3d;

// ─────────────────────────────────────────────────────────────────────────────
// Dry 3D runner
// ─────────────────────────────────────────────────────────────────────────────

/// Dry 3D active nematic: fused Euler Beris-Edwards + per-step-seeded Langevin noise.
///
/// Noise convention: one `LangevinNoise::per_step_seed(step, 0xdead_beef_cafe_1234)`
/// per step (outside the vertex loop), then `fill5` per vertex. This matches the
/// legacy `SmallRng::seed_from_u64(step as u64 ^ 0xdead_beef_cafe_1234)` exactly.
pub struct Cartesian3DDry {
    /// Physics parameters.
    pub params: ActiveNematicParams3D,
    /// Current step counter (incremented each `PhysicsStep::step` call).
    pub step_idx: usize,
}

impl PhysicsStep for Cartesian3DDry {
    type Field = QField3D;

    fn step(&mut self, q: &mut QField3D, _t: f64) -> StepStats {
        let step = self.step_idx;
        let t = step as f64 * self.params.dt;
        let p = &self.params;

        // 1+2. Fused molecular field + Euler step.
        crate::mol_field_3d::euler_step_fused_par(q, p, t);

        // 3. Langevin noise: one RNG per step, drawn per vertex.
        if p.noise_amp > 0.0 {
            let amp = p.noise_amp * p.dt.sqrt();
            let n_verts = q.len();
            let mut noise = LangevinNoise::per_step_seed(step, 0xdead_beef_cafe_1234);
            for k in 0..n_verts {
                let mut samples = [0.0f64; 5];
                noise.fill5(&mut samples);
                for c in 0..5 {
                    q.q[k][c] += amp * samples[c];
                }
            }
        }

        self.step_idx += 1;
        StepStats::default().with_order_param(q.mean_s())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BECH 3D runner
// ─────────────────────────────────────────────────────────────────────────────

/// 3D BECH state: Q-tensor + lipid concentration + most recent velocity.
///
/// The velocity field is updated in-place each step so that snapshot writers
/// can access the velocity at the moment of the snapshot without re-solving Stokes.
pub struct BechState3D {
    /// Rotor Q-tensor field.
    pub q: QField3D,
    /// Lipid volume-fraction field.
    pub phi: ScalarField3D,
    /// Velocity field from the most recent Stokes solve (for snapshot writing).
    pub vel: VelocityField3D,
}

/// The state a wet run advances: the tensor field and the flow it drives.
#[derive(Debug, Clone)]
pub struct WetState3D {
    /// Q-tensor field.
    pub q: QField3D,
    /// Velocity from the most recent Stokes solve.
    pub vel: VelocityField3D,
}

/// 3D wet active nematic: Stokes, then Beris-Edwards in the flow it found.
///
/// The dry stepper solves no flow and advects nothing. This one solves the
/// steady Stokes problem driven by the active stress `-zeta Q` at every step and
/// hands the velocity to the Beris-Edwards right-hand side, so the texture is
/// advected and sheared by a flow its own activity generates. Switching the
/// activity off returns the dry stepper exactly, which is what
/// `tests/wet_3d.rs` asserts.
///
/// Lipids are absent, which is the difference from [`Cartesian3DBech`]: this is
/// the two-field problem rather than the three-field one.
pub struct Cartesian3DWet {
    /// Physics parameters.
    pub params: ActiveNematicParams3D,
    /// Current step counter.
    pub step_idx: usize,
}

impl PhysicsStep for Cartesian3DWet {
    type Field = WetState3D;

    fn step(&mut self, st: &mut WetState3D, _t: f64) -> StepStats {
        let step = self.step_idx;
        let p = &self.params;
        let t = step as f64 * p.dt;

        let (vel, _pressure) = stokes_solve_3d(&st.q, p);
        let rhs = beris_edwards_rhs_3d(&st.q, Some(&vel), p, t);
        st.q = EulerIntegrator3D.step(&st.q, p.dt, &rhs);

        if p.noise_amp > 0.0 {
            let amp = p.noise_amp * p.dt.sqrt();
            let mut noise = LangevinNoise::per_step_seed(step, 0xdead_beef_cafe_5678);
            for k in 0..st.q.len() {
                let mut samples = [0.0f64; 5];
                noise.fill5(&mut samples);
                for c in 0..5 {
                    st.q.q[k][c] += amp * samples[c];
                }
            }
        }

        st.vel = vel;
        self.step_idx += 1;
        StepStats::default().with_order_param(st.q.mean_s())
    }
}

/// 3D BECH runner: Stokes + Euler Beris-Edwards + Langevin noise + CH-ETD.
///
/// Noise convention: one `LangevinNoise::per_step_seed(step, 0xdead_beef_cafe_5678)`
/// per step, drawn per vertex. Matches the legacy tag exactly.
pub struct Cartesian3DBech {
    /// Physics parameters.
    pub params: ActiveNematicParams3D,
    /// Current step counter (incremented each call).
    pub step_idx: usize,
}

impl PhysicsStep for Cartesian3DBech {
    type Field = BechState3D;

    fn step(&mut self, st: &mut BechState3D, _t: f64) -> StepStats {
        let step = self.step_idx;
        let p = &self.params;
        let t = step as f64 * p.dt;

        // 1. Stokes solve: get incompressible velocity driven by active stress.
        let (vel, _pressure) = stokes_solve_3d(&st.q, p);

        // 2. Beris-Edwards RHS with flow.
        let rhs = beris_edwards_rhs_3d(&st.q, Some(&vel), p, t);

        // 3. Euler step on Q.
        let euler = EulerIntegrator3D;
        st.q = euler.step(&st.q, p.dt, &rhs);

        // 4. Langevin noise: one RNG per step, drawn per vertex.
        if p.noise_amp > 0.0 {
            let amp = p.noise_amp * p.dt.sqrt();
            let n_verts = st.q.len();
            let mut noise = LangevinNoise::per_step_seed(step, 0xdead_beef_cafe_5678);
            for k in 0..n_verts {
                let mut samples = [0.0f64; 5];
                noise.fill5(&mut samples);
                for c in 0..5 {
                    st.q.q[k][c] += amp * samples[c];
                }
            }
        }

        // 5. Cahn-Hilliard ETD step.
        //    Approximation: pass &q as q_lip (see runner_3d doc for rationale).
        st.phi = ch_step_etd_3d(&st.phi, &st.q, p, p.dt);

        // Store velocity for snapshot access.
        st.vel = vel;

        self.step_idx += 1;
        StepStats::default().with_order_param(st.q.mean_s())
    }
}
