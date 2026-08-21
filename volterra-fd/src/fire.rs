//! FIRE (Fast Inertial Relaxation Engine) minimiser for the 3D passive
//! Landau-de Gennes free energy, against the same molecular field the
//! Euler time stepper already computes (`beris_edwards_rhs_3d_par_dry`).
//!
//! Ported step for step from open-Qmin's reference implementation
//! (`src/updaters/energyMinimizerFIRE.{h,cpp}`, `velocityVerlet.cpp`), so the
//! two codes are measured against the same stopping quantity. In particular
//! `force_max` is **not** an elementwise maximum despite the name inherited
//! from open-Qmin's `getMaxForce()`: it is `sqrt(sum_i |f_i|^2) / N`, the
//! root-sum-square of the per-site force vector divided by the site count.
//! Matching this exactly (rather than a literal per-site max) is what makes
//! the "time to reach 1e-3" comparison in `BENCHMARKS.md` apples to apples.
//!
//! ## Algorithm
//!
//! Each FIRE iteration is one velocity-Verlet step against the force field
//! `f = beris_edwards_rhs_3d_par_dry(q, p, t)` (mass = 1, matching
//! open-Qmin's `velocityVerlet::integrateEOMCPU`):
//!
//! ```text
//! d  = deltaT * v + 0.5 * deltaT^2 * f_old
//! v += 0.5 * deltaT * f_old
//! q += d
//! f_new = force(q)
//! v += 0.5 * deltaT * f_new
//! ```
//!
//! followed by the FIRE velocity mixing and adaptive-timestep rule
//! (`energyMinimizerFIRE::fireStepCPU`):
//!
//! ```text
//! forceNorm    = sum_i |f_i|^2
//! velocityNorm = sum_i |v_i|^2
//! Power        = sum_i f_i . v_i
//! forceMax     = sqrt(forceNorm) / N
//! scaling      = sqrt(velocityNorm / forceNorm)   (0 if forceNorm == 0)
//! v            = (1 - alpha) * v + alpha * scaling * f
//! ```
//!
//! and, on a Power sign check performed every step except every 500th
//! iteration (open-Qmin's own quirk, kept for parity): either accelerate
//! (grow deltaT, shrink alpha) after `n_min` consecutive steps of positive
//! power, or reset (shrink deltaT, restore alpha, zero the velocity).

use rayon::prelude::*;
use volterra_core::ActiveNematicParams3D;
use volterra_core::QField3D;

use crate::beris_3d::{beris_edwards_rhs_3d_par_dry, beris_edwards_rhs_3d_par_dry_into};

/// FIRE tuning parameters, matching open-Qmin's own defaults
/// (`openQmin.cpp`'s `energyMinimizerFIRE` construction) unless overridden.
#[derive(Debug, Clone, Copy)]
pub struct FireParams {
    /// Initial (and per-reset floor-relative) MD timestep.
    pub delta_t: f64,
    /// Minimum timestep: fixed at `0.01 * delta_t` at construction, exactly
    /// as open-Qmin's `setDeltaT` does.
    pub delta_t_min: f64,
    /// Maximum timestep the adaptive rule may grow to.
    pub delta_t_max: f64,
    /// Growth factor applied to `deltaT` on sustained positive power.
    pub delta_t_inc: f64,
    /// Shrink factor applied to `deltaT` on a reset.
    pub delta_t_dec: f64,
    /// Starting value of the velocity-mixing parameter `alpha`, restored on
    /// every reset.
    pub alpha_start: f64,
    /// Shrink factor applied to `alpha` on sustained positive power.
    pub alpha_dec: f64,
    /// Floor `alpha` may not decay below.
    pub alpha_min: f64,
    /// Consecutive positive-power steps required before `deltaT`/`alpha`
    /// start adapting.
    pub n_min: i32,
    /// Stop when `force_max <= force_cutoff`.
    pub force_cutoff: f64,
    /// Hard iteration ceiling.
    pub max_iterations: usize,
}

impl FireParams {
    /// open-Qmin's own default FIRE parameters (`openQmin.cpp`), parameterised
    /// by the initial timestep and the target residual.
    pub fn open_qmin_defaults(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t,
            delta_t_min: delta_t * 0.01,
            delta_t_max: 100.0 * delta_t,
            delta_t_inc: 1.1,
            delta_t_dec: 0.95,
            alpha_start: 0.99,
            alpha_dec: 0.9,
            alpha_min: 0.0,
            n_min: 4,
            force_cutoff,
            max_iterations,
        }
    }

    /// Retuned for volterra's own bulk/elastic constants, which are not
    /// open-Qmin's (see `BENCHMARKS.md`'s "Scale mismatch" caveat): open-Qmin
    /// tuned `delta_t_inc`, `alpha_dec` and `n_min` for its own energy
    /// landscape, and there is no reason those values are optimal for a
    /// different one. A sweep over these three constants
    /// (`volterra-solver/examples/sweep_fire_params.rs`), holding the
    /// initial condition and every other parameter fixed, cuts the
    /// scale-matched-target step count at N=100 from 43 to 19 (56%),
    /// consistently across four random seeds (not tuned to one IC). Grows
    /// the adaptive timestep faster (`delta_t_inc`), decays `alpha` faster on
    /// a good run (`alpha_dec`), and drops the run-up delay before adapting
    /// at all (`n_min = 0`).
    pub fn volterra_tuned(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t_inc: 1.6,
            alpha_dec: 0.7,
            n_min: 0,
            ..Self::open_qmin_defaults(delta_t, force_cutoff, max_iterations)
        }
    }

    /// Retuned again for the matched-physics landscape (`a_eff, b_landau,
    /// c_landau, k_r` set to open-Qmin's own defaults under the mapping in
    /// `mol_field_3d.rs`'s module header): `volterra_tuned` above was tuned
    /// on volterra's pre-cubic-term landscape, a different energy surface,
    /// so there is no reason it stays optimal on this one. The identical
    /// sweep procedure (`examples/sweep_fire_params_matched.rs`), pushed as
    /// far past its own swept ranges as the matching open-Qmin sweep was,
    /// found the CPU
    /// sweep's own plateau near `delta_t_inc=3.5, alpha_dec=0.99` (10 steps
    /// at the literal `1e-3` target). Every point with `alpha_dec` pushed
    /// up near open-Qmin's own `alpha_start=0.99` (`2.2` through `3.5`) sits
    /// close enough to a chaotic bifurcation in FIRE's own adaptive
    /// timestep rule, since `alpha` barely decaying keeps the dynamics
    /// inertia-dominated, i.e. closer to conservative MD, far longer, and
    /// conservative MD is exponentially sensitive to rounding -- that the
    /// GPU reduction's run-to-run atomic accumulation order (not fixed on
    /// real hardware) flips the outcome of the N=8 tight-tolerance
    /// GPU-vs-CPU correctness check between repeated runs of the *same*
    /// binary (`2.2` passed on one run, failed the next five, CPU/GPU
    /// iteration counts one apart either way, 2-4e-9 against the 1e-9
    /// tolerance). A correctness check that nondeterministically passes
    /// checks nothing. Holding `alpha_dec=0.7` fixed at `volterra_tuned`'s
    /// own value (proven stable at N=8 across six repeated device runs,
    /// agreement ~1e-16 every time) and pushing only `delta_t_inc` stays
    /// stable up to at least `3.0`; `delta_t_inc=2.5` passed the N=8 check
    /// on six repeated device runs with that same ~1e-16 margin and is used
    /// here: 16 steps at the literal `1e-3` target (matched physics needs
    /// no scale-matching
    /// construction), against `volterra_tuned`'s own 20 and the untuned
    /// baseline's 56, robust across the same four seeds.
    pub fn matched_tuned(delta_t: f64, force_cutoff: f64, max_iterations: usize) -> Self {
        Self {
            delta_t_inc: 2.5,
            alpha_dec: 0.7,
            n_min: 0,
            ..Self::open_qmin_defaults(delta_t, force_cutoff, max_iterations)
        }
    }
}

/// Mutable FIRE integration state, carried across steps.
pub struct FireState {
    pub v: QField3D,
    /// The velocity mix owed from the previous iteration, `1 - alpha` and
    /// `alpha * scaling`, applied by the next position pass rather than in a
    /// pass of its own. Both zero on a reset, which is what zeroing `v` did.
    pub mix_v: f64,
    pub mix_f: f64,
    pub delta_t: f64,
    pub alpha: f64,
    pub n_since_negative_power: i32,
    pub iterations: usize,
    pub force_max: f64,
    pub power: f64,
    pub scaling: f64,
}

impl FireState {
    pub fn new(q: &QField3D, params: &FireParams) -> Self {
        Self {
            v: QField3D::zeros(q.nx, q.ny, q.nz, q.dx),
            mix_v: 1.0,
            mix_f: 0.0,
            delta_t: params.delta_t,
            alpha: params.alpha_start,
            n_since_negative_power: 0,
            iterations: 0,
            force_max: f64::INFINITY,
            power: 0.0,
            scaling: 0.0,
        }
    }
}

/// The result of a `fire_minimize_3d_par` run.
pub struct FireResult {
    pub q: QField3D,
    pub iterations: usize,
    pub force_max: f64,
    pub converged: bool,
}

#[inline]
fn dot5(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4]
}

/// open-Qmin's own residual quantity, `sqrt(sum_i |f_i|^2) / N`, applied to
/// any force-shaped field (the RHS/molecular field for Euler, or the FIRE
/// force). Exposed so every integrator in a comparison is scored against the
/// identical stopping quantity described in `BENCHMARKS.md`.
pub fn force_max_metric(f: &QField3D) -> f64 {
    let n = f.len();
    let sum_sq: f64 = f
        .q
        .par_iter()
        .map(|fk| dot5(fk, fk))
        .sum();
    sum_sq.sqrt() / (n as f64)
}

/// One FIRE iteration: velocity-Verlet against `beris_edwards_rhs_3d_par_dry`,
/// then the FIRE velocity mix and adaptive-timestep rule.
///
/// `f` holds the force at `q` on entry and is updated in place to the force
/// at the new `q` on return. Returns the updated `force_max` (open-Qmin's
/// `sqrt(sum|f|^2)/N` quantity, evaluated at the new `q`).
pub fn fire_step_3d_par(
    q: &mut QField3D,
    f: &mut QField3D,
    state: &mut FireState,
    params: &FireParams,
    p: &ActiveNematicParams3D,
    t: f64,
) -> f64 {
    let n = q.len();
    let dt = state.delta_t;

    // velocity-Verlet step (mass = 1), against the old force.
    // The position update and the first half-kick read the same `v` and `f`
    // and write `q` and `v`, so they run as one pass in place. Collecting a
    // fresh `q` here instead, as an earlier version did, allocated 40 MB per
    // iteration at N=100 and faulted the pages in again each time.
    let half_dt = 0.5 * dt;
    let half_dt2 = half_dt * dt;
    let (mix_v, mix_f) = (state.mix_v, state.mix_f);
    q.q.par_iter_mut()
        .zip(state.v.q.par_iter_mut())
        .zip(f.q.par_iter())
        .for_each(|((qk, vk), fk)| {
            for c in 0..5 {
                // The previous iteration's velocity mix, deferred to here so it
                // shares this pass's reads of v and f rather than taking a pass
                // of its own. Both coefficients are zero on a reset, which is
                // what zeroing the velocity did.
                let vm = mix_v * vk[c] + mix_f * fk[c];
                qk[c] += dt * vm + half_dt2 * fk[c];
                vk[c] = vm + half_dt * fk[c];
            }
        });

    // recompute the force at the new configuration, into the buffer already
    // held rather than into a fresh one.
    beris_edwards_rhs_3d_par_dry_into(q, p, t, f);

    // second half-kick and the FIRE reductions, in one pass.
    // The reduction reads exactly the `f` and `v` the half-kick just touched,
    // so running them separately reads both fields twice.
    let (force_norm, velocity_norm, power) = state
        .v
        .q
        .par_iter_mut()
        .zip(f.q.par_iter())
        .map(|(vk, fk)| {
            let mut fs = 0.0_f64;
            let mut vs = 0.0_f64;
            let mut ps = 0.0_f64;
            for c in 0..5 {
                vk[c] += half_dt * fk[c];
                fs += fk[c] * fk[c];
                vs += vk[c] * vk[c];
                ps += fk[c] * vk[c];
            }
            (fs, vs, ps)
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0.0_f64),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

    let force_max = force_norm.sqrt() / (n as f64);
    let scaling = if force_norm > 0.0 {
        (velocity_norm / force_norm).sqrt()
    } else {
        0.0
    };

    // FIRE velocity mix: v = (1 - alpha) v + alpha * scaling * f.
    // Recorded rather than applied: the next iteration's position pass folds it
    // in. Nothing between here and there reads `v`, and a run that stops here
    // returns `q`, which the mix does not touch.
    state.mix_v = 1.0 - state.alpha;
    state.mix_f = state.alpha * scaling;

    // adaptive deltaT / alpha, open-Qmin's Power-sign rule.
    state.iterations += 1;
    if power > 0.0 && state.iterations % 500 != 0 {
        if state.n_since_negative_power > params.n_min {
            state.delta_t = (state.delta_t * params.delta_t_inc).min(params.delta_t_max);
            state.alpha = (state.alpha * params.alpha_dec).max(params.alpha_min);
        }
        state.n_since_negative_power += 1;
    } else {
        state.n_since_negative_power = 0;
        state.delta_t = (state.delta_t * params.delta_t_dec).max(params.delta_t_min);
        state.alpha = params.alpha_start;
        // A reset zeroes `v` after the mix; zero coefficients express that in
        // the deferred mix.
        state.mix_v = 0.0;
        state.mix_f = 0.0;
    }

    state.force_max = force_max;
    state.power = power;
    state.scaling = scaling;
    force_max
}

/// Minimise `q0` under the passive dry Beris-Edwards molecular field via
/// FIRE, stopping when `force_max <= params.force_cutoff` or
/// `params.max_iterations` is reached (whichever comes first). Always runs
/// at least one iteration, matching open-Qmin's `minimize()`.
pub fn fire_minimize_3d_par(
    q0: &QField3D,
    p: &ActiveNematicParams3D,
    params: &FireParams,
    t: f64,
) -> FireResult {
    let mut q = q0.clone();
    let mut f = beris_edwards_rhs_3d_par_dry(&q, p, t);
    let mut state = FireState::new(&q, params);

    loop {
        let force_max = fire_step_3d_par(&mut q, &mut f, &mut state, params, p, t);
        if state.iterations >= params.max_iterations || force_max <= params.force_cutoff {
            break;
        }
    }

    FireResult {
        q,
        iterations: state.iterations,
        force_max: state.force_max,
        converged: state.force_max <= params.force_cutoff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fire_reduces_uniform_bulk_force_below_cutoff() {
        // Uniform director along z, S offset from equilibrium: purely bulk
        // dynamics (Laplacian is exactly zero on a uniform field), so this
        // exercises the FIRE machinery without depending on the elastic term.
        let mut p = ActiveNematicParams3D::default_test();
        p.nx = 4;
        p.ny = 4;
        p.nz = 4;
        p.zeta_eff = 0.0;
        p.noise_amp = 0.0;

        let s = 0.1_f64; // away from equilibrium S0
        let q0 = QField3D::uniform(p.nx, p.ny, p.nz, p.dx, [-s / 3.0, 0.0, 0.0, -s / 3.0, 0.0]);

        let params = FireParams::open_qmin_defaults(0.005, 1e-6, 2000);
        let result = fire_minimize_3d_par(&q0, &p, &params, 0.0);
        assert!(
            result.converged,
            "FIRE did not converge: force_max={}",
            result.force_max
        );
    }
}
