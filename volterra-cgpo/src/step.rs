//! Full timestep driver for the CGPO solver (Task E).
//!
//! Ports `update_step_inner` and `update_step` from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py` lines 738–767.
//!
//! # Step ordering (matching Python `update_step_inner`)
//!
//! ```text
//! 1. H_S_from_Q(u, Q, H, S, ...)
//! 2. apply_H_boundary_conditions(H, γ, Q, u, S, boundary)
//! 3. calculate_Π(Π_S, Π_A, H, Q, ...)
//! 4. relax_pressure(u, ρ, p, Π_S, Π_A, ν, ...)
//! 5. get_Q_update(dQ, Q, H, S, u, γ, bounds)
//! 6. get_u_update(dudt, u, p, ρ, Π_S, Π_A, ν, bounds)
//! 7. Q += dt * dQ
//! 8. u += dt * dudt
//! 9. apply_Q_boundary_conditions(Q, boundary, S0)
//! 10. apply_u_boundary_conditions(u, boundary)
//! ```
//!
//! # Q-update formula
//!
//! `get_Q_update` ports Python lines 114–125:
//! ```python
//! dQ[:] = (1/γ) * H[:] + S[:]
//! upwind_advective_term(u, Q, dQ, bounds)  # default coeff=-1
//! ```
//! The upwind call **accumulates** `-(u·∇)Q` into `dQ` (coeff=-1 in Python
//! default, which matches our Rust `upwind_advective_term` with coeff=-1).

use crate::{
    bc::{
        apply_h_boundary_conditions, apply_q_boundary_conditions,
        apply_u_boundary_conditions, apply_p_boundary_conditions,
    },
    nematic::{calculate_pi, h_s_from_q},
    ops::upwind_advective_term,
    par_gate::use_parallel,
    stokes::get_u_update,
    Boundary, Params,
};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Q update
// ---------------------------------------------------------------------------

/// Compute the Q time-derivative.
///
/// Ports `get_Q_update(dQ, Q, H, S, u, γ, bounds)` (lines 113–125).
///
/// ```text
/// dQ = (1/γ) * H + S
/// dQ += upwind_advective_term(u, Q, coeff=-1)   // subtract (u·∇)Q
/// ```
///
/// On entry `dq` need not be zero — it is **fully overwritten** before
/// `upwind_advective_term` accumulates into it.
pub fn get_q_update(
    dq: &mut [f64],
    q: &[f64],
    h: &[f64],
    s: &[f64],
    u: &[f64],
    gamma: f64,
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let n2 = lx * ly * 2;
    let inv_gamma = 1.0 / gamma;

    // dQ = (1/γ)*H + S
    if use_parallel(lx, ly) {
        dq[..n2].par_iter_mut()
            .zip(h[..n2].par_iter())
            .zip(s[..n2].par_iter())
            .for_each(|((dqi, hi), si)| {
                *dqi = inv_gamma * hi + si;
            });
    } else {
        for i in 0..n2 {
            dq[i] = inv_gamma * h[i] + s[i];
        }
    }

    // subtract (u·∇)Q via upwind scheme (coeff=-1, accumulates)
    upwind_advective_term(u, q, dq, bounds, -1.0);
}

// ---------------------------------------------------------------------------
// State struct
// ---------------------------------------------------------------------------

/// All dynamic and scratch arrays for one simulation instance.
///
/// Allocate once with [`State::new`]; the fields are passed by mutable
/// reference through the step functions.
pub struct State {
    /// Velocity field, 2-component, length `lx*ly*2`.
    pub u: Vec<f64>,
    /// Q-tensor, 2-component, length `lx*ly*2`.
    pub q: Vec<f64>,
    /// Pressure, scalar, length `lx*ly`.
    pub p: Vec<f64>,

    // --- scratch (allocated once, reused each step) ---
    /// Velocity time-derivative (scratch).
    pub dudt: Vec<f64>,
    /// Q time-derivative (scratch).
    pub dq: Vec<f64>,
    /// Molecular field H (scratch).
    pub h: Vec<f64>,
    /// Co-rotation tensor S (scratch).
    pub s: Vec<f64>,
    /// Symmetric stress Π_S (scratch).
    pub pi_s: Vec<f64>,
    /// Antisymmetric stress Π_A, scalar (scratch).
    pub pi_a: Vec<f64>,
    /// Pressure auxiliary array for Jacobi double-buffer (scratch).
    pub p_aux: Vec<f64>,
    /// Pressure Poisson RHS (scratch).
    pub rhs: Vec<f64>,
}

impl State {
    /// Allocate all arrays for an `lx × ly` grid, initialised to zero.
    pub fn new(lx: usize, ly: usize) -> Self {
        let n = lx * ly;
        let n2 = n * 2;
        Self {
            u: vec![0.0; n2],
            q: vec![0.0; n2],
            p: vec![0.0; n],
            dudt: vec![0.0; n2],
            dq: vec![0.0; n2],
            h: vec![0.0; n2],
            s: vec![0.0; n2],
            pi_s: vec![0.0; n2],
            pi_a: vec![0.0; n],
            p_aux: vec![0.0; n],
            rhs: vec![0.0; n],
        }
    }
}

// ---------------------------------------------------------------------------
// Inner Euler step
// ---------------------------------------------------------------------------

/// One Euler substep.
///
/// Ports `update_step_inner(arrs, consts, bounds, boundary)` (lines 738–767).
///
/// Returns the number of pressure Jacobi iterations performed.
pub fn update_step_inner(
    state: &mut State,
    params: &Params,
    bnd: &Boundary,
    target_rel_change: f64,
) -> usize {
    let lx = params.lx;
    let ly = params.ly;
    let n2 = lx * ly * 2;

    // 1. H and S from Q
    h_s_from_q(
        &state.u,
        &state.q,
        &mut state.h,
        &mut state.s,
        params.a_landau,
        params.c_landau,
        params.k_elastic,
        params.lambda,
        bnd,
    );

    // 2. Apply H boundary conditions
    // Need to split borrows: h, q, u, s are all in state.
    // Use raw pointer trick via temporary slices for the immutable reads.
    {
        // Safety: apply_h_bc reads q, u, s and writes h.
        // We need to pass h as mutable and q/u/s as immutable simultaneously.
        // Since they are distinct Vec fields, we can take their slices separately.
        let h_ptr = state.h.as_mut_ptr();
        let h_len = state.h.len();
        // SAFETY: h, q, u, s are distinct fields of State — no aliasing.
        let h_mut = unsafe { std::slice::from_raw_parts_mut(h_ptr, h_len) };
        apply_h_boundary_conditions(h_mut, params.gamma, &state.q, &state.u, &state.s, bnd);
    }

    // 3. Calculate Π
    calculate_pi(
        &mut state.pi_s,
        &mut state.pi_a,
        &state.h,
        &state.q,
        params.lambda,
        params.zeta,
        params.k_elastic,
        bnd,
    );

    // 4. Relax pressure (with real Neumann BCs applied after each sweep)
    let p_iters = relax_pressure_with_bc(
        state,
        params,
        bnd,
        target_rel_change,
        params.max_p_iters as i64,
    );

    // 5. Q update
    get_q_update(
        &mut state.dq,
        &state.q,
        &state.h,
        &state.s,
        &state.u,
        params.gamma,
        bnd,
    );

    // 6. u update
    get_u_update(
        &mut state.dudt,
        &state.u,
        &state.p,
        params.rho,
        &state.pi_s,
        &state.pi_a,
        params.eta, // Python passes ν = eta directly
        bnd,
    );

    // 7. Q += dt * dQ
    let dt = params.dt;
    if use_parallel(lx, ly) {
        state.q[..n2].par_iter_mut()
            .zip(state.dq[..n2].par_iter())
            .for_each(|(qi, dqi)| *qi += dt * dqi);
    } else {
        for i in 0..n2 {
            state.q[i] += dt * state.dq[i];
        }
    }

    // 8. u += dt * dudt
    if use_parallel(lx, ly) {
        state.u[..n2].par_iter_mut()
            .zip(state.dudt[..n2].par_iter())
            .for_each(|(ui, dudti)| *ui += dt * dudti);
    } else {
        for i in 0..n2 {
            state.u[i] += dt * state.dudt[i];
        }
    }

    // 9. Apply Q boundary conditions (Dirichlet anchoring)
    apply_q_boundary_conditions(&mut state.q, bnd, params.s0);

    // 10. Apply u boundary conditions (no-slip)
    apply_u_boundary_conditions(&mut state.u, bnd);

    p_iters
}

// ---------------------------------------------------------------------------
// Pressure relaxation with real Neumann BCs
// ---------------------------------------------------------------------------

/// Pressure relaxation with real Neumann BC applied after each Jacobi sweep.
///
/// Ports Python's `relax_pressure` (lines 280–309) with the BC stub replaced
/// by the real `apply_p_boundary_conditions`.
fn relax_pressure_with_bc(
    state: &mut State,
    params: &Params,
    bnd: &Boundary,
    target_rel_change: f64,
    max_p_iters: i64,
) -> usize {
    use crate::ops::div_vector;
    use crate::stokes::relax_pressure_inner_loop;

    let lx = params.lx;
    let ly = params.ly;

    // Step 1–2: rhs = (ρ/dt) · ∇·u
    div_vector(&state.u, &mut state.rhs, bnd);
    let scale = params.rho / params.dt;
    if use_parallel(lx, ly) {
        state.rhs.par_iter_mut().for_each(|v| *v *= scale);
    } else {
        state.rhs.iter_mut().for_each(|v| *v *= scale);
    }

    // Step 3: accumulate ∇·F and −ρ·convective onto rhs
    use crate::stokes::calculate_pressure_terms;
    calculate_pressure_terms(&state.u, params.rho, &state.pi_s, &mut state.rhs, bnd);

    // Step 4: Jacobi iteration with Neumann BC
    let mut p_iters: usize = 0;
    let mut rel_change = target_rel_change + 1.0;

    loop {
        if max_p_iters >= 0 && p_iters >= max_p_iters as usize {
            break;
        }
        if p_iters > 0 && rel_change <= target_rel_change {
            break;
        }

        // Copy p → p_aux (Jacobi double-buffer)
        state.p_aux.copy_from_slice(&state.p);

        // Inner sweep: read p_aux, write p
        relax_pressure_inner_loop(&mut state.p, &state.p_aux, &state.rhs, bnd);

        // Real Neumann BC on p
        {
            // Need p (mut), p_aux (immut), u/pi_s/pi_a (immut)
            // All are distinct fields — use raw pointer for p to allow simultaneous borrows.
            let p_ptr = state.p.as_mut_ptr();
            let p_len = state.p.len();
            let p_mut = unsafe { std::slice::from_raw_parts_mut(p_ptr, p_len) };
            apply_p_boundary_conditions(
                p_mut,
                &state.p_aux,
                &state.u,
                params.rho,
                params.eta,
                &state.pi_s,
                &state.pi_a,
                bnd,
            );
        }

        // Convergence: rel_change = Σ|p_aux−p| / (1e-7 + Σp_aux)
        // p_aux = old p, p = new p.
        let (sum_diff, sum_old) = if use_parallel(lx, ly) {
            let sd: f64 = state.p_aux.par_iter()
                .zip(state.p.par_iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let so: f64 = state.p_aux.par_iter().sum();
            (sd, so)
        } else {
            let sd: f64 = state.p_aux.iter()
                .zip(state.p.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let so: f64 = state.p_aux.iter().sum();
            (sd, so)
        };
        rel_change = sum_diff / (1e-7 + sum_old);

        p_iters += 1;
    }

    p_iters
}

// ---------------------------------------------------------------------------
// Outer step driver
// ---------------------------------------------------------------------------

/// Run `n_steps` Euler updates on `state`.
///
/// Ports `update_step(arrs, consts, stepcount, t, boundary, bounds, n_steps)`
/// (lines 720–735).
///
/// `target_rel_change` controls the pressure relaxation stopping criterion
/// (Python: `p_target_rel_change`).
///
/// Returns `(steps_done, t_final)`.
pub fn update_step(
    state: &mut State,
    params: &Params,
    boundary: &Boundary,
    n_steps: usize,
    target_rel_change: f64,
) -> (usize, f64) {
    let mut t = 0.0_f64;
    for _ in 0..n_steps {
        update_step_inner(state, params, boundary, target_rel_change);
        t += params.dt;
    }
    (n_steps, t)
}
