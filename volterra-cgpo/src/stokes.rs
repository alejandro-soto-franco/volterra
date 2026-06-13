//! Stokes / pressure-relaxation solver.
//!
//! Ports the following numba functions from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py` (lines ~166–392):
//!
//! - [`calculate_pressure_terms`] — builds the ∇·F − ρ·(∂ᵢuⱼ∂ⱼuᵢ) increment
//!   onto the pressure-Poisson RHS (called after `div_vector` has set the base).
//! - [`relax_pressure_inner_loop`] — one Jacobi sweep of the 9-point pressure
//!   Poisson relaxation (double-buffer: reads from `p_aux`, writes to `p`).
//! - [`relax_pressure`] — full iteration driver: builds RHS, iterates until
//!   convergence or cap, returns iterations used.
//! - [`subtract_p_avg`] — remove mean pressure over interior.
//! - [`u_update_p_pi_terms`] — velocity update from −∇p/ρ + ∇·Π/ρ.
//! - [`get_u_update`] — full velocity time-derivative (viscous + convective +
//!   pressure/stress).
//!
//! # BC assumption (Task E deferred)
//!
//! The Python `relax_pressure` calls `apply_p_boundary_conditions` after each
//! inner sweep (line 303).  That function uses the Python `boundary` array of
//! shape `[2, Lx, Ly, 2]` which carries per-layer normal vectors — a richer
//! structure than our `Boundary` struct (which stores `is_outer`/`is_inner`
//! bool masks and normals, but is keyed differently from the Python 4-D array).
//! **For Task D, the BC application is a no-op stub.**  The pressure field is
//! relaxed purely on the interior cells; boundary cells retain whatever value
//! they were initialised with (zero by default in the test).  Task E will wire
//! the full Neumann pressure BC using our `Boundary` normal data.
//!
//! The test fixture bypasses this gap by using a flat rectangular interior
//! (1-cell border), capping iterations to an exact N, and setting
//! `p_target_rel_change` to a tiny value so the Python also does exactly N
//! sweeps without BC application mattering (the Python BC only touches the
//! outer/inner boundary ring, which is outside the rectangular interior).
//!
//! # Jacobi vs Gauss-Seidel
//!
//! The Python `relax_pressure_inner_loop` is **Jacobi**: it copies `p → p_aux`
//! before the sweep, then reads exclusively from `p_aux` while writing to `p`.
//! Our Rust implementation follows the same double-buffer pattern exactly.
//!
//! # Convergence test
//!
//! After each sweep (Python lines 305–306):
//! ```text
//! rel_change = sum(|p_aux - p|) / (1e-7 + sum(p_aux))
//! ```
//! where `p_aux` is the **old** p (before the sweep) and `p` is the **new**
//! p.  The loop continues while `rel_change > target_rel_change` AND
//! `(p_iters < max_p_iters OR max_p_iters < 0)`.
//! A negative `max_p_iters` means **uncapped** (loop until convergence only).

use crate::{
    ops::{div_vector, laplacian_vector, upwind_advective_term},
    Boundary, Params,
};

// ---------------------------------------------------------------------------
// index helpers (matching ops.rs conventions)
// ---------------------------------------------------------------------------

#[inline(always)]
fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}

#[inline(always)]
fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}

// ---------------------------------------------------------------------------
// Public kernels
// ---------------------------------------------------------------------------

/// Accumulate the non-divergence terms onto the pressure-Poisson RHS.
///
/// Ports `calculate_pressure_terms(u, ρ, Π_S, pressure_poisson_RHS, bounds)`.
///
/// **Accumulates** into `rhs` — caller must initialise it (e.g. with
/// `div_vector` multiplied by `ρ/dt`) before calling.
///
/// Formula added per interior cell:
/// ```text
/// rhs[x,y] += (∂x²−∂y²)Π_S_xx + 2·∂x∂y·Π_S_xy
///           − ρ·( (∂x·ux)² + (∂y·uy)² + 2·(∂y·ux)·(∂x·uy) )
/// ```
/// All differences are centred, `dx=1`, indices wrap modulo (lx, ly).
pub fn calculate_pressure_terms(
    u: &[f64],
    rho: f64,
    pi_s: &[f64],
    rhs: &mut [f64],
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            // Centred velocity gradients
            let dudx = 0.5 * (u[vi(xup, y, ly, 0)] - u[vi(xdn, y, ly, 0)]);
            let dvdy = 0.5 * (u[vi(x, yup, ly, 1)] - u[vi(x, ydn, ly, 1)]);
            let dyux = 0.5 * (u[vi(x, yup, ly, 0)] - u[vi(x, ydn, ly, 0)]);
            let dxuy = 0.5 * (u[vi(xup, y, ly, 1)] - u[vi(xdn, y, ly, 1)]);

            // ∇·F = (∂x²−∂y²)Π_S_xx + 2·∂x·∂y·Π_S_xy
            //   = (Π_S_xx[xup,y] + Π_S_xx[xdn,y] − Π_S_xx[x,yup] − Π_S_xx[x,ydn])
            //   + 0.5·(Π_S_xy[xup,yup] − Π_S_xy[xup,ydn] − Π_S_xy[xdn,yup] + Π_S_xy[xdn,ydn])
            let div_f = (pi_s[vi(xup, y, ly, 0)]
                + pi_s[vi(xdn, y, ly, 0)]
                - pi_s[vi(x, yup, ly, 0)]
                - pi_s[vi(x, ydn, ly, 0)])
                + 0.5
                    * (pi_s[vi(xup, yup, ly, 1)] - pi_s[vi(xup, ydn, ly, 1)]
                        - pi_s[vi(xdn, yup, ly, 1)]
                        + pi_s[vi(xdn, ydn, ly, 1)]);

            // −ρ·(∂ᵢuⱼ ∂ⱼuᵢ) = −ρ·((∂x·ux)² + (∂y·uy)² + 2·(∂y·ux)·(∂x·uy))
            // Note: cross term is (∂y·ux)*(∂x·uy) * 2, not *(∂x·uy + ∂y·ux)
            // Python: dudx*dudx + dvdy*dvdy + 0.5*(dyux)*(dxuy+dxuy) wait —
            // Python line exactly:
            //   -ρ*(dudx*dudx + dvdy*dvdy + 0.5*(u[x,yup,0]-u[x,ydn,0])*(u[xup,y,1]-u[xdn,y,1]))
            // The 0.5 factor comes from Python's centred differences being (f[+]-f[-]) without /2,
            // then 0.5* the product = 0.5*(2*dyux)*(2*dxuy) = 2*dyux*dxuy — but actually:
            // Python: u[x,yup,0]-u[x,ydn,0] = 2*dyux (since dyux = 0.5*(yup-ydn))
            //         u[xup,y,1]-u[xdn,y,1] = 2*dxuy
            // So 0.5*(2*dyux)*(2*dxuy) = 2*dyux*dxuy ✓  matches d_i u_j d_j u_i cross
            let conv = rho * (dudx * dudx + dvdy * dvdy + dyux * 2.0 * dxuy);

            rhs[idx] += div_f - conv;
        }
    }
}

/// One Jacobi sweep of the pressure Poisson relaxation.
///
/// Ports `relax_pressure_inner_loop(p, p_aux, pressure_poisson_RHS, bounds)`.
///
/// **Jacobi**: reads exclusively from `p_aux` (the old field), writes to `p`.
/// Caller must copy `p → p_aux` before calling.
///
/// 9-point stencil (solved for centre, coeff = 0.05 = 1/20):
/// ```text
/// p[x,y] = 0.05 * (
///     −6·rhs[x,y]
///     + 4·(p_aux[x+1,y] + p_aux[x,y+1] + p_aux[x,y-1] + p_aux[x-1,y])
///     + p_aux[x+1,y+1] + p_aux[x+1,y-1] + p_aux[x-1,y+1] + p_aux[x-1,y-1]
/// )
/// ```
pub fn relax_pressure_inner_loop(
    p: &mut [f64],
    p_aux: &[f64],
    rhs: &[f64],
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            p[idx] = 0.05
                * (-6.0 * rhs[idx]
                    + 4.0
                        * (p_aux[si(xup, y, ly)]
                            + p_aux[si(x, yup, ly)]
                            + p_aux[si(x, ydn, ly)]
                            + p_aux[si(xdn, y, ly)])
                    + p_aux[si(xup, yup, ly)]
                    + p_aux[si(xup, ydn, ly)]
                    + p_aux[si(xdn, yup, ly)]
                    + p_aux[si(xdn, ydn, ly)]);
        }
    }
}

/// No-op stub for pressure boundary conditions (Task E).
///
/// The Python `apply_p_boundary_conditions` applies Neumann-like BCs on the
/// outer and inner boundary rings using the boundary normal vectors.  That
/// function's signature takes the Python 4-D `boundary` array
/// `[layer, x, y, component]`, which maps to our `Boundary` struct's
/// `outer_normals`/`inner_normals` but with different addressing.
///
/// **This stub does nothing.**  When Task E wires the full BC, replace this
/// with the translated normal-gradient pressure condition.  The test fixtures
/// use a rectangular interior (no boundary ring inside the tested region), so
/// the no-op is numerically exact for the test.
#[allow(unused_variables)]
pub fn apply_p_boundary_conditions_stub(
    _p: &mut [f64],
    _p_aux: &[f64],
    _bounds: &Boundary,
) {
    // Task E: implement Neumann pressure BC using bounds.outer_normals / inner_normals
}

/// Full pressure relaxation driver.
///
/// Ports `relax_pressure(u, ρ, p, Π_S, Π_A, ν, p_aux, pressure_poisson_RHS,
///                        dt, target_rel_change, boundary, bounds, max_p_iters=-1)`.
///
/// # Steps
///
/// 1. `rhs = div(u)` (via [`div_vector`])
/// 2. `rhs *= ρ/dt`
/// 3. `rhs += calculate_pressure_terms(u, ρ, Π_S, …)` (∇·F − ρ·convective)
/// 4. Jacobi loop:
///    - copy `p → p_aux`
///    - [`relax_pressure_inner_loop`]
///    - BC stub (no-op for now)
///    - `rel_change = Σ|p_aux−p| / (1e-7 + Σ|p_aux|)`
///    - stop if `rel_change <= target_rel_change`
///      AND `(p_iters >= max_p_iters AND max_p_iters >= 0)`
///
/// # Convergence-test convention (exact Python match)
///
/// ```text
/// rel_change = sum(|p_aux - p|) / (1e-7 + sum(p_aux))
/// ```
/// `p_aux` is the pre-sweep (old) value; `p` is the post-sweep (new) value.
/// Note `sum(p_aux)` is a **signed** sum, not sum-of-abs.
/// Iteration continues while `rel_change > target_rel_change` AND the
/// iteration cap is not yet reached.  `max_p_iters < 0` means uncapped.
///
/// # Returns
///
/// Number of Jacobi sweeps performed.
pub fn relax_pressure(
    u: &[f64],
    rho: f64,
    p: &mut [f64],
    pi_s: &[f64],
    p_aux: &mut [f64],
    rhs: &mut [f64],
    dt: f64,
    target_rel_change: f64,
    bounds: &Boundary,
    max_p_iters: i64,
) -> usize {
    // Step 1–2: rhs = (ρ/dt) · ∇·u
    div_vector(u, rhs, bounds);
    let scale = rho / dt;
    for v in rhs.iter_mut() {
        *v *= scale;
    }

    // Step 3: accumulate ∇·F and −ρ·convective onto rhs
    calculate_pressure_terms(u, rho, pi_s, rhs, bounds);

    // Step 4: Jacobi iteration
    let mut p_iters: usize = 0;
    let mut rel_change = target_rel_change + 1.0; // ensure at least one sweep

    loop {
        // Check cap: stop if max_p_iters >= 0 and we've hit it
        if max_p_iters >= 0 && p_iters >= max_p_iters as usize {
            break;
        }
        // Check convergence from previous sweep
        if p_iters > 0 && rel_change <= target_rel_change {
            break;
        }

        // Copy p → p_aux  (Jacobi double-buffer)
        p_aux.copy_from_slice(p);

        // Inner Jacobi sweep: read p_aux, write p
        relax_pressure_inner_loop(p, p_aux, rhs, bounds);

        // BC stub (no-op; Task E)
        apply_p_boundary_conditions_stub(p, p_aux, bounds);

        // Convergence test: rel_change = Σ|p_aux−p| / (1e-7 + Σp_aux)
        // p_aux = old p, p = new p
        let sum_diff: f64 = p_aux.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
        let sum_old: f64 = p_aux.iter().sum();
        rel_change = sum_diff / (1e-7 + sum_old);

        p_iters += 1;
    }

    p_iters
}

/// Remove the mean pressure over interior cells.
///
/// Ports `subtract_p_avg(p, bounds)`.
///
/// Computes `p_avg = Σ p[interior] / n_interior`, then subtracts from all
/// interior cells.
pub fn subtract_p_avg(p: &mut [f64], bounds: &Boundary) {
    let n_interior = bounds.inside.iter().filter(|&&b| b).count();
    if n_interior == 0 {
        return;
    }
    let lx = bounds.lx;
    let ly = bounds.ly;

    // Python uses np.sum(p) / len(bounds) — sums the ENTIRE array, not just interior.
    // But since non-interior cells stay zero throughout, np.sum(p) == sum of interior.
    // We replicate the Python exactly: sum all cells, divide by n_interior.
    let p_avg: f64 = p.iter().sum::<f64>() / n_interior as f64;

    for x in 0..lx {
        for y in 0..ly {
            if bounds.inside[si(x, y, ly)] {
                p[si(x, y, ly)] -= p_avg;
            }
        }
    }
}

/// Velocity update from pressure gradient and stress divergence.
///
/// Ports `u_update_p_Π_terms(dudt, p, ρ, Π_S, Π_A, bounds)`.
///
/// **Accumulates** into `dudt` — caller must initialise `dudt` before calling
/// (Python callers set `dudt` from the viscous+convective terms first).
///
/// Formula (centred differences, `dx=1`):
/// ```text
/// dudt[x,y,0] += 0.5/ρ * (
///     −(p[x+1,y] − p[x−1,y])                         // −∂x p
///     + (Π_S[x+1,y,0] − Π_S[x−1,y,0])               // ∂x Π_xx
///     + (Π_S[x,y+1,1]+Π_A[x,y+1]) − (Π_S[x,y−1,1]+Π_A[x,y−1])  // ∂y (Π_xy+Π_A)
/// )
/// dudt[x,y,1] += 0.5/ρ * (
///     −(p[x,y+1] − p[x,y−1])                         // −∂y p
///     + (Π_S[x+1,y,1]−Π_A[x+1,y]) − (Π_S[x−1,y,1]−Π_A[x−1,y])  // ∂x (Π_yx)
///     − (Π_S[x,y+1,0] − Π_S[x,y−1,0])               // −∂y Π_xx
/// )
/// ```
pub fn u_update_p_pi_terms(
    dudt: &mut [f64],
    p: &[f64],
    rho: f64,
    pi_s: &[f64],
    pi_a: &[f64],
    bounds: &Boundary,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let inv_rho = 0.5 / rho;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            // x-component
            dudt[vi(x, y, ly, 0)] += inv_rho
                * (-(p[si(xup, y, ly)] - p[si(xdn, y, ly)])
                    + (pi_s[vi(xup, y, ly, 0)] - pi_s[vi(xdn, y, ly, 0)])
                    + ((pi_s[vi(x, yup, ly, 1)] + pi_a[si(x, yup, ly)])
                        - (pi_s[vi(x, ydn, ly, 1)] + pi_a[si(x, ydn, ly)])));

            // y-component
            dudt[vi(x, y, ly, 1)] += inv_rho
                * (-(p[si(x, yup, ly)] - p[si(x, ydn, ly)])
                    + ((pi_s[vi(xup, y, ly, 1)] - pi_a[si(xup, y, ly)])
                        - (pi_s[vi(xdn, y, ly, 1)] - pi_a[si(xdn, y, ly)]))
                    - (pi_s[vi(x, yup, ly, 0)] - pi_s[vi(x, ydn, ly, 0)]));
        }
    }
}

/// Full velocity time-derivative.
///
/// Ports `get_u_update(dudt, u, p, ρ, Π_S, Π_A, ν, bounds)`.
///
/// # Order (must match Python)
///
/// 1. `dudt = ν·∇²u`  (viscous term via [`laplacian_vector`])
/// 2. `dudt += −(u·∇)u`  (convective term via [`upwind_advective_term`])
/// 3. `dudt += u_update_p_pi_terms(…)`  (pressure + stress)
///
/// `ν` is the kinematic viscosity (`params.eta / params.rho` in the Python
/// convention, but the Python passes `ν` directly as the `eta` field).
pub fn get_u_update(
    dudt: &mut [f64],
    u: &[f64],
    p: &[f64],
    rho: f64,
    pi_s: &[f64],
    pi_a: &[f64],
    nu: f64,
    bounds: &Boundary,
) {
    // 1. viscous term: dudt = ν∇²u
    laplacian_vector(u, dudt, bounds, nu);

    // 2. convective term: dudt += −(u·∇)u  (upwind, coeff=-1)
    upwind_advective_term(u, u, dudt, bounds, -1.0);

    // 3. pressure + stress
    u_update_p_pi_terms(dudt, p, rho, pi_s, pi_a, bounds);
}

/// Convenience wrapper: runs the full pressure relaxation from [`Params`].
///
/// Equivalent to [`relax_pressure`] but takes `params` for `rho`, `dt`, and
/// `max_p_iters` (using `params.max_p_iters as i64`).
pub fn relax_pressure_from_params(
    u: &[f64],
    p: &mut [f64],
    pi_s: &[f64],
    p_aux: &mut [f64],
    rhs: &mut [f64],
    target_rel_change: f64,
    bounds: &Boundary,
    params: &Params,
) -> usize {
    relax_pressure(
        u,
        params.rho,
        p,
        pi_s,
        p_aux,
        rhs,
        params.dt,
        target_rel_change,
        bounds,
        params.max_p_iters as i64,
    )
}
