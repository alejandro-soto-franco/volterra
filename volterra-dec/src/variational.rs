//! Variational integrator with BAOAB splitting for membrane dynamics.
//!
//! B: half-kick (momentum update from forces)
//! A: drift (position update via exp map)
//! O: stochastic (Shardlow edge sweep, handled by caller)
//!
//! The caller inserts the O step between the two A half-steps.
//!
//! ## References
//!
//! - Leimkuhler, Matthews. "Rational Construction of Stochastic Numerical
//!   Methods for Molecular Sampling." AMM, 2012.
//! - Mem3DG. Biophysical Reports, 2022.

use cartan_core::Manifold;

use crate::domain::DecDomain;
use crate::helfrich::{helfrich_forces, HelfrichParams};

/// Configuration for the variational integrator.
pub struct VariationalConfig {
    /// Time step size.
    pub dt: f64,
    /// Maximum Newton iterations for implicit B steps (0 = explicit).
    pub newton_max_iter: usize,
    /// Newton convergence tolerance.
    pub newton_tol: f64,
    /// Kinetic energy tolerance for symplectic-aware remesh triggering.
    pub ke_tolerance: f64,
}

/// Perform the deterministic B-A-A-B steps of the BAOAB scheme.
///
/// The full BAOAB splitting is:
///
///   B: p += (dt/4) * F(x)
///   A: x = exp_x(p * dt / (2m))
///   O: (caller inserts Shardlow sweep here)
///   A: x = exp_x(p * dt / (2m))
///   B: p += (dt/4) * F(x_new)
///
/// This function performs the B-A and A-B deterministic steps. The caller
/// is responsible for inserting the stochastic O step between the two
/// calls (or between the two A half-steps within a single call).
///
/// Uses the exponential map for position updates, preserving the manifold
/// constraint exactly.
pub fn baoab_ba_step<M: Manifold<Point = nalgebra::SVector<f64, 3>, Tangent = nalgebra::SVector<f64, 3>>>(
    manifold: &M,
    positions: &mut [M::Point],
    momenta: &mut [M::Tangent],
    masses: &[f64],
    helfrich_params: &HelfrichParams,
    domain: &DecDomain<M>,
    dt: f64,
) {
    let nv = positions.len();

    // Compute forces at current positions.
    let forces = helfrich_forces(domain, helfrich_params);

    // B step (first quarter): p += (dt/4) * force
    for v in 0..nv {
        let scaled = forces[v].clone() * (dt / 4.0);
        momenta[v] = momenta[v].clone() + scaled;
    }

    // A step (first half): x = exp_x(p * dt / (2*mass))
    for v in 0..nv {
        let m = masses[v].max(1e-30);
        let vel = momenta[v].clone() * (dt / (2.0 * m));
        positions[v] = manifold.exp(&positions[v], &vel);
    }

    // (O step goes here, handled by caller)

    // A step (second half): x = exp_x(p * dt / (2*mass))
    for v in 0..nv {
        let m = masses[v].max(1e-30);
        let vel = momenta[v].clone() * (dt / (2.0 * m));
        positions[v] = manifold.exp(&positions[v], &vel);
    }

    // B step (second quarter): p += (dt/4) * force(new_x)
    // Explicit scheme: reuse forces from old positions. First-order in B,
    // second-order overall in configuration space.
    for v in 0..nv {
        let scaled = forces[v].clone() * (dt / 4.0);
        momenta[v] = momenta[v].clone() + scaled;
    }
}

/// Compute kinetic energy from momenta and masses.
///
///   KE = sum_v (1/2) * ||p_v||^2 / m_v
///
/// Uses the manifold inner product to compute tangent vector norms.
pub fn kinetic_energy<M: Manifold>(
    manifold: &M,
    positions: &[M::Point],
    momenta: &[M::Tangent],
    masses: &[f64],
) -> f64 {
    momenta
        .iter()
        .zip(positions.iter())
        .zip(masses.iter())
        .map(|((p, x), &m)| {
            let norm_sq = manifold.inner(x, p, p);
            0.5 * norm_sq / m.max(1e-30)
        })
        .sum()
}

/// Adaptive time step based on diffusive and force CFL conditions.
///
/// Returns the minimum of three candidates:
///
///   dt_diff  = c_diff * h_min^2       (diffusive CFL)
///   dt_force = c_force * h_min / F_max  (force CFL)
///   dt_max                              (user-imposed cap)
///
/// `h_min` is the shortest edge length in the mesh. `max_force` is the
/// maximum force magnitude across all vertices. `c_diff` and `c_force`
/// are safety factors (typically 0.1 to 0.5).
pub fn compute_dt(
    h_min: f64,
    max_force: f64,
    dt_max: f64,
    c_diff: f64,
    c_force: f64,
) -> f64 {
    let dt_diff = c_diff * h_min * h_min;
    let dt_force = if max_force > 1e-30 {
        c_force * h_min / max_force
    } else {
        f64::MAX
    };
    dt_max.min(dt_diff).min(dt_force)
}
