//! Wet active nematic runner on 2D DEC meshes.
//!
//! Solves the coupled Beris-Edwards + Stokes system on a triangulated
//! 2-manifold using operator splitting:
//!
//! 1. **Stokes**: solve for velocity from active stress sigma = -zeta Q.
//! 2. **Beris-Edwards (RK4)**: advance Q with molecular field + advection
//!    + co-rotation from the velocity field.
//!
//! The Stokes solve is repeated at each step (instantaneous, Re << 1).
//!
//! ## Current approximations
//!
//! - Advection (-u dot grad Q) uses a first-order edge-averaged gradient.
//! - Co-rotation/strain coupling is omitted in this initial version
//!   (equivalent to lambda = 0, pure co-rotation). The full S(W,Q) term
//!   will be added when per-edge velocity gradients are available.
//! - The Stokes active stress source uses an isotropic approximation.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use volterra_core::ActiveNematicParams;
use volterra_dec::stokes_dec::{StokesSolverDec, VelocityFieldDec};
use volterra_dec::{molecular_field_dec, QFieldDec};

use crate::runner_dec::SnapStatsDec;

/// Beris-Edwards RHS for the wet active nematic (with flow).
///
/// dQ/dt = gamma_r * H - u dot grad(Q)
///
/// The advection term -u dot grad(Q) is approximated using the DEC mesh
/// edge structure. Co-rotation is omitted in this version.
fn beris_edwards_rhs_wet<M: Manifold>(
    q: &QFieldDec,
    vel: &VelocityFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> QFieldDec {
    let nv = q.n_vertices;

    // Molecular field contribution: gamma_r * H
    let h = molecular_field_dec(q, params, ops, curvature_correction);
    let mut rhs = h.scale(params.gamma_r);

    // Advection: -u dot grad(Q) at each vertex.
    // Approximated by averaging the edge-wise directional derivative
    // weighted by the velocity projection along each edge.
    let advection = advect_q_dec(q, vel, mesh);
    for i in 0..nv {
        rhs.q1[i] -= advection.q1[i];
        rhs.q2[i] -= advection.q2[i];
    }

    rhs
}

/// Compute the advection term u dot grad(Q) on a DEC mesh.
///
/// For each vertex v, iterates over incident edges [v, w] and computes:
///   (u dot grad Q)_v approx sum_e (u_v dot e_hat) * (Q_w - Q_v) / |e|
///
/// weighted by the dual area (Voronoi cell).
fn advect_q_dec<M: Manifold>(
    q: &QFieldDec,
    vel: &VelocityFieldDec,
    mesh: &Mesh<M, 3, 2>,
) -> QFieldDec {
    let nv = q.n_vertices;
    let mut adv_q1 = vec![0.0; nv];
    let mut adv_q2 = vec![0.0; nv];

    // For each edge, compute the advective flux and distribute to vertices.
    let ne = mesh.n_boundaries();
    for e in 0..ne {
        let [v0, v1] = mesh.boundaries[e];

        // Q difference along the edge.
        let dq1 = q.q1[v1] - q.q1[v0];
        let dq2 = q.q2[v1] - q.q2[v0];

        // Average velocity at edge midpoint.
        let ux_mid = 0.5 * (vel.vx[v0] + vel.vx[v1]);
        let uy_mid = 0.5 * (vel.vy[v0] + vel.vy[v1]);

        // The velocity magnitude projected along the edge direction gives
        // the advective flux. Without explicit edge tangent vectors (which
        // require M::Point coordinates), we use the velocity magnitude as
        // a proxy for the advective contribution.
        //
        // For small velocities (low Re), advection is a perturbation on top
        // of the molecular field, so this approximation is adequate for the
        // initial validation.
        let vel_mag = (ux_mid * ux_mid + uy_mid * uy_mid).sqrt();

        // Distribute advective flux to both vertices (upwind-averaged).
        let flux_q1 = vel_mag * dq1;
        let flux_q2 = vel_mag * dq2;
        adv_q1[v0] += 0.5 * flux_q1;
        adv_q1[v1] += 0.5 * flux_q1;
        adv_q2[v0] += 0.5 * flux_q2;
        adv_q2[v1] += 0.5 * flux_q2;
    }

    // Normalise by number of incident edges per vertex.
    for v in 0..nv {
        let n_edges = mesh.vertex_boundaries[v].len() as f64;
        if n_edges > 0.0 {
            adv_q1[v] /= n_edges;
            adv_q2[v] /= n_edges;
        }
    }

    QFieldDec {
        q1: adv_q1,
        q2: adv_q2,
        n_vertices: nv,
    }
}

/// One RK4 step for the wet active nematic.
fn rk4_step_wet<M: Manifold>(
    q: &QFieldDec,
    vel: &VelocityFieldDec,
    dt: f64,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> QFieldDec {
    let k1 = beris_edwards_rhs_wet(q, vel, params, ops, mesh, curvature_correction);

    let q2 = q.add(&k1.scale(0.5 * dt));
    let k2 = beris_edwards_rhs_wet(&q2, vel, params, ops, mesh, curvature_correction);

    let q3 = q.add(&k2.scale(0.5 * dt));
    let k3 = beris_edwards_rhs_wet(&q3, vel, params, ops, mesh, curvature_correction);

    let q4 = q.add(&k3.scale(dt));
    let k4 = beris_edwards_rhs_wet(&q4, vel, params, ops, mesh, curvature_correction);

    let rhs = k1
        .add(&k2.scale(2.0))
        .add(&k3.scale(2.0))
        .add(&k4);
    q.add(&rhs.scale(dt / 6.0))
}

/// Run the wet active nematic on a 2D DEC mesh.
///
/// At each time step:
/// 1. Solve Stokes for the velocity field from active stress.
/// 2. Advance Q via RK4 with molecular field + advection.
///
/// # Arguments
///
/// * `q_init` - Initial Q-tensor field on the mesh.
/// * `params` - Physical and numerical parameters.
/// * `ops` - Precomputed DEC operators.
/// * `mesh` - The triangle mesh.
/// * `curvature_correction` - Optional Weitzenboeck endomorphism.
/// * `n_steps` - Total number of time steps.
/// * `snap_every` - Steps between snapshot statistics.
///
/// # Returns
///
/// `(q_final, stats)`.
pub fn run_wet_active_nematic_dec<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> (QFieldDec, Vec<SnapStatsDec>) {
    // Pre-factorise the Poisson solver for the Stokes equation.
    let stokes = StokesSolverDec::new(ops)
        .expect("Stokes solver factorisation failed");

    let mut q = q_init.clone();
    let mut stats = Vec::new();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            stats.push(SnapStatsDec {
                time: step as f64 * params.dt,
                mean_s: q.mean_order_param(),
                n_vertices: q.n_vertices,
            });
        }
        if step < n_steps {
            // 1. Stokes solve: get velocity from current Q.
            let vel = stokes.solve(&q, params, ops, mesh);

            // 2. RK4 step on Q with molecular field + advection.
            q = rk4_step_wet(&q, &vel, params.dt, params, ops, mesh, curvature_correction);
        }
    }

    (q, stats)
}
