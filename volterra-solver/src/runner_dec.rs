//! Dry active nematic runner on 2D DEC meshes.
//!
//! Solves the Beris-Edwards equation (no flow coupling) on a triangulated
//! 2-manifold using RK4 time integration.
//!
//! ## Governing equation
//!
//! ```text
//! dQ/dt = gamma_r * H
//! ```
//!
//! where H is the molecular field from [`volterra_dec::molecular_field_dec`].
//! Flow coupling (Stokes solver) is deferred to Sub-project B.

use cartan_core::Manifold;
use cartan_dec::Operators;
use volterra_core::ActiveNematicParams;
use volterra_dec::{molecular_field_dec, QFieldDec};

/// Per-snapshot statistics from a DEC simulation.
#[derive(Debug, Clone)]
pub struct SnapStatsDec {
    /// Simulation time.
    pub time: f64,
    /// Mean scalar order parameter.
    pub mean_s: f64,
    /// Number of vertices in the mesh.
    pub n_vertices: usize,
}

/// Beris-Edwards RHS for the dry active nematic (no flow).
///
/// dQ/dt = gamma_r * H
fn beris_edwards_rhs_dec<M: Manifold>(
    q: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> QFieldDec {
    let h = molecular_field_dec(q, params, ops, curvature_correction);
    h.scale(params.gamma_r)
}

/// One RK4 step for the Q-tensor field on a DEC mesh.
fn rk4_step<M: Manifold>(
    q: &QFieldDec,
    dt: f64,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> QFieldDec {
    let k1 = beris_edwards_rhs_dec(q, params, ops, curvature_correction);

    let q2 = q.add(&k1.scale(0.5 * dt));
    let k2 = beris_edwards_rhs_dec(&q2, params, ops, curvature_correction);

    let q3 = q.add(&k2.scale(0.5 * dt));
    let k3 = beris_edwards_rhs_dec(&q3, params, ops, curvature_correction);

    let q4 = q.add(&k3.scale(dt));
    let k4 = beris_edwards_rhs_dec(&q4, params, ops, curvature_correction);

    // q_next = q + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
    let rhs = k1
        .add(&k2.scale(2.0))
        .add(&k3.scale(2.0))
        .add(&k4);
    q.add(&rhs.scale(dt / 6.0))
}

/// Run the dry active nematic on a 2D DEC mesh.
///
/// Evolves Q via RK4 for `n_steps` time steps with no hydrodynamic flow.
/// Records statistics every `snap_every` steps.
///
/// # Arguments
///
/// * `q_init` - Initial Q-tensor field on the mesh.
/// * `params` - Physical and numerical parameters (uses `k_r`, `gamma_r`,
///   `a_landau`, `zeta_eff`, `c_landau`, `dt`).
/// * `ops` - Precomputed DEC operators (Laplace-Beltrami, Hodge stars).
/// * `curvature_correction` - Optional Weitzenboeck endomorphism for curved
///   surfaces. Pass `None` for flat meshes.
/// * `n_steps` - Total number of time steps.
/// * `snap_every` - Steps between snapshot statistics.
///
/// # Returns
///
/// `(q_final, stats)`: final Q-field and one [`SnapStatsDec`] per snapshot.
pub fn run_dry_active_nematic_dec<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> (QFieldDec, Vec<SnapStatsDec>) {
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
            q = rk4_step(&q, params.dt, params, ops, curvature_correction);
        }
    }

    (q, stats)
}
