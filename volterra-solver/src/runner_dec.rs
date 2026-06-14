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
//! Flow coupling (Stokes solver) is in [`crate::run_wet_active_nematic_dec`].
//!
//! The per-step physics lives in [`crate::sim_impls::dec::DecDry`]; this module
//! is the thin wrapper driving it through `volterra_core::sim::SimulationRunner`.

use cartan_core::Manifold;
use cartan_dec::Operators;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::{Observer, RunConfig, SimulationRunner};
use volterra_core::ActiveNematicParams;
use volterra_dec::QFieldDec;

use crate::sim_impls::dec::DecDry;

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

/// Build a partial `SnapStatsDec` from the runner-core `StepStats` currency.
///
/// Carries the shared `time` and `mean_s` (the DEC `order_param`); `n_vertices`
/// is not a per-step diagnostic and is filled in by the snapshot sink from the
/// observed field.
impl From<StepStats> for SnapStatsDec {
    fn from(s: StepStats) -> Self {
        SnapStatsDec {
            time: s.time.unwrap_or(0.0),
            mean_s: s.order_param.unwrap_or(0.0),
            n_vertices: 0,
        }
    }
}

/// Snapshot sink that builds `SnapStatsDec` from the observed field and stats.
///
/// `mean_s` is read from the field (`q.mean_order_param()`) so the step-0
/// snapshot (whose `StepStats` predate any physics step) reports the initial
/// field's order parameter, exactly as the legacy loop did.
pub(crate) struct DecSink {
    /// Collected snapshots, in order.
    pub out: Vec<SnapStatsDec>,
}

impl Observer<QFieldDec> for DecSink {
    fn observe(&mut self, _step: usize, t: f64, q: &QFieldDec, _stats: &StepStats) {
        self.out.push(SnapStatsDec {
            time: t,
            mean_s: q.mean_order_param(),
            n_vertices: q.n_vertices,
        });
    }
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
    let mut physics = DecDry {
        params: params.clone(),
        ops,
        cc: curvature_correction,
    };
    let mut q = q_init.clone();
    let mut sink = DecSink { out: Vec::new() };
    let runner = SimulationRunner {
        config: RunConfig {
            steps: n_steps,
            snap_every,
            dt: params.dt,
            seed: 0,
        },
    };
    runner.run(&mut q, &mut physics, &mut sink);
    (q, sink.out)
}
