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
use cartan_manifolds::sphere::Sphere;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::{Observer, RunConfig, SimulationRunner};
use volterra_core::ActiveNematicParams;
use volterra_dec::connection_laplacian::{molecular_field_conn, ConnectionLaplacian};
use volterra_dec::mesh_gen::icosphere;
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

/// Smoke helper: run a short rotor-backed connection Laplacian simulation on S^2.
///
/// Builds a `ConnectionLaplacian` from an icosphere at the given refinement level,
/// initialises a random Q-tensor field, and advances it for `steps` RK4 steps
/// using [`molecular_field_conn`]. Returns per-vertex `sqrt(q1^2 + q2^2)` after
/// the run.
///
/// This function exists to make the rotor-backed connection path testable at the
/// solver level without importing cartan types into the test binary.
pub fn run_dry_active_nematic_dec_smoke(refinement: usize, steps: usize) -> Vec<f64> {
    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();

    let manifold = Sphere::<3>;
    let ops = Operators::from_mesh_generic(&mesh, &manifold)
        .expect("Operators::from_mesh_generic failed on icosphere");
    let star0: Vec<f64> = (0..ops.hodge.star0().len()).map(|i| ops.hodge.star0()[i]).collect();
    let star1: Vec<f64> = (0..ops.hodge.star1().len()).map(|i| ops.hodge.star1()[i]).collect();
    let cl = ConnectionLaplacian::new(&mesh, &coords, &star0, &star1);

    let mut params = ActiveNematicParams::default_test();
    // Small dt for CFL stability on the sphere (h ~ 1/sqrt(nv), so dt ~ h^2).
    params.dt = 1e-3;

    let mut q = QFieldDec::random_perturbation(nv, 0.1, 42);

    for _ in 0..steps {
        // RK4 using molecular_field_conn as the RHS.
        let k1 = molecular_field_conn(&q, params.k_r, params.a_eff(), params.c_landau, &cl)
            .scale(params.gamma_r);
        let q2 = q.add(&k1.scale(0.5 * params.dt));
        let k2 = molecular_field_conn(&q2, params.k_r, params.a_eff(), params.c_landau, &cl)
            .scale(params.gamma_r);
        let q3 = q.add(&k2.scale(0.5 * params.dt));
        let k3 = molecular_field_conn(&q3, params.k_r, params.a_eff(), params.c_landau, &cl)
            .scale(params.gamma_r);
        let q4 = q.add(&k3.scale(params.dt));
        let k4 = molecular_field_conn(&q4, params.k_r, params.a_eff(), params.c_landau, &cl)
            .scale(params.gamma_r);
        let rhs = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        q = q.add(&rhs.scale(params.dt / 6.0));
    }

    (0..nv)
        .map(|i| (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]).sqrt())
        .collect()
}
