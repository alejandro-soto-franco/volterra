//! Wet active nematic runner on 2D DEC meshes.
//!
//! Solves the coupled Beris-Edwards + Stokes system on a triangulated
//! 2-manifold using operator splitting:
//!
//! 1. **Stokes**: solve for velocity from active stress sigma = -zeta Q.
//! 2. **Beris-Edwards (RK4)**: advance Q with molecular field + advection.
//!
//! The per-step physics lives in [`crate::sim_impls::dec::DecWet`]; these are
//! the thin wrappers driving it through `volterra_core::sim::SimulationRunner`.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use volterra_core::sim::{RunConfig, SimulationRunner};
use volterra_core::ActiveNematicParams;
use volterra_dec::stokes_dec::StokesSolverDec;
use volterra_dec::QFieldDec;

use crate::runner_dec::{DecSink, SnapStatsDec};
use crate::sim_impls::dec::DecWet;

/// Shared coordinate extraction helper (avoids duplicating the Debug-parse hack).
fn extract_coords_runner<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Vec<[f64; 3]> {
    mesh.vertices.iter().map(|v| {
        let s = format!("{:?}", v);
        let nums: Vec<f64> = s
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',' || *c == ' ' || *c == 'e' || *c == '+')
            .collect::<String>()
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        match nums.len() {
            2 => [nums[0], nums[1], 0.0],
            n if n >= 3 => [nums[0], nums[1], nums[2]],
            _ => panic!(
                "extract_coords_runner: failed to parse vertex coordinates from Debug repr {s:?} \
                 (parsed {} numbers, expected 2 or 3); a silent zero would corrupt advection",
                nums.len()
            ),
        }
    }).collect()
}

/// Drive a [`DecWet`] physics through the shared `SimulationRunner` loop.
///
/// The Stokes solver is constructed by the caller (closed or confined) and
/// moved into the physics; the loop itself never fails.
#[allow(clippy::too_many_arguments)] // physics driver: many fields and parameters
fn run_wet_inner<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    stokes: StokesSolverDec,
    coords: Vec<[f64; 3]>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> (QFieldDec, Vec<SnapStatsDec>) {
    let mut physics = DecWet {
        params: params.clone(),
        stokes,
        ops,
        mesh,
        coords,
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

/// Run the wet active nematic on a 2D DEC mesh.
///
/// # Returns
///
/// `Ok((q_final, stats))` on success, or an error if the Poisson
/// factorisation for the Stokes solver fails.
#[allow(clippy::too_many_arguments)]
pub fn run_wet_active_nematic_dec<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> Result<(QFieldDec, Vec<SnapStatsDec>), String> {
    let stokes = StokesSolverDec::new(ops, mesh)?;
    let coords = extract_coords_runner(mesh);
    Ok(run_wet_inner(
        q_init,
        params,
        ops,
        mesh,
        stokes,
        coords,
        curvature_correction,
        n_steps,
        snap_every,
    ))
}

/// Run the wet active nematic on a **confined (bounded) 2D DEC mesh** with no-slip BCs.
///
/// Identical to [`run_wet_active_nematic_dec`] except the Stokes solver enforces
/// ψ = 0 on all `boundary_vertices` (Dirichlet stream-function), giving true
/// no-slip at the domain boundary.
///
/// # Parameters
///
/// - `boundary_vertices`: indices of domain-boundary vertices (e.g. from
///   [`volterra_dec::epitrochoid::ConfinedMesh::boundary_vertices`]).
///
/// # Returns
///
/// `Ok((q_final, stats))` on success, or an error if the Poisson factorisation fails.
#[allow(clippy::too_many_arguments)]
pub fn run_wet_active_nematic_dec_confined<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    boundary_vertices: &[usize],
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> Result<(QFieldDec, Vec<SnapStatsDec>), String> {
    let stokes = StokesSolverDec::new_confined(ops, mesh, boundary_vertices)?;
    let coords = extract_coords_runner(mesh);
    Ok(run_wet_inner(
        q_init,
        params,
        ops,
        mesh,
        stokes,
        coords,
        curvature_correction,
        n_steps,
        snap_every,
    ))
}
