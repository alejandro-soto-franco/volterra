#![allow(clippy::needless_range_loop)]
// ~/volterra/volterra-solver/src/runner_3d.rs

//! High-level 3D simulation runners for dry active nematic and BECH
//! (Beris-Edwards + Cahn-Hilliard) models.
//!
//! Two entry points are provided:
//!
//! | Function | Model | Fields evolved |
//! |----------|-------|----------------|
//! | [`run_dry_active_nematic_3d`] | dry active nematic | Q only |
//! | [`run_bech_3d`] | full BECH | Q + φ + Stokes velocity |
//!
//! Both runners:
//! - Accept an initial field and a [`volterra_core::ActiveNematicParams3D`].
//! - Advance by `n_steps` Euler steps, writing `.npy` snapshots every
//!   `snap_every` steps to `out_dir`.
//! - Return the final field(s) together with a vector of per-snapshot
//!   statistics structs ([`SnapStats3D`] / [`BechStats3D`]).

use std::path::Path;

use serde::{Deserialize, Serialize};

use volterra_core::ActiveNematicParams3D;
use volterra_core::{QField3D, ScalarField3D, VelocityField3D};

use volterra_braid::disclination::{DisclinationCurve, disclination_lines_at_fraction};

// ─────────────────────────────────────────────────────────────────────────────
// Statistics types
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics for the dry active nematic run ([`run_dry_active_nematic_3d`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapStats3D {
    /// Simulation time at this snapshot.
    pub time: f64,
    /// Spatial mean of the scalar order parameter S = (3/2) λ_max.
    pub mean_s: f64,
    /// Spatial mean of the biaxiality parameter P = λ_mid − λ_min.
    pub biaxiality_p: f64,
    /// The disclination density the lines were read off at this snapshot.
    ///
    /// Set as a fraction of the field's own interior peak, so it moves with the
    /// field. Read the counts below against it: a snapshot whose threshold has
    /// fallen by orders of magnitude has no disclination left, whatever it
    /// reports finding.
    pub disclination_threshold: f64,
    /// Number of connected disclination lines detected.
    pub n_disclination_lines: usize,
    /// How many of those close on themselves.
    pub n_disclination_loops: usize,
    /// Total disclination line length, in the grid's own length units.
    pub total_line_length: f64,
    /// Length-weighted mean curvature of the lines themselves.
    pub mean_line_curvature: f64,
    /// Length-weighted mean curvature of the `s` isosurface around them.
    ///
    /// Positive, since `s` rises towards a core and the surface normal follows
    /// it, and near `1/(2R)` at the tube radius `R`. That makes it much the
    /// larger of the two curvatures.
    pub mean_surface_mean_curvature: f64,
    /// Length-weighted mean Gaussian curvature of that surface.
    pub mean_surface_gaussian_curvature: f64,
    /// Length-weighted mean of `cos(beta)`, which is `+1` on a `+1/2` wedge,
    /// `-1` on a `-1/2` wedge and `0` on a twist.
    pub mean_cos_beta: f64,
    /// Fastest flow anywhere in the box.
    ///
    /// Zero in a dry run, which solves no flow, so the field distinguishes the
    /// two without a second statistics type.
    pub max_speed: f64,
    /// Mean speed over the box.
    pub mean_speed: f64,
}

/// Per-snapshot statistics for the full BECH run ([`run_bech_3d`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BechStats3D {
    /// Simulation time at this snapshot.
    pub time: f64,
    /// Spatial mean of the scalar order parameter S.
    pub mean_s: f64,
    /// Spatial mean of the biaxiality parameter P.
    pub biaxiality_p: f64,
    /// Spatial mean of the lipid concentration φ.
    pub mean_phi: f64,
    /// The disclination density the lines were read off at this snapshot.
    pub disclination_threshold: f64,
    /// Number of connected disclination lines detected.
    pub n_disclination_lines: usize,
    /// How many of those close on themselves.
    pub n_disclination_loops: usize,
    /// Total disclination line length, in the grid's own length units.
    pub total_line_length: f64,
    /// Length-weighted mean curvature of the lines themselves.
    pub mean_line_curvature: f64,
    /// Length-weighted mean curvature of the `s` isosurface around them.
    pub mean_surface_mean_curvature: f64,
    /// Length-weighted mean Gaussian curvature of that surface.
    pub mean_surface_gaussian_curvature: f64,
    /// Length-weighted mean of `cos(beta)`, the wedge-against-twist character.
    pub mean_cos_beta: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public runners
// ─────────────────────────────────────────────────────────────────────────────

/// Run the **dry** 3D active nematic model (no Stokes coupling).
///
/// Advances `Q` by `n_steps` Euler + Langevin steps. Writes `.npy` snapshots
/// of the Q-tensor field and a `stats.json` file to `out_dir`.
///
/// # Snapshot trigger
///
/// A snapshot is written when `(step + 1) % snap_every == 0`. With
/// `n_steps = snap_every` this yields exactly one snapshot.
///
/// # Arguments
///
/// * `q_init`       - Initial Q-tensor field.
/// * `p`            - Active nematic parameters (grid, physics, noise).
/// * `n_steps`      - Number of time steps to advance.
/// * `snap_every`   - Write a snapshot every this many steps.
/// * `out_dir`      - Directory for `.npy` and `stats.json` output.
/// * `track_defects`- When `true`, run the full disclination detection
///   pipeline and topology-event tracker at each snapshot.
///
/// # Returns
///
/// `(q_final, stats)`: final Q-field and one [`SnapStats3D`] per snapshot.
pub fn run_dry_active_nematic_3d(
    q_init: &QField3D,
    p: &ActiveNematicParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &Path,
    track_defects: bool,
) -> (QField3D, Vec<SnapStats3D>) {
    use crate::sim_impls::cartesian3d::Cartesian3DDry;
    use volterra_core::sim::PhysicsStep;
    use volterra_core::sim::snapshot::write_npy;

    let mut q = q_init.clone();
    let mut stats: Vec<SnapStats3D> = Vec::new();

    let mut physics = Cartesian3DDry { params: p.clone(), step_idx: 0 };

    for step in 0..n_steps {
        // Advance physics (fused Euler step + Langevin noise).
        // Pass t=0.0; Cartesian3DDry computes t from its internal step_idx before
        // incrementing, so the correct t = step * dt is used each call.
        physics.step(&mut q, 0.0);

        // Snapshot trigger: when (step+1) % snap_every == 0.
        if snap_every > 0 && (step + 1) % snap_every == 0 {

            let t_snap = (step + 1) as f64 * p.dt;

            let (lines, threshold) = if track_defects {
                disclination_lines_at_fraction(
                    &q.q,
                    q.nx,
                    q.ny,
                    q.nz,
                    q.dx,
                    p.disclination_threshold_fraction,
                    p.disclination_floor(),
                )
            } else {
                (Vec::new(), 0.0)
            };

            let s = compute_snap_stats(&q, &lines, threshold, t_snap, None);
            stats.push(s);

            let npy_path = out_dir.join(format!("q_{step:06}.npy"));
            let flat: Vec<f64> = q.q.iter().flat_map(|arr| arr.iter().copied()).collect();
            if let Err(e) = write_npy(&npy_path, &flat, p.nx, p.ny, p.nz, 5) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", npy_path.display());
            }
        }
    }

    let stats_path = out_dir.join("stats.json");
    if let Ok(json) = serde_json::to_string_pretty(&stats) {
        let _ = std::fs::write(&stats_path, json);
    }

    (q, stats)
}

/// Run the **wet** 3D active nematic: Beris-Edwards coupled to Stokes flow.
///
/// Each step solves the steady incompressible Stokes problem driven by the
/// active stress `-zeta Q`, then advances `Q` in that flow, so a disclination
/// line is advected and sheared by hydrodynamics its own texture generates. The
/// dry runner solves no flow at all, and with the activity switched off this one
/// reproduces it exactly.
///
/// Writes `.npy` snapshots of `Q` and of the velocity, and a `stats.json` whose
/// records now report the flow as well as the lines.
///
/// # Returns
///
/// `(q_final, velocity_final, stats)`.
pub fn run_wet_active_nematic_3d(
    q_init: &QField3D,
    p: &ActiveNematicParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &Path,
    track_defects: bool,
) -> (QField3D, VelocityField3D, Vec<SnapStats3D>) {
    use crate::sim_impls::cartesian3d::{Cartesian3DWet, WetState3D};
    use volterra_core::sim::PhysicsStep;
    use volterra_core::sim::snapshot::write_npy;

    let mut st = WetState3D {
        q: q_init.clone(),
        vel: VelocityField3D::zeros(q_init.nx, q_init.ny, q_init.nz, q_init.dx),
    };
    let mut stats: Vec<SnapStats3D> = Vec::new();
    let mut physics = Cartesian3DWet { params: p.clone(), step_idx: 0 };

    for step in 0..n_steps {
        physics.step(&mut st, 0.0);

        if snap_every > 0 && (step + 1) % snap_every == 0 {
            let t_snap = (step + 1) as f64 * p.dt;
            let (lines, threshold) = if track_defects {
                disclination_lines_at_fraction(
                    &st.q.q,
                    st.q.nx,
                    st.q.ny,
                    st.q.nz,
                    st.q.dx,
                    p.disclination_threshold_fraction,
                    p.disclination_floor(),
                )
            } else {
                (Vec::new(), 0.0)
            };
            stats.push(compute_snap_stats(&st.q, &lines, threshold, t_snap, Some(&st.vel)));

            let flat: Vec<f64> = st.q.q.iter().flat_map(|a| a.iter().copied()).collect();
            if let Err(e) = write_npy(&out_dir.join(format!("q_{step:06}.npy")), &flat, p.nx, p.ny, p.nz, 5) {
                eprintln!("[runner_3d] warn: failed to write the Q snapshot: {e}");
            }
            let flow: Vec<f64> = st.vel.u.iter().flat_map(|a| a.iter().copied()).collect();
            if let Err(e) = write_npy(&out_dir.join(format!("u_{step:06}.npy")), &flow, p.nx, p.ny, p.nz, 3) {
                eprintln!("[runner_3d] warn: failed to write the velocity snapshot: {e}");
            }
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&stats) {
        let _ = std::fs::write(out_dir.join("stats.json"), json);
    }
    (st.q, st.vel, stats)
}

/// Run the **full BECH** 3D model: Beris-Edwards + Stokes + Cahn-Hilliard.
///
/// At each step the coupled system is advanced by operator-splitting:
/// 1. Stokes solve (active stress from current Q).
/// 2. Beris-Edwards Euler step (advection + co-rotation + molecular field).
/// 3. Langevin noise on Q.
/// 4. Cahn-Hilliard ETD step for φ.
///
/// ## Approximation note
///
/// The Cahn-Hilliard equation requires a *lipid* Q-tensor `q_lip`. In this
/// sprint the rotor Q-field `q` is passed as `q_lip` to `ch_step_etd_3d`.
/// This is a leading-order approximation: the rotational orientational order
/// drives the Maier-Saupe coupling term as if `q_lip ≈ q`. A separate
/// lipid Q evolution will be added in a future sprint.
///
/// # Snapshot trigger
///
/// A snapshot is written when `(step + 1) % snap_every == 0`.
///
/// # Returns
///
/// `(q_final, phi_final, stats)`.
pub fn run_bech_3d(
    q_init: &QField3D,
    phi_init: &ScalarField3D,
    p: &ActiveNematicParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &Path,
    track_defects: bool,
) -> (QField3D, ScalarField3D, Vec<BechStats3D>) {
    use crate::sim_impls::cartesian3d::{BechState3D, Cartesian3DBech};
    use volterra_core::sim::PhysicsStep;
    use volterra_core::sim::snapshot::write_npy;
    use volterra_core::VelocityField3D;

    let mut st = BechState3D {
        q: q_init.clone(),
        phi: phi_init.clone(),
        vel: VelocityField3D::zeros(p.nx, p.ny, p.nz, p.dx),
    };
    let mut stats: Vec<BechStats3D> = Vec::new();

    let mut physics = Cartesian3DBech { params: p.clone(), step_idx: 0 };

    for step in 0..n_steps {
        // Advance physics: Stokes + Euler BE + noise + CH-ETD.
        // Pass t=0.0; Cartesian3DBech computes t from its internal step_idx.
        physics.step(&mut st, 0.0);

        // Snapshot trigger.
        if snap_every > 0 && (step + 1) % snap_every == 0 {

            let t_snap = (step + 1) as f64 * p.dt;

            let (lines, threshold) = if track_defects {
                disclination_lines_at_fraction(
                    &st.q.q,
                    st.q.nx,
                    st.q.ny,
                    st.q.nz,
                    st.q.dx,
                    p.disclination_threshold_fraction,
                    p.disclination_floor(),
                )
            } else {
                (Vec::new(), 0.0)
            };

            let s = compute_bech_stats(&st.q, &st.phi, &lines, threshold, t_snap);
            stats.push(s);

            // Write Q snapshot.
            let q_path = out_dir.join(format!("q_{step:06}.npy"));
            let flat_q: Vec<f64> = st.q.q.iter().flat_map(|arr| arr.iter().copied()).collect();
            if let Err(e) = write_npy(&q_path, &flat_q, p.nx, p.ny, p.nz, 5) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", q_path.display());
            }

            // Write phi snapshot as (nx,ny,nz,1).
            let phi_path = out_dir.join(format!("phi_{step:06}.npy"));
            if let Err(e) = write_npy(&phi_path, &st.phi.phi, p.nx, p.ny, p.nz, 1) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", phi_path.display());
            }

            // Write velocity snapshot as (nx,ny,nz,3).
            let vel_path = out_dir.join(format!("vel_{step:06}.npy"));
            let flat_vel: Vec<f64> = st.vel.u.iter().flat_map(|arr| arr.iter().copied()).collect();
            if let Err(e) = write_npy(&vel_path, &flat_vel, p.nx, p.ny, p.nz, 3) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", vel_path.display());
            }
        }
    }

    let stats_path = out_dir.join("stats.json");
    if let Ok(json) = serde_json::to_string_pretty(&stats) {
        let _ = std::fs::write(&stats_path, json);
    }

    (st.q, st.phi, stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// The aggregates both runners report over a set of disclination lines.
///
/// Every mean is weighted by contour length, so a line that runs the depth of
/// the box counts for more than a two-voxel fragment beside it.
struct LineGeometry {
    n_lines: usize,
    n_loops: usize,
    total_length: f64,
    curvature: f64,
    surface_mean: f64,
    surface_gaussian: f64,
    cos_beta: f64,
}

impl LineGeometry {
    fn of(lines: &[DisclinationCurve]) -> Self {
        let total_length: f64 = lines.iter().map(|c| c.length).sum();
        let weighted = |f: fn(&DisclinationCurve) -> f64| {
            if total_length > 0.0 {
                lines.iter().map(|c| c.length * f(c)).sum::<f64>() / total_length
            } else {
                0.0
            }
        };
        Self {
            n_lines: lines.len(),
            n_loops: lines.iter().filter(|c| c.is_loop).count(),
            total_length,
            curvature: weighted(|c| c.mean_curvature),
            surface_mean: weighted(|c| c.surface_mean_curvature),
            surface_gaussian: weighted(|c| c.surface_gaussian_curvature),
            cos_beta: weighted(|c| c.mean_cos_beta),
        }
    }
}

/// Compute [`SnapStats3D`] from the current Q-field and disclination lines.
fn compute_snap_stats(
    q: &QField3D,
    lines: &[DisclinationCurve],
    threshold: f64,
    time: f64,
    vel: Option<&VelocityField3D>,
) -> SnapStats3D {
    let g = LineGeometry::of(lines);
    let (max_speed, mean_speed) = speeds(vel);
    SnapStats3D {
        time,
        mean_s: q.mean_s(),
        biaxiality_p: q.biaxiality_p().iter().sum::<f64>() / q.len() as f64,
        disclination_threshold: threshold,
        n_disclination_lines: g.n_lines,
        n_disclination_loops: g.n_loops,
        total_line_length: g.total_length,
        mean_line_curvature: g.curvature,
        mean_surface_mean_curvature: g.surface_mean,
        mean_surface_gaussian_curvature: g.surface_gaussian,
        mean_cos_beta: g.cos_beta,
        max_speed,
        mean_speed,
    }
}

/// The fastest and mean speed of a flow, or zeros where there is none.
fn speeds(vel: Option<&VelocityField3D>) -> (f64, f64) {
    let Some(v) = vel else { return (0.0, 0.0) };
    if v.u.is_empty() {
        return (0.0, 0.0);
    }
    let mut max = 0.0_f64;
    let mut sum = 0.0_f64;
    for u in &v.u {
        let s = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        max = max.max(s);
        sum += s;
    }
    (max, sum / v.u.len() as f64)
}

/// Compute [`BechStats3D`] from the current Q, φ, and disclination lines.
fn compute_bech_stats(
    q: &QField3D,
    phi: &ScalarField3D,
    lines: &[DisclinationCurve],
    threshold: f64,
    time: f64,
) -> BechStats3D {
    let g = LineGeometry::of(lines);
    BechStats3D {
        time,
        mean_s: q.mean_s(),
        biaxiality_p: q.biaxiality_p().iter().sum::<f64>() / q.len() as f64,
        mean_phi: phi.mean(),
        disclination_threshold: threshold,
        n_disclination_lines: g.n_lines,
        n_disclination_loops: g.n_loops,
        total_line_length: g.total_length,
        mean_line_curvature: g.curvature,
        mean_surface_mean_curvature: g.surface_mean,
        mean_surface_gaussian_curvature: g.surface_gaussian,
        mean_cos_beta: g.cos_beta,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::ActiveNematicParams3D;
    use volterra_core::{QField3D, ScalarField3D, VelocityField3D};

    /// Smoke test: 5 steps of dry active turbulence on a tiny grid, no crash.
    #[test]
    fn test_run_dry_active_nematic_3d_dry_smoke() {
        let p = ActiveNematicParams3D::default_test(); // 16^3
        let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
        let tmp = std::env::temp_dir().join("volterra_test_run");
        std::fs::create_dir_all(&tmp).unwrap();
        let (q_final, stats) = run_dry_active_nematic_3d(&q_init, &p, 5, 5, &tmp, false);
        assert_eq!(q_final.len(), q_init.len());
        assert_eq!(stats.len(), 1);
        assert!(stats[0].mean_s >= 0.0);
    }

    /// Smoke test for the full BECH runner.
    #[test]
    fn test_run_bech_3d_smoke() {
        let p = ActiveNematicParams3D::default_test();
        let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
        let phi_init = ScalarField3D::uniform(p.nx, p.ny, p.nz, p.dx, 0.3);
        let tmp = std::env::temp_dir().join("volterra_test_full");
        std::fs::create_dir_all(&tmp).unwrap();
        let (q_f, phi_f, stats) = run_bech_3d(&q_init, &phi_init, &p, 5, 5, &tmp, false);
        assert_eq!(q_f.len(), q_init.len());
        assert!((phi_f.mean() - 0.3).abs() < 0.01, "mass roughly conserved, got mean={}", phi_f.mean());
        assert_eq!(stats.len(), 1);
    }
}
