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
use volterra_fields::{QField3D, ScalarField3D};

use crate::defects_3d::{scan_defects_3d, track_defect_events};
use cartan_geo::disclination::DisclinationLine;

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
    /// Number of connected disclination lines detected.
    pub n_disclination_lines: usize,
    /// Total disclination line length (in vertex units).
    pub total_line_length: f64,
    /// Mean Frenet curvature along all disclination lines.
    pub mean_line_curvature: f64,
    /// Number of topological events (creation / annihilation / reconnection)
    /// detected since the previous snapshot.
    pub n_events: usize,
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
    /// Number of connected disclination lines detected.
    pub n_disclination_lines: usize,
    /// Total disclination line length (in vertex units).
    pub total_line_length: f64,
    /// Mean Frenet curvature along all disclination lines.
    pub mean_line_curvature: f64,
    /// Number of topological events since the previous snapshot.
    pub n_events: usize,
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
    let mut prev_lines: Option<Vec<DisclinationLine>> = None;

    let mut physics = Cartesian3DDry { params: p.clone(), step_idx: 0 };

    for step in 0..n_steps {
        // Advance physics (fused Euler step + Langevin noise).
        // Pass t=0.0; Cartesian3DDry computes t from its internal step_idx before
        // incrementing, so the correct t = step * dt is used each call.
        physics.step(&mut q, 0.0);

        // Snapshot trigger: when (step+1) % snap_every == 0.
        if snap_every > 0 && (step + 1) % snap_every == 0 {
            let snap_idx = (step + 1) / snap_every;
            let t_snap = (step + 1) as f64 * p.dt;

            let (lines, n_events) = if track_defects {
                let current = scan_defects_3d(&q);
                let n_ev = if let Some(ref prev) = prev_lines {
                    track_defect_events(prev, &current, snap_idx, p.dx).len()
                } else {
                    0
                };
                prev_lines = Some(current.clone());
                (current, n_ev)
            } else {
                (Vec::new(), 0)
            };

            let s = compute_snap_stats(&q, &lines, n_events, t_snap);
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
    use volterra_fields::VelocityField3D;

    let mut st = BechState3D {
        q: q_init.clone(),
        phi: phi_init.clone(),
        vel: VelocityField3D::zeros(p.nx, p.ny, p.nz, p.dx),
    };
    let mut stats: Vec<BechStats3D> = Vec::new();
    let mut prev_lines: Option<Vec<DisclinationLine>> = None;

    let mut physics = Cartesian3DBech { params: p.clone(), step_idx: 0 };

    for step in 0..n_steps {
        // Advance physics: Stokes + Euler BE + noise + CH-ETD.
        // Pass t=0.0; Cartesian3DBech computes t from its internal step_idx.
        physics.step(&mut st, 0.0);

        // Snapshot trigger.
        if snap_every > 0 && (step + 1) % snap_every == 0 {
            let snap_idx = (step + 1) / snap_every;
            let t_snap = (step + 1) as f64 * p.dt;

            let (lines, n_events) = if track_defects {
                let current = scan_defects_3d(&st.q);
                let n_ev = if let Some(ref prev) = prev_lines {
                    track_defect_events(prev, &current, snap_idx, p.dx).len()
                } else {
                    0
                };
                prev_lines = Some(current.clone());
                (current, n_ev)
            } else {
                (Vec::new(), 0)
            };

            let s = compute_bech_stats(&st.q, &st.phi, &lines, n_events, t_snap);
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

/// Compute [`SnapStats3D`] from the current Q-field and disclination lines.
fn compute_snap_stats(
    q: &QField3D,
    lines: &[DisclinationLine],
    n_events: usize,
    time: f64,
) -> SnapStats3D {
    let mean_s = q.mean_s();
    let biaxiality_p = q.biaxiality_p().iter().sum::<f64>() / q.len() as f64;
    let total_length: f64 = lines.iter().map(|l| l.vertices.len() as f64).sum();
    let mean_curv = if total_length > 0.0 {
        lines.iter().flat_map(|l| l.curvatures.iter()).sum::<f64>() / total_length
    } else {
        0.0
    };
    SnapStats3D {
        time,
        mean_s,
        biaxiality_p,
        n_disclination_lines: lines.len(),
        total_line_length: total_length,
        mean_line_curvature: mean_curv,
        n_events,
    }
}

/// Compute [`BechStats3D`] from the current Q, φ, and disclination lines.
fn compute_bech_stats(
    q: &QField3D,
    phi: &ScalarField3D,
    lines: &[DisclinationLine],
    n_events: usize,
    time: f64,
) -> BechStats3D {
    let mean_s = q.mean_s();
    let biaxiality_p = q.biaxiality_p().iter().sum::<f64>() / q.len() as f64;
    let mean_phi = phi.mean();
    let total_length: f64 = lines.iter().map(|l| l.vertices.len() as f64).sum();
    let mean_curv = if total_length > 0.0 {
        lines.iter().flat_map(|l| l.curvatures.iter()).sum::<f64>() / total_length
    } else {
        0.0
    };
    BechStats3D {
        time,
        mean_s,
        biaxiality_p,
        mean_phi,
        n_disclination_lines: lines.len(),
        total_line_length: total_length,
        mean_line_curvature: mean_curv,
        n_events,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::ActiveNematicParams3D;
    use volterra_fields::{QField3D, ScalarField3D};

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
