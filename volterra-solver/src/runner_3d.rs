// ~/volterra/volterra-solver/src/runner_3d.rs

//! High-level 3D simulation runners for the MARS and BECH (Beris-Edwards +
//! Cahn-Hilliard) models.
//!
//! Two entry points are provided:
//!
//! | Function | Model | Fields evolved |
//! |----------|-------|----------------|
//! | [`run_mars_3d`] | dry active nematic | Q only |
//! | [`run_mars_3d_full`] | full BECH | Q + φ + Stokes velocity |
//!
//! Both runners:
//! - Accept an initial field and a [`volterra_core::MarsParams3D`].
//! - Advance by `n_steps` Euler steps, writing `.npy` snapshots every
//!   `snap_every` steps to `out_dir`.
//! - Return the final field(s) together with a vector of per-snapshot
//!   statistics structs ([`SnapStats3D`] / [`BechStats3D`]).

use std::io::Write;
use std::path::Path;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, ScalarField3D, VelocityField3D};

use crate::beris_3d::{beris_edwards_rhs_3d, EulerIntegrator3D};
use crate::ch_3d::ch_step_etd_3d;
use crate::defects_3d::{scan_defects_3d, track_defect_events};
use crate::stokes_3d::stokes_solve_3d;
use cartan_geo::disclination::DisclinationLine;

// ─────────────────────────────────────────────────────────────────────────────
// Statistics types
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics for the dry active nematic run ([`run_mars_3d`]).
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

/// Per-snapshot statistics for the full BECH run ([`run_mars_3d_full`]).
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
/// * `p`            - MARS parameters (grid, physics, noise).
/// * `n_steps`      - Number of time steps to advance.
/// * `snap_every`   - Write a snapshot every this many steps.
/// * `out_dir`      - Directory for `.npy` and `stats.json` output.
/// * `track_defects`- When `true`, run the full disclination detection
///                    pipeline and topology-event tracker at each snapshot.
///
/// # Returns
///
/// `(q_final, stats)` — final Q-field and one [`SnapStats3D`] per snapshot.
pub fn run_mars_3d(
    q_init: &QField3D,
    p: &MarsParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &Path,
    track_defects: bool,
) -> (QField3D, Vec<SnapStats3D>) {
    let euler = EulerIntegrator3D;
    let mut q = q_init.clone();
    let mut stats: Vec<SnapStats3D> = Vec::new();

    // Previous lines for event tracking (populated lazily).
    let mut prev_lines: Option<Vec<DisclinationLine>> = None;

    // Deterministic RNG seeded by a fixed constant; each step re-seeds from
    // the step index to keep runs reproducible regardless of snap_every.
    let mut rng = SmallRng::seed_from_u64(0xdead_beef_cafe_1234);

    for step in 0..n_steps {
        let t = step as f64 * p.dt;

        // 1. Compute Beris-Edwards RHS (dry: no velocity).
        let rhs = beris_edwards_rhs_3d(&q, None, p, t);

        // 2. Euler time step.
        q = euler.step(&q, p.dt, &rhs);

        // 3. Add Langevin noise to each of the 5 Q-components at every vertex.
        //    The 5 independent Gaussian increments are already in the
        //    symmetric-traceless basis, so no symmetrization is needed.
        if p.noise_amp > 0.0 {
            let amp = p.noise_amp * p.dt.sqrt();
            let n_verts = q.len();
            // Re-seed per step for reproducibility; cheap with SmallRng.
            rng = SmallRng::seed_from_u64(step as u64 ^ 0xdead_beef_cafe_1234);
            for k in 0..n_verts {
                // Box-Muller to generate 5 independent N(0,1) samples.
                let samples = box_muller_5(&mut rng);
                for c in 0..5 {
                    q.q[k][c] += amp * samples[c];
                }
            }
        }

        // 4. Snapshot trigger: when (step+1) % snap_every == 0.
        if snap_every > 0 && (step + 1) % snap_every == 0 {
            let snap_idx = (step + 1) / snap_every;
            let t_snap = (step + 1) as f64 * p.dt;

            // Defect detection and event tracking.
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

            // Write Q snapshot.
            let npy_path = out_dir.join(format!("q_{snap_idx:06}.npy"));
            if let Err(e) = write_npy(&npy_path, &q.q, p.nx, p.ny, p.nz, 5) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", npy_path.display());
            }
        }
    }

    // Write stats.json at the end.
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
pub fn run_mars_3d_full(
    q_init: &QField3D,
    phi_init: &ScalarField3D,
    p: &MarsParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &Path,
    track_defects: bool,
) -> (QField3D, ScalarField3D, Vec<BechStats3D>) {
    let euler = EulerIntegrator3D;
    let mut q = q_init.clone();
    let mut phi = phi_init.clone();
    let mut stats: Vec<BechStats3D> = Vec::new();

    let mut prev_lines: Option<Vec<DisclinationLine>> = None;
    let mut rng = SmallRng::seed_from_u64(0xdead_beef_cafe_5678);

    for step in 0..n_steps {
        let t = step as f64 * p.dt;

        // 1. Stokes solve: get incompressible velocity driven by active stress.
        let (vel, _pressure) = stokes_solve_3d(&q, p);

        // 2. Beris-Edwards RHS with flow.
        let rhs = beris_edwards_rhs_3d(&q, Some(&vel), p, t);

        // 3. Euler step on Q.
        q = euler.step(&q, p.dt, &rhs);

        // 4. Langevin noise on Q.
        if p.noise_amp > 0.0 {
            let amp = p.noise_amp * p.dt.sqrt();
            let n_verts = q.len();
            rng = SmallRng::seed_from_u64(step as u64 ^ 0xdead_beef_cafe_5678);
            for k in 0..n_verts {
                let samples = box_muller_5(&mut rng);
                for c in 0..5 {
                    q.q[k][c] += amp * samples[c];
                }
            }
        }

        // 5. Cahn-Hilliard ETD step.
        //    Approximation: pass &q as q_lip (see doc comment above).
        phi = ch_step_etd_3d(&phi, &q, p, p.dt);

        // 6. Snapshot trigger.
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

            let s = compute_bech_stats(&q, &phi, &lines, n_events, t_snap);
            stats.push(s);

            // Write Q snapshot.
            let q_path = out_dir.join(format!("q_{snap_idx:06}.npy"));
            if let Err(e) = write_npy(&q_path, &q.q, p.nx, p.ny, p.nz, 5) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", q_path.display());
            }

            // Write phi snapshot as (nx,ny,nz,1) for uniform API — wrap each f64 in [f64;1].
            let phi_wrapped: Vec<[f64; 1]> = phi.phi.iter().map(|&v| [v]).collect();
            let phi_path = out_dir.join(format!("phi_{snap_idx:06}.npy"));
            if let Err(e) = write_npy(&phi_path, &phi_wrapped, p.nx, p.ny, p.nz, 1) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", phi_path.display());
            }

            // Write velocity snapshot as (nx,ny,nz,3).
            let vel_path = out_dir.join(format!("vel_{snap_idx:06}.npy"));
            if let Err(e) = write_npy(&vel_path, &vel.u, p.nx, p.ny, p.nz, 3) {
                eprintln!("[runner_3d] warn: failed to write {}: {e}", vel_path.display());
            }
        }
    }

    // Write stats.json.
    let stats_path = out_dir.join("stats.json");
    if let Ok(json) = serde_json::to_string_pretty(&stats) {
        let _ = std::fs::write(&stats_path, json);
    }

    (q, phi, stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate 5 independent standard-normal samples using Box-Muller.
///
/// Consumes 6 uniform(0,1) draws from `rng` (3 pairs), discarding the 6th
/// sample from the last pair to produce exactly 5 outputs.
fn box_muller_5(rng: &mut SmallRng) -> [f64; 5] {
    let mut out = [0.0f64; 5];
    let mut i = 0;
    while i < 5 {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE); // avoid ln(0)
        let u2: f64 = rng.random::<f64>();
        let (z0, z1) = box_muller(u1, u2);
        out[i] = z0;
        i += 1;
        if i < 5 {
            out[i] = z1;
            i += 1;
        }
    }
    out
}

/// Box-Muller transform: maps two uniform(0,1) draws to two independent N(0,1).
#[inline]
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

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

/// Write a C-contiguous `f64` array to a minimal NumPy `.npy` v1.0 file.
///
/// The on-disk shape is `(nx, ny, nz, n_comp)`. `data` must have exactly
/// `nx * ny * nz` entries, each of length `n_comp`.
///
/// The header is padded with spaces to a multiple of 64 bytes, followed by a
/// newline, as required by the NumPy format specification.
fn write_npy<const N: usize>(
    path: &Path,
    data: &[[f64; N]],
    nx: usize,
    ny: usize,
    nz: usize,
    n_comp: usize,
) -> std::io::Result<()> {
    // Build the Python-dict header string.
    let header_dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({nx}, {ny}, {nz}, {n_comp}), }}"
    );

    // The magic + header-length field occupies 10 bytes:
    //   6 bytes magic (\x93NUMPY\x01\x00) + 2 bytes HEADER_LEN (uint16 LE)
    //   + 2 bytes for version (already in magic) = total 10.
    // Wait: magic is \x93NUMPY\x01\x00 = 8 bytes, HEADER_LEN is 2 bytes = 10 total.
    // We need total_file_size_up_to_data to be a multiple of 64.
    // The header block (HEADER_LEN bytes) must include the dict + padding + \n.
    // total = 10 + HEADER_LEN; we want this to be a multiple of 64.
    let magic: &[u8] = b"\x93NUMPY\x01\x00"; // 8 bytes

    // The header_len field counts the bytes that follow it (dict + pad + newline).
    // We need: 10 + header_len ≡ 0 (mod 64), i.e. header_len ≡ 54 (mod 64).
    // But any multiple of 64 >= header_dict.len()+1 works.
    let dict_plus_newline = header_dict.len() + 1; // +1 for \n
    // Round up to next multiple of 64 after the 10-byte prefix.
    let header_len = {
        let needed = dict_plus_newline;
        let rounded = ((needed + 64 - 1) / 64) * 64;
        // Ensure also (10 + rounded) % 64 == 0.
        // Since rounded is already a multiple of 64, (10 + rounded) % 64 = 10 % 64 = 10 != 0.
        // The spec says the TOTAL of (magic + header_len_field + header_data) must be divisible
        // by 64. That means header_data must have length such that (8 + 2 + header_data_len) % 64 == 0,
        // i.e. header_data_len % 64 == 54.
        let rem = (needed as isize).rem_euclid(64) as usize;
        let pad_needed = if rem == 54 { 0 } else { (54 + 64 - rem) % 64 };
        needed + pad_needed
    };

    let padding = header_len - dict_plus_newline;

    let mut padded_header = header_dict.clone();
    for _ in 0..padding {
        padded_header.push(' ');
    }
    padded_header.push('\n');

    debug_assert_eq!(padded_header.len(), header_len);
    debug_assert_eq!((8 + 2 + header_len) % 64, 0);

    let header_len_u16: u16 = header_len as u16;

    let mut f = std::fs::File::create(path)?;
    f.write_all(magic)?;
    f.write_all(&header_len_u16.to_le_bytes())?;
    f.write_all(padded_header.as_bytes())?;
    for entry in data {
        for &v in entry[..n_comp].iter() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::MarsParams3D;
    use volterra_fields::{QField3D, ScalarField3D};

    /// Smoke test: 5 steps of dry active turbulence on a tiny grid, no crash.
    #[test]
    fn test_run_mars_3d_dry_smoke() {
        let p = MarsParams3D::default_test(); // 16^3
        let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
        let tmp = std::env::temp_dir().join("volterra_test_run");
        std::fs::create_dir_all(&tmp).unwrap();
        let (q_final, stats) = run_mars_3d(&q_init, &p, 5, 5, &tmp, false);
        assert_eq!(q_final.len(), q_init.len());
        assert_eq!(stats.len(), 1);
        assert!(stats[0].mean_s >= 0.0);
    }

    /// Smoke test for the full BECH runner.
    #[test]
    fn test_run_mars_3d_full_smoke() {
        let p = MarsParams3D::default_test();
        let q_init = QField3D::random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.01, 42);
        let phi_init = ScalarField3D::uniform(p.nx, p.ny, p.nz, p.dx, 0.3);
        let tmp = std::env::temp_dir().join("volterra_test_full");
        std::fs::create_dir_all(&tmp).unwrap();
        let (q_f, phi_f, stats) = run_mars_3d_full(&q_init, &phi_init, &p, 5, 5, &tmp, false);
        assert_eq!(q_f.len(), q_init.len());
        assert!((phi_f.mean() - 0.3).abs() < 0.01, "mass roughly conserved, got mean={}", phi_f.mean());
        assert_eq!(stats.len(), 1);
    }
}
