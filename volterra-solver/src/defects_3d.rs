// ~/volterra/volterra-solver/src/defects_3d.rs

//! 3-D disclination detection and event tracking.
//!
//! Wraps the three-layer pipeline exposed by `cartan_geo::disclination`:
//!
//! | Layer | Function |
//! |-------|----------|
//! | 1: segment scan | [`scan_disclination_lines_3d`]: holonomy per grid edge |
//! | 2: line assembly | [`connect_disclination_lines`]: BFS connected components |
//! | 3: event tracking | [`track_disclination_events`]: creation/annihilation/reconnection |

use volterra_fields::QField3D;
use cartan_geo::disclination::{
    scan_disclination_lines_3d, connect_disclination_lines,
    track_disclination_events, DisclinationLine, DisclinationEvent,
};

/// Scan a 3D Q-tensor field for disclination lines.
///
/// Runs the full Layer-1 + Layer-2 pipeline from `cartan_geo::disclination`:
///
/// 1. `scan_disclination_lines_3d` detects individual pierced edges via
///    dual-loop holonomy.
/// 2. `connect_disclination_lines` assembles the edge segments into ordered
///    lines with Frenet-Serret geometry.
///
/// Returns an empty `Vec` when no defects are present (e.g. a uniform or
/// weakly-perturbed ordered field).
///
/// # Parameters
///
/// - `q`: reference to the 3D Q-tensor field on a regular Cartesian grid.
///
/// # Returns
///
/// `Vec<DisclinationLine>`, one entry per connected disclination line.
pub fn scan_defects_3d(q: &QField3D) -> Vec<DisclinationLine> {
    let segs = scan_disclination_lines_3d(q);
    connect_disclination_lines(&segs, q.dx)
}

/// Track topology-changing events between two consecutive frames of
/// disclination lines.
///
/// Wraps `cartan_geo::disclination` Layer 3 (`track_disclination_events`).
/// Events detected include creation, annihilation, and reconnection of
/// disclination lines.
///
/// # Parameters
///
/// - `lines_a`: disclination lines at the earlier frame.
/// - `lines_b`: disclination lines at the later frame.
/// - `frame`: index of the later frame (used in returned `DisclinationEvent`s).
/// - `proximity_threshold`: distance below which two line endpoints are
///   considered the same point for event classification (in grid units).
///
/// # Returns
///
/// `Vec<DisclinationEvent>`, one entry per detected topological event.
pub fn track_defect_events(
    lines_a: &[DisclinationLine],
    lines_b: &[DisclinationLine],
    frame: usize,
    proximity_threshold: f64,
) -> Vec<DisclinationEvent> {
    track_disclination_events(lines_a, lines_b, frame, proximity_threshold)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_defects_3d_uniform_no_defects() {
        let q = QField3D::uniform(8, 8, 8, 1.0, [0.2, 0.0, 0.0, -0.1, 0.0]);
        let lines = scan_defects_3d(&q);
        assert!(lines.is_empty());
    }
}
