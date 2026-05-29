//! Frame-to-frame tracking of defects into worldlines.

use crate::defect::Defect;

/// The trajectory of a single tracked defect over time.
#[derive(Debug, Clone, PartialEq)]
pub struct Worldline {
    /// Position `[x, y]` at each tracked frame, in frame order.
    pub positions: Vec<[f64; 2]>,
    /// Topological charge sign of the defect (`+1` or `-1`).
    pub charge: i8,
}

/// Track defects across frames by greedy nearest-neighbour assignment.
///
/// The number of worldlines is fixed by the first frame (one per defect there).
/// For each subsequent frame, each worldline is extended with the nearest
/// remaining defect (smallest squared distance to its previous position),
/// matching the reference Python tracker.
///
/// Returns one [`Worldline`] per defect in `frames[0]`. Panics if `frames` is
/// empty.
pub fn track(frames: &[Vec<Defect>]) -> Vec<Worldline> {
    assert!(!frames.is_empty(), "track requires at least one frame");
    let dim = frames[0].len();
    let mut worldlines: Vec<Worldline> = frames[0]
        .iter()
        .map(|d| Worldline {
            positions: vec![d.pos],
            charge: d.charge,
        })
        .collect();

    for frame in &frames[1..] {
        assert!(
            frame.len() >= dim,
            "frame has {} defects, fewer than the {dim} tracked worldlines",
            frame.len()
        );
        // Greedy nearest-neighbour with removal: each worldline claims the nearest
        // not-yet-claimed defect (a per-frame bijection onto a subset of defects).
        let mut claimed = vec![false; frame.len()];
        for wl in worldlines.iter_mut() {
            let prev = *wl.positions.last().unwrap();
            let mut best = usize::MAX;
            let mut best_d2 = f64::INFINITY;
            for (j, def) in frame.iter().enumerate() {
                if claimed[j] {
                    continue;
                }
                let dx = def.pos[0] - prev[0];
                let dy = def.pos[1] - prev[1];
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = j;
                }
            }
            claimed[best] = true;
            wl.positions.push(frame[best].pos);
        }
    }
    worldlines
}

#[cfg(test)]
mod track_tests {
    use super::*;

    fn d(x: f64, y: f64, charge: i8) -> Defect {
        Defect {
            pos: [x, y],
            charge,
        }
    }

    #[test]
    fn two_defects_tracked_through_shuffled_order() {
        // A moves along +x at y=0; B sits high at x=10 moving along +y.
        // Frame 1 lists them in swapped order; the tracker must still follow each.
        let frames = vec![
            vec![d(0.0, 0.0, 1), d(10.0, 0.0, -1)],
            vec![d(10.0, 1.0, -1), d(1.0, 0.0, 1)],
            vec![d(2.0, 0.0, 1), d(10.0, 2.0, -1)],
        ];
        let wls = track(&frames);
        assert_eq!(wls.len(), 2);
        assert_eq!(wls[0].positions, vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        assert_eq!(
            wls[1].positions,
            vec![[10.0, 0.0], [10.0, 1.0], [10.0, 2.0]]
        );
    }

    #[test]
    fn charge_carried_from_first_frame() {
        let frames = vec![
            vec![d(0.0, 0.0, 1), d(5.0, 5.0, -1)],
            vec![d(0.5, 0.0, 1), d(5.0, 5.5, -1)],
        ];
        let wls = track(&frames);
        assert_eq!(wls[0].charge, 1);
        assert_eq!(wls[1].charge, -1);
    }
}
