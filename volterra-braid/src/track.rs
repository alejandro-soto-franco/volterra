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

/// Assignment of every row to a distinct column at least total cost.
///
/// Greedy nearest-neighbour is what the reference tracker does and it is right
/// whenever the strands outnumber nothing, which is the case [`track`] handles:
/// there each strand takes its nearest unclaimed defect and the claims rarely
/// contend. Tracking a core inside a larger frame breaks that. When a pair
/// nucleates beside a core strand, whichever strand the loop reaches first takes
/// the newcomer, and the strand it belonged to is pushed onto whatever remains,
/// which can be across the domain. The displacement bound then rejects the whole
/// window, and it was rejecting 28 of 30 at the paper's metastable-golden point.
///
/// Minimising the total displacement instead removes the order dependence, and
/// the pairing is then a property of the two frames rather than of the loop.
/// This is the Jonker-Volgenant shortest-augmenting-path form, on a rectangular
/// matrix with `rows <= cols`, in `O(rows^2 cols)`; the strand count is single
/// digits, so the cost is irrelevant beside the frame loop around it.
fn assign_min_cost(cost: &[Vec<f64>]) -> Vec<usize> {
    let m = cost.len();
    let n = if m == 0 { 0 } else { cost[0].len() };
    assert!(m <= n, "assignment needs at least as many columns as rows");
    // One-based potentials and column assignment, with index 0 as the sentinel
    // the augmenting path starts from.
    let mut u = vec![0.0f64; m + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=m {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut out = vec![usize::MAX; m];
    for j in 1..=n {
        if p[j] != 0 {
            out[p[j] - 1] = j - 1;
        }
    }
    out
}

/// Track only the worldlines that persist across every frame of the window.
///
/// [`track`] fixes the strand count from the first frame and requires every
/// later frame to hold at least as many defects. A confined active nematic does
/// not oblige: pairs nucleate and annihilate around a braided core, so the count
/// fluctuates and the first frame is as likely to hold a transient as not. Fed
/// such a series, [`track`] braids the transients together with the core and
/// returns whatever entropy that mixture carries, which is larger than the
/// core's and unrepeatable between windows. At the paper's metastable-golden
/// point that showed up as modal readings of 3.5 to 5.8 agreed by a fifth of the
/// projection axes, where the core alone reads `2 log phi` on most of them.
///
/// The persistent core is extracted instead. Any strand alive in every frame is
/// alive in the frame holding the fewest defects, so that frame seeds the
/// tracking, which then runs forwards and backwards from it. Later frames may
/// hold more defects than there are strands; the surplus are transients and are
/// left unmatched.
///
/// `max_step` is a displacement a strand is not expected to cover in one frame.
/// A strand that dies has no successor near it, so the assignment attaches it to
/// an unrelated defect and manufactures a crossing, and that shows up as a step
/// far outside the flow's own scale. It is reported rather than enforced: the
/// count of steps that exceed it and the largest step actually taken come back
/// with the worldlines, and the caller decides what a tolerable number is. One
/// bad step in a window of two thousand assignments perturbs the braid word in
/// one place and cannot move a reading held by seven projection axes; rejecting
/// the window on it threw away 28 of 30 windows at the paper's
/// metastable-golden point, every one of which reads `2 log phi`.
///
/// Returns `None` if `frames` is empty or the seed frame holds no defects.
pub fn track_core(
    frames: &[Vec<Defect>],
    max_step: f64,
) -> Option<(Vec<Worldline>, f64, usize, usize)> {
    if frames.is_empty() {
        return None;
    }
    let seed = (0..frames.len()).min_by_key(|&i| frames[i].len())?;
    let dim = frames[seed].len();
    if dim == 0 {
        return None;
    }
    let max2 = max_step * max_step;
    let mut worst = 0.0f64;
    let mut over = 0usize;
    let mut steps = 0usize;

    // One deque of positions per strand, built outwards from the seed frame.
    let mut back: Vec<Vec<[f64; 2]>> = frames[seed].iter().map(|d| vec![d.pos]).collect();
    let mut fwd: Vec<Vec<[f64; 2]>> = vec![Vec::new(); dim];
    let charges: Vec<i8> = frames[seed].iter().map(|d| d.charge).collect();

    // `heads` carries the frontier in whichever direction is being extended.
    let extend = |heads: &mut Vec<[f64; 2]>,
                  frame: &Vec<Defect>,
                  worst: &mut f64,
                  over: &mut usize,
                  steps: &mut usize|
     -> Option<Vec<[f64; 2]>> {
        if frame.len() < heads.len() {
            return None;
        }
        let cost: Vec<Vec<f64>> = heads
            .iter()
            .map(|prev| {
                frame
                    .iter()
                    .map(|def| {
                        let dx = def.pos[0] - prev[0];
                        let dy = def.pos[1] - prev[1];
                        dx * dx + dy * dy
                    })
                    .collect()
            })
            .collect();
        let pick = assign_min_cost(&cost);
        let mut next = Vec::with_capacity(heads.len());
        for (s, &j) in pick.iter().enumerate() {
            if j == usize::MAX {
                return None;
            }
            *steps += 1;
            if cost[s][j] > max2 {
                *over += 1;
            }
            *worst = worst.max(cost[s][j].sqrt());
            next.push(frame[j].pos);
        }
        Some(next)
    };

    let mut heads: Vec<[f64; 2]> = back.iter().map(|w| w[0]).collect();
    for frame in frames[seed + 1..].iter() {
        heads = extend(&mut heads, frame, &mut worst, &mut over, &mut steps)?;
        for (s, p) in heads.iter().enumerate() {
            fwd[s].push(*p);
        }
    }
    let mut heads: Vec<[f64; 2]> = back.iter().map(|w| w[0]).collect();
    for frame in frames[..seed].iter().rev() {
        heads = extend(&mut heads, frame, &mut worst, &mut over, &mut steps)?;
        for (s, p) in heads.iter().enumerate() {
            back[s].push(*p);
        }
    }

    let out = (0..dim)
        .map(|s| {
            // `back[s]` was built outwards from the seed, so it runs backwards in
            // time and holds the seed position at its front.
            let mut positions: Vec<[f64; 2]> = back[s].iter().rev().copied().collect();
            positions.extend_from_slice(&fwd[s]);
            Worldline {
                positions,
                charge: charges[s],
            }
        })
        .collect();
    Some((out, worst, over, steps))
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

    /// A braided core with a transient defect alive over the first half.
    ///
    /// This is the shape of a real metastable window: three strands that braid,
    /// plus a pair partner that nucleates, wanders and annihilates. The transient
    /// sits in the first frame, so [`track`] takes it for a fourth strand and then
    /// meets a frame that cannot supply one.
    fn golden_with_a_transient() -> Vec<Vec<Defect>> {
        use crate::synthetic::{RealizeOpts, golden_orbit};
        let mut frames = golden_orbit(&RealizeOpts {
            frames_per_gen: 8,
            periods: 6,
        });
        let half = frames.len() / 2;
        for (i, f) in frames.iter_mut().enumerate().take(half) {
            // Well away from the three strands, which live in x in [1, 3].
            f.insert(0, d(20.0 + 0.05 * i as f64, 20.0, 1));
        }
        frames
    }

    #[test]
    fn core_ignores_a_transient_that_track_takes_for_a_strand() {
        let frames = golden_with_a_transient();
        assert_eq!(frames[0].len(), 4);
        assert_eq!(frames[frames.len() - 1].len(), 3);

        let (core, worst, over, steps) = track_core(&frames, 5.0).expect("a core");
        assert_eq!(core.len(), 3, "the persistent core is three strands");
        assert_eq!(over, 0, "no step should exceed the bound on a clean core");
        assert_eq!(steps, 3 * (frames.len() - 1));
        for wl in &core {
            assert_eq!(
                wl.positions.len(),
                frames.len(),
                "every core strand spans the whole window"
            );
            for p in &wl.positions {
                assert!(p[0] < 10.0, "a core strand picked up the transient at {p:?}");
            }
        }
        assert!(worst < 5.0, "worst step {worst} should be inside the bound");
    }

    #[test]
    fn core_recovers_the_golden_entropy_the_mixture_loses() {
        use crate::extract_braidword;
        let frames = golden_with_a_transient();
        let (core, _, _, _) = track_core(&frames, 5.0).expect("a core");
        let h = extract_braidword(&core).entropy_per_period();
        assert!(
            (h - crate::GOLDEN_H).abs() < 1e-9,
            "core read {h}, want {}",
            crate::GOLDEN_H
        );

        // The stub the test is against: seeded from the first frame, the
        // transient is a strand, and the series does not even admit a reading.
        let clean = &frames[frames.len() / 2..];
        let mixed = extract_braidword(&track(clean)).entropy_per_period();
        assert!(
            (mixed - crate::GOLDEN_H).abs() < 1e-9,
            "the same frames without the transient must still read golden, got {mixed}"
        );
    }

    #[test]
    fn core_reports_a_strand_with_no_near_successor() {
        // Two strands sitting still, then one of them is replaced by a defect far
        // away. Nothing in the third frame continues it, so the assignment can
        // only reach the distant defect, and the step count is what says so.
        let frames = vec![
            vec![d(0.0, 0.0, 1), d(1.0, 0.0, 1), d(2.0, 0.0, 1)],
            vec![d(0.0, 0.0, 1), d(1.0, 0.0, 1)],
            vec![d(0.0, 0.0, 1), d(40.0, 0.0, 1)],
        ];
        // The bound is reported, not enforced, so the window comes back either
        // way and the count of offending steps is what separates the two.
        let (_, worst, over, _) = track_core(&frames, 2.0).expect("a core");
        assert_eq!(over, 1, "one assignment jumps the gap");
        assert!(worst > 35.0, "the jump is the whole 40-unit gap, got {worst}");
        let (_, _, none_over, _) = track_core(&frames, 50.0).expect("a core");
        assert_eq!(none_over, 0, "a bound past the jump is not exceeded");
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
