//! Defect-seeded initial conditions.
//!
//! A director field built to have prescribed disclinations at prescribed
//! places, rather than the random or uniform field the reference codes ship.
//!
//! # Why this exists
//!
//! Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing active
//! nematics", Phys. Rev. E 109, 014606 (2024), report a periodic orbit of four
//! defects on a torus and map its stability by continuation: "The black curve
//! uses an initial Q-field taken from the periodic state at `ell_a = 3`". A
//! random initial condition at the same parameters competes with a defect-free
//! stationary state and does not reliably reach the orbit, so reproducing the
//! orbit needs a field that starts with the right number of defects.
//!
//! # Construction
//!
//! For defects of charge `q_k` at `r_k`, the one-constant equilibrium director
//! angle is the harmonic superposition
//!
//! ```text
//! theta(r) = theta_0 + sum_k q_k arg(r - r_k),
//! ```
//!
//! and `Q = S_0 (n (x) n - I/2)` with `n = (cos theta, sin theta)`. On a torus
//! the sum is single-valued only because the total charge vanishes, which
//! [`seeded_q`] checks. `arg` is still not periodic term by term, so each defect
//! is summed over its periodic images out to `images` boxes; with zero total
//! charge the far field cancels and a `3 x 3` tiling already leaves the seam
//! smooth to a part in a hundred, which the elastic relaxation in the first few
//! hundred steps removes entirely.
//!
//! Half-integer charges make `theta` double-valued as a function, which is
//! exactly right for a nematic: `n` and `-n` are the same director, and `Q`,
//! built from `cos^2` and `cos sin`, is single-valued.

use crate::index::{si, vi};

/// A disclination to place in the initial field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeededDefect {
    /// Position in lattice units.
    pub pos: [f64; 2],
    /// Topological charge, `+0.5` or `-0.5` for a nematic disclination.
    pub charge: f64,
}

impl SeededDefect {
    /// A `+1/2` disclination at `(x, y)`.
    pub fn plus(x: f64, y: f64) -> Self {
        Self { pos: [x, y], charge: 0.5 }
    }
    /// A `-1/2` disclination at `(x, y)`.
    pub fn minus(x: f64, y: f64) -> Self {
        Self { pos: [x, y], charge: -0.5 }
    }
}

/// The four-defect arrangement of Mitchell et al. (2024), Fig. 2.
///
/// Two `+1/2` defects on one diagonal of the fundamental cell and two `-1/2` on
/// the other, at the quarter points. The `+1/2` pair is what braids; the
/// `-1/2` pair is what the paper reports tracing "a strikingly square orbit,
/// braiding around no other defects".
pub fn mitchell_four_defect(lx: usize, ly: usize) -> Vec<SeededDefect> {
    let (fx, fy) = (lx as f64, ly as f64);
    vec![
        SeededDefect::plus(0.25 * fx, 0.25 * fy),
        SeededDefect::plus(0.75 * fx, 0.75 * fy),
        SeededDefect::minus(0.75 * fx, 0.25 * fy),
        SeededDefect::minus(0.25 * fx, 0.75 * fy),
    ]
}

/// The four-defect arrangement as Mitchell et al. (2024) draw it, Fig. 2a.
///
/// The `+1/2` pair sits at the midpoints of the left and the bottom edge, which
/// on the torus is one point each; the `-1/2` pair sits at the corner and at the
/// centre. [`mitchell_four_defect`] is the same configuration translated by
/// `(L/4, L/4)` and reflected, which reverses the sense the pair passes in; the
/// model is achiral, so the two are equivalent, and this one overlays the
/// published figure without a change of frame.
pub fn mitchell_figure_2a(lx: usize, ly: usize) -> Vec<SeededDefect> {
    let (fx, fy) = (lx as f64, ly as f64);
    vec![
        SeededDefect::plus(0.0, 0.5 * fy),
        SeededDefect::plus(0.5 * fx, 0.0),
        SeededDefect::minus(0.0, 0.0),
        SeededDefect::minus(0.5 * fx, 0.5 * fy),
    ]
}

/// Director angle field from a set of defects, on a periodic lattice.
///
/// `images` is the half-width of the periodic image sum: `1` sums the `3 x 3`
/// tiling, `2` the `5 x 5`. `theta_0` sets the far-field orientation.
pub fn director_from_defects(
    defects: &[SeededDefect],
    lx: usize,
    ly: usize,
    images: i32,
    theta_0: f64,
) -> Vec<f64> {
    let mut theta = vec![theta_0; lx * ly];
    let (fx, fy) = (lx as f64, ly as f64);
    for x in 0..lx {
        for y in 0..ly {
            let mut acc = 0.0;
            for d in defects {
                for ix in -images..=images {
                    for iy in -images..=images {
                        let dx = x as f64 - (d.pos[0] + ix as f64 * fx);
                        let dy = y as f64 - (d.pos[1] + iy as f64 * fy);
                        if dx == 0.0 && dy == 0.0 {
                            continue;
                        }
                        acc += d.charge * dy.atan2(dx);
                    }
                }
            }
            theta[si(x, y, ly)] += acc;
        }
    }
    theta
}

/// `Q = S_0 (n (x) n - I/2)` from a director angle field.
///
/// The same map `flow-solver.py`'s `initialize_Q_from_theta` uses:
/// `Q_xx = S_0 (cos^2 theta - 1/2)`, `Q_xy = S_0 cos theta sin theta`.
pub fn q_from_theta(theta: &[f64], s0: f64, lx: usize, ly: usize) -> Vec<f64> {
    let mut q = vec![0.0; lx * ly * 2];
    for x in 0..lx {
        for y in 0..ly {
            let (s, c) = theta[si(x, y, ly)].sin_cos();
            q[vi(x, y, ly, 0)] = s0 * (c * c - 0.5);
            q[vi(x, y, ly, 1)] = s0 * (c * s);
        }
    }
    q
}

/// Build a `Q` field with the given defects.
///
/// Returns `None` when the charges do not sum to zero, which no field on a
/// torus can satisfy.
pub fn seeded_q(
    defects: &[SeededDefect],
    lx: usize,
    ly: usize,
    s0: f64,
    theta_0: f64,
) -> Option<Vec<f64>> {
    let total: f64 = defects.iter().map(|d| d.charge).sum();
    if total.abs() > 1e-12 {
        return None;
    }
    let theta = director_from_defects(defects, lx, ly, 1, theta_0);
    Some(q_from_theta(&theta, s0, lx, ly))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A charge that does not cancel has no field on a torus.
    #[test]
    fn a_net_charge_is_refused() {
        let d = vec![SeededDefect::plus(4.0, 4.0)];
        assert!(seeded_q(&d, 16, 16, 1.0, 0.0).is_none());
        let ok = vec![SeededDefect::plus(4.0, 4.0), SeededDefect::minus(12.0, 12.0)];
        assert!(seeded_q(&ok, 16, 16, 1.0, 0.0).is_some());
    }

    /// The winding of the director about a small loop around each seeded
    /// position must be the charge that was asked for.
    #[test]
    fn the_winding_about_each_defect_is_its_charge() {
        let (lx, ly) = (128, 128);
        let defects = mitchell_four_defect(lx, ly);
        let theta = director_from_defects(&defects, lx, ly, 1, 0.3);

        for d in &defects {
            // Walk a square loop of radius 6 cells about the defect and sum the
            // change in 2 theta, taken to the nearest branch each step, since
            // the director identifies theta with theta + pi.
            let r = 6_i32;
            let (cx, cy) = (d.pos[0] as i32, d.pos[1] as i32);
            let mut path: Vec<(i32, i32)> = Vec::new();
            for k in -r..r { path.push((cx + k, cy - r)); }
            for k in -r..r { path.push((cx + r, cy + k)); }
            for k in -r..r { path.push((cx - k, cy + r)); }
            for k in -r..r { path.push((cx - r, cy - k)); }
            path.push(path[0]);

            let at = |p: (i32, i32)| {
                let x = p.0.rem_euclid(lx as i32) as usize;
                let y = p.1.rem_euclid(ly as i32) as usize;
                2.0 * theta[si(x, y, ly)]
            };
            let mut total = 0.0;
            for w in path.windows(2) {
                let mut dphi = at(w[1]) - at(w[0]);
                while dphi > std::f64::consts::PI { dphi -= 2.0 * std::f64::consts::PI; }
                while dphi < -std::f64::consts::PI { dphi += 2.0 * std::f64::consts::PI; }
                total += dphi;
            }
            let winding = total / (2.0 * std::f64::consts::PI);
            assert!(
                (winding - 2.0 * d.charge).abs() < 0.05,
                "at {:?}: winding of 2 theta is {winding}, wanted {}",
                d.pos,
                2.0 * d.charge
            );
        }
    }

    /// `Q` is single-valued across the seam even though `theta` is not.
    #[test]
    fn q_is_continuous_across_the_periodic_seam() {
        let (lx, ly) = (100, 100);
        let defects = mitchell_four_defect(lx, ly);
        let q = seeded_q(&defects, lx, ly, 2.0_f64.sqrt(), 0.0).unwrap();
        // Compare the seam column against its neighbour on the other side, well
        // away from every defect.
        let mut worst = 0.0_f64;
        for y in 0..ly {
            for c in 0..2 {
                let a = q[vi(0, y, ly, c)];
                let b = q[vi(lx - 1, y, ly, c)];
                worst = worst.max((a - b).abs());
            }
        }
        // A one-cell step in a smooth field, not a branch cut of size S_0.
        assert!(worst < 0.2, "seam jump {worst}");
    }

    /// The seeded field is what the crate's own detector finds.
    #[test]
    fn the_detector_finds_the_seeded_defects() {
        let (lx, ly) = (100, 100);
        let s0 = 2.0_f64.sqrt();
        let defects = mitchell_four_defect(lx, ly);
        let q = seeded_q(&defects, lx, ly, s0, 0.0).unwrap();
        let qxx: Vec<f64> = (0..lx * ly).map(|i| q[i * 2]).collect();
        let qxy: Vec<f64> = (0..lx * ly).map(|i| q[i * 2 + 1]).collect();
        let mask = vec![true; lx * ly];
        let found = volterra_braid::detect_defects(&qxx, &qxy, lx, ly, 0.05 * s0.powi(4), &mask);
        assert_eq!(found.len(), 4, "found {found:?}");
        assert_eq!(found.iter().filter(|d| d.charge > 0).count(), 2, "{found:?}");
        for d in &defects {
            let near = found.iter().any(|f| {
                let dx = (f.pos[0] - d.pos[0]).abs().min(lx as f64 - (f.pos[0] - d.pos[0]).abs());
                let dy = (f.pos[1] - d.pos[1]).abs().min(ly as f64 - (f.pos[1] - d.pos[1]).abs());
                (dx * dx + dy * dy).sqrt() < 3.0
            });
            assert!(near, "no detected defect near {:?}: {found:?}", d.pos);
        }
    }
}
