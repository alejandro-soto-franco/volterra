//! Topological-defect detection from a 2D Q-tensor grid.
//!
//! Matches the scheme in the reference `braid_tracker.py`: the defect density
//!
//! ```text
//! ss = (2 dx Qxy)(2 dy Qxx) - (2 dx Qxx)(2 dy Qxy)
//! ```
//!
//! (the Jacobian of `(Qxx, Qxy)` with respect to `(x, y)`, via central
//! differences) spikes at disclination cores. Cells with `|ss| > threshold` are
//! defect candidates; 8-connected components are clustered and the centroid of
//! each cluster is the defect position. The charge sign is `-sign(ss)` at the
//! cluster (the negated sign matches the reference convention).

/// A detected topological defect: position and charge sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Defect {
    /// Position `[x, y]` (grid-index units; cluster centroid).
    pub pos: [f64; 2],
    /// Charge sign, `+1` or `-1`.
    pub charge: i8,
}

/// Detect defects in a row-major `nx * ny` Q-tensor grid.
///
/// `qxx[x * ny + y]` and `qxy[x * ny + y]` give the two independent components
/// at grid cell `(x, y)`. `mask[x * ny + y] == false` forces `ss = 0` at that
/// cell (used to zero a boundary ring or cells outside the simulated domain).
/// Central differences use periodic wraparound, matching the reference.
///
/// Returns one [`Defect`] per connected cluster of super-threshold cells.
pub fn detect_defects(
    qxx: &[f64],
    qxy: &[f64],
    nx: usize,
    ny: usize,
    threshold: f64,
    mask: &[bool],
) -> Vec<Defect> {
    let idx = |x: usize, y: usize| x * ny + y;

    // Defect density ss via central differences with periodic wraparound; zeroed
    // wherever the mask is inactive. Cells with |ss| <= threshold are non-defect.
    let mut field = vec![0.0f64; nx * ny];
    for x in 0..nx {
        for y in 0..ny {
            if !mask[idx(x, y)] {
                continue;
            }
            let (xup, xdn) = ((x + 1) % nx, (x + nx - 1) % nx);
            let (yup, ydn) = ((y + 1) % ny, (y + ny - 1) % ny);
            let dx_qxx = qxx[idx(xup, y)] - qxx[idx(xdn, y)];
            let dx_qxy = qxy[idx(xup, y)] - qxy[idx(xdn, y)];
            let dy_qxx = qxx[idx(x, yup)] - qxx[idx(x, ydn)];
            let dy_qxy = qxy[idx(x, yup)] - qxy[idx(x, ydn)];
            let ss = dx_qxy * dy_qxx - dx_qxx * dy_qxy;
            if ss.abs() > threshold {
                field[idx(x, y)] = ss;
            }
        }
    }

    // Same-sign 8-connected components of the thresholded field. Each component is
    // one defect; position = centroid, charge = -sign(ss) (reference convention).
    let mut visited = vec![false; nx * ny];
    let mut defects = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            let seed = field[idx(x, y)];
            if seed == 0.0 || visited[idx(x, y)] {
                continue;
            }
            let sign = seed.signum();
            let mut stack = vec![(x, y)];
            visited[idx(x, y)] = true;
            let (mut sx, mut sy, mut count) = (0.0f64, 0.0f64, 0usize);
            while let Some((cx, cy)) = stack.pop() {
                sx += cx as f64;
                sy += cy as f64;
                count += 1;
                for (dx, dy) in [
                    (1i64, 0i64),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    let nxp = cx as i64 + dx;
                    let nyp = cy as i64 + dy;
                    if nxp < 0 || nyp < 0 || nxp >= nx as i64 || nyp >= ny as i64 {
                        continue;
                    }
                    let (nxp, nyp) = (nxp as usize, nyp as usize);
                    let ni = idx(nxp, nyp);
                    if !visited[ni] && field[ni] != 0.0 && field[ni].signum() == sign {
                        visited[ni] = true;
                        stack.push((nxp, nyp));
                    }
                }
            }
            defects.push(Defect {
                pos: [sx / count as f64, sy / count as f64],
                charge: -(sign as i8),
            });
        }
    }
    defects
}

#[cfg(test)]
mod defect_tests {
    use super::*;

    /// Build an `nx*ny` row-major Q field for a single `charge`-half defect at
    /// `(cx, cy)`. `+1/2`: (Qxx, Qxy) = (X/r, Y/r); `-1/2`: (X/r, -Y/r).
    pub(super) fn winding_field(
        nx: usize,
        ny: usize,
        cx: f64,
        cy: f64,
        plus: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut qxx = vec![0.0; nx * ny];
        let mut qxy = vec![0.0; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let i = x * ny + y;
                if r < 0.5 {
                    continue;
                }
                qxx[i] = dx / r;
                qxy[i] = if plus { dy / r } else { -dy / r };
            }
        }
        (qxx, qxy)
    }

    /// Mask that zeroes a one-cell border ring (true = active interior cell).
    pub(super) fn interior_mask(nx: usize, ny: usize) -> Vec<bool> {
        let mut m = vec![true; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                if x == 0 || y == 0 || x == nx - 1 || y == ny - 1 {
                    m[x * ny + y] = false;
                }
            }
        }
        m
    }

    #[test]
    fn uniform_field_has_no_defects() {
        let (nx, ny) = (20, 20);
        let qxx = vec![0.7; nx * ny];
        let qxy = vec![0.3; nx * ny];
        let mask = vec![true; nx * ny];
        assert!(detect_defects(&qxx, &qxy, nx, ny, 0.1, &mask).is_empty());
    }

    #[test]
    fn single_plus_half_defect_found_near_core() {
        let (nx, ny) = (41, 41);
        let (qxx, qxy) = winding_field(nx, ny, 20.0, 20.0, true);
        let mask = interior_mask(nx, ny);
        let defects = detect_defects(&qxx, &qxy, nx, ny, 0.5, &mask);
        assert_eq!(
            defects.len(),
            1,
            "expected one defect, got {}",
            defects.len()
        );
        let [x, y] = defects[0].pos;
        assert!(
            (x - 20.0).abs() < 2.0 && (y - 20.0).abs() < 2.0,
            "core at ({x},{y})"
        );
    }

    #[test]
    fn plus_and_minus_half_have_opposite_charge() {
        let (nx, ny) = (41, 41);
        let mask = interior_mask(nx, ny);
        let (pxx, pxy) = winding_field(nx, ny, 20.0, 20.0, true);
        let (mxx, mxy) = winding_field(nx, ny, 20.0, 20.0, false);
        let dp = detect_defects(&pxx, &pxy, nx, ny, 0.5, &mask);
        let dm = detect_defects(&mxx, &mxy, nx, ny, 0.5, &mask);
        assert_eq!(dp.len(), 1);
        assert_eq!(dm.len(), 1);
        assert_eq!(dp[0].charge, -dm[0].charge, "charges should be opposite");
    }
}

// ---------------------------------------------------------------------------
// Winding detection
// ---------------------------------------------------------------------------

/// Detect defects by the winding of the director around each lattice plaquette.
///
/// The Jacobian scheme above thresholds a quantity that scales as the square of
/// the director gradient, so a threshold calibrated at one core size is wrong at
/// another: a coherence length of 0.975 lattice spacings and one of 12.9 put
/// `ss` about two hundred and eighty times apart at the core. Every epitrochoid
/// point in arXiv:2503.10880 Fig. 7 sits at a coherence length an order of
/// magnitude above the steady-winding-circle runs of Figs. 2-4, which is what
/// makes a threshold-free detector necessary rather than merely tidier.
///
/// Winding carries no such scale. The director angle is
/// `phi = atan2(Qxy, Qxx) / 2`, defined modulo `pi`; summing the differences
/// around the four corners of a plaquette, each wrapped into `(-pi/2, pi/2]`,
/// gives `0` away from a core and `+/- pi` at one, whatever the core's size or
/// the field's amplitude. The charge is that sum over `2 pi`, in units of
/// `1/2`: `+1` for a `+1/2` defect, matching [`Defect::charge`].
///
/// A plaquette contributes only when all four of its corners are unmasked, so a
/// domain boundary neither generates nor absorbs winding. Adjacent same-sign
/// plaquettes are merged, since a core sitting between two lattice sites can
/// register on more than one.
///
/// A core sitting exactly on a lattice site is enclosed by no plaquette and is
/// invisible here. That position is also the one where the site's own director
/// is undefined, `S` being zero at a core, so no lattice scheme reads it. A
/// core anywhere else lies strictly inside a plaquette and registers there.
pub fn detect_defects_winding(
    qxx: &[f64],
    qxy: &[f64],
    nx: usize,
    ny: usize,
    mask: &[bool],
) -> Vec<Defect> {
    use std::f64::consts::PI;

    let idx = |x: usize, y: usize| x * ny + y;
    let phi = |x: usize, y: usize| 0.5 * qxy[idx(x, y)].atan2(qxx[idx(x, y)]);
    // Wrap into (-pi/2, pi/2]: the director is a line, not a vector, so a
    // corner-to-corner turn of more than a right angle is read as the shorter
    // turn the other way.
    let wrap_half = |mut d: f64| {
        while d > PI / 2.0 {
            d -= PI;
        }
        while d <= -PI / 2.0 {
            d += PI;
        }
        d
    };

    // Charge per plaquette, indexed by its lower-left corner.
    let mut charge = vec![0i8; nx * ny];
    for x in 0..nx.saturating_sub(1) {
        for y in 0..ny.saturating_sub(1) {
            let corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
            if !corners.iter().all(|&(cx, cy)| mask[idx(cx, cy)]) {
                continue;
            }
            let mut total = 0.0;
            for i in 0..4 {
                let (ax, ay) = corners[i];
                let (bx, by) = corners[(i + 1) % 4];
                total += wrap_half(phi(bx, by) - phi(ax, ay));
            }
            // total is a multiple of pi; +pi is a +1/2 defect.
            charge[idx(x, y)] = (total / PI).round() as i8;
        }
    }

    // Merge 8-connected same-sign plaquettes into one defect each. The position
    // is the centroid of the plaquette centres, so it is offset by half a cell
    // from the corner index, which is where the defect actually sits.
    let mut visited = vec![false; nx * ny];
    let mut defects = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            let seed = charge[idx(x, y)];
            if seed == 0 || visited[idx(x, y)] {
                continue;
            }
            let sign = seed.signum();
            let mut stack = vec![(x, y)];
            visited[idx(x, y)] = true;
            let (mut sx, mut sy, mut count) = (0.0f64, 0.0f64, 0usize);
            while let Some((cx, cy)) = stack.pop() {
                sx += cx as f64 + 0.5;
                sy += cy as f64 + 0.5;
                count += 1;
                for (dx, dy) in [
                    (1i64, 0i64),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    let nxp = cx as i64 + dx;
                    let nyp = cy as i64 + dy;
                    if nxp < 0 || nyp < 0 || nxp >= nx as i64 || nyp >= ny as i64 {
                        continue;
                    }
                    let (nxp, nyp) = (nxp as usize, nyp as usize);
                    let ni = idx(nxp, nyp);
                    if !visited[ni] && charge[ni].signum() == sign && charge[ni] != 0 {
                        visited[ni] = true;
                        stack.push((nxp, nyp));
                    }
                }
            }
            defects.push(Defect {
                pos: [sx / count as f64, sy / count as f64],
                charge: sign,
            });
        }
    }
    defects
}

#[cfg(test)]
mod winding_tests {
    use super::*;
    use super::defect_tests::{interior_mask, winding_field};

    /// A uniform director has no winding anywhere.
    #[test]
    fn uniform_field_has_no_winding_defects() {
        let (nx, ny) = (20, 20);
        let qxx = vec![0.7; nx * ny];
        let qxy = vec![0.3; nx * ny];
        let mask = vec![true; nx * ny];
        assert!(detect_defects_winding(&qxx, &qxy, nx, ny, &mask).is_empty());
    }

    /// A single `+1/2` core is found, with the charge sign the Jacobian scheme
    /// gives, so the two detectors are interchangeable downstream.
    #[test]
    fn agrees_with_the_jacobian_scheme_on_sign_and_position() {
        let (nx, ny) = (41, 41);
        let mask = interior_mask(nx, ny);
        for plus in [true, false] {
            let (qxx, qxy) = winding_field(nx, ny, 20.5, 20.5, plus);
            let w = detect_defects_winding(&qxx, &qxy, nx, ny, &mask);
            let j = detect_defects(&qxx, &qxy, nx, ny, 0.5, &mask);
            assert_eq!(w.len(), 1, "winding found {} defects", w.len());
            assert_eq!(j.len(), 1);
            assert_eq!(w[0].charge, j[0].charge, "charge sign disagrees");
            assert!(
                (w[0].pos[0] - 20.5).abs() < 2.0 && (w[0].pos[1] - 20.5).abs() < 2.0,
                "core at {:?}",
                w[0].pos
            );
        }
    }

    /// Winding is independent of the field's amplitude, where the Jacobian
    /// threshold is not. Scaling Q by 1/20 scales `ss` by 1/400, which takes a
    /// real defect below any fixed threshold; the winding detector still finds
    /// it. This is the property the epitrochoid runs need.
    #[test]
    fn winding_survives_a_scaling_that_defeats_a_fixed_threshold() {
        let (nx, ny) = (41, 41);
        let mask = interior_mask(nx, ny);
        let (qxx, qxy) = winding_field(nx, ny, 20.5, 20.5, true);
        let faint_xx: Vec<f64> = qxx.iter().map(|v| v / 20.0).collect();
        let faint_xy: Vec<f64> = qxy.iter().map(|v| v / 20.0).collect();

        assert_eq!(
            detect_defects_winding(&faint_xx, &faint_xy, nx, ny, &mask).len(),
            1,
            "winding should be blind to the amplitude"
        );
        assert!(
            detect_defects(&faint_xx, &faint_xy, nx, ny, 0.5, &mask).is_empty(),
            "the fixed threshold should have missed it, or this test proves nothing"
        );
    }

    /// A `+1/2` and a `-1/2` at a distance are found as two defects of opposite
    /// sign, and the net charge is zero.
    #[test]
    fn a_pair_is_found_with_zero_net_charge() {
        let (nx, ny) = (61, 41);
        let mask = interior_mask(nx, ny);
        let mut qxx = vec![0.0; nx * ny];
        let mut qxy = vec![0.0; nx * ny];
        // Superpose the two director angles, which is how a real pair looks far
        // from either core: phi = phi_+ + phi_-.
        for x in 0..nx {
            for y in 0..ny {
                let a = (y as f64 - 20.5).atan2(x as f64 - 20.5) * 0.5;
                let b = -(y as f64 - 20.5).atan2(x as f64 - 40.5) * 0.5;
                let phi = a + b;
                qxx[x * ny + y] = (2.0 * phi).cos();
                qxy[x * ny + y] = (2.0 * phi).sin();
            }
        }
        let d = detect_defects_winding(&qxx, &qxy, nx, ny, &mask);
        assert_eq!(d.len(), 2, "expected a pair, got {d:?}");
        assert_eq!(
            d.iter().map(|d| d.charge as i32).sum::<i32>(),
            0,
            "a pair must have zero net charge"
        );
    }
}
