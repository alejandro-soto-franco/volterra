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
    fn winding_field(nx: usize, ny: usize, cx: f64, cy: f64, plus: bool) -> (Vec<f64>, Vec<f64>) {
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
    fn interior_mask(nx: usize, ny: usize) -> Vec<bool> {
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
