//! Topological entropy by material-line stretching.
//!
//! The measurement of Mitchell, Sabbir, Geumhan, Smith, Klein and Beller,
//! "Maximally mixing active nematics", Phys. Rev. E 109, 014606 (2024): advect
//! an initial line segment as a passive material curve, record its length
//! against time, and read the topological entropy `h` off the slope of
//! `log(length)`. In two dimensions `h` is the asymptotic exponential
//! stretching rate of such a curve, and the paper's dimensionless form is
//! `h_tilde = h t_a` with `t_a = K / (zeta nu)` the active time scale
//! ([`crate::Params::active_time`]).
//!
//! # Refinement
//!
//! A curve stretched exponentially loses resolution exponentially, so a segment
//! longer than `max_segment` is halved, repeatedly, at every step. The paper's
//! own note that this becomes exponentially expensive in time, which is why
//! they move to the E-tec ensemble algorithm for the parameter sweep, is the
//! same cost: [`MaterialLine::points`] grows like `exp(h t)`. [`MaterialLine`]
//! stops refining at `max_points` and records the step it stopped at in
//! [`MaterialLine::saturated_at`], so a fit can be restricted to the interval
//! where the curve was still resolved.
//!
//! # Domain
//!
//! Periodic in both directions, with unit lattice spacing, matching
//! [`crate::boundary::periodic_boundary`]. Segment lengths use the minimum
//! image, so a segment that crosses the seam is measured across it rather than
//! around the box.

use crate::index::vi;

/// A passively advected polyline on the periodic lattice.
#[derive(Debug, Clone)]
pub struct MaterialLine {
    /// Vertex positions, wrapped into `[0, lx) x [0, ly)`.
    pub points: Vec<[f64; 2]>,
    lx: f64,
    ly: f64,
    /// Longest segment tolerated before a midpoint is inserted.
    pub max_segment: f64,
    /// Refinement stops here, to bound the cost.
    pub max_points: usize,
    /// Step index at which refinement first stopped, if it has.
    pub saturated_at: Option<usize>,
    /// `(t, length)` for every recorded step.
    pub history: Vec<(f64, f64)>,
}

impl MaterialLine {
    /// A straight open segment from `a` to `b`, resolved at `max_segment`.
    pub fn segment(a: [f64; 2], b: [f64; 2], lx: usize, ly: usize, n: usize) -> Self {
        let pts = (0..=n)
            .map(|i| {
                let s = i as f64 / n as f64;
                [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])]
            })
            .collect();
        Self {
            points: pts,
            lx: lx as f64,
            ly: ly as f64,
            max_segment: 0.5,
            max_points: 4_000_000,
            saturated_at: None,
            history: Vec::new(),
        }
    }

    /// Set the refinement tolerance and the point cap.
    pub fn with_limits(mut self, max_segment: f64, max_points: usize) -> Self {
        self.max_segment = max_segment;
        self.max_points = max_points;
        self
    }

    /// Displacement from `a` to `b`, by the minimum image on the torus.
    #[inline]
    fn delta(&self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut dx = b[0] - a[0];
        let mut dy = b[1] - a[1];
        if dx > 0.5 * self.lx {
            dx -= self.lx;
        } else if dx < -0.5 * self.lx {
            dx += self.lx;
        }
        if dy > 0.5 * self.ly {
            dy -= self.ly;
        } else if dy < -0.5 * self.ly {
            dy += self.ly;
        }
        [dx, dy]
    }

    /// Total arc length of the polyline.
    pub fn length(&self) -> f64 {
        self.points
            .windows(2)
            .map(|w| {
                let d = self.delta(w[0], w[1]);
                (d[0] * d[0] + d[1] * d[1]).sqrt()
            })
            .sum()
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the curve has no vertices.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Advance every vertex by `dt` under `u`, then refine and record.
    ///
    /// `u` is the two-component velocity field of
    /// [`crate::step::State`], sampled bilinearly. The field is held fixed
    /// across the step, so the integrator is fourth order in space and inherits
    /// the solver's own first-order treatment of the field in time; at the
    /// timestep an explicit Beris-Edwards run needs, that is the smaller error.
    pub fn advect(&mut self, u: &[f64], ly_cells: usize, dt: f64, step: usize, t: f64) {
        // Once refinement has stopped the curve is no longer resolved, so its
        // length is a lower bound that drifts further off with every step and no
        // fit may use it. Freezing here also bounds the cost, which would
        // otherwise sit at `max_points` samples per step for the rest of a run.
        if self.saturated_at.is_some() {
            return;
        }
        let ly = ly_cells;
        let lxn = self.lx as usize;
        for p in self.points.iter_mut() {
            let k1 = sample(u, *p, lxn, ly);
            let p2 = wrap([p[0] + 0.5 * dt * k1[0], p[1] + 0.5 * dt * k1[1]], self.lx, self.ly);
            let k2 = sample(u, p2, lxn, ly);
            let p3 = wrap([p[0] + 0.5 * dt * k2[0], p[1] + 0.5 * dt * k2[1]], self.lx, self.ly);
            let k3 = sample(u, p3, lxn, ly);
            let p4 = wrap([p[0] + dt * k3[0], p[1] + dt * k3[1]], self.lx, self.ly);
            let k4 = sample(u, p4, lxn, ly);
            let vx = (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
            let vy = (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
            *p = wrap([p[0] + dt * vx, p[1] + dt * vy], self.lx, self.ly);
        }
        self.refine(step);
        self.history.push((t, self.length()));
    }

    /// Insert midpoints until no segment is longer than `max_segment`.
    fn refine(&mut self, step: usize) {
        if self.points.len() >= self.max_points {
            self.saturated_at.get_or_insert(step);
            return;
        }
        let mut out: Vec<[f64; 2]> = Vec::with_capacity(self.points.len() * 2);
        let mut hit_cap = false;
        for i in 0..self.points.len() - 1 {
            let a = self.points[i];
            let b = self.points[i + 1];
            out.push(a);
            let d = self.delta(a, b);
            let seg = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if seg > self.max_segment {
                // Enough midpoints to bring every piece under tolerance in one
                // pass; a single midpoint would need repeated sweeps where the
                // flow has stretched a segment by more than a factor of two.
                let n = (seg / self.max_segment).ceil() as usize;
                if out.len() + n >= self.max_points {
                    hit_cap = true;
                } else {
                    for j in 1..n {
                        let s = j as f64 / n as f64;
                        out.push(wrap([a[0] + s * d[0], a[1] + s * d[1]], self.lx, self.ly));
                    }
                }
            }
        }
        out.push(*self.points.last().unwrap());
        if hit_cap || out.len() >= self.max_points {
            self.saturated_at.get_or_insert(step);
        }
        self.points = out;
    }

    /// Least-squares slope of `log(length)` against `t`, over `[t0, t1]`.
    pub fn fit(&self, t0: f64, t1: f64) -> Option<StretchFit> {
        let pts: Vec<(f64, f64)> = self
            .history
            .iter()
            .filter(|&&(t, l)| t >= t0 && t <= t1 && l > 0.0)
            .map(|&(t, l)| (t, l.ln()))
            .collect();
        least_squares(&pts)
    }
}

/// A fitted exponential stretching rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StretchFit {
    /// Slope of `log(length)` against time: the topological entropy `h`, in
    /// reciprocal units of the integration time.
    pub h: f64,
    /// Standard error on the slope.
    pub stderr: f64,
    /// Intercept of the fit.
    pub intercept: f64,
    /// Coefficient of determination.
    pub r2: f64,
    /// Number of samples in the fit.
    pub n: usize,
}

/// Ordinary least squares of `y` on `x`, with the standard error of the slope.
pub fn least_squares(pts: &[(f64, f64)]) -> Option<StretchFit> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / nf;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / nf;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let syy: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let h = sxy / sxx;
    let intercept = my - h * mx;
    let ss_res = syy - h * sxy;
    let r2 = if syy > 0.0 { 1.0 - ss_res / syy } else { 1.0 };
    let stderr = if n > 2 {
        (ss_res.max(0.0) / ((nf - 2.0) * sxx)).sqrt()
    } else {
        f64::NAN
    };
    Some(StretchFit { h, stderr, intercept, r2, n })
}

/// Wrap a point into `[0, lx) x [0, ly)`.
#[inline]
fn wrap(p: [f64; 2], lx: f64, ly: f64) -> [f64; 2] {
    [p[0].rem_euclid(lx), p[1].rem_euclid(ly)]
}

/// Bilinear sample of a two-component field at a real-valued lattice position.
///
/// Cell `(x, y)` carries the value at lattice point `(x, y)`, and neighbours
/// wrap, matching every stencil in [`crate::ops`].
#[inline]
pub fn sample(u: &[f64], p: [f64; 2], lx: usize, ly: usize) -> [f64; 2] {
    let x0 = p[0].floor();
    let y0 = p[1].floor();
    let fx = p[0] - x0;
    let fy = p[1] - y0;
    let i0 = (x0 as i64).rem_euclid(lx as i64) as usize;
    let j0 = (y0 as i64).rem_euclid(ly as i64) as usize;
    let i1 = (i0 + 1) % lx;
    let j1 = (j0 + 1) % ly;

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let mut out = [0.0; 2];
    for (c, o) in out.iter_mut().enumerate() {
        *o = w00 * u[vi(i0, j0, ly, c)]
            + w10 * u[vi(i1, j0, ly, c)]
            + w01 * u[vi(i0, j1, ly, c)]
            + w11 * u[vi(i1, j1, ly, c)];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_field(lx: usize, ly: usize, v: [f64; 2]) -> Vec<f64> {
        let mut u = vec![0.0; lx * ly * 2];
        for i in 0..lx * ly {
            u[i * 2] = v[0];
            u[i * 2 + 1] = v[1];
        }
        u
    }

    #[test]
    fn bilinear_sample_recovers_a_linear_field() {
        let (lx, ly) = (16, 16);
        let mut u = vec![0.0; lx * ly * 2];
        for x in 0..lx {
            for y in 0..ly {
                // Linear in y only, so the periodic seam in x does not enter.
                u[vi(x, y, ly, 0)] = 3.0 * y as f64;
                u[vi(x, y, ly, 1)] = 0.0;
            }
        }
        let s = sample(&u, [4.25, 7.5], lx, ly);
        assert!((s[0] - 22.5).abs() < 1e-12, "{s:?}");
    }

    #[test]
    fn uniform_flow_translates_without_stretching() {
        let (lx, ly) = (32, 32);
        let u = uniform_field(lx, ly, [1.0, 0.0]);
        let mut line = MaterialLine::segment([4.0, 4.0], [12.0, 4.0], lx, ly, 32);
        let l0 = line.length();
        for k in 0..100 {
            line.advect(&u, ly, 0.01, k, k as f64 * 0.01);
        }
        let l1 = line.length();
        assert!((l1 - l0).abs() < 1e-9, "{l0} -> {l1}");
        // Translated by exactly 1.0 in x.
        assert!((line.points[0][0] - 5.0).abs() < 1e-9, "{:?}", line.points[0]);
    }

    /// A pure shear stretches a transverse segment at a rate the fit must
    /// recover. `u = (a y, 0)` maps a vertical segment of length `L` onto one of
    /// length `L sqrt(1 + (a t)^2)`, which is not exponential, so use the
    /// uniform-strain field `u = (a x, -a y)` instead: a segment along `x`
    /// stretches as `exp(a t)` exactly.
    #[test]
    fn uniform_strain_gives_the_exact_exponential_rate() {
        let (lx, ly) = (256, 256);
        let a = 0.5;
        let mut u = vec![0.0; lx * ly * 2];
        for x in 0..lx {
            for y in 0..ly {
                // Centred so the segment stays away from the seam.
                u[vi(x, y, ly, 0)] = a * (x as f64 - 128.0);
                u[vi(x, y, ly, 1)] = -a * (y as f64 - 128.0);
            }
        }
        let mut line = MaterialLine::segment([127.0, 128.0], [129.0, 128.0], lx, ly, 8)
            .with_limits(0.5, 1_000_000);
        let dt = 1e-3;
        for k in 0..2000 {
            line.advect(&u, ly, dt, k, k as f64 * dt);
        }
        let fit = line.fit(0.5, 2.0).expect("fit");
        assert!((fit.h - a).abs() < 1e-3, "h = {} vs {a}", fit.h);
        assert!(fit.r2 > 0.9999, "r2 = {}", fit.r2);
    }

    #[test]
    fn refinement_keeps_segments_under_tolerance() {
        let (lx, ly) = (64, 64);
        let a = 2.0;
        let mut u = vec![0.0; lx * ly * 2];
        for x in 0..lx {
            for y in 0..ly {
                u[vi(x, y, ly, 0)] = a * (x as f64 - 32.0);
                u[vi(x, y, ly, 1)] = -a * (y as f64 - 32.0);
            }
        }
        let mut line = MaterialLine::segment([31.0, 32.0], [33.0, 32.0], lx, ly, 4)
            .with_limits(0.25, 1_000_000);
        for k in 0..500 {
            line.advect(&u, ly, 1e-3, k, k as f64 * 1e-3);
        }
        let worst = line
            .points
            .windows(2)
            .map(|w| {
                let d = line.delta(w[0], w[1]);
                (d[0] * d[0] + d[1] * d[1]).sqrt()
            })
            .fold(0.0_f64, f64::max);
        assert!(worst <= 0.25 * 1.5, "longest segment {worst}");
    }

    #[test]
    fn minimum_image_measures_across_the_seam() {
        let (lx, ly) = (10, 10);
        let line = MaterialLine::segment([0.0, 0.0], [1.0, 0.0], lx, ly, 1);
        let d = line.delta([9.5, 0.0], [0.5, 0.0]);
        assert!((d[0] - 1.0).abs() < 1e-12, "{d:?}");
    }
}
