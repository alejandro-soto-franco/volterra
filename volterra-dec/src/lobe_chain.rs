//! A linear chain of lobes, as the confining wall of a chip micromixer.
//!
//! # The wall
//!
//! One lobe of the upper wall, over `s` in `[0, 2 pi)`:
//!
//! ```text
//! x = a (s - sin s)
//! y = y_tip + A (1 - cos s),      A = (h - y_tip) / 2
//! ```
//!
//! Both derivatives vanish at `s = 0`, where `dx/ds` goes as `s^2` and `dy/ds`
//! as `s`, so the tangent approaches the vertical from one side and leaves it
//! on the other. That is a true cusp, and it is one for every `h > y_tip`.
//!
//! The lower wall is the mirror image, so each junction between lobes has two
//! cusps facing each other across the channel. `y_tip` is the only thing
//! setting how far apart their tips are.
//!
//! # Why the cusp and the neck are separate numbers
//!
//! The device varies the neck and needs the forced defect count to stay put
//! while it does. A cycloid, where the cusp exists only at one value of its
//! single shape parameter, cannot do that: opening the neck smooths the cusps
//! away and the imposed charge falls to zero. Splitting the tip depth from the
//! chamber height is what keeps the cusp through the whole range.
//!
//! # Charge
//!
//! `n` lobes give `n - 1` interior cusps on each wall, so `k = 2(n - 1)`, and a
//! closed curve with `k` cusps imposes `1 + k/2`, which is `n`. The count is
//! independent of the neck by construction, and `tests/lobe_chain.rs` measures
//! that rather than assuming it.
//!
//! The four junctions where the train meets the two end walls are C1: the
//! train's tangent is vertical at each end of its span and the end walls are
//! vertical, so they contribute no cusp and no corner. That matters. A corner
//! of exactly a right angle puts `2 (theta_{i+1} - theta_i)` at `pi`, the
//! branch point of the wrap in `imposed_charge`, and a variant of this curve
//! with a straight lower wall reads exactly one less than its cusp count for
//! that reason.

use std::f64::consts::TAU;

use crate::curve::PlaneCurve;

/// Fraction of the period spent on each of the two lobe trains.
const F_WALL: f64 = 0.45;
/// Fraction of the period spent on each of the two end walls.
const F_END: f64 = 0.05;

/// A chain of `n` lobes in a line, closed by an end wall at each end.
///
/// The end walls span the neck gap, so a port cut into one of them is the
/// width of a neck.
#[derive(Debug, Clone, Copy)]
pub struct LobeChain {
    /// Number of lobes.
    pub n: usize,
    /// Half-height of a lobe at its widest.
    pub h: f64,
    /// Height of a cusp tip above the axis, so the gap between opposing tips
    /// is `2 y_tip`. Must be below `h` for the cusps to exist.
    pub y_tip: f64,
    /// Length scale of one lobe along the channel; a lobe spans `2 pi a`.
    pub a: f64,
}

impl LobeChain {
    /// A chain of `n` lobes whose neck gap is `neck` times the half-height.
    ///
    /// `neck` at zero pinches the chambers apart at a point, and approaching
    /// one flattens the lobes away. The cusps are present throughout the open
    /// interval between.
    pub fn new(n: usize, neck: f64) -> Self {
        let h = 1.0;
        Self {
            n: n.max(1),
            h,
            y_tip: neck.clamp(0.0, 1.0) * h,
            a: 0.55,
        }
    }

    /// Gap between opposing cusp tips.
    pub fn neck_width(&self) -> f64 {
        2.0 * self.y_tip
    }

    /// Imposed charge the wall's cusp count predicts, `1 + k/2` for
    /// `k = 2(n - 1)`.
    pub fn predicted_charge(&self) -> f64 {
        self.n as f64
    }

    /// Interior cusps on each wall.
    pub fn cusps_per_wall(&self) -> usize {
        self.n - 1
    }

    fn amp(&self) -> f64 {
        (self.h - self.y_tip) / 2.0
    }

    fn s_span(&self) -> f64 {
        TAU * self.n as f64
    }

    /// Horizontal extent of the chain.
    pub fn length(&self) -> f64 {
        self.a * self.s_span()
    }

    fn wall(&self, s: f64, upper: bool) -> [f64; 2] {
        let y = self.y_tip + self.amp() * (1.0 - s.cos());
        [self.a * (s - s.sin()), if upper { y } else { -y }]
    }

    fn dwall(&self, s: f64, upper: bool) -> [f64; 2] {
        let dx = self.a * (1.0 - s.cos());
        let dy = self.amp() * s.sin();
        [dx, if upper { dy } else { -dy }]
    }

    fn ddwall(&self, s: f64, upper: bool) -> [f64; 2] {
        let dx = self.a * s.sin();
        let dy = self.amp() * s.cos();
        [dx, if upper { dy } else { -dy }]
    }
}

/// Where a curve parameter falls in the circuit.
enum Leg {
    /// Lower wall, left to right, at train parameter `s`.
    Lower(f64),
    /// Right end wall, rising, at fraction `f` of its length.
    Right(f64),
    /// Upper wall, right to left, at train parameter `s`.
    Upper(f64),
    /// Left end wall, falling, at fraction `f` of its length.
    Left(f64),
}

impl LobeChain {
    fn leg(&self, u: f64) -> Leg {
        let t = u.rem_euclid(TAU) / TAU;
        let span = self.s_span();
        if t < F_WALL {
            Leg::Lower(span * t / F_WALL)
        } else if t < F_WALL + F_END {
            Leg::Right((t - F_WALL) / F_END)
        } else if t < 2.0 * F_WALL + F_END {
            Leg::Upper(span * (1.0 - (t - F_WALL - F_END) / F_WALL))
        } else {
            Leg::Left((t - 2.0 * F_WALL - F_END) / F_END)
        }
    }

    /// Chain rule factor from the curve parameter to the train parameter.
    fn wall_rate(&self) -> f64 {
        self.s_span() / (TAU * F_WALL)
    }

    fn end_rate(&self) -> f64 {
        2.0 * self.y_tip / (TAU * F_END)
    }
}

impl PlaneCurve for LobeChain {
    fn point(&self, u: f64) -> [f64; 2] {
        match self.leg(u) {
            Leg::Lower(s) => self.wall(s, false),
            Leg::Right(f) => [self.length(), -self.y_tip + f * 2.0 * self.y_tip],
            Leg::Upper(s) => self.wall(s, true),
            Leg::Left(f) => [0.0, self.y_tip - f * 2.0 * self.y_tip],
        }
    }

    fn d1(&self, u: f64) -> [f64; 2] {
        let k = self.wall_rate();
        match self.leg(u) {
            Leg::Lower(s) => {
                let d = self.dwall(s, false);
                [d[0] * k, d[1] * k]
            }
            Leg::Right(_) => [0.0, self.end_rate()],
            Leg::Upper(s) => {
                let d = self.dwall(s, true);
                [-d[0] * k, -d[1] * k]
            }
            Leg::Left(_) => [0.0, -self.end_rate()],
        }
    }

    fn d2(&self, u: f64) -> [f64; 2] {
        let k = self.wall_rate();
        match self.leg(u) {
            Leg::Lower(s) => {
                let d = self.ddwall(s, false);
                [d[0] * k * k, d[1] * k * k]
            }
            Leg::Upper(s) => {
                let d = self.ddwall(s, true);
                [d[0] * k * k, d[1] * k * k]
            }
            Leg::Right(_) | Leg::Left(_) => [0.0, 0.0],
        }
    }

    fn features(&self) -> Vec<f64> {
        let per = TAU * F_WALL / self.n as f64;
        let mut f = Vec::with_capacity(2 * self.n + 2);
        for m in 1..self.n {
            f.push(m as f64 * per);
        }
        f.push(TAU * F_WALL);
        f.push(TAU * (F_WALL + F_END));
        for m in 1..self.n {
            f.push(TAU * (F_WALL + F_END) + m as f64 * per);
        }
        f.push(TAU * (2.0 * F_WALL + F_END));
        f.sort_by(|p, q| p.partial_cmp(q).expect("feature parameters are finite"));
        f
    }

    fn interior(&self) -> [f64; 2] {
        [self.length() / 2.0, 0.0]
    }
}
