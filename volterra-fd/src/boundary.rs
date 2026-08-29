/// Epitrochoid boundary construction (cardioid, nephroid, trefoiloid).
///
/// Ports the `'epitrochoid'` branch of `set_boundary` from
/// `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`, generalised from that
/// script's fixed 2-cusp nephroid to the whole epitrochoid family of
/// arXiv:2503.10880 Eq. SI.6.
///
/// Index convention: flat index = x * ly + y  (row-major over (x,y)),
/// matching Python's `obj[:,:,i].flatten()` with C-order (x is the outer axis).
use std::f64::consts::PI;
use rayon::prelude::*;

/// Default cusp regularisation. arXiv:2503.10880 SI: "We use d = 0.99 to
/// approximate the epicycloids near their sharp limit."
pub const EPITROCHOID_D: f64 = 0.99;

/// Full boundary description for a nephroid-confined grid.
///
/// - `inside`       : cell is in `sim_points` (interior of the nephroid).
/// - `is_outer`     : cell is inside AND has at least one 4-neighbour outside.
/// - `is_inner`     : cell is inside, not outer, but has an outer 4-neighbour.
/// - `outer_normals`: unit outward normal for outer-layer cells; [0,0] otherwise.
/// - `inner_normals`: unit outward normal for inner-layer cells; [0,0] otherwise.
///
/// Python's `boundary[1, x, y, :]` = outer layer = `outer_normals[x*ly+y]`.
/// Python's `boundary[0, x, y, :]` = inner layer = `inner_normals[x*ly+y]`.
#[derive(Debug, Clone)]
pub struct Boundary {
    pub lx: usize,
    pub ly: usize,
    /// Whether each cell belongs to sim_points (is inside the nephroid).
    pub inside: Vec<bool>,
    /// Outer boundary layer (layer 1 in Python).
    pub is_outer: Vec<bool>,
    /// Inner boundary layer (layer 0 in Python).
    pub is_inner: Vec<bool>,
    /// Per-cell unit normal for the outer layer. [0.0, 0.0] off-boundary.
    pub outer_normals: Vec<[f64; 2]>,
    /// Per-cell unit normal for the inner layer. [0.0, 0.0] off-boundary.
    pub inner_normals: Vec<[f64; 2]>,
}

impl Boundary {
    /// Number of cells in `sim_points` (the confined interior).
    pub fn interior_count(&self) -> usize {
        self.inside.iter().filter(|&&b| b).count()
    }

    /// `sqrt(A_sys)` in lattice units, the length arXiv:2503.10880 divides by
    /// to report a dimensionless active or coherence length (p. 3).
    ///
    /// `A_sys` is the confined area measured as a count of interior lattice
    /// sites, so a dimensionless `ell_tilde` converts to raw pixels as
    /// `ell_tilde * sqrt_area()`.
    pub fn sqrt_area(&self) -> f64 {
        (self.interior_count() as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Epitrochoid geometry
// ---------------------------------------------------------------------------

/// An epitrochoid confinement boundary, arXiv:2503.10880 Eq. SI.6:
///
/// ```text
/// x(u) = r/(2q) [(2q-1) cos(u) + d cos((2q-1) u)]
/// y(u) = r/(2q) [(2q-1) sin(u) + d sin((2q-1) u)]
/// ```
///
/// The curve carries `2(q - 1)` cusps, each of which pins a `-1/2` defect under
/// strong tangential anchoring, so the interior holds net topological charge
/// `q`. `q = 3/2` is the cardioid, `q = 2` the nephroid, `q = 5/2` the
/// trefoiloid. `d` interpolates between the circle (`d = 0`) and the sharp
/// epicycloid (`d = 1`); the paper uses `d = 0.99`, which keeps the curve
/// `C^1`-continuous so the finite-difference normals stay well defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epitrochoid {
    /// Net topological charge the boundary imposes on the interior, a
    /// half-integer at or above `3/2`.
    pub q: f64,
    /// Cusp regularisation. See [`EPITROCHOID_D`].
    pub d: f64,
}

impl Epitrochoid {
    /// A boundary with the paper's regularisation.
    pub fn new(q: f64) -> Self {
        Self { q, d: EPITROCHOID_D }
    }

    /// The cardioid, `q = 3/2`, one cusp.
    pub fn cardioid() -> Self {
        Self::new(1.5)
    }

    /// The nephroid, `q = 2`, two cusps.
    pub fn nephroid() -> Self {
        Self::new(2.0)
    }

    /// The trefoiloid, `q = 5/2`, three cusps.
    pub fn trefoiloid() -> Self {
        Self::new(2.5)
    }

    /// Number of cusps, `2(q - 1)`.
    pub fn cusps(&self) -> f64 {
        2.0 * (self.q - 1.0)
    }

    /// The curve point at parameter `u`, for scale `r`. Eq. SI.6.
    pub fn point(&self, u: f64, r: f64) -> [f64; 2] {
        let m = 2.0 * self.q - 1.0;
        let a = r / (2.0 * self.q);
        [
            a * (m * u.cos() + self.d * (m * u).cos()),
            a * (m * u.sin() + self.d * (m * u).sin()),
        ]
    }

    /// The unit outward normal at parameter `u`, independent of scale.
    ///
    /// The normal winds through `2 pi q` over one circuit, which is what makes
    /// the tangential anchoring built on it impose net charge `q` on the
    /// interior. Its magnitude before normalisation falls to `1 - d` at a cusp,
    /// so `d < 1` is what keeps the direction defined there.
    pub fn normal(&self, u: f64) -> [f64; 2] {
        let m = 2.0 * self.q - 1.0;
        let k = self.cusps();
        let norm = (1.0 + self.d * self.d + 2.0 * self.d * (k * u).cos()).sqrt();
        [
            (u.cos() + self.d * (m * u).cos()) / norm,
            (u.sin() + self.d * (m * u).sin()) / norm,
        ]
    }

    /// Enclosed area for scale `r`, in the same units as `r` squared.
    ///
    /// From Green's theorem on Eq. SI.6:
    /// `A = pi m (m + d^2) (r / 2q)^2` with `m = 2q - 1`. At `d = 0` this
    /// reduces to the area of the circle of radius `m r / 2q`, as it must.
    pub fn area(&self, r: f64) -> f64 {
        let m = 2.0 * self.q - 1.0;
        let a = r / (2.0 * self.q);
        std::f64::consts::PI * m * (m + self.d * self.d) * a * a
    }
}

impl Default for Epitrochoid {
    fn default() -> Self {
        Self::nephroid()
    }
}

// ---------------------------------------------------------------------------
// u-solver: find u ∈ (-π, π] such that
//   atan2((2q-1)sin(u)+d*sin((2q-1)u), (2q-1)cos(u)+d*cos((2q-1)u)) == theta
//
// The polar angle phi(u) of the curve is strictly increasing in u whenever
// d < 2q-1, so the root is unique on (-π, π]. A coarse scan locates it and
// Newton refines; `solve_u` checks the residual and falls back to bisection on
// the bracketing scan interval when Newton stalls, which it can near a cusp,
// where phi'(u) drops to O(1 - d).
// ---------------------------------------------------------------------------

/// Evaluate the epitrochoid's polar angle for parameter u.
#[inline]
fn epi_angle(u: f64, epi: &Epitrochoid) -> f64 {
    let m = 2.0 * epi.q - 1.0;
    f64::atan2(
        m * u.sin() + epi.d * (m * u).sin(),
        m * u.cos() + epi.d * (m * u).cos(),
    )
}

/// Wrap an angle difference into (-π, π].
#[inline]
fn wrap(a: f64) -> f64 {
    let mut v = a % (2.0 * PI);
    if v > PI {
        v -= 2.0 * PI;
    } else if v <= -PI {
        v += 2.0 * PI;
    }
    v
}

/// Solve for u ∈ (-π, π] such that epi_angle(u) ≈ theta.
///
/// Strategy:
///   1. Coarse scan of N_SCAN equally-spaced u values; pick best candidate.
///   2. Newton refinement (up to MAX_NEWTON steps) from that candidate.
///   3. Bisection on the scan interval bracketing the root, if Newton left a
///      residual above `TOL_ACCEPT`.
///
/// Steps 1 and 2 match scipy fsolve(f, 0.1) in the Python code. Step 3 is
/// this port's addition: `phi` is strictly increasing, so the sign of the
/// wrapped residual brackets the root between adjacent scan points, and
/// bisection converges there unconditionally. Newton alone is enough for the
/// nephroid, and stalls for a fraction of the parameter circle at `q = 3/2`,
/// where `phi'` falls to 0.02 next to the single cusp.
fn solve_u(theta: f64, epi: &Epitrochoid) -> f64 {
    const N_SCAN: usize = 2000;
    const MAX_NEWTON: usize = 30;
    const TOL: f64 = 1e-12;
    /// Residual above which Newton's answer is rejected for bisection's.
    const TOL_ACCEPT: f64 = 1e-9;

    // coarse scan
    let mut best_i = 0usize;
    let mut best_err = f64::INFINITY;
    let scan_u = |i: usize| -PI + (2.0 * PI) * (i as f64) / (N_SCAN as f64);
    for i in 0..N_SCAN {
        let err = wrap(epi_angle(scan_u(i), epi) - theta).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        }
    }
    let best_u = scan_u(best_i);

    // Newton refinement
    // f(u)  = epi_angle(u) - theta  (wrapped)
    // f'(u) ≈ (f(u+h) - f(u-h)) / (2h), the numerical derivative
    let h = 1e-7_f64;
    let mut u = best_u;
    for _ in 0..MAX_NEWTON {
        let fu = wrap(epi_angle(u, epi) - theta);
        if fu.abs() < TOL {
            break;
        }
        let fp = (wrap(epi_angle(u + h, epi) - theta) - wrap(epi_angle(u - h, epi) - theta))
            / (2.0 * h);
        if fp.abs() < 1e-15 {
            break;
        }
        u -= fu / fp;
        // Keep u in (-2π, 2π) to avoid drift.
        u = u.rem_euclid(2.0 * PI);
        if u > PI {
            u -= 2.0 * PI;
        }
    }
    if wrap(epi_angle(u, epi) - theta).abs() <= TOL_ACCEPT {
        return u;
    }

    // Bisection fallback. phi is increasing, so the root lies in whichever of
    // the two intervals adjoining the best scan point has a sign change.
    // The bracket is taken in unwrapped u, not modulo the scan index, so that
    // a root next to u = ±π still gets an interval with lo < hi. epi_angle is
    // 2π-periodic in u, so evaluating outside (-π, π] is well defined.
    let residual = |u: f64| wrap(epi_angle(u, epi) - theta);
    let step = 2.0 * PI / N_SCAN as f64;
    let mut lo = best_u - step;
    let mut hi = best_u + step;
    if residual(lo) * residual(best_u) <= 0.0 {
        hi = best_u;
    } else if residual(best_u) * residual(hi) <= 0.0 {
        lo = best_u;
    } else {
        // No sign change adjoining the scan minimum: keep Newton's answer.
        return u;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if residual(lo) * residual(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Compute the unit outward normal for a boundary cell at grid position (x, y).
///
/// Matches Python, with `k = 2(q-1)` cusps and `k + 1 = 2q - 1`:
///   norm = sqrt(1 + d^2 + 2*d*cos(k*u))
///   nx = (cos(u) + d*cos((k+1)*u)) / norm
///   ny = (sin(u) + d*sin((k+1)*u)) / norm
#[inline]
fn boundary_normal(x: usize, y: usize, radius: usize, epi: &Epitrochoid) -> [f64; 2] {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    let theta = dy.atan2(dx);
    epi.normal(solve_u(theta, epi))
}

/// Test whether grid cell (x, y) is inside the epitrochoid.
///
/// Matches Python:
///   (x-r)^2 + (y-r)^2 <= r^2/(2q)^2 * ((2q-1)^2 + d^2 + 2*(2q-1)*d*cos(k*u))
fn is_inside(x: usize, y: usize, radius: usize, epi: &Epitrochoid) -> bool {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    let lhs = dx * dx + dy * dy;
    let theta = dy.atan2(dx);
    let u = solve_u(theta, epi);
    let k = epi.cusps();
    let m = 2.0 * epi.q - 1.0;
    let d = epi.d;
    let two_q = 2.0 * epi.q;
    let rhs = (r * r) / (two_q * two_q) * (m * m + d * d + 2.0 * m * d * (k * u).cos());
    lhs <= rhs
}

/// Build the nephroid (`q = 2`) boundary for an `lx × ly` grid.
pub fn nephroid_boundary(lx: usize, ly: usize) -> Boundary {
    epitrochoid_boundary(lx, ly, Epitrochoid::nephroid())
}

/// Build the cardioid (`q = 3/2`) boundary for an `lx × ly` grid.
pub fn cardioid_boundary(lx: usize, ly: usize) -> Boundary {
    epitrochoid_boundary(lx, ly, Epitrochoid::cardioid())
}

/// Build the trefoiloid (`q = 5/2`) boundary for an `lx × ly` grid.
pub fn trefoiloid_boundary(lx: usize, ly: usize) -> Boundary {
    epitrochoid_boundary(lx, ly, Epitrochoid::trefoiloid())
}

/// Build an epitrochoid boundary for an `lx × ly` grid.
///
/// `radius = lx / 2 - 1` (integer division, as in Python).
///
/// The interior test and the normals each solve for the curve parameter `u`
/// from a cell's polar angle, at a couple of thousand trigonometric
/// evaluations per cell, so both passes go through rayon regardless of grid
/// size. [`crate::par_gate`] governs the per-step kernels, where the tradeoff
/// against spawn overhead is a real one; it does not apply to a construction
/// this heavy that runs once.
pub fn epitrochoid_boundary(lx: usize, ly: usize, epi: Epitrochoid) -> Boundary {
    let n = lx * ly;
    let radius = lx / 2 - 1;

    // Pass 1: determine sim_points (inside)
    let mut inside = vec![false; n];
    inside
        .par_chunks_mut(ly)
        .enumerate()
        .for_each(|(x, row)| {
            for (y, cell) in row.iter_mut().enumerate() {
                *cell = is_inside(x, y, radius, &epi);
            }
        });

    // Pass 2: outer boundary (inside cells with a non-inside 4-neighbour)
    let mut is_outer = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if !inside[idx] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let neighbours = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)];
            let has_outside_neighbour = neighbours.iter().any(|&(nx, ny)| {
                if nx < 0 || ny < 0 || nx >= lx as i64 || ny >= ly as i64 {
                    return true; // out of grid → outside
                }
                !inside[nx as usize * ly + ny as usize]
            });
            if has_outside_neighbour {
                is_outer[idx] = true;
            }
        }
    }

    // Pass 3: inner boundary (inside, not outer, with an outer 4-neighbour)
    let mut is_inner = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if !inside[idx] || is_outer[idx] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let neighbours = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)];
            let has_outer_neighbour = neighbours.iter().any(|&(nx, ny)| {
                if nx < 0 || ny < 0 || nx >= lx as i64 || ny >= ly as i64 {
                    return false;
                }
                is_outer[nx as usize * ly + ny as usize]
            });
            if has_outer_neighbour {
                is_inner[idx] = true;
            }
        }
    }

    // Pass 4: compute normals for boundary cells
    let zero = [0.0_f64; 2];
    let mut outer_normals = vec![zero; n];
    let mut inner_normals = vec![zero; n];

    outer_normals
        .par_chunks_mut(ly)
        .zip(inner_normals.par_chunks_mut(ly))
        .enumerate()
        .for_each(|(x, (outer_row, inner_row))| {
            for y in 0..ly {
                let idx = x * ly + y;
                if is_outer[idx] || is_inner[idx] {
                    let normal = boundary_normal(x, y, radius, &epi);
                    if is_outer[idx] {
                        outer_row[y] = normal;
                    }
                    if is_inner[idx] {
                        inner_row[y] = normal;
                    }
                }
            }
        });

    Boundary {
        lx,
        ly,
        inside,
        is_outer,
        is_inner,
        outer_normals,
        inner_normals,
    }
}

// ---------------------------------------------------------------------------
// Circular ("steady-winding circle") boundary
// ---------------------------------------------------------------------------

/// Radial unit normal at (x, y) for a disk centred at (radius, radius),
/// rounded to 4 decimal places to match the Python `round(..., 4)` calls.
///
/// Matches `set_boundary`'s `'circular'` branch in flow-solver.py:
///   boundary[l, x, y, 0] = round((x - radius) / dist, 4)
///   boundary[l, x, y, 1] = round((y - radius) / dist, 4)
fn circular_normal(x: usize, y: usize, radius: usize) -> [f64; 2] {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    let dist = (dx * dx + dy * dy).sqrt();
    [
        (dx / dist * 10000.0).round() / 10000.0,
        (dy / dist * 10000.0).round() / 10000.0,
    ]
}

/// Test whether grid cell (x, y) lies inside the disk of the given radius,
/// centred at (radius, radius).
///
/// Matches Python: `(x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2`.
fn circular_is_inside(x: usize, y: usize, radius: usize) -> bool {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    dx * dx + dy * dy <= r * r
}

/// Build the "steady-winding circle" boundary for an `lx x ly` grid: a plain
/// disk of radius `lx / 2 - 1` (integer division), centred at
/// `(radius, radius)`.
///
/// Faithfully ports the `'circular'` branch of `set_boundary` in
/// `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`. This is the boundary
/// used for the paper's circular-confinement results (Klein et al.,
/// arXiv:2503.10880, Eq. 1 and Figs. 2-4): a smooth disk with a tangential
/// anchoring direction that winds through angle `2*pi*q` around the
/// boundary, `q` set separately in [`crate::bc::apply_q_boundary_conditions`]
/// via its `net_charge` argument.
pub fn circular_boundary(lx: usize, ly: usize) -> Boundary {
    let n = lx * ly;
    let radius = lx / 2 - 1;

    let mut inside = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            if circular_is_inside(x, y, radius) {
                inside[x * ly + y] = true;
            }
        }
    }

    let mut is_outer = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if !inside[idx] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let neighbours = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)];
            let has_outside_neighbour = neighbours.iter().any(|&(nx, ny)| {
                if nx < 0 || ny < 0 || nx >= lx as i64 || ny >= ly as i64 {
                    return true;
                }
                !inside[nx as usize * ly + ny as usize]
            });
            if has_outside_neighbour {
                is_outer[idx] = true;
            }
        }
    }

    let mut is_inner = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if !inside[idx] || is_outer[idx] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let neighbours = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)];
            let has_outer_neighbour = neighbours.iter().any(|&(nx, ny)| {
                if nx < 0 || ny < 0 || nx >= lx as i64 || ny >= ly as i64 {
                    return false;
                }
                is_outer[nx as usize * ly + ny as usize]
            });
            if has_outer_neighbour {
                is_inner[idx] = true;
            }
        }
    }

    let zero = [0.0_f64; 2];
    let mut outer_normals = vec![zero; n];
    let mut inner_normals = vec![zero; n];

    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if is_outer[idx] {
                outer_normals[idx] = circular_normal(x, y, radius);
            }
            if is_inner[idx] {
                inner_normals[idx] = circular_normal(x, y, radius);
            }
        }
    }

    Boundary {
        lx,
        ly,
        inside,
        is_outer,
        is_inner,
        outer_normals,
        inner_normals,
    }
}

/// Build a fully periodic `lx x ly` domain: a flat torus with no wall.
///
/// Every cell is interior and neither boundary ring has a member, so the four
/// boundary-condition passes in [`crate::step::update_step_inner`] each visit
/// nothing and the domain closes on itself through the modular neighbour
/// indexing every stencil in [`crate::ops`] already uses. Nothing else in the
/// solver changes.
///
/// This is the domain of Mitchell, Sabbir, Geumhan, Smith, Klein and Beller,
/// "Maximally mixing active nematics", Phys. Rev. E 109, 014606 (2024), whose
/// result is that a square with periodic boundaries, confined tightly enough,
/// settles into a periodic four-defect orbit; and of Mitchell, Sabbir, Klein
/// and Beller, "Modelling active nematics via the nematic locking principle",
/// Soft Matter (2025), arXiv:2506.20996, whose simulations run on a 200x200
/// periodic domain.
///
/// # Pressure gauge
///
/// The pressure Poisson problem on a torus has the constant functions in its
/// null space, so `p` is fixed only up to an additive constant and the
/// relative-change convergence test in [`crate::stokes::relax_pressure`]
/// divides by a sum that is near zero once the mean is removed. Drive a
/// periodic run with a finite `max_p_iters` and subtract the mean with
/// [`crate::stokes::subtract_p_avg`]; only `grad p` enters the velocity
/// update, so the gauge is free.
pub fn periodic_boundary(lx: usize, ly: usize) -> Boundary {
    let n = lx * ly;
    Boundary {
        lx,
        ly,
        inside: vec![true; n],
        is_outer: vec![false; n],
        is_inner: vec![false; n],
        outer_normals: vec![[0.0, 0.0]; n],
        inner_normals: vec![[0.0, 0.0]; n],
    }
}
