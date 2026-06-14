/// Nephroid (epitrochoid k=2) boundary construction.
///
/// Faithfully ports the `'epitrochoid'` branch of `set_boundary` from
/// `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`, with the task-specified
/// parameters d=0.99, k=2 (2-cusp nephroid).
///
/// Index convention: flat index = x * ly + y  (row-major over (x,y)),
/// matching Python's `obj[:,:,i].flatten()` with C-order (x is the outer axis).
use std::f64::consts::PI;

const D: f64 = 0.99;
const K: f64 = 2.0; // number-of-cusps variable in the epitrochoid formula

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
    /// Number of cells in `sim_points` (interior of the nephroid).
    pub fn interior_count(&self) -> usize {
        self.inside.iter().filter(|&&b| b).count()
    }

}

// ---------------------------------------------------------------------------
// u-solver: find u ∈ (-π, π] such that
//   atan2((k+1)sin(u)+d*sin((k+1)u), (k+1)cos(u)+d*cos((k+1)u)) == theta
//
// The epitrochoid tangent angle phi(u) is monotone enough over (-π,π] that a
// coarse scan + Newton refinement is robust at all grid positions.
// ---------------------------------------------------------------------------

/// Evaluate the epitrochoid angle for parameter u.
#[inline]
fn epi_angle(u: f64) -> f64 {
    let kp1 = K + 1.0;
    f64::atan2(
        kp1 * u.sin() + D * (kp1 * u).sin(),
        kp1 * u.cos() + D * (kp1 * u).cos(),
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
///
/// This matches scipy fsolve(f, 0.1) in the Python code — the coarse scan
/// handles the non-trivial structure near the two cusps of the nephroid (k=2).
fn solve_u(theta: f64) -> f64 {
    const N_SCAN: usize = 2000;
    const MAX_NEWTON: usize = 30;
    const TOL: f64 = 1e-12;

    // --- coarse scan ---
    let mut best_u = 0.0_f64;
    let mut best_err = f64::INFINITY;
    for i in 0..N_SCAN {
        let u = -PI + (2.0 * PI) * (i as f64) / (N_SCAN as f64);
        let err = wrap(epi_angle(u) - theta).abs();
        if err < best_err {
            best_err = err;
            best_u = u;
        }
    }

    // --- Newton refinement ---
    // f(u)  = epi_angle(u) - theta  (wrapped)
    // f'(u) ≈ (f(u+h) - f(u-h)) / (2h)  — numerical derivative
    let h = 1e-7_f64;
    let mut u = best_u;
    for _ in 0..MAX_NEWTON {
        let fu = wrap(epi_angle(u) - theta);
        if fu.abs() < TOL {
            break;
        }
        let fp = (wrap(epi_angle(u + h) - theta) - wrap(epi_angle(u - h) - theta)) / (2.0 * h);
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
    u
}

/// Compute the unit outward normal for a boundary cell at grid position (x, y).
///
/// Matches Python:
///   norm = sqrt(1 + d^2 + 2*d*cos(k*u))
///   nx = (cos(u) + d*cos((k+1)*u)) / norm
///   ny = (sin(u) + d*sin((k+1)*u)) / norm
#[inline]
fn boundary_normal(x: usize, y: usize, radius: usize) -> [f64; 2] {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    let theta = dy.atan2(dx);
    let u = solve_u(theta);
    let kp1 = K + 1.0;
    let norm = (1.0 + D * D + 2.0 * D * (K * u).cos()).sqrt();
    let nx = (u.cos() + D * (kp1 * u).cos()) / norm;
    let ny = (u.sin() + D * (kp1 * u).sin()) / norm;
    [nx, ny]
}

/// Test whether grid cell (x, y) is inside the nephroid.
///
/// Matches Python:
///   (x-r)^2 + (y-r)^2 <= r^2/(k+2)^2 * ((k+1)^2 + d^2 + 2*(k+1)*d*cos(k*u))
fn is_inside(x: usize, y: usize, radius: usize) -> bool {
    let r = radius as f64;
    let dx = x as f64 - r;
    let dy = y as f64 - r;
    let lhs = dx * dx + dy * dy;
    let theta = dy.atan2(dx);
    let u = solve_u(theta);
    let kp1 = K + 1.0;
    let rhs = (r * r) / ((K + 2.0) * (K + 2.0))
        * (kp1 * kp1 + D * D + 2.0 * kp1 * D * (K * u).cos());
    lhs <= rhs
}

/// Build the nephroid boundary for an `lx × ly` grid.
///
/// Parameters match the Python epitrochoid branch: d=0.99, k=2.
/// `radius = lx / 2 - 1` (integer division, as in Python).
pub fn nephroid_boundary(lx: usize, ly: usize) -> Boundary {
    let n = lx * ly;
    let radius = lx / 2 - 1;

    // --- Pass 1: determine sim_points (inside) ---
    let mut inside = vec![false; n];
    for x in 0..lx {
        for y in 0..ly {
            if is_inside(x, y, radius) {
                inside[x * ly + y] = true;
            }
        }
    }

    // --- Pass 2: outer boundary (inside cells with a non-inside 4-neighbour) ---
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

    // --- Pass 3: inner boundary (inside, not outer, with an outer 4-neighbour) ---
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

    // --- Pass 4: compute normals for boundary cells ---
    let zero = [0.0_f64; 2];
    let mut outer_normals = vec![zero; n];
    let mut inner_normals = vec![zero; n];

    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            if is_outer[idx] {
                outer_normals[idx] = boundary_normal(x, y, radius);
            }
            if is_inner[idx] {
                inner_normals[idx] = boundary_normal(x, y, radius);
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
