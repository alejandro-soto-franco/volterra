//! Finite-difference operators for the CGPO solver.
//!
//! # Field layout conventions
//!
//! **Scalar fields** (`&[f64]`, length `lx*ly`):
//!   flat index = `x * ly + y`  (row-major, x outer axis).
//!
//! **2-component vector/tensor fields** (`&[f64]`, length `lx*ly*2`):
//!   flat index = `(x * ly + y) * 2 + c`  where `c ∈ {0,1}`.
//!   This matches NumPy's `arr[x, y, c]` with C-order (last axis contiguous).
//!
//! # Stencil notes
//!
//! All operators faithfully replicate the Python/Numba stencils from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py` lines 783–898.
//!
//! Grid spacing `dx = 1` throughout (the Python solver uses unit spacing).
//!
//! **Index wrapping**: following the Python exactly, all neighbour indices are
//! computed modulo the grid dimensions — `xdn = (x + lx - 1) % lx`, etc.
//! Python's bare `x - 1` at `x = 0` yields `-1`, which NumPy/Python wraps to
//! index `lx - 1`; our modular arithmetic replicates that exactly.
//!
//! **Operators only write to cells where `bounds.inside[idx]` is true** — the
//! same set the Python calls `sim_points` / `bounds`.

use crate::Boundary;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Flat scalar index.
#[inline(always)]
fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}

/// Flat vector index for component `c ∈ {0,1}`.
#[inline(always)]
fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}

// ---------------------------------------------------------------------------
// Public operators
// ---------------------------------------------------------------------------

/// 9-point isotropic Laplacian of a **scalar** field.
///
/// Replicates `Laplacian(arr, out, bounds, coeff=1.)` from flow-solver.py.
///
/// Stencil (dx=1):
/// ```text
/// out[x,y] = (coeff/6) * (
///     -20*arr[x,y]
///     + 4*(arr[x+1,y] + arr[x-1,y] + arr[x,y+1] + arr[x,y-1])
///     +    arr[x+1,y+1] + arr[x+1,y-1] + arr[x-1,y+1] + arr[x-1,y-1]
/// )
/// ```
///
/// Updates every cell where `bounds.inside[idx]` is true.
pub fn laplacian(arr: &[f64], out: &mut [f64], bounds: &Boundary, coeff: f64) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let c = coeff / 6.0;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            out[idx] = c * (
                -20.0 * arr[si(x, y, ly)]
                + 4.0 * (
                    arr[si(xup, y, ly)]
                    + arr[si(xdn, y, ly)]
                    + arr[si(x, yup, ly)]
                    + arr[si(x, ydn, ly)]
                )
                + arr[si(xup, yup, ly)]
                + arr[si(xup, ydn, ly)]
                + arr[si(xdn, yup, ly)]
                + arr[si(xdn, ydn, ly)]
            );
        }
    }
}

/// 9-point isotropic Laplacian of a **2-component vector** field.
///
/// Replicates `Laplacian_vector(arr, out, bounds, coeff=1.)`.
/// Applied component-wise with the same stencil as `laplacian`.
pub fn laplacian_vector(arr: &[f64], out: &mut [f64], bounds: &Boundary, coeff: f64) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let c = coeff / 6.0;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            for comp in 0..2 {
                out[vi(x, y, ly, comp)] = c * (
                    -20.0 * arr[vi(x, y, ly, comp)]
                    + 4.0 * (
                        arr[vi(xup, y, ly, comp)]
                        + arr[vi(xdn, y, ly, comp)]
                        + arr[vi(x, yup, ly, comp)]
                        + arr[vi(x, ydn, ly, comp)]
                    )
                    + arr[vi(xup, yup, ly, comp)]
                    + arr[vi(xup, ydn, ly, comp)]
                    + arr[vi(xdn, yup, ly, comp)]
                    + arr[vi(xdn, ydn, ly, comp)]
                );
            }
        }
    }
}

/// Divergence of a 2-component vector field.
///
/// Replicates `div_vector(arr, out, bounds)`.
///
/// Stencil (dx=1):
/// ```text
/// out[x,y] = 0.5 * (
///     (arr[x+1,y,0] - arr[x-1,y,0])   // ∂x vx
///   + (arr[x,y+1,1] - arr[x,y-1,1])   // ∂y vy
/// )
/// ```
pub fn div_vector(arr: &[f64], out: &mut [f64], bounds: &Boundary) {
    let lx = bounds.lx;
    let ly = bounds.ly;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }
            let xup = (x + 1) % lx;
            let xdn = (x + lx - 1) % lx;
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;

            out[idx] = 0.5 * (
                (arr[vi(xup, y, ly, 0)] - arr[vi(xdn, y, ly, 0)])
                + (arr[vi(x, yup, ly, 1)] - arr[vi(x, ydn, ly, 1)])
            );
        }
    }
}

/// Second-order upwind advective term: adds `-(u·∇)[arr]` to `out`.
///
/// Replicates `upwind_advective_term(u, arr, out, bounds, coeff=-1)`.
///
/// **Accumulates** into `out` (zero `out` before calling if a fresh result is
/// needed — matching the Python caller pattern).
///
/// For `ux > 0` (upwind from the left):
/// ```text
/// out[x,y,:] += (coeff/2)*ux * (3*arr[x,y,:] - 4*arr[x-1,y,:] + arr[x-2,y,:])
/// ```
/// For `ux <= 0` (upwind from the right):
/// ```text
/// out[x,y,:] += (coeff/2)*ux * (-3*arr[x,y,:] + 4*arr[x+1,y,:] - arr[x+2,y,:])
/// ```
/// (analogously for the y-component).
///
/// All indices wrap modulo (lx, ly) following the Python convention.
pub fn upwind_advective_term(
    u: &[f64],
    arr: &[f64],
    out: &mut [f64],
    bounds: &Boundary,
    coeff: f64,
) {
    let lx = bounds.lx;
    let ly = bounds.ly;
    let half_coeff = coeff * 0.5;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if !bounds.inside[idx] {
                continue;
            }

            // --- x-direction ---
            let xup   = (x + 1) % lx;
            let xdn   = (x + lx - 1) % lx;
            let xupup = (x + 2) % lx;
            let xdndn = (x + lx - 2) % lx;

            let tmp_x = half_coeff * u[vi(x, y, ly, 0)];
            if u[vi(x, y, ly, 0)] > 0.0 {
                for c in 0..2 {
                    out[vi(x, y, ly, c)] += tmp_x * (
                        3.0 * arr[vi(x,     y, ly, c)]
                        - 4.0 * arr[vi(xdn,   y, ly, c)]
                        +       arr[vi(xdndn, y, ly, c)]
                    );
                }
            } else {
                for c in 0..2 {
                    out[vi(x, y, ly, c)] += tmp_x * (
                        -3.0 * arr[vi(x,     y, ly, c)]
                        + 4.0 * arr[vi(xup,   y, ly, c)]
                        -       arr[vi(xupup, y, ly, c)]
                    );
                }
            }

            // --- y-direction ---
            let yup   = (y + 1) % ly;
            let ydn   = (y + ly - 1) % ly;
            let yupup = (y + 2) % ly;
            let ydndn = (y + ly - 2) % ly;

            let tmp_y = half_coeff * u[vi(x, y, ly, 1)];
            if u[vi(x, y, ly, 1)] > 0.0 {
                for c in 0..2 {
                    out[vi(x, y, ly, c)] += tmp_y * (
                        3.0 * arr[vi(x, y,     ly, c)]
                        - 4.0 * arr[vi(x, ydn,   ly, c)]
                        +       arr[vi(x, ydndn, ly, c)]
                    );
                }
            } else {
                for c in 0..2 {
                    out[vi(x, y, ly, c)] += tmp_y * (
                        -3.0 * arr[vi(x, y,     ly, c)]
                        + 4.0 * arr[vi(x, yup,   ly, c)]
                        -       arr[vi(x, yupup, ly, c)]
                    );
                }
            }
        }
    }
}
