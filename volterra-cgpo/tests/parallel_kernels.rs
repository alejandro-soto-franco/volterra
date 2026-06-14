//! Parallel-path coverage for the `par_gate`-gated rayon kernels.
//!
//! The kernels in `ops` switch to the rayon row-chunked path once the grid has
//! at least `PAR_THRESHOLD` cells. The other integration tests all use small
//! grids (< threshold), so they only exercise the serial path. These tests run
//! on a 512x512 = 262_144 cell grid (>= 250_000), which engages the parallel
//! path WITHOUT touching the `CGPO_FORCE_PARALLEL` env override (no OnceLock or
//! unsafe set_var ordering hazards).
//!
//! Both oracles are exact on the parallel path: for a linear vector field
//! v = (x, y) the centred-difference divergence is exactly 2, and for a linear
//! scalar field f = x + y the consistent Laplacian stencil is exactly 0. A
//! chunk-offset or row-indexing bug in the parallel path would produce large
//! localized errors that these tight tolerances catch.

use volterra_cgpo::{
    ops::{div_vector, laplacian},
    par_gate::{use_parallel, PAR_THRESHOLD},
    Boundary,
};

// A grid that is guaranteed to engage the parallel path.
const LX: usize = 512;
const LY: usize = 512;

fn rect_boundary(lx: usize, ly: usize) -> Boundary {
    let n = lx * ly;
    Boundary {
        lx,
        ly,
        inside: vec![true; n],
        is_outer: vec![false; n],
        is_inner: vec![false; n],
        outer_normals: vec![[0.0; 2]; n],
        inner_normals: vec![[0.0; 2]; n],
    }
}

#[inline]
fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}
#[inline]
fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}

#[test]
#[allow(clippy::assertions_on_constants)] // intentional compile-time premise guard below
fn parallel_path_is_engaged_at_test_grid() {
    // Guards the premise of this file: 512x512 must take the parallel branch.
    // If PAR_THRESHOLD is ever raised above 262_144, bump LX/LY here too.
    assert!(LX * LY >= PAR_THRESHOLD, "test grid below threshold");
    assert!(
        use_parallel(LX, LY),
        "use_parallel({LX}, {LY}) is false; parallel kernels would not be exercised"
    );
}

#[test]
fn parallel_div_vector_linear_is_two() {
    let bounds = rect_boundary(LX, LY);

    // v[x,y,0] = x,  v[x,y,1] = y  ->  div v = 2 everywhere interior, exactly.
    let mut arr = vec![0.0_f64; LX * LY * 2];
    for x in 0..LX {
        for y in 0..LY {
            arr[vi(x, y, LY, 0)] = x as f64;
            arr[vi(x, y, LY, 1)] = y as f64;
        }
    }

    let mut out = vec![0.0_f64; LX * LY];
    div_vector(&arr, &mut out, &bounds);

    // Interior strip (avoid the periodic-wrap discontinuity of the linear field).
    let margin = 2usize;
    let mut max_err = 0.0_f64;
    for x in margin..(LX - margin) {
        for y in margin..(LY - margin) {
            let err = (out[si(x, y, LY)] - 2.0).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    assert!(
        max_err < 1e-6,
        "parallel div_vector linear field max error = {max_err:.2e} (want < 1e-6)"
    );
}

#[test]
fn parallel_laplacian_linear_is_zero() {
    let bounds = rect_boundary(LX, LY);

    // f = x + y  ->  Laplacian is exactly 0 everywhere interior.
    let mut arr = vec![0.0_f64; LX * LY];
    for x in 0..LX {
        for y in 0..LY {
            arr[si(x, y, LY)] = x as f64 + y as f64;
        }
    }

    let mut out = vec![0.0_f64; LX * LY];
    laplacian(&arr, &mut out, &bounds, 1.0);

    let margin = 2usize;
    let mut max_abs = 0.0_f64;
    for x in margin..(LX - margin) {
        for y in margin..(LY - margin) {
            let a = out[si(x, y, LY)].abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    assert!(
        max_abs < 1e-6,
        "parallel Laplacian of a linear field max |out| = {max_abs:.2e} (want < 1e-6)"
    );
}
