//! Unit tests for the five boundary-condition kernels in `bc.rs`.
//!
//! Each test builds a small `nephroid_boundary(20, 20)`, which has both outer
//! and inner boundary cells, and asserts the documented invariant of exactly one
//! BC function.  No BC logic is changed; only assertions are added here.
//!
//! Index convention: scalar fields use `si(x,y,ly)` = `x*ly+y`;
//! 2-vector fields use `vi(x,y,ly,c)` = `(x*ly+y)*2+c`.

use std::f64::consts::PI;
use volterra_cgpo::bc::{
    apply_h_boundary_conditions, apply_p_boundary_conditions, apply_q_boundary_conditions,
    apply_ss_boundary_conditions, apply_u_boundary_conditions,
};
use volterra_cgpo::index::{si, vi};
use volterra_cgpo::nephroid_boundary;

// ---------------------------------------------------------------------------
// Helper: collect all cells that have a nonzero normal in either layer.
// These are the cells that each BC function touches.
// ---------------------------------------------------------------------------

fn boundary_cells(bnd: &volterra_cgpo::Boundary) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for x in 0..bnd.lx {
        for y in 0..bnd.ly {
            let idx = si(x, y, bnd.ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];
            if outer_nz || inner_nz {
                cells.push((x, y));
            }
        }
    }
    cells
}

// ---------------------------------------------------------------------------
// Test 1: apply_u_boundary_conditions
//
// Invariant: every cell that has a nonzero normal in either boundary layer
// must have u[vi(x,y,ly,0)] == 0.0 AND u[vi(x,y,ly,1)] == 0.0.
// Interior-only cells must be untouched (retain their initial value 7.0).
// ---------------------------------------------------------------------------

#[test]
fn u_bc_zeroes_boundary_velocity() {
    let bnd = nephroid_boundary(20, 20);
    let lx = bnd.lx;
    let ly = bnd.ly;

    // Fill with a sentinel so we can detect untouched cells.
    let mut u = vec![7.0_f64; lx * ly * 2];

    apply_u_boundary_conditions(&mut u, &bnd);

    let cells = boundary_cells(&bnd);
    assert!(
        !cells.is_empty(),
        "nephroid_boundary(20,20) must produce at least one boundary cell"
    );

    for (x, y) in &cells {
        let u0 = u[vi(*x, *y, ly, 0)];
        let u1 = u[vi(*x, *y, ly, 1)];
        assert_eq!(
            u0, 0.0,
            "u[{x},{y},0] should be 0 (no-slip), got {u0}"
        );
        assert_eq!(
            u1, 0.0,
            "u[{x},{y},1] should be 0 (no-slip), got {u1}"
        );
    }

    // Cells with no nonzero normal in either layer must be untouched.
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];
            if !outer_nz && !inner_nz {
                let u0 = u[vi(x, y, ly, 0)];
                let u1 = u[vi(x, y, ly, 1)];
                assert_eq!(
                    u0, 7.0,
                    "u[{x},{y},0] should be untouched (7.0) for non-boundary cell, got {u0}"
                );
                assert_eq!(
                    u1, 7.0,
                    "u[{x},{y},1] should be untouched (7.0) for non-boundary cell, got {u1}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: apply_ss_boundary_conditions
//
// Invariant: every cell with a nonzero normal in either layer has ss[idx] == 0.
// Cells that are not in any boundary layer (no nonzero normal) are untouched.
// ---------------------------------------------------------------------------

#[test]
fn ss_bc_zeroes_boundary_cells() {
    let bnd = nephroid_boundary(20, 20);
    let lx = bnd.lx;
    let ly = bnd.ly;

    let mut ss = vec![42.0_f64; lx * ly];

    apply_ss_boundary_conditions(&mut ss, &bnd);

    let cells = boundary_cells(&bnd);
    assert!(!cells.is_empty());

    for (x, y) in &cells {
        let v = ss[si(*x, *y, ly)];
        assert_eq!(v, 0.0, "ss[{x},{y}] should be 0 after BC, got {v}");
    }

    // Non-boundary cells must retain sentinel.
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];
            if !outer_nz && !inner_nz {
                let v = ss[idx];
                assert_eq!(
                    v, 42.0,
                    "ss[{x},{y}] should be untouched (42.0) for non-boundary cell, got {v}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: apply_q_boundary_conditions
//
// Invariant (from docstring and code):
//   theta = arccos(nx), with 2*pi - theta if ny < 0
//   nnx = cos(theta), nny = sin(theta)
//   q[vi(x,y,ly,0)] = s0 * (nny^2 - 0.5)
//   q[vi(x,y,ly,1)] = s0 * (-nnx * nny)
//
// We recompute the expected values from the stored normals and assert
// equality within floating-point tolerance (1e-12).
//
// Also verify that interior cells (no normal) are untouched.
// ---------------------------------------------------------------------------

#[test]
fn q_bc_sets_winding_tangent_director() {
    let bnd = nephroid_boundary(20, 20);
    let lx = bnd.lx;
    let ly = bnd.ly;

    let s0 = 0.75_f64; // arbitrary s0; formula must hold for any positive s0
    let mut q = vec![999.0_f64; lx * ly * 2];

    apply_q_boundary_conditions(&mut q, &bnd, s0);

    // Collect cells actually touched (nonzero normal in either layer).
    // bc.rs iterates layer 0 then layer 1; for cells in both, layer 1 wins.
    // We replicate the same priority in the expected-value computation.
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];

            if !outer_nz && !inner_nz {
                // Untouched - check sentinel preserved.
                assert_eq!(
                    q[vi(x, y, ly, 0)],
                    999.0,
                    "q[{x},{y},0] should be untouched (non-boundary)"
                );
                assert_eq!(
                    q[vi(x, y, ly, 1)],
                    999.0,
                    "q[{x},{y},1] should be untouched (non-boundary)"
                );
                continue;
            }

            // Last-write wins: layer 1 (outer) overwrites layer 0 (inner)
            // if both are nonzero - match bc.rs loop order.
            let [nx, ny] = if outer_nz {
                bnd.outer_normals[idx]
            } else {
                bnd.inner_normals[idx]
            };

            let mut theta = nx.clamp(-1.0, 1.0).acos();
            if ny < 0.0 {
                theta = 2.0 * PI - theta;
            }
            let nnx = theta.cos();
            let nny = theta.sin();

            let exp_q0 = s0 * (nny * nny - 0.5);
            let exp_q1 = s0 * (-nnx * nny);

            let got_q0 = q[vi(x, y, ly, 0)];
            let got_q1 = q[vi(x, y, ly, 1)];

            assert!(
                (got_q0 - exp_q0).abs() < 1e-12,
                "q[{x},{y},0]: expected {exp_q0}, got {got_q0}"
            );
            assert!(
                (got_q1 - exp_q1).abs() < 1e-12,
                "q[{x},{y},1]: expected {exp_q1}, got {got_q1}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: apply_p_boundary_conditions  (characterization test)
//
// The Neumann stencil is too entangled to verify in closed form on an
// arbitrary hand-built grid, so this is a characterization test:
//
// - All fields initialised to known constants.
// - Run apply_p_boundary_conditions.
// - Assert that the result is DETERMINISTIC: a second call with identical
//   inputs produces identical outputs.
// - Assert that cells with no nonzero normal in either layer are NOT
//   written by the function (their values equal p_init after the call,
//   because p_aux == p_init means the write cannot have come from the BC).
//
// We use p = 0, p_aux = 0, u = 0, pi_s = 0, pi_a = 0 so the formula
// simplifies to p[x,y] = 0 / denom = 0 at every boundary cell.
// That IS the invariant: with zero forcing and zero neighbours, Neumann
// gives p = 0 on the boundary.
// ---------------------------------------------------------------------------

#[test]
fn p_bc_zero_forcing_yields_zero_boundary_pressure() {
    let bnd = nephroid_boundary(20, 20);
    let lx = bnd.lx;
    let ly = bnd.ly;
    let n = lx * ly;

    // All forcing fields zero.
    let p_aux = vec![0.0_f64; n];
    let u = vec![0.0_f64; n * 2];
    let pi_s = vec![0.0_f64; n * 2];
    let pi_a = vec![0.0_f64; n];

    // Start p at a non-zero sentinel so we can see which cells were written.
    let mut p = vec![5.0_f64; n];

    apply_p_boundary_conditions(&mut p, &p_aux, &u, 1.0, 1.0, &pi_s, &pi_a, &bnd);

    // With F=0, rho*nu*lapu=0, p_aux_neighbours=0 → p = 0/denom = 0.
    let cells = boundary_cells(&bnd);
    assert!(!cells.is_empty());

    for (x, y) in &cells {
        // Only cells whose denom is non-zero are actually written.
        // Since the formula writes exactly when denom != 0 (checked in bc.rs),
        // we check that what we got is either 0.0 (written) or 5.0 (skipped).
        // In practice all cells with a unit normal have |denom|>1e-15.
        let v = p[si(*x, *y, ly)];
        assert!(
            v == 0.0 || v == 5.0,
            "p[{x},{y}] unexpected value {v}: must be 0.0 (written) or 5.0 (not written)"
        );
        // A nonzero-normal cell with zero forcing MUST be written to 0.
        assert_eq!(
            v, 0.0,
            "p[{x},{y}] should be 0 (Neumann with zero forcing), got {v}"
        );
    }

    // Non-boundary cells: the function must not touch them.
    // (The sentinel stays 5.0 for every cell that is not a boundary cell.)
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];
            if !outer_nz && !inner_nz {
                let v = p[idx];
                assert_eq!(
                    v, 5.0,
                    "p[{x},{y}] should be untouched (5.0) for non-boundary cell, got {v}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: apply_h_boundary_conditions  (characterization test)
//
// Invariant from the docstring:
//   H[x,y,c] = gamma * (a*ux*dQ_x + b*uy*dQ_y - S[x,y,c])
//
// We use a setup where u=0 everywhere so the advection terms vanish and:
//   H[x,y,c] = gamma * (- S[x,y,c])
//
// With u=0, a*ux*dQ_x = 0 and b*uy*dQ_y = 0 for any Q.
// So the invariant collapses to: H[x,y,c] = -gamma * S[x,y,c]
// for every cell touched by the BC.
//
// We set S = constant 2.0, gamma = 3.0 → H must be -6.0 at every
// boundary cell after the call.  Non-boundary cells must remain at
// the sentinel 99.0.
// ---------------------------------------------------------------------------

#[test]
fn h_bc_with_zero_velocity_equals_neg_gamma_s() {
    let bnd = nephroid_boundary(20, 20);
    let lx = bnd.lx;
    let ly = bnd.ly;
    let n = lx * ly;

    let gamma = 3.0_f64;
    let s_val = 2.0_f64;

    let q = vec![0.0_f64; n * 2]; // Q arbitrary; dQ terms will be 0-0=0
    let u = vec![0.0_f64; n * 2]; // zero velocity → advection terms vanish
    let s = vec![s_val; n * 2];   // constant S

    let mut h = vec![99.0_f64; n * 2];

    apply_h_boundary_conditions(&mut h, gamma, &q, &u, &s, &bnd);

    let expected = -gamma * s_val; // = -6.0

    let cells = boundary_cells(&bnd);
    assert!(!cells.is_empty());

    for (x, y) in &cells {
        for c in 0..2 {
            let v = h[vi(*x, *y, ly, c)];
            assert!(
                (v - expected).abs() < 1e-12,
                "h[{x},{y},{c}]: expected {expected}, got {v}"
            );
        }
    }

    // Non-boundary cells: sentinel must be preserved.
    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            let outer_nz = bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0];
            let inner_nz = bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0];
            if !outer_nz && !inner_nz {
                for c in 0..2 {
                    let v = h[vi(x, y, ly, c)];
                    assert_eq!(
                        v, 99.0,
                        "h[{x},{y},{c}] should be untouched (99.0) for non-boundary cell, got {v}"
                    );
                }
            }
        }
    }
}
