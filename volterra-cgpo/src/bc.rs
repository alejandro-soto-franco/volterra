//! Boundary condition kernels for the CGPO solver (Task E).
//!
//! Ports the five BC functions from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py` lines 555–630:
//!
//! - [`apply_u_boundary_conditions`]   — no-slip: u=0 on both boundary layers.
//! - [`apply_ss_boundary_conditions`]  — zero saddle-splay on boundary cells
//!   (plotting helper; not used in the physics loop but included for completeness).
//! - [`apply_p_boundary_conditions`]   — Neumann pressure BC from ∇p·n̂ = F·n̂.
//! - [`apply_q_boundary_conditions`]   — Dirichlet Q anchoring from winding tangent.
//! - [`apply_h_boundary_conditions`]   — H BC to enforce ∂_t Q|_∂ = 0.
//!
//! # Field layout
//!
//! Follows `ops.rs` conventions:
//! - Scalar fields: `&[f64]` length `lx*ly`, index `x*ly+y`.
//! - 2-component fields: `&[f64]` length `lx*ly*2`, index `(x*ly+y)*2+c`.
//!
//! # Python `boundary` array vs `Boundary` struct
//!
//! Python uses `boundary[layer, x, y, component]` where layer 0 = inner,
//! layer 1 = outer.  Our `Boundary` struct stores the same information in
//! `is_outer`/`outer_normals` (layer 1) and `is_inner`/`inner_normals` (layer 0).
//! Every Python loop `for l in range(2)` is replicated here by iterating over
//! both layers in the same order (layer 0 = inner first, then layer 1 = outer).

use crate::index::{si, vi};
use crate::Boundary;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// sign helper (matches Python's round(nx / abs(nx)))
// ---------------------------------------------------------------------------

/// Integer sign: returns -1, 0, or +1 as i64.
#[inline(always)]
fn sign_i(v: f64) -> i64 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Wrapping neighbour index (Python: (i - a) where a can be -1,0,1; wraps via %.
#[inline(always)]
fn wrap(i: usize, delta: i64, n: usize) -> usize {
    let r = (i as i64 + n as i64 + delta).rem_euclid(n as i64) as usize;
    r
}

// ---------------------------------------------------------------------------
// Helper: is this (x,y) a boundary cell at layer l (0=inner, 1=outer)?
// Returns the normal [nx, ny] if so, else None.
// ---------------------------------------------------------------------------

#[inline(always)]
fn get_normal(bnd: &Boundary, x: usize, y: usize, layer: usize) -> Option<[f64; 2]> {
    let idx = si(x, y, bnd.ly);
    match layer {
        0 => {
            if bnd.is_inner[idx] {
                let n = bnd.inner_normals[idx];
                if n[0] != 0.0 || n[1] != 0.0 {
                    Some(n)
                } else {
                    None
                }
            } else {
                None
            }
        }
        1 => {
            if bnd.is_outer[idx] {
                let n = bnd.outer_normals[idx];
                if n[0] != 0.0 || n[1] != 0.0 {
                    Some(n)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Public BC kernels
// ---------------------------------------------------------------------------

/// No-slip velocity boundary condition.
///
/// Ports `apply_u_boundary_conditions(u, boundary)`.
///
/// Python code (lines 556–567) first computes a Lions slip BC, then immediately
/// overwrites with `u[x,y,:] = 0`.  The net effect is: set u=0 at every cell
/// that has a nonzero normal in either boundary layer.
pub fn apply_u_boundary_conditions(u: &mut [f64], bnd: &Boundary) {
    let lx = bnd.lx;
    let ly = bnd.ly;

    for layer in 0..2_usize {
        for x in 0..lx {
            for y in 0..ly {
                if get_normal(bnd, x, y, layer).is_some() {
                    // Python: compute Lions BC then unconditionally set u[x,y,:] = 0
                    u[vi(x, y, ly, 0)] = 0.0;
                    u[vi(x, y, ly, 1)] = 0.0;
                }
            }
        }
    }
}

/// Zero saddle-splay on all boundary cells (plotting helper).
///
/// Ports `apply_ss_boundary_conditions(ss, boundary)`.
///
/// Sets `ss[x,y] = 0` for every cell that is in either boundary layer.
pub fn apply_ss_boundary_conditions(ss: &mut [f64], bnd: &Boundary) {
    let lx = bnd.lx;
    let ly = bnd.ly;

    for x in 0..lx {
        for y in 0..ly {
            let idx = si(x, y, ly);
            if (bnd.is_outer[idx] && bnd.outer_normals[idx] != [0.0, 0.0])
                || (bnd.is_inner[idx] && bnd.inner_normals[idx] != [0.0, 0.0])
            {
                ss[idx] = 0.0;
            }
        }
    }
}

/// Pressure Neumann BC: ∂_ν p = F·ν at the boundary.
///
/// Ports `apply_p_boundary_conditions(p, p_aux, boundary, u, ρ, Π_S, Π_A, ν)`.
///
/// For each boundary cell (both layers, matching Python's `for l in range(2)`):
/// ```text
/// a = sign(nx), b = sign(ny)
/// Fx = a*(Π_S[x,y,0] - Π_S[x-a,y,0]) + b*(Π_S[x,y,1]+Π_A[x,y] - Π_S[x,y-b,1]-Π_A[x,y-b])
/// Fy = a*(Π_S[x,y,1]-Π_A[x,y] - Π_S[x-a,y,1]+Π_A[x-a,y]) - b*(Π_S[x,y,0]-Π_S[x,y-b,0])
/// lapu = 2*u[x,y] - 2*(u[x-a,y]+u[x,y-b]) + u[x-2a,y] + u[x,y-2b]
/// p[x,y] = (n·(F + ρ*ν*lapu) + a*nx*p_aux[x-a,y] + b*ny*p_aux[x,y-b]) / (a*nx + b*ny)
/// ```
pub fn apply_p_boundary_conditions(
    p: &mut [f64],
    p_aux: &[f64],
    u: &[f64],
    rho: f64,
    nu: f64,
    pi_s: &[f64],
    pi_a: &[f64],
    bnd: &Boundary,
) {
    let lx = bnd.lx;
    let ly = bnd.ly;

    for layer in 0..2_usize {
        for x in 0..lx {
            for y in 0..ly {
                let Some([nx, ny]) = get_normal(bnd, x, y, layer) else {
                    continue;
                };

                let a = sign_i(nx); // -1, 0, or +1
                let b = sign_i(ny);

                let denom = a as f64 * nx + b as f64 * ny;
                if denom.abs() < 1e-15 {
                    continue; // degenerate normal (shouldn't happen)
                }

                // Neighbours (wrap around the grid)
                let xa = wrap(x, -a, lx);
                let xaa = wrap(x, -2 * a, lx);
                let yb = wrap(y, -b, ly);
                let ybb = wrap(y, -2 * b, ly);

                // Force components from Π_S and Π_A
                let fx = a as f64 * (pi_s[vi(x, y, ly, 0)] - pi_s[vi(xa, y, ly, 0)])
                    + b as f64
                        * (pi_s[vi(x, y, ly, 1)] + pi_a[si(x, y, ly)]
                            - pi_s[vi(x, yb, ly, 1)]
                            - pi_a[si(x, yb, ly)]);

                let fy = a as f64
                    * (pi_s[vi(x, y, ly, 1)] - pi_a[si(x, y, ly)]
                        - pi_s[vi(xa, y, ly, 1)]
                        + pi_a[si(xa, y, ly)])
                    - b as f64 * (pi_s[vi(x, y, ly, 0)] - pi_s[vi(x, yb, ly, 0)]);

                // 1D Laplacian of u along the normal direction
                // Python: lapu = 2*u[x,y] - 2*(u[x-a,y]+u[x,y-b]) + u[x-2a,y] + u[x,y-2b]
                let lapu0 = 2.0 * u[vi(x, y, ly, 0)]
                    - 2.0 * (u[vi(xa, y, ly, 0)] + u[vi(x, yb, ly, 0)])
                    + u[vi(xaa, y, ly, 0)]
                    + u[vi(x, ybb, ly, 0)];
                let lapu1 = 2.0 * u[vi(x, y, ly, 1)]
                    - 2.0 * (u[vi(xa, y, ly, 1)] + u[vi(x, yb, ly, 1)])
                    + u[vi(xaa, y, ly, 1)]
                    + u[vi(x, ybb, ly, 1)];

                // n · (F + ρ*ν*∇²u)
                let n_dot = nx * (fx + rho * nu * lapu0) + ny * (fy + rho * nu * lapu1);

                // Neumann stencil: p from p_aux neighbours along the inward normal
                let p_neighbours =
                    a as f64 * nx * p_aux[si(xa, y, ly)] + b as f64 * ny * p_aux[si(x, yb, ly)];

                p[si(x, y, ly)] = (n_dot + p_neighbours) / denom;
            }
        }
    }
}

/// Q anchoring BC: Dirichlet, director set to winding tangent.
///
/// Ports `apply_Q_boundary_conditions(Q, boundary)`.
///
/// Python (lines 599–612):
/// ```python
/// net_charge = 2 / 2   # = 1.0  (the "winding index" for k=2 nephroid)
/// theta = arccos(nx)
/// if ny < 0: theta = 2*pi - theta
/// nnx, nny = cos(theta * net_charge), sin(theta * net_charge)
/// Q[x,y,0] = S0 * (nny**2 - 0.5)
/// Q[x,y,1] = S0 * (-nnx * nny)
/// ```
/// With `net_charge = 1`, `nnx = cos(arccos(nx)) = nx` and `nny = sin(arccos(nx))`.
/// For ny < 0 we use the reflection branch so that `nny = -|sin(arccos(nx))|`.
/// The resulting Q is for the **tangent** director `n = (nny, -nnx)`, i.e. the
/// outward normal rotated 90°.
///
/// `s0` is the preferred scalar order parameter.
pub fn apply_q_boundary_conditions(q: &mut [f64], bnd: &Boundary, s0: f64) {
    let lx = bnd.lx;
    let ly = bnd.ly;
    // net_charge = 2/2 = 1 (Python integer division of two int literals evaluates to 1)
    let net_charge = 1.0_f64;

    for layer in 0..2_usize {
        for x in 0..lx {
            for y in 0..ly {
                let Some([nx, ny]) = get_normal(bnd, x, y, layer) else {
                    continue;
                };

                // theta = arccos(nx), adjusted for ny < 0
                let mut theta = nx.clamp(-1.0, 1.0).acos();
                if ny < 0.0 {
                    theta = 2.0 * PI - theta;
                }

                let nnx = (theta * net_charge).cos();
                let nny = (theta * net_charge).sin();

                // Q for director (nnx, nny):
                // Q_xx = S0*(n_y^2 - 1/2) but n = (nny, -nnx) → Q_xx = S0*(nny^2 - 1/2)
                // Wait — Python literally writes:
                //   Q[x,y,0] = S0 * (nny**2 - 1/2)
                //   Q[x,y,1] = S0 * (-nnx * nny)
                // which is Q for director n=(nnx, nny) using the traceless-symmetric form
                // Q_ij = S0*(n_i n_j - delta_ij/2) with xx=(nx^2-1/2), xy=nx*ny,
                // but the Python uses (nny^2-1/2) for Q_xx and (-nnx*nny) for Q_xy.
                // This corresponds to director n = (nny, nnx) rotated (the tangent).
                q[vi(x, y, ly, 0)] = s0 * (nny * nny - 0.5);
                q[vi(x, y, ly, 1)] = s0 * (-nnx * nny);
            }
        }
    }
}

/// H boundary condition: enforce ∂_t Q|_∂ = 0.
///
/// Ports `apply_H_boundary_conditions(H, γ, Q, u, S, boundary)`.
///
/// Python (lines 615–629): for every cell that is in ANY boundary layer
/// (note: Python checks `boundary[0,...] OR boundary[1,...]` for the outer
/// condition, but then uses `boundary[l,x,y]` for the normal):
///
/// ```python
/// for l in range(2):
///   for x,y where boundary[l,x,y] is nonzero:
///     nx, ny = boundary[l,x,y]
///     a = sign(nx), b = sign(ny)
///     H[x,y,:] = γ * (a*u[x,y,0]*(Q[x,y]-Q[x-a,y]) + b*u[x,y,1]*(Q[x,y]-Q[x,y-b]) - S[x,y,:])
/// ```
///
/// Note the Python's outer loop `for l in range(2)` means it iterates both
/// layers; for cells in BOTH layers the inner update runs twice (layer 0 first,
/// then layer 1 overwrites).
pub fn apply_h_boundary_conditions(
    h: &mut [f64],
    gamma: f64,
    q: &[f64],
    u: &[f64],
    s: &[f64],
    bnd: &Boundary,
) {
    let lx = bnd.lx;
    let ly = bnd.ly;

    // Python: outer if-condition checks ANY layer nonzero, then inner update uses boundary[l,x,y]
    // for l in range(2). So layer 0 applies first, layer 1 second (possibly overwriting).
    for layer in 0..2_usize {
        for x in 0..lx {
            for y in 0..ly {
                // Python outer condition: checks boundary[0,...] OR boundary[1,...] for *any* nonzero
                // But then uses boundary[l,x,y] for the normal of the current layer.
                // So we only apply when THIS layer has a nonzero normal.
                let Some([nx, ny]) = get_normal(bnd, x, y, layer) else {
                    continue;
                };

                let a = sign_i(nx);
                let b = sign_i(ny);

                let xa = wrap(x, -a, lx);
                let yb = wrap(y, -b, ly);

                let ux = u[vi(x, y, ly, 0)];
                let uy = u[vi(x, y, ly, 1)];

                for c in 0..2 {
                    let dq_x = q[vi(x, y, ly, c)] - q[vi(xa, y, ly, c)];
                    let dq_y = q[vi(x, y, ly, c)] - q[vi(x, yb, ly, c)];
                    h[vi(x, y, ly, c)] =
                        gamma * (a as f64 * ux * dq_x + b as f64 * uy * dq_y - s[vi(x, y, ly, c)]);
                }
            }
        }
    }
}
