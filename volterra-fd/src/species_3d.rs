//! Transport of a composition field, in flux form.
//!
//! Nucleation is driven by a local supersaturation, so a chemistry model needs
//! concentrations moved by the same flow that moves the director. Two properties
//! decide the scheme.
//!
//! **Conservation is exact.** Every interior face contributes its flux to one
//! cell and takes it from the other, so the total is conserved to round-off
//! rather than to a tolerance. Solute a scheme loses is nucleation the model
//! loses, silently, and a drift is worst where gradients are sharp, which is
//! where nucleation happens.
//!
//! **A front stays between its own bounds.** The face values come from a
//! van Leer limited reconstruction, second order where the field is smooth and
//! first order at a front. An unlimited scheme overshoots at a step, and a
//! negative concentration is a rate law evaluated where no material is.
//!
//! # What is reported rather than enforced
//!
//! [`species_step_limit`] gives the explicit bound the caller has to respect,
//! and [`batchelor_scale`] says how fine the scalar's own structure is. A
//! liquid's Schmidt number is of order a thousand, which puts that structure
//! some thirty times below the flow's, and no grid affordable in three
//! dimensions resolves it. The numbers are reported so a run can state whether
//! it could, since an unresolved scalar mixes at the scheme's rate rather than
//! the fluid's.

use volterra_core::{SpeciesField3D, VelocityField3D};

/// The van Leer limiter, which is what keeps a front monotone.
#[inline]
fn van_leer(r: f64) -> f64 {
    if !r.is_finite() || r <= 0.0 {
        0.0
    } else {
        2.0 * r / (1.0 + r)
    }
}

/// One explicit step of advection and diffusion, in flux form.
///
/// The velocity is taken at cell centres and averaged onto faces. Boundaries
/// wrap, matching every other 3D stencil in this crate.
pub fn advect_diffuse_species(
    field: &SpeciesField3D,
    vel: &VelocityField3D,
    dt: f64,
) -> SpeciesField3D {
    let (nx, ny, nz, dx) = (field.nx, field.ny, field.nz, field.dx);
    assert_eq!(
        vel.u.len(),
        nx * ny * nz,
        "the flow and the field disagree in size"
    );
    let mut out = field.clone();

    let idx = |i: usize, j: usize, l: usize| ((i % nx) * ny + (j % ny)) * nz + (l % nz);
    let step = [[1usize, 0, 0], [0, 1, 0], [0, 0, 1]];
    let dims = [nx, ny, nz];

    for (s, conc) in field.c.iter().enumerate() {
        let d = field.diffusivity.get(s).copied().unwrap_or(0.0);
        let mut delta = vec![0.0f64; conc.len()];

        for axis in 0..3 {
            let (sx, sy, sz) = (step[axis][0], step[axis][1], step[axis][2]);
            let n_axis = dims[axis];
            for i in 0..nx {
                for j in 0..ny {
                    for l in 0..nz {
                        // The face between this cell and its neighbour up-axis.
                        let here = idx(i, j, l);
                        let up = idx(i + sx, j + sy, l + sz);
                        let up2 = idx(i + 2 * sx, j + 2 * sy, l + 2 * sz);
                        let down = idx(
                            i + (n_axis - 1) * sx,
                            j + (n_axis - 1) * sy,
                            l + (n_axis - 1) * sz,
                        );

                        let u_face = 0.5 * (vel.u[here][axis] + vel.u[up][axis]);
                        let jump = conc[up] - conc[here];

                        // A limited reconstruction from whichever side the flow
                        // comes from.
                        let face_value = if u_face >= 0.0 {
                            let r = if jump.abs() > 0.0 {
                                (conc[here] - conc[down]) / jump
                            } else {
                                0.0
                            };
                            conc[here] + 0.5 * van_leer(r) * jump
                        } else {
                            let r = if jump.abs() > 0.0 {
                                (conc[up2] - conc[up]) / jump
                            } else {
                                0.0
                            };
                            conc[up] - 0.5 * van_leer(r) * jump
                        };

                        let flux = u_face * face_value - d * jump / dx;
                        let carried = dt * flux / dx;
                        delta[here] -= carried;
                        delta[up] += carried;
                    }
                }
            }
        }

        for (k, v) in delta.iter().enumerate() {
            out.c[s][k] = conc[k] + v;
        }
    }
    out
}

/// The largest step the explicit scheme takes, from the advective and diffusive
/// limits together.
///
/// The diffusive bound is `dx^2 / (6 D)` in three dimensions and the advective
/// one is `dx / |u|`. Reported rather than imposed, since a caller with its own
/// stepper needs the number rather than an opinion.
pub fn species_step_limit(diffusivity: &[f64], max_speed: f64, dx: f64) -> f64 {
    let diffusive = diffusivity
        .iter()
        .filter(|d| **d > 0.0)
        .map(|d| dx * dx / (6.0 * d))
        .fold(f64::INFINITY, f64::min);
    let advective = if max_speed > 0.0 {
        dx / max_speed
    } else {
        f64::INFINITY
    };
    let limit = diffusive.min(advective);
    if limit.is_finite() { limit } else { f64::MAX }
}

/// The Batchelor scale, the finest structure a scalar develops in a given flow.
///
/// `eta / sqrt(Sc)`, with `eta` the Kolmogorov scale. At a liquid's Schmidt
/// number of order a thousand this is some thirty times below the flow's own
/// finest scale, so a grid that resolves the velocity does not resolve the
/// concentration, and the mixing rate a run reports is the scheme's rather than
/// the fluid's until a sub-grid model says otherwise.
pub fn batchelor_scale(kolmogorov: f64, schmidt: f64) -> f64 {
    if schmidt <= 0.0 {
        return kolmogorov;
    }
    kolmogorov / schmidt.sqrt()
}
