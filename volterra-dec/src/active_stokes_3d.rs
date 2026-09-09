//! Active nematic forcing through the bounded three-dimensional Stokes solver.
//!
//! `stokes_solve_3d` in `volterra-fd` is spectral and periodic, so a chamber
//! there is a mask applied to the velocity after the solve. The mask leaves a
//! slip layer of unmeasured thickness and destroys the exact incompressibility
//! the spectral projector had. This module routes the same active stress through
//! [`BoundedStokes3D`] instead, where the wall is a boundary condition: the
//! normal velocity vanishes exactly, the tangential velocity vanishes weakly,
//! and the cellwise divergence sits at machine precision.
//!
//! # The two projections
//!
//! The active stress lives on the Q-field's grid and the solver wants a 2-form
//! on faces, so the force is evaluated at each face's quadrature points by
//! trilinear interpolation and integrated against the face normal. The solve
//! returns face fluxes and the caller wants a velocity per grid point, so the
//! fluxes are reconstructed through the lowest-order Raviart-Thomas basis of the
//! cell the point falls in.
//!
//! Both maps are lossy in the way any projection is, so each has a test that a
//! constant field survives it exactly.
//!
//! # Geometry
//!
//! The chamber is the box `[0, (nx-1) dx] x [0, (ny-1) dx] x [0, (nz-1) dx]`,
//! so the outermost grid layer sits on the wall and the no-slip condition is
//! stated where the periodic lane wrapped. The mesh resolution is chosen
//! independently of the grid, since the direct factorisation grows faster than
//! the field storage does.
//!
//! # Boundary differences
//!
//! The force is `f_a = -zeta (d_b Q_ab)`, differenced centrally inside and
//! one-sidedly on the wall. The periodic lane wraps instead, which on a bounded
//! chamber differences across the chamber rather than along it.

use volterra_core::{ActiveNematicParams3D, QField3D, VelocityField3D};

use crate::stokes_3d::{BoundedStokes3D, Flow3D, Stokes3DError};
use crate::tet_mesh::StructuredBox;

/// The active body force on the Q-field's own grid.
///
/// The active stress is `sigma_ij = -zeta Q_ij`, so the body force is
/// `f_a = (div sigma)_a = -zeta sum_b d_b Q_ab`. `Q` is stored as
/// `[q11, q12, q13, q22, q23]` with `q33 = -(q11 + q22)`, so the three rows are
/// `(q11, q12, q13)`, `(q12, q22, q23)` and `(q13, q23, q33)`.
///
/// Differences are central in the interior and one-sided on the wall. The
/// spectral lane wraps, which on a bounded chamber takes a difference across the
/// chamber rather than along it, and puts a spurious force one cell deep all
/// round the boundary.
pub fn active_force_on_grid(q: &QField3D, zeta_eff: f64) -> Vec<[f64; 3]> {
    let (nx, ny, nz, dx) = (q.nx, q.ny, q.nz, q.dx);
    let at = |i: usize, j: usize, k: usize| q.q[(k * ny + j) * nx + i];
    // Row `a` of Q, as a function of the five stored components.
    let row = |c: [f64; 5], a: usize| match a {
        0 => [c[0], c[1], c[2]],
        1 => [c[1], c[3], c[4]],
        _ => [c[2], c[4], -(c[0] + c[3])],
    };

    let mut f = vec![[0.0f64; 3]; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut div = [0.0f64; 3];
                for (b, n) in [(0usize, nx), (1, ny), (2, nz)] {
                    let idx = [i, j, k][b];
                    if n < 2 {
                        continue;
                    }
                    let (lo, hi, h) = if idx == 0 {
                        (0, 1, dx)
                    } else if idx == n - 1 {
                        (n - 2, n - 1, dx)
                    } else {
                        (idx - 1, idx + 1, 2.0 * dx)
                    };
                    let mut plo = [i, j, k];
                    let mut phi = [i, j, k];
                    plo[b] = lo;
                    phi[b] = hi;
                    let clo = at(plo[0], plo[1], plo[2]);
                    let chi = at(phi[0], phi[1], phi[2]);
                    for a in 0..3 {
                        div[a] += (row(chi, a)[b] - row(clo, a)[b]) / h;
                    }
                }
                f[(k * ny + j) * nx + i] = [-zeta_eff * div[0], -zeta_eff * div[1], -zeta_eff * div[2]];
            }
        }
    }
    f
}

/// A vector field on a grid, sampled at arbitrary points by trilinear
/// interpolation.
///
/// The grid point `(i, j, k)` sits at `(i dx, j dx, k dx)`, and a point outside
/// the grid is clamped to it rather than extrapolated: the chamber wall is the
/// outermost grid layer, so a quadrature point can sit on the boundary but never
/// beyond it by more than a rounding error.
#[derive(Debug, Clone)]
pub struct GridField {
    values: Vec<[f64; 3]>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
}

impl GridField {
    /// Wrap a per-grid-point vector field.
    pub fn new(values: Vec<[f64; 3]>, nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        assert_eq!(values.len(), nx * ny * nz, "one vector per grid point");
        Self { values, nx, ny, nz, dx }
    }

    /// The field at a point, by trilinear interpolation.
    pub fn sample(&self, x: [f64; 3]) -> [f64; 3] {
        let axis = |v: f64, n: usize| {
            if n < 2 {
                return (0usize, 0usize, 0.0);
            }
            let f = (v / self.dx).clamp(0.0, (n - 1) as f64);
            let lo = (f.floor() as usize).min(n - 2);
            (lo, lo + 1, f - lo as f64)
        };
        let (i0, i1, tx) = axis(x[0], self.nx);
        let (j0, j1, ty) = axis(x[1], self.ny);
        let (k0, k1, tz) = axis(x[2], self.nz);
        let at = |i: usize, j: usize, k: usize| self.values[(k * self.ny + j) * self.nx + i];
        let mut out = [0.0f64; 3];
        for (i, wi) in [(i0, 1.0 - tx), (i1, tx)] {
            for (j, wj) in [(j0, 1.0 - ty), (j1, ty)] {
                for (k, wk) in [(k0, 1.0 - tz), (k1, tz)] {
                    let w = wi * wj * wk;
                    if w == 0.0 {
                        continue;
                    }
                    let v = at(i, j, k);
                    for a in 0..3 {
                        out[a] += w * v[a];
                    }
                }
            }
        }
        out
    }
}

/// The bounded Stokes solver, wired to a Q-field grid.
///
/// The factorisation is built once and reused for every Q-field on the same
/// grid, which is what an outer nematic loop needs: the matrix depends on
/// neither the viscosity nor the activity.
pub struct ConfinedActiveStokes3D {
    solver: BoundedStokes3D,
    boxed: StructuredBox,
    /// Grid divisions the chamber was built for.
    grid: (usize, usize, usize),
    dx: f64,
}

impl ConfinedActiveStokes3D {
    /// Build the chamber for a grid, meshed at the given cell divisions.
    ///
    /// The chamber is `[0, (gnx-1) dx] x [0, (gny-1) dx] x [0, (gnz-1) dx]`, so
    /// the outermost grid layer sits on the wall. The mesh divisions are
    /// independent of the grid: a direct factorisation grows faster than field
    /// storage does, so a fine field on a coarse chamber is the usual case.
    pub fn new(
        grid: (usize, usize, usize),
        dx: f64,
        cells: (usize, usize, usize),
    ) -> Result<Self, Stokes3DError> {
        let (gnx, gny, gnz) = grid;
        assert!(gnx >= 2 && gny >= 2 && gnz >= 2, "a bounded chamber needs two layers per axis");
        let boxed = StructuredBox::new(
            cells.0,
            cells.1,
            cells.2,
            (gnx - 1) as f64 * dx,
            (gny - 1) as f64 * dx,
            (gnz - 1) as f64 * dx,
        );
        let mesh = boxed.build().map_err(|_| Stokes3DError::DegenerateCell)?;
        Ok(Self {
            solver: BoundedStokes3D::new(mesh)?,
            boxed,
            grid,
            dx,
        })
    }

    /// The chamber meshed one cell per grid cell.
    pub fn matching_grid(grid: (usize, usize, usize), dx: f64) -> Result<Self, Stokes3DError> {
        Self::new(grid, dx, (grid.0 - 1, grid.1 - 1, grid.2 - 1))
    }

    /// The underlying solver.
    pub fn solver(&self) -> &BoundedStokes3D {
        &self.solver
    }

    /// The chamber descriptor.
    pub fn chamber(&self) -> StructuredBox {
        self.boxed
    }

    /// Face flux degrees of freedom of a grid vector field.
    ///
    /// The field is sampled at each face's degree-five quadrature points and
    /// integrated against the face normal, so a field constant on the grid
    /// reproduces its exact flux.
    pub fn project_to_faces(&self, field: &GridField) -> Vec<f64> {
        self.solver
            .mesh()
            .flux_dofs(|x: [f64; 3]| field.sample(x))
    }

    /// Solve for the velocity driven by an active Q-field, with no-slip walls.
    ///
    /// Returns the raw discrete solution. [`Self::sample_to_grid`] turns it into
    /// a `VelocityField3D` for the rest of the lane.
    pub fn solve_flow(
        &self,
        q: &QField3D,
        p: &ActiveNematicParams3D,
    ) -> Result<Flow3D, Stokes3DError> {
        if (q.nx, q.ny, q.nz) != self.grid {
            return Err(Stokes3DError::WrongLength {
                expected: self.grid.0 * self.grid.1 * self.grid.2,
                got: q.nx * q.ny * q.nz,
            });
        }
        let force = GridField::new(
            active_force_on_grid(q, p.zeta_eff),
            q.nx,
            q.ny,
            q.nz,
            q.dx,
        );
        let f = self.project_to_faces(&force);
        let wall = vec![0.0; self.solver.mesh().n_faces()];
        self.solver.solve(&f, &wall, p.eta)
    }

    /// The solved velocity at every grid point.
    ///
    /// A grid point on the wall reconstructs from the cell it borders, where the
    /// normal flux is exactly zero and the tangential velocity is whatever the
    /// weak condition left. That number is the wall slip, and it is reported
    /// rather than masked away.
    pub fn sample_to_grid(&self, flow: &Flow3D) -> VelocityField3D {
        let (gnx, gny, gnz) = self.grid;
        let mesh = self.solver.mesh();
        let mut out = VelocityField3D::zeros(gnx, gny, gnz, self.dx);
        for k in 0..gnz {
            for j in 0..gny {
                for i in 0..gnx {
                    let x = [i as f64 * self.dx, j as f64 * self.dx, k as f64 * self.dx];
                    let v = match self.boxed.locate(mesh, x) {
                        Some(t) => flow.velocity_in_cell(mesh, t, x),
                        None => [0.0; 3],
                    };
                    out.u[(k * gny + j) * gnx + i] = v;
                }
            }
        }
        out
    }

    /// Solve and sample in one call, which is what an outer loop wants.
    pub fn solve(
        &self,
        q: &QField3D,
        p: &ActiveNematicParams3D,
    ) -> Result<VelocityField3D, Stokes3DError> {
        let flow = self.solve_flow(q, p)?;
        Ok(self.sample_to_grid(&flow))
    }
}
