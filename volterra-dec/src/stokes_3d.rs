//! Bounded three-dimensional Stokes flow on a tetrahedral complex.
//!
//! `stokes_solve_3d` in `volterra-fd` is spectral and periodic, and a chamber
//! wall there is a mask on the velocity, which leaves a slip layer of unmeasured
//! thickness and abandons exact incompressibility where it matters most. A chip
//! chamber is a slab of finite depth, so the depth-averaged two-dimensional
//! model is an approximation whose error nobody has measured. This solver
//! resolves the depth instead.
//!
//! # Formulation
//!
//! The velocity is a 2-form on faces, so its degree of freedom is the flux
//! through a face, `d2 u` is the cellwise net outflux and is combinatorial, and
//! incompressibility is exact at machine precision by construction. The normal
//! trace is essential, so no-penetration is exact and free.
//!
//! Momentum written in terms of `u` alone puts an `M1` inverse in the operator
//! and destroys sparsity, so the vorticity enters as an unknown. With `w` in
//! `Lambda^1` on edges, `u` in `Lambda^2` on faces and `p` in `Lambda^3` on
//! cells, the weak form is
//!
//! ```text
//! (w, tau) - (u, d1 tau) = 0                   for all tau in Lambda^1
//! eta (d1 w, phi) - (p, d2 phi) = (f, phi)     for all phi in Lambda^2
//! (d2 u, chi) = 0                              for all chi in Lambda^3
//! ```
//!
//! Scaling the first two rows by `1 / eta` and writing `p = eta p_hat` removes
//! the viscosity from every block:
//!
//! ```text
//! [ -M1      d1^T M2    0        ] [w    ]   [ 0          ]
//! [  M2 d1   0         -d2^T M3  ] [u    ] = [ M2 f / eta ]
//! [  0      -M3 d2      0        ] [p_hat]   [ 0          ]
//! ```
//!
//! One matrix therefore serves every viscosity, and a viscosity sweep reuses one
//! factorisation. Eliminating `w` gives a velocity block
//! `M2 d1 M1^-1 d1^T M2`, which is symmetric positive semi-definite: that sign
//! is the difference between viscous dissipation and viscous growth, and it is
//! what fixes the sign of the first row against its transpose.
//!
//! # Boundary conditions
//!
//! No-penetration is exact: every boundary face's flux is essential, set to zero
//! at a wall or to a prescribed value at a port, eliminated from the system with
//! its contribution moved to the right-hand side.
//!
//! Tangential no-slip is weak, and it costs nothing. The identity
//!
//! ```text
//! (curl v, tau) = (v, curl tau) + boundary integral of (n x v) . tau
//! ```
//!
//! says that writing the first row as `(w, tau) = (u, d1 tau)`, with the
//! boundary integral dropped and `tau` ranging over every edge including the
//! boundary ones, imposes `n x v = 0` on the wall weakly. That is the tangential
//! velocity, so the formulation states full no-slip: normal component exactly,
//! tangential component weakly. Free slip is the variant that needs the boundary
//! integral put back with the vorticity trace prescribed.
//!
//! The claim is a derivation, so [`Flow3D::wall_slip`] measures it. No-slip
//! sends the reconstructed tangential wall velocity to zero under refinement,
//! free slip sends it to a constant, and the duct case separates the two on the
//! shape of its profile alone.
//!
//! With every boundary flux prescribed the pressure is determined up to the mode
//! `p_t = c |T_t|`, so one cell's pressure is pinned and its equation dropped.
//! That equation is implied by the others exactly when the prescribed fluxes sum
//! to zero over the boundary, and a boundary condition violating that is
//! rejected rather than answered in a least-squares sense.

use faer::linalg::solvers::SolveCore;
use faer::sparse::{SparseColMat, Triplet};
use faer::Mat;
use sprs::CsMat;

use crate::mimetic::assemble_star;
use crate::tet_mesh::TetComplex;

/// What can go wrong in the three-dimensional solver.
#[derive(Debug, Clone, PartialEq)]
pub enum Stokes3DError {
    /// A cell of the complex admits no mimetic star.
    DegenerateCell,
    /// The assembled saddle point could not be factorised.
    Singular,
    /// The prescribed boundary fluxes do not sum to zero, so the incompressible
    /// problem has no solution. The two numbers are the net flux and the scale
    /// it was compared against.
    IncompatibleBoundaryFlux { net: f64, scale: f64 },
    /// A supplied vector has the wrong length.
    WrongLength { expected: usize, got: usize },
}

impl std::fmt::Display for Stokes3DError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateCell => write!(f, "a cell admits no mimetic star"),
            Self::Singular => write!(f, "the saddle point could not be factorised"),
            Self::IncompatibleBoundaryFlux { net, scale } => write!(
                f,
                "prescribed boundary fluxes sum to {net} against a scale of {scale}, \
                 so the incompressible problem has no solution"
            ),
            Self::WrongLength { expected, got } => {
                write!(f, "expected {expected} entries, received {got}")
            }
        }
    }
}

impl std::error::Error for Stokes3DError {}

/// A factorised bounded Stokes operator on a tetrahedral complex.
///
/// The matrix is independent of the viscosity and of the boundary data, so one
/// instance serves every right-hand side, every viscosity and every set of
/// prescribed wall fluxes on the same mesh.
pub struct BoundedStokes3D {
    mesh: TetComplex,
    d1: CsMat<f64>,
    d2: CsMat<f64>,
    m2: CsMat<f64>,
    /// The diagonal of `M3`, which is the reciprocal cell volume.
    m3: Vec<f64>,
    /// The free faces, in reduced order.
    free_faces: Vec<usize>,
    /// The cell whose pressure is pinned.
    pin: usize,
    n_w: usize,
    n_u: usize,
    n_p: usize,
    lu: faer::sparse::linalg::solvers::Lu<usize, f64>,
}

impl BoundedStokes3D {
    /// Assemble and factorise the operator for a complex.
    ///
    /// Every boundary face is essential. Interior faces are unknowns, every edge
    /// is an unknown, and every cell but the pinned one has a pressure unknown.
    pub fn new(mesh: TetComplex) -> Result<Self, Stokes3DError> {
        let d1 = mesh.d1();
        let d2 = mesh.d2();
        let m1 = assemble_star(&mesh, 1).ok_or(Stokes3DError::DegenerateCell)?;
        let m2 = assemble_star(&mesh, 2).ok_or(Stokes3DError::DegenerateCell)?;
        let m3: Vec<f64> = (0..mesh.n_tets())
            .map(|t| 1.0 / mesh.tet_volume(t))
            .collect();

        let mut face_slot = vec![None; mesh.n_faces()];
        let mut free_faces = Vec::new();
        for f in 0..mesh.n_faces() {
            if !mesh.is_boundary_face(f) {
                face_slot[f] = Some(free_faces.len());
                free_faces.push(f);
            }
        }

        let n_w = mesh.n_edges();
        let n_u = free_faces.len();
        let n_p = mesh.n_tets() - 1;
        let pin = 0usize;
        let n = n_w + n_u + n_p;

        let row_w = |e: usize| e;
        let row_u = |i: usize| n_w + i;
        let row_p = |t: usize| n_w + n_u + (t - 1);

        let mut trip: Vec<Triplet<usize, usize, f64>> = Vec::new();

        // (1,1): -M1.
        for (v, (r, c)) in m1.iter() {
            trip.push(Triplet::new(row_w(r), row_w(c), -v));
        }

        // (1,2) and (2,1). The first is `(d1^T M2)[:, free]`, the second its
        // transpose `(M2 d1)[free, :]`, and both are generated from the same
        // pass so they cannot drift apart.
        for (v, (g, f)) in m2.iter() {
            let Some(row_g) = d1.outer_view(g) else {
                continue;
            };
            if let Some(slot) = face_slot[f] {
                for (e, &s) in row_g.iter() {
                    trip.push(Triplet::new(row_w(e), row_u(slot), s * v));
                }
            }
            if let Some(slot_g) = face_slot[g] {
                if let Some(row_f) = d1.outer_view(f) {
                    for (e, &s) in row_f.iter() {
                        trip.push(Triplet::new(row_u(slot_g), row_w(e), v * s));
                    }
                }
            }
        }

        // (2,3) and (3,2): both `-(d2^T M3)` on the free rows and columns.
        for (s, (t, f)) in d2.iter() {
            let Some(slot) = face_slot[f] else { continue };
            if t == pin {
                continue;
            }
            let v = -s * m3[t];
            trip.push(Triplet::new(row_u(slot), row_p(t), v));
            trip.push(Triplet::new(row_p(t), row_u(slot), v));
        }

        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip)
            .map_err(|_| Stokes3DError::Singular)?;
        let lu = mat.sp_lu().map_err(|_| Stokes3DError::Singular)?;

        Ok(Self {
            mesh,
            d1,
            d2,
            m2,
            m3,
            free_faces,
            pin,
            n_w,
            n_u,
            n_p,
            lu,
        })
    }

    /// The complex the operator was built on.
    pub fn mesh(&self) -> &TetComplex {
        &self.mesh
    }

    /// Solve for a body force and a set of prescribed boundary fluxes.
    ///
    /// `force` is one entry per face: the integral of the body force's normal
    /// component over that face, in the face's canonical orientation.
    /// `boundary_flux` is one entry per face as well, read only on boundary
    /// faces, in the same orientation. A wall is a zero entry, a port a nonzero
    /// one.
    pub fn solve(
        &self,
        force: &[f64],
        boundary_flux: &[f64],
        eta: f64,
    ) -> Result<Flow3D, Stokes3DError> {
        let nf = self.mesh.n_faces();
        if force.len() != nf {
            return Err(Stokes3DError::WrongLength {
                expected: nf,
                got: force.len(),
            });
        }
        if boundary_flux.len() != nf {
            return Err(Stokes3DError::WrongLength {
                expected: nf,
                got: boundary_flux.len(),
            });
        }
        assert!(eta > 0.0, "viscosity must be positive");

        // The prescribed part of the velocity, zero on every interior face.
        let mut u_bc = vec![0.0; nf];
        for f in 0..nf {
            if self.mesh.is_boundary_face(f) {
                u_bc[f] = boundary_flux[f];
            }
        }

        // Compatibility: the incompressible problem needs the prescribed fluxes
        // to sum to zero over the boundary.
        let div_bc = matvec(&self.d2, &u_bc);
        let net: f64 = div_bc.iter().sum();
        let scale: f64 = u_bc
            .iter()
            .zip(0..nf)
            .map(|(x, f)| x.abs().max(0.0) * if self.mesh.is_boundary_face(f) { 1.0 } else { 0.0 })
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        if net.abs() > 1e-10 * scale {
            return Err(Stokes3DError::IncompatibleBoundaryFlux { net, scale });
        }

        let n = self.n_w + self.n_u + self.n_p;
        let mut rhs = Mat::<f64>::zeros(n, 1);

        // Row w: the eliminated columns of `d1^T M2`.
        let m2_ubc = matvec(&self.m2, &u_bc);
        let y = matvec_transpose(&self.d1, &m2_ubc, self.n_w);
        for e in 0..self.n_w {
            rhs[(e, 0)] = -y[e];
        }

        // Row u: the source, scaled by the viscosity.
        let m2_f = matvec(&self.m2, force);
        for (i, &f) in self.free_faces.iter().enumerate() {
            rhs[(self.n_w + i, 0)] = m2_f[f] / eta;
        }

        // Row p: the eliminated columns of `-M3 d2`.
        for t in 0..self.mesh.n_tets() {
            if t == self.pin {
                continue;
            }
            rhs[(self.n_w + self.n_u + (t - 1), 0)] = self.m3[t] * div_bc[t];
        }

        self.lu.solve_in_place_with_conj(faer::Conj::No, rhs.as_mut());

        let vorticity: Vec<f64> = (0..self.n_w).map(|e| rhs[(e, 0)]).collect();
        let mut flux = u_bc;
        for (i, &f) in self.free_faces.iter().enumerate() {
            flux[f] = rhs[(self.n_w + i, 0)];
        }
        let mut pressure = vec![0.0; self.mesh.n_tets()];
        for t in 0..self.mesh.n_tets() {
            if t == self.pin {
                continue;
            }
            pressure[t] = eta * rhs[(self.n_w + self.n_u + (t - 1), 0)];
        }

        let div = matvec(&self.d2, &flux);
        let divergence_residual = div.iter().fold(0.0_f64, |a, x| a.max(x.abs()));

        Ok(Flow3D {
            vorticity,
            flux,
            pressure,
            divergence_residual,
        })
    }

    /// Degrees of freedom of the reduced system: vorticity, free fluxes, free
    /// pressures.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.n_w, self.n_u, self.n_p)
    }
}

/// The result of one solve.
#[derive(Debug, Clone)]
pub struct Flow3D {
    /// Vorticity 1-form, one entry per edge: the circulation along the edge in
    /// its ascending orientation.
    pub vorticity: Vec<f64>,
    /// Velocity 2-form, one entry per face: the flux through the face in its
    /// canonical orientation.
    pub flux: Vec<f64>,
    /// Pressure 3-form, one entry per cell: the pressure times the cell volume,
    /// with the pinned cell at zero.
    pub pressure: Vec<f64>,
    /// The largest cellwise net outflux, which is the incompressibility the
    /// masked spectral solver never had.
    pub divergence_residual: f64,
}

impl Flow3D {
    /// The velocity vector at a point of a cell, from the lowest-order
    /// Raviart-Thomas reconstruction.
    ///
    /// For the face opposite vertex `l` of a cell of volume `V`, the basis field
    /// `(x - p_l) / (3 V)` has unit outward flux through that face and none
    /// through the other three, so the velocity is the signed sum of the four
    /// face fluxes against it. The sign is `d2[t, f]`, which turns a canonically
    /// oriented flux into an outward one.
    pub fn velocity_in_cell(&self, mesh: &TetComplex, t: usize, x: [f64; 3]) -> [f64; 3] {
        let v = 3.0 * mesh.tet_volume(t);
        let o = mesh.tet_orient[t];
        let mut out = [0.0; 3];
        for k in 0..4 {
            let f = mesh.tet_faces[t][k];
            let sign = if k % 2 == 0 { o } else { -o };
            let p = mesh.vertices[mesh.tets[t][k]];
            let c = sign * self.flux[f] / v;
            for i in 0..3 {
                out[i] += c * (x[i] - p[i]);
            }
        }
        out
    }

    /// The velocity at each vertex, as a volume-weighted mean over the cells
    /// that meet there.
    pub fn velocity_at_vertices(&self, mesh: &TetComplex) -> Vec<[f64; 3]> {
        let incident = mesh.vertex_tets();
        (0..mesh.n_vertices())
            .map(|v| {
                let x = mesh.vertices[v];
                let mut acc = [0.0; 3];
                let mut wsum = 0.0;
                for &t in &incident[v] {
                    let vol = mesh.tet_volume(t);
                    let u = self.velocity_in_cell(mesh, t, x);
                    for i in 0..3 {
                        acc[i] += vol * u[i];
                    }
                    wsum += vol;
                }
                if wsum > 0.0 {
                    for a in &mut acc {
                        *a /= wsum;
                    }
                }
                acc
            })
            .collect()
    }

    /// The tangential velocity on the boundary, reconstructed at each boundary
    /// face's centroid.
    ///
    /// Returns the largest and the area-weighted root mean square, both in the
    /// units of velocity. Tangential no-slip is imposed weakly, so these are the
    /// measurement that decides whether the derivation is right: they fall with
    /// mesh size under no-slip and settle on a constant under free slip.
    pub fn wall_slip(&self, mesh: &TetComplex) -> WallSlip {
        let mut owner = vec![usize::MAX; mesh.n_faces()];
        for t in 0..mesh.n_tets() {
            for &f in &mesh.tet_faces[t] {
                if mesh.is_boundary_face(f) {
                    owner[f] = t;
                }
            }
        }
        let mut worst = 0.0_f64;
        let mut acc = 0.0;
        let mut area = 0.0;
        for f in 0..mesh.n_faces() {
            if !mesh.is_boundary_face(f) {
                continue;
            }
            let t = owner[f];
            let c = mesh.face_centroid(f);
            let v = self.velocity_in_cell(mesh, t, c);
            let n = mesh.face_normal(f);
            let vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            let tang = [v[0] - vn * n[0], v[1] - vn * n[1], v[2] - vn * n[2]];
            let mag = (tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]).sqrt();
            let a = mesh.face_area(f);
            worst = worst.max(mag);
            acc += a * mag * mag;
            area += a;
        }
        WallSlip {
            max: worst,
            rms: if area > 0.0 { (acc / area).sqrt() } else { 0.0 },
        }
    }

    /// The kinetic-energy-like norm of the flux, `sqrt(u^T M2 u)`, which is the
    /// L2 norm of the velocity field.
    pub fn velocity_norm(&self, m2: &CsMat<f64>) -> f64 {
        let mut s = 0.0;
        for (v, (r, c)) in m2.iter() {
            s += self.flux[r] * v * self.flux[c];
        }
        s.max(0.0).sqrt()
    }
}

/// The measured tangential velocity on the wall.
#[derive(Debug, Clone, Copy)]
pub struct WallSlip {
    /// The largest tangential speed on any boundary face centroid.
    pub max: f64,
    /// The area-weighted root mean square over the boundary.
    pub rms: f64,
}

/// Dense mat-vec against a sparse matrix.
pub fn matvec(m: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; m.rows()];
    for (v, (r, c)) in m.iter() {
        y[r] += v * x[c];
    }
    y
}

/// Dense mat-vec against the transpose of a sparse matrix.
pub fn matvec_transpose(m: &CsMat<f64>, x: &[f64], cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; cols];
    for (v, (r, c)) in m.iter() {
        y[c] += v * x[r];
    }
    y
}
