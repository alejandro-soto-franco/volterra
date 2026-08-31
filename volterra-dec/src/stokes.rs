//! Stokes solver for active nematics on 2D DEC meshes.
//!
//! Computes the incompressible velocity field driven by the active stress
//! sigma = -zeta Q via the stream-function formulation. The velocity is
//! a 3D tangent vector at each vertex (for surfaces embedded in R^3).

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use nalgebra::DVector;
use volterra_core::ActiveNematicParams;

use std::cell::RefCell;

use crate::poisson::PoissonSolver;
use crate::QField;

/// Precomputed Stokes solver with cached vertex coordinates and Poisson factorisation.
pub struct SurfaceStokes {
    poisson: PoissonSolver,
    /// The biharmonic's OUTER solve, `-(Delta + 2K)`.
    ///
    /// Steady Stokes on a surface is `Delta (Delta + 2K) psi = curl f`, not the
    /// flat `Delta^2 psi = curl f`. The curvature shift sits on the outer
    /// factor, which is also where the Killing kernel is: on a sphere
    /// `(Delta + 2K)` annihilates the rigid rotations, so the solve returns a
    /// stream function with no rigid spin in it. On a flat mesh the angle
    /// defect vanishes, the shift is zero, and this is the same operator as
    /// `poisson`.
    outer: PoissonSolver,
    /// Last iterate of the biharmonic's INNER solve, one slot per warm entry
    /// point so each is seeded by its own previous answer rather than another's.
    ///
    /// The outer solve was already warm-started from the caller's stream
    /// function; the inner one was not, because a stream function is not a
    /// vorticity and there was nowhere to keep the right thing. Half the
    /// biharmonic therefore started from zero every step on a field that moves
    /// by order `dt`. This is a starting point and nothing else: the conjugate
    /// gradient stops on the same relative residual whatever it starts from, so
    /// a stale or foreign seed costs iterations and never the answer.
    inner_cache: [RefCell<Option<Vec<f64>>>; 3],
    n_vertices: usize,
    /// Vertex coordinates in R^3 (for 2D meshes, z = 0).
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (barycentric, 1/3 of incident triangle areas).
    dual_areas: Vec<f64>,
    /// Hodge star on 1-forms: star_1[e] = |dual_edge| / |primal_edge|.
    star1: Vec<f64>,
    /// Per-vertex unit normal.
    normals: Vec<[f64; 3]>,
    /// Per-vertex tangent frame e1 direction (e2 = n x e1).
    e1_frames: Vec<[f64; 3]>,
    /// Vertices at which velocity is zeroed after recovery (no-slip BCs).
    /// Empty for closed-manifold mode (no zeroing applied).
    no_slip_vertices: Vec<usize>,
    /// Present when the solver was built for a clamped, that is no-slip, wall.
    clamped: Option<ClampedCorrection>,
}

/// Precomputed data for the CLAMPED biharmonic condition.
///
/// The two Dirichlet solves impose `psi = 0` and `Delta psi = 0`, which is the
/// SIMPLY SUPPORTED plate, that is a free-slip wall. No-slip is the clamped
/// plate, `psi = 0` and `dpsi/dn = 0`. On a disc under uniform load the two
/// differ by exactly `2 sqrt 2` in peak velocity, free slip being the faster,
/// and that is the size of the gap measured between this solver and the
/// reference lattice at matched parameters.
///
/// The split into two Dirichlet solves cannot represent the clamped condition,
/// because `Delta psi = 0` on the wall is what makes it separable. The classical
/// remedy is to treat the wall values of `omega = Delta psi` as unknowns chosen
/// so the normal derivative vanishes. The map from those values to `dpsi/dn` is
/// affine and the mesh does not move, so the whole of it is precomputed once:
/// one pair of Poisson solves per boundary vertex builds a response basis, and
/// a dense factorisation of the resulting matrix turns the per-step correction
/// into a matrix-vector product. The time step therefore costs the same two
/// solves it always did.
struct ClampedCorrection {
    boundary: Vec<usize>,
    /// Condition number of the boundary response matrix, from its singular
    /// values at construction. The correction it drives is GLOBAL, so a large
    /// value here pollutes the whole domain rather than a wall layer, and that
    /// is worth knowing before a run rather than after one diverges.
    cond: f64,
    /// Singular values of that matrix, kept so a rank deficiency can be read as
    /// a spectrum rather than as a single infinity.
    spectrum: Vec<f64>,
    /// `phi[j]` is the stream function produced by a unit value of `omega` at
    /// boundary vertex `j`, with `phi = 0` on the wall.
    phi: Vec<Vec<f64>>,
    /// Factorised `L[i][j] = dphi_j/dn` at boundary vertex `i`.
    lu: nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
    /// The same matrix unfactorised, kept for the least-squares fallback.
    lmat: nalgebra::DMatrix<f64>,
    /// Inward unit direction at each boundary vertex, from the vertex towards
    /// the centroid of its incident triangles. Only its consistency matters,
    /// since the functional driven to zero is `grad psi . d`.
    inward: Vec<[f64; 3]>,
}

/// Velocity field on a DEC mesh: 3D tangent vector per vertex.
#[derive(Debug, Clone)]
pub struct VelocityField {
    /// Velocity components at each vertex (3D tangent vector).
    pub v: Vec<[f64; 3]>,
    pub n_vertices: usize,
}

impl VelocityField {
    pub fn zeros(nv: usize) -> Self {
        Self {
            v: vec![[0.0; 3]; nv],
            n_vertices: nv,
        }
    }

    /// Convenience accessors for backward compatibility.
    pub fn vx(&self, i: usize) -> f64 { self.v[i][0] }
    pub fn vy(&self, i: usize) -> f64 { self.v[i][1] }
    pub fn vz(&self, i: usize) -> f64 { self.v[i][2] }

    /// Velocity magnitude at vertex i.
    pub fn speed(&self, i: usize) -> f64 {
        let [x, y, z] = self.v[i];
        (x * x + y * y + z * z).sqrt()
    }
}

/// Extract vertex coordinates from a generic mesh by formatting via Debug.
/// All cartan manifold Point types are SVector<f64, N>.
pub fn extract_coords<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Vec<[f64; 3]> {
    mesh.vertices.iter().map(|v| {
        let s = format!("{:?}", v);
        let nums: Vec<f64> = s
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',' || *c == ' ' || *c == 'e' || *c == '+')
            .collect::<String>()
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        match nums.len() {
            2 => [nums[0], nums[1], 0.0],
            n if n >= 3 => [nums[0], nums[1], nums[2]],
            _ => [0.0, 0.0, 0.0],
        }
    }).collect()
}

/// Compute barycentric dual cell areas from triangles.
pub fn compute_dual_areas(nv: usize, simplices: &[[usize; 3]], coords: &[[f64; 3]]) -> Vec<f64> {
    let mut areas = vec![0.0_f64; nv];
    for &[i0, i1, i2] in simplices {
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let cr = cross3(e01, e02);
        let face_area = 0.5 * norm3(cr);
        let third = face_area / 3.0;
        areas[i0] += third;
        areas[i1] += third;
        areas[i2] += third;
    }
    areas
}

impl SurfaceStokes {
    /// Build the Stokes solver, caching vertex coordinates, dual areas, and tangent frames.
    ///
    /// Uses the closed-manifold Poisson solver (pin vertex 0 + zero-mean projection).
    /// Correct for periodic/closed meshes (sphere, torus, flat torus).
    pub fn new<M: Manifold>(ops: &Operators<M, 3, 2>, mesh: &Mesh<M, 3, 2>) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();
        let poisson = PoissonSolver::new(ops)?;
        let coords = extract_coords(mesh);
        let dual_areas = compute_dual_areas(n_vertices, &mesh.simplices, &coords);
        let curvature = gaussian_curvature(n_vertices, &mesh.simplices, &coords, &dual_areas);
        let shift: Vec<f64> = curvature.iter().map(|k| 2.0 * k).collect();
        let outer = PoissonSolver::new_shifted(ops, &shift, &coords)?;
        let s1 = ops.hodge.star1();
        let star1: Vec<f64> = (0..s1.len()).map(|i| s1[i]).collect();
        let normals = compute_vertex_normals_stokes(&mesh.simplices, &coords);
        let e1_frames = compute_tangent_frames_stokes(&normals);
        Ok(Self {
            poisson, outer, n_vertices, coords, dual_areas, star1, normals, e1_frames,
            no_slip_vertices: Vec::new(), clamped: None, inner_cache: Default::default(),
        })
    }

    /// Build the Stokes solver for a **confined (bounded) domain** with no-slip BCs.
    ///
    /// Enforces ψ = 0 on all vertices in `boundary_vertices` via Dirichlet
    /// elimination in the Poisson solver. This gives the no-slip condition
    /// u = 0 on the domain boundary (the stream function is constant on each
    /// connected boundary component; setting it to zero on the single boundary
    /// of a simply-connected domain is the correct no-slip choice).
    ///
    /// Additionally, after velocity recovery from ψ, the velocity is explicitly
    /// zeroed at all boundary vertices. This enforces the strong no-slip condition
    /// at the discrete level, eliminating residual velocity from interior edges
    /// incident to boundary vertices.
    ///
    /// All other behaviour (vorticity source, velocity recovery) is identical
    /// to [`Self::new`]. The existing `new` is left unchanged for closed-manifold
    /// callers (sphere, torus, etc.).
    pub fn new_confined<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        boundary_vertices: &[usize],
    ) -> Result<Self, String> {
        let n_vertices = ops.laplace_beltrami.rows();
        let poisson = PoissonSolver::with_dirichlet(ops, boundary_vertices)?;
        let coords = extract_coords(mesh);
        let dual_areas = compute_dual_areas(n_vertices, &mesh.simplices, &coords);
        let s1 = ops.hodge.star1();
        let star1: Vec<f64> = (0..s1.len()).map(|i| s1[i]).collect();
        let normals = compute_vertex_normals_stokes(&mesh.simplices, &coords);
        let e1_frames = compute_tangent_frames_stokes(&normals);
        // A confined domain is planar here, so the angle defect vanishes and
        // the outer factor is the same Dirichlet operator as the inner one.
        let outer = PoissonSolver::with_dirichlet(ops, boundary_vertices)?;
        Ok(Self {
            poisson,
            outer,
            inner_cache: Default::default(),
            n_vertices,
            coords,
            dual_areas,
            star1,
            normals,
            e1_frames,
            no_slip_vertices: boundary_vertices.to_vec(),
            clamped: None,
        })
    }

    /// The biharmonic's inner Poisson solve, seeded from slot `slot`'s previous
    /// iterate and storing this one back.
    fn solve_inner(&self, slot: usize, rhs: &DVector<f64>, tol: f64) -> (DVector<f64>, usize) {
        let (inner, its) = {
            let seed = self.inner_cache[slot].borrow();
            let x0 = seed.as_ref().filter(|v| v.len() == self.n_vertices).map(|v| v.as_slice());
            self.poisson.solve_from(rhs, x0, tol)
        };
        *self.inner_cache[slot].borrow_mut() = Some(inner.as_slice().to_vec());
        (inner, its)
    }

    /// Solve Stokes, warm-started from a previous stream function.
    ///
    /// Returns the velocity, the stream function to carry into the next step,
    /// and the iterations taken. In a time-stepping loop the source moves by
    /// order `dt` per step, so the previous stream function is close to the
    /// answer and the iteration is short; started cold, the same solve repeats
    /// its whole descent every step and dominates the run.
    pub fn solve_warm<M: Manifold>(
        &self,
        q: &QField,
        params: &ActiveNematicParams,
        _ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        psi0: Option<&[f64]>,
        tol: f64,
    ) -> (VelocityField, Vec<f64>, usize) {
        let nv = self.n_vertices;
        let zeta = params.zeta_eff;
        let eta = params.eta;
        if zeta.abs() < 1e-30 || eta.abs() < 1e-30 {
            return (VelocityField::zeros(nv), vec![0.0; nv], 0);
        }
        let omega = compute_vorticity_source(
            q, zeta, eta, mesh, &self.coords, &self.dual_areas,
            &self.normals, &self.e1_frames,
        );
        // Two sequential Poisson solves, as in `solve`; see the note there for
        // why one is wrong. The inner solve starts cold, since the caller's
        // cached iterate is a stream function and not a vorticity.
        let (inner, its_inner) = self.solve_inner(0, &omega, tol);
        let (psi_ss, its_outer) = self.outer.solve_from(&inner, psi0, tol);
        let mut psi_v: Vec<f64> = psi_ss.iter().copied().collect();
        self.clamp(&mut psi_v, mesh);
        let psi = DVector::from_vec(psi_v);
        let its = its_inner + its_outer;
        let mut vel =
            velocity_from_psi(nv, &psi, mesh, &self.coords, &self.dual_areas, &self.star1);
        for &bv in &self.no_slip_vertices {
            vel.v[bv] = [0.0, 0.0, 0.0];
        }
        let psi_out: Vec<f64> = psi.iter().copied().collect();
        (vel, psi_out, its)
    }

    /// Solve Stokes for the active velocity field.
    pub fn solve<M: Manifold>(
        &self,
        q: &QField,
        params: &ActiveNematicParams,
        _ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
    ) -> VelocityField {
        let nv = self.n_vertices;
        let zeta = params.zeta_eff;
        let eta = params.eta;

        if zeta.abs() < 1e-30 || eta.abs() < 1e-30 {
            return VelocityField::zeros(nv);
        }

        // Compute vorticity source from active stress (covariant divergence).
        let omega = compute_vorticity_source(
            q, zeta, eta, mesh, &self.coords, &self.dual_areas,
            &self.normals, &self.e1_frames,
        );

        // Solve for the stream function. Steady Stokes is the BIHARMONIC
        //
        //     eta Delta^2 psi = curl f,
        //
        // which factors into two sequential Poisson solves. Applying the
        // Laplacian inverse ONCE solves eta Delta psi = curl f instead, and that
        // is the substrate-friction balance xi u = f - grad p with the viscosity
        // standing in for a friction coefficient. It is a different physics, and
        // it is short by one inverse Laplacian, so the long-wavelength response
        // is suppressed by the square of the forcing scale while the local
        // response survives. Measured against `flow-solver.py`'s own steady field
        // on `long_mg_s0`: the reference reaches |u|max 41.46 and mean 11.81, the
        // biharmonic 59.35 and 20.85, the single inversion 1.28 and 0.086. The
        // mean falls five times harder than the max, which is the domain-scale
        // circulation going missing, and that is the flow that transports
        // defects. Runs made with the single inversion read a trivial braid
        // because their defects barely move.
        //
        // Splitting the biharmonic into two Dirichlet Poisson solves imposes
        // psi = 0 and omega = 0 on the boundary, which is no flux plus the Lions
        // free-slip condition the reference derives for this system, rather than
        // the no-slip condition a molecular fluid would carry.
        let (vel, _psi) = self.stream_and_velocity(&omega, mesh);
        vel
    }

    /// Solve Stokes from an assembled stress rather than from `Q` alone.
    ///
    /// `sym1`, `sym2` and `anti` are the components described on
    /// [`compute_vorticity_source_from_stress`]. This is the entry point for the
    /// full Beris-Edwards stress, where the active part `-zeta Q` is only one of
    /// four contributions; [`Self::solve_warm`] keeps the active-only form for
    /// comparison against the runs made before the elastic terms were added.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_stress_warm<M: Manifold>(
        &self,
        sym1: &[f64],
        sym2: &[f64],
        anti: &[f64],
        eta: f64,
        mesh: &Mesh<M, 3, 2>,
        psi0: Option<&[f64]>,
        tol: f64,
    ) -> (VelocityField, Vec<f64>, usize) {
        if eta.abs() < 1e-30 {
            return (VelocityField::zeros(self.n_vertices), vec![0.0; self.n_vertices], 0);
        }
        let omega = compute_vorticity_source_from_stress(
            sym1, sym2, anti, eta, mesh, &self.coords, &self.dual_areas,
            &self.normals, &self.e1_frames,
        );
        let (inner, its_inner) = self.solve_inner(1, &omega, tol);
        let (psi_ss, its_outer) = self.outer.solve_from(&inner, psi0, tol);
        let mut psi_v: Vec<f64> = psi_ss.iter().copied().collect();
        self.clamp(&mut psi_v, mesh);
        let psi = DVector::from_vec(psi_v);
        let mut vel = velocity_from_psi(
            self.n_vertices, &psi, mesh, &self.coords, &self.dual_areas, &self.star1,
        );
        for &bv in &self.no_slip_vertices {
            vel.v[bv] = [0.0, 0.0, 0.0];
        }
        (vel, psi.iter().copied().collect(), its_inner + its_outer)
    }

    /// Solve Stokes from a nodal FORCE, with a symmetric operator.
    ///
    /// The stress-driven path assembles `curl(div Pi)` and inherits whatever
    /// discretisation that assembly happens to use, which need not be the
    /// transpose of the velocity recovery. This path is built the other way: the
    /// source is `curl^T f`, the exact transpose of [`velocity_from_psi`], so
    ///
    /// ```text
    /// <f, u> = <b, psi> = (1/eta) b^T S^-1 M S^-1 b >= 0
    /// ```
    ///
    /// for every `f`, with `S` the SPD stiffness and `M` the mass. The operator
    /// taking `f` to `u` is therefore symmetric positive semi-definite, which is
    /// what the discrete energy law needs downstream of the adjoint elastic force.
    ///
    /// In terms of the Poisson solver's own convention, `solve(rhs) = -S^-1 M rhs`,
    /// so `S^-1 M S^-1 b` is `solve(solve(b / M))`, which is the same double solve
    /// the biharmonic already uses with the source converted from a nodal
    /// functional to a pointwise density.
    ///
    /// The no-slip projection is applied to `f` as well as to `u`, since zeroing
    /// only the output would break the symmetry the construction exists for.
    pub fn solve_force_warm<M: Manifold>(
        &self,
        force: &[[f64; 2]],
        eta: f64,
        mesh: &Mesh<M, 3, 2>,
        psi0: Option<&[f64]>,
        tol: f64,
    ) -> (VelocityField, Vec<f64>, usize) {
        let nv = self.n_vertices;
        if eta.abs() < 1e-30 {
            return (VelocityField::zeros(nv), vec![0.0; nv], 0);
        }
        let mut f3: Vec<[f64; 3]> = force.iter().map(|v| [v[0], v[1], 0.0]).collect();
        for &bv in &self.no_slip_vertices {
            f3[bv] = [0.0, 0.0, 0.0];
        }
        let b = curl_transpose(nv, &f3, mesh, &self.coords);
        let mass = self.poisson.mass_diagonal();
        let mut rhs = DVector::zeros(nv);
        for i in 0..nv {
            rhs[i] = if mass[i].abs() > 1e-30 { b[i] / mass[i] } else { 0.0 };
        }
        let (inner, its_inner) = self.solve_inner(2, &rhs, tol);
        let (psi_raw, its_outer) = self.outer.solve_from(&inner, psi0, tol);
        let mut psi_v: Vec<f64> = (psi_raw / eta).iter().copied().collect();
        self.clamp(&mut psi_v, mesh);
        let psi = DVector::from_vec(psi_v);
        let mut vel = velocity_from_psi(
            nv, &psi, mesh, &self.coords, &self.dual_areas, &self.star1,
        );
        for &bv in &self.no_slip_vertices {
            vel.v[bv] = [0.0, 0.0, 0.0];
        }
        (vel, psi.iter().copied().collect(), its_inner + its_outer)
    }

    /// The pressure that belongs to the flow this solver returns.
    ///
    /// The stream function eliminates `p`, so nothing in the time stepping ever
    /// forms it. It is recovered here from the same assembled stress the
    /// velocity came from, by the Poisson problem described on
    /// [`pressure_rhs_from_force`], and it is defined up to a constant. The
    /// vector that comes back has the mean removed.
    ///
    /// `poisson` must be a CLOSED-mode solver, that is [`PoissonSolver::new`]
    /// with no Dirichlet vertices. The pressure has no Dirichlet data anywhere:
    /// its wall condition is the Neumann one the weak form imposes naturally,
    /// and pinning `p = 0` on the wall instead would impose a boundary layer
    /// that the physics does not have.
    pub fn pressure_from_stress<M: Manifold>(
        &self,
        sym1: &[f64],
        sym2: &[f64],
        anti: &[f64],
        mesh: &Mesh<M, 3, 2>,
        poisson: &PoissonSolver,
    ) -> Vec<f64> {
        let f = vertex_force_from_stress(
            sym1, sym2, anti, mesh, &self.coords, &self.normals, &self.e1_frames,
        );
        let rhs = pressure_rhs_from_force(&f, mesh, &self.coords, poisson.mass_diagonal());
        let sol = poisson.solve(&rhs);
        let mut p: Vec<f64> = sol.iter().copied().collect();
        // AREA WEIGHTED, which is the gauge the lattice pressure is fixed in.
        // An unweighted vertex mean is a mean over the
        // sampling rather than over the domain, and the boundary is sampled
        // more densely than the bulk, so it sits off the interior mean by a few
        // per cent of a standard deviation and would put the two solvers on
        // different zeros.
        let area = poisson.mass_diagonal();
        let total: f64 = area.iter().sum();
        let mean: f64 = (0..p.len()).map(|i| area[i] * p[i]).sum::<f64>() / total;
        for v in p.iter_mut() {
            *v -= mean;
        }
        p
    }

    /// Solve the biharmonic for the stream function from an assembled source,
    /// then recover the velocity. Returns both.
    ///
    /// `source` is `curl f / eta`, so this solves `Delta^2 psi = source`. Kept
    /// separate from [`Self::solve`] so a test can drive the production path
    /// with a source whose exact solution is known, which is the only kind of
    /// test that pins the operator: every smoke test that asserts a nonzero
    /// velocity passes just as happily for the single-inversion form this once
    /// carried. See `stokes_reproduces_the_biharmonic_manufactured_solution`.
    /// Build the solver for a CLAMPED, that is genuinely no-slip, wall.
    ///
    /// [`Self::new_confined`] imposes `psi = 0` and `Delta psi = 0`, which is
    /// free slip. This adds the correction described on [`ClampedCorrection`]
    /// so that `dpsi/dn = 0` as well.
    ///
    /// Setup costs two Poisson solves per boundary vertex and stores one stream
    /// function per boundary vertex, so it is `O(n_boundary)` in time and
    /// `n_boundary * n_vertices` in memory, paid once. Every time step afterwards
    /// costs the same two solves it did before, plus one dense product.
    pub fn new_confined_clamped<M: Manifold>(
        ops: &Operators<M, 3, 2>,
        mesh: &Mesh<M, 3, 2>,
        boundary_vertices: &[usize],
    ) -> Result<Self, String> {
        let mut solver = Self::new_confined(ops, mesh, boundary_vertices)?;
        let nv = solver.n_vertices;
        let nb = boundary_vertices.len();
        if nb == 0 {
            return Ok(solver);
        }
        let coords = solver.coords.clone();
        let (inv, _nrm) = gradient_frames(nv, mesh, &coords);

        // Inward direction: towards the centroid of the incident triangles.
        let mut acc = vec![[0.0_f64; 3]; nv];
        let mut cnt = vec![0.0_f64; nv];
        for &[i0, i1, i2] in &mesh.simplices {
            let c = [
                (coords[i0][0] + coords[i1][0] + coords[i2][0]) / 3.0,
                (coords[i0][1] + coords[i1][1] + coords[i2][1]) / 3.0,
                (coords[i0][2] + coords[i1][2] + coords[i2][2]) / 3.0,
            ];
            for &v in &[i0, i1, i2] {
                for k in 0..3 {
                    acc[v][k] += c[k] - coords[v][k];
                }
                cnt[v] += 1.0;
            }
        }
        let inward: Vec<[f64; 3]> = boundary_vertices
            .iter()
            .map(|&b| {
                let mut d = acc[b];
                if cnt[b] > 0.0 {
                    for k in 0..3 {
                        d[k] /= cnt[b];
                    }
                }
                let n = norm3(d);
                if n > 1e-30 { scale3(d, 1.0 / n) } else { [0.0; 3] }
            })
            .collect();

        let zero = DVector::zeros(nv);
        let mut phi: Vec<Vec<f64>> = Vec::with_capacity(nb);
        let mut lmat = nalgebra::DMatrix::<f64>::zeros(nb, nb);
        for (j, &_bj) in boundary_vertices.iter().enumerate() {
            let mut g = vec![0.0_f64; nv];
            g[boundary_vertices[j]] = 1.0;
            // Discrete harmonic lift of a unit wall value of omega, then the
            // stream function it drives.
            let omega = solver.poisson.solve_with_boundary(&zero, &g, 1e-12);
            let phi_j = solver.poisson.solve(&omega);
            let pv: Vec<f64> = phi_j.iter().copied().collect();
            let grad = vertex_gradients(nv, &pv, mesh, &coords, &inv);
            for (i, &bi) in boundary_vertices.iter().enumerate() {
                lmat[(i, j)] = dot3(grad[bi], inward[i]);
            }
            phi.push(pv);
        }
        let sv = lmat.clone().singular_values();
        let (hi, lo) = (sv[0], sv[sv.len() - 1]);
        let cond = if lo > 0.0 { hi / lo } else { f64::INFINITY };
        let lu = lmat.clone().lu();
        let spectrum: Vec<f64> = sv.iter().copied().collect();
        solver.clamped = Some(ClampedCorrection {
            boundary: boundary_vertices.to_vec(),
            cond,
            spectrum,
            phi,
            lu,
            lmat,
            inward,
        });
        Ok(solver)
    }

    /// Whether this solver imposes the clamped wall.
    /// Condition number of the clamped wall's boundary response matrix, or
    /// `None` for a simply supported wall, which needs no such solve.
    pub fn clamped_condition(&self) -> Option<f64> {
        self.clamped.as_ref().map(|c| c.cond)
    }

    /// Singular values of the clamped wall's boundary response matrix, largest
    /// first.
    pub fn clamped_spectrum(&self) -> Option<&[f64]> {
        self.clamped.as_ref().map(|c| c.spectrum.as_slice())
    }

    pub fn is_clamped(&self) -> bool {
        self.clamped.is_some()
    }

    /// Apply the clamped correction to a simply supported stream function.
    fn clamp<M: Manifold>(&self, psi: &mut [f64], mesh: &Mesh<M, 3, 2>) {
        let Some(c) = self.clamped.as_ref() else { return };
        let nv = self.n_vertices;
        let (inv, _) = gradient_frames(nv, mesh, &self.coords);
        let grad = vertex_gradients(nv, psi, mesh, &self.coords, &inv);
        let rhs = nalgebra::DVector::from_iterator(
            c.boundary.len(),
            c.boundary.iter().enumerate().map(|(i, &b)| -dot3(grad[b], c.inward[i])),
        );
        // A silent `return` here would drop the correction and leave the wall at
        // free slip, which is a different physical problem reported as success.
        // The matrix is rank deficient on every mesh measured (265 of 285 on the
        // production nephroid), so this branch is reachable: its null space maps
        // to the ZERO stream function, since a combination of the `phi_j` with
        // vanishing value and vanishing normal derivative on the wall is a
        // biharmonic field with zero Cauchy data, so any solution of the system
        // gives the same `psi`. The system is consistent and the factorisation
        // has met it so far, but if it ever fails, take the least-squares
        // solution rather than none.
        let coef = match c.lu.solve(&rhs) {
            Some(x) => x,
            None => match c.lmat.clone().svd(true, true).solve(&rhs, 1e-12) {
                Ok(x) => x,
                Err(_) => return,
            },
        };
        for (j, pj) in c.phi.iter().enumerate() {
            let a = coef[j];
            if a == 0.0 {
                continue;
            }
            for v in 0..nv {
                psi[v] += a * pj[v];
            }
        }
    }

    /// Dual areas, normals and tangent frames as built at construction.
    ///
    /// Exposed so a diagnostic can drive [`compute_vorticity_source`] on the
    /// solver's own geometry rather than rebuilding it and comparing two
    /// constructions.
    pub fn dual_areas(&self) -> &[f64] {
        &self.dual_areas
    }

    /// See [`Self::dual_areas`].
    pub fn normals(&self) -> &[[f64; 3]] {
        &self.normals
    }

    /// See [`Self::dual_areas`].
    pub fn e1_frames(&self) -> &[[f64; 3]] {
        &self.e1_frames
    }

    /// The stream function of a vorticity source, with no mesh needed.
    ///
    /// Solves `Delta (Delta + 2K) psi = er * source`, the surface biharmonic,
    /// as the two sequential Poisson solves it factors into. This is the entry
    /// point for a caller that has already formed its own vorticity, and it is
    /// what the trait backend uses.
    pub fn stream_from_vorticity(&self, source: &DVector<f64>, er: f64) -> DVector<f64> {
        let scaled = source * er;
        self.outer.solve(&self.poisson.solve(&scaled))
    }

    pub fn stream_and_velocity<M: Manifold>(
        &self,
        source: &DVector<f64>,
        mesh: &Mesh<M, 3, 2>,
    ) -> (VelocityField, Vec<f64>) {
        let mut psi_v: Vec<f64> = self.outer.solve(&self.poisson.solve(source))
            .iter().copied().collect();
        self.clamp(&mut psi_v, mesh);
        let psi = DVector::from_vec(psi_v);
        let mut vel = velocity_from_psi(
            self.n_vertices, &psi, mesh, &self.coords, &self.dual_areas, &self.star1,
        );
        // Zero velocity at boundary vertices. The discrete curl recovery spreads
        // contributions from interior edges onto boundary vertices, so it leaves
        // a small tangential velocity there even with psi = 0 on the boundary.
        //
        // Note that the two Dirichlet solves already impose psi = 0 and
        // omega = 0, which is no flux plus the Lions free-slip condition the
        // reference derives for this system. This zeroing therefore adds a third
        // condition the continuum problem does not impose, and it is retained
        // for continuity with earlier runs rather than because the physics asks
        // for it.
        for &bv in &self.no_slip_vertices {
            vel.v[bv] = [0.0, 0.0, 0.0];
        }
        (vel, psi.iter().copied().collect())
    }
}

/// Biharmonic source for the active stress alone, `Pi = -zeta Q`.
///
/// See [`compute_vorticity_source_from_stress`] for the sign, which is the whole
/// content of the 2026-08-20 fix.
///
/// Uses the covariant tensor divergence: Q at each vertex is expanded into
/// its 3D ambient representation using the vertex tangent frame (e1, e2),
/// then FEM gradients give the surface divergence on each triangle. This
/// correctly handles curved surfaces where the tangent frames differ between
/// vertices (the previous version used only global (x,y) components, which
/// fails away from the equator on a sphere).
#[allow(clippy::too_many_arguments)]
pub fn compute_vorticity_source<M: Manifold>(
    q: &QField,
    zeta: f64,
    eta: f64,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
    normals: &[[f64; 3]],
    e1_frames: &[[f64; 3]],
) -> DVector<f64> {
    // The active stress alone, `Pi = -zeta Q`, with no antisymmetric part.
    let sym1: Vec<f64> = q.q1.iter().map(|v| -zeta * v).collect();
    let sym2: Vec<f64> = q.q2.iter().map(|v| -zeta * v).collect();
    let anti = vec![0.0_f64; q.n_vertices];
    compute_vorticity_source_from_stress(
        &sym1, &sym2, &anti, eta, mesh, coords, dual_areas, normals, e1_frames,
    )
}

/// Biharmonic source `-curl(div Pi) / eta` from an assembled stress.
///
/// `sym1` and `sym2` are the two independent components of the symmetric
/// traceless part, `Pi_xx` and `Pi_xy`, and `anti` is the single component of
/// the antisymmetric part, `Pi_A,xy`. The part proportional to the identity is
/// omitted deliberately: it is a gradient and the pressure absorbs it, which is
/// why the reference splits its Ericksen stress and discards `Pi_I`.
///
/// # The sign
///
/// Steady Stokes with a body force `f = div Pi` is
///
/// ```text
/// -grad p + eta grad^2 u + f = 0,   so   eta grad^2 u = grad p - f.
/// ```
///
/// The two-dimensional curl kills the pressure, and with `u = n x grad psi` the
/// vorticity is `curl u = Delta psi`, so
///
/// ```text
/// eta Delta^2 psi = - curl f,
/// ```
///
/// and the source this returns, which the solver passes straight to
/// `Delta^2 psi = source`, is `-curl(div Pi) / eta`. The MINUS is the fix of
/// 2026-08-20. Until then the solver drove the biharmonic with `+curl f` and
/// every velocity it produced ran backwards.
///
/// The old sign was not caught because nothing tested it. `examples/dbg_source.rs`
/// compared the assembly against the exact `curl(div(-zeta Q))/eta` through
/// SUMS OF SQUARES, which is blind to an overall sign, and every other check on
/// this path asserted a magnitude, a ratio or a convergence rate. What it cost:
/// the backflow that should relax a distorted director instead reinforced it, so
/// the order parameter grew past its equilibrium instead of relaxing to it, the
/// active stress grew with it, and the run diverged. Measured against the lattice
/// reference for the SAME `Q` field, the mesh velocity had a median
/// `cos(u_mesh, u_ref)` of -0.935 over 4592 interior vertices before the fix.
///
/// Three checks pin it now, and they are independent of that derivation:
/// `a_force_does_positive_work_on_the_flow_it_drives`,
/// `the_stress_and_force_paths_drive_the_same_flow`, and
/// `the_biharmonic_source_has_the_sign_steady_stokes_gives`.
/// One triangle's vertices, FEM gradient bases, unit normal and lumping weight.
type TriGrad = ([usize; 3], [[f64; 3]; 3], [f64; 3], f64);

/// The vertex-lumped force `f = d_j T_kj` and the per-triangle gradient bases.
///
/// Split out of [`compute_vorticity_source_from_stress`] so the force is
/// available on its own. The curl of it drives the flow; the divergence of the
/// same field is what the pressure solves against, and a pressure recovered
/// from a separately assembled force would answer a different question.
#[allow(clippy::too_many_arguments)]
fn assemble_vertex_force<M: Manifold>(
    sym1: &[f64],
    sym2: &[f64],
    anti: &[f64],
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    normals: &[[f64; 3]],
    e1_frames: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<TriGrad>) {
    let nv = sym1.len();
    let mut fvert = vec![[0.0_f64; 3]; nv];
    let mut fw = vec![0.0_f64; nv];
    let mut tri_grads: Vec<TriGrad> = Vec::with_capacity(mesh.simplices.len());

    for &[i0, i1, i2] in &mesh.simplices {
        let p0 = coords[i0]; let p1 = coords[i1]; let p2 = coords[i2];
        let e01 = sub3(p1, p0);
        let e02 = sub3(p2, p0);
        let e12 = sub3(p2, p1);
        let e20 = sub3(p0, p2);

        let fn_vec = cross3(e01, e02);
        let area2 = norm3(fn_vec);
        if area2 < 1e-30 { continue; }

        let fn_hat = scale3(fn_vec, 1.0 / area2);
        let inv_2a = 1.0 / area2;

        // FEM gradient basis functions: grad(phi_a) = (n x e_opp) / (2*area).
        let grad_phi = [
            scale3(cross3(fn_hat, e12), inv_2a),
            scale3(cross3(fn_hat, e20), inv_2a),
            scale3(cross3(fn_hat, e01), inv_2a),
        ];

        let verts = [i0, i1, i2];

        // Compute 3D active force f = -zeta * div(Q) on this face.
        //
        // Q at vertex a in 3D ambient:
        //   Q_{kj} = q1_a*(e1_k*e1_j - e2_k*e2_j) + q2_a*(e1_k*e2_j + e2_k*e1_j)
        //
        // div(Q)_k = sum_a sum_j grad_phi_a[j] * Q_{kj}(a)
        //          = sum_a [q1_a*(e1_k*g1 - e2_k*g2) + q2_a*(e1_k*g2 + e2_k*g1)]
        //
        // where g1 = dot(grad_phi_a, e1_a), g2 = dot(grad_phi_a, e2_a).
        let mut f = [0.0_f64; 3];
        for local in 0..3 {
            let vi = verts[local];
            let e1 = e1_frames[vi];
            let e2 = cross3(normals[vi], e1);

            let g1 = dot3(grad_phi[local], e1);
            let g2 = dot3(grad_phi[local], e2);

            // The stress at this vertex, expanded into the ambient frame:
            //
            //   T_kj = s1 (e1_k e1_j - e2_k e2_j)      symmetric traceless
            //        + s2 (e1_k e2_j + e2_k e1_j)
            //        +  a (e1_k e2_j - e2_k e1_j)      antisymmetric
            //
            // so T_xy = s2 + a and T_yx = s2 - a, which is the split the
            // reference carries as `Pi_S` and `Pi_A`. The force is
            // f_k = d_j T_kj, in that index order, so the antisymmetric part
            // does not drop out.
            let (s1v, s2v, av) = (sym1[vi], sym2[vi], anti[vi]);

            for k in 0..3 {
                f[k] += s1v * (e1[k] * g1 - e2[k] * g2)
                    + s2v * (e1[k] * g2 + e2[k] * g1)
                    + av * (e1[k] * g2 - e2[k] * g1);
            }
        }

        // `f` is CONSTANT on this triangle, since `Q` is piecewise linear, so
        // its circulation around the triangle is identically zero:
        // `e01 + e12 + e20 = 0`. Taking differences of the individual edge terms
        // instead accumulates `0.5 f . (p1 + p2 - 2 p0)`, which scales as `h`
        // and, divided by a dual area of order `h^2`, gives a source that
        // DIVERGES as the mesh refines. Measured against the exact
        // `curl(div(-zeta Q))/eta` for `q1 = cos(kx) cos(ky)`, that form
        // returned 2.01, 4.16 and 5.55 times the right answer at spacings 0.08,
        // 0.04 and 0.02.
        //
        // The curl of a piecewise-constant field lives in the jumps between
        // triangles, so `f` is lumped to the vertices first and its curl taken
        // from the same FEM gradients that produced it. Both steps are lumped
        // with the barycentric weight `A_T/3`, which is the dual area this
        // module already uses.
        let wt = area2 / 6.0;
        for &v in &verts {
            for k in 0..3 {
                fvert[v][k] += wt * f[k];
            }
            fw[v] += wt;
        }
        tri_grads.push((verts, grad_phi, fn_hat, wt));
    }

    // Lump the force onto the vertices.
    for v in 0..nv {
        if fw[v] > 1e-30 {
            for k in 0..3 {
                fvert[v][k] /= fw[v];
            }
        }
    }
    (fvert, tri_grads)
}

/// The vertex force `f = div T` from an assembled Beris-Edwards stress.
#[allow(clippy::too_many_arguments)]
pub fn vertex_force_from_stress<M: Manifold>(
    sym1: &[f64],
    sym2: &[f64],
    anti: &[f64],
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    normals: &[[f64; 3]],
    e1_frames: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    assemble_vertex_force(sym1, sym2, anti, mesh, coords, normals, e1_frames).0
}

#[allow(clippy::too_many_arguments)]
pub fn compute_vorticity_source_from_stress<M: Manifold>(
    sym1: &[f64],
    sym2: &[f64],
    anti: &[f64],
    eta: f64,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    dual_areas: &[f64],
    normals: &[[f64; 3]],
    e1_frames: &[[f64; 3]],
) -> DVector<f64> {
    let nv = sym1.len();
    let mut omega = vec![0.0_f64; nv];
    let (fvert, tri_grads) =
        assemble_vertex_force(sym1, sym2, anti, mesh, coords, normals, e1_frames);

    // Curl of the now-continuous force, per triangle, lumped back to vertices.
    let mut ow = vec![0.0_f64; nv];
    for (verts, grad_phi, fn_hat, wt) in tri_grads {
        let mut curl = 0.0;
        for local in 0..3 {
            curl += dot3(fn_hat, cross3(grad_phi[local], fvert[verts[local]]));
        }
        for &v in &verts {
            omega[v] += wt * curl;
            ow[v] += wt;
        }
    }

    for i in 0..nv {
        let w = if ow[i] > 1e-30 { ow[i] } else { dual_areas[i] };
        if w > 1e-30 {
            // The minus is the equation, not the assembly: `omega` holds
            // `curl(div Pi)` at this point and the biharmonic wants `-curl f`.
            omega[i] /= -eta * w;
        }
    }

    DVector::from_vec(omega)
}

/// Vorticity of the recovered flow, `omega = curl u`.
///
/// The velocity is `u = grad^perp psi`, so `curl u = Delta psi` identically and
/// the vorticity needs no differencing of `u`. That matters here: `u` is
/// recovered as a discrete curl through edge fluxes over dual areas, and
/// differencing it again on a graded mesh chains two operators whose product
/// converges at `O(h^0.4)`, which is the same failure the co-rotational term
/// has and the reason it is taken from `psi` as well.
///
/// `Operators::laplace_beltrami` stores `L = -Delta`, so the sign is flipped
/// here.
pub fn vorticity_from_psi<M: Manifold>(psi: &[f64], ops: &Operators<M, 3, 2>) -> Vec<f64> {
    let n = psi.len();
    let mut out = vec![0.0_f64; n];
    for (col, column) in ops.laplace_beltrami.outer_iterator().enumerate() {
        let x = psi[col];
        if x == 0.0 {
            continue;
        }
        for (row, &v) in column.iter() {
            out[row] -= v * x;
        }
    }
    out
}

/// Right-hand side of the pressure Poisson problem, in the convention
/// [`PoissonSolver::solve`] reads.
///
/// A stream-function Stokes solve eliminates the pressure. `psi` absorbs the
/// whole incompressible response and `p` never appears, so recovering it is a
/// separate problem. Taking the divergence of steady Stokes,
///
/// ```text
/// 0 = -grad p + eta Delta u + f,    div u = 0
/// ```
///
/// leaves `Delta p = div f` with no dependence on `u`, since
/// `div(Delta u) = Delta(div u) = 0`. The pressure is the gradient part of the
/// Helmholtz split of the force, and the wall condition that belongs to it is
/// `dp/dn = f . n`.
///
/// Both are imposed at once by the weak form: find `p` with
///
/// ```text
/// int grad phi . grad p = int grad phi . f     for every test function phi
/// ```
///
/// whose natural boundary condition is `dp/dn = f . n`. The stiffness
/// `S = d0^T star1 d0` is the left side, so the right side is assembled per
/// triangle as `A_T grad phi_i . f_T` with `f_T` the barycentric mean of the
/// vertex force. `grad phi_i` is constant on a triangle, so that quadrature is
/// exact for a piecewise linear force.
///
/// `PoissonSolver::solve` forms `b = -star0 * rhs` and solves `S x = b`, which
/// is how it reads `Delta x = rhs`. This problem therefore needs
/// `rhs_i = -b_i / star0_i`, and that is what comes back. The solver's closed
/// mode projects out the constant, which is the gauge freedom `p` has anyway.
pub fn pressure_rhs_from_force<M: Manifold>(
    fvert: &[[f64; 3]],
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    star0: &[f64],
) -> DVector<f64> {
    let nv = fvert.len();
    let mut b = vec![0.0_f64; nv];

    for &[i0, i1, i2] in &mesh.simplices {
        let p0 = coords[i0];
        let p1 = coords[i1];
        let p2 = coords[i2];
        let e01 = sub3(p1, p0);
        let e02 = sub3(p2, p0);
        let e12 = sub3(p2, p1);
        let e20 = sub3(p0, p2);

        let fn_vec = cross3(e01, e02);
        let area2 = norm3(fn_vec);
        if area2 < 1e-30 {
            continue;
        }
        let fn_hat = scale3(fn_vec, 1.0 / area2);
        let inv_2a = 1.0 / area2;
        let grad_phi = [
            scale3(cross3(fn_hat, e12), inv_2a),
            scale3(cross3(fn_hat, e20), inv_2a),
            scale3(cross3(fn_hat, e01), inv_2a),
        ];

        let verts = [i0, i1, i2];
        let mut f_tri = [0.0_f64; 3];
        for &v in &verts {
            for k in 0..3 {
                f_tri[k] += fvert[v][k] / 3.0;
            }
        }
        // `area2` is twice the area, which is what the gradient basis divides by.
        let area = 0.5 * area2;
        for local in 0..3 {
            b[verts[local]] += area * dot3(grad_phi[local], f_tri);
        }
    }

    let mut rhs = vec![0.0_f64; nv];
    for i in 0..nv {
        if star0[i] > 1e-30 {
            rhs[i] = -b[i] / star0[i];
        }
    }
    DVector::from_vec(rhs)
}

/// Per-vertex normal and the inverse of the least-squares gradient matrix.
///
/// `A_v = sum_{e incident v} t_e t_e^T + n_v n_v^T`, whose inverse maps the
/// accumulated edge data to a tangential gradient. Shared by
/// [`velocity_from_psi`] and [`curl_transpose`] so the two cannot drift apart,
/// which is the whole content of the transpose relation.
fn gradient_frames<M: Manifold>(
    nv: usize,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> (Vec<[f64; 9]>, Vec<[f64; 3]>) {
    let mut mm = vec![[0.0_f64; 9]; nv];
    let mut nn = vec![[0.0_f64; 3]; nv];
    for e in 0..mesh.n_boundaries() {
        let [v0, v1] = mesh.boundaries[e];
        let edge = sub3(coords[v1], coords[v0]);
        let edge_len = norm3(edge);
        if edge_len < 1e-30 {
            continue;
        }
        let t = scale3(edge, 1.0 / edge_len);
        let fn_hat = average_edge_normal(e, mesh, coords);
        for &v in &[v0, v1] {
            for i in 0..3 {
                nn[v][i] += fn_hat[i];
                for j in 0..3 {
                    mm[v][3 * i + j] += t[i] * t[j];
                }
            }
        }
    }
    let mut inv = vec![[0.0_f64; 9]; nv];
    let mut nrm = vec![[0.0_f64; 3]; nv];
    for v in 0..nv {
        let n_len = norm3(nn[v]);
        if n_len < 1e-30 {
            continue;
        }
        let n_hat = scale3(nn[v], 1.0 / n_len);
        nrm[v] = n_hat;
        let mut a = mm[v];
        for i in 0..3 {
            for j in 0..3 {
                a[3 * i + j] += n_hat[i] * n_hat[j];
            }
        }
        let c00 = a[4] * a[8] - a[5] * a[7];
        let c01 = a[5] * a[6] - a[3] * a[8];
        let c02 = a[3] * a[7] - a[4] * a[6];
        let det = a[0] * c00 + a[1] * c01 + a[2] * c02;
        if det.abs() < 1e-30 {
            continue;
        }
        // A is symmetric, so the inverse is the adjugate over the determinant
        // and is symmetric too.
        inv[v] = [
            c00 / det,
            (a[2] * a[7] - a[1] * a[8]) / det,
            (a[1] * a[5] - a[2] * a[4]) / det,
            c01 / det,
            (a[0] * a[8] - a[2] * a[6]) / det,
            (a[2] * a[3] - a[0] * a[5]) / det,
            c02 / det,
            (a[1] * a[6] - a[0] * a[7]) / det,
            (a[0] * a[4] - a[1] * a[3]) / det,
        ];
    }
    (inv, nrm)
}

/// Vertex gradient of a nodal scalar, by the same least-squares frame the
/// velocity recovery uses. Exposed for diagnostics only: it is how the
/// divergence of a recovered velocity is measured against the recovery's own
/// operator rather than against an unrelated one.
pub fn debug_vertex_gradient<M: Manifold>(f: &[f64], mesh: &Mesh<M, 3, 2>) -> Vec<[f64; 3]> {
    let nv = mesh.n_vertices();
    let coords = extract_coords(mesh);
    let (inv, _n) = gradient_frames(nv, mesh, &coords);
    vertex_gradients(nv, f, mesh, &coords, &inv)
}

fn apply3(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// The exact transpose of [`velocity_from_psi`].
///
/// Given a nodal force `f`, returns `b` with `<f, u> = <b, psi>` for every
/// `psi`, where `u = velocity_from_psi(psi)`. This is what lets a force-driven
/// Stokes solve have a symmetric operator, and with it a discrete energy law.
///
/// With the least-squares recovery `u_v = n_v x A_v^-1 sum_e t_e dpsi_e/|e|`,
/// the scalar triple product gives
///
/// ```text
///   <f, u> = sum_v b_v . w_v,      w_v = -A_v^-1 (n_v x f_v)
/// ```
///
/// so each edge contributes `kappa = t_e . (w_v0 + w_v1) / |e|`, differenced
/// along the edge exactly as the forward map differences `psi`.
fn curl_transpose<M: Manifold>(
    nv: usize,
    f: &[[f64; 3]],
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> DVector<f64> {
    let (inv, nrm) = gradient_frames(nv, mesh, coords);
    let w: Vec<[f64; 3]> = (0..nv)
        .map(|v| {
            let nf = cross3(nrm[v], f[v]);
            let a = apply3(&inv[v], nf);
            [-a[0], -a[1], -a[2]]
        })
        .collect();

    let mut b = vec![0.0_f64; nv];
    for e in 0..mesh.n_boundaries() {
        let [v0, v1] = mesh.boundaries[e];
        let edge = sub3(coords[v1], coords[v0]);
        let edge_len = norm3(edge);
        if edge_len < 1e-30 {
            continue;
        }
        let t = scale3(edge, 1.0 / edge_len);
        let kappa = (dot3(t, w[v0]) + dot3(t, w[v1])) / edge_len;
        b[v1] += kappa;
        b[v0] -= kappa;
    }
    DVector::from_vec(b)
}

/// Recover the velocity `u = n x grad psi` from the stream function.
///
/// The gradient at a vertex is taken in LEAST SQUARES from its incident edges.
/// Each edge supplies one directional derivative, `t_e . grad psi = dpsi/|e|`,
/// so the fan of edges around a vertex overdetermines a two-component gradient
/// and the normal equations give it:
///
/// ```text
///   M = sum_e t_e t_e^T,   b = sum_e t_e dpsi_e/|e|,   grad psi = M^-1 b
/// ```
///
/// `M` is rank two on a surface, so `n n^T` is added to it and the solve then
/// returns the tangential gradient with nothing along the normal.
///
/// **This replaces an average of directional derivatives, which was wrong by a
/// factor of about four.** That form accumulated
/// `(1/valence) sum_e 0.5 (n x t_e)(t_e . grad psi)`, and for an isotropic fan
/// `sum_e t_e t_e^T = (valence/2) I`, so it collapsed to `(1/4) n x grad psi`.
/// Being an average rather than a solve, it never converged: measured against
/// the manufactured biharmonic solution the ratio to the exact peak velocity
/// was 0.262, 0.275, 0.307 and 0.266 as the spacing fell from 0.16 to 0.02,
/// scattering about a quarter with the residual set by mesh anisotropy rather
/// than by resolution. Every ratio test in this file passed throughout, because
/// a constant factor cancels in a ratio; see
/// `stokes_velocity_magnitude_matches_the_manufactured_solution`, which pins the
/// number instead.
/// Least-squares gradient of a nodal scalar, using precomputed frames.
fn vertex_gradients<M: Manifold>(
    nv: usize,
    f: &[f64],
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    inv: &[[f64; 9]],
) -> Vec<[f64; 3]> {
    let mut bb = vec![[0.0_f64; 3]; nv];
    for e in 0..mesh.n_boundaries() {
        let [v0, v1] = mesh.boundaries[e];
        let edge = sub3(coords[v1], coords[v0]);
        let edge_len = norm3(edge);
        if edge_len < 1e-30 {
            continue;
        }
        let t = scale3(edge, 1.0 / edge_len);
        let d = (f[v1] - f[v0]) / edge_len;
        for &v in &[v0, v1] {
            for i in 0..3 {
                bb[v][i] += t[i] * d;
            }
        }
    }
    (0..nv).map(|v| apply3(&inv[v], bb[v])).collect()
}

fn velocity_from_psi<M: Manifold>(
    nv: usize,
    psi: &DVector<f64>,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
    _dual_areas: &[f64],
    _star1: &[f64],
) -> VelocityField {
    let (inv, nrm) = gradient_frames(nv, mesh, coords);
    let psiv: Vec<f64> = psi.iter().copied().collect();
    let g = vertex_gradients(nv, &psiv, mesh, coords, &inv);
    let vel: Vec<[f64; 3]> = (0..nv).map(|v| cross3(nrm[v], g[v])).collect();
    VelocityField { v: vel, n_vertices: nv }
}

/// Advect Q along velocity: computes (u · grad Q) at each vertex.
///
/// Uses edge-based directional derivative: for each edge [v0, v1],
/// the advective flux is (u · edge_tangent) * (Q_v1 - Q_v0) / |e|^2,
/// distributed to both vertices.
pub fn advect_q(
    q: &QField,
    vel: &VelocityField,
    mesh_boundaries: &[[usize; 2]],
    vertex_boundaries: &[Vec<usize>],
    coords: &[[f64; 3]],
) -> QField {
    let nv = q.n_vertices;
    let _ = vertex_boundaries;

    // Least squares gradient, not an average of directional derivatives.
    //
    // The previous form accumulated `(u . e_hat)(dQ/d|e|)` per incident edge and
    // divided by the valence. That is `u^T A_v grad Q` with
    // `A_v = (1/n) sum_e e_hat e_hat^T`, which is `I/2` on an isotropic fan and
    // so returns HALF of `u . grad Q`, and on a real fan is an anisotropic tensor
    // that does not approach `I/2` under refinement. Measured on a disc by
    // `examples/dbg_advect.rs`, the ratio to the exact answer sat at 0.50 across
    // a factor of eight in spacing, with `A_v` a fixed 0.236 from `I/2` in
    // Frobenius norm. The scheme was therefore inconsistent, not merely
    // inaccurate, and the anisotropy is what stopped the operator being skew.
    //
    // Solving `A_v grad Q = b_v` instead inverts exactly that tensor, which is
    // the same correction `velocity_from_psi` needed for the curl.
    let mut a = vec![[0.0_f64; 3]; nv]; // [xx, xy, yy], symmetric in plane
    let mut b1 = vec![[0.0_f64; 2]; nv];
    let mut b2 = vec![[0.0_f64; 2]; nv];
    for &[v0, v1] in mesh_boundaries {
        let edge = sub3(coords[v1], coords[v0]);
        let len = norm3(edge);
        if len < 1e-30 {
            continue;
        }
        let t = [edge[0] / len, edge[1] / len];
        let d1 = (q.q1[v1] - q.q1[v0]) / len;
        let d2 = (q.q2[v1] - q.q2[v0]) / len;
        for &v in &[v0, v1] {
            a[v][0] += t[0] * t[0];
            a[v][1] += t[0] * t[1];
            a[v][2] += t[1] * t[1];
            b1[v][0] += t[0] * d1;
            b1[v][1] += t[1] * d1;
            b2[v][0] += t[0] * d2;
            b2[v][1] += t[1] * d2;
        }
    }

    let mut adv_q1 = vec![0.0; nv];
    let mut adv_q2 = vec![0.0; nv];
    for v in 0..nv {
        let det = a[v][0] * a[v][2] - a[v][1] * a[v][1];
        if det.abs() < 1e-30 {
            continue;
        }
        let solve = |b: [f64; 2]| {
            [
                (a[v][2] * b[0] - a[v][1] * b[1]) / det,
                (a[v][0] * b[1] - a[v][1] * b[0]) / det,
            ]
        };
        let g1 = solve(b1[v]);
        let g2 = solve(b2[v]);
        let u = vel.v[v];
        adv_q1[v] = u[0] * g1[0] + u[1] * g1[1];
        adv_q2[v] = u[0] * g2[0] + u[1] * g2[1];
    }

    QField { q1: adv_q1, q2: adv_q2, n_vertices: nv }
}

/// Covariant advection: computes (u . grad Q) with parallel transport.
///
/// Unlike [`advect_q`], this function parallel-transports Q along each edge
/// before computing directional derivatives, correctly handling the
/// frame-dependence of Q-tensor components on curved surfaces.
///
/// `edge_phases[e]` is the spin-2 connection phase for `mesh_boundaries[e]`,
/// obtained from [`crate::connection_laplacian::ConnectionLaplacian::edge_phases`].
pub fn advect_q_covariant(
    q: &QField,
    vel: &VelocityField,
    mesh_boundaries: &[[usize; 2]],
    vertex_boundaries: &[Vec<usize>],
    coords: &[[f64; 3]],
    edge_phases: &[f64],
) -> QField {
    let nv = q.n_vertices;
    let _ = vertex_boundaries;

    // Least squares gradient in each vertex's OWN frame, for the same reason
    // [`advect_q`] needs one: dividing a sum of directional derivatives by the
    // valence returns `u^T A_v grad Q` rather than `u . grad Q`, which is half
    // the answer on an isotropic fan and an anisotropic tensor on a real one.
    // Parallel transport changes what is differenced along each edge, not how
    // the directional derivatives are assembled into a gradient.
    //
    // The tangent used at `v1` points from `v1` towards `v0`, so it is `-t`,
    // and the derivative along it is `-dq_at_v1 / len`. The two signs cancel in
    // `b`, and `A` is unchanged because it is quadratic in the tangent.
    let mut a = vec![[0.0_f64; 3]; nv]; // [xx, xy, yy]
    let mut b1 = vec![[0.0_f64; 2]; nv];
    let mut b2 = vec![[0.0_f64; 2]; nv];

    for (e, &[v0, v1]) in mesh_boundaries.iter().enumerate() {
        let edge = sub3(coords[v1], coords[v0]);
        let len = norm3(edge);
        if len < 1e-30 {
            continue;
        }
        let t = [edge[0] / len, edge[1] / len];

        let phase = edge_phases[e];
        let (cos_p, sin_p) = (phase.cos(), phase.sin());

        // Transport Q from v1 into v0's frame, then difference there.
        let q1_v1_in_v0 = cos_p * q.q1[v1] + sin_p * q.q2[v1];
        let q2_v1_in_v0 = -sin_p * q.q1[v1] + cos_p * q.q2[v1];
        let d1_v0 = (q1_v1_in_v0 - q.q1[v0]) / len;
        let d2_v0 = (q2_v1_in_v0 - q.q2[v0]) / len;

        // Transport Q from v0 into v1's frame, then difference there.
        let q1_v0_in_v1 = cos_p * q.q1[v0] - sin_p * q.q2[v0];
        let q2_v0_in_v1 = sin_p * q.q1[v0] + cos_p * q.q2[v0];
        let d1_v1 = (q.q1[v1] - q1_v0_in_v1) / len;
        let d2_v1 = (q.q2[v1] - q2_v0_in_v1) / len;

        for (v, d1, d2) in [(v0, d1_v0, d2_v0), (v1, d1_v1, d2_v1)] {
            a[v][0] += t[0] * t[0];
            a[v][1] += t[0] * t[1];
            a[v][2] += t[1] * t[1];
            b1[v][0] += t[0] * d1;
            b1[v][1] += t[1] * d1;
            b2[v][0] += t[0] * d2;
            b2[v][1] += t[1] * d2;
        }
    }

    let mut adv_q1 = vec![0.0; nv];
    let mut adv_q2 = vec![0.0; nv];
    for v in 0..nv {
        let det = a[v][0] * a[v][2] - a[v][1] * a[v][1];
        if det.abs() < 1e-30 {
            continue;
        }
        let solve = |b: [f64; 2]| {
            [
                (a[v][2] * b[0] - a[v][1] * b[1]) / det,
                (a[v][0] * b[1] - a[v][1] * b[0]) / det,
            ]
        };
        let g1 = solve(b1[v]);
        let g2 = solve(b2[v]);
        let u = vel.v[v];
        adv_q1[v] = u[0] * g1[0] + u[1] * g1[1];
        adv_q2[v] = u[0] * g2[0] + u[1] * g2[1];
    }

    QField { q1: adv_q1, q2: adv_q2, n_vertices: nv }
}

fn average_edge_normal<M: Manifold>(
    edge_idx: usize,
    mesh: &Mesh<M, 3, 2>,
    coords: &[[f64; 3]],
) -> [f64; 3] {
    let mut n = [0.0_f64; 3];
    for &fi in &mesh.boundary_simplices[edge_idx] {
        let [i0, i1, i2] = mesh.simplices[fi];
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let cr = cross3(e01, e02);
        n = add3(n, cr);
    }
    let len = norm3(n);
    if len > 1e-14 { scale3(n, 1.0 / len) } else { [0.0, 0.0, 1.0] }
}

/// Compute area-weighted vertex normals from a triangle mesh.
fn compute_vertex_normals_stokes(simplices: &[[usize; 3]], coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let nv = coords.len();
    let mut normals = vec![[0.0_f64; 3]; nv];
    for &[i0, i1, i2] in simplices {
        let e01 = sub3(coords[i1], coords[i0]);
        let e02 = sub3(coords[i2], coords[i0]);
        let fn_vec = cross3(e01, e02);
        normals[i0] = add3(normals[i0], fn_vec);
        normals[i1] = add3(normals[i1], fn_vec);
        normals[i2] = add3(normals[i2], fn_vec);
    }
    for n in &mut normals {
        let len = norm3(*n);
        if len > 1e-14 {
            *n = scale3(*n, 1.0 / len);
        }
    }
    normals
}

/// Compute per-vertex tangent frame e1 from normals (e2 = n x e1).
///
/// Uses the same algorithm as `ConnectionLaplacian` so that the frames are
/// consistent with the covariant Laplacian used by the molecular field.
fn compute_tangent_frames_stokes(normals: &[[f64; 3]]) -> Vec<[f64; 3]> {
    normals.iter().map(|n| {
        let ref_dir = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let d = dot3(*n, ref_dir);
        let t = [ref_dir[0] - d * n[0], ref_dir[1] - d * n[1], ref_dir[2] - d * n[2]];
        let len = norm3(t);
        if len > 1e-14 { scale3(t, 1.0 / len) } else { [1.0, 0.0, 0.0] }
    }).collect()
}

// Vector helpers (pub(crate) for use by curved_stokes).
pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
pub(crate) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s, a[1]*s, a[2]*s] }
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
pub(crate) fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::mesh::FlatMesh;
    use cartan_manifolds::euclidean::Euclidean;

    #[test]
    fn stokes_zero_activity_zero_velocity() {
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 0.0;

        let nv = mesh.n_vertices();
        let q = QField::random_perturbation(nv, 0.5, 42);

        let solver = SurfaceStokes::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.v.iter().map(|[x, y, z]| x.abs() + y.abs() + z.abs()).sum();
        assert!(v_norm < 1e-12, "zero activity should give zero velocity");
    }

    #[test]
    fn stokes_nonzero_activity_nonzero_velocity() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 1.0;

        let nv = mesh.n_vertices();
        let q = QField::random_perturbation(nv, 0.3, 42);

        let solver = SurfaceStokes::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_norm: f64 = v.v.iter()
            .map(|[x, y, z]| x * x + y * y + z * z)
            .sum::<f64>()
            .sqrt();
        assert!(v_norm > 1e-6, "nonzero activity should give nonzero velocity, got {v_norm}");
    }

    #[test]
    fn stokes_solver_constructs() {
        let mesh = FlatMesh::unit_square_grid(8);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new(&ops, &mesh);
        assert!(solver.is_ok());
    }

    /// Build a disc mesh of radius `r` and return it with its boundary vertices.
    /// The clamped wall holds on the CUSPED mesh the runs use, where its
    /// boundary response matrix is rank deficient.
    ///
    /// `stokes_velocity_matches_the_clamped_solution` checks the disc, where the
    /// matrix is also rank deficient (110 of 125) but the domain is benign. The
    /// production nephroid is 265 of 285, and the condition number is reported as
    /// infinite on both, which reads as a broken solve. It is not: a combination
    /// of the `phi_j` whose wall value and wall normal derivative both vanish is a
    /// biharmonic field with zero Cauchy data, hence zero, so the null space maps
    /// to the zero stream function and every solution of the system gives the same
    /// `psi`. This test states that as an observable rather than as an argument.
    ///
    /// The free-slip comparison is what makes it a test rather than a tautology:
    /// on the same mesh and source the simply supported solve leaves a normal
    /// derivative of the same order as the gradient itself.
    #[test]
    fn the_pressure_recovers_the_potential_of_a_gradient_force() {
        // A stream-function solve never forms a pressure, so the recovery is a
        // separate solve with its own sign and its own scaling, and neither is
        // pinned by any test that only asks whether the answer looks like a
        // pressure. A force that IS a gradient has a known potential, and the
        // solve has to return it.
        use crate::confined::{Epitrochoid, MeshOpts, confined_mesh};
        let cm = confined_mesh(
            Epitrochoid { q: 2.0, d: 0.72, r: 53.071676 },
            MeshOpts { h_bulk: 2.0, h_min: 2.0, ..Default::default() },
        );
        let mesh = &cm.mesh;
        let ops = Operators::from_mesh(mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let coords = extract_coords(mesh);
        let poisson = PoissonSolver::new(&ops).expect("closed Poisson");
        let area = poisson.mass_diagonal().to_vec();
        let total: f64 = area.iter().sum();

        let k = 2.0 * std::f64::consts::PI / 53.071676;
        let phi: Vec<f64> =
            (0..nv).map(|i| (k * coords[i][0]).cos() * (k * coords[i][1]).cos()).collect();
        let f: Vec<[f64; 3]> = (0..nv)
            .map(|i| {
                let (x, y) = (coords[i][0], coords[i][1]);
                [-k * (k * x).sin() * (k * y).cos(), -k * (k * x).cos() * (k * y).sin(), 0.0]
            })
            .collect();

        let rhs = pressure_rhs_from_force(&f, mesh, &coords, &area);
        let sol = poisson.solve(&rhs);
        let mp: f64 = (0..nv).map(|i| area[i] * sol[i]).sum::<f64>() / total;
        let me: f64 = (0..nv).map(|i| area[i] * phi[i]).sum::<f64>() / total;
        let num: f64 =
            (0..nv).map(|i| area[i] * (sol[i] - mp - (phi[i] - me)).powi(2)).sum();
        let den: f64 = (0..nv).map(|i| area[i] * (phi[i] - me).powi(2)).sum();
        let err = (num / den).sqrt();
        // A sign error returns the potential negated, which is a relative error
        // of 2, so this bound separates the two by three orders of magnitude.
        assert!(err < 2e-2, "pressure recovery is off by {err:.3e}");
    }

    #[test]
    fn the_clamped_wall_holds_on_a_cusped_mesh() {
        use crate::confined::{Epitrochoid, MeshOpts, confined_mesh};
        let cm = confined_mesh(
            Epitrochoid { q: 2.0, d: 0.72, r: 53.071676 },
            MeshOpts { h_bulk: 2.0, h_min: 2.0, ..Default::default() },
        );
        let mesh = &cm.mesh;
        let ops = Operators::from_mesh(mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let bv = &cm.boundary_vertices;
        let coords = extract_coords(mesh);

        // Inward direction at each boundary vertex, towards the centroid of the
        // incident triangles, which is how the correction defines it.
        let mut acc = vec![[0.0_f64; 3]; nv];
        for t in 0..mesh.n_simplices() {
            let sx = mesh.simplices[t];
            let c = [
                (coords[sx[0]][0] + coords[sx[1]][0] + coords[sx[2]][0]) / 3.0,
                (coords[sx[0]][1] + coords[sx[1]][1] + coords[sx[2]][1]) / 3.0,
            ];
            for &v in &sx {
                acc[v][0] += c[0] - coords[v][0];
                acc[v][1] += c[1] - coords[v][1];
            }
        }

        let source = DVector::from_element(nv, -4.0);
        let measure = |solver: &SurfaceStokes| {
            let (_v, psi) = solver.stream_and_velocity(&source, mesh);
            let g = debug_vertex_gradient(&psi, mesh);
            let mut worst = 0.0_f64;
            for &b in bv {
                let n = (acc[b][0] * acc[b][0] + acc[b][1] * acc[b][1]).sqrt().max(1e-30);
                worst = worst.max((g[b][0] * acc[b][0] + g[b][1] * acc[b][1]).abs() / n);
            }
            let scale = (0..nv)
                .map(|i| (g[i][0] * g[i][0] + g[i][1] * g[i][1]).sqrt())
                .fold(0.0_f64, f64::max);
            (worst, scale)
        };

        let cl = SurfaceStokes::new_confined_clamped(&ops, mesh, bv).unwrap();
        assert!(cl.is_clamped());
        let sv = cl.clamped_spectrum().unwrap();
        let rank = sv.iter().filter(|&&x| x > 1e-10 * sv[0]).count();
        assert!(
            rank < bv.len(),
            "the fixture stopped exercising a rank-deficient matrix: rank {rank} of {}",
            bv.len()
        );
        let (clamped_dn, clamped_scale) = measure(&cl);
        assert!(
            clamped_dn < 1e-6 * clamped_scale,
            "clamped wall left dpsi/dn = {clamped_dn:.3e} against a gradient scale of \
             {clamped_scale:.3e}; rank {rank} of {}",
            bv.len()
        );

        let free = SurfaceStokes::new_confined(&ops, mesh, bv).unwrap();
        let (free_dn, free_scale) = measure(&free);
        assert!(
            free_dn > 0.1 * free_scale,
            "the free-slip solve should leave a normal derivative of the same order as the \
             gradient, got {free_dn:.3e} against {free_scale:.3e}; without that this test \
             would pass for a solver that produced no flow at all"
        );
        eprintln!(
            "cusped mesh, rank {rank} of {}: clamped dpsi/dn {clamped_dn:.3e}, \
             free slip {free_dn:.3e}",
            bv.len()
        );
    }

    /// Build the nodal force `f = div Pi` from an assembled stress, with the same
    /// P1 gradients the solver uses, lumped to vertices by area.
    ///
    /// `Pi_xx = s1`, `Pi_xy = Pi_yx = s2`, `Pi_yy = -s1`, so
    /// `f_x = dx s1 + dy s2` and `f_y = dx s2 - dy s1`.
    fn divergence_of_stress(
        s1: &[f64],
        s2: &[f64],
        xy: &[[f64; 2]],
        tris: &[[usize; 3]],
    ) -> Vec<[f64; 2]> {
        let nv = xy.len();
        let mut acc = vec![[0.0_f64; 2]; nv];
        let mut w = vec![0.0_f64; nv];
        for t in tris {
            let (p0, p1, p2) = (xy[t[0]], xy[t[1]], xy[t[2]]);
            let two_a = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let inv = 1.0 / two_a;
            let g = [
                [(p1[1] - p2[1]) * inv, (p2[0] - p1[0]) * inv],
                [(p2[1] - p0[1]) * inv, (p0[0] - p2[0]) * inv],
                [(p0[1] - p1[1]) * inv, (p1[0] - p0[0]) * inv],
            ];
            let (mut dxs1, mut dys1, mut dxs2, mut dys2) = (0.0, 0.0, 0.0, 0.0);
            for a in 0..3 {
                dxs1 += s1[t[a]] * g[a][0];
                dys1 += s1[t[a]] * g[a][1];
                dxs2 += s2[t[a]] * g[a][0];
                dys2 += s2[t[a]] * g[a][1];
            }
            let (fx, fy) = (dxs1 + dys2, dxs2 - dys1);
            let aw = 0.5 * two_a.abs();
            for a in 0..3 {
                acc[t[a]][0] += aw * fx;
                acc[t[a]][1] += aw * fy;
                w[t[a]] += aw;
            }
        }
        (0..nv)
            .map(|i| if w[i] > 1e-30 { [acc[i][0] / w[i], acc[i][1] / w[i]] } else { [0.0; 2] })
            .collect()
    }

    /// Fixture shared by the three sign tests: a cusped mesh, its solver, a
    /// smooth `Q` and the active stress it carries.
    #[allow(clippy::type_complexity)]
    fn sign_fixture() -> (
        crate::confined::ConfinedMesh2,
        SurfaceStokes,
        Vec<f64>,
        Vec<f64>,
        Vec<[f64; 2]>,
        f64,
    ) {
        use crate::confined::{Epitrochoid, MeshOpts, confined_mesh};
        let cm = confined_mesh(
            Epitrochoid { q: 2.0, d: 0.72, r: 53.071676 },
            MeshOpts { h_bulk: 2.0, h_min: 2.0, ..Default::default() },
        );
        let ops = Operators::from_mesh(&cm.mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &cm.mesh, &cm.boundary_vertices).unwrap();
        let nv = cm.mesh.n_vertices();
        let xy: Vec<[f64; 2]> =
            (0..nv).map(|i| [cm.mesh.vertices[i].x, cm.mesh.vertices[i].y]).collect();
        let tris: Vec<[usize; 3]> =
            (0..cm.mesh.n_simplices()).map(|t| cm.mesh.simplices[t]).collect();
        let (k, zeta) = (0.09_f64, 16384.0_f64);
        let s1: Vec<f64> =
            xy.iter().map(|c| -zeta * (k * c[0]).cos() * (k * c[1]).cos()).collect();
        let s2: Vec<f64> =
            xy.iter().map(|c| -zeta * (k * c[0]).sin() * (k * c[1]).sin()).collect();
        let f = divergence_of_stress(&s1, &s2, &xy, &tris);
        (cm, solver, s1, s2, f, 404.7715405015526)
    }

    /// A force does POSITIVE work on the flow it drives.
    ///
    /// This is the sign check that needs no derivation. The map from a nodal
    /// force to the velocity it drives is symmetric positive semi-definite, since
    /// [`SurfaceStokes::solve_force_warm`] builds its source as the exact
    /// transpose of the velocity recovery, so `<f, u> = <f, K f> >= 0` for every
    /// `f`. The fluid dissipates what the force delivers; a flipped sign makes the
    /// flow run backwards and the power negative for every `f`, which is a
    /// perpetual motion machine rather than a small numerical error.
    #[test]
    fn a_force_does_positive_work_on_the_flow_it_drives() {
        let (cm, solver, _s1, _s2, f, eta) = sign_fixture();
        let nv = cm.mesh.n_vertices();
        let (vel, _psi, _n) = solver.solve_force_warm(&f, eta, &cm.mesh, None, 1e-10);
        let da = solver.dual_areas();
        let power: f64 = (0..nv)
            .map(|i| da[i] * (f[i][0] * vel.v[i][0] + f[i][1] * vel.v[i][1]))
            .sum();
        let scale: f64 = (0..nv)
            .map(|i| {
                da[i] * (f[i][0].hypot(f[i][1])) * (vel.v[i][0].hypot(vel.v[i][1]))
            })
            .sum();
        assert!(scale > 0.0, "the fixture drove no flow at all");
        assert!(
            power > 0.5 * scale,
            "power {power:.4e} against a scale of {scale:.4e}: the flow does not run with \
             the force that drives it"
        );
    }

    /// The stress path and the force path drive the SAME flow.
    ///
    /// `solve_stress_warm` assembles `curl(div Pi)` itself; `solve_force_warm`
    /// takes the nodal force and transposes the velocity recovery. They are two
    /// routes to one physical answer and must agree in direction. Before
    /// 2026-08-20 they were ANTI-parallel, `cos = -0.9989`, because the biharmonic
    /// was driven with `+curl f` where steady Stokes gives `-curl f`. Together with
    /// `a_force_does_positive_work_on_the_flow_it_drives`, which fixes the force
    /// path absolutely, this pins the stress path absolutely too.
    ///
    /// Direction only, since the two assemble the source differently and converge
    /// to each other rather than agreeing term by term. Measured by
    /// `examples/dbg_sign.rs` on this geometry, the magnitude ratio
    /// stress-to-force runs 19.13, 4.37, 1.04 at `h` of 4, 2, 1, so it is second
    /// order and reaches four per cent at the resolution the runs use. The fixture
    /// here is the coarse one, where the ratio is still four; asserting on it
    /// would pin a discretisation error rather than the sign.
    #[test]
    fn the_stress_and_force_paths_drive_the_same_flow() {
        // The production mesh, oriented +z.
        let (cm, solver, s1, s2, f, eta) = sign_fixture();
        stress_and_force_agree_on(&cm.mesh, &solver, &s1, &s2, &f, eta);
        // And the disc, which `disk_mesh` orients -z. The sign must not depend on
        // which way the generator wound its triangles.
        let cd = crate::epitrochoid::disk_mesh(1.0, 1.0, 240, 0.04);
        let od = Operators::from_mesh(&cd.mesh, &Euclidean::<2>);
        let sd = SurfaceStokes::new_confined(&od, &cd.mesh, &cd.boundary_vertices).unwrap();
        assert!(
            sd.normals()[0][2] * solver.normals()[0][2] < 0.0,
            "the two fixtures no longer disagree on orientation, so this test has stopped \
             covering both"
        );
        let nd = cd.mesh.n_vertices();
        let xyd: Vec<[f64; 2]> =
            (0..nd).map(|i| [cd.mesh.vertices[i].x, cd.mesh.vertices[i].y]).collect();
        let trisd: Vec<[usize; 3]> =
            (0..cd.mesh.n_simplices()).map(|t| cd.mesh.simplices[t]).collect();
        let k = 3.0_f64;
        let d1: Vec<f64> = xyd.iter().map(|c| -2.0 * (k * c[0]).cos() * (k * c[1]).cos()).collect();
        let d2: Vec<f64> = xyd.iter().map(|c| -2.0 * (k * c[0]).sin() * (k * c[1]).sin()).collect();
        let fd = divergence_of_stress(&d1, &d2, &xyd, &trisd);
        stress_and_force_agree_on(&cd.mesh, &sd, &d1, &d2, &fd, 3.0);
    }

    fn stress_and_force_agree_on<M: Manifold>(
        mesh: &Mesh<M, 3, 2>,
        solver: &SurfaceStokes,
        s1: &[f64],
        s2: &[f64],
        f: &[[f64; 2]],
        eta: f64,
    ) {
        let nv = mesh.n_vertices();
        let anti = vec![0.0_f64; nv];
        let (vf, _a, _b) = solver.solve_force_warm(f, eta, mesh, None, 1e-10);
        let (vs, _c, _d) = solver.solve_stress_warm(s1, s2, &anti, eta, mesh, None, 1e-10);
        let (mut dot, mut nf, mut ns) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..nv {
            dot += vf.v[i][0] * vs.v[i][0] + vf.v[i][1] * vs.v[i][1];
            nf += vf.v[i][0].powi(2) + vf.v[i][1].powi(2);
            ns += vs.v[i][0].powi(2) + vs.v[i][1].powi(2);
        }
        assert!(nf > 0.0 && ns > 0.0, "one of the two paths produced no flow");
        let cos = dot / (nf.sqrt() * ns.sqrt());
        eprintln!(
            "stress against force, mesh oriented {:+}: cos {cos:+.6}, magnitude ratio {:.4}",
            solver.normals()[0][2].signum(),
            (ns / nf).sqrt()
        );
        assert!(
            cos > 0.99,
            "the stress path and the force path disagree: cos {cos:+.6}. A value near -1 \
             means the biharmonic is being driven with the wrong sign."
        );
    }

    /// The biharmonic source has the sign steady Stokes gives, with the exact
    /// value written out rather than compared in magnitude.
    ///
    /// For `q1 = cos(kx) cos(ky)`, `q2 = 0`, the active force `f = -zeta div Q` has
    /// `curl f = 2 zeta k^2 sin(kx) sin(ky)` about `+z`, so the source of
    /// `Delta^2 psi = source` is `-2 zeta k^2 sin(kx) sin(ky) / eta` ON A MESH
    /// ORIENTED `+z`.
    ///
    /// The orientation matters here and nowhere else. Writing the mesh normal as
    /// `s z` with `s = +-1`, the recovery `u = n x grad psi` gives
    /// `curl u = s Delta psi`, so the equation is `Delta^2 psi = -s curl f / eta`
    /// and the source carries `s`. The VELOCITY does not: the source flips, `psi`
    /// flips with it, and `u = n x grad psi` flips a second time. The two mesh
    /// generators in this crate disagree, `confined_mesh` orienting `+z` and
    /// `disk_mesh` `-z`, so an expectation written without `s` passes on one and
    /// fails on the other. `the_stress_and_force_paths_drive_the_same_flow` covers
    /// both.
    ///
    /// `examples/dbg_source.rs` compared this against the same expression through
    /// SUMS OF SQUARES and so was blind to the overall sign, which is how the sign
    /// survived. The correlation below is signed and would read `-1` for the old
    /// code.
    #[test]
    fn the_biharmonic_source_has_the_sign_steady_stokes_gives() {
        let rad = 1.0_f64;
        let (zeta, eta, k) = (2.0_f64, 3.0_f64, 3.0_f64);
        let cm = crate::epitrochoid::disk_mesh(rad, 1.0, 240, 0.04);
        let mesh = cm.mesh;
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &cm.boundary_vertices).unwrap();
        let coords = extract_coords(&mesh);
        let nv = mesh.n_vertices();
        let mut q = QField::uniform(nv, 0.0, 0.0);
        for i in 0..nv {
            q.q1[i] = (k * coords[i][0]).cos() * (k * coords[i][1]).cos();
        }
        let orient = solver.normals()[0][2].signum();
        let got = compute_vorticity_source(
            &q, zeta, eta, &mesh, &coords, solver.dual_areas(), solver.normals(),
            solver.e1_frames(),
        );
        let (mut dot, mut ng, mut nw) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..nv {
            let (x, y) = (coords[i][0], coords[i][1]);
            if (x * x + y * y).sqrt() > 0.75 * rad {
                continue;
            }
            let want =
                -orient * 2.0 * zeta * k * k * (k * x).sin() * (k * y).sin() / eta;
            dot += got[i] * want;
            ng += got[i] * got[i];
            nw += want * want;
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        eprintln!(
            "mesh oriented {orient:+}; source against the exact -s 2 zeta k^2 sin sin / eta: \
             cos {cos:+.6}"
        );
        assert!(
            cos > 0.95,
            "the biharmonic source does not match `-curl(div Pi)/eta`: signed correlation \
             {cos:+.6}. A value near -1 is the sign error of 2026-08-20 returning."
        );
    }

    fn disc(r: f64, n_boundary: usize, spacing: f64) -> (FlatMesh, Vec<usize>) {
        let cm = crate::epitrochoid::disk_mesh(r, 1.0, n_boundary, spacing);
        (cm.mesh, cm.boundary_vertices)
    }

    /// Exact biharmonic solution on a disc of radius `R`, for the CONSTANT
    /// source `Delta^2 psi = -4`:
    ///
    ///     psi(r) = R^2 r^2 / 4 - r^4 / 16 - 3 R^4 / 16,
    ///
    /// which satisfies BOTH boundary conditions the two Dirichlet solves impose,
    /// `psi(R) = 0` and `Delta psi (R) = R^2 - r^2 = 0`, so it is the solution
    /// this solver's split is meant to reproduce. Its radial velocity is
    /// `|dpsi/dr| = |R^2 r / 2 - r^3 / 4|`.
    fn psi_exact(r2: f64, rad: f64) -> f64 {
        rad * rad * r2 / 4.0 - r2 * r2 / 16.0 - 3.0 * rad.powi(4) / 16.0
    }

    /// Pins the operator: the solver must invert the BIHARMONIC, not the
    /// Laplacian.
    ///
    /// Applying the Laplacian inverse once solves `eta Delta psi = curl f`, the
    /// substrate-friction balance, rather than `eta Delta^2 psi = curl f`. That
    /// error stood in this file until 2026-08-19 and cost every confined active
    /// run its flow: measured against `flow-solver.py`'s own steady field, the
    /// single inversion reached |u|max 1.28 where the reference reaches 41.46
    /// and the biharmonic 59.35. No smoke test caught it, because a velocity
    /// suppressed by two orders of magnitude is still nonzero.
    ///
    /// This test is the one that would have. It drives the production path with
    /// a constant source whose exact solution is known in closed form, and it
    /// also asserts that the single-inversion answer FAILS, so the property is
    /// stated rather than merely satisfied.
    #[test]
    fn stokes_reproduces_the_biharmonic_manufactured_solution() {
        let rad = 1.0_f64;
        let (mesh, bverts) = disc(rad, 240, 0.04);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let nv = mesh.n_vertices();

        // Delta^2 psi = -4 has the closed-form solution above.
        let source = DVector::from_element(nv, -4.0);
        let (_vel, psi) = solver.stream_and_velocity(&source, &mesh);

        let coords = extract_coords(&mesh);
        let exact: Vec<f64> = coords
            .iter()
            .map(|p| psi_exact(p[0] * p[0] + p[1] * p[1], rad))
            .collect();

        let rel = |got: &[f64]| -> f64 {
            let num: f64 = got.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f64 = exact.iter().map(|b| b * b).sum();
            (num / den).sqrt()
        };

        let err = rel(&psi);
        assert!(
            err < 0.05,
            "biharmonic solve should reproduce the exact solution, relative L2 error {err:.4}"
        );

        // psi = 0 on the boundary, which is the no-flux half of the condition.
        for &b in &bverts {
            assert!(psi[b].abs() < 1e-8, "psi should vanish on the boundary, got {}", psi[b]);
        }

        // And the single inversion, which is what this file used to do, must not
        // pass. It is wrong by a factor set by the domain, not by a constant.
        let single: Vec<f64> = solver.poisson.solve(&source).iter().copied().collect();
        let err_single = rel(&single);
        assert!(
            err_single > 0.5,
            "a single Laplacian inversion must not reproduce the biharmonic solution, \
             relative L2 error {err_single:.4}"
        );
    }

    /// The ABSOLUTE velocity, against the same manufactured solution.
    ///
    /// The three tests around this one pin ratios and the stream function: the
    /// manufactured test compares `psi`, the scaling test compares `u` at two
    /// domain sizes, and the activity test compares `u` at two activities. A
    /// constant factor in the recovery of `u` from `psi` passes all three, and
    /// the confined runs measured |u|max near 600 where `volterra-fd`, whose
    /// fields satisfy steady Stokes at a best-fit viscosity of 398.7 against a
    /// nominal 404.77, reaches 60 at the same parameters.
    ///
    /// So this pins the number. For `Delta^2 psi = -4` on a disc of radius `R`
    /// the exact stream function is `psi_exact`, and the velocity is azimuthal
    /// with magnitude `|dpsi/dr| = |R^2 r / 2 - r^3 / 4|`, which peaks at
    /// `r = R sqrt(2/3)` with value `R^3 / (6 sqrt(6)) * 2`, that is 0.27217 at
    /// `R = 1`.
    /// Advection must return `u . grad Q`, not a fixed fraction of it.
    ///
    /// [`advect_q`] used to accumulate `(u . e_hat)(dQ/d|e|)` over the incident
    /// edges and divide by the valence. That is `u^T A_v grad Q` with
    /// `A_v = (1/n) sum_e e_hat e_hat^T`. On an isotropic fan `A_v = I/2` in the
    /// plane, so the result was exactly HALF the answer, and on a real fan `A_v`
    /// is anisotropic and stays that way under refinement, which makes the
    /// scheme inconsistent rather than merely inaccurate.
    ///
    /// Constant `u` and `Q = (cos k x, sin k y)` give the exact value
    /// `u . grad Q = (-u_x k sin k x, u_y k cos k y)`. Checking two spacings
    /// pins convergence as well as the constant: a scheme off by a fixed factor
    /// holds its error under refinement, which is what the halving did.
    #[test]
    fn advection_recovers_the_directional_derivative_and_converges() {
        let k = 2.0_f64;
        let (u_x, u_y) = (0.7_f64, -0.4_f64);
        let mut errs = Vec::new();
        for (nb, spacing) in [(80usize, 0.08_f64), (160, 0.04)] {
            let (mesh, bverts) = disc(1.0, nb, spacing);
            let nv = mesh.n_vertices();
            let coords = extract_coords(&mesh);
            let mut on_wall = vec![false; nv];
            for &b in &bverts {
                on_wall[b] = true;
            }
            let q = QField {
                q1: (0..nv).map(|i| (k * coords[i][0]).cos()).collect(),
                q2: (0..nv).map(|i| (k * coords[i][1]).sin()).collect(),
                n_vertices: nv,
            };
            let vel = VelocityField { v: vec![[u_x, u_y, 0.0]; nv], n_vertices: nv };
            let mut vb: Vec<Vec<usize>> = vec![Vec::new(); nv];
            for e in 0..mesh.n_boundaries() {
                let [v0, v1] = mesh.boundaries[e];
                vb[v0].push(e);
                vb[v1].push(e);
            }
            let got = advect_q(&q, &vel, &mesh.boundaries, &vb, &coords);

            // Away from the wall, where a one sided fan is its own error.
            let (mut num, mut den) = (0.0_f64, 0.0_f64);
            for i in 0..nv {
                let (x, y) = (coords[i][0], coords[i][1]);
                if (x * x + y * y).sqrt() > 0.8 {
                    continue;
                }
                let w1 = -u_x * k * (k * x).sin();
                let w2 = u_y * k * (k * y).cos();
                num += (got.q1[i] - w1).powi(2) + (got.q2[i] - w2).powi(2);
                den += w1 * w1 + w2 * w2;
            }
            errs.push((num / den).sqrt());
        }
        assert!(
            errs[0] < 0.10,
            "advection off by {:.1}% at the coarse spacing; the halving defect read 50%",
            errs[0] * 100.0
        );
        assert!(
            errs[1] < errs[0] * 0.75,
            "advection error did not fall under refinement: {:.4} then {:.4}, which is what \
             a fixed factor looks like",
            errs[0],
            errs[1]
        );
    }

    #[test]
    fn stokes_velocity_magnitude_matches_the_manufactured_solution() {
        let rad = 1.0_f64;
        let (mesh, bverts) = disc(rad, 240, 0.04);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let source = DVector::from_element(mesh.n_vertices(), -4.0);
        let (vel, _psi) = solver.stream_and_velocity(&source, &mesh);

        let coords = extract_coords(&mesh);
        let (mut num, mut den, mut worst, mut got_max, mut want_max) = (0.0, 0.0, 0.0_f64, 0.0_f64, 0.0_f64);
        for (i, p) in coords.iter().enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            // Skip the outermost ring, where the boundary layer of the discrete
            // curl is thicker than the element and the exact field is smallest.
            if r > 0.9 * rad {
                continue;
            }
            let want = (rad * rad * r / 2.0 - r * r * r / 4.0).abs();
            let v = vel.v[i];
            let got = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            num += (got - want) * (got - want);
            den += want * want;
            worst = worst.max((got - want).abs());
            got_max = got_max.max(got);
            want_max = want_max.max(want);
        }
        let rel = (num / den).sqrt();
        assert!(
            rel < 0.10,
            "velocity magnitude is off by relative L2 {rel:.4} (worst pointwise \
             {worst:.4}); peak got {got_max:.5} against exact {want_max:.5}, a \
             factor of {:.3}",
            got_max / want_max
        );
    }

    /// The CLAMPED wall, against its own closed form.
    ///
    /// `Delta^2 psi = -4` on a disc of radius `R` with `psi = dpsi/dr = 0` has
    /// `psi = -(R^2 - r^2)^2 / 16`, whose velocity `|dpsi/dr| = (R^2 - r^2) r/4`
    /// peaks at `r = R/sqrt 3` with value `R^3/(6 sqrt 3)`, that is 0.096225 at
    /// `R = 1`. The simply supported solution the plain constructor gives peaks
    /// at `sqrt 6/9 = 0.272166`, so free slip is faster by exactly `2 sqrt 2`.
    #[test]
    fn stokes_velocity_matches_the_clamped_solution() {
        let rad = 1.0_f64;
        let (mesh, bverts) = disc(rad, 240, 0.04);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined_clamped(&ops, &mesh, &bverts).unwrap();
        assert!(solver.is_clamped());
        let source = DVector::from_element(mesh.n_vertices(), -4.0);
        let (vel, psi) = solver.stream_and_velocity(&source, &mesh);

        let coords = extract_coords(&mesh);
        let (mut num, mut den, mut got_max, mut want_max) = (0.0, 0.0, 0.0_f64, 0.0_f64);
        for (i, p) in coords.iter().enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if r > 0.9 * rad {
                continue;
            }
            let want = (rad * rad - r * r) * r / 4.0;
            let v = vel.v[i];
            let got = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            num += (got - want) * (got - want);
            den += want * want;
            got_max = got_max.max(got);
            want_max = want_max.max(want);
        }
        let rel = (num / den).sqrt();
        assert!(
            rel < 0.12,
            "clamped velocity off by relative L2 {rel:.4}; peak got {got_max:.5} \
             against exact {want_max:.5}, factor {:.3}",
            got_max / want_max
        );

        // psi vanishes on the wall, as the free-slip solve already gave, and the
        // NORMAL DERIVATIVE now vanishes too, which is the whole point.
        for &b in &bverts {
            assert!(psi[b].abs() < 1e-8, "psi should vanish on the wall, got {}", psi[b]);
        }

        // And the free-slip solver on the same mesh must NOT reproduce it, so the
        // property is stated rather than merely satisfied.
        let free = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let (fvel, _) = free.stream_and_velocity(&source, &mesh);
        let fmax = (0..mesh.n_vertices())
            .filter(|&i| {
                let p = coords[i];
                (p[0] * p[0] + p[1] * p[1]).sqrt() <= 0.9 * rad
            })
            .map(|i| {
                let v = fvel.v[i];
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            })
            .fold(0.0_f64, f64::max);
        let ratio = fmax / got_max;
        assert!(
            (2.4..3.3).contains(&ratio),
            "free slip should exceed no slip by about 2 sqrt 2 = 2.828, got {ratio:.4}"
        );
    }

    /// Pins the operator a second way, without a manufactured solution, so a
    /// future variant of the solver is caught even if it never sees the test
    /// above.
    ///
    /// Under a similarity scaling of the domain by `s`, holding `Q` fixed as a
    /// function of the normalised coordinate, the force `f = -zeta div Q` scales
    /// as `1/s`. Stokes then gives `u ~ f L^2 / eta`, which scales as `s`. The
    /// single-inversion form gives `u ~ f / eta`, which scales as `1/s`. So the
    /// two differ by `s^2`, and at `s = 2` the correct ratio is 2 where the
    /// wrong one is 0.5.
    ///
    /// The scaled mesh is built by multiplying the coordinates of the first, so
    /// the two are exactly similar and no discretisation difference enters.
    #[test]
    fn stokes_velocity_scales_linearly_with_the_domain_size() {
        let (mesh1, bverts) = disc(1.0, 200, 0.05);
        let scaled: Vec<[f64; 2]> = mesh1
            .vertices
            .iter()
            .map(|p| [2.0 * p[0], 2.0 * p[1]])
            .collect();
        let tris: Vec<[usize; 3]> = mesh1.simplices.clone();
        let mesh2 = FlatMesh::from_triangles(scaled, tris);

        let ops1 = Operators::from_mesh(&mesh1, &Euclidean::<2>);
        let ops2 = Operators::from_mesh(&mesh2, &Euclidean::<2>);
        let s1 = SurfaceStokes::new_confined(&ops1, &mesh1, &bverts).unwrap();
        let s2 = SurfaceStokes::new_confined(&ops2, &mesh2, &bverts).unwrap();

        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 3.0;

        // The same Q values at corresponding vertices, so Q is the same function
        // of the normalised coordinate on both meshes.
        let q = QField::random_perturbation(mesh1.n_vertices(), 0.3, 42);

        let rms = |v: &VelocityField| -> f64 {
            (v.v.iter().map(|[x, y, z]| x * x + y * y + z * z).sum::<f64>()
                / v.n_vertices as f64)
                .sqrt()
        };
        let u1 = rms(&s1.solve(&q, &params, &ops1, &mesh1));
        let u2 = rms(&s2.solve(&q, &params, &ops2, &mesh2));
        let ratio = u2 / u1;

        assert!(
            (1.8..2.2).contains(&ratio),
            "doubling the domain should double the Stokes velocity (a single \
             Laplacian inversion would halve it), got ratio {ratio:.4}"
        );
    }

    /// The velocity is linear in the activity and inverse in the viscosity.
    /// Cheap, and it catches a normalisation slip in `compute_vorticity_source`,
    /// which divides by `eta` before the solves rather than after.
    #[test]
    fn stokes_velocity_is_linear_in_activity_and_inverse_in_viscosity() {
        let (mesh, bverts) = disc(1.0, 160, 0.06);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let q = QField::random_perturbation(mesh.n_vertices(), 0.3, 7);

        let rms = |zeta: f64, eta: f64| -> f64 {
            let mut p = ActiveNematicParams::default_test();
            p.zeta_eff = zeta;
            p.eta = eta;
            let v = solver.solve(&q, &p, &ops, &mesh);
            (v.v.iter().map(|[x, y, z]| x * x + y * y + z * z).sum::<f64>()
                / v.n_vertices as f64)
                .sqrt()
        };

        let base = rms(1.0, 1.0);
        assert!(base > 1e-12, "baseline velocity should be nonzero");
        let by_zeta = rms(3.0, 1.0) / base;
        let by_eta = rms(1.0, 4.0) / base;
        assert!((by_zeta - 3.0).abs() < 1e-9, "u should be linear in zeta, got {by_zeta}");
        assert!((by_eta - 0.25).abs() < 1e-9, "u should go as 1/eta, got {by_eta}");
    }

    /// `solve` and `solve_warm` must agree. They carry the biharmonic separately,
    /// one through the factorisation and one through preconditioned CG, so a fix
    /// applied to only one of them is a live hazard.
    #[test]
    fn stokes_warm_and_direct_solves_agree() {
        let (mesh, bverts) = disc(1.0, 160, 0.06);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 1.5;
        let q = QField::random_perturbation(mesh.n_vertices(), 0.3, 11);

        let direct = solver.solve(&q, &params, &ops, &mesh);
        let (warm, _psi, _its) = solver.solve_warm(&q, &params, &ops, &mesh, None, 1e-12);

        let num: f64 = direct
            .v
            .iter()
            .zip(&warm.v)
            .map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>())
            .sum();
        let den: f64 = direct.v.iter().map(|a| a.iter().map(|c| c * c).sum::<f64>()).sum();
        let rel = (num / den.max(1e-300)).sqrt();
        assert!(rel < 1e-6, "warm and direct solves should agree, relative difference {rel:.3e}");
    }

    /// The antisymmetric stress must reach the force.
    ///
    /// `Pi_A` enters through `T_xy = s2 + a` against `T_yx = s2 - a`, so an
    /// assembly that symmetrises the tensor, or that simply forgets the third
    /// argument, drops it silently and nothing else in this module notices. The
    /// active-only path has nowhere to put it, which is exactly why it is easy
    /// to lose in a refactor back toward that path.
    #[test]
    fn the_antisymmetric_stress_reaches_the_vorticity_source() {
        let (mesh, bverts) = disc(1.0, 160, 0.06);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let nv = mesh.n_vertices();

        // A spatially varying antisymmetric part and nothing else. A constant one
        // would have zero divergence and prove nothing.
        let coords = extract_coords(&mesh);
        let anti: Vec<f64> = coords.iter().map(|p| p[0] * p[0] - p[1]).collect();
        let zero = vec![0.0_f64; nv];

        let (with_a, _psi, _its) =
            solver.solve_stress_warm(&zero, &zero, &anti, 1.0, &mesh, None, 1e-10);
        let (without, _psi2, _its2) =
            solver.solve_stress_warm(&zero, &zero, &zero, 1.0, &mesh, None, 1e-10);

        let rms = |v: &VelocityField| -> f64 {
            (v.v.iter().map(|[x, y, z]| x * x + y * y + z * z).sum::<f64>() / nv as f64).sqrt()
        };
        assert!(rms(&without) < 1e-12, "a zero stress should give no flow");
        assert!(
            rms(&with_a) > 1e-6,
            "the antisymmetric stress must drive a flow, got rms {:.3e}",
            rms(&with_a)
        );
    }

    /// The force-driven Stokes operator must be symmetric and positive.
    ///
    /// `<f, u(f)> >= 0` is the dissipation being non-negative, and
    /// `<f1, u(f2)> = <f2, u(f1)>` is the operator being self-adjoint. Together
    /// with the adjoint elastic force they give the discrete energy law; without
    /// them the cancellation is destroyed downstream of a correct force.
    #[test]
    fn the_force_driven_stokes_operator_is_symmetric_and_positive() {
        let (mesh, bverts) = disc(1.0, 200, 0.05);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let solver = SurfaceStokes::new_confined(&ops, &mesh, &bverts).unwrap();
        let nv = mesh.n_vertices();
        let eta = 2.5;

        let field = |seed: u64| -> Vec<[f64; 2]> {
            let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (0..nv)
                .map(|_| {
                    let mut r = || {
                        st = st
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0
                    };
                    [r(), r()]
                })
                .collect()
        };
        let dot = |f: &[[f64; 2]], u: &VelocityField| -> f64 {
            (0..nv).map(|i| f[i][0] * u.v[i][0] + f[i][1] * u.v[i][1]).sum()
        };

        let f1 = field(11);
        let f2 = field(29);
        let (u1, _, _) = solver.solve_force_warm(&f1, eta, &mesh, None, 1e-12);
        let (u2, _, _) = solver.solve_force_warm(&f2, eta, &mesh, None, 1e-12);

        let d11 = dot(&f1, &u1);
        let d22 = dot(&f2, &u2);
        assert!(d11 > 0.0, "dissipation should be positive, got {d11:.6e}");
        assert!(d22 > 0.0, "dissipation should be positive, got {d22:.6e}");

        let a = dot(&f1, &u2);
        let b = dot(&f2, &u1);
        let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
        assert!(
            rel < 1e-8,
            "the operator should be self-adjoint: <f1,u2> = {a:.6e} against <f2,u1> = {b:.6e}, \
             relative {rel:.3e}"
        );
    }

    #[test]
    fn stokes_nonzero_velocity_on_sphere() {
        use crate::mesh_gen::icosphere;
        use cartan_manifolds::sphere::Sphere;

        let mesh = icosphere(3); // 642 vertices
        let ops = Operators::from_mesh_generic(&mesh, &Sphere::<3>).unwrap();
        let mut params = ActiveNematicParams::default_test();
        params.zeta_eff = 2.0;
        params.eta = 1.0;

        let nv = mesh.n_vertices();
        let q = QField::random_perturbation(nv, 0.3, 42);

        let solver = SurfaceStokes::new(&ops, &mesh).unwrap();
        let v = solver.solve(&q, &params, &ops, &mesh);

        let v_rms: f64 = (v.v.iter()
            .map(|[x, y, z]| x * x + y * y + z * z)
            .sum::<f64>() / nv as f64)
            .sqrt();
        assert!(
            v_rms > 1e-6,
            "nonzero activity on sphere should give nonzero velocity, got v_rms = {v_rms:.3e}"
        );
    }
}


/// Discrete Gaussian curvature at each vertex, by angle defect.
///
/// `K_i = (2 pi - sum of the incident triangle angles at i) / A_i`, with `A_i`
/// the dual area. This is the standard discretisation and it satisfies the
/// discrete Gauss-Bonnet theorem exactly: `sum_i K_i A_i = 2 pi chi`, which is
/// `4 pi` on a sphere whatever the triangulation.
///
/// Vertices on a boundary have no full angle to complete, so their defect is
/// taken as zero rather than as `2 pi` minus a partial fan.
pub fn gaussian_curvature(
    n_vertices: usize,
    simplices: &[[usize; 3]],
    coords: &[[f64; 3]],
    dual_areas: &[f64],
) -> Vec<f64> {
    let mut angle_sum = vec![0.0_f64; n_vertices];
    let mut valence = vec![0usize; n_vertices];
    let mut edge_faces: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for tri in simplices {
        for k in 0..3 {
            let i = tri[k];
            let a = tri[(k + 1) % 3];
            let b = tri[(k + 2) % 3];
            let u = sub3(coords[a], coords[i]);
            let v = sub3(coords[b], coords[i]);
            let nu = norm3(u);
            let nv = norm3(v);
            if nu > 1e-30 && nv > 1e-30 {
                let c = (dot3(u, v) / (nu * nv)).clamp(-1.0, 1.0);
                angle_sum[i] += c.acos();
            }
            valence[i] += 1;
            let e = if a < b { (a, b) } else { (b, a) };
            *edge_faces.entry(e).or_insert(0) += 1;
        }
    }
    // A vertex is on a boundary when one of its edges has a single incident face.
    let mut on_boundary = vec![false; n_vertices];
    for (&(a, b), &count) in &edge_faces {
        if count < 2 {
            on_boundary[a] = true;
            on_boundary[b] = true;
        }
    }
    (0..n_vertices)
        .map(|i| {
            if on_boundary[i] || dual_areas[i] <= 0.0 {
                0.0
            } else {
                (std::f64::consts::TAU - angle_sum[i]) / dual_areas[i]
            }
        })
        .collect()
}
