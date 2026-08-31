//! Klein's Landau-de Gennes nematic on a conforming DEC mesh.
//!
//! The molecular field, the strong anchoring and the defect count, written to
//! match `flow-solver.py` term for term so a mesh result and a lattice result are
//! the same physics at two discretisations.
//!
//! ## The molecular field
//!
//! ```text
//! H = K grad^2 Q - (A + C Tr(Q^2)) Q,     Tr(Q^2) = 2 (Qxx^2 + Qxy^2)
//! ```
//!
//! `cartan_dec` assembles `laplace_beltrami = star0^-1 d0^T star1 d0`, which is
//! positive semidefinite as a quadratic form and so represents `-grad^2`. The
//! elastic term therefore enters as `-K L Q`. On a flat mesh the two components of
//! `Q` are ordinary scalars, so the scalar Laplacian applied component-wise is the
//! whole operator and no connection term is needed.
//!
//! ## The anchoring
//!
//! `apply_Q_boundary_conditions` sets, at each wall site with outward normal at
//! angle `theta`,
//!
//! ```text
//! nn = (cos(q theta), sin(q theta))
//! Qxx = S0 (nn_y^2 - 1/2)      Qxy = -S0 nn_x nn_y
//! ```
//!
//! which is `Q = S0 (m m - I/2)` for `m = (nn_y, -nn_x)`, the vector at angle
//! `q theta` turned by a quarter turn. At `q = 1` that is the wall tangent, which
//! is the tangential anchoring the published videos show.
//!
//! Note the amplitude convention: with `Q = S0 (m m - I/2)` the invariant is
//! `Tr(Q^2) = S0^2 / 2`, so the ordered state has `sqrt(Tr(Q^2)) = S0 / sqrt(2)`,
//! which is 1 at Klein's `A = -C`. The reference's own `S` diagnostic is
//! `sqrt(Tr(Q^2))`, so a fully ordered run reports 1 and not `S0`.
//!
//! ## Timestepping
//!
//! The passive relaxation is `dQ/dt = H / gamma`. Explicitly that is stable only
//! for `dt < gamma h^2 / (4 K)`, which is 1.5e-3 at Klein's constants and `h = 1`
//! and 1.5e-9 at the `h = 1e-3` a graded cusp needs, so the elastic term is taken
//! implicitly:
//!
//! ```text
//! (I + dt K / gamma * L) Q^{n+1} = Q^n - dt / gamma * (A + C Tr(Q^2)^n) Q^n
//! ```
//!
//! The matrix is symmetric positive definite, so conjugate gradients solve it, and
//! the mesh being graded costs iterations rather than stability. Anchored vertices
//! are pinned by row replacement, which keeps the system symmetric because the
//! known values are substituted into the right-hand side.

use nalgebra::DVector;

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;

use crate::confined::ConfinedMesh2;
use crate::nematic_params::NematicParams;
use crate::qfield::QField;
use crate::semi_lagrangian::SemiLagrangian;
use crate::stokes::VelocityField;

/// Which mass matrix the weak form uses.
///
/// The DEC operator `star0^-1 d0^T star1 d0` is the stiffness matrix
/// preconditioned by a *lumped* mass, the diagonal of the circumcentric dual
/// areas. That diagonal is only guaranteed positive when every triangle is well
/// centred, which is a stronger condition than non-degeneracy and one a graded
/// mesh of a cusped domain does not meet everywhere: the meshes here carry about
/// 1.5 per cent obtuse triangles.
///
/// The finite-element exterior calculus alternative is the *consistent* mass
/// matrix of the same space. Whitney 0-forms are the P1 Lagrange basis, so the
/// consistent mass is the textbook P1 element matrix, `area / 12 * (2 on the
/// diagonal, 1 off)`, assembled over triangles. It is symmetric positive definite
/// for any non-degenerate triangle, obtuse or not, so it removes the
/// well-centredness requirement. The stiffness matrix is the same cotangent matrix
/// either way, so the maximum principle is unaffected: this fixes one of the two
/// quality constraints, not both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mass {
    /// Diagonal dual areas, which is what the DEC operator carries.
    Lumped,
    /// The P1 element mass matrix, which is the FEEC mass for Whitney 0-forms.
    Consistent,
}

/// A confined Landau-de Gennes problem: mesh, operators, parameters, anchoring.
pub struct LdgProblem {
    pub mesh: ConfinedMesh2,
    pub ops: Operators<Euclidean<2>, 3, 2>,
    pub params: NematicParams,
    /// Anchoring winding, Klein's `net_charge`.
    pub q_anchor: f64,
    /// Anchored values at the boundary vertices, `(Qxx, Qxy)`.
    pub anchor: Vec<(f64, f64)>,
    /// True at a vertex whose value is imposed.
    pinned: Vec<bool>,
    /// Which mass matrix the weak form uses.
    pub mass: Mass,
    /// Consistent P1 mass, as (row, col, value) triples; empty when lumped.
    mass_tri: Vec<(usize, usize, f64)>,
    /// Lumped dual areas, the diagonal of `star0`.
    mass_lumped: Vec<f64>,
    /// Diagonal of the mass-normalised Laplacian, kept so the Jacobi
    /// preconditioner can be rebuilt for any `alpha` without touching the
    /// sparse matrix again.
    l_diag: Vec<f64>,
    /// One- and two-ring neighbours of every vertex, the stencil
    /// [`Self::velocity_gradients_from_psi`] fits its quadratic over. Built once.
    two_ring: Vec<Vec<usize>>,
}

/// Anchored `(Qxx, Qxy)` for an outward normal at angle `theta`.
///
/// Written out rather than reduced so it can be read against
/// `apply_Q_boundary_conditions` line by line.
pub fn anchored_q(theta: f64, q_anchor: f64, s0: f64) -> (f64, f64) {
    let a = q_anchor * theta;
    let (nnx, nny) = (a.cos(), a.sin());
    (s0 * (nny * nny - 0.5), -s0 * nnx * nny)
}

impl LdgProblem {
    /// Assemble the operators and the anchoring for one mesh and parameter set.
    pub fn new(
        mesh: ConfinedMesh2,
        params: NematicParams,
        q_anchor: f64,
    ) -> Result<Self, cartan_dec::DecError> {
        let manifold = Euclidean::<2>;
        let ops = Operators::from_mesh_generic(&mesh.mesh, &manifold)?;
        let s0 = params.s0();
        let mut anchor = Vec::with_capacity(mesh.boundary_vertices.len());
        for i in 0..mesh.boundary_vertices.len() {
            let n = mesh.boundary_normals[i];
            // The reference's normals point out of the liquid crystal; the mesh
            // stores the inward one, so the angle is that of its negative.
            let theta = (-n[1]).atan2(-n[0]);
            anchor.push(anchored_q(theta, q_anchor, s0));
        }
        let mut pinned = vec![false; mesh.mesh.n_vertices()];
        for &v in &mesh.boundary_vertices {
            pinned[v] = true;
        }
        let mass_lumped: Vec<f64> = ops.mass0.iter().copied().collect();
        let nvx = mesh.mesh.n_vertices();
        let l_diag: Vec<f64> = (0..nvx)
            .map(|i| ops.laplace_beltrami.get(i, i).copied().unwrap_or(0.0))
            .collect();
        let two_ring = {
            let m = &mesh.mesh;
            let nv = m.n_vertices();
            let mut one = vec![std::collections::BTreeSet::new(); nv];
            for t in 0..m.n_simplices() {
                let s = m.simplices[t];
                for a in 0..3 {
                    for b in 0..3 {
                        if a != b {
                            one[s[a]].insert(s[b]);
                        }
                    }
                }
            }
            (0..nv)
                .map(|v| {
                    let mut r = one[v].clone();
                    for &j in &one[v] {
                        for &k in &one[j] {
                            r.insert(k);
                        }
                    }
                    r.remove(&v);
                    r.into_iter().collect::<Vec<usize>>()
                })
                .collect::<Vec<_>>()
        };

        Ok(Self {
            mesh,
            ops,
            params,
            q_anchor,
            anchor,
            pinned,
            mass: Mass::Lumped,
            mass_tri: Vec::new(),
            mass_lumped,
            l_diag,
            two_ring,
        })
    }

    /// Switch to the consistent P1 mass matrix and assemble it.
    ///
    /// `M_e = area / 12 * [[2,1,1],[1,2,1],[1,1,2]]`, the exact integral of
    /// `phi_i phi_j` over a triangle for the linear basis. Assembled once.
    pub fn with_consistent_mass(mut self) -> Self {
        let m = &self.mesh.mesh;
        let mut tri = Vec::with_capacity(9 * m.n_simplices());
        for t in 0..m.n_simplices() {
            let s = m.simplices[t];
            let (v0, v1, v2) = (m.vertices[s[0]], m.vertices[s[1]], m.vertices[s[2]]);
            let area = 0.5 * ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs();
            let c = area / 12.0;
            for a in 0..3 {
                for b in 0..3 {
                    tri.push((s[a], s[b], if a == b { 2.0 * c } else { c }));
                }
            }
        }
        self.mass_tri = tri;
        self.mass = Mass::Consistent;
        self
    }

    /// Apply the chosen mass matrix.
    fn apply_mass(&self, x: &[f64], out: &mut [f64]) {
        match self.mass {
            Mass::Lumped => {
                for i in 0..x.len() {
                    out[i] = self.mass_lumped[i] * x[i];
                }
            }
            Mass::Consistent => {
                out.iter_mut().for_each(|v| *v = 0.0);
                for &(i, j, v) in &self.mass_tri {
                    out[i] += v * x[j];
                }
            }
        }
    }

    /// Apply the cotangent stiffness matrix.
    ///
    /// `cartan_dec` assembles the lumped Laplacian `star0^-1 A`, so multiplying
    /// back by the dual areas recovers `A` itself, which is the quantity both mass
    /// choices share.
    fn apply_stiffness(&self, x: &[f64], out: &mut [f64]) {
        let lx = self
            .ops
            .apply_laplace_beltrami(&DVector::from_column_slice(x));
        for i in 0..x.len() {
            out[i] = self.mass_lumped[i] * lx[i];
        }
    }

    /// A random initial director, matching the reference's own initial condition:
    /// an angle uniform on `[0, pi)` at every interior vertex, with the anchoring
    /// imposed on the wall.
    pub fn random_state(&self, seed: u64) -> QField {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let nv = self.mesh.mesh.n_vertices();
        let s0 = self.params.s0();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut q = QField::zeros(nv);
        for i in 0..nv {
            let th = std::f64::consts::PI * rng.random::<f64>();
            let (c, s) = (th.cos(), th.sin());
            q.q1[i] = s0 * (c * c - 0.5);
            q.q2[i] = s0 * c * s;
        }
        self.impose_anchoring(&mut q);
        q
    }

    /// Overwrite the boundary vertices with their anchored values.
    pub fn impose_anchoring(&self, q: &mut QField) {
        for (k, &v) in self.mesh.boundary_vertices.iter().enumerate() {
            q.q1[v] = self.anchor[k].0;
            q.q2[v] = self.anchor[k].1;
        }
    }

    /// Klein's molecular field, `H = K grad^2 Q - (A + C Tr(Q^2)) Q`.
    pub fn molecular_field(&self, q: &QField) -> QField {
        let nv = q.n_vertices;
        let l1 = self
            .ops
            .apply_laplace_beltrami(&DVector::from_column_slice(&q.q1));
        let l2 = self
            .ops
            .apply_laplace_beltrami(&DVector::from_column_slice(&q.q2));
        let (k, a, c) = (
            self.params.k_frank,
            self.params.a_landau,
            self.params.c_landau,
        );
        let mut h = QField::zeros(nv);
        for i in 0..nv {
            let tr = 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]);
            let bulk = a + c * tr;
            // The assembled Laplacian is `-grad^2`, hence the leading minus.
            h.q1[i] = -k * l1[i] - bulk * q.q1[i];
            h.q2[i] = -k * l2[i] - bulk * q.q2[i];
        }
        h
    }

    /// The rate at which the flow transports `Q`, exactly as `step_active` uses
    /// it: `-u . grad Q + S(Q, grad u)`.
    ///
    /// Split out so the elastic force can be built as its exact adjoint and the
    /// pair can be tested against each other.
    pub fn transport_rate(&self, q: &QField, vel: &[[f64; 2]]) -> QField {
        let nv = q.n_vertices;
        let dq = self.q_gradients(q);
        let du = self.velocity_gradients(vel);
        let s = self.corotational(q, &du);
        let mut out = QField::zeros(nv);
        for i in 0..nv {
            let adv1 = vel[i][0] * dq[i][0] + vel[i][1] * dq[i][1];
            let adv2 = vel[i][0] * dq[i][2] + vel[i][1] * dq[i][3];
            out.q1[i] = -adv1 + s.q1[i];
            out.q2[i] = -adv2 + s.q2[i];
        }
        out
    }

    /// The elastic force the free energy exerts on the fluid, built as the exact
    /// discrete ADJOINT of [`Self::transport_rate`].
    ///
    /// This is the fix for the missing energy law. Assembling the elastic stress
    /// independently, as `beris_edwards_stress` does, gives two operators that are
    /// adjoint in the continuum and not discretely, so the power the stress
    /// delivers to the fluid does not cancel the energy the transport removes from
    /// `Q`. On a mild mesh the residual is small; graded a thousand to one it is
    /// unbounded, and the passive system reaches NaN in twenty steps with the
    /// activity switched off.
    ///
    /// The construction. The discrete free energy has
    /// `dF/dq1_i = -2 w_i h1_i` exactly, with `w_i` the lumped mass and `h` the
    /// molecular field, because `apply_laplace_beltrami` is mass-normalised and
    /// the bulk term is integrated with the same lumped mass. The energy the
    /// transport removes in a step is therefore `-2 dt <w h, T(u)>`, and the law
    /// requires the fluid to receive exactly that:
    ///
    /// ```text
    /// <f, u> = 2 sum_i w_i [ h1_i T1_i(u) + h2_i T2_i(u) ]   for every u,
    /// ```
    ///
    /// which DEFINES `f` as that functional's Riesz representer. Since `T` is
    /// linear in `u`, `f` is obtained by differentiating it.
    ///
    /// The advective half is local: `u . grad Q` reads `u` at the vertex, so its
    /// transpose is `-2 w_i (h1 grad q1 + h2 grad q2)_i`.
    ///
    /// The co-rotational half is linear in `grad u`, which
    /// [`Self::velocity_gradients`] builds by averaging the per-triangle P1
    /// gradients onto vertices by area. Its transpose therefore runs the other
    /// way: form the per-vertex coefficients of `grad u`, push them back to the
    /// triangles through the same area weights, and contract with the same basis
    /// gradients. Written that way the two are adjoint to rounding, at any
    /// grading, which is what `the_elastic_force_is_the_adjoint_of_transport`
    /// holds them to.
    pub fn elastic_force(&self, q: &QField) -> Vec<[f64; 2]> {
        let mut h = self.molecular_field(q);
        // The anchored vertices are pinned by the Dirichlet condition, so the flow
        // never transports them and the free energy never gives up their share.
        // Booking that share delivers power the field does not pay for. Measured
        // 2026-08-22 at `d = 0.99`: the pairing over all vertices is `-1.44e11`,
        // over the interior `+2.28e6`, so the wall holds the entire budget and the
        // interior alone INJECTS energy. At `d = 0.9` both readings share a sign,
        // which is why this stayed invisible until the mesh graded into a cusp.
        for &v in &self.mesh.boundary_vertices {
            h.q1[v] = 0.0;
            h.q2[v] = 0.0;
        }
        self.elastic_force_from_h(q, &h)
    }

    /// [`Self::elastic_force`] without the Dirichlet restriction, kept so the A/B
    /// can be run. This is the adjoint of the UNCONSTRAINED transport operator,
    /// which the dynamics does not apply.
    pub fn elastic_force_unconstrained(&self, q: &QField) -> Vec<[f64; 2]> {
        let h = self.molecular_field(q);
        self.elastic_force_from_h(q, &h)
    }

    /// [`Self::elastic_force`] with the molecular field supplied.
    pub fn elastic_force_from_h(&self, q: &QField, h: &QField) -> Vec<[f64; 2]> {
        let m = &self.mesh.mesh;
        let nv = q.n_vertices;
        let dq = self.q_gradients(q);
        let lam = self.params.lambda;

        // Advective half, local at each vertex.
        let mut f = vec![[0.0_f64; 2]; nv];
        for i in 0..nv {
            let w = 2.0 * self.mass_lumped[i];
            f[i][0] = -w * (h.q1[i] * dq[i][0] + h.q2[i] * dq[i][2]);
            f[i][1] = -w * (h.q1[i] * dq[i][1] + h.q2[i] * dq[i][3]);
        }

        // Co-rotational half. `corotational` is linear in `du = (D0,D1,D2,D3)`:
        //
        //   s1 = D0 (lam_s - 4 q1^2) + D1 (-q2 - 2 q1 q2) + D2 (q2 - 2 q1 q2)
        //   s2 = D0 (-4 q1 q2) + D1 (lam_s/2 + q1 - 2 q2^2) + D2 (lam_s/2 - q1 - 2 q2^2)
        //
        // with D3 unused. Contract with `2 w h` to get the coefficient of each
        // component of `du` at the vertex.
        let mut c = vec![[0.0_f64; 4]; nv];
        for i in 0..nv {
            let (q1, q2) = (q.q1[i], q.q2[i]);
            let (h1, h2) = (h.q1[i], h.q2[i]);
            let lam_s = lam * (4.0 * (q1 * q1 + q2 * q2)).sqrt();
            let a10 = lam_s - 4.0 * q1 * q1;
            let a11 = -q2 - 2.0 * q1 * q2;
            let a12 = q2 - 2.0 * q1 * q2;
            let a20 = -4.0 * q1 * q2;
            let a21 = 0.5 * lam_s + q1 - 2.0 * q2 * q2;
            let a22 = 0.5 * lam_s - q1 - 2.0 * q2 * q2;
            let w = 2.0 * self.mass_lumped[i];
            c[i][0] = w * (h1 * a10 + h2 * a20);
            c[i][1] = w * (h1 * a11 + h2 * a21);
            c[i][2] = w * (h1 * a12 + h2 * a22);
            c[i][3] = 0.0;
        }

        // The area weights `velocity_gradients` divides by, recomputed identically
        // so the transpose matches the forward map term for term.
        let mut wsum = vec![0.0_f64; nv];
        for t in 0..m.n_simplices() {
            let sv = m.simplices[t];
            let (p0, p1, p2) = (m.vertices[sv[0]], m.vertices[sv[1]], m.vertices[sv[2]]);
            let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let w = 0.5 * two_a.abs();
            for a in 0..3 {
                wsum[sv[a]] += w;
            }
        }

        for t in 0..m.n_simplices() {
            let sv = m.simplices[t];
            let (p0, p1, p2) = (m.vertices[sv[0]], m.vertices[sv[1]], m.vertices[sv[2]]);
            let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let inv = 1.0 / two_a;
            let g = [
                [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv],
                [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv],
                [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv],
            ];
            let w = 0.5 * two_a.abs();
            // e_t[k] = sum over the triangle's vertices of c[i][k] / wsum[i].
            let mut e = [0.0_f64; 4];
            for a in 0..3 {
                let i = sv[a];
                if wsum[i] > 1e-30 {
                    for k in 0..4 {
                        e[k] += c[i][k] / wsum[i];
                    }
                }
            }
            // d_t[0] = sum_a ux_a g_a[0], d_t[1] = sum_a uy_a g_a[0],
            // d_t[2] = sum_a ux_a g_a[1], d_t[3] = sum_a uy_a g_a[1].
            for a in 0..3 {
                let j = sv[a];
                f[j][0] += w * (e[0] * g[a][0] + e[2] * g[a][1]);
                f[j][1] += w * (e[1] * g[a][0] + e[3] * g[a][1]);
            }
        }
        f
    }

    /// The Landau-de Gennes free energy,
    ///
    /// ```text
    /// F[Q] = integral of  (K/2)|grad Q|^2 + (A/2) Tr Q^2 + (C/4) (Tr Q^2)^2
    /// ```
    ///
    /// with `Tr Q^2 = 2(q1^2 + q2^2)` and `|grad Q|^2 = 2(|grad q1|^2 + |grad q2|^2)`
    /// in the two-component representation.
    ///
    /// This is the potential the variational step descends. Onsager's principle
    /// makes the time-incremental problem a minimisation whose potential is
    /// evaluated at the NEW state, so the step decreases the Rayleighian
    /// monotonically and is stable without a step restriction; a scheme that
    /// evaluates the same potential's gradient at the OLD state inherits the
    /// stiffest length scale in the mesh instead. Zhu, Saintillan and Chern state
    /// the property as the integrator being "unconditionally stable by design, as
    /// it preserves the system's dissipative structure", so that "the allowable
    /// time step size depends only on the solvability of the optimization"
    /// (arXiv:2407.14025v2, section "Variational integrator by Onsager's
    /// variational principle").
    ///
    /// Reported so the claim can be checked rather than asserted: in the passive
    /// limit this must not increase, and
    /// `the_variational_step_descends_the_free_energy` holds it to that.
    pub fn free_energy(&self, q: &QField) -> f64 {
        let dq = self.q_gradients(q);
        let (k, a, c) = (
            self.params.k_frank,
            self.params.a_landau,
            self.params.c_landau,
        );
        let nv = q.n_vertices;
        let dens: Vec<f64> = (0..nv)
            .map(|i| {
                let [dxq1, dyq1, dxq2, dyq2] = dq[i];
                let grad2 = 2.0 * (dxq1 * dxq1 + dyq1 * dyq1 + dxq2 * dxq2 + dyq2 * dyq2);
                let tr = 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]);
                0.5 * k * grad2 + 0.5 * a * tr + 0.25 * c * tr * tr
            })
            .collect();
        let mut w = vec![0.0; nv];
        let ones = vec![1.0; nv];
        self.apply_mass(&ones, &mut w);
        (0..nv).map(|i| w[i] * dens[i]).sum()
    }

    /// The same free energy assembled so that `-2 w h` is EXACTLY its gradient.
    ///
    /// [`Self::free_energy`] forms `|grad q|^2` from [`Self::q_gradients`], the
    /// area-weighted vertex average of the per-triangle P1 gradients. The gradient
    /// of that functional is not the cotangent stiffness, so the molecular field is
    /// not its derivative: measured 2026-08-22, `dF/dq . delta` disagrees with
    /// `-2 <w h, delta>` by 1.0% at `d = 0.5`, 4.6% at `d = 0.9` and 4.4% at
    /// `d = 0.99` on a smoothed field, and by 73% to 109% on a rough one.
    ///
    /// Summing `A_T |grad q|_T^2` over triangles instead gives the standard P1
    /// stiffness form, whose gradient IS the assembled Laplacian. Measured against
    /// the same directional derivative: relative 2.5e-10, 4.3e-10 and 1.2e-9 at the
    /// same three gradings, and 1e-9 on a rough field, since the identity is
    /// algebraic rather than asymptotic.
    ///
    /// This is the functional the scheme actually descends, so it is the one an
    /// energy law must be stated about.
    pub fn free_energy_fem(&self, q: &QField) -> f64 {
        let m = &self.mesh.mesh;
        let (k, a, c) = (
            self.params.k_frank,
            self.params.a_landau,
            self.params.c_landau,
        );
        let nv = q.n_vertices;
        let mut total = 0.0_f64;
        for t in 0..m.n_simplices() {
            let sv = m.simplices[t];
            let (p0, p1, p2) = (m.vertices[sv[0]], m.vertices[sv[1]], m.vertices[sv[2]]);
            let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let inv = 1.0 / two_a;
            let g = [
                [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv],
                [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv],
                [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv],
            ];
            let mut d = [0.0_f64; 4];
            for aa in 0..3 {
                d[0] += q.q1[sv[aa]] * g[aa][0];
                d[1] += q.q1[sv[aa]] * g[aa][1];
                d[2] += q.q2[sv[aa]] * g[aa][0];
                d[3] += q.q2[sv[aa]] * g[aa][1];
            }
            let area = 0.5 * two_a.abs();
            let grad2 = 2.0 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
            total += area * 0.5 * k * grad2;
        }
        let ones = vec![1.0_f64; nv];
        let mut w = vec![0.0_f64; nv];
        self.apply_mass(&ones, &mut w);
        for i in 0..nv {
            let tr = 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]);
            total += w[i] * (0.5 * a * tr + 0.25 * c * tr * tr);
        }
        total
    }

    /// The full Beris-Edwards stress, term for term against `flow-solver.py`'s
    /// `calculate_Pi`, `get_Erickson_stress` and `get_TrQH_term`.
    ///
    /// Returns `(Pi_xx, Pi_xy, Pi_A,xy)`, the two independent components of the
    /// symmetric traceless part and the one component of the antisymmetric part:
    ///
    /// ```text
    /// Pi_S,xx = -lambda H1 - zeta q1 + TrQH q1
    ///           - K [ (dx q1)^2 + (dx q2)^2 - (dy q1)^2 - (dy q2)^2 ]
    /// Pi_S,xy = -lambda H2 - zeta q2 + TrQH q2
    ///           - 2 K [ dx q1 dy q1 + dx q2 dy q2 ]
    /// Pi_A,xy = 2 (q1 H2 - H1 q2)
    /// ```
    ///
    /// with `TrQH = 2 (q1 H1 + q2 H2)`.
    ///
    /// The active part `-zeta Q` was for a long time the only term the mesh
    /// carried, so the elastic backflow that opposes the active flow was absent.
    ///
    /// Two points where the reference's prose and its code disagree, and the
    /// CODE is followed, as with the `-2 Tr(QE) Q` term in the co-rotation. Its
    /// derivation writes `2 Tr[QH] Q` while `get_TrQH_term` adds `TrQH * Q` once,
    /// with `TrQH` already carrying the factor of two from the trace. And the
    /// Ericksen stress is not traceless, so only its traceless part enters here;
    /// the isotropic remainder is a gradient the pressure absorbs.
    ///
    /// The gradients come from [`Self::q_gradients`], which averages the
    /// per-triangle P1 gradients onto vertices by area. A per-triangle Ericksen
    /// stress would be piecewise constant and its elementwise divergence would
    /// vanish, so the projection to vertices is what makes the term act at all.
    pub fn beris_edwards_stress(&self, q: &QField) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.beris_edwards_stress_masked(q, &[])
    }

    /// As [`Self::beris_edwards_stress`], with the ELASTIC terms suppressed on
    /// the listed vertices. The active term `-zeta Q` is kept everywhere.
    ///
    /// The elastic terms are what make the explicit stress unstable on a graded
    /// mesh. `-lambda H` is third order in `Q` through `H = K grad^2 Q` and the
    /// Ericksen term is `K (grad Q)^2`, so both grow as inverse powers of the
    /// local element size, while the active term carries no derivative of `Q` at
    /// all. On the matched-scale nephroid at `d = 0.99` the driver's own
    /// diffusive limit is 2.12e-9 against a step of 5e-5, and the field reaches
    /// `S = 3.1e8` by step 17.
    ///
    /// The vertices to pass are the wall layer `ACT_WALLH` already selects, on
    /// elements below 0.05. That layer exists because a cusp tip 0.005 across is
    /// eight hundred times below the core size `ncl = 4`, so the continuum model
    /// resolves nothing inside it and the velocity there is a recovery artefact
    /// rather than a flow. The same argument covers the elastic stress: a free
    /// energy differentiated twice across a sliver the physics cannot see is not
    /// a force the physics exerts.
    ///
    /// This is an approximation and should be reported as one. The principled
    /// fix is to stop treating the stress explicitly at all: Zhu, Saintillan and
    /// Chern derive the whole step from a time-incremental Onsager variational
    /// principle on a discrete Rayleighian, which is "unconditionally stable by
    /// design, as it preserves the system's dissipative structure", so that "the
    /// allowable time step size depends only on the solvability of the
    /// optimization" (arXiv:2407.14025v2, section "Variational integrator by
    /// Onsager's variational principle"). `volterra-dec/src/flow.rs` already
    /// carries that pattern for the Helfrich step.
    pub fn beris_edwards_stress_masked(
        &self,
        q: &QField,
        suppress_elastic: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nv = q.n_vertices;
        let h = self.molecular_field(q);
        let dq = self.q_gradients(q);
        let (k, lambda, zeta) = (self.params.k_frank, self.params.lambda, self.params.zeta);
        let mut sym1 = vec![0.0_f64; nv];
        let mut sym2 = vec![0.0_f64; nv];
        let mut anti = vec![0.0_f64; nv];
        let mut masked = vec![false; nv];
        for &i in suppress_elastic {
            if i < nv {
                masked[i] = true;
            }
        }
        for i in 0..nv {
            let (a, b, c) = if masked[i] {
                // Active term only: K = 0, lambda = 0, H ignored.
                stress_at_vertex(q.q1[i], q.q2[i], 0.0, 0.0, [0.0; 4], 0.0, 0.0, zeta)
            } else {
                stress_at_vertex(q.q1[i], q.q2[i], h.q1[i], h.q2[i], dq[i], k, lambda, zeta)
            };
            sym1[i] = a;
            sym2[i] = b;
            anti[i] = c;
        }
        (sym1, sym2, anti)
    }

    /// `sqrt(Tr(Q^2))` per vertex, which is the reference's `S` diagnostic and
    /// equals 1 in the ordered state at `A = -C`.
    pub fn order_parameter(&self, q: &QField) -> Vec<f64> {
        (0..q.n_vertices)
            .map(|i| (2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i])).sqrt())
            .collect()
    }

    /// One semi-implicit step of the passive relaxation, returning the largest
    /// change in `Q` so a caller can watch convergence.
    ///
    /// The elastic term is implicit and the bulk term explicit, which is the
    /// standard splitting: the bulk term is a local cubic with no stiffness of its
    /// own, and the elastic term is where the mesh's smallest edge would otherwise
    /// dictate the step.
    pub fn step_passive(&self, q: &mut QField, dt: f64, cg_tol: f64) -> f64 {
        let (k, a, c, g) = (
            self.params.k_frank,
            self.params.a_landau,
            self.params.c_landau,
            self.params.gamma,
        );
        let nv = q.n_vertices;
        let alpha = dt * k / g;

        // The weak form is `M dQ/dt = -(K / gamma) A Q - (1 / gamma) M b`, so the
        // semi-implicit system is `(M + alpha A) Q^{n+1} = M (Q^n - dt/gamma b^n)`.
        // With the lumped mass this reduces to the earlier `(I + alpha L)` form
        // after dividing through by the dual areas, so the two paths agree on a
        // well-centred mesh and differ only where the lumping is questionable.
        let mut pre1 = vec![0.0; nv];
        let mut pre2 = vec![0.0; nv];
        for i in 0..nv {
            let tr = 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]);
            let bulk = (a + c * tr) * dt / g;
            pre1[i] = q.q1[i] - bulk * q.q1[i];
            pre2[i] = q.q2[i] - bulk * q.q2[i];
        }
        let mut rhs1 = vec![0.0; nv];
        let mut rhs2 = vec![0.0; nv];
        self.apply_mass(&pre1, &mut rhs1);
        self.apply_mass(&pre2, &mut rhs2);
        for (kk, &v) in self.mesh.boundary_vertices.iter().enumerate() {
            rhs1[v] = self.anchor[kk].0;
            rhs2[v] = self.anchor[kk].1;
        }

        let mut new1 = q.q1.clone();
        let mut new2 = q.q2.clone();
        self.cg(&rhs1, &mut new1, alpha, cg_tol);
        self.cg(&rhs2, &mut new2, alpha, cg_tol);

        let mut worst = 0.0_f64;
        for i in 0..nv {
            worst = worst
                .max((new1[i] - q.q1[i]).abs())
                .max((new2[i] - q.q2[i]).abs());
        }
        q.q1 = new1;
        q.q2 = new2;
        self.impose_anchoring(q);
        worst
    }

    /// Apply `(M + alpha A)` with pinned rows replaced by the identity.
    fn apply(&self, x: &[f64], alpha: f64, out: &mut [f64]) {
        let n = x.len();
        let mut mx = vec![0.0; n];
        let mut ax = vec![0.0; n];
        self.apply_mass(x, &mut mx);
        self.apply_stiffness(x, &mut ax);
        for i in 0..n {
            out[i] = if self.pinned[i] {
                x[i]
            } else {
                mx[i] + alpha * ax[i]
            };
        }
    }

    /// Jacobi preconditioner for `(M + alpha A)` at this `alpha`.
    ///
    /// `A = diag(star0) * laplace_beltrami`, so its diagonal is the dual area
    /// times the Laplacian's own. Pinned rows are the identity and precondition
    /// to one. The mesh is graded a thousand to one into the cusp, and the
    /// stiffness there exceeds the mass by the square of that ratio, so the
    /// unpreconditioned iteration count is set by the grading rather than by the
    /// physics: 305 iterations per solve at `d = 0.99`, against 40 for the same
    /// problem at `d = 0.7`.
    fn jacobi(&self, alpha: f64) -> Vec<f64> {
        let n = self.mass_lumped.len();
        // The consistent mass diagonal is gathered in ONE pass over the
        // triplets. Scanning the whole list once per vertex is
        // `O(n |mass_tri|)`, and `mass_tri` grows with `n`, so on the 20k
        // vertex graded meshes this runs to billions of operations for a
        // preconditioner that is rebuilt on every solve.
        let m_diag = match self.mass {
            Mass::Lumped => None,
            Mass::Consistent => {
                let mut d = vec![0.0; n];
                for &(a, b, v) in &self.mass_tri {
                    if a == b {
                        d[a] += v;
                    }
                }
                Some(d)
            }
        };
        (0..n)
            .map(|i| {
                if self.pinned[i] {
                    return 1.0;
                }
                let m = match &m_diag {
                    None => self.mass_lumped[i],
                    Some(d) => d[i],
                };
                let d = m + alpha * self.mass_lumped[i] * self.l_diag[i];
                if d.abs() < 1e-300 { 1.0 } else { 1.0 / d }
            })
            .collect()
    }

    /// Conjugate gradients on `(I + alpha L) x = b`, pinned rows held fixed.
    ///
    /// The operator is symmetric positive definite off the pinned rows, and the
    /// pinned rows are the identity, so the iteration is well posed provided the
    /// residual is zeroed there, which is what keeps the search directions from
    /// moving a boundary value.
    fn cg(&self, b: &[f64], x: &mut [f64], alpha: f64, tol: f64) -> usize {
        self.pcg(b, x, alpha, tol, &self.jacobi(alpha))
    }

    /// The same iteration with the preconditioner supplied, so a caller stepping
    /// at a fixed `alpha` builds it once rather than once per solve.
    fn pcg(&self, b: &[f64], x: &mut [f64], alpha: f64, tol: f64, inv_diag: &[f64]) -> usize {
        let n = b.len();
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        self.apply(x, alpha, &mut ax);
        for i in 0..n {
            r[i] = if self.pinned[i] { 0.0 } else { b[i] - ax[i] };
        }
        let mut z: Vec<f64> = (0..n).map(|i| inv_diag[i] * r[i]).collect();
        let mut p = z.clone();
        let mut rs: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        // Relative to the right-hand side, because the mass matrix carries the
        // element areas and so sets the scale of the system: an absolute tolerance
        // that suits the lumped identity form is meaningless once M is consistent.
        let scale: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let stop = tol * scale;
        // The stopping test is on the true residual, not on the preconditioned
        // one, so the tolerance means the same thing with and without a
        // preconditioner and the two can be compared.
        let mut rnorm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rnorm <= stop {
            return 0;
        }
        let mut ap = vec![0.0; n];
        for it in 0..2000 {
            self.apply(&p, alpha, &mut ap);
            for i in 0..n {
                if self.pinned[i] {
                    ap[i] = 0.0;
                }
            }
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-300 {
                return it;
            }
            let a_step = rs / pap;
            for i in 0..n {
                x[i] += a_step * p[i];
                r[i] -= a_step * ap[i];
            }
            rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rnorm <= stop {
                return it + 1;
            }
            for i in 0..n {
                z[i] = inv_diag[i] * r[i];
            }
            let rs_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rs_new / rs;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rs = rs_new;
        }
        2000
    }

    /// Relax to equilibrium, returning the step count and the final change.
    pub fn relax(&self, q: &mut QField, dt: f64, max_steps: usize, tol: f64) -> (usize, f64) {
        let mut last = f64::INFINITY;
        for s in 0..max_steps {
            last = self.step_passive(q, dt, 1e-10);
            if last < tol {
                return (s + 1, last);
            }
        }
        (max_steps, last)
    }

    /// Defect charges by triangle, in units of a half.
    ///
    /// The winding of the doubled director angle round a triangle, which is the
    /// mesh analogue of the lattice's plaquette winding and is exact per triangle:
    /// each increment is wrapped into a half turn, and the total can only be a
    /// multiple of `2 pi`.
    pub fn defect_charges(&self, q: &QField) -> Vec<(usize, i32, [f64; 2])> {
        let m = &self.mesh.mesh;
        let mut out = Vec::new();
        let wrap =
            |x: f64| x - 2.0 * std::f64::consts::PI * (x / (2.0 * std::f64::consts::PI)).round();
        for t in 0..m.n_simplices() {
            let s = m.simplices[t];
            let phi: Vec<f64> = s.iter().map(|&v| q.q2[v].atan2(q.q1[v])).collect();
            let total = wrap(phi[1] - phi[0]) + wrap(phi[2] - phi[1]) + wrap(phi[0] - phi[2]);
            // The winding is signed by the direction the loop is traversed, and a
            // triangulator does not promise a consistent vertex order. Multiplying
            // by the sign of the signed area measures every triangle
            // anticlockwise, so the charge does not depend on the mesher's
            // bookkeeping. Without this the total comes out at minus the anchoring
            // winding whenever the triangles happen to be listed clockwise.
            let (v0, v1, v2) = (m.vertices[s[0]], m.vertices[s[1]], m.vertices[s[2]]);
            let twice_area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
            let orient = if twice_area >= 0.0 { 1.0 } else { -1.0 };
            let charge = (orient * total / (2.0 * std::f64::consts::PI)).round() as i32;
            if charge != 0 {
                let c = [
                    (m.vertices[s[0]].x + m.vertices[s[1]].x + m.vertices[s[2]].x) / 3.0,
                    (m.vertices[s[0]].y + m.vertices[s[1]].y + m.vertices[s[2]].y) / 3.0,
                ];
                out.push((t, charge, c));
            }
        }
        out
    }

    /// Defect count by sign and the total charge, after merging neighbouring
    /// triangles of the same sign into one core.
    pub fn defect_summary(
        &self,
        q: &QField,
        merge: f64,
    ) -> (usize, usize, f64, Vec<(f64, f64, i32)>) {
        let raw = self.defect_charges(q);
        let mut cores: Vec<(f64, f64, i32, usize)> = Vec::new();
        for (_, ch, c) in &raw {
            let mut placed = false;
            for k in 0..cores.len() {
                if cores[k].2 == *ch {
                    let dx = cores[k].0 / cores[k].3 as f64 - c[0];
                    let dy = cores[k].1 / cores[k].3 as f64 - c[1];
                    if (dx * dx + dy * dy).sqrt() < merge {
                        cores[k].0 += c[0];
                        cores[k].1 += c[1];
                        cores[k].3 += 1;
                        placed = true;
                        break;
                    }
                }
            }
            if !placed {
                cores.push((c[0], c[1], *ch, 1));
            }
        }
        let list: Vec<(f64, f64, i32)> = cores
            .iter()
            .map(|&(x, y, ch, n)| (x / n as f64, y / n as f64, ch))
            .collect();
        let pos = list.iter().filter(|c| c.2 > 0).count();
        let neg = list.iter().filter(|c| c.2 < 0).count();
        // The total is summed over the raw triangle windings rather than over the
        // merged cores, because it has to be independent of the merge radius: two
        // cores closer together than the radius merge into one entry and the count
        // loses a defect, while the winding sum cannot lose one. A total that
        // disagrees with the anchoring winding is then a statement about the
        // physics, and a count that disagrees with the total is a statement about
        // the merge.
        let total = raw.iter().map(|&(_, ch, _)| ch as f64).sum::<f64>() / 2.0;
        (pos, neg, total, list)
    }

    /// Per-vertex velocity gradient `[dx ux, dx uy, dy ux, dy uy]`.
    ///
    /// The gradient of a P1 field is constant on a triangle, so the vertex value
    /// is the area-weighted average over the incident triangles. That is the same
    /// recovery the molecular field's Laplacian uses, so the strain rate and the
    /// elasticity see one discretisation rather than two.
    ///
    /// The reference takes these as central differences on its lattice, which is
    /// the same object at second order; the difference is that here the stencil
    /// follows the graded elements into the cusp instead of stopping at the cell
    /// size.
    pub fn velocity_gradients(&self, vel: &[[f64; 2]]) -> Vec<[f64; 4]> {
        let m = &self.mesh.mesh;
        let nv = m.n_vertices();
        let mut acc = vec![[0.0_f64; 4]; nv];
        let mut wsum = vec![0.0_f64; nv];
        for t in 0..m.n_simplices() {
            let s = m.simplices[t];
            let (p0, p1, p2) = (m.vertices[s[0]], m.vertices[s[1]], m.vertices[s[2]]);
            let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let inv = 1.0 / two_a;
            // grad of the P1 basis: the opposite edge turned by a quarter turn.
            let g = [
                [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv],
                [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv],
                [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv],
            ];
            let mut d = [0.0_f64; 4];
            for a in 0..3 {
                let u = vel[s[a]];
                d[0] += u[0] * g[a][0];
                d[1] += u[1] * g[a][0];
                d[2] += u[0] * g[a][1];
                d[3] += u[1] * g[a][1];
            }
            let w = 0.5 * two_a.abs();
            for a in 0..3 {
                for k in 0..4 {
                    acc[s[a]][k] += w * d[k];
                }
                wsum[s[a]] += w;
            }
        }
        for i in 0..nv {
            if wsum[i] > 1e-30 {
                for k in 0..4 {
                    acc[i][k] /= wsum[i];
                }
            }
        }
        acc
    }

    /// Per-vertex gradient of `Q`, as `[dx q1, dy q1, dx q2, dy q2]`.
    pub fn q_gradients(&self, q: &QField) -> Vec<[f64; 4]> {
        let m = &self.mesh.mesh;
        let nv = m.n_vertices();
        let mut acc = vec![[0.0_f64; 4]; nv];
        let mut wsum = vec![0.0_f64; nv];
        for t in 0..m.n_simplices() {
            let s = m.simplices[t];
            let (p0, p1, p2) = (m.vertices[s[0]], m.vertices[s[1]], m.vertices[s[2]]);
            let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if two_a.abs() < 1e-30 {
                continue;
            }
            let inv = 1.0 / two_a;
            let g = [
                [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv],
                [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv],
                [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv],
            ];
            let mut d = [0.0_f64; 4];
            for a in 0..3 {
                d[0] += q.q1[s[a]] * g[a][0];
                d[1] += q.q1[s[a]] * g[a][1];
                d[2] += q.q2[s[a]] * g[a][0];
                d[3] += q.q2[s[a]] * g[a][1];
            }
            let w = 0.5 * two_a.abs();
            for a in 0..3 {
                for k in 0..4 {
                    acc[s[a]][k] += w * d[k];
                }
                wsum[s[a]] += w;
            }
        }
        for i in 0..nv {
            if wsum[i] > 1e-30 {
                for k in 0..4 {
                    acc[i][k] /= wsum[i];
                }
            }
        }
        acc
    }

    /// Velocity gradient taken from the STREAM FUNCTION in one step, rather than
    /// by differentiating a recovered velocity.
    ///
    /// `u = n x grad psi`, so `grad u` is the Hessian of `psi` turned by a
    /// quarter turn:
    ///
    /// ```text
    /// dx u_x = -psi_xy    dx u_y =  psi_xx
    /// dy u_x = -psi_yy    dy u_y =  psi_xy
    /// ```
    ///
    /// Why not simply differentiate `u`. The production path recovers `u` from
    /// `psi` with a vertex-gradient operator and then applies a SECOND, different
    /// vertex-gradient operator to get `grad u`. The recovery is accurate in `u`
    /// but its error has grid-scale structure, and differentiating divides that
    /// structure by `h`, so the strain does not converge. Measured on the
    /// nephroid by `examples/dbg_strain.rs` against `psi = sin(ax) sin(by)`, the
    /// relative error in `E_xy` over the interior runs
    ///
    /// ```text
    ///   h      chained     this
    ///   2      1.61e-1     6.88e-2
    ///   1      1.20e-1     3.08e-2
    ///   0.5    9.75e-2     1.49e-2
    /// ```
    ///
    /// which is `O(h^0.4)` against `O(h^1.1)`. The chained form is therefore
    /// inconsistent, not merely inaccurate, and the co-rotational term it feeds
    /// is what drives the order parameter. The lattice reference has no such
    /// error, since it central-differences its own velocity field, and its
    /// interior `S` never exceeds `0.93 S0` over two million steps while the mesh
    /// reached `2.04 S0` and diverged.
    ///
    /// The construction is also EXACTLY divergence-free at every vertex, since
    /// `dx u_x + dy u_y = -psi_xy + psi_xy` identically, which the chained form is
    /// not, and its vorticity is `(psi_xx + psi_yy) / 2`, the discrete Laplacian
    /// the solver already works with.
    ///
    /// The fit is a weighted least squares of
    /// `psi_j - psi_v` against `[dx, dy, dx^2/2, dx dy, dy^2/2]` over the two-ring,
    /// with weights `1/r^2`. Five unknowns against about eighteen neighbours, so
    /// the system is well overdetermined away from the boundary. Where it is
    /// singular the vertex keeps a zero gradient rather than a wild one.
    ///
    /// Returns `[dx u_x, dx u_y, dy u_x, dy u_y]`, the convention
    /// [`Self::corotational`] reads.
    pub fn velocity_gradients_from_psi(&self, psi: &[f64]) -> Vec<[f64; 4]> {
        let m = &self.mesh.mesh;
        let nv = m.n_vertices();
        let xy: Vec<[f64; 2]> = (0..nv)
            .map(|i| [m.vertices[i].x, m.vertices[i].y])
            .collect();
        let mut out = vec![[0.0_f64; 4]; nv];
        for v in 0..nv {
            let mut ata = [[0.0_f64; 5]; 5];
            let mut atb = [0.0_f64; 5];
            for &j in &self.two_ring[v] {
                let (dx, dy) = (xy[j][0] - xy[v][0], xy[j][1] - xy[v][1]);
                let r2 = dx * dx + dy * dy;
                if r2 < 1e-30 {
                    continue;
                }
                let w = 1.0 / r2;
                let b = [dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy];
                let rhs = psi[j] - psi[v];
                for a in 0..5 {
                    atb[a] += w * b[a] * rhs;
                    for c in 0..5 {
                        ata[a][c] += w * b[a] * b[c];
                    }
                }
            }
            let mut mat = ata;
            let mut r = atb;
            let mut ok = true;
            for c in 0..5 {
                let mut piv = c;
                for k in c + 1..5 {
                    if mat[k][c].abs() > mat[piv][c].abs() {
                        piv = k;
                    }
                }
                if mat[piv][c].abs() < 1e-14 {
                    ok = false;
                    break;
                }
                mat.swap(c, piv);
                r.swap(c, piv);
                for k in c + 1..5 {
                    let f = mat[k][c] / mat[c][c];
                    for j in c..5 {
                        mat[k][j] -= f * mat[c][j];
                    }
                    r[k] -= f * r[c];
                }
            }
            if !ok {
                continue;
            }
            let mut x = [0.0_f64; 5];
            for c in (0..5).rev() {
                let mut acc = r[c];
                for j in c + 1..5 {
                    acc -= mat[c][j] * x[j];
                }
                x[c] = acc / mat[c][c];
            }
            let (pxx, pxy, pyy) = (x[2], x[3], x[4]);
            out[v] = [-pxy, pxx, -pyy, pxy];
        }
        out
    }

    /// Shortest incident edge at each vertex, the local length the explicit
    /// advective term is measured against.
    pub fn local_h(&self) -> Vec<f64> {
        let m = &self.mesh.mesh;
        let nv = m.n_vertices();
        let mut h = vec![f64::INFINITY; nv];
        for t in 0..m.n_simplices() {
            let sx = m.simplices[t];
            for a in 0..3 {
                let (i, j) = (sx[a], sx[(a + 1) % 3]);
                let (pi, pj) = (m.vertices[i], m.vertices[j]);
                let d = ((pi.x - pj.x).powi(2) + (pi.y - pj.y).powi(2)).sqrt();
                h[i] = h[i].min(d);
                h[j] = h[j].min(d);
            }
        }
        for v in h.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        h
    }

    /// The largest local Courant number `dt |u_i| / h_i`, and where it occurs.
    ///
    /// Pairing the global fastest speed with the globally smallest element
    /// overstates this badly on a graded cusped mesh, because the two live in
    /// different places: the small elements are all within a cusp radius of the
    /// wall, and no-slip pins the velocity to zero there. The bound that governs
    /// the explicit advective term is the local one, so it is the local one that
    /// is measured.
    pub fn courant(&self, vel: &[[f64; 2]], dt: f64, h: &[f64]) -> (f64, usize) {
        let mut worst = 0.0_f64;
        let mut at = 0usize;
        for i in 0..vel.len() {
            if h[i] <= 0.0 {
                continue;
            }
            let sp = (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1]).sqrt();
            let c = dt * sp / h[i];
            if c > worst {
                worst = c;
                at = i;
            }
        }
        (worst, at)
    }

    /// Klein's co-rotation tensor `S`, term for term against `H_S_from_Q`.
    ///
    /// ```text
    /// S_ij = lambda S E_ij + Q_ik omega_kj - omega_ik Q_kj - 2 Tr(Q E) Q_ij
    /// ```
    ///
    /// with `E` the strain rate, `omega` the vorticity tensor and
    /// `lambda S = lambda sqrt(2 Tr(Q^2))`. The last term is the one the
    /// reference's derivation in prose omits and its code carries: it is what
    /// keeps `Q` traceless under the flow coupling, and dropping it lets the
    /// amplitude drift wherever the strain is large, which is exactly at a core.
    pub fn corotational(&self, q: &QField, du: &[[f64; 4]]) -> QField {
        let nv = q.n_vertices;
        let lam = self.params.lambda;
        let mut s = QField::zeros(nv);
        for i in 0..nv {
            let (dxux, dxuy, dyux, dyuy) = (du[i][0], du[i][1], du[i][2], du[i][3]);
            let _ = dyuy;
            let w_xy = 0.5 * (dxuy - dyux);
            let tr_q_sq = 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i]);
            let lam_s = lam * (2.0 * tr_q_sq).sqrt();
            let tr_qe = 2.0 * q.q1[i] * dxux + q.q2[i] * (dyux + dxuy);
            s.q1[i] = lam_s * dxux - 2.0 * w_xy * q.q2[i] - 2.0 * tr_qe * q.q1[i];
            s.q2[i] = lam_s * 0.5 * (dxuy + dyux) + 2.0 * w_xy * q.q1[i] - 2.0 * tr_qe * q.q2[i];
        }
        s
    }

    /// One semi-implicit step of the active dynamics, returning the largest
    /// change in `Q`.
    ///
    /// The equation is the reference's own,
    ///
    /// ```text
    /// dQ/dt + u . grad Q = H / gamma + S
    /// ```
    ///
    /// split exactly as [`Self::step_passive`] splits the passive case: the
    /// elastic part of `H` implicit, everything local explicit. The co-rotation
    /// is algebraic in the velocity gradient, so it joins the bulk term in the
    /// right-hand side and the matrix is the one the passive path already uses.
    /// The velocity is held fixed across the step, which is the splitting the
    /// Stokes limit invites: there is no velocity time derivative to integrate.
    ///
    /// `sl` chooses how the advective term is carried.
    ///
    /// - `None`: `u . grad Q` is differenced on the mesh and added to the
    ///   right-hand side. This is the direct reading of the reference, and it is
    ///   stable only while `dt |u| < h`. On a mesh graded into a cusp that is the
    ///   smallest element rather than the bulk one, and at `d = 0.99` the grading
    ///   ratio is a thousand, so the condition fails by a factor of thirty at the
    ///   speeds the system reaches.
    /// - `Some(sl)`: the transport is taken by a backward trace, stable at any
    ///   step, leaving the co-rotation untouched. The two agree as `dt` falls and
    ///   differ by the splitting error at finite `dt`, which is first order.
    ///
    /// Returns `(largest change in Q, conjugate-gradient iterations)`, the
    /// second being the larger of the two component solves. The iteration count
    /// is the cost of the step and it is reported rather than hidden: on a mesh
    /// graded into a cusp the stiffness dominates the mass by the square of the
    /// grading ratio, so an unpreconditioned solve can reach its own iteration
    /// ceiling and return an unconverged field while still looking like progress.
    pub fn step_active(
        &self,
        q: &mut QField,
        vel: &[[f64; 2]],
        dt: f64,
        cg_tol: f64,
        sl: Option<&SemiLagrangian>,
    ) -> (f64, usize) {
        let du = self.velocity_gradients(vel);
        self.step_active_with_du(q, vel, &du, dt, cg_tol, sl)
    }

    /// [`Self::step_active`] with the velocity gradient supplied rather than
    /// differenced from `vel`.
    ///
    /// The co-rotational term is driven by `grad u` alone, and taking it from the
    /// stream function with [`Self::velocity_gradients_from_psi`] is consistent
    /// where differentiating the recovered velocity is not. `vel` is still needed,
    /// for the advective term and for the backward trace.
    pub fn step_active_with_du(
        &self,
        q: &mut QField,
        vel: &[[f64; 2]],
        du: &[[f64; 4]],
        dt: f64,
        cg_tol: f64,
        sl: Option<&SemiLagrangian>,
    ) -> (f64, usize) {
        let (k, a, c, g) = (
            self.params.k_frank,
            self.params.a_landau,
            self.params.c_landau,
            self.params.gamma,
        );
        let nv = q.n_vertices;
        let alpha = dt * k / g;

        let s = self.corotational(q, du);

        // The transport half of the split, taken first so the co-rotation and the
        // bulk term act on the field at the arrival point.
        let base = match sl {
            Some(op) => {
                let mut v3 = VelocityField::zeros(nv);
                for i in 0..nv {
                    v3.v[i] = [vel[i][0], vel[i][1], 0.0];
                }
                let mut t = op.transport(q, &v3, dt);
                self.impose_anchoring(&mut t);
                t
            }
            None => q.clone(),
        };
        let dq = if sl.is_none() {
            self.q_gradients(q)
        } else {
            Vec::new()
        };

        let mut pre1 = vec![0.0; nv];
        let mut pre2 = vec![0.0; nv];
        for i in 0..nv {
            let tr = 2.0 * (base.q1[i] * base.q1[i] + base.q2[i] * base.q2[i]);
            let bulk = (a + c * tr) / g;
            let (adv1, adv2) = match sl {
                Some(_) => (0.0, 0.0),
                None => (
                    vel[i][0] * dq[i][0] + vel[i][1] * dq[i][1],
                    vel[i][0] * dq[i][2] + vel[i][1] * dq[i][3],
                ),
            };
            pre1[i] = base.q1[i] + dt * (-bulk * base.q1[i] - adv1 + s.q1[i]);
            pre2[i] = base.q2[i] + dt * (-bulk * base.q2[i] - adv2 + s.q2[i]);
        }
        let mut rhs1 = vec![0.0; nv];
        let mut rhs2 = vec![0.0; nv];
        self.apply_mass(&pre1, &mut rhs1);
        self.apply_mass(&pre2, &mut rhs2);
        for (kk, &v) in self.mesh.boundary_vertices.iter().enumerate() {
            rhs1[v] = self.anchor[kk].0;
            rhs2[v] = self.anchor[kk].1;
        }

        let mut new1 = q.q1.clone();
        let mut new2 = q.q2.clone();
        let it1 = self.cg(&rhs1, &mut new1, alpha, cg_tol);
        let it2 = self.cg(&rhs2, &mut new2, alpha, cg_tol);

        let mut worst = 0.0_f64;
        for i in 0..nv {
            worst = worst
                .max((new1[i] - q.q1[i]).abs())
                .max((new2[i] - q.q2[i]).abs());
        }
        q.q1 = new1;
        q.q2 = new2;
        self.impose_anchoring(q);
        (worst, it1.max(it2))
    }
}

/// The pointwise Beris-Edwards stress at one vertex.
///
/// `dq` is `[dx q1, dy q1, dx q2, dy q2]`. Returns `(Pi_S,xx, Pi_S,xy, Pi_A,xy)`.
///
/// Split out from [`LdgProblem::beris_edwards_stress`] so the algebra can be
/// checked against values produced by `flow-solver.py`'s own `calculate_Pi`,
/// which is what `stress_matches_the_reference_calculate_pi` does. The
/// transcription was verified against the reference over a random field at
/// relative 1e-16 on all three components before those values were frozen.
#[allow(clippy::too_many_arguments)]
pub fn stress_at_vertex(
    q1: f64,
    q2: f64,
    h1: f64,
    h2: f64,
    dq: [f64; 4],
    k: f64,
    lambda: f64,
    zeta: f64,
) -> (f64, f64, f64) {
    let [dxq1, dyq1, dxq2, dyq2] = dq;
    let tr_qh = 2.0 * (q1 * h1 + q2 * h2);
    // Traceless part of the Ericksen stress. Its isotropic remainder is a
    // gradient and the pressure absorbs it, so it is dropped rather than passed
    // on, exactly as the reference drops `Pi_I`.
    let ericksen_b = -k * (dxq1 * dxq1 + dxq2 * dxq2 - dyq1 * dyq1 - dyq2 * dyq2);
    let ericksen_c = -2.0 * k * (dxq1 * dyq1 + dxq2 * dyq2);
    (
        -lambda * h1 - zeta * q1 + tr_qh * q1 + ericksen_b,
        -lambda * h2 - zeta * q2 + tr_qh * q2 + ericksen_c,
        2.0 * (q1 * h2 - h1 * q2),
    )
}

#[cfg(test)]
mod tests {

    /// The adjoint identity, which is the whole content of the fix.
    ///
    /// `elastic_force` must satisfy, for EVERY discrete velocity,
    ///
    ///     <f, u> = 2 sum_i w_i [ h1_i T1_i(u) + h2_i T2_i(u) ],
    ///
    /// where `T` is `transport_rate` and `-2 w h` is the discrete energy
    /// gradient. That is what makes the power the elastic stress delivers to the
    /// fluid cancel the energy the transport removes from `Q`, term by term
    /// rather than in the continuum limit, and it is what the independently
    /// assembled `beris_edwards_stress` does not do.
    ///
    /// Tested on a graded mesh, since that is where the old assembly fails:
    /// exactness here cannot depend on the elements being uniform.
    #[test]
    fn the_elastic_force_is_the_adjoint_of_transport() {
        for d in [0.5_f64, 0.9, 0.99] {
            let p = problem(2.0, d, 1.0, 2.0);
            let q = p.random_state(3);
            let nv = q.n_vertices;
            let h = p.molecular_field(&q);
            let f = p.elastic_force(&q);
            let mut is_b = vec![false; nv];
            for &v in &p.mesh.boundary_vertices {
                is_b[v] = true;
            }

            // Several unrelated velocity fields, so the identity is not satisfied
            // by accident on one direction.
            for seed in 0..4u64 {
                let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mut rnd = || {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0
                };
                let u: Vec<[f64; 2]> = (0..nv).map(|_| [rnd(), rnd()]).collect();
                let t = p.transport_rate(&q, &u);

                let lhs: f64 = (0..nv).map(|i| f[i][0] * u[i][0] + f[i][1] * u[i][1]).sum();
                // Over the INTERIOR: the anchored vertices are pinned, so the
                // transport never acts there and the pairing must not book them.
                let rhs: f64 = (0..nv)
                    .filter(|i| !is_b[*i])
                    .map(|i| 2.0 * p.mass_lumped[i] * (h.q1[i] * t.q1[i] + h.q2[i] * t.q2[i]))
                    .sum();
                let scale = lhs.abs().max(rhs.abs()).max(1e-300);
                let rel = (lhs - rhs).abs() / scale;
                assert!(
                    rel < 1e-10,
                    "d = {d}, seed {seed}: <f,u> = {lhs:.6e} against 2<w h, T(u)> = {rhs:.6e}, \
                     relative {rel:.3e}. The elastic force is not the adjoint of the transport."
                );
            }
        }
    }

    /// The passive system must not gain free energy.
    ///
    /// With the activity off, the only driving is the elastic stress, and the
    /// only sinks are rotational and viscous dissipation. So `F[Q]` must decrease
    /// monotonically, whatever the step. That is the discrete energy law, and a
    /// scheme that has one cannot blow up the way this one does.
    ///
    /// The law holds only if the elastic stress driving the flow is the exact
    /// discrete adjoint of the operator that transports `Q`. `corotational` and
    /// `beris_edwards_stress` are assembled independently here, so there is no
    /// reason for them to be adjoint, and this test is what settles whether they
    /// are.
    #[test]
    fn the_passive_system_does_not_gain_free_energy() {
        // d = 0.9 holds it exactly: worst single-step gain 0.000e0 over 20 steps.
        energy_law_at(0.9, 1e-4);
    }

    /// How the energy-law residual depends on mesh grading, with the lattice as
    /// the limiting case.
    ///
    /// The lattice this study compares against is a uniform grid: its grading
    /// ratio is 1, and it integrates three million steps at `d = 0.99` without
    /// trouble. The mesh at the same `d` is graded about two thousand to one and
    /// loses the energy law entirely. If grading is the control parameter rather
    /// than the geometry, the residual should grow with the ratio and vanish as
    /// the mesh approaches the lattice's uniformity.
    ///
    /// Diagnostic rather than a guard, so it prints and is ignored by default:
    ///
    ///     cargo test -p volterra-dec --lib grading_sweep -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic sweep, prints the energy-law residual against mesh grading"]
    fn energy_law_residual_against_grading_sweep() {
        println!("  h_min    ratio    worst single-step relative gain in F");
        for h_min in [1.0_f64, 0.5, 0.25, 0.1, 0.05, 0.02] {
            let curve = Epitrochoid {
                q: 2.0,
                d: 0.99,
                r: 98.0,
            };
            let mesh = confined_mesh(
                curve,
                MeshOpts {
                    h_bulk: 2.0,
                    h_min,
                    ..Default::default()
                },
            );
            let params = NematicParams::klein(1.5, 4.0, 100).passive();
            let mut p = LdgProblem::new(mesh, params, 1.0).expect("operators");
            p.params.zeta = 0.0;
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let mut f_prev = p.free_energy(&q);
            let mut worst = 0.0_f64;
            for _ in 0..20 {
                let (s1, s2, sa) = p.beris_edwards_stress(&q);
                let (vel, _psi, _its) = stokes.solve_stress_warm(
                    &s1,
                    &s2,
                    &sa,
                    p.params.eta,
                    &p.mesh.mesh,
                    None,
                    1e-10,
                );
                let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
                p.step_active(&mut q, &v2, 1e-4, 1e-8, None);
                let f = p.free_energy(&q);
                if f.is_finite() && f_prev.is_finite() {
                    worst = worst.max((f - f_prev) / f_prev.abs().max(1e-30));
                } else {
                    worst = f64::INFINITY;
                }
                f_prev = f;
            }
            println!("  {h_min:<8} {:<8.0} {worst:.3e}", 2.0 / h_min);
        }
    }

    /// The same law at the sharpness the study needs, where it FAILS.
    ///
    /// Measured 2026-08-19, with the activity off so the elastic stress is the
    /// only driving: at `d = 0.9` the free energy falls monotonically with a
    /// worst single-step gain of exactly zero, and at `d = 0.99` it reaches NaN
    /// inside twenty steps. Nothing about the activity is involved.
    ///
    /// The cause is discrete: `corotational` transports `Q` by the flow, and
    /// `beris_edwards_stress` assembles the force the elastic energy exerts on
    /// the flow, and the two are built independently from different
    /// discretisations of the same gradients. A discrete energy law needs the
    /// second to be the exact adjoint of the first, so that the power the elastic
    /// stress delivers to the fluid cancels the energy the transport removes from
    /// `Q`, term by term and not merely in the continuum limit. Where the mesh is
    /// mild the mismatch is a small residual; where it is graded a thousand to
    /// one it is unbounded.
    ///
    /// This is the acceptance criterion for that work. Ignored rather than
    /// deleted, so the defect stays visible and the fix has a test to satisfy.
    /// Removing the `ignore` is the last step of the fix, not the first.
    #[test]
    #[ignore = "no discrete energy law on a strongly graded mesh; the elastic stress is not the                 adjoint of the transport operator. This is the acceptance test for that fix."]
    fn the_passive_system_does_not_gain_free_energy_on_a_graded_mesh() {
        energy_law_at(0.99, 1e-4);
        energy_law_at(0.99, 5e-5);
    }

    /// The same law through the ADJOINT path.
    ///
    /// `elastic_force` is the transpose of `transport_rate` against INTERIOR test
    /// functions, `free_energy_fem` is the functional `-2 w h` differentiates, and
    /// `solve_force_warm` has a symmetric positive operator. Together those force
    /// `dF/dt|transport = -<f, L^-1 f> <= 0`, so the passive system cannot gain
    /// free energy.
    ///
    /// Measured 2026-08-22 at `d = 0.9`, predicted rate `-3.808387e7` against
    /// realised `-3.808385e7`, `-3.808387e7`, `-3.808380e7` at `dt` 1e-8, 1e-9 and
    /// 1e-10: ratio `+1.0000` across three decades.
    ///
    /// `d = 0.99` is excluded and the reason is the time step rather than the
    /// energy structure. `h_min` there is 9.498e-4 and the first step already runs
    /// at Courant 3.75, while `step_active` with `sl = None` differences the
    /// advection and is stable only while `dt |u| < h`. See
    /// `diag_courant_trace_at_d99` and
    /// `~/planning/cgpo-reproduction/mesh-energy-law-2026-08-22.md`.
    #[test]
    fn the_adjoint_path_keeps_the_energy_law_on_a_graded_mesh() {
        for (d, dt, passes) in [
            (0.5_f64, 1e-4_f64, 1usize),
            (0.9, 1e-4, 1),
            (0.9, 1e-5, 1),
        ] {
            energy_law_adjoint_at(d, dt, passes);
        }
    }

    fn energy_law_adjoint_at(d: f64, dt: f64, passes: usize) {
        let mut p = problem(2.0, d, 1.0, 2.0);
        p.params.zeta = 0.0;
        let mut q = p.random_state(7);
        for _ in 0..40 {
            p.step_passive(&mut q, 1e-3, 1e-8);
        }
        let stokes = crate::stokes::SurfaceStokes::new_confined(
            &p.ops,
            &p.mesh.mesh,
            &p.mesh.boundary_vertices,
        )
        .unwrap();
        let nv = q.n_vertices;
        let mut f_prev = p.free_energy_fem(&q);
        let f_start = f_prev;
        let mut worst = 0.0_f64;
        for _ in 0..20 {
            // Evaluate the elastic force at the current iterate rather than at the
            // old state, so the potential is seen at the end of the step. With the
            // force and the Stokes operator finally consistent, the map is the
            // gradient of a convex functional and the iteration should contract.
            let mut iterate = q.clone();
            let mut v2 = vec![[0.0_f64; 2]; nv];
            for _ in 0..passes {
                let force = p.elastic_force(&iterate);
                let (vel, _psi, _its) =
                    stokes.solve_force_warm(&force, p.params.eta, &p.mesh.mesh, None, 1e-10);
                for i in 0..nv {
                    v2[i] = [vel.v[i][0], vel.v[i][1]];
                }
                let mut trial = q.clone();
                p.step_active(&mut trial, &v2, dt, 1e-8, None);
                iterate = trial;
            }
            q = iterate;
            let f = p.free_energy_fem(&q);
            if f.is_finite() && f_prev.is_finite() {
                worst = worst.max((f - f_prev) / f_prev.abs().max(1e-30));
            } else {
                worst = f64::INFINITY;
            }
            f_prev = f;
        }
        eprintln!(
            "adjoint path, d = {d}, dt = {dt:.0e}: F {f_start:.6e} -> {f_prev:.6e}, \
             worst single-step relative gain {worst:.3e}"
        );
        assert!(
            worst < 1e-9,
            "adjoint path at d = {d}, dt = {dt:.0e}: the passive system gained free energy, \
             worst relative gain {worst:.3e}"
        );
    }

    /// DIAGNOSTIC (2026-08-22): is the discrete energy gradient the molecular field?
    ///
    /// `elastic_force` is built on the claim `dF/dq1_i = -2 w_i h1_i` exactly.
    /// But `free_energy` forms `|grad q|^2` from `q_gradients`, the vertex-averaged
    /// P1 gradients, while `molecular_field` uses `apply_laplace_beltrami`. Those
    /// are different discretisations, so the claim needs measuring rather than
    /// assuming. Central-difference the discrete `free_energy` along an
    /// interior-only direction and compare.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_energy_gradient_is_minus_two_w_h() {
        for d in [0.5_f64, 0.9, 0.99] {
            let p = problem(2.0, d, 1.0, 2.0);
            let mut q = p.random_state(3);
            // Control: the energy-law tests smooth first, so measure on the state
            // they actually use. `SMOOTH=0` in the environment keeps the raw field.
            let smooth = std::env::var("DIAG_SMOOTH").map(|v| v != "0").unwrap_or(true);
            if smooth {
                for _ in 0..40 {
                    p.step_passive(&mut q, 1e-3, 1e-8);
                }
            }
            eprintln!("--- d = {d}, smoothed = {smooth} ---");
            let nv = q.n_vertices;
            let mut is_boundary = vec![false; nv];
            for &v in &p.mesh.boundary_vertices {
                is_boundary[v] = true;
            }
            let h = p.molecular_field(&q);

            // The two lumped masses in play, checked against each other first.
            let ones = vec![1.0_f64; nv];
            let mut w_apply = vec![0.0_f64; nv];
            p.apply_mass(&ones, &mut w_apply);
            let mass_rel = (0..nv)
                .map(|i| (w_apply[i] - p.mass_lumped[i]).abs() / p.mass_lumped[i].abs().max(1e-300))
                .fold(0.0_f64, f64::max);

            let mut st = 12345_u64;
            let mut rnd = || {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            };
            let mut d1 = vec![0.0_f64; nv];
            let mut d2 = vec![0.0_f64; nv];
            let mut n_int = 0usize;
            for i in 0..nv {
                if !is_boundary[i] {
                    d1[i] = rnd();
                    d2[i] = rnd();
                    n_int += 1;
                }
            }

            let analytic: f64 = (0..nv)
                .map(|i| -2.0 * p.mass_lumped[i] * (h.q1[i] * d1[i] + h.q2[i] * d2[i]))
                .sum();

            // The same directional derivative of the FEM form of the same energy,
            // `sum_T A_T |grad q|_T^2` rather than `sum_i w_i |grad q|_avg,i^2`.
            let fem = |qq: &QField| -> f64 {
                let m = &p.mesh.mesh;
                let (k, a, c) = (
                    p.params.k_frank,
                    p.params.a_landau,
                    p.params.c_landau,
                );
                let mut dir = 0.0_f64;
                for t in 0..m.n_simplices() {
                    let sv = m.simplices[t];
                    let (p0, p1, p2) =
                        (m.vertices[sv[0]], m.vertices[sv[1]], m.vertices[sv[2]]);
                    let two_a = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                    if two_a.abs() < 1e-30 {
                        continue;
                    }
                    let inv = 1.0 / two_a;
                    let g = [
                        [(p1.y - p2.y) * inv, (p2.x - p1.x) * inv],
                        [(p2.y - p0.y) * inv, (p0.x - p2.x) * inv],
                        [(p0.y - p1.y) * inv, (p1.x - p0.x) * inv],
                    ];
                    let mut dd = [0.0_f64; 4];
                    for aa in 0..3 {
                        dd[0] += qq.q1[sv[aa]] * g[aa][0];
                        dd[1] += qq.q1[sv[aa]] * g[aa][1];
                        dd[2] += qq.q2[sv[aa]] * g[aa][0];
                        dd[3] += qq.q2[sv[aa]] * g[aa][1];
                    }
                    let area = 0.5 * two_a.abs();
                    let grad2 = 2.0 * (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2] + dd[3] * dd[3]);
                    dir += area * 0.5 * k * grad2;
                }
                // Bulk, lumped exactly as `free_energy` lumps it.
                let ones = vec![1.0_f64; nv];
                let mut w = vec![0.0_f64; nv];
                p.apply_mass(&ones, &mut w);
                for i in 0..nv {
                    let tr = 2.0 * (qq.q1[i] * qq.q1[i] + qq.q2[i] * qq.q2[i]);
                    dir += w[i] * (0.5 * a * tr + 0.25 * c * tr * tr);
                }
                dir
            };

            // Central difference, with a step ladder so the answer is not an
            // artefact of one epsilon.
            for eps in [1e-5_f64, 1e-6, 1e-7] {
                let mut qp = q.clone();
                let mut qm = q.clone();
                for i in 0..nv {
                    qp.q1[i] += eps * d1[i];
                    qp.q2[i] += eps * d2[i];
                    qm.q1[i] -= eps * d1[i];
                    qm.q2[i] -= eps * d2[i];
                }
                let fd = (p.free_energy(&qp) - p.free_energy(&qm)) / (2.0 * eps);
                let fd_fem = (fem(&qp) - fem(&qm)) / (2.0 * eps);
                let rel =
                    (fd - analytic).abs() / fd.abs().max(analytic.abs()).max(1e-300);
                let rel_fem = (fd_fem - analytic).abs()
                    / fd_fem.abs().max(analytic.abs()).max(1e-300);
                eprintln!(
                    "d = {d}, nv = {nv}, int = {n_int}, mass rel = {mass_rel:.1e}, \
                     eps = {eps:.0e}\n    free_energy : fd = {fd:.8e}  rel = {rel:.3e}\n\
                         FEM form    : fd = {fd_fem:.8e}  rel = {rel_fem:.3e}\n\
                         -2<w h, del>: {analytic:.8e}"
                );
            }
        }
    }

    /// DIAGNOSTIC (2026-08-22): the energy law, measured on both functionals.
    ///
    /// `energy_law_at` monitors `free_energy`, whose gradient is not the molecular
    /// field. `free_energy_fem` is the functional `-2 w h` actually differentiates.
    /// If the reported gain is an artefact of monitoring the wrong quantity it
    /// disappears from the second column and not the first.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_energy_law_on_both_functionals() {
        for (d, dt) in [
            (0.5_f64, 1e-4_f64),
            (0.9, 1e-4),
            (0.99, 1e-4),
            (0.99, 5e-5),
            (0.99, 1e-5),
        ] {
            let mut p = problem(2.0, d, 1.0, 2.0);
            p.params.zeta = 0.0;
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let (mut fa_prev, mut fb_prev) = (p.free_energy(&q), p.free_energy_fem(&q));
            let (fa0, fb0) = (fa_prev, fb_prev);
            let (mut worst_a, mut worst_b) = (0.0_f64, 0.0_f64);
            for _ in 0..20 {
                let (s1, s2, sa) = p.beris_edwards_stress(&q);
                let (vel, _psi, _its) = stokes.solve_stress_warm(
                    &s1, &s2, &sa, p.params.eta, &p.mesh.mesh, None, 1e-10,
                );
                let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
                p.step_active(&mut q, &v2, dt, 1e-8, None);
                let (fa, fb) = (p.free_energy(&q), p.free_energy_fem(&q));
                worst_a = worst_a.max((fa - fa_prev) / fa_prev.abs().max(1e-30));
                worst_b = worst_b.max((fb - fb_prev) / fb_prev.abs().max(1e-30));
                fa_prev = fa;
                fb_prev = fb;
            }
            eprintln!(
                "d = {d}, dt = {dt:.0e}  stress path\n\
                 \x20   free_energy     {fa0:.6e} -> {fa_prev:.6e}   worst gain {worst_a:.3e}\n\
                 \x20   free_energy_fem {fb0:.6e} -> {fb_prev:.6e}   worst gain {worst_b:.3e}"
            );
        }
    }

    /// DIAGNOSTIC (2026-08-22): the same on the ADJOINT path, which is the one
    /// the energy law is supposed to hold for.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_adjoint_energy_law_on_both_functionals() {
        for (d, dt) in [
            (0.5_f64, 1e-4_f64),
            (0.9, 1e-4),
            (0.99, 1e-4),
            (0.99, 1e-5),
            (0.99, 1e-6),
        ] {
            let mut p = problem(2.0, d, 1.0, 2.0);
            p.params.zeta = 0.0;
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let (mut fa_prev, mut fb_prev) = (p.free_energy(&q), p.free_energy_fem(&q));
            let (fa0, fb0) = (fa_prev, fb_prev);
            let (mut worst_a, mut worst_b) = (0.0_f64, 0.0_f64);
            for _ in 0..20 {
                let force = p.elastic_force(&q);
                let (vel, _psi, _its) =
                    stokes.solve_force_warm(&force, p.params.eta, &p.mesh.mesh, None, 1e-10);
                let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
                p.step_active(&mut q, &v2, dt, 1e-8, None);
                let (fa, fb) = (p.free_energy(&q), p.free_energy_fem(&q));
                if !fa.is_finite() || !fb.is_finite() {
                    worst_a = f64::INFINITY;
                    worst_b = f64::INFINITY;
                    break;
                }
                worst_a = worst_a.max((fa - fa_prev) / fa_prev.abs().max(1e-30));
                worst_b = worst_b.max((fb - fb_prev) / fb_prev.abs().max(1e-30));
                fa_prev = fa;
                fb_prev = fb;
            }
            eprintln!(
                "d = {d}, dt = {dt:.0e}  ADJOINT path\n\
                 \x20   free_energy     {fa0:.6e} -> {fa_prev:.6e}   worst gain {worst_a:.3e}\n\
                 \x20   free_energy_fem {fb0:.6e} -> {fb_prev:.6e}   worst gain {worst_b:.3e}"
            );
        }
    }

    /// DIAGNOSTIC (2026-08-22): per-step trace at the grading that fails, with the
    /// Courant number alongside, to separate a CFL blow-up of the explicit
    /// advection from a missing energy law.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_courant_trace_at_d99() {
        let d = 0.99_f64;
        let mut p = problem(2.0, d, 1.0, 2.0);
        p.params.zeta = 0.0;
        let m = &p.mesh.mesh;
        let mut h_min = f64::INFINITY;
        for t in 0..m.n_simplices() {
            let sv = m.simplices[t];
            for a in 0..3 {
                let (u, v) = (m.vertices[sv[a]], m.vertices[sv[(a + 1) % 3]]);
                let e = ((u.x - v.x).powi(2) + (u.y - v.y).powi(2)).sqrt();
                if e > 0.0 && e < h_min {
                    h_min = e;
                }
            }
        }
        eprintln!("d = {d}, h_min (shortest edge) = {h_min:.4e}");

        for dt in [1e-4_f64, 1e-6, 1e-8] {
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let mut f_prev = p.free_energy_fem(&q);
            eprintln!("  dt = {dt:.0e}   F0 = {f_prev:.6e}");
            for step in 0..12 {
                let force = p.elastic_force(&q);
                let (vel, _psi, _its) =
                    stokes.solve_force_warm(&force, p.params.eta, &p.mesh.mesh, None, 1e-10);
                let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
                let umax = v2
                    .iter()
                    .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
                    .fold(0.0_f64, f64::max);
                let courant = dt * umax / h_min;
                p.step_active(&mut q, &v2, dt, 1e-8, None);
                let f = p.free_energy_fem(&q);
                let gain = (f - f_prev) / f_prev.abs().max(1e-30);
                eprintln!(
                    "    step {step:2}  |u|max = {umax:.4e}  Courant = {courant:.4e}  \
                     F = {f:.6e}  gain = {gain:+.3e}"
                );
                f_prev = f;
                if !f.is_finite() {
                    break;
                }
            }
        }
    }

    /// DIAGNOSTIC (2026-08-22): the semi-discrete energy balance, term by term.
    ///
    /// Adjointness and `dF/dq = -2 w h` together force
    /// `dF/dt|transport = -<f, u> = -<f, L^-1 f> <= 0`, so the semi-discrete
    /// scheme cannot inject energy. Measure the predicted rate against the rate
    /// the step actually realises as `dt -> 0`. If they part company the stepped
    /// operator is not the analysed one.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_semidiscrete_energy_balance() {
        for d in [0.9_f64, 0.99] {
            let mut p = problem(2.0, d, 1.0, 2.0);
            p.params.zeta = 0.0;
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let g = p.params.gamma;

            let f = p.elastic_force(&q);
            let (vel, _psi, _its) =
                stokes.solve_force_warm(&f, p.params.eta, &p.mesh.mesh, None, 1e-10);
            let u: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();

            let power: f64 = (0..nv).map(|i| f[i][0] * u[i][0] + f[i][1] * u[i][1]).sum();
            let h = p.molecular_field(&q);
            let t = p.transport_rate(&q, &u);
            let rate_transport: f64 = (0..nv)
                .map(|i| -2.0 * p.mass_lumped[i] * (h.q1[i] * t.q1[i] + h.q2[i] * t.q2[i]))
                .sum();
            // The anchored vertices are pinned rather than relaxed, so the rate
            // must be read over the interior only.
            let mut is_b = vec![false; nv];
            for &v in &p.mesh.boundary_vertices {
                is_b[v] = true;
            }
            let rate_relax: f64 = (0..nv)
                .filter(|&i| !is_b[i])
                .map(|i| {
                    -2.0 * p.mass_lumped[i] * (h.q1[i] * h.q1[i] + h.q2[i] * h.q2[i]) / g
                })
                .sum();
            let rate_relax_all: f64 = (0..nv)
                .map(|i| {
                    -2.0 * p.mass_lumped[i] * (h.q1[i] * h.q1[i] + h.q2[i] * h.q2[i]) / g
                })
                .sum();
            let rate_transport_int: f64 = (0..nv)
                .filter(|&i| !is_b[i])
                .map(|i| -2.0 * p.mass_lumped[i] * (h.q1[i] * t.q1[i] + h.q2[i] * t.q2[i]))
                .sum();
            let predicted = rate_transport_int + rate_relax;
            eprintln!(
                "  interior-only: rate_transport {rate_transport_int:+.6e}                   rate_relax {rate_relax:+.6e}  (all-vertex relax {rate_relax_all:+.6e})"
            );

            eprintln!("--- d = {d} ---");
            eprintln!(
                "  <f,u> = {power:+.6e}   rate_transport = {rate_transport:+.6e}  \
                 (must be -<f,u>, rel {:.2e})",
                (rate_transport + power).abs() / power.abs().max(1e-300)
            );
            eprintln!("  rate_relax = {rate_relax:+.6e}   predicted dF/dt = {predicted:+.6e}");

            let f0 = p.free_energy_fem(&q);
            for dt in [1e-8_f64, 1e-9, 1e-10] {
                let mut qq = q.clone();
                let (_w, its) = p.step_active(&mut qq, &u, dt, 1e-12, None);
                let realised = (p.free_energy_fem(&qq) - f0) / dt;
                eprintln!(
                    "  dt = {dt:.0e}: realised dF/dt = {realised:+.6e}   \
                     ratio to predicted = {:+.4}   cg its = {its}",
                    realised / predicted
                );
            }
        }
    }

    /// DIAGNOSTIC (2026-08-22): the energy law through the CONSTRAINED adjoint.
    #[test]
    #[ignore = "diagnostic, run explicitly"]
    fn diag_constrained_adjoint_energy_law() {
        for d in [0.5_f64, 0.9, 0.99] {
            let mut p = problem(2.0, d, 1.0, 2.0);
            p.params.zeta = 0.0;
            let mut q = p.random_state(7);
            for _ in 0..40 {
                p.step_passive(&mut q, 1e-3, 1e-8);
            }
            let stokes = crate::stokes::SurfaceStokes::new_confined(
                &p.ops,
                &p.mesh.mesh,
                &p.mesh.boundary_vertices,
            )
            .unwrap();
            let nv = q.n_vertices;
            let mut is_b = vec![false; nv];
            for &v in &p.mesh.boundary_vertices {
                is_b[v] = true;
            }
            let mm = &p.mesh.mesh;
            let mut h_min = f64::INFINITY;
            for t in 0..mm.n_simplices() {
                let sv = mm.simplices[t];
                for a in 0..3 {
                    let (x, y) = (mm.vertices[sv[a]], mm.vertices[sv[(a + 1) % 3]]);
                    let e = ((x.x - y.x).powi(2) + (x.y - y.y).powi(2)).sqrt();
                    if e > 0.0 && e < h_min {
                        h_min = e;
                    }
                }
            }

            // Balance check first: the constrained pairing must equal <f,u> when
            // the transport is read over the interior only.
            let fc = p.elastic_force(&q);
            let (vel, _psi, _its) =
                stokes.solve_force_warm(&fc, p.params.eta, &p.mesh.mesh, None, 1e-10);
            let u: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
            let power: f64 = (0..nv).map(|i| fc[i][0] * u[i][0] + fc[i][1] * u[i][1]).sum();
            let h = p.molecular_field(&q);
            let t = p.transport_rate(&q, &u);
            let pairing_int: f64 = (0..nv)
                .filter(|&i| !is_b[i])
                .map(|i| 2.0 * p.mass_lumped[i] * (h.q1[i] * t.q1[i] + h.q2[i] * t.q2[i]))
                .sum();
            eprintln!(
                "--- d = {d} ---\n  <f_c,u> = {power:+.6e}   2<w h, T(u)>_interior = \
                 {pairing_int:+.6e}   rel = {:.3e}",
                (power - pairing_int).abs() / power.abs().max(1e-300)
            );

            for dt in [1e-4_f64, 1e-5] {
                let mut qq = q.clone();
                let mut f_prev = p.free_energy_fem(&qq);
                let f0 = f_prev;
                let mut worst = 0.0_f64;
                let mut nsteps = 0usize;
                for _ in 0..20 {
                    let force = p.elastic_force(&qq);
                    let (vel, _psi, _its) =
                        stokes.solve_force_warm(&force, p.params.eta, &p.mesh.mesh, None, 1e-10);
                    let v2: Vec<[f64; 2]> =
                        (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
                    let umax = v2
                        .iter()
                        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
                        .fold(0.0_f64, f64::max);
                    if nsteps < 4 {
                        eprintln!(
                            "      step {nsteps}: |u|max = {umax:.4e}  Courant = {:.4e}  \
                             F = {f_prev:.6e}",
                            dt * umax / h_min
                        );
                    }
                    nsteps += 1;
                    p.step_active(&mut qq, &v2, dt, 1e-10, None);
                    let f = p.free_energy_fem(&qq);
                    if !f.is_finite() {
                        worst = f64::INFINITY;
                        break;
                    }
                    worst = worst.max((f - f_prev) / f_prev.abs().max(1e-30));
                    f_prev = f;
                }
                eprintln!(
                    "  dt = {dt:.0e}: F_fem {f0:.6e} -> {f_prev:.6e}   worst gain {worst:.3e}"
                );
            }
        }
    }

    fn energy_law_at(d: f64, dt: f64) {
        // Passive: the activity is off, so only the elastic stress drives the flow.
        let mut p = problem(2.0, d, 1.0, 2.0);
        p.params.zeta = 0.0;
        let mut q = p.random_state(7);
        for _ in 0..40 {
            p.step_passive(&mut q, 1e-3, 1e-8);
        }

        let stokes = crate::stokes::SurfaceStokes::new_confined(
            &p.ops,
            &p.mesh.mesh,
            &p.mesh.boundary_vertices,
        )
        .unwrap();
        let nv = q.n_vertices;

        let mut f_prev = p.free_energy_fem(&q);
        let f_start = f_prev;
        let mut worst_gain = 0.0_f64;
        for _ in 0..20 {
            let (s1, s2, sa) = p.beris_edwards_stress(&q);
            let (vel, _psi, _its) =
                stokes.solve_stress_warm(&s1, &s2, &sa, p.params.eta, &p.mesh.mesh, None, 1e-10);
            let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.v[i][0], vel.v[i][1]]).collect();
            p.step_active(&mut q, &v2, dt, 1e-8, None);
            let f = p.free_energy_fem(&q);
            worst_gain = worst_gain.max((f - f_prev) / f_prev.abs().max(1e-30));
            f_prev = f;
        }
        eprintln!(
            "d = {d}, dt = {dt:.0e}: F {f_start:.6e} -> {f_prev:.6e}, \
             worst single-step relative gain {worst_gain:.3e}"
        );
        assert!(
            worst_gain < 1e-9,
            "at d = {d}, dt = {dt:.0e}: the passive system gained free energy, worst relative \
             gain {worst_gain:.3e}. \
             The elastic stress is not the discrete adjoint of the transport operator, so the \
             scheme has no energy law."
        );
    }

    /// The stress algebra against values produced by `flow-solver.py`'s own
    /// `calculate_Pi`, at one interior point of a random field.
    ///
    /// This is the only test here that can catch a transcription error in the
    /// formula, because the expected values come from the reference rather than
    /// from a restatement of the same algebra. The full-field comparison behind
    /// them agreed at relative 1e-16 on all three components; the numbers below
    /// are point (5, 7) of that comparison, seed 0.
    #[test]
    fn stress_matches_the_reference_calculate_pi() {
        let (q1, q2) = (-5.565_867_409_637_493_6e-3, 4.167_359_036_851_288_6e-1);
        let (h1, h2) = (7.581_374_500_747_724e-1, -2.094_337_939_869_166e-1);
        let dq = [
            -1.208_157_937_191_461_2e-1,
            2.087_185_755_204_356e-1,
            -5.015_025_125_751_170_2e-1,
            -3.445_836_672_342_819_7e-2,
        ];
        let (k, lambda, zeta) = (1.6384e4, 1.0, 1.337_469_387_755_102_2e3);
        let want = (
            -3.619_917_949_742_146_5e3,
            -2.972_064_724_868_323_5e2,
            -6.295_548_292_920_421e-1,
        );
        let got = stress_at_vertex(q1, q2, h1, h2, dq, k, lambda, zeta);
        for (name, g, w) in [
            ("Pi_S,xx", got.0, want.0),
            ("Pi_S,xy", got.1, want.1),
            ("Pi_A,xy", got.2, want.2),
        ] {
            let rel = (g - w).abs() / w.abs().max(1e-300);
            assert!(
                rel < 1e-12,
                "{name}: got {g:.17e}, reference {w:.17e}, relative {rel:.3e}"
            );
        }
    }

    /// With no elasticity and no molecular field the stress must collapse to the
    /// active term alone, which is what the solver carried before 2026-08-19.
    /// Pins the sign of `-zeta Q` and confirms the antisymmetric part vanishes
    /// when `Q` and `H` are parallel.
    #[test]
    fn stress_reduces_to_the_active_term_when_the_elastic_terms_vanish() {
        let zeta = 3.0;
        let dq = [0.0; 4];
        let (s1, s2, sa) = stress_at_vertex(0.2, -0.5, 0.0, 0.0, dq, 0.0, 0.0, zeta);
        assert!((s1 - (-zeta * 0.2)).abs() < 1e-15, "got {s1}");
        assert!((s2 - (-zeta * -0.5)).abs() < 1e-15, "got {s2}");
        assert!(
            sa.abs() < 1e-15,
            "antisymmetric part should vanish, got {sa}"
        );
    }

    /// Each elastic term must reach the answer. Dropping any one of them is an
    /// easy regression and none of the structural tests above would see it, since
    /// they all sit at the point where the term is switched off.
    #[test]
    fn every_elastic_term_moves_the_stress() {
        let (q1, q2, h1, h2) = (0.3, -0.2, 0.7, 0.4);
        let dq = [0.5, -0.3, 0.2, 0.6];
        let base = stress_at_vertex(q1, q2, h1, h2, dq, 1.5, 2.0, 3.0);

        // lambda H, by switching lambda off.
        let no_lambda = stress_at_vertex(q1, q2, h1, h2, dq, 1.5, 0.0, 3.0);
        assert!(
            (base.0 - no_lambda.0).abs() > 1e-9,
            "the -lambda H term must act"
        );

        // The Ericksen stress, by switching K off.
        let no_k = stress_at_vertex(q1, q2, h1, h2, dq, 0.0, 2.0, 3.0);
        assert!(
            (base.0 - no_k.0).abs() > 1e-9,
            "the Ericksen term must act on Pi_xx"
        );
        assert!(
            (base.1 - no_k.1).abs() > 1e-9,
            "the Ericksen term must act on Pi_xy"
        );

        // Tr(QH) Q, isolated by switching lambda and K off so that only
        // -zeta Q and Tr(QH) Q survive. Comparing two cases that both already
        // have a vanishing trace does NOT isolate it: an earlier version of this
        // check did exactly that and passed with the term deleted.
        let only_trqh = stress_at_vertex(q1, q2, h1, h2, dq, 0.0, 0.0, 3.0);
        assert!(
            (only_trqh.0 - (-3.0 * q1)).abs() > 1e-9,
            "the Tr(QH) Q term must act, got {} against the active-only {}",
            only_trqh.0,
            -3.0 * q1
        );

        // The antisymmetric part, which is nonzero whenever Q and H are not
        // parallel and is the term most easily lost in a refactor, since the
        // symmetric solver path has nowhere to put it.
        assert!(
            base.2.abs() > 1e-9,
            "Pi_A must be nonzero for non-parallel Q and H"
        );
    }

    use super::*;
    use crate::confined::{Epitrochoid, MeshOpts, confined_mesh};

    /// A passive problem with the core resolved by the mesh.
    ///
    /// `ncl` has to sit above about twice the bulk element for the defect count to
    /// mean anything: the per-triangle winding is exact only while the director
    /// turns by less than a quarter turn along an edge, and inside a core of width
    /// `ncl` on elements of size `h` that needs `h <~ ncl / 2`. Below it the
    /// winding sum stops telescoping and a core goes missing, which reads as a
    /// fractional total charge rather than as an obvious failure.
    /// A stream function whose Hessian is a constant must be recovered exactly.
    ///
    /// The fit is a quadratic least squares, so a quadratic `psi` sits in its own
    /// space and the answer is exact to rounding. That makes this the sharpest
    /// available check on the four entries and their signs: every one of
    ///
    /// ```text
    /// dx u_x = -psi_xy    dx u_y =  psi_xx
    /// dy u_x = -psi_yy    dy u_y =  psi_xy
    /// ```
    ///
    /// is pinned independently, since the three second derivatives are given
    /// three different values. A sign slipped anywhere, or the pair transposed,
    /// moves at least one entry by more than the tolerance.
    #[test]
    fn the_strain_from_the_stream_function_is_exact_on_a_quadratic() {
        let p = problem(2.0, 0.72, 1.0, 2.0);
        let m = &p.mesh.mesh;
        let nv = m.n_vertices();
        // psi = a x^2 / 2 + b x y + c y^2 / 2 + linear, so psi_xx = a, psi_xy = b,
        // psi_yy = c. Three distinct values, none of them zero, and a linear part
        // that must not leak into the second derivatives.
        let (a, b, c) = (0.7_f64, -0.3, 1.9);
        let psi: Vec<f64> = (0..nv)
            .map(|i| {
                let (x, y) = (m.vertices[i].x, m.vertices[i].y);
                0.5 * a * x * x + b * x * y + 0.5 * c * y * y + 2.5 * x - 1.25 * y + 4.0
            })
            .collect();
        let du = p.velocity_gradients_from_psi(&psi);
        let want = [-b, a, -c, b];
        let names = ["dx u_x", "dx u_y", "dy u_x", "dy u_y"];
        // Interior only: a boundary vertex's two-ring is one-sided, and the fit is
        // still exact there for a quadratic, but a vertex whose ring is degenerate
        // is left at zero by design and would read as a failure.
        let mut onb = vec![false; nv];
        for &v in &p.mesh.boundary_vertices {
            onb[v] = true;
        }
        let mut worst = 0.0_f64;
        for i in 0..nv {
            if onb[i] {
                continue;
            }
            for k in 0..4 {
                let e = (du[i][k] - want[k]).abs();
                assert!(
                    e < 1e-8,
                    "{} at vertex {i} ({:.2}, {:.2}): got {}, exact {}",
                    names[k],
                    m.vertices[i].x,
                    m.vertices[i].y,
                    du[i][k],
                    want[k]
                );
                worst = worst.max(e);
            }
        }
        eprintln!("quadratic stream function: worst entry error {worst:.3e}");
    }

    /// The recovered velocity gradient is divergence-free at every vertex, for
    /// ANY stream function, to rounding.
    ///
    /// This is a property of the construction rather than of the field:
    /// `dx u_x + dy u_y = -psi_xy + psi_xy`. The chained form, which differentiates
    /// a recovered velocity, has no such identity and does not satisfy it. A test
    /// on a smooth field alone would not separate the two, so the field here is
    /// deliberately rough.
    #[test]
    fn the_strain_from_the_stream_function_is_exactly_divergence_free() {
        let p = problem(2.0, 0.72, 1.0, 2.0);
        let m = &p.mesh.mesh;
        let nv = m.n_vertices();
        let mut st = 987_654_321_u64;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let psi: Vec<f64> = (0..nv).map(|_| rnd()).collect();
        let du = p.velocity_gradients_from_psi(&psi);
        let scale = du
            .iter()
            .flat_map(|d| d.iter())
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(scale > 1e-6, "the fixture produced no gradient at all");
        let worst = du
            .iter()
            .map(|d| (d[0] + d[3]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 1e-9 * scale,
            "divergence {worst:.3e} against a gradient scale of {scale:.3e}"
        );
        eprintln!("random stream function: worst divergence {worst:.3e}, scale {scale:.3e}");
    }

    /// The strain converges under refinement, and the chained form does not.
    ///
    /// This is the defect the whole change exists for. `corotational` is driven by
    /// `grad u`, and the production path used to build it by recovering `u` from
    /// `psi` with one vertex-gradient operator and then applying a second,
    /// different one. The recovery is accurate in `u`, but its error has grid-scale
    /// structure, and differentiating divides that structure by `h`.
    ///
    /// Asserting a small error at one resolution would NOT catch this: at `h = 2`
    /// the chained form is within sixteen per cent, which reads as merely coarse.
    /// Only the ratio across a refinement separates an inconsistent operator from
    /// an inaccurate one, so the assertion is on the ratio.
    ///
    /// Measured 2026-08-20 on the nephroid at `d = 0.72`, relative error in
    /// `E_xy` over the interior:
    ///
    /// ```text
    ///   h      chained     from psi
    ///   2      1.61e-1     6.88e-2
    ///   1      1.20e-1     3.08e-2
    ///   0.5    9.75e-2     1.49e-2
    /// ```
    #[test]
    fn the_strain_from_the_stream_function_converges_where_the_chained_form_does_not() {
        let (ka, kb) = (0.09_f64, 0.07);
        let mut errs = Vec::new();
        for h in [2.0_f64, 1.0] {
            let p = problem(2.0, 0.72, 1.0, h);
            let m = &p.mesh.mesh;
            let nv = m.n_vertices();
            let xy: Vec<[f64; 2]> = (0..nv)
                .map(|i| [m.vertices[i].x, m.vertices[i].y])
                .collect();
            let psi: Vec<f64> = xy
                .iter()
                .map(|c| (ka * c[0]).sin() * (kb * c[1]).sin())
                .collect();

            let du_psi = p.velocity_gradients_from_psi(&psi);
            // The chained form: recover u from psi with the P1 vertex gradient,
            // then differentiate it again with the same operator.
            let gq = p.q_gradients(&QField {
                q1: psi.clone(),
                q2: vec![0.0; nv],
                n_vertices: nv,
            });
            let u_rec: Vec<[f64; 2]> = (0..nv).map(|i| [-gq[i][1], gq[i][0]]).collect();
            let du_chain = p.velocity_gradients(&u_rec);

            let mut onb = vec![false; nv];
            for &v in &p.mesh.boundary_vertices {
                onb[v] = true;
            }
            let (mut ep, mut ec, mut den) = (0.0_f64, 0.0_f64, 0.0_f64);
            for i in 0..nv {
                if onb[i] {
                    continue;
                }
                let (sa, sb) = ((ka * xy[i][0]).sin(), (kb * xy[i][1]).sin());
                let exy = 0.5 * (kb * kb - ka * ka) * sa * sb;
                ep += (0.5 * (du_psi[i][1] + du_psi[i][2]) - exy).powi(2);
                ec += (0.5 * (du_chain[i][1] + du_chain[i][2]) - exy).powi(2);
                den += exy * exy;
            }
            errs.push(((ep / den).sqrt(), (ec / den).sqrt()));
            eprintln!(
                "  h = {h}: E_xy relative error, from psi {:.3e}, chained {:.3e}",
                errs.last().unwrap().0,
                errs.last().unwrap().1
            );
        }
        let (coarse_psi, coarse_chain) = errs[0];
        let (fine_psi, fine_chain) = errs[1];
        assert!(
            fine_psi < 0.5 * coarse_psi,
            "the strain from psi did not converge: {coarse_psi:.3e} to {fine_psi:.3e} \
             across a halving of h"
        );
        assert!(
            fine_chain > 0.6 * coarse_chain,
            "the chained form now converges ({coarse_chain:.3e} to {fine_chain:.3e}); if that is \
             a real improvement this test has served its purpose and should be rewritten, but \
             until then it means the fixture stopped exercising the defect"
        );
        assert!(
            fine_psi < 0.4 * fine_chain,
            "from psi {fine_psi:.3e} is not clearly better than chained {fine_chain:.3e}"
        );
    }

    fn problem_with(q: f64, d: f64, q_anchor: f64, h_bulk: f64, ncl: f64) -> LdgProblem {
        assert!(
            h_bulk <= ncl / 2.0,
            "h_bulk {h_bulk} too coarse for ncl {ncl}"
        );
        let curve = Epitrochoid { q, d, r: 98.0 };
        let mesh = confined_mesh(
            curve,
            MeshOpts {
                h_bulk,
                h_min: (curve.cusp_radius() / 4.0).min(h_bulk),
                ..Default::default()
            },
        );
        let params = NematicParams::klein(1.5, ncl, 100).passive();
        LdgProblem::new(mesh, params, q_anchor).expect("operators")
    }

    fn problem(q: f64, d: f64, q_anchor: f64, h_bulk: f64) -> LdgProblem {
        problem_with(q, d, q_anchor, h_bulk, 2.0 * h_bulk)
    }

    /// A uniform flow has no gradient, so it neither strains nor rotates `Q`.
    ///
    /// This is the test that catches a sign or an index slipped into the P1
    /// gradient: a wrong entry survives a symmetric field and shows up the moment
    /// the flow is translation, which is the one case where the answer must be
    /// exactly zero rather than merely small.
    #[test]
    fn a_uniform_flow_neither_strains_nor_rotates_q() {
        let p = problem(1.5, 0.5, 1.0, 2.0);
        let nv = p.mesh.mesh.n_vertices();
        let vel = vec![[0.7_f64, -0.3_f64]; nv];
        let du = p.velocity_gradients(&vel);
        let q = p.random_state(3);
        let s = p.corotational(&q, &du);
        // Interior vertices only: a boundary vertex sees a one-sided stencil, so
        // its recovered gradient is the average over a partial fan and is not
        // required to vanish.
        let mut interior = 0;
        for i in 0..nv {
            if p.mesh.boundary_vertices.contains(&i) {
                continue;
            }
            interior += 1;
            for k in 0..4 {
                assert!(du[i][k].abs() < 1e-9, "gradient {} at {i}: {}", k, du[i][k]);
            }
            assert!(s.q1[i].abs() < 1e-9 && s.q2[i].abs() < 1e-9);
        }
        assert!(
            interior > 100,
            "fixture degenerate: {interior} interior vertices"
        );
    }

    /// A rigid rotation turns `Q` at twice the angular rate and does not strain it.
    ///
    /// For `u = omega (-y, x)` the strain rate vanishes and `omega_xy = omega`, so
    /// the reference's co-rotation reduces to `S = 2 omega (-Qxy, Qxx)`, which is
    /// the derivative of a director turning at `omega`: the doubling is the spin-2
    /// character of `Q`, not a factor out of place. Checked against the closed
    /// form rather than against a stored number.
    #[test]
    fn a_rigid_rotation_turns_q_at_twice_the_angular_rate() {
        let p = problem(1.5, 0.5, 1.0, 2.0);
        let m = &p.mesh.mesh;
        let nv = m.n_vertices();
        let omega = 0.037_f64;
        // Centred on the mesh so the velocities stay of one scale.
        let (cx, cy) = {
            let mut sx = 0.0;
            let mut sy = 0.0;
            for v in &m.vertices {
                sx += v.x;
                sy += v.y;
            }
            (sx / nv as f64, sy / nv as f64)
        };
        let vel: Vec<[f64; 2]> = m
            .vertices
            .iter()
            .map(|v| [-omega * (v.y - cy), omega * (v.x - cx)])
            .collect();
        let du = p.velocity_gradients(&vel);
        let q = p.random_state(5);
        let s = p.corotational(&q, &du);
        let mut checked = 0;
        for i in 0..nv {
            if p.mesh.boundary_vertices.contains(&i) {
                continue;
            }
            checked += 1;
            // No strain: E = 0 means dx ux = dy uy = 0 and dx uy = -dy ux.
            assert!(du[i][0].abs() < 1e-9, "dx ux at {i}: {}", du[i][0]);
            assert!(du[i][3].abs() < 1e-9, "dy uy at {i}: {}", du[i][3]);
            assert!((du[i][1] + du[i][2]).abs() < 1e-9);
            let want1 = -2.0 * omega * q.q2[i];
            let want2 = 2.0 * omega * q.q1[i];
            assert!(
                (s.q1[i] - want1).abs() < 1e-9,
                "S1 at {i}: {} vs {want1}",
                s.q1[i]
            );
            assert!(
                (s.q2[i] - want2).abs() < 1e-9,
                "S2 at {i}: {} vs {want2}",
                s.q2[i]
            );
        }
        assert!(
            checked > 100,
            "fixture degenerate: {checked} interior vertices"
        );
    }

    /// A zero flow must transport nothing.
    ///
    /// IGNORED: this fails, and it is the minimal reproduction of a defect in
    /// [`SemiLagrangian`]'s point location on a flat confined mesh. Zero flow
    /// moves `Q` by 1.4139 at vertex 1042, which is the full magnitude `s0`, so
    /// the departure point is being located in the wrong triangle rather than in
    /// the one containing it. The operator was written for closed curved meshes
    /// (sphere, torus) and this is the first use on a planar domain with a
    /// boundary. Recorded rather than fixed because the confined active runs use
    /// the differenced advection, which the wall layer makes stable; see
    /// [`LdgProblem::step_active`].
    #[ignore = "reproduces a point-location defect in SemiLagrangian on flat meshes"]
    ///
    /// With `u = 0` the backward trace ends where it started, so locating that
    /// point and interpolating there has to return the vertex's own value. This
    /// isolates the point location and the interpolation from the splitting: if
    /// it fails, nothing downstream of it means anything.
    #[test]
    fn the_backward_trace_at_zero_flow_is_the_identity() {
        let p = problem(1.5, 0.5, 1.0, 2.0);
        let m = &p.mesh.mesh;
        let nv = m.n_vertices();
        let sl = SemiLagrangian::new(
            m.vertices.iter().map(|v| [v.x, v.y, 0.0]).collect(),
            m.simplices.clone(),
        );
        let q0 = p.random_state(23);
        let vel = VelocityField::zeros(nv);
        let out = sl.transport(&q0, &vel, 1e-3);
        let mut worst = 0.0_f64;
        let mut at = 0usize;
        for i in 0..nv {
            let d = (out.q1[i] - q0.q1[i])
                .abs()
                .max((out.q2[i] - q0.q2[i]).abs());
            if d > worst {
                worst = d;
                at = i;
            }
        }
        assert!(worst < 1e-9, "zero flow moved Q by {worst} at vertex {at}");
    }

    /// The backward trace and the differenced advection agree as the step falls.
    ///
    /// IGNORED: fails for the same reason as
    /// `the_backward_trace_at_zero_flow_is_the_identity`, which is the narrower
    /// statement of the same defect. 3142 of 3361 vertices disagree, none of them
    /// on the boundary, so this is not an edge effect.
    ///
    /// They are two discretisations of the same term, so the difference between
    /// them is the splitting and interpolation error and must shrink with `dt`.
    /// A term added to one path and not the other would leave a difference that
    /// does not shrink, which is what this rules out. The rate is not asserted:
    /// the semi-Lagrangian interpolation is first order in the departure point
    /// and the trace is fourth, so the observed rate is a mixture and pinning it
    /// would be asserting the mesh rather than the physics.
    #[test]
    #[ignore = "blocked by the SemiLagrangian point-location defect above"]
    fn the_two_advective_paths_agree_as_the_step_falls() {
        let p = problem(1.5, 0.5, 1.0, 2.0);
        let m = &p.mesh.mesh;
        let nv = m.n_vertices();
        let sl = SemiLagrangian::new(
            m.vertices.iter().map(|v| [v.x, v.y, 0.0]).collect(),
            m.simplices.clone(),
        );
        // A solid-body rotation about the mesh centre, slow enough that the
        // differenced form is stable at the larger step and so the two are being
        // compared where both are valid.
        let (cx, cy) = {
            let (mut sx, mut sy) = (0.0, 0.0);
            for v in &m.vertices {
                sx += v.x;
                sy += v.y;
            }
            (sx / nv as f64, sy / nv as f64)
        };
        let omega = 0.05_f64;
        let vel: Vec<[f64; 2]> = m
            .vertices
            .iter()
            .map(|v| [-omega * (v.y - cy), omega * (v.x - cx)])
            .collect();
        // A smooth field, not `random_state`. The random initial condition is an
        // independent director at every vertex, so it has no coherence length and
        // neither transport is consistent on it: interpolating white noise and
        // differencing white noise disagree at order one however small the step,
        // which says nothing about either scheme. The physical field acquires a
        // coherence length of `ncl` within the first few steps, and that is the
        // regime the two have to agree in.
        let q0 = {
            let s0 = p.params.s0();
            let mut f = QField::zeros(nv);
            for (i, v) in m.vertices.iter().enumerate() {
                let th = 0.01 * (v.x - cx) + 0.007 * (v.y - cy);
                let (c, sn) = (th.cos(), th.sin());
                f.q1[i] = s0 * (c * c - 0.5);
                f.q2[i] = s0 * c * sn;
            }
            p.impose_anchoring(&mut f);
            f
        };

        let mut diffs = Vec::new();
        for &dt in &[4e-3_f64, 1e-3] {
            let mut qa = q0.clone();
            let mut qb = q0.clone();
            p.step_active(&mut qa, &vel, dt, 1e-12, None);
            p.step_active(&mut qb, &vel, dt, 1e-12, Some(&sl));
            let mut worst = 0.0_f64;
            for i in 0..nv {
                worst = worst
                    .max((qa.q1[i] - qb.q1[i]).abs())
                    .max((qa.q2[i] - qb.q2[i]).abs());
            }
            diffs.push(worst);
        }
        // Where the disagreement lives, before asserting anything about it.
        {
            let dt = 1e-3_f64;
            let mut qa = q0.clone();
            let mut qb = q0.clone();
            p.step_active(&mut qa, &vel, dt, 1e-12, None);
            p.step_active(&mut qb, &vel, dt, 1e-12, Some(&sl));
            let hl = p.local_h();
            let onb: std::collections::HashSet<usize> =
                p.mesh.boundary_vertices.iter().copied().collect();
            let mut bad = Vec::new();
            for i in 0..nv {
                let d = (qa.q1[i] - qb.q1[i]).abs().max((qa.q2[i] - qb.q2[i]).abs());
                if d > 1e-3 {
                    bad.push((i, d, hl[i], onb.contains(&i)));
                }
            }
            let nb = bad.iter().filter(|b| b.3).count();
            println!(
                "vertices disagreeing by more than 1e-3: {} of {nv}, {} on the \
                 boundary; worst {:?}",
                bad.len(),
                nb,
                bad.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|b| (b.0, b.1, b.2, b.3))
            );
        }
        assert!(
            diffs[1] < diffs[0],
            "difference did not fall with dt: {:?}",
            diffs
        );
    }

    /// With the flow switched off the active step is the passive step.
    ///
    /// The active path adds advection and co-rotation to the same semi-implicit
    /// solve, so at zero velocity the two have to agree to the solver tolerance.
    /// This is what says the new terms were added to the right-hand side and did
    /// not disturb the operator.
    #[test]
    fn the_active_step_reduces_to_the_passive_one_at_zero_flow() {
        let p = problem(1.5, 0.5, 1.0, 2.0);
        let nv = p.mesh.mesh.n_vertices();
        let q0 = p.random_state(11);
        let mut qa = q0.clone();
        let mut qp = q0.clone();
        let vel = vec![[0.0_f64, 0.0_f64]; nv];
        p.step_active(&mut qa, &vel, 1e-3, 1e-12, None);
        p.step_passive(&mut qp, 1e-3, 1e-12);
        let mut worst = 0.0_f64;
        for i in 0..nv {
            worst = worst
                .max((qa.q1[i] - qp.q1[i]).abs())
                .max((qa.q2[i] - qp.q2[i]).abs());
        }
        assert!(
            worst < 1e-9,
            "active and passive differ by {worst} at zero flow"
        );
    }

    #[test]
    fn anchoring_has_the_equilibrium_amplitude() {
        // Q = S0 (m m - I/2) gives Tr(Q^2) = S0^2 / 2, so the reference's own S
        // diagnostic is S0 / sqrt(2), which is 1 at A = -C. Anything else means the
        // wall is not at equilibrium and would relax away from the anchoring.
        let p = problem(2.0, 0.9, 1.0, 4.0);
        let s0 = p.params.s0();
        for &(q1, q2) in &p.anchor {
            let tr = 2.0 * (q1 * q1 + q2 * q2);
            assert!((tr - s0 * s0 / 2.0).abs() < 1e-12, "Tr(Q^2) at the wall");
            assert!((tr.sqrt() - 1.0).abs() < 1e-12, "S at the wall");
        }
    }

    #[test]
    fn anchoring_at_q_one_is_tangential() {
        // At q = 1 the imposed director is the wall tangent. The director of
        // Q = S0 (m m - I/2) is the eigenvector of the larger eigenvalue, whose
        // angle is atan2(Qxy, Qxx) / 2.
        let p = problem(2.0, 0.7, 1.0, 4.0);
        for i in (0..p.anchor.len()).step_by(11) {
            let (q1, q2) = p.anchor[i];
            let director = 0.5 * q2.atan2(q1);
            let t = p.mesh.curve.tangent(p.mesh.boundary_params[i]);
            let tangent = t[1].atan2(t[0]);
            // Equal modulo pi, since a director has no arrow.
            let diff = (director - tangent).rem_euclid(std::f64::consts::PI);
            let off = diff.min(std::f64::consts::PI - diff);
            assert!(
                off < 1e-8,
                "vertex {i}: director {director}, tangent {tangent}"
            );
        }
    }

    #[test]
    fn a_uniform_field_has_no_molecular_field_from_elasticity() {
        // The Laplacian of a constant is zero, so H reduces to the bulk term, and
        // at the equilibrium amplitude the bulk term vanishes too.
        let p = problem(2.0, 0.7, 1.0, 6.0);
        let nv = p.mesh.mesh.n_vertices();
        let s0 = p.params.s0();
        let q = QField::uniform(nv, s0 * (1.0 - 0.5), 0.0);
        let h = p.molecular_field(&q);
        // Interior vertices only: the wall is anchored to a different direction and
        // is expected to carry a field.
        let mut worst = 0.0_f64;
        for i in 0..nv {
            if !p.pinned[i] {
                worst = worst.max(h.q1[i].abs()).max(h.q2[i].abs());
            }
        }
        // Tr(Q^2) = 2 (S0/2)^2 = S0^2/2 = 1, and A + C = 0, so the bulk term is
        // zero and only the boundary ring of the Laplacian contributes.
        assert!(
            worst < 1e-6 * p.params.k_frank,
            "worst interior |H| = {worst}"
        );
    }

    #[test]
    fn the_two_mass_matrices_agree_on_the_equilibrium_charge() {
        // The consistent mass changes the discrete inner product, not the physics,
        // so the topological answer has to be the same. Where they differ is
        // robustness: the lumped diagonal is only positive on a well-centred mesh,
        // and these meshes carry obtuse triangles.
        for (q, d) in [(2.0, 0.9), (2.5, 0.9)] {
            let lumped = problem(q, d, 1.0, 4.0);
            let mut a = lumped.random_state(3);
            lumped.relax(&mut a, 1e-3, 3000, 1e-9);
            let (_, _, ca, _) = lumped.defect_summary(&a, 4.0);

            let consistent = problem(q, d, 1.0, 4.0).with_consistent_mass();
            let mut b = consistent.random_state(3);
            consistent.relax(&mut b, 1e-3, 3000, 1e-9);
            let (_, _, cb, _) = consistent.defect_summary(&b, 4.0);

            assert!((ca - 1.0).abs() < 1e-9, "lumped charge {ca}");
            assert!((cb - 1.0).abs() < 1e-9, "consistent charge {cb}");
        }
    }

    #[test]
    fn consistent_mass_is_positive_definite_on_this_mesh() {
        // The property the lumped diagonal cannot promise. Tested by the Rayleigh
        // quotient on random vectors rather than by an eigenvalue solve, which is
        // enough to catch a sign error in the assembly.
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let p = problem(2.0, 0.95, 1.0, 4.0).with_consistent_mass();
        let n = p.mesh.mesh.n_vertices();
        let mut rng = StdRng::seed_from_u64(7);
        let mut worst = f64::INFINITY;
        for _ in 0..20 {
            let x: Vec<f64> = (0..n).map(|_| 2.0 * rng.random::<f64>() - 1.0).collect();
            let mut mx = vec![0.0; n];
            p.apply_mass(&x, &mut mx);
            let quad: f64 = x.iter().zip(&mx).map(|(a, b)| a * b).sum();
            worst = worst.min(quad);
        }
        assert!(worst > 0.0, "mass matrix is not positive definite: {worst}");
    }

    #[test]
    fn relaxation_reaches_the_anchoring_charge() {
        // The point of the whole exercise: the equilibrium defect charge equals the
        // anchoring winding, on both shapes and at a cusp the lattice cannot
        // represent.
        for (q, d, name) in [(2.0, 0.9, "nephroid"), (2.5, 0.9, "trefoiloid")] {
            let p = problem(q, d, 1.0, 4.0);
            let mut state = p.random_state(0);
            let (steps, last) = p.relax(&mut state, 1e-3, 4000, 1e-9);
            let (pos, neg, total, _) = p.defect_summary(&state, 4.0);
            assert!(
                (total - 1.0).abs() < 1e-9,
                "{name} d={d}: {pos} (+1/2) and {neg} (-1/2) give {total}, wanted 1 \
                 after {steps} steps (last change {last:.3e})"
            );
        }
    }
}
