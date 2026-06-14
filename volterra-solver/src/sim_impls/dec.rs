//! `PhysicsStep` implementations for the DEC (discrete exterior calculus)
//! runners on triangulated 2-manifolds.
//!
//! These reproduce the legacy `runner_dec.rs` / `runner_dec_wet.rs` loops
//! bit-for-bit:
//! - [`DecDry`]: RK4 of the dry Beris-Edwards RHS `gamma_r * H` (no flow).
//! - [`DecWet`]: a per-step Stokes solve for the velocity, then RK4 of the
//!   molecular-field + directional-advection RHS, with the velocity held fixed
//!   across the four RK4 stages (exactly as the legacy hand-rolled loop).
//!
//! The generic `volterra_core::sim::integrate::rk4` is used in place of the
//! legacy hand-rolled RK4; the two produce the identical op tree (proven by the
//! `rk4_matches_legacy_rk4_step_bit_for_bit` unit test in `volterra-dec`).
//!
//! DEC runners have **no Langevin noise**: the only numerics are the RK4 of the
//! respective RHS.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use volterra_core::sim::integrate::rk4;
use volterra_core::sim::stats::StepStats;
use volterra_core::sim::PhysicsStep;
use volterra_core::ActiveNematicParams;
use volterra_dec::stokes_dec::{advect_q, StokesSolverDec};
use volterra_dec::{molecular_field_dec, QFieldDec};

/// Dry active nematic on a 2D DEC mesh: RK4 of `dQ/dt = gamma_r * H` (no flow).
///
/// Reproduces the legacy `run_dry_active_nematic_dec` per-step physics
/// bit-for-bit. Borrows the precomputed operators and (optionally) the
/// curvature-correction endomorphism for the lifetime of the run.
pub struct DecDry<'a, M: Manifold> {
    /// Physics parameters (uses `dt`, `gamma_r`, and the molecular-field terms).
    pub params: ActiveNematicParams,
    /// Precomputed DEC operators (Laplace-Beltrami, Hodge stars).
    pub ops: &'a Operators<M, 3, 2>,
    /// Optional Weitzenboeck endomorphism for curved surfaces (`None` for flat).
    pub cc: Option<&'a dyn Fn(usize) -> [[f64; 3]; 3]>,
}

impl<M: Manifold> PhysicsStep for DecDry<'_, M> {
    type Field = QFieldDec;

    fn step(&mut self, q: &mut QFieldDec, _t: f64) -> StepStats {
        let params = &self.params;
        let ops = self.ops;
        let cc = self.cc;
        // dQ/dt = gamma_r * H, integrated with RK4. Same op tree as the legacy
        // `rk4_step` of `beris_edwards_rhs_dec`.
        *q = rk4(q, params.dt, |qq| {
            let h = molecular_field_dec(qq, params, ops, cc);
            h.scale(params.gamma_r)
        });
        StepStats::default().with_order_param(q.mean_order_param())
    }
}

/// Wet active nematic on a 2D DEC mesh: Stokes solve + RK4 of molecular field +
/// directional advection.
///
/// Reproduces the legacy `run_wet_active_nematic_dec` (and `_confined`) per-step
/// physics bit-for-bit:
/// 1. Solve Stokes once per step for the 3D tangent velocity.
/// 2. RK4-advance Q with RHS `gamma_r * H - advect(Q, vel)`, the velocity held
///    fixed across the four RK4 stages.
///
/// The [`StokesSolverDec`] is constructed by the wrapper (closed or confined)
/// and owned here; `ops`/`mesh` are borrowed and `coords` is the owned vertex
/// coordinate table the legacy runner built once via `extract_coords_runner`.
pub struct DecWet<'a, M: Manifold> {
    /// Physics parameters.
    pub params: ActiveNematicParams,
    /// The Stokes solver (closed via `new`, or confined via `new_confined`).
    pub stokes: StokesSolverDec,
    /// Precomputed DEC operators.
    pub ops: &'a Operators<M, 3, 2>,
    /// The mesh (needed for the Stokes solve and advection boundary data).
    pub mesh: &'a Mesh<M, 3, 2>,
    /// Vertex coordinate table (legacy `extract_coords_runner` output).
    pub coords: Vec<[f64; 3]>,
    /// Optional curvature-correction endomorphism (`None` for flat meshes).
    pub cc: Option<&'a dyn Fn(usize) -> [[f64; 3]; 3]>,
}

impl<M: Manifold> PhysicsStep for DecWet<'_, M> {
    type Field = QFieldDec;

    fn step(&mut self, q: &mut QFieldDec, _t: f64) -> StepStats {
        let params = &self.params;
        let ops = self.ops;
        let mesh = self.mesh;
        let cc = self.cc;
        let coords = &self.coords;

        // 1. Stokes solve: 3D tangent velocity, fixed across the RK4 stages.
        let vel = self.stokes.solve(q, params, ops, mesh);

        // 2. RK4 step: molecular field + directional advection. The RHS closure
        //    is preserved exactly from the legacy runner (gamma_r scaling,
        //    advect_q args, the per-component subtraction loop).
        *q = rk4(q, params.dt, |qq| {
            let h = molecular_field_dec(qq, params, ops, cc);
            let mut dq = h.scale(params.gamma_r);
            let adv = advect_q(
                qq,
                &vel,
                &mesh.boundaries,
                &mesh.vertex_boundaries,
                coords,
            );
            for i in 0..qq.n_vertices {
                dq.q1[i] -= adv.q1[i];
                dq.q2[i] -= adv.q2[i];
            }
            dq
        });

        StepStats::default().with_order_param(q.mean_order_param())
    }
}
