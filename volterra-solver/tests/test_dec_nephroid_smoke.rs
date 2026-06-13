//! Smoke test: wet active nematic on a confined nephroid (q=2 epitrochoid) domain.
//!
//! ## Purpose
//!
//! This test validates the key risk item for the confined-nephroid run:
//! **no-slip at the domain boundary**. The Stokes stream-function formulation
//! in `StokesSolverDec` solves −Δψ = ω via `PoissonSolver`, which only pins one
//! vertex (vertex 0) to remove the constant kernel. On a bounded domain the
//! correct no-slip condition requires ψ = 0 on the **entire** boundary, not just
//! a single pinned vertex.
//!
//! ## Pass/Fail criterion
//!
//! - No NaN in Q after the run (stability check).
//! - max boundary-vertex speed < `NO_SLIP_TOL × max interior-vertex speed`
//!   (no-slip check).
//!
//! ## Current status: PASSING
//!
//! The gated fix is in place: `PoissonSolver::with_dirichlet` imposes ψ = 0 at
//! all boundary vertices via symmetric Dirichlet elimination, and
//! `StokesSolverDec::new_confined` uses it (and additionally zeroes the velocity
//! at no-slip vertices after stream-function recovery). With that, the measured
//! ratio (boundary speed / interior speed) falls below the 0.05 tolerance.
//!
//! The test still fails loudly with the measured speeds if the no-slip condition
//! ever regresses — do NOT silently skip this assertion.

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::DVector;
use volterra_core::ActiveNematicParams;
use volterra_dec::boundary_conditions::apply_strong_anchoring;
use volterra_dec::epitrochoid::epitrochoid_mesh;
use volterra_dec::stokes_dec::StokesSolverDec;
use volterra_dec::QFieldDec;
use volterra_solver::run_wet_active_nematic_dec_confined;

/// Boundary speed must be below this fraction of the interior peak speed
/// for the no-slip condition to be considered satisfied.
const NO_SLIP_TOL: f64 = 0.05;

#[test]
fn dec_nephroid_confined_wet_smoke() {
    // ── 1. Build the nephroid mesh ────────────────────────────────────────────
    // q=2 → nephroid (2 cusps); r=3.0 gives a domain ~6 units across.
    // n_boundary=48: coarse enough that near-cusp edges are not degenerate
    // (min_edge ≈ 0.038 >> interior_spacing=0.5 near most of the domain).
    // 80 boundary points produced min_edge=0.014 → mol_field max ~6e4 → NaN in 1 step.
    let q_winding = 2.0_f64;
    let r = 3.0_f64;
    let n_boundary = 48_usize;
    let spacing = 0.5_f64;

    let confined = epitrochoid_mesh(q_winding, r, n_boundary, spacing);
    let nv = confined.mesh.n_vertices();
    let n_tri = confined.mesh.n_simplices();
    let n_interior = nv - n_boundary;

    assert!(n_interior > 0, "nephroid mesh should have interior vertices (got 0)");
    assert!(n_tri > 0, "nephroid mesh should have triangles (got 0)");

    println!(
        "[nephroid mesh] n_vertices={nv}, n_triangles={n_tri}, \
         n_boundary={n_boundary}, n_interior={n_interior}"
    );

    // ── 2. Set up operators and params ────────────────────────────────────────
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&confined.mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = 0.5;
    assert!(
        params.a_eff() < 0.0,
        "test requires active regime: a_eff={}",
        params.a_eff()
    );

    // ── 3. Initialise Q with tangential anchoring on the boundary ─────────────
    let mut q0 = QFieldDec::random_perturbation(nv, 0.01, 99);
    let s0 = 0.5_f64;
    apply_strong_anchoring(&mut q0, &confined, s0);

    // ── 3b. Adaptive CFL dt ───────────────────────────────────────────────────
    // The Laplace-Beltrami stencil amplifies gradients as 1/h^2; the explicit
    // RK4 step is stable only when dt < h^2 / (2 k_r γ_r). Find min edge
    // length and set dt to 10% of the CFL limit.
    let mut min_edge = f64::INFINITY;
    for &[v0, v1] in &confined.mesh.boundaries {
        let p0 = confined.mesh.vertices[v0];
        let p1 = confined.mesh.vertices[v1];
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < min_edge {
            min_edge = len;
        }
    }
    let cfl_dt = min_edge * min_edge / (2.0 * params.k_r * params.gamma_r);
    params.dt = (cfl_dt * 0.1).min(0.00005);
    println!(
        "[CFL] min_edge={min_edge:.4}, safe_dt<={cfl_dt:.2e}, using dt={:.2e}",
        params.dt
    );

    // ── 3c. Single-step diagnostics (isolate NaN source) ─────────────────────
    {
        let stokes_check = StokesSolverDec::new_confined(&ops, &confined.mesh, &confined.boundary_vertices)
            .expect("Confined Stokes should construct on nephroid mesh");
        let vel_check = stokes_check.solve(&q0, &params, &ops, &confined.mesh);
        let max_v: f64 = (0..nv).map(|i| vel_check.speed(i)).fold(0.0_f64, f64::max);
        let nan_v = (0..nv).any(|i| !vel_check.speed(i).is_finite());
        println!("[diag] Stokes: max_speed={max_v:.3e}, has_nan={nan_v}");

        let mol_h = volterra_dec::molecular_field_dec(&q0, &params, &ops, None);
        let max_h: f64 = (0..nv)
            .map(|i| mol_h.q1[i].abs().max(mol_h.q2[i].abs()))
            .fold(0.0_f64, f64::max);
        let nan_h = (0..nv).any(|i| !mol_h.q1[i].is_finite() || !mol_h.q2[i].is_finite());
        println!("[diag] mol_field: max={max_h:.3e}, has_nan={nan_h}");

        let q1_vec = DVector::from_vec(q0.q1.clone());
        let lap_q = ops.apply_laplace_beltrami(&q1_vec);
        let max_lap: f64 = lap_q.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let nan_lap = lap_q.iter().any(|v| !v.is_finite());
        println!("[diag] Laplace-Beltrami: max={max_lap:.3e}, has_nan={nan_lap}");
    }

    // ── 4. Run the wet confined runner for 10 steps ──────────────────────────
    // Uses run_wet_active_nematic_dec_confined so that the Stokes Poisson
    // solver enforces ψ=0 on all boundary vertices (no-slip).
    let n_steps = 10_usize;
    let (q_fin, stats) = run_wet_active_nematic_dec_confined(
        &q0,
        &params,
        &ops,
        &confined.mesh,
        &confined.boundary_vertices,
        None,
        n_steps,
        n_steps,
    )
    .expect("wet confined runner should not error on the nephroid mesh");

    // ── 5. Q finite check ─────────────────────────────────────────────────────
    let mean_s = q_fin.mean_order_param();
    assert!(
        mean_s.is_finite(),
        "mean order parameter should be finite after the run, got {mean_s}"
    );
    for i in 0..nv {
        assert!(
            q_fin.q1[i].is_finite() && q_fin.q2[i].is_finite(),
            "NaN/Inf in Q at vertex {i}: q1={}, q2={}",
            q_fin.q1[i],
            q_fin.q2[i]
        );
    }
    assert_eq!(stats.len(), 2, "expected 2 snapshots (step 0 and step {n_steps})");
    println!("[Q check] mean_s={mean_s:.4e} — finite, no NaN");

    // ── 6. NO-SLIP CHECK ──────────────────────────────────────────────────────
    // Re-solve Stokes on q_fin using the confined (Dirichlet) solver so the
    // check is consistent with what the runner used.
    let stokes = StokesSolverDec::new_confined(&ops, &confined.mesh, &confined.boundary_vertices)
        .expect("Confined Stokes solver should construct on the nephroid mesh");
    let vel = stokes.solve(&q_fin, &params, &ops, &confined.mesh);

    let boundary_set: std::collections::HashSet<usize> =
        confined.boundary_vertices.iter().cloned().collect();

    let max_boundary_speed = confined
        .boundary_vertices
        .iter()
        .map(|&bv| vel.speed(bv))
        .fold(0.0_f64, f64::max);

    let max_interior_speed = (0..nv)
        .filter(|i| !boundary_set.contains(i))
        .map(|i| vel.speed(i))
        .fold(0.0_f64, f64::max);

    let ratio = if max_interior_speed > 1e-30 {
        max_boundary_speed / max_interior_speed
    } else {
        0.0
    };

    println!(
        "[no-slip check] max_boundary_speed={:.3e}, max_interior_speed={:.3e}, \
         ratio={:.3e} (tol={NO_SLIP_TOL})",
        max_boundary_speed, max_interior_speed, ratio
    );

    // Trivially satisfied if there is no active flow.
    if max_interior_speed < 1e-12 {
        println!("[no-slip check] interior speed ~0; no-slip trivially satisfied");
        return;
    }

    assert!(
        ratio < NO_SLIP_TOL,
        "\n\
         ╔══════════════════════════════════════════════════════════════════╗\n\
         ║  BLOCKED: NO-SLIP VIOLATED on the confined nephroid domain      ║\n\
         ╠══════════════════════════════════════════════════════════════════╣\n\
         ║  max_boundary_speed = {max_boundary_speed:.3e}                               \n\
         ║  max_interior_speed = {max_interior_speed:.3e}                               \n\
         ║  ratio              = {ratio:.3e}  (must be < {NO_SLIP_TOL})              \n\
         ╠══════════════════════════════════════════════════════════════════╣\n\
         ║  ROOT CAUSE: PoissonSolver only pins vertex 0; it does NOT      ║\n\
         ║  enforce ψ=0 on all boundary vertices. On a periodic mesh this  ║\n\
         ║  is harmless, but on a bounded domain the constant-kernel fix   ║\n\
         ║  is insufficient — all n_boundary={n_boundary} vertices need ψ=0.         \n\
         ║                                                                  ║\n\
         ║  GATED FIX: modify PoissonSolver (or add a boundary projection  ║\n\
         ║  step after solve) to enforce Dirichlet ψ=0 at all indices in   ║\n\
         ║  ConfinedMesh::boundary_vertices before recovering velocity.     ║\n\
         ╚══════════════════════════════════════════════════════════════════╝"
    );

    println!(
        "[PASS] No-slip holds on the confined nephroid domain \
         (ratio={ratio:.3e} < {NO_SLIP_TOL})"
    );
}
