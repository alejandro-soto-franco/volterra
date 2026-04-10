//! Active nematic on S^2 using Zhu-Saintillan-Chern (2024) framework.
//!
//! Parameters from arXiv:2405.06044, Section 4.3, Figure 6A:
//!   Pe = 1 (low activity): 4 +1/2 defects in periodic orbiting motions
//!   lambda = 1 (flow-aligning)
//!   epsilon -> 0 (normalisation handles it)
//!
//! Mesh: well-centred icosphere level 3 (642 vertices).
//! Stokes backend: stream function (direct LDL^T, fast for 2-manifolds).
//! Runs for t = 30 nondimensional time units.

use std::time::Instant;

use cartan_dec::hodge::HodgeStar;
use cartan_dec::line_bundle::{ConnectionAngles, defect_charges, Section};
use cartan_dec::mesh_gen::icosphere;
use cartan_manifolds::sphere::Sphere;

use volterra_solver::nematic_field_2d::NematicField2D;
use volterra_solver::stokes_trait::KillingOperatorSolver;
use volterra_solver::{ActiveNematicEngine, EngineParams, StokesSolver};

fn main() {
    // ── Mesh ────────────────────────────────────────────────────────────
    let level = 2; // 162 vertices
    let manifold = Sphere::<3>;

    println!("Building well-centred icosphere (level {level})...");
    let mesh = icosphere(&manifold, level, true);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  {nv} vertices, {nf} faces");

    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold)
        .expect("circumcentric Hodge star requires well-centred mesh");

    // ── Parameters (Zhu et al. 2024, Figure 6A) ────────────────────────
    let dt: f64 = 0.01;
    let t_final: f64 = 30.0;
    let n_steps = (t_final / dt).round() as usize;

    let params = EngineParams {
        pe: 1.0,              // Low activity (Fig 6A)
        lambda: 1.0,          // Flow-aligning
        epsilon: 0.0,         // Not used (normalise step)
        dt,
        activity_sign: -1.0,  // Extensile
    };

    println!("  Pe = {}, lambda = {}, dt = {}, t_final = {}", params.pe, params.lambda, dt, t_final);
    println!("  {n_steps} steps, extensile activity");

    // ── Stokes solver (stream function, fast for 2-manifolds) ──────────
    // Unit sphere: K = 1 everywhere.
    println!("Building Killing operator Stokes solver...");
    // Reduced iterations for speed: 10 AL iters, 200 CG iters.
    let stokes: Box<dyn StokesSolver> = Box::new(
        KillingOperatorSolver::new_with_iters(&mesh, 1e2, 1e-3, 5, 100)
    );

    // ── Engine ─────────────────────────────────────────────────────────
    println!("Building ActiveNematicEngine...");
    let mut engine = ActiveNematicEngine::new(&mesh, &manifold, &hodge, params, stokes);

    // ── Initial condition ──────────────────────────────────────────────
    // Random perturbation around uniform nematic (as in the paper).
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut field = NematicField2D::from_section(
        Section::<2>::from_real_components(
            &(0..nv).map(|_| 0.3 + 0.1 * rng.random_range(-1.0..1.0)).collect::<Vec<f64>>(),
            &(0..nv).map(|_| 0.1 * rng.random_range(-1.0..1.0)).collect::<Vec<f64>>(),
        ),
    );
    field.normalise();

    // Precompute connection for defect detection.
    let connection = ConnectionAngles::from_mesh(&mesh, &manifold);

    // Integrated Gaussian curvature per face: K_f = K * area_f.
    // On the unit sphere K = 1 everywhere, so K_f = area_f.
    let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let gauss_k_faces: Vec<f64> = mesh.simplices.iter().map(|&[i0, i1, i2]| {
        let e01 = [coords[i1][0] - coords[i0][0], coords[i1][1] - coords[i0][1], coords[i1][2] - coords[i0][2]];
        let e02 = [coords[i2][0] - coords[i0][0], coords[i2][1] - coords[i0][1], coords[i2][2] - coords[i0][2]];
        let cx = e01[1] * e02[2] - e01[2] * e02[1];
        let cy = e01[2] * e02[0] - e01[0] * e02[2];
        let cz = e01[0] * e02[1] - e01[1] * e02[0];
        0.5 * (cx * cx + cy * cy + cz * cz).sqrt() // area_f = integrated K on unit sphere
    }).collect();

    // ── Run ────────────────────────────────────────────────────────────
    let print_every = 100;
    let t0 = Instant::now();

    println!("\n{:>6}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>8}",
        "step", "time", "<S>", "div_res", "+1/2", "-1/2", "wall_s");
    println!("{}", "-".repeat(68));

    for i in 1..=n_steps {
        let diag = engine.step(&mut field);

        if i % print_every == 0 || i == n_steps || i == 1 {
            // Detect defects.
            let charges = defect_charges::<2>(&field.section, &connection, &mesh, &gauss_k_faces);
            let mut n_pos = 0usize;
            let mut n_neg = 0usize;
            for &c in &charges {
                if c > 0.1 { n_pos += 1; }
                if c < -0.1 { n_neg += 1; }
            }

            let wall = t0.elapsed().as_secs_f64();
            println!(
                "{:>6}  {:>8.3}  {:>8.4}  {:>8.2e}  {:>6}  {:>6}  {:>8.1}",
                i, diag.time, diag.mean_order, diag.stokes_residual, n_pos, n_neg, wall
            );

            // Poincare-Hopf check every 500 steps.
            if i % 500 == 0 {
                let total_charge: f64 = charges.iter().sum();
                println!("        sum(Z) = {total_charge:.6} (expect 2.0 for S^2)");
            }
        }
    }

    let wall_total = t0.elapsed().as_secs_f64();
    println!("\nDone: {n_steps} steps, t = {t_final}, wall time = {wall_total:.1}s");
    println!("  mean step time = {:.3}ms", 1000.0 * wall_total / n_steps as f64);

    // Final defect census.
    let charges = defect_charges::<2>(&field.section, &connection, &mesh, &gauss_k_faces);
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    for &c in &charges {
        if c > 0.25 { n_pos += 1; }
        if c < -0.25 { n_neg += 1; }
    }
    let total: f64 = charges.iter().sum();
    println!("  final defects: {} positive, {} negative, sum(Z) = {:.6}", n_pos, n_neg, total);
    println!("  final <S> = {:.4}", field.mean_scalar_order());
}
