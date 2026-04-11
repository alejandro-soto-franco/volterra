//! Active nematic on S^2 at Zhu et al. (2024) Peclet numbers.
//!
//! Runs the Beris-Edwards + Stokes DEC solver on a unit sphere with
//! parameters mapped from Zhu, Saintillan, Chern's nondimensionalisation.
//!
//! Pe controls the activity-to-relaxation ratio:
//!   Pe = 1:     4 tetrahedral +1/2 defects, gentle oscillation
//!   Pe = 10:    4 defects, stronger oscillation
//!   Pe = 100:   chaotic 4-defect dynamics
//!   Pe = 1000:  active turbulence onset
//!   Pe = 10000: fully developed active turbulence
//!
//! Usage:
//!     cargo run --release -p volterra-solver --example sim_sphere_zhu -- --pe 100
//!     cargo run --release -p volterra-solver --example sim_sphere_zhu -- --pe 1000 --refinement 5

use std::path::Path;
use std::time::Instant;

use cartan_manifolds::sphere::Sphere;
use volterra_core::ActiveNematicParams;
use volterra_dec::connection_laplacian::{ConnectionLaplacian, molecular_field_conn};
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::{write_snapshot, write_velocity_snapshot};
use volterra_dec::stokes_dec::{StokesSolverDec, advect_q_covariant};
use volterra_dec::QFieldDec;
use volterra_dec::DecDomain;

/// Map Zhu's nondimensional Pe to volterra's dimensional parameters.
///
/// Zhu's nondimensionalisation (arXiv:2405.06044, Eq. 34):
///   Pe = |alpha| * r^2 / mu    (activity / elastic relaxation)
///   eps = varepsilon / r        (defect core / domain)
///
/// We fix K_frank = 0.01, eta = 1, gamma_r = 1, R = 1 (unit sphere).
/// Then Pe = zeta * R^2 / K = zeta / K = 100 * zeta.
///
/// The Ginzburg-Landau bulk sets the equilibrium order:
///   |z|_eq = sqrt(-a_eff / (4c))
/// We target |z|_eq ~ 0.5 (S_eq ~ 1.0) by fixing c = 1 and a_eff = -1.
fn zhu_params(pe: f64) -> (ActiveNematicParams, f64, f64, usize) {
    let k_frank = 0.01;
    let eta = 1.0;
    let gamma_r = 1.0;
    let c_landau = 1.0;
    let a_eff_target = -1.0;
    let lambda = 1.0; // flow-aligning (Zhu uses lambda=1)

    // Pe = zeta / K => zeta = Pe * K
    let zeta = pe * k_frank;

    // a_eff = a_landau - zeta/2 => a_landau = a_eff + zeta/2
    let a_landau = a_eff_target + zeta / 2.0;

    // Defect core size: eps ~ sqrt(K / |a_eff|) = sqrt(0.01) = 0.1
    // Resolved by ~2 edges on L5 icosphere (edge ~ 0.06)

    // Timestep: conservative for explicit RK4.
    // Diffusive CFL: dt < dx^2 / (gamma_r * K) ~ 0.0036 / 0.01 = 0.36
    // GL stiffness: max rate ~ max(|a_eff|, 4c) = max(1, 4) = 4, dt < 0.25
    // Advective CFL: dt < dx / max|v|. max|v| ~ zeta/eta ~ Pe*K = 0.01*Pe.
    //   Pe=1: max|v|~0.01, dt < 6. Pe=1000: max|v|~10, dt < 0.006.
    let dt = if pe <= 1.0 {
        0.005
    } else if pe <= 10.0 {
        0.002
    } else if pe <= 100.0 {
        0.001
    } else if pe <= 1000.0 {
        0.0005
    } else {
        0.0002
    };

    // Simulation time: T = 30 (nondimensional, matching Zhu).
    // nondimensional time = t * |a_eff| * gamma_r (relaxation units)
    // Since gamma_r = 1 and |a_eff| = 1, physical t = T_nd.
    let t_final = 30.0;
    let _n_steps = ((t_final / dt) as f64).ceil() as usize;

    // Snapshot every ~0.1 nondimensional time units.
    let snap_every = ((0.1_f64 / dt).round() as usize).max(1);

    let mut params = ActiveNematicParams::default_test();
    params.dt = dt;
    params.zeta_eff = zeta;
    params.k_r = k_frank;
    params.gamma_r = gamma_r;
    params.eta = eta;
    params.a_landau = a_landau;
    params.c_landau = c_landau;
    params.lambda = lambda;

    (params, t_final, dt, snap_every)
}

fn main() {
    // Parse command-line arguments.
    let args: Vec<String> = std::env::args().collect();
    let mut pe = 100.0_f64;
    let mut refinement = 5_usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pe" => {
                i += 1;
                pe = args[i].parse().expect("invalid --pe value");
            }
            "--refinement" | "-r" => {
                i += 1;
                refinement = args[i].parse().expect("invalid --refinement value");
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let (params, t_final, dt, snap_every) = zhu_params(pe);
    let n_steps = ((t_final / dt) as f64).ceil() as usize;

    let out_dir = format!("output/sphere_pe{}", pe as u64);
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("=== Zhu et al. (2024) S^2 Active Nematic ===");
    println!("  Pe = {pe}");
    println!("  refinement = {refinement}");
    println!("  dt = {dt}");
    println!("  T_final = {t_final}");
    println!("  n_steps = {n_steps}");
    println!("  snap_every = {snap_every}");
    println!("  zeta = {:.4}", params.zeta_eff);
    println!("  K = {:.4}", params.k_r);
    println!("  a_eff = {:.4}", params.a_eff());
    println!("  c = {:.4}", params.c_landau);
    println!("  lambda = {:.4}", params.lambda);
    println!("  eps ~ {:.4}", (params.k_r / params.a_eff().abs()).sqrt());
    println!();

    println!("Building icosphere (L{refinement})...");
    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let nf = mesh.n_simplices();
    println!("  vertices: {nv}, faces: {nf}, chi: {}", mesh.euler_characteristic());

    println!("Assembling DEC operators...");
    let domain = DecDomain::new(mesh, Sphere::<3>).expect("DecDomain assembly failed");

    // Write mesh JSON.
    let mesh_json = serde_json::json!({
        "vertices": domain.mesh.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>(),
        "triangles": domain.mesh.simplices,
    });
    std::fs::write(out.join("mesh.json"), serde_json::to_string(&mesh_json).unwrap())
        .expect("failed to write mesh.json");

    let stokes_coords: Vec<[f64; 3]> = domain.mesh.vertices.iter()
        .map(|v| [v[0], v[1], v[2]]).collect();

    let conn_lap = ConnectionLaplacian::new(
        &domain.mesh, &stokes_coords,
        &(0..domain.ops.hodge.star0().len()).map(|i| domain.ops.hodge.star0()[i]).collect::<Vec<_>>(),
        &(0..domain.ops.hodge.star1().len()).map(|i| domain.ops.hodge.star1()[i]).collect::<Vec<_>>(),
    );

    println!("Factorising Stokes solver...");
    let stokes = StokesSolverDec::new(&domain.ops, &domain.mesh)
        .expect("Stokes solver factorisation failed");

    let edge_phases = conn_lap.edge_phases();

    // Initial condition: random perturbation around half-order.
    // |z| ~ 0.5 matches the equilibrium, with spatial noise to seed instability.
    let mut q = QFieldDec::random_perturbation(nv, 0.5, 42);

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "sphere",
        "mode": "wet_zhu",
        "pe": pe,
        "refinement": refinement,
        "n_vertices": nv,
        "n_faces": nf,
        "n_steps": n_steps,
        "snap_every": snap_every,
        "dt": dt,
        "t_final": t_final,
        "zeta_eff": params.zeta_eff,
        "k_r": params.k_r,
        "eta": params.eta,
        "a_eff": params.a_eff(),
        "c_landau": params.c_landau,
        "lambda": params.lambda,
        "eps": (params.k_r / params.a_eff().abs()).sqrt(),
    });
    volterra_dec::snapshot::write_meta(&out.join("meta.json"), &meta)
        .expect("failed to write meta.json");

    println!("Running: Pe={pe}, T={t_final}, {n_steps} steps...");
    let t0 = Instant::now();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            write_snapshot(&q, &out.join(format!("q_{step:06}.npy")))
                .expect("failed to write snapshot");
        }

        if step % (snap_every * 5) == 0 {
            let s = q.mean_order_param();
            let t_sim = step as f64 * dt;
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
            println!("  t={t_sim:6.2}/{t_final}  step {step:>7}/{n_steps}  <S>={s:.4}  wall={elapsed:.1}s  ({rate:.0} steps/s)");
        }

        if step < n_steps {
            // 1. Stokes solve.
            let vel = stokes.solve(&q, &params, &domain.ops, &domain.mesh);

            // Write velocity snapshot.
            if step % snap_every == 0 {
                write_velocity_snapshot(&vel.v, &out.join(format!("vel_{step:06}.npy")))
                    .expect("failed to write velocity snapshot");
            }

            // 2. RK4 step: molecular field + covariant advection.
            let coords = &stokes_coords;
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = molecular_field_conn(
                    qq, params.k_r, params.a_eff(), params.c_landau, &conn_lap,
                );
                let mut dq = h.scale(params.gamma_r);

                let adv = advect_q_covariant(
                    qq, &vel,
                    &domain.mesh.boundaries,
                    &domain.mesh.vertex_boundaries,
                    coords,
                    &edge_phases,
                );
                let nv = qq.n_vertices;
                for i in 0..nv {
                    dq.q1[i] -= adv.q1[i];
                    dq.q2[i] -= adv.q2[i];
                }
                dq
            };

            let k1 = rhs(&q);
            let q2 = q.add(&k1.scale(0.5 * dt));
            let k2 = rhs(&q2);
            let q3 = q.add(&k2.scale(0.5 * dt));
            let k3 = rhs(&q3);
            let q4 = q.add(&k3.scale(dt));
            let k4 = rhs(&q4);
            let update = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
            q = q.add(&update.scale(dt / 6.0));
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let n_snaps = (n_steps / snap_every) + 1;
    println!();
    println!("Done: {n_snaps} snapshots in {elapsed:.1}s");
    println!("Output: {out_dir}");
    println!();
    println!("Render with:");
    println!("  python tools/viz/render_surface_pv.py {out_dir} --video {out_dir}/pe{}.mp4 --orbit", pe as u64);
}
