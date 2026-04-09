//! Parameter sweep for activity on S^2: find the threshold for motile defects.
//!
//! Runs short simulations at increasing zeta and reports whether S fluctuates
//! (indicating active turbulence) or monotonically converges (equilibrium).

use std::time::Instant;

use cartan_manifolds::sphere::Sphere;
use volterra_core::ActiveNematicParams;
use volterra_dec::connection_laplacian::{ConnectionLaplacian, molecular_field_conn};
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::stokes_dec::{StokesSolverDec, advect_q};
use volterra_dec::QFieldDec;
use volterra_dec::DecDomain;

fn main() {
    let refinement = 3; // 642 vertices (fast sweeps)
    let n_steps = 5000;
    let sample_window = 500; // check S fluctuation over last 500 steps

    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let domain = DecDomain::new(mesh, Sphere::<3>).unwrap();

    let coords: Vec<[f64; 3]> = domain.mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let star0: Vec<f64> = (0..domain.ops.hodge.star0().len()).map(|i| domain.ops.hodge.star0()[i]).collect();
    let star1: Vec<f64> = (0..domain.ops.hodge.star1().len()).map(|i| domain.ops.hodge.star1()[i]).collect();
    let conn_lap = ConnectionLaplacian::new(&domain.mesh, &coords, &star0, &star1);
    let stokes = StokesSolverDec::new(&domain.ops, &domain.mesh).unwrap();

    println!("S^2 activity sweep (refinement={refinement}, {nv} vertices, {n_steps} steps)");
    println!("{:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<8} {:<8}",
        "zeta", "eta", "K", "S_final", "S_min", "S_max", "fluct?", "time(s)");

    // Sweep: vary zeta and eta together to explore the phase space.
    let configs: Vec<(f64, f64, f64)> = vec![
        // (zeta, eta, K)
        (0.1, 0.1, 0.01),
        (0.5, 0.1, 0.01),
        (1.0, 0.1, 0.01),
        (2.0, 0.1, 0.01),
        (5.0, 0.1, 0.01),
        (0.5, 0.1, 0.1),
        (1.0, 0.1, 0.1),
        (2.0, 0.1, 0.1),
        (0.5, 0.05, 0.05),
        (1.0, 0.05, 0.05),
        (2.0, 0.05, 0.05),
    ];

    for &(zeta, eta, k_r) in &configs {
        let mut params = ActiveNematicParams::default_test();
        params.dt = 0.0001; // conservative for low eta
        params.zeta_eff = zeta;
        params.k_r = k_r;
        params.gamma_r = 1.0;
        params.eta = eta;
        params.a_landau = -0.4;
        params.c_landau = 2.0;
        params.lambda = 0.7;

        let mut q = QFieldDec::random_perturbation(nv, 0.2, 42);
        let mut s_history = Vec::new();

        let t0 = Instant::now();
        let mut blew_up = false;

        for step in 0..n_steps {
            // Stokes solve.
            let vel = stokes.solve(&q, &params, &domain.ops, &domain.mesh);

            // RK4 with connection Laplacian + advection.
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = molecular_field_conn(qq, params.k_r, params.a_eff(), params.c_landau, &conn_lap);
                let mut dq = h.scale(params.gamma_r);
                let adv = advect_q(qq, &vel, &domain.mesh.boundaries, &domain.mesh.vertex_boundaries, &coords);
                for i in 0..qq.n_vertices {
                    dq.q1[i] -= adv.q1[i];
                    dq.q2[i] -= adv.q2[i];
                }
                dq
            };

            let k1 = rhs(&q);
            let q2 = q.add(&k1.scale(0.5 * params.dt));
            let k2 = rhs(&q2);
            let q3 = q.add(&k2.scale(0.5 * params.dt));
            let k3 = rhs(&q3);
            let q4 = q.add(&k3.scale(params.dt));
            let k4 = rhs(&q4);
            let update = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
            q = q.add(&update.scale(params.dt / 6.0));

            let s = q.mean_order_param();
            if s.is_nan() || s > 100.0 {
                blew_up = true;
                break;
            }

            if step >= n_steps - sample_window {
                s_history.push(s);
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();

        if blew_up {
            println!("{:<8.3} {:<8.1} {:<8.3} {:<10} {:<10} {:<10} {:<8} {:<8.1}",
                zeta, params.eta, params.k_r, "NaN", "-", "-", "BLOW", elapsed);
            continue;
        }

        let s_final = s_history.last().copied().unwrap_or(0.0);
        let s_min = s_history.iter().cloned().fold(f64::INFINITY, f64::min);
        let s_max = s_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let fluctuation = (s_max - s_min) / s_final.max(0.01);
        let is_active = fluctuation > 0.05; // >5% variation = active

        println!("{:<8.3} {:<8.1} {:<8.3} {:<10.4} {:<10.4} {:<10.4} {:<8} {:<8.1}",
            zeta, params.eta, params.k_r, s_final, s_min, s_max,
            if is_active { "YES" } else { "no" }, elapsed);
    }
}
