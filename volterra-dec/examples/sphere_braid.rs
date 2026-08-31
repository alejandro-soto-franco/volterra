//! Active nematic on a sphere, with the defects tracked so the braid can be read.
//!
//! Runs Beris-Edwards with a Stokes flow on an icosphere and writes, at every
//! snapshot, the position and charge of every defect. Four `+1/2` defects is
//! what a sphere owes by Poincare-Hopf when the field is half-charged
//! throughout, and at low activity they sit at the corners of a tetrahedron,
//! which is the configuration a braid is read from.
//!
//! The activity is set by a Peclet number, the ratio of the active stress to
//! elastic relaxation:
//!
//! ```text
//! Pe = zeta R^2 / K
//! ```
//!
//! Everything else is held: `K = 0.01`, `eta = 1`, `gamma_r = 1`, `R = 1`,
//! `lambda = 1`, and a Ginzburg-Landau bulk with `c = 1` and `a_eff = -1`, so
//! the equilibrium order is `|z| = 0.5` and the core size is
//! `sqrt(K/|a_eff|) = 0.1`, about two edges at refinement 5.
//!
//!     cargo run --release -p volterra-solver --example sphere_braid -- --pe 1
//!     cargo run --release -p volterra-solver --example sphere_braid -- --pe 100 -r 5
use std::path::Path;
use std::time::Instant;

use cartan_manifolds::sphere::Sphere;
use volterra_core::ActiveNematicParams;
use volterra_dec::connection_laplacian::{ConnectionLaplacian, molecular_field_conn};
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::snapshot::{write_snapshot, write_velocity_snapshot};
use volterra_dec::stokes::{SurfaceStokes, advect_q_covariant};
use volterra_dec::QField;
use volterra_dec::DecDomain;
use volterra_dec::surface_defects::{detect_defects_surface, total_charge};
use std::io::Write;

/// Map Zhu's nondimensional Pe to volterra's dimensional parameters.
///
/// Nondimensionalisation:
///   Pe  = |alpha| r^2 / mu   (activity against elastic relaxation)
///   eps = core size / radius
///
/// We fix K_frank = 0.01, eta = 1, gamma_r = 1, R = 1 (unit sphere).
/// Then Pe = zeta * R^2 / K = zeta / K = 100 * zeta.
///
/// The Ginzburg-Landau bulk sets the equilibrium order:
///   |z|_eq = sqrt(-a_eff / (4c))
/// We target |z|_eq ~ 0.5 (S_eq ~ 1.0) by fixing c = 1 and a_eff = -1.
/// Read a `(n_vertices, 2)` float64 `.npy` snapshot back into a Q field.
///
/// Accepts only the header this crate's own writer produces, so a file from
/// anywhere else fails loudly instead of being reinterpreted.
fn read_snapshot(path: &std::path::Path, nv: usize) -> std::io::Result<QField> {
    use std::io::{Error, ErrorKind, Read};
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 10];
    f.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(Error::new(ErrorKind::InvalidData, "not a .npy file"));
    }
    let hlen = u16::from_le_bytes([magic[8], magic[9]]) as usize;
    let mut header = vec![0u8; hlen];
    f.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header).to_string();
    let want = format!("'descr': '<f8', 'fortran_order': False, 'shape': ({nv}, 2)");
    if !header.contains(&want) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("header is {}, wanted {want}", header.trim()),
        ));
    }
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() != nv * 2 * 8 {
        return Err(Error::new(ErrorKind::InvalidData, "wrong payload length"));
    }
    let val = |i: usize| f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    let mut q = QField::zeros(nv);
    for v in 0..nv {
        q.q1[v] = val(2 * v);
        q.q2[v] = val(2 * v + 1);
    }
    Ok(q)
}

fn activity_params(pe: f64) -> (ActiveNematicParams, f64, f64, usize) {
    let k_frank = 0.01;
    let eta = 1.0;
    let gamma_r = 1.0;
    let c_landau = 1.0;
    let a_eff_target = -1.0;
    let lambda = 1.0; // flow-aligning

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

    // Simulation time, in relaxation units.
    // nondimensional time = t * |a_eff| * gamma_r (relaxation units)
    // Since gamma_r = 1 and |a_eff| = 1, physical t = T_nd.
    let t_final: f64 = 30.0;
    let _n_steps = (t_final / dt).ceil() as usize;

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
    let mut t_override: Option<f64> = None;
    let mut tag = String::new();
    let mut seed = 42_u64;
    let mut q_init: Option<String> = None;
    let mut dt_override: Option<f64> = None;
    let mut snap_dt: Option<f64> = None;
    let mut tol = 1e-8_f64;

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
            "--t-final" => {
                i += 1;
                t_override = Some(args[i].parse().expect("invalid --t-final value"));
            }
            "--tag" => {
                i += 1;
                tag = args[i].clone();
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid --seed value");
            }
            "--q-init" => {
                i += 1;
                q_init = Some(args[i].clone());
            }
            "--dt" => {
                i += 1;
                dt_override = Some(args[i].parse().expect("invalid --dt value"));
            }
            "--snap-dt" => {
                i += 1;
                snap_dt = Some(args[i].parse().expect("invalid --snap-dt value"));
            }
            "--tol" => {
                i += 1;
                tol = args[i].parse().expect("invalid --tol value");
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let (params, t_default, dt_default, snap_every_default) = activity_params(pe);
    let dt = dt_override.unwrap_or(dt_default);
    let snap_every = match (snap_dt, dt_override) {
        (Some(sd), _) => ((sd / dt).round() as usize).max(1),
        (None, Some(_)) => ((0.1_f64 / dt).round() as usize).max(1),
        (None, None) => snap_every_default,
    };
    let mut params = params;
    params.dt = dt;
    let t_final = t_override.unwrap_or(t_default);
    let n_steps = (t_final / dt).ceil() as usize;

    let out_dir = if tag.is_empty() {
        format!("output/sphere_braid_pe{pe}")
    } else {
        format!("output/sphere_braid_{tag}")
    };
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).expect("failed to create output directory");

    println!("=== active nematic on a sphere ===");
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
    let stokes = SurfaceStokes::new(&domain.ops, &domain.mesh)
        .expect("Stokes solver factorisation failed");

    let edge_phases = conn_lap.edge_phases();

    // Initial condition: random perturbation around half-order.
    // |z| ~ 0.5 matches the equilibrium, with spatial noise to seed instability.
    // A random start finds the tetrahedron only sometimes: of five seeds
    // relaxed with no activity, two reached it, two stopped part way and one
    // settled into a coplanar square. Continuing from a state already relaxed
    // into the tetrahedron enters the configuration rather than waiting for it.
    let mut q = match &q_init {
        Some(path) => {
            let f = read_snapshot(std::path::Path::new(path), nv)
                .expect("failed to read the initial Q snapshot");
            println!("  initial Q from {path}");
            f
        }
        None => QField::random_perturbation(nv, 0.5, seed),
    };

    // Write metadata.
    let meta = serde_json::json!({
        "geometry": "sphere",
        "mode": "sphere_braid",
        "pe": pe,
        "refinement": refinement,
        "seed": seed,
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

    let mut fdef = std::fs::File::create(out.join("defects.csv"))
        .expect("failed to open defects.csv");
    writeln!(fdef, "step,t,x,y,z,charge").unwrap();
    let mut fstat = std::fs::File::create(out.join("stats.csv"))
        .expect("failed to open stats.csv");
    writeln!(fstat, "step,t,n_plus,n_minus,mean_S,total_charge,u_rms,pe_measured").unwrap();

    println!("Running: Pe={pe}, T={t_final}, {n_steps} steps...");
    let t0 = Instant::now();

    // The Peclet number the run actually realises, as the reference defines it:
    // the advective rate against the elastic relaxation rate at the domain
    // scale, `u R / (gamma_r K)` with `R = 1`. The flag sets an active stress,
    // which fixes this only once the order parameter has settled, so it is
    // measured rather than declared.
    let mut u_rms = 0.0_f64;
    let mut psi_prev: Option<Vec<f64>> = None;
    let mut cg_iters = 0usize;

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            write_snapshot(&q, &out.join(format!("q_{step:06}.npy")))
                .expect("failed to write snapshot");

            let t_sim = step as f64 * dt;
            let defs = detect_defects_surface(
                &stokes_coords, &domain.mesh.simplices, &domain.mesh.boundaries,
                &domain.mesh.simplex_boundary_ids, &edge_phases, &q,
            );
            for (p, c) in &defs {
                writeln!(fdef, "{step},{t_sim:.6},{:.6},{:.6},{:.6},{c}", p[0], p[1], p[2])
                    .unwrap();
            }
            let npl = defs.iter().filter(|d| d.1 > 0).count();
            // The total is Poincare-Hopf and must be four halves on a sphere at
            // every step. It is recorded rather than asserted so a run that
            // breaks it is diagnosable afterwards rather than merely dead.
            let pe_measured = u_rms / (params.gamma_r * params.k_r);
            writeln!(
                fstat, "{step},{t_sim:.6},{npl},{},{:.6},{},{u_rms:.6},{pe_measured:.4}",
                defs.len() - npl, q.mean_order_param(), total_charge(&defs)
            ).unwrap();
        }

        if step % (snap_every * 5) == 0 {
            let s = q.mean_order_param();
            let t_sim = step as f64 * dt;
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { step as f64 / elapsed } else { 0.0 };
            let per_step = if step > 0 { cg_iters as f64 / step as f64 } else { 0.0 };
            println!("  t={t_sim:6.2}/{t_final}  step {step:>7}/{n_steps}  <S>={s:.4}  wall={elapsed:.1}s  ({rate:.0} steps/s, {per_step:.0} cg/step)");
        }

        if step < n_steps {
            // 1. Stokes solve, warm-started from the previous step.
            //
            // The source moves by order `dt` per step, so last step's stream
            // function is already close to this step's and the iteration is
            // short. Started cold the solve repeats its whole descent every
            // step, which is where the run was spending most of its time.
            let (vel, psi_next, its) = stokes.solve_warm(
                &q, &params, &domain.ops, &domain.mesh, psi_prev.as_deref(), tol,
            );
            psi_prev = Some(psi_next);
            cg_iters += its;
            u_rms = (vel.v.iter().map(|u| u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
                .sum::<f64>() / vel.v.len() as f64).sqrt();

            // Write velocity snapshot.
            if step % snap_every == 0 {
                write_velocity_snapshot(&vel.v, &out.join(format!("vel_{step:06}.npy")))
                    .expect("failed to write velocity snapshot");
            }

            // 2. RK4 step: molecular field + covariant advection.
            let coords = &stokes_coords;
            let rhs = |qq: &QField| -> QField {
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
