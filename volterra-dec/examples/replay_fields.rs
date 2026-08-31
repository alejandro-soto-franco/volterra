//! Velocity, stream function and pressure for a finished `active_confined` run.
//!
//! The solver saves `Q` and nothing else, because `Q` is the state: the flow is
//! an instantaneous functional of it, so a saved frame determines the velocity
//! exactly and there is no reason to store it. A film that wants the flow can
//! therefore recover it after the fact, and this replays each saved frame
//! through the SAME Stokes path the run itself used.
//!
//! The pressure is not a replay. A stream-function solve eliminates it, so no
//! run has ever formed it; it is solved here from the same assembled stress,
//! by the Poisson problem on `pressure_rhs_from_force`.
//!
//! The mesh is rebuilt from `consts.json` and checked against the run's own
//! `vertices.tsv` before anything is written. A mesh that failed to reproduce
//! would give fields on the wrong domain, silently.
//!
//!   REPLAY_RUN=runs/prod_neph_d0.72_s0 \
//!     cargo run --release --example replay_fields
//!
//! `REPLAY_STRIDE` takes every nth frame, `REPLAY_FROM` and `REPLAY_TO` bound
//! the range, and `REPLAY_SELFTEST=1` checks the pressure solve against a force
//! whose potential is known and exits.

use std::io::Write;
use std::path::{Path, PathBuf};

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};
use volterra_dec::confined_ldg::LdgProblem;
use volterra_dec::nematic_params::NematicParams;
use volterra_dec::poisson::PoissonSolver;
use volterra_dec::qfield::QField;
use volterra_dec::stokes::{SurfaceStokes, pressure_rhs_from_force, vorticity_from_psi};

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn frame_index(path: &Path) -> Option<usize> {
    path.file_stem()?.to_str()?.split('_').nth(1)?.parse().ok()
}

fn write_f32_pairs(path: &Path, pairs: impl Iterator<Item = (f64, f64)>) {
    let mut buf: Vec<u8> = Vec::new();
    for (a, b) in pairs {
        buf.extend_from_slice(&(a as f32).to_le_bytes());
        buf.extend_from_slice(&(b as f32).to_le_bytes());
    }
    std::fs::write(path, buf).expect("field frame");
}

fn write_f32(path: &Path, vals: &[f64]) {
    let mut buf: Vec<u8> = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        buf.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    std::fs::write(path, buf).expect("field frame");
}

fn main() {
    let run = PathBuf::from(std::env::var("REPLAY_RUN").expect("set REPLAY_RUN to the run dir"));
    let c: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("consts.json")).expect("consts.json"))
            .expect("consts.json parse");
    let g = |k: &str| c[k].as_f64().unwrap_or_else(|| panic!("consts.json has no {k}"));

    let shape = c["shape"].as_str().expect("shape").to_string();
    let qc = match shape.as_str() {
        "cardioid" => 1.5,
        "nephroid" => 2.0,
        "trefoiloid" => 2.5,
        "quatrefoiloid" => 3.0,
        "quintefoiloid" => 3.5,
        other => panic!("unknown shape {other}"),
    };
    let (d, r, als, ncl) = (g("d"), g("r"), g("als"), g("ncl"));
    let (h_bulk, h_min, q_anchor) = (g("h_bulk"), g("h_min"), g("q_anchor"));
    let seed = g("seed") as u64;
    let dt_frame = g("dt") * g("save_every");

    // `h_min` and the cusp treatment are read back rather than re-derived. The
    // run's own value is the one its mesh was built from, and a derivation that
    // drifted would put the fields on a different mesh from the defects.
    let cusp_edge = env_f64("ACT_CUSPEDGE", if d >= 1.0 { h_bulk } else { 0.0 });
    let curve = Epitrochoid { q: qc, d, r };
    let mesh_opts = MeshOpts { h_bulk, h_min, cusp_edge, seed, ..Default::default() };
    let mesh = confined_mesh(curve, mesh_opts);
    let nv = mesh.mesh.n_vertices();
    let boundary: Vec<usize> = mesh.boundary_vertices.clone();

    // The rebuilt mesh has to BE the run's mesh, not merely one like it.
    {
        let text = std::fs::read_to_string(run.join("vertices.tsv")).expect("vertices.tsv");
        let mut worst = 0.0_f64;
        let mut n = 0usize;
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let mut it = line.split('\t');
            let x: f64 = it.next().unwrap().parse().unwrap();
            let y: f64 = it.next().unwrap().parse().unwrap();
            assert!(n < nv, "vertices.tsv has more rows than the rebuilt mesh has vertices");
            let v = mesh.mesh.vertices[n];
            worst = worst.max(((v.x - x).powi(2) + (v.y - y).powi(2)).sqrt());
            n += 1;
        }
        assert_eq!(n, nv, "rebuilt mesh has {nv} vertices, the run wrote {n}");
        assert!(worst < 1e-5, "rebuilt mesh moved by {worst:.3e}, it is not the run's mesh");
        println!("  mesh reproduced: {nv} vertices, worst vertex offset {worst:.2e}");
    }

    let params = NematicParams::klein(als, ncl, r.round() as usize);
    let p = LdgProblem::new(mesh, params, q_anchor).expect("operators");
    let hloc = p.local_h();

    // The same two thresholds the run used, and for the same reasons: `wall_h`
    // bounds where the discrete curl recovery invents a velocity, `elastic_h`
    // bounds where the explicit elastic stress is stable. Neither is recorded in
    // `consts.json`, so they come from the environment with the solver's own
    // defaults, and the resume script sets neither.
    let wall_h = env_f64("ACT_WALLH", 0.05);
    let mut noslip = boundary.clone();
    {
        let on_wall: std::collections::HashSet<usize> = boundary.iter().copied().collect();
        for i in 0..nv {
            if hloc[i] > 0.0 && hloc[i] < wall_h && !on_wall.contains(&i) {
                noslip.push(i);
            }
        }
    }
    let elastic_h = env_f64("ACT_ELASTICH", 0.5);
    let elastic_mask: Vec<usize> = if env_usize("ACT_ELASTICMASK", 1) != 0 {
        let mut m = boundary.clone();
        for i in 0..nv {
            if hloc[i] > 0.0 && hloc[i] < elastic_h && !m.contains(&i) {
                m.push(i);
            }
        }
        m
    } else {
        Vec::new()
    };

    let slip_wall = std::env::var("ACT_WALL").map(|v| v == "slip").unwrap_or(false);
    let stokes = if slip_wall {
        SurfaceStokes::new_confined(&p.ops, &p.mesh.mesh, &noslip)
    } else {
        SurfaceStokes::new_confined_clamped(&p.ops, &p.mesh.mesh, &noslip)
    }
    .expect("confined Stokes factorisation");

    // CLOSED mode, no Dirichlet vertices. The pressure's wall condition is the
    // Neumann one the weak form imposes for itself.
    let poisson = PoissonSolver::new(&p.ops).expect("pressure Poisson");
    let area = poisson.mass_diagonal().to_vec();
    let area_total: f64 = area.iter().sum();

    // A force whose potential is known, so the pressure solve is checked against
    // an answer rather than against its own plausibility.
    if env_usize("REPLAY_SELFTEST", 0) != 0 {
        let k = 2.0 * std::f64::consts::PI / r;
        let phi: Vec<f64> = (0..nv)
            .map(|i| {
                let v = p.mesh.mesh.vertices[i];
                (k * v.x).cos() * (k * v.y).cos()
            })
            .collect();
        let f: Vec<[f64; 3]> = (0..nv)
            .map(|i| {
                let v = p.mesh.mesh.vertices[i];
                [
                    -k * (k * v.x).sin() * (k * v.y).cos(),
                    -k * (k * v.x).cos() * (k * v.y).sin(),
                    0.0,
                ]
            })
            .collect();
        let rhs = pressure_rhs_from_force(&f, &p.mesh.mesh, &volterra_dec::stokes::extract_coords(&p.mesh.mesh), &area);
        let sol = poisson.solve(&rhs);
        let mean_p: f64 = (0..nv).map(|i| area[i] * sol[i]).sum::<f64>() / area_total;
        let mean_e: f64 = (0..nv).map(|i| area[i] * phi[i]).sum::<f64>() / area_total;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..nv {
            num += area[i] * (sol[i] - mean_p - (phi[i] - mean_e)).powi(2);
            den += area[i] * (phi[i] - mean_e).powi(2);
        }
        println!(
            "  selftest: grad-potential force, relative L2 error {:.3e}",
            (num / den).sqrt()
        );
        return;
    }

    let qdir = run.join("qframes");
    let mut frames: Vec<usize> = std::fs::read_dir(&qdir)
        .expect("qframes")
        .filter_map(|e| frame_index(&e.ok()?.path()))
        .collect();
    frames.sort_unstable();
    let stride = env_usize("REPLAY_STRIDE", 1).max(1);
    let from = env_usize("REPLAY_FROM", 0);
    let to = env_usize("REPLAY_TO", usize::MAX);
    let frames: Vec<usize> = frames
        .into_iter()
        .filter(|f| *f >= from && *f <= to && f % stride == 0)
        .collect();
    assert!(!frames.is_empty(), "no q frames selected");

    for sub in ["ufields", "psifields", "pfields", "wfields"] {
        std::fs::create_dir_all(run.join(sub)).expect("field dir");
    }
    // The triangulation, once. Without it a renderer has to re-triangulate the
    // vertices, and a Delaunay of a non-convex domain fills the concavities.
    {
        let mut t = String::from("# i0 i1 i2\n");
        for s in &p.mesh.mesh.simplices {
            t.push_str(&format!("{}\t{}\t{}\n", s[0], s[1], s[2]));
        }
        std::fs::write(run.join("triangles.tsv"), t).expect("triangles.tsv");
    }

    let mut series = std::io::BufWriter::new(
        std::fs::File::create(run.join("fields.tsv")).expect("fields.tsv"),
    );
    // Appended, never inserted: the panel scripts read these by column index.
    writeln!(series, "# frame t u_rms u_max p_rms p_min p_max psi_max w_absmax").unwrap();

    // The run's own `speed_max`, so the replay can be checked against the solver
    // that produced the frames rather than trusted.
    let mut recorded: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    if let Ok(text) = std::fs::read_to_string(run.join("series.tsv")) {
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 8 {
                if let (Ok(fr), Ok(v)) = (f[0].parse::<usize>(), f[7].parse::<f64>()) {
                    recorded.insert(fr, v);
                }
            }
        }
    }
    let mut worst_rel = 0.0_f64;
    let mut worst_frame = 0usize;
    let mut checked = 0usize;

    let tol = env_f64("ACT_STOKESTOL", 1e-8);
    let mut psi_warm: Option<Vec<f64>> = None;
    let t0 = std::time::Instant::now();
    for (n, &fid) in frames.iter().enumerate() {
        let bytes = std::fs::read(qdir.join(format!("q_{fid:05}.f32"))).expect("q frame");
        assert_eq!(bytes.len(), nv * 8, "q frame {fid} is not {nv} vertex pairs");
        let mut q = QField::zeros(nv);
        for i in 0..nv {
            let a = f32::from_le_bytes(bytes[8 * i..8 * i + 4].try_into().unwrap());
            let b = f32::from_le_bytes(bytes[8 * i + 4..8 * i + 8].try_into().unwrap());
            q.q1[i] = a as f64;
            q.q2[i] = b as f64;
        }
        // The frame is f32, so the anchored values arrive truncated. Re-imposing
        // restores them exactly, which is what the solver stepped with.
        p.impose_anchoring(&mut q);

        let (s1, s2, sa) = p.beris_edwards_stress_masked(&q, &elastic_mask);
        let (vel, psi, _its) = stokes.solve_stress_warm(
            &s1, &s2, &sa, p.params.eta, &p.mesh.mesh, psi_warm.as_deref(), tol,
        );
        let pres = stokes.pressure_from_stress(&s1, &s2, &sa, &p.mesh.mesh, &poisson);
        let vort = vorticity_from_psi(&psi, &p.ops);
        psi_warm = Some(psi.clone());

        write_f32_pairs(
            &run.join("ufields").join(format!("u_{fid:05}.f32")),
            (0..nv).map(|i| (vel.v[i][0], vel.v[i][1])),
        );
        write_f32(&run.join("psifields").join(format!("psi_{fid:05}.f32")), &psi);
        write_f32(&run.join("pfields").join(format!("p_{fid:05}.f32")), &pres);
        write_f32(&run.join("wfields").join(format!("w_{fid:05}.f32")), &vort);

        let u2: f64 = (0..nv)
            .map(|i| area[i] * (vel.v[i][0].powi(2) + vel.v[i][1].powi(2)))
            .sum();
        let u_rms = (u2 / area_total).sqrt();
        let u_max = (0..nv).map(|i| vel.speed(i)).fold(0.0_f64, f64::max);
        if let Some(&rec) = recorded.get(&fid) {
            if rec.abs() > 1e-12 {
                let rel = (u_max - rec).abs() / rec.abs();
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_frame = fid;
                }
                checked += 1;
            }
        }
        let p_rms = ((0..nv).map(|i| area[i] * pres[i] * pres[i]).sum::<f64>() / area_total).sqrt();
        let p_min = pres.iter().copied().fold(f64::INFINITY, f64::min);
        let p_max = pres.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let psi_max = psi.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let w_absmax = vort.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        writeln!(
            series,
            "{fid}\t{:.6}\t{u_rms:.6}\t{u_max:.6}\t{p_rms:.6}\t{p_min:.6}\t{p_max:.6}\t{psi_max:.6}\t{w_absmax:.6}",
            fid as f64 * dt_frame
        )
        .unwrap();

        if n % 50 == 0 || n + 1 == frames.len() {
            series.flush().unwrap();
            println!(
                "  frame {fid:5} t {:6.2}  u_rms {u_rms:8.3}  u_max {u_max:9.3}  \
                 p in [{p_min:9.2}, {p_max:8.2}]  [{:.0}s]",
                fid as f64 * dt_frame,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!("{} frames -> {}", frames.len(), run.display());
    // A replay that disagrees with the run is a replay of a DIFFERENT solver.
    // The frames carry `Q` and nothing about the code that produced them, so a
    // run made before a change to the Stokes path, the Poisson solver or the
    // stress replays into fields the run never had. Nothing else detects that.
    if checked > 0 {
        println!(
            "  checked against the run's own speed_max on {checked} frames: \
             worst relative difference {worst_rel:.3e} at frame {worst_frame}"
        );
        if worst_rel > 1e-4 {
            println!(
                "  WARNING: the replay does not reproduce this run. Either the \
                 environment differs from the one it was launched with, or the \
                 solver has changed since and these frames are STALE."
            );
        }
    }
}
