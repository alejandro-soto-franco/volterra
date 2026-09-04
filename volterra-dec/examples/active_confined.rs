//! Active nematic in a cusped cavity, on a conforming mesh.
//!
//! The lattice cannot carry the four- and five-cusp geometries. Two to four
//! adjacent boundary cells turn the director by more than a quarter turn, the
//! winding sum picks the wrong branch, and the imposed winding comes out at two
//! or three times the value requested. Measured on the reference's own
//! boundaries:
//!
//! ```text
//! quatrefoiloid d = 0.90  L = 250  imposes 3x, worst step 112.7 deg, 4 steps over
//! cinquefoiloid d = 0.95  L = 250  imposes 3x, worst step 134.0 deg, 4 steps over
//! cinquefoiloid d = 0.99  L = 250  imposes 3x, and the mask pinches into 2 holes
//! ```
//!
//! A conforming mesh samples the boundary against the local curvature radius
//! rather than against a fixed cell, so the director step between neighbours
//! stays small where the wall turns fastest, and it follows the tip down to a
//! radius of a few times `1e-2` where the lattice loses it above `1`.
//!
//! Below that the mesh reads the cusped winding too, and correctly: a tip of
//! radius `1e-3` is not a feature any affordable discretisation represents, so
//! at `d = 0.99` the wall this integrates IS cusped and imposes `(k+2)/2`. The
//! run's own `imposed_charge` is the number to read, never `d`. Derivation and
//! the threshold in `d` are in `tests/index_law.rs` and in
//! `cgpo-reproduction/symbolic-review/forms/sympy/index_law.py`.
//!
//! This drives the full wet system on that mesh: Stokes for the velocity from the
//! active stress, then the Beris-Edwards equation the reference integrates,
//!
//! ```text
//! dQ/dt + u . grad Q = H / gamma + S
//! ```
//!
//! with `H` and `S` term for term against `flow-solver.py`'s `H_S_from_Q`. It
//! writes the defect worldlines in the same `defects.tsv` the lattice runs write,
//! so the braid reader and the classifier take it unchanged.
//!
//! ```text
//! ACT_SHAPE=quatrefoiloid ACT_D=0.99 ACT_ALS=3.5 ACT_NCL=4.0 ACT_H=1.5 \
//! ACT_STEPS=3000000 ACT_SAVE=1000 ACT_SEED=0 ACT_R=245 \
//! ACT_OUT=runs/mesh_quatre_d99_s0 \
//!   cargo run --release -p volterra-dec --example active_confined
//! ```
//!
//! `ACT_DRYRUN=1` stops after the mesh and the anchoring, which is how a
//! geometry is checked for the imposed winding and the element count before any
//! time is spent on it.

use std::io::Write;

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};
use volterra_dec::confined_ldg::LdgProblem;
use volterra_dec::semi_lagrangian::SemiLagrangian;
use volterra_dec::nematic_params::NematicParams;
use volterra_dec::stokes::{SurfaceStokes, VelocityField};

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let shape = env_str("ACT_SHAPE", "quatrefoiloid");
    let qc = match shape.as_str() {
        "cardioid" => 1.5,
        "nephroid" => 2.0,
        "trefoiloid" => 2.5,
        "quatrefoiloid" => 3.0,
        "quintefoiloid" => 3.5,
        "cinquefoiloid" => 3.5,
        other => panic!("unknown shape {other}"),
    };
    let d = env_f64("ACT_D", 0.99);
    // The lobe tip sits at r / 2, so r = 245 puts it at 122.5, the radius of an
    // L = 250 lattice. Keeping that equal is what lets the braid reader's step
    // bound and the classifier's window carry over from the lattice runs without
    // being retuned.
    let r = env_f64("ACT_R", 245.0);
    let als = env_f64("ACT_ALS", 3.5);
    let ncl = env_f64("ACT_NCL", 4.0);
    let h_bulk = env_f64("ACT_H", 1.5);
    let steps = env_usize("ACT_STEPS", 3_000_000);
    let save_every = env_usize("ACT_SAVE", 1000);
    let seed = env_usize("ACT_SEED", 0) as u64;
    let q_anchor = env_f64("ACT_Q", 1.0);
    let cg_tol = env_f64("ACT_CGTOL", 1e-8);
    let relax_steps = env_usize("ACT_RELAX", 0);
    let out = env_str("ACT_OUT", "");
    assert!(!out.is_empty(), "set ACT_OUT to the run directory");

    // The per-triangle winding is exact only while the director turns by less
    // than a quarter turn along an edge, so a core of width ncl needs elements at
    // or below half of it. This is the same constraint the passive comparison
    // carries, and it is the one the lattice fails at the cusp.
    assert!(
        h_bulk <= ncl / 2.0 + 1e-12,
        "ACT_H {h_bulk} too coarse for ACT_NCL {ncl}"
    );

    let curve = Epitrochoid { q: qc, d, r };
    let rc = curve.cusp_radius();
    // Four elements across the tip's own radius is what the passive charge test
    // needs, and at d = 0.99 that is 1.3e-3, a grading ratio above a thousand.
    // The active problem cannot pay for it: the advective step is explicit, so
    // the smallest element sets a CFL limit, and the conjugate-gradient solve
    // costs the grading ratio in iterations. `ACT_HMIN` puts a floor under it,
    // and the imposed winding printed below is the property that floor must not
    // break.
    let h_floor = env_f64("ACT_HMIN", 0.0);
    // At `d = 1` the curve is the paper's own epicycloid and its derivative
    // vanishes at every cusp, so there is no tangent to anchor against and no
    // curvature radius to refine towards: `rc` is zero and the grading would run
    // the element size to zero with it. The cusp is excised at the element size
    // instead, which is the sharpness the reference lattice itself carries, and
    // the element size then has nothing left to resolve below `h_bulk`. Default
    // it off below `d = 1`, where the curve is smooth and the existing grading
    // is what the passive charge test was calibrated against.
    // The sharp treatment at d = 1: one vertex at the exact cusp, with the two
    // boundary edges meeting there an element long. A re-entrant cusp has
    // interior angle `2 pi` and contributes `-pi` to the boundary's turning, so
    // the nephroid imposes 2 rather than 1, and the interior holds that whole
    // number. Rounding the cusp AND resolving it drops the turning number to 1,
    // which the nephroid pays for by gaining two `-1/2` cores: its complement is
    // `(4, 0)` at `d = 1` and `(4, 2)` at `d = 0.72`. Resolving is the second
    // half of that and it is not free, so read the imposed winding printed
    // below rather than `d`. See `tests/index_law.rs`.
    let cusp_edge = env_f64("ACT_CUSPEDGE", if d >= 1.0 { h_bulk } else { 0.0 });
    // With a fillet the element size near the cusp is a choice, not a
    // consequence: there is no curvature radius on the curve to refine towards,
    // and the arc is resolved at whatever `h_min` says. `ACT_HMIN` therefore
    // LOWERS it here, where on a smooth curve it raises a floor under the
    // grading. The binding constraint is the explicit diffusive limit
    // `gamma h^2 / (4 K)`, which has to stay above `dt`.
    let h_min = if cusp_edge > 0.0 {
        if h_floor > 0.0 { h_floor.min(h_bulk) } else { h_bulk }
    } else {
        (rc / 4.0).max(h_floor).min(h_bulk)
    };
    let mesh_opts = MeshOpts {
        h_bulk,
        h_min,
        cusp_edge,
        ..Default::default()
    };
    let mesh = confined_mesh(curve, mesh_opts);
    let quality = mesh.quality.clone();
    let (imposed, worst_step, n_steps_b) = mesh.imposed_charge(q_anchor);
    let nv = mesh.mesh.n_vertices();
    let boundary: Vec<usize> = mesh.boundary_vertices.clone();

    let params = NematicParams::from_length_scales(als, ncl, r.round() as usize);
    // The reference integrates at 1e-4, which its lattice needs because its elastic term
    // is explicit. Here it is implicit and the transport can be taken by a
    // backward trace, so the step is bounded by accuracy rather than by
    // stability, and the run length the paper's protocol asks for is otherwise
    // out of reach: 300 time units at 1e-4 is three million steps.
    let dt = env_f64("ACT_DT", params.dt);
    // The graded cusp sets the explicit limit, and the elastic term is implicit
    // precisely so this does not have to hold; it is reported because the bulk
    // and advective terms are explicit and do have to.
    let dt_diff = params.q_diffusive_dt_limit(mesh_opts.h_min);

    let p = LdgProblem::new(mesh, params, q_anchor).expect("operators");

    // The wall layer.
    //
    // The velocity is recovered as a discrete curl of the stream function, which
    // divides edge fluxes by dual areas. On a mesh graded a thousand to one those
    // areas fall by a million, so any error in psi is amplified by the same
    // factor, and the speed comes out inversely proportional to the element size:
    // 0.29 in the bulk against 70.9 on the 226 smallest vertices, with a local
    // Courant number of 50 where the bulk sits at 1e-4. That is a recovery
    // artefact, not a flow feature. A cusp tip is a stagnation corner of a
    // no-slip wall, and the sliver in question is 0.005 across where a defect
    // core is `ncl = 4`, so the continuum model resolves nothing inside it.
    //
    // Vertices on elements below `ACT_WALLH` are therefore held at no-slip along
    // with the wall itself. The anchoring is untouched: `Q` is still pinned only
    // on the true boundary, so the winding the mesh was chosen for is unaffected.
    let hloc = p.local_h();
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
    let layer = noslip.len() - boundary.len();
    println!(
        "  wall layer: {layer} interior vertices on elements below {wall_h}          held at no-slip, {} total",
        noslip.len()
    );
    // The CLAMPED wall. `new_confined` imposes psi = 0 and Delta psi = 0, which
    // is the simply supported plate, that is free slip, and on a disc under
    // uniform load it runs faster than no slip by exactly 2 sqrt 2. Measured
    // against the reference lattice at matched als and ncl, this solver sat 2.25
    // times high at the same defect state with everything else already pinned.
    // The clamped constructor adds dpsi/dn = 0 and costs one setup pass.
    // `ACT_WALL=slip` reverts to the simply supported wall, which exists so the
    // two conditions can be A/B'd on one binary. The clamped correction adds a
    // GLOBAL field, `sum_j coef_j phi_j` with `coef` from a dense solve on the
    // boundary unknowns, so a badly conditioned boundary system pollutes the
    // whole domain rather than a layer, and only holding everything else fixed
    // separates that from the physics.
    let slip_wall = std::env::var("ACT_WALL").map(|v| v == "slip").unwrap_or(false);
    let stokes = if slip_wall {
        SurfaceStokes::new_confined(&p.ops, &p.mesh.mesh, &noslip)
    } else {
        SurfaceStokes::new_confined_clamped(&p.ops, &p.mesh.mesh, &noslip)
    }
    .expect("confined Stokes factorisation");
    println!(
        "  wall: {} on {} vertices",
        if slip_wall { "simply supported (free slip), psi = 0, lap psi = 0" }
        else { "clamped (no-slip), psi = 0, dpsi/dn = 0" },
        noslip.len()
    );

    // `SurfaceStokes::solve` reads only `zeta_eff` and `eta` off this struct;
    // the molecular field comes from `LdgProblem`, which carries the reference's own
    // `H = K grad^2 Q - (A + C Tr(Q^2)) Q`, so the rotor convention's `a_eff`
    // never enters. Activity therefore reaches the physics through the stress
    // alone, which is where the reference puts it.
    let mut sp = volterra_core::ActiveNematicParams::default_test();
    sp.zeta_eff = p.params.zeta;
    sp.eta = p.params.eta;

    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).expect("run directory");

    let consts = format!(
        "{{\n  \"solver\": \"volterra-dec active_confined\",\n  \
         \"shape\": \"{shape}\",\n  \"bc\": \"epitrochoid\",\n  \
         \"cusps\": {},\n  \"q_curve\": {qc},\n  \"d\": {d},\n  \"r\": {r},\n  \
         \"q_anchor\": {q_anchor},\n  \"als\": {als},\n  \"ncl\": {ncl},\n  \
         \"h_bulk\": {h_bulk},\n  \"h_min\": {},\n  \"cusp_radius\": {rc},\n  \
         \"vertices\": {nv},\n  \"triangles\": {},\n  \"boundary_vertices\": {},\n  \
         \"min_angle_deg\": {},\n  \"obtuse\": {},\n  \
         \"imposed_charge\": {imposed},\n  \"imposed_worst_step_deg\": {worst_step},\n  \
         \"imposed_steps\": {n_steps_b},\n  \"wall_layer_h\": {wall_h},\n  \
         \"dt\": {dt},\n  \"dt_reference\": {dtk},\n  \"dt_diffusive_limit\": {dt_diff},\n  \
         \"steps\": {steps},\n  \"save_every\": {save_every},\n  \"seed\": {seed},\n  \
         \"K\": {},\n  \"A\": {},\n  \"C\": {},\n  \"gamma\": {},\n  \
         \"lambda\": {},\n  \"zeta\": {},\n  \"eta\": {}\n}}\n",
        curve.cusps(),
        mesh_opts.h_min,
        quality.triangles,
        quality.boundary_vertices,
        quality.min_angle_deg,
        quality.obtuse,
        p.params.k_frank,
        p.params.a_landau,
        p.params.c_landau,
        p.params.gamma,
        p.params.lambda,
        p.params.zeta,
        p.params.eta,
        dtk = params.dt,
    );
    std::fs::write(dir.join("consts.json"), &consts).expect("consts.json");

    // The mesh geometry, so the analysis and any render have the domain without
    // re-deriving it from the parameters.
    {
        let m = &p.mesh.mesh;
        let mut g = String::from("# x y\n");
        for v in &m.vertices {
            g.push_str(&format!("{:.6}\t{:.6}\n", v.x, v.y));
        }
        std::fs::write(dir.join("vertices.tsv"), g).expect("vertices.tsv");
        let mut b = String::from("# vertex x y nx ny\n");
        for (k, &vi) in p.mesh.boundary_vertices.iter().enumerate() {
            let v = m.vertices[vi];
            let n = p.mesh.boundary_normals[k];
            b.push_str(&format!(
                "{vi}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                v.x, v.y, n[0], n[1]
            ));
        }
        std::fs::write(dir.join("boundary.tsv"), b).expect("boundary.tsv");
    }

    println!(
        "{shape} d {d} q {q_anchor}: {nv} vertices, {} triangles, {} boundary, \
         min angle {:.1} deg, {} obtuse, worst cot {:.4}",
        quality.triangles, quality.boundary_vertices, quality.min_angle_deg,
        quality.obtuse,
        quality.worst_cot_weight
    );
    println!(
        "  R_cusp {rc:.5}, h_min {:.5}, imposed charge {imposed:+.2} \
         (worst boundary step {worst_step:.1} deg over {n_steps_b})",
        mesh_opts.h_min
    );
    println!(
        "  dt {dt:.2e}, explicit diffusive limit at h_min {dt_diff:.2e}, \
         {steps} steps = {:.1} time units, saving every {save_every}",
        steps as f64 * dt
    );

    // Sizing without stepping, so a geometry can be checked for the imposed
    // winding and the element count before any time is spent on it.
    if env_usize("ACT_DRYRUN", 0) != 0 {
        println!("  dry run: mesh and anchoring only, no time stepping");
        return;
    }

    // The advective transport. A backward trace by default: the mesh is graded
    // into the cusp, so an explicit `u . grad Q` is limited by the smallest
    // element and not by the bulk one, and at d = 0.99 that is a thousandth of a
    // bulk element. `ACT_SL=0` selects the differenced form, which is the direct
    // reading of the reference and is what the two agree on as `dt` falls.
    let sl = if env_usize("ACT_SL", 0) != 0 {
        // Refuses by default. `SemiLagrangian`'s point location is wrong on a
        // flat confined mesh: at zero flow it moves Q by the full magnitude s0,
        // reproduced by `the_backward_trace_at_zero_flow_is_the_identity`. The
        // wall layer removes the reason this path was wanted, so it is off unless
        // asked for and says so when it is.
        eprintln!(
            "  WARNING: ACT_SL=1 selects the backward trace, whose point location \
             is known wrong on flat confined meshes. Results are not trustworthy."
        );
        let m = &p.mesh.mesh;
        println!("  advection: backward trace (semi-Lagrangian), building BVH");
        Some(SemiLagrangian::new(
            m.vertices.iter().map(|v| [v.x, v.y, 0.0]).collect(),
            m.simplices.clone(),
        ))
    } else {
        println!("  advection: differenced on the mesh, CFL bound by h_min");
        None
    };

    // Where the fast vertices are. The explicit advective bound is local, so a
    // Courant number of thirty means either a real flow feature on a small
    // element or a recovery artefact on one, and the two call for different
    // fixes. Bucketing the speed by local element size separates them.
    if env_usize("ACT_DIAG", 0) != 0 {
        let q0 = p.random_state(seed);
        let vel = stokes.solve(&q0, &sp, &p.ops, &p.mesh.mesh);
        let mut buckets: Vec<(f64, f64, f64, usize)> = vec![(0.0, 0.0, 0.0, 0); 6];
        for i in 0..nv {
            let h = hloc[i];
            let b = if h < 0.005 { 0 } else if h < 0.05 { 1 }
                else if h < 0.2 { 2 } else if h < 0.6 { 3 }
                else if h < 1.2 { 4 } else { 5 };
            let sp_i = vel.speed(i);
            buckets[b].0 += sp_i;
            buckets[b].1 = buckets[b].1.max(sp_i);
            buckets[b].2 = buckets[b].2.max(dt * sp_i / h.max(1e-300));
            buckets[b].3 += 1;
        }
        println!("  local h        vertices   mean |u|    max |u|    max CFL");
        let names = ["< 0.005", "0.005-0.05", "0.05-0.2", "0.2-0.6", "0.6-1.2", "> 1.2"];
        for (b, n) in names.iter().enumerate() {
            let (sum, mx, cfl, cnt) = buckets[b];
            if cnt == 0 { continue; }
            println!("  {n:<12} {cnt:>8}  {:>9.3}  {mx:>9.3}  {cfl:>9.3}",
                     sum / cnt as f64);
        }
        return;
    }

    // Resume from a checkpoint if one is there and `ACT_RESUME` asks for it, so
    // a run can be extended rather than repeated.
    let mut q = p.random_state(seed);
    let mut resume_step = 0usize;
    let resuming = env_usize("ACT_RESUME", 0) != 0;
    if resuming {
        let f = dir.join("Q_state.tsv");
        match std::fs::read_to_string(&f) {
            Ok(text) => {
                let mut n = 0usize;
                for line in text.lines() {
                    if let Some(rest) = line.strip_prefix("# step ") {
                        resume_step = rest.trim().parse().unwrap_or(0);
                    }
                    if line.starts_with('#') || line.trim().is_empty() {
                        continue;
                    }
                    let mut it = line.split('\t');
                    let (a, b) = (it.next(), it.next());
                    if let (Some(a), Some(b)) = (a, b) {
                        if n < nv {
                            q.q1[n] = a.parse().unwrap_or(q.q1[n]);
                            q.q2[n] = b.parse().unwrap_or(q.q2[n]);
                            n += 1;
                        }
                    }
                }
                assert_eq!(n, nv, "checkpoint has {n} vertices, mesh has {nv}");
                p.impose_anchoring(&mut q);
                println!(
                    "  resumed from {} ({n} vertices, at step {resume_step})",
                    f.display()
                );
            }
            Err(e) => panic!("ACT_RESUME set but {} unreadable: {e}", f.display()),
        }
    }
    // An optional passive settle, which starts the active run from an ordered
    // field rather than from noise. Zero by default: the lattice runs start from
    // noise and the comparison is against those.
    for _ in 0..relax_steps {
        p.step_passive(&mut q, dt, cg_tol);
    }

    // A resume must extend the record rather than replace it. Opening for
    // writing truncates, so a continued run would silently destroy the frames it
    // is continuing from, which is the one outcome a checkpoint exists to
    // prevent.
    //
    // The frame to continue from is the checkpoint's, not the last one written.
    // A run stopped between checkpoints has frames on disk that the restored
    // state is behind, and appending after those would record two different
    // times under one frame index. Rows from the checkpoint frame onward are
    // dropped and rewritten, so the record is a single consistent series with no
    // instant recorded twice.
    let frame0 = if resuming { resume_step / save_every } else { 0 };
    if resuming {
        for (name, key) in [("defects.tsv", 0usize), ("series.tsv", 0usize)] {
            let _ = key;
            let path = dir.join(name);
            if let Ok(text) = std::fs::read_to_string(&path) {
                let mut kept = String::new();
                let mut dropped = 0usize;
                for line in text.lines() {
                    let keep = if line.starts_with('#') || line.trim().is_empty() {
                        true
                    } else {
                        line.split('\t')
                            .next()
                            .and_then(|f| f.parse::<usize>().ok())
                            .map(|f| f < frame0)
                            .unwrap_or(false)
                    };
                    if keep {
                        kept.push_str(line);
                        kept.push('\n');
                    } else {
                        dropped += 1;
                    }
                }
                if dropped > 0 {
                    println!("  {name}: dropped {dropped} row(s) past the checkpoint");
                }
                std::fs::write(&path, kept).expect(name);
            }
        }
    }
    let open = |name: &str| -> std::fs::File {
        if resuming {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(dir.join(name))
                .expect(name)
        } else {
            std::fs::File::create(dir.join(name)).expect(name)
        }
    };
    let mut tsv = std::io::BufWriter::new(open("defects.tsv"));
    if !resuming {
        writeln!(tsv, "# frame x y charge").unwrap();
    }
    // Written as the run goes rather than at the end. A run of this length that
    // is interrupted, and one will be, otherwise leaves no series and no state to
    // resume from, which is the whole diagnostic record.
    let mut series = std::io::BufWriter::new(open("series.tsv"));
    if !resuming {
        writeln!(
            series,
            "# frame step t n_plus n_minus charge S_median speed_max worst_dq"
        )
        .unwrap();
    }
    let checkpoint_every = env_usize("ACT_CHECKPOINT", 50);
    let write_state = |q: &volterra_dec::qfield::QField, step: usize| {
        let mut f = format!("# step {step}\n# q1 q2\n");
        for i in 0..q.n_vertices {
            f.push_str(&format!("{:.9}\t{:.9}\n", q.q1[i], q.q2[i]));
        }
        let tmp = dir.join("Q_state.tsv.tmp");
        std::fs::write(&tmp, f).expect("Q_state");
        std::fs::rename(&tmp, dir.join("Q_state.tsv")).expect("Q_state rename");
    };

    // Per-frame director field, for rendering. `Q_state.tsv` is a checkpoint and
    // gets overwritten, so it carries one instant and cannot make a film. These
    // are f32 pairs, little-endian, `q1 q2` per vertex in `vertices.tsv` order:
    // 41 kB a frame against 100 kB as text, and the renderer wants floats rather
    // than digits anyway. `ACT_QFRAMES=0` turns them off.
    let q_frames = env_usize("ACT_QFRAMES", 1) != 0;
    let q_dir = dir.join("qframes");
    if q_frames {
        std::fs::create_dir_all(&q_dir).expect("qframes dir");
    }
    let write_q_frame = |q: &volterra_dec::qfield::QField, frame: usize| {
        if !q_frames {
            return;
        }
        let mut buf = Vec::with_capacity(q.n_vertices * 8);
        for i in 0..q.n_vertices {
            buf.extend_from_slice(&(q.q1[i] as f32).to_le_bytes());
            buf.extend_from_slice(&(q.q2[i] as f32).to_le_bytes());
        }
        let path = q_dir.join(format!("q_{frame:05}.f32"));
        std::fs::write(path, buf).expect("q frame");
    };

    // Per-frame velocity, `ACT_DUMPVEL=1`. Same layout as the Q frames: f32
    // pairs, little-endian, `(ux, uy)` per vertex in `vertices.tsv` order. Off by
    // default, since a run that does not need them should not pay for them.
    let vel_frames = env_usize("ACT_DUMPVEL", 0) != 0;
    let v_dir = dir.join("vframes");
    if vel_frames {
        std::fs::create_dir_all(&v_dir).expect("vframes dir");
    }
    let write_vel_frame = |vel: &VelocityField, frame: usize| {
        if !vel_frames {
            return;
        }
        let mut buf = Vec::with_capacity(vel.n_vertices * 8);
        for i in 0..vel.n_vertices {
            buf.extend_from_slice(&(vel.vx(i) as f32).to_le_bytes());
            buf.extend_from_slice(&(vel.vy(i) as f32).to_le_bytes());
        }
        std::fs::write(v_dir.join(format!("u_{frame:05}.f32")), buf).expect("vel frame");
    };

    let merge = 1.5 * h_bulk;
    let hloc = p.local_h();
    let mut cfl_worst = 0.0_f64;
    let s0_ref = p.params.s0();
    // Passes of the variational fixed point. DEFAULT 1, which is the sequential
    // scheme.
    //
    // Iterating the coupling does NOT implement Onsager's principle and does not
    // stabilise it. Measured on the nephroid at d = 0.99, dt = 5e-5, no elastic
    // mask: one pass reaches S = 3.1e8 at step 17, three passes reach S = 1.2e109
    // at step 10, and the first pass alone raises |u|max from 141.7 to 610.4. The
    // fixed point is unstable, so a Picard substitution converges away from it.
    //
    // The property that makes the variational integrator stable is monotone
    // decrease of the Rayleighian, which a substitution does not have. Getting it
    // requires solving the minimisation, a descent on R with a line search or a
    // metric chosen to guarantee R_(eps) <= R_(0), rather than iterating a
    // sequential update to consistency. `free_energy` is the piece of that which
    // exists; the descent is not written.
    let picard = env_usize("ACT_PICARD", 1).max(1);
    // Chain two vertex-gradient operators for `grad u` instead of taking it from
    // the stream function. Kept only so the A/B can be run.
    let du_chain = env_usize("ACT_DUCHAIN", 0) != 0;
    // Per-report budget of `d|q|^2/dt` at the vertex of largest S, split into the
    // four terms that drive it. A runaway in S has to come from one of them, and
    // guessing which has cost more than measuring it would have.
    let budget = env_usize("ACT_BUDGET", 0) != 0;
    let mut picard_worst = 0.0_f64;
    let mut t_stokes = 0.0_f64;
    let mut t_ldg = 0.0_f64;
    // The stream function carried between steps, so the Stokes solve starts from
    // the previous answer instead of from zero.
    let mut psi: Option<Vec<f64>> = None;
    let mut pcg_worst = 0usize;
    let stokes_tol = env_f64("ACT_STOKESTOL", 1e-8);
    // The full Beris-Edwards stress, or the active term alone. The active term
    // alone is what every run before 2026-08-19 carried, and it omits the elastic
    // backflow that opposes the active flow, so it is kept only for comparison.
    let full_stress = env_usize("ACT_FULLSTRESS", 1) != 0;
    // Vertices where the ELASTIC stress is suppressed, the active term kept.
    // The wall layer by default: those elements are below the core size, so a
    // free energy differentiated twice across them is not a force the physics
    // exerts, and it is what makes the explicit stress unstable. ACT_ELASTICMASK=0
    // keeps the elastic terms everywhere, which is what blows up at d = 0.99.
    // Its own threshold, not the no-slip one. The two answer different questions.
    // ACT_WALLH bounds where the discrete curl recovery produces a spurious
    // velocity; ACT_ELASTICH bounds where the explicit elastic stress is stable.
    // The second is set by the diffusive limit, which goes as h^2: measured on
    // this mesh, h = 0.05 gives 3.81e-6, h = 0.25 gives 9.54e-5 and h = 0.5 gives
    // 3.81e-4, so a step of 2e-4 needs the elastic terms off below about 0.4.
    // Masking at ACT_WALLH alone moved the blow-up from step 17 to step 177 and
    // no further, which is what pins the threshold to the diffusive limit rather
    // than to the recovery artefact.
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
    if full_stress {
        eprintln!(
            "  elastic stress suppressed on {} vertices of {nv} (elements below {elastic_h})",
            elastic_mask.len()
        );
    }

    if !full_stress {
        eprintln!("  ACT_FULLSTRESS=0: active stress only, no elastic backflow");
    }

    let t0 = std::time::Instant::now();
    let mut frame = frame0;
    let mut cg_worst = 0usize;
    let mut dq_worst = 0.0_f64;
    for step in 0..=steps {
        // One Stokes solve per step, shared by the diagnostics and the update.
        // Solving twice on a saved frame was costing a full Poisson solve every
        // save, which is invisible at a cadence of a thousand and doubles the
        // cost of a short probe, so it flattered nothing and confused timing.
        let t_a = std::time::Instant::now();
        // Time-incremental variational step, after Onsager's principle as Zhu,
        // Saintillan and Chern apply it: the potential is evaluated at the NEW
        // state, so the step is a descent and the stiffest element in the mesh
        // does not set the step size.
        //
        // Sequentially, the elastic stress is read off the OLD `Q`, the flow
        // follows, and `Q` is advanced. That makes the free energy's gradient
        // explicit, and it inherits `h^-3` through `-lambda H` and `h^-2` through
        // the Ericksen term. On this mesh the diffusive limit is 2.12e-9 against
        // a step of 5e-5, and the field reaches S = 3.1e8 by step 17.
        //
        // Closing the loop removes that. Each pass recomputes the stress from the
        // current iterate rather than from the old state, so at convergence the
        // stress, the flow and `Q` are all consistent at the end of the step,
        // which is the stationarity condition of the discrete Rayleighian. The
        // Q solve is already implicit in the Frank term, so what was missing was
        // only the coupling back through the flow. `ACT_PICARD=1` recovers the
        // sequential scheme.
        let mut vel_out: Option<VelocityField> = None;
        let mut psi_out: Vec<f64> = Vec::new();
        let mut pits_out = 0usize;
        let mut q_next = q.clone();
        let mut picard_gap = 0.0_f64;
        for pass in 0..picard {
            let src = if pass == 0 { &q } else { &q_next };
            let (vel_p, psi_p, pits_p) = if full_stress {
                let (s1, s2, sa) = p.beris_edwards_stress_masked(src, &elastic_mask);
                stokes.solve_stress_warm(
                    &s1, &s2, &sa, p.params.eta, &p.mesh.mesh, psi.as_deref(), stokes_tol,
                )
            } else {
                stokes.solve_warm(src, &sp, &p.ops, &p.mesh.mesh, psi.as_deref(), stokes_tol)
            };
            let mut v2p = vec![[0.0_f64; 2]; nv];
            for i in 0..nv {
                v2p[i] = [vel_p.v[i][0], vel_p.v[i][1]];
            }
            let mut trial = q.clone();
            // The co-rotational term is driven by `grad u`. Differentiating the
            // recovered velocity chains two different vertex-gradient operators
            // and does not converge: on this mesh the relative error in `E_xy`
            // falls only from 1.2e-1 to 7.2e-2 across a factor of four in `h`,
            // which is `O(h^0.4)`. Taking the same quantity from the stream
            // function is `O(h^1.1)` and is exactly divergence-free at every
            // vertex. `ACT_DUCHAIN=1` restores the chained form for comparison.
            if du_chain {
                p.step_active(&mut trial, &v2p, dt, cg_tol, sl.as_ref());
            } else {
                let du = p.velocity_gradients_from_psi(&psi_p);
                p.step_active_with_du(&mut trial, &v2p, &du, dt, cg_tol, sl.as_ref());
            }
            picard_gap = (0..nv)
                .map(|i| {
                    (trial.q1[i] - q_next.q1[i]).abs().max((trial.q2[i] - q_next.q2[i]).abs())
                })
                .fold(0.0_f64, f64::max);
            q_next = trial;
            vel_out = Some(vel_p);
            psi_out = psi_p;
            pits_out = pits_p;
        }
        picard_worst = picard_worst.max(picard_gap);
        let (vel, psi_new, pits) = (vel_out.unwrap(), psi_out, pits_out);
        psi = Some(psi_new);
        pcg_worst = pcg_worst.max(pits);
        t_stokes += t_a.elapsed().as_secs_f64();
        if step % save_every == 0 {
            let (pos, neg, charge, list) = p.defect_summary(&q, merge);
            for (x, y, ch) in &list {
                writeln!(tsv, "{frame}\t{x:.4}\t{y:.4}\t{ch}").unwrap();
            }
            let mut s = p.order_parameter(&q);
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            let vmax = (0..nv).map(|i| vel.speed(i)).fold(0.0_f64, f64::max);
            // Absolute step and time, so a resumed run's series continues the
            // one it resumed from instead of restarting its own clock.
            let astep = step + resume_step;
            writeln!(
                series,
                "{frame}\t{astep}\t{:.6}\t{pos}\t{neg}\t{charge:+.2}\t{med:.6}\t\
                 {vmax:.6}\t{dq_worst:.3e}",
                astep as f64 * dt
            )
            .unwrap();
            write_q_frame(&q, frame);
            write_vel_frame(&vel, frame);
            if frame % checkpoint_every == 0 {
                tsv.flush().unwrap();
                series.flush().unwrap();
                write_state(&q, step + resume_step);
            }
            if budget {
                // The vertex whose order parameter is largest, and the four
                // contributions to d|q|^2/dt there. `2 (q . rate)` for each, so
                // they sum to the total and their signs are directly comparable.
                let (bi, bs) = (0..nv)
                    .map(|i| (i, 2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i])))
                    .fold((0usize, 0.0_f64), |a, x| if x.1 > a.1 { x } else { a });
                let v2b: Vec<[f64; 2]> = (0..nv).map(|i| [vel.vx(i), vel.vy(i)]).collect();
                let dub = if du_chain {
                    p.velocity_gradients(&v2b)
                } else {
                    p.velocity_gradients_from_psi(psi.as_deref().unwrap_or(&[]))
                };
                let cor = p.corotational(&q, &dub);
                let hh = p.molecular_field(&q);
                let gq = p.q_gradients(&q);
                let (q1, q2) = (q.q1[bi], q.q2[bi]);
                let trq = 2.0 * (q1 * q1 + q2 * q2);
                let gam = p.params.gamma;
                // Bulk and elastic halves of H, separated so the restoring term
                // can be read on its own.
                let bulkc = p.params.a_landau + p.params.c_landau * trq;
                let (hb1, hb2) = (-bulkc * q1, -bulkc * q2);
                let (he1, he2) = (hh.q1[bi] - hb1, hh.q2[bi] - hb2);
                let adv1 = v2b[bi][0] * gq[bi][0] + v2b[bi][1] * gq[bi][1];
                let adv2 = v2b[bi][0] * gq[bi][2] + v2b[bi][1] * gq[bi][3];
                let d_cor = 2.0 * (q1 * cor.q1[bi] + q2 * cor.q2[bi]);
                let d_bulk = 2.0 * (q1 * hb1 + q2 * hb2) / gam;
                let d_el = 2.0 * (q1 * he1 + q2 * he2) / gam;
                let d_adv = -2.0 * (q1 * adv1 + q2 * adv2);
                let (dxux, dxuy, dyux) = (dub[bi][0], dub[bi][1], dub[bi][2]);
                let tr_qe = 2.0 * q1 * dxux + q2 * (dyux + dxuy);
                let mut onb = vec![false; nv];
                for &v in &p.mesh.boundary_vertices {
                    onb[v] = true;
                }
                // Strain statistics over the interior, the quantity to compare
                // with the lattice rather than argue about: the reference at
                // these constants has |E| rms 2.88, a divergence of 1.3e-2, and
                // its max-S cell sits at cos(Q, E) = +0.05, so its co-rotation
                // pulls S back DOWN.
                let (mut e2, mut w2, mut dv2, mut emax, mut cnt) =
                    (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0usize);
                for i in 0..nv {
                    if onb[i] {
                        continue;
                    }
                    let (a0, a1, a2, a3) = (dub[i][0], dub[i][1], dub[i][2], dub[i][3]);
                    let (exx, exy) = (a0, 0.5 * (a1 + a2));
                    let em = (exx * exx + exy * exy).sqrt();
                    e2 += em * em;
                    w2 += (0.5 * (a1 - a2)).powi(2);
                    dv2 += (a0 + a3).powi(2);
                    emax = emax.max(em);
                    cnt += 1;
                }
                let n = cnt.max(1) as f64;
                // Steady-state power balance, the check that needs neither code
                // to be right: the active stress injects, the fluid dissipates
                // viscously and the director rotationally, so
                //     P_a = zeta <Q:E> = 2 eta <E:E> + <H:H>/gamma = D_v + D_rot.
                // Measured on the two lattices this closes at 0.968 and 0.973.
                {
                    let (mut pa, mut dv, mut dr, mut cnt) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
                    for i in 0..nv {
                        if onb[i] {
                            continue;
                        }
                        let (a0, a1, a2) = (dub[i][0], dub[i][1], dub[i][2]);
                        let (exx, exy) = (a0, 0.5 * (a1 + a2));
                        pa += 2.0 * (q.q1[i] * exx + q.q2[i] * exy);
                        dv += 2.0 * (exx * exx + exy * exy);
                        dr += 2.0 * (hh.q1[i] * hh.q1[i] + hh.q2[i] * hh.q2[i]);
                        cnt += 1;
                    }
                    let n2 = cnt.max(1) as f64;
                    let (pa, dv, dr) = (
                        p.params.zeta * pa / n2,
                        2.0 * p.params.eta * dv / n2,
                        dr / (n2 * p.params.gamma),
                    );
                    println!(
                        "   power  P_a {pa:.4e}  D_v {dv:.4e}  D_rot {dr:.4e}                           (D_v + D_rot)/P_a {:.4}",
                        (dv + dr) / pa
                    );
                }
                let exy_b = 0.5 * (dub[bi][1] + dub[bi][2]);
                let em_b = (dub[bi][0] * dub[bi][0] + exy_b * exy_b).sqrt();
                let cos_b =
                    (q1 * dub[bi][0] + q2 * exy_b) / ((q1 * q1 + q2 * q2).sqrt() * em_b + 1e-30);
                println!(
                    "   budget v{bi} S {:.4} |u| {:.2} TrQE {tr_qe:+.3e} |E| {em_b:.3} cos {cos_b:+.3} | corot {d_cor:+.3e} bulk {d_bulk:+.3e} elast {d_el:+.3e} adv {d_adv:+.3e} sum {:+.3e} | field |E|rms {:.3} max {emax:.3} |w|rms {:.3} divrms {:.4}",
                    bs.sqrt(),
                    (v2b[bi][0].powi(2) + v2b[bi][1].powi(2)).sqrt(),
                    d_cor + d_bulk + d_el + d_adv,
                    (e2 / n).sqrt(),
                    (w2 / n).sqrt(),
                    (dv2 / n).sqrt()
                );
            }
            if frame % env_usize("ACT_REPORT", 5) == 0 {
                println!(
                    "  frame {frame:>5} step {astep:>9} t {:>8.2}  {pos} (+1/2) \
                     {neg} (-1/2) charge {charge:+.2}  S {med:.4}  |u|max {vmax:.3}  \
                     cg {cg_worst}/{pcg_worst}  dQ {dq_worst:.2e}  CFL {cfl_worst:.3}  [{:.1}s]",
                    astep as f64 * dt,
                    t0.elapsed().as_secs_f64()
                );
                std::io::stdout().flush().ok();
            }
            // Per frame, not per report. These four are running maxima, and
            // resetting them only when a line is printed made every one of them
            // a maximum over the report window instead: at `ACT_REPORT=50` the
            // `worst_dq` column held the initial transient's 1.2e-1 for all
            // forty-one frames of a run whose true per-frame value was 1e-4 and
            // falling, which reads exactly like a step limiter binding. The
            // series carries every frame, so thinning the printing loses nothing.
            cg_worst = 0;
            dq_worst = 0.0;
            cfl_worst = 0.0;
            pcg_worst = 0;
            frame += 1;
        }
        if step == steps {
            break;
        }
        let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.vx(i), vel.vy(i)]).collect();
        cfl_worst = cfl_worst.max(p.courant(&v2, dt, &hloc).0);
        let t_b = std::time::Instant::now();
        let dq_max = (0..nv)
            .map(|i| (q_next.q1[i] - q.q1[i]).abs().max((q_next.q2[i] - q.q2[i]).abs()))
            .fold(0.0_f64, f64::max);
        let its = 0usize;
        q = q_next;
        t_ldg += t_b.elapsed().as_secs_f64();
        cg_worst = cg_worst.max(its);
        dq_worst = dq_worst.max(dq_max);

        // Stop on a field that has left the reals, rather than writing frames of
        // it. A run that blew up on 2026-08-19 carried on for 300 frames after
        // `S` reached 5e38, every one of them recorded as a legitimate frame with
        // `|u|max 0.000` because the solver returns zero on a NaN right-hand
        // side. The output was indistinguishable from an arrested run at a
        // glance, which is the reason this check exists.
        // Finiteness alone is not the check. The run that motivated this reached
        // S = 5.5e38, which is a perfectly finite f64, and `dQ` then read zero
        // because the saturated field is a fixed point of the semi-implicit
        // solve. So bound the magnitude against the equilibrium the Landau
        // potential sets: `S` cannot exceed `s0` by any meaningful factor, and a
        // hundredfold is far outside any transient.
        let (worst_v, s_max) = (0..nv)
            .map(|i| (i, (2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i])).sqrt()))
            .fold((0usize, 0.0_f64), |acc, x| if x.1 > acc.1 { x } else { acc });
        if !dq_max.is_finite() || !s_max.is_finite() || s_max > 100.0 * s0_ref {
            // WHERE it diverges decides what is wrong. A blow-up sitting on the
            // wall is the boundary condition or the recovery there; one in the
            // bulk on the smallest elements is the step against the grading; one
            // at a cusp is the corner singularity the geometry actually has.
            let m = &p.mesh.mesh;
            let pw = m.vertex(worst_v);
            let on_wall = boundary.contains(&worst_v);
            let d_wall = boundary
                .iter()
                .map(|&b| {
                    let pb = m.vertex(b);
                    ((pw[0] - pb[0]).powi(2) + (pw[1] - pb[1]).powi(2)).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            // How many vertices have already left the equilibrium, which
            // separates one bad point from a field-wide failure.
            let hot = (0..nv)
                .filter(|&i| (2.0 * (q.q1[i] * q.q1[i] + q.q2[i] * q.q2[i])).sqrt() > 2.0 * s0_ref)
                .count();
            eprintln!(
                "ABORT at step {step}, t {:.4}: S reached {s_max:.3e} against an equilibrium \
                 s0 of {s0_ref:.4} (worst dQ {dq_max:.3e}). The explicit stress is unstable at \
                 this step and mesh grading; reduce ACT_DT or raise ACT_HMIN."
            , step as f64 * dt);
            eprintln!(
                "  worst vertex {worst_v} at ({:.3}, {:.3}), h {:.4}, {} the wall \
                 ({:.3} away); {hot} of {nv} vertices are above 2 s0",
                pw[0], pw[1], hloc[worst_v],
                if on_wall { "ON" } else { "off" }, d_wall
            );
            // Flush before leaving. `process::exit` runs no destructors, so an
            // aborted run used to discard every buffered series row since the
            // last checkpoint, which is exactly the approach to the blow-up that
            // the abort exists to let someone look at.
            tsv.flush().ok();
            series.flush().ok();
            std::process::exit(2);
        }
    }
    tsv.flush().unwrap();
    series.flush().unwrap();

    // The final field, so a run can be restarted or re-measured without the
    // frames, which are not written: the defect worldlines are the observable
    // and the fields are what the lattice study had to purge.
    {
        let mut f = String::from("# q1 q2\n");
        for i in 0..nv {
            f.push_str(&format!("{:.9}\t{:.9}\n", q.q1[i], q.q2[i]));
        }
        std::fs::write(dir.join("Q_final.tsv"), f).expect("Q_final.tsv");
    }

    println!(
        "done: {frame} frames, {:.1} time units, {:.0} s  \
         (Stokes {t_stokes:.1} s, Beris-Edwards {t_ldg:.1} s, \
         {:.3} s per step)",
        steps as f64 * dt,
        t0.elapsed().as_secs_f64(),
        (t_stokes + t_ldg) / steps.max(1) as f64
    );
}
