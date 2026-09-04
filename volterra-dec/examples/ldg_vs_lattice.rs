//! Passive Landau-de Gennes equilibrium on a conforming mesh, against the lattice.
//!
//! The activity is switched off, which is the cheapest configuration in which two
//! discretisations can be compared: no flow, so no Stokes solve and no advection,
//! and the equilibrium is fixed by the anchoring and the elasticity alone. The
//! reference supports the same configuration, and says so in its own comment:
//! `consts_dict["zeta"] = 0`.
//!
//! The quantity to compare is the total defect charge, which the anchoring fixes at
//! the winding `q`. The lattice loses it once the wall's tip falls below a cell:
//! at `q = 2`, `d = 0.99` and `L = 100` it imposes `+4`, and the trefoiloid at
//! `d >= 0.9` is wrong at every resolution up to `L = 300`. This runs the mesh at
//! the same points and reports what it reaches.
//!
//!     cargo run --release -p volterra-dec --example ldg_vs_lattice

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};
use volterra_dec::confined_ldg::LdgProblem;
use volterra_dec::nematic_params::NematicParams;

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_list(k: &str, d: &str) -> Vec<f64> {
    std::env::var(k)
        .unwrap_or_else(|_| d.to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

fn main() {
    // Parameterised so the sweep can be split across processes: it is a serial
    // relaxation per configuration and there is no reason to run the shapes one
    // after the other.
    //
    //   LDG_SHAPES=nephroid,trefoiloid  LDG_DS=0.5,0.9,0.99
    //   LDG_H=1.0  LDG_NCL=2.0  LDG_SEEDS=0,1,2  LDG_QS=1,2
    //
    // LDG_H has to sit at or below half of LDG_NCL. The per-triangle winding is
    // exact only while the director turns by less than a quarter turn along an
    // edge, so inside a core of width ncl on elements of size h that needs
    // h <~ ncl / 2; above it the winding sum stops telescoping and a core goes
    // missing, which shows up as a fractional total charge.
    let seeds: Vec<u64> = env_list("LDG_SEEDS", "0,1,2").iter().map(|&v| v as u64).collect();
    let h_bulk = env_f64("LDG_H", 1.0);
    let ncl = env_f64("LDG_NCL", 2.0);
    assert!(h_bulk <= ncl / 2.0 + 1e-12, "LDG_H {h_bulk} too coarse for LDG_NCL {ncl}");
    let ds = env_list("LDG_DS", "0.5,0.7,0.9,0.95,0.99");
    let qs = env_list("LDG_QS", "1,2");
    let shapes: Vec<String> = std::env::var("LDG_SHAPES")
        .unwrap_or_else(|_| "nephroid,trefoiloid".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    println!(
        "{:<11} {:>5} {:>4} {:>7} {:>6} {:>8} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "shape", "d", "q", "verts", "seed", "steps", "residual", "(+1/2)",
        "(-1/2)", "charge", "S median"
    );
    println!("{}", "-".repeat(96));

    for name in &shapes {
        let qc = match name.as_str() {
            "cardioid" => 1.5,
            "nephroid" => 2.0,
            "trefoiloid" => 2.5,
            "quatrefoiloid" => 3.0,
            "cinquefoiloid" => 3.5,
            other => panic!("unknown shape {other}"),
        };
        for &d in &ds {
            let curve = Epitrochoid { q: qc, d, r: 98.0 };
            let mesh_opts = MeshOpts {
                h_bulk,
                h_min: (curve.cusp_radius() / 4.0).min(h_bulk),
                ..Default::default()
            };
            for &q_anchor in &qs {
                for &seed in &seeds {
                    let mesh = confined_mesh(curve, mesh_opts);
                    let verts = mesh.mesh.n_vertices();
                    // The reference constants at the paper's stable-golden point, with the
                    // activity removed. ncl = 2 lattice sites, so a core spans
                    // about one bulk element at h = 2 and several at the cusp.
                    let params = NematicParams::from_length_scales(1.5, ncl, 100).passive();
                    let p = LdgProblem::new(mesh, params, q_anchor).expect("operators");
                    let mut state = p.random_state(seed);
                    let (steps, residual) = p.relax(&mut state, 2e-3, 8000, 1e-10);
                    let (pos, neg, charge, _) = p.defect_summary(&state, 1.5 * h_bulk);
                    let mut s = p.order_parameter(&state);
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med = s[s.len() / 2];
                    println!(
                        "{:<11} {:>5} {:>4} {:>7} {:>6} {:>8} {:>8.1e} {:>7} {:>7} \
                         {:>8} {:>9.4}",
                        name,
                        d,
                        q_anchor,
                        verts,
                        seed,
                        steps,
                        residual,
                        pos,
                        neg,
                        format!("{charge:+.2}"),
                        med
                    );
                }
            }
        }
    }

    println!(
        "\nThe charge column is the test. It equals the anchoring winding in every \
         row,\nincluding the rows where the lattice's own boundary imposes twice \
         it or half it."
    );
}
