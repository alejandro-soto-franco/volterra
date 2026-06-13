//! Substep-by-substep comparison against correct Python reference intermediates (Task E).
//! References from dump_step_p.py (correct numba kernels, same IC as dump_step.py).

use volterra_cgpo::{
    bc::{apply_h_boundary_conditions, apply_p_boundary_conditions},
    nematic::{calculate_pi, h_s_from_q},
    nephroid_boundary,
    ops::{div_vector, upwind_advective_term},
    stokes::{calculate_pressure_terms, get_u_update, relax_pressure_inner_loop},
};

const LX: usize = 30;
const LY: usize = 30;
const N: usize = LX * LY;
const N2: usize = N * 2;

const K_ELASTIC: f64 = 16384.0;
const GAMMA: f64 = 100.0;
const LAMBDA: f64 = 0.7;
const ZETA: f64 = K_ELASTIC;
const A_LANDAU: f64 = -K_ELASTIC / 81.0;
const C_LANDAU: f64 = K_ELASTIC / 81.0;
const DT: f64 = 1e-4;
const NU: f64 = 404.771_540_501_552_6;
const RHO: f64 = 1.0;
const MAX_P_ITERS: usize = 20;

fn load_txt(path: &str) -> Vec<f64> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap_or_else(|e| panic!("parse error in {path}: {e}")))
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

fn max_abs_diff_at(a: &[f64], b: &[f64]) -> (f64, usize) {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).enumerate()
        .map(|(i, (x, y))| ((x - y).abs(), i))
        .fold((0.0_f64, 0), |(md, mi), (d, i)| if d > md { (d, i) } else { (md, mi) })
}

#[test]
fn debug_substeps() {
    let ref_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ref");

    // IC from dump_step.py (same as used in the one-step reference run)
    let q_ic = load_txt(&format!("{ref_dir}/step_Q_ic.txt"));
    let u_ic = load_txt(&format!("{ref_dir}/step_u_ic.txt"));
    let p_ic = load_txt(&format!("{ref_dir}/step_p_ic.txt"));

    // Intermediate references from dump_step_p.py (correct numba kernels)
    let h_ref     = load_txt(&format!("{ref_dir}/step_H.txt"));
    let s_ref     = load_txt(&format!("{ref_dir}/step_S.txt"));
    let pi_s_ref  = load_txt(&format!("{ref_dir}/step_Pi_S.txt"));
    let pi_a_ref  = load_txt(&format!("{ref_dir}/step_Pi_A.txt"));
    let p_rhs_ref = load_txt(&format!("{ref_dir}/step_p_rhs.txt"));
    let p_mid_ref = load_txt(&format!("{ref_dir}/step_p_mid.txt"));

    // Final references from dump_step.py (the correctness gate)
    let q_final_ref = load_txt(&format!("{ref_dir}/step_Q_ref.txt"));
    let u_final_ref = load_txt(&format!("{ref_dir}/step_u_ref.txt"));
    let p_final_ref = load_txt(&format!("{ref_dir}/step_p_ref.txt"));

    let bnd = nephroid_boundary(LX, LY);

    let q = q_ic.clone();
    let u = u_ic.clone();
    let mut h = vec![0.0_f64; N2];
    let mut s = vec![0.0_f64; N2];

    // ── Step 1: H_S_from_Q ──────────────────────────────────────────────────
    h_s_from_q(&u, &q, &mut h, &mut s, A_LANDAU, C_LANDAU, K_ELASTIC, LAMBDA, &bnd);

    let diff_h = max_abs_diff(&h, &h_ref);
    let diff_s = max_abs_diff(&s, &s_ref);
    eprintln!("Step 1 H_S_from_Q:   max|H diff|={diff_h:.3e}  max|S diff|={diff_s:.3e}");

    // ── Step 2: apply_H_bc (u=0 so H should be -gamma*S on boundary) ────────
    {
        let h_ptr = h.as_mut_ptr();
        let h_slice = unsafe { std::slice::from_raw_parts_mut(h_ptr, N2) };
        apply_h_boundary_conditions(h_slice, GAMMA, &q, &u, &s, &bnd);
    }
    // After H_bc, the Python step_H.txt was saved AFTER apply_H_bc,
    // so compare against step_H.txt
    let diff_h_bc = max_abs_diff(&h, &h_ref);
    eprintln!("Step 2 apply_H_bc:   max|H diff|={diff_h_bc:.3e}  (vs step_H which is POST-bc)");

    // ── Step 3: calculate_Pi ────────────────────────────────────────────────
    let mut pi_s = vec![0.0_f64; N2];
    let mut pi_a = vec![0.0_f64; N];
    calculate_pi(&mut pi_s, &mut pi_a, &h, &q, LAMBDA, ZETA, K_ELASTIC, &bnd);

    let diff_pi_s = max_abs_diff(&pi_s, &pi_s_ref);
    let diff_pi_a = max_abs_diff(&pi_a, &pi_a_ref);
    eprintln!("Step 3 calculate_Pi: max|Pi_S diff|={diff_pi_s:.3e}  max|Pi_A diff|={diff_pi_a:.3e}");

    // ── Step 4a: pressure RHS ───────────────────────────────────────────────
    let mut rhs = vec![0.0_f64; N];
    div_vector(&u, &mut rhs, &bnd);
    for v in rhs.iter_mut() { *v *= RHO / DT; }
    calculate_pressure_terms(&u, RHO, &pi_s, &mut rhs, &bnd);

    let diff_rhs = max_abs_diff(&rhs, &p_rhs_ref);
    let (_, rhs_idx) = max_abs_diff_at(&rhs, &p_rhs_ref);
    let rx = rhs_idx / LY; let ry = rhs_idx % LY;
    eprintln!("Step 4a p_rhs:       max|rhs diff|={diff_rhs:.3e}  (at [{rx},{ry}] rust={:.4e} py={:.4e})",
              rhs[rhs_idx], p_rhs_ref[rhs_idx]);

    // ── Step 4b: pressure relaxation (20 iters, with Neumann BC) ────────────
    let mut p = p_ic.clone();
    let mut p_aux = vec![0.0_f64; N];
    for _ in 0..MAX_P_ITERS {
        p_aux.copy_from_slice(&p);
        relax_pressure_inner_loop(&mut p, &p_aux, &rhs, &bnd);
        {
            let p_ptr = p.as_mut_ptr();
            let p_mut = unsafe { std::slice::from_raw_parts_mut(p_ptr, N) };
            apply_p_boundary_conditions(p_mut, &p_aux, &u, RHO, NU, &pi_s, &pi_a, &bnd);
        }
    }
    let diff_p_mid = max_abs_diff(&p, &p_mid_ref);
    let (_, pm_idx) = max_abs_diff_at(&p, &p_mid_ref);
    let pmx = pm_idx / LY; let pmy = pm_idx % LY;
    eprintln!("Step 4b p_mid:       max|p diff|={diff_p_mid:.3e}  (at [{pmx},{pmy}] rust={:.4e} py={:.4e})",
              p[pm_idx], p_mid_ref[pm_idx]);

    // ── Step 5: Q update ────────────────────────────────────────────────────
    let mut dq = vec![0.0_f64; N2];
    for i in 0..N2 { dq[i] = (1.0 / GAMMA) * h[i] + s[i]; }
    upwind_advective_term(&u, &q, &mut dq, &bnd, -1.0);
    eprintln!("Step 5 Q_update:     max|dQ|={:.3e}", dq.iter().cloned().fold(0.0_f64, f64::max));

    // ── Step 6: u update ────────────────────────────────────────────────────
    let mut dudt = vec![0.0_f64; N2];
    get_u_update(&mut dudt, &u, &p, RHO, &pi_s, &pi_a, NU, &bnd);
    eprintln!("Step 6 u_update:     max|dudt|={:.3e}", dudt.iter().cloned().fold(0.0_f64, f64::max));

    // ── Final: apply Q and u BC, compare against step_*_ref.txt ────────────
    let mut q_new = q.clone();
    let mut u_new = u.clone();
    for i in 0..N2 { q_new[i] += DT * dq[i]; }
    for i in 0..N2 { u_new[i] += DT * dudt[i]; }
    volterra_cgpo::bc::apply_q_boundary_conditions(&mut q_new, &bnd, std::f64::consts::SQRT_2);
    volterra_cgpo::bc::apply_u_boundary_conditions(&mut u_new, &bnd);

    let diff_q_final = max_abs_diff(&q_new, &q_final_ref);
    let diff_u_final = max_abs_diff(&u_new, &u_final_ref);
    let diff_p_final = max_abs_diff(&p, &p_final_ref);
    eprintln!("Final: max|Q diff|={diff_q_final:.3e}  max|u diff|={diff_u_final:.3e}  max|p diff|={diff_p_final:.3e}");

    // Report all diffs — don't assert hard limits here, just gather information
    eprintln!("Summary:\n  H: {diff_h:.3e}\n  S: {diff_s:.3e}\n  H_bc: {diff_h_bc:.3e}\n  Pi_S: {diff_pi_s:.3e}\n  Pi_A: {diff_pi_a:.3e}\n  p_rhs: {diff_rhs:.3e}\n  p_mid: {diff_p_mid:.3e}\n  Q_final: {diff_q_final:.3e}\n  u_final: {diff_u_final:.3e}\n  p_final: {diff_p_final:.3e}");

    // Only assert that computation completed without panic (NaN check)
    assert!(!q_new.iter().any(|x| x.is_nan()), "Q has NaN");
    assert!(!u_new.iter().any(|x| x.is_nan()), "u has NaN");
    assert!(!p.iter().any(|x| x.is_nan()), "p has NaN");
}

/// Dump Rust boundary normals to ref/ so Python can use the same values.
/// Writes bnd_outer_normals.txt and bnd_inner_normals.txt (flat x*LY+y order,
/// two values per cell: nx ny, each on its own line).
#[test]
fn dump_boundary_normals() {
    let ref_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ref");
    let bnd = nephroid_boundary(LX, LY);

    let mut outer_lines = Vec::with_capacity(LX * LY * 2);
    let mut inner_lines = Vec::with_capacity(LX * LY * 2);
    for x in 0..LX {
        for y in 0..LY {
            let idx = x * LY + y;
            outer_lines.push(format!("{:.18e}", bnd.outer_normals[idx][0]));
            outer_lines.push(format!("{:.18e}", bnd.outer_normals[idx][1]));
            inner_lines.push(format!("{:.18e}", bnd.inner_normals[idx][0]));
            inner_lines.push(format!("{:.18e}", bnd.inner_normals[idx][1]));
        }
    }
    std::fs::write(
        format!("{ref_dir}/bnd_outer_normals.txt"),
        outer_lines.join("\n") + "\n",
    ).expect("write bnd_outer_normals.txt");
    std::fs::write(
        format!("{ref_dir}/bnd_inner_normals.txt"),
        inner_lines.join("\n") + "\n",
    ).expect("write bnd_inner_normals.txt");
    eprintln!("Wrote boundary normal files to {ref_dir}/");
}
