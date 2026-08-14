//! CPU-side check of `volterra-cuda`'s `force_fused_aos`/`force_fused_soa`
//! kernel arithmetic (`volterra-cuda/src/kernels.rs`) against the trusted
//! CPU reference, `beris_edwards_rhs_3d_par_dry`.
//!
//! Lives in `volterra-solver`, not `volterra-cuda`, on purpose: a `#[kernel]`
//! function's body is a device-only stub (see `cartan-cuda`'s own README),
//! so this cannot call the GPU kernel directly either way, and gains nothing
//! from building against the CUDA toolchain's pinned nightly. It
//! reimplements the same formula in ordinary Rust -- transcribed, not
//! called -- and checks it against the already-validated CPU path. This is
//! the correctness half of "fuse the two passes" that does not need the
//! device; the GPU kernel's fidelity to this same formula is checked against
//! CPU FIRE once the device is free.
//!
//! Index-based throughout, deliberately: this is a line-for-line transcript
//! of the kernels' own indexing (see `volterra-cuda/src/kernels.rs`), and
//! rewriting it as iterators would trade the direct visual correspondence
//! for style.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use volterra_core::ActiveNematicParams3D;
use volterra_fields::QField3D;
use volterra_solver::beris_edwards_rhs_3d_par_dry;

/// Exactly the arithmetic `force_fused_aos` performs, transcribed from
/// `kernels.rs` (same order of operations, so agreement should be exact to
/// several ULPs, not merely close).
fn force_fused_aos_reference(
    q: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    a_eff: f64,
    b_landau: f64,
    c_landau: f64,
    k_r: f64,
    gamma_r: f64,
    inv_dx2: f64,
) -> Vec<[f64; 5]> {
    let n_sites = nx * ny * nz;
    let nynz = ny * nz;
    let mut out = vec![[0.0_f64; 5]; n_sites];

    for s in 0..n_sites {
        let l = s % nz;
        let ij = s / nz;
        let jc = ij % ny;
        let i = ij / ny;

        let ip = ((i + 1) % nx) * nynz + jc * nz + l;
        let im = ((i + nx - 1) % nx) * nynz + jc * nz + l;
        let jp = i * nynz + ((jc + 1) % ny) * nz + l;
        let jm = i * nynz + ((jc + ny - 1) % ny) * nz + l;
        let lp = i * nynz + jc * nz + (l + 1) % nz;
        let lm = i * nynz + jc * nz + (l + nz - 1) % nz;

        let b = s * 5;
        let q11 = q[b];
        let q12 = q[b + 1];
        let q13 = q[b + 2];
        let q22 = q[b + 3];
        let q23 = q[b + 4];
        let q33 = -(q11 + q22);
        let trq2 = q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        let bulk = -a_eff - 2.0 * c_landau * trq2;

        let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
        let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
        let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
        let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
        let qq23 = q12 * q13 + q22 * q23 + q23 * q33;
        let h_cubic = [
            -3.0 * b_landau * qq11 + b_landau * trq2,
            -3.0 * b_landau * qq12,
            -3.0 * b_landau * qq13,
            -3.0 * b_landau * qq22 + b_landau * trq2,
            -3.0 * b_landau * qq23,
        ];

        for c in 0..5 {
            let lap = (q[ip * 5 + c]
                + q[im * 5 + c]
                + q[jp * 5 + c]
                + q[jm * 5 + c]
                + q[lp * 5 + c]
                + q[lm * 5 + c]
                - 6.0 * q[b + c])
                * inv_dx2;
            out[s][c] = gamma_r * (k_r * lap + bulk * q[b + c] + h_cubic[c]);
        }
    }
    out
}

fn analytic_s0(p: &ActiveNematicParams3D) -> f64 {
    (-3.0 * p.a_eff() / (4.0 * p.c_landau)).sqrt()
}

#[test]
fn fused_aos_formula_matches_cpu_reference() {
    let n = 6usize;
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    // Nonzero, exercising the cubic bulk term the fused kernel must also
    // carry; the initial field below is still built from the b=0 magnitude
    // (an arbitrary off-equilibrium state for a formula check, not an
    // equilibrium claim).
    p.b_landau = -1.5;
    // chi_a = 0 in default_test, so the magnetic torque term the reference
    // also carries is exactly zero and the fused kernel's simpler formula
    // (no magnetic term at all, matching the existing split force/trq2
    // kernels) is a fair comparison.
    assert_eq!(p.chi_a, 0.0, "test assumes no magnetic torque term");

    let s0 = analytic_s0(&p);
    let q = QField3D::random_director_field(n, n, n, p.dx, s0, 11);

    let reference = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);

    let q_flat: Vec<f64> = q.q.iter().flat_map(|c| c.iter().copied()).collect();
    let inv_dx2 = 1.0 / (p.dx * p.dx);
    let fused = force_fused_aos_reference(
        &q_flat,
        n,
        n,
        n,
        p.a_eff(),
        p.b_landau,
        p.c_landau,
        p.k_r,
        p.gamma_r,
        inv_dx2,
    );

    let mut max_diff = 0.0_f64;
    for s in 0..n * n * n {
        for c in 0..5 {
            let diff = (fused[s][c] - reference.q[s][c]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff < 1e-12,
        "fused-AoS formula disagrees with beris_edwards_rhs_3d_par_dry by {max_diff:e}"
    );
}

/// The SoA formula is the identical arithmetic over 5 separate planes; check
/// it against the same reference, and against the AoS transcription above
/// (same numbers, different memory layout, must agree exactly).
#[test]
fn fused_soa_formula_matches_cpu_reference_and_aos() {
    let n = 6usize;
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.zeta_eff = 0.0;
    p.noise_amp = 0.0;
    p.b_landau = -1.5;

    let s0 = analytic_s0(&p);
    let q = QField3D::random_director_field(n, n, n, p.dx, s0, 11);
    let n_sites = n * n * n;

    // Deinterleave into 5 planes, exactly what Device::force_soa's host-side
    // conversion will do.
    let mut planes: [Vec<f64>; 5] = Default::default();
    for plane in &mut planes {
        *plane = vec![0.0; n_sites];
    }
    for s in 0..n_sites {
        for c in 0..5 {
            planes[c][s] = q.q[s][c];
        }
    }

    let inv_dx2 = 1.0 / (p.dx * p.dx);
    let nynz = n * n;
    let a_eff = p.a_eff();
    let b_landau = p.b_landau;
    let c_landau = p.c_landau;
    let k_r = p.k_r;
    let gamma_r = p.gamma_r;

    let mut soa_out: [Vec<f64>; 5] = Default::default();
    for plane in &mut soa_out {
        *plane = vec![0.0; n_sites];
    }

    for s in 0..n_sites {
        let l = s % n;
        let ij = s / n;
        let jc = ij % n;
        let i = ij / n;

        let ip = ((i + 1) % n) * nynz + jc * n + l;
        let im = ((i + n - 1) % n) * nynz + jc * n + l;
        let jp = i * nynz + ((jc + 1) % n) * n + l;
        let jm = i * nynz + ((jc + n - 1) % n) * n + l;
        let lp = i * nynz + jc * n + (l + 1) % n;
        let lm = i * nynz + jc * n + (l + n - 1) % n;

        let q11 = planes[0][s];
        let q12 = planes[1][s];
        let q13 = planes[2][s];
        let q22 = planes[3][s];
        let q23 = planes[4][s];
        let q33 = -(q11 + q22);
        let trq2 = q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
        let bulk = -a_eff - 2.0 * c_landau * trq2;

        let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
        let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
        let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
        let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
        let qq23 = q12 * q13 + q22 * q23 + q23 * q33;
        let h_cubic = [
            -3.0 * b_landau * qq11 + b_landau * trq2,
            -3.0 * b_landau * qq12,
            -3.0 * b_landau * qq13,
            -3.0 * b_landau * qq22 + b_landau * trq2,
            -3.0 * b_landau * qq23,
        ];

        let centres = [q11, q12, q13, q22, q23];
        for c in 0..5 {
            let plane = &planes[c];
            let lap = (plane[ip] + plane[im] + plane[jp] + plane[jm] + plane[lp] + plane[lm]
                - 6.0 * centres[c])
                * inv_dx2;
            soa_out[c][s] = gamma_r * (k_r * lap + bulk * centres[c] + h_cubic[c]);
        }
    }

    let reference = beris_edwards_rhs_3d_par_dry(&q, &p, 0.0);

    let mut max_diff_ref = 0.0_f64;
    for s in 0..n_sites {
        for c in 0..5 {
            max_diff_ref = max_diff_ref.max((soa_out[c][s] - reference.q[s][c]).abs());
        }
    }
    assert!(
        max_diff_ref < 1e-12,
        "fused-SoA formula disagrees with beris_edwards_rhs_3d_par_dry by {max_diff_ref:e}"
    );

    let q_flat: Vec<f64> = q.q.iter().flat_map(|c| c.iter().copied()).collect();
    let aos = force_fused_aos_reference(
        &q_flat, n, n, n, a_eff, b_landau, c_landau, k_r, gamma_r, inv_dx2,
    );
    let mut max_diff_aos = 0.0_f64;
    for s in 0..n_sites {
        for c in 0..5 {
            max_diff_aos = max_diff_aos.max((soa_out[c][s] - aos[s][c]).abs());
        }
    }
    assert!(
        max_diff_aos < 1e-14,
        "SoA and AoS formulas disagree by {max_diff_aos:e} (should be bitwise identical)"
    );
}
