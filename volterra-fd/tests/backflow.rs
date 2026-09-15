//! Passive backflow in the 2D wet solver.
//!
//! With `backflow` set, every flow term X of the Q equation has a conjugate
//! force F in the Stokes forcing, chosen so that the power the term takes from
//! the free energy, Σ h·X, equals the work the force does on the flow, Σ v·F.
//! The pairs are the aligning source with the symmetric stress -λSH/2,
//! co-rotation with the antisymmetric stress Q·H - H·Q, and advection with the
//! Ericksen body force -Σ h_α∇q_α. The first test checks each pair term by
//! term with central differences, on a smooth field and at defect cores, and
//! that the solver's velocity feels all three forces. Summed, the balance is
//! the passive energy law: a run's free energy can only fall.

use num_complex::Complex;
use rustfft::FftPlanner;
use volterra_core::{ActiveNematicParams, QField2D, VelocityField2D};
use volterra_fd::{
    beris_edwards_rhs, corotation_strain, molecular_field, run_active_nematic_hydro, stokes_solve,
    strain_alignment,
};

const N: usize = 64;

fn params(backflow: bool) -> ActiveNematicParams {
    let mut p = ActiveNematicParams::default_test();
    p.nx = N;
    p.ny = N;
    p.zeta_eff = 0.0;
    p.a_landau = -0.2;
    p.backflow = backflow;
    p
}

/// A smooth ordered field with a long-wavelength bend and splay.
fn bent() -> QField2D {
    let mut q = vec![[0.0; 2]; N * N];
    let tau = std::f64::consts::TAU;
    for i in 0..N {
        for j in 0..N {
            let x = i as f64 / N as f64;
            let y = j as f64 / N as f64;
            let theta = 0.5 * (tau * y).sin() + 0.3 * (tau * (x + 2.0 * y)).cos();
            let s = 0.12 + 0.03 * (tau * x).sin();
            q[i * N + j] = [s * (2.0 * theta).cos(), s * (2.0 * theta).sin()];
        }
    }
    QField2D {
        q,
        nx: N,
        ny: N,
        dx: 1.0,
    }
}

/// A +1/2 and -1/2 pair, whose cores give the steep gradients the Ericksen
/// stress acts on. The phase is periodic because the two windings cancel.
fn defect_pair() -> QField2D {
    let mut q = vec![[0.0; 2]; N * N];
    let (xa, xb, yc) = (
        N as f64 * 0.35 + 0.3,
        N as f64 * 0.65 + 0.3,
        N as f64 * 0.5 + 0.3,
    );
    for i in 0..N {
        for j in 0..N {
            let (x, y) = (i as f64, j as f64);
            let theta = 0.5 * (y - yc).atan2(x - xa) - 0.5 * (y - yc).atan2(x - xb);
            let r = ((x - xa).hypot(y - yc)).min((x - xb).hypot(y - yc));
            let s = 0.15 * (r / 2.0).tanh();
            q[i * N + j] = [s * (2.0 * theta).cos(), s * (2.0 * theta).sin()];
        }
    }
    QField2D {
        q,
        nx: N,
        ny: N,
        dx: 1.0,
    }
}

fn free_energy(q: &QField2D, p: &ActiveNematicParams) -> f64 {
    let a = p.a_eff();
    let mut f = 0.0;
    for i in 0..N {
        for j in 0..N {
            let k = i * N + j;
            let kx = ((i + 1) % N) * N + j;
            let ky = i * N + (j + 1) % N;
            let [q1, q2] = q.q[k];
            let s2 = q1 * q1 + q2 * q2;
            let grad: f64 = (0..2)
                .map(|al| (q.q[kx][al] - q.q[k][al]).powi(2) + (q.q[ky][al] - q.q[k][al]).powi(2))
                .sum();
            f += 0.5 * p.k_r * grad + a * s2 + p.c_landau * s2 * s2;
        }
    }
    f
}

fn viscous_dissipation(v: &VelocityField2D, eta: f64) -> f64 {
    let mut d = 0.0;
    for i in 0..N {
        for j in 0..N {
            let ip = v.idx_i(i as i64 + 1, j as i64);
            let im = v.idx_i(i as i64 - 1, j as i64);
            let jp = v.idx_i(i as i64, j as i64 + 1);
            let jm = v.idx_i(i as i64, j as i64 - 1);
            for al in 0..2 {
                let gx = (v.v[ip][al] - v.v[im][al]) / 2.0;
                let gy = (v.v[jp][al] - v.v[jm][al]) / 2.0;
                d += gx * gx + gy * gy;
            }
        }
    }
    eta * d
}

fn d(f: &[f64], i: usize, j: usize, axis: usize) -> f64 {
    let (ip, im) = if axis == 0 {
        (((i + 1) % N) * N + j, ((i + N - 1) % N) * N + j)
    } else {
        (i * N + (j + 1) % N, i * N + (j + N - 1) % N)
    };
    (f[ip] - f[im]) / 2.0
}

/// Per pair: (power of the Q-equation term, work of its conjugate force).
fn pairs(q: &QField2D, v: &VelocityField2D, p: &ActiveNematicParams) -> [(f64, f64); 3] {
    let h = molecular_field(q, p);
    let dot = |x: &QField2D| -> f64 {
        (0..N * N)
            .map(|k| h.q[k][0] * x.q[k][0] + h.q[k][1] * x.q[k][1])
            .sum()
    };
    let comp = |f: &QField2D, al: usize| -> Vec<f64> { f.q.iter().map(|x| x[al]).collect() };
    let (q1, q2) = (comp(q, 0), comp(q, 1));
    let vx: Vec<f64> = v.v.iter().map(|x| x[0]).collect();
    let vy: Vec<f64> = v.v.iter().map(|x| x[1]).collect();
    let mut s1 = vec![0.0; N * N];
    let mut s2 = vec![0.0; N * N];
    let mut a = vec![0.0; N * N];
    for k in 0..N * N {
        let [a1, a2] = q.q[k];
        let [h1, h2] = h.q[k];
        let lam_s = p.lambda * 2.0 * (a1 * a1 + a2 * a2).sqrt();
        s1[k] = -0.5 * lam_s * h1;
        s2[k] = -0.5 * lam_s * h2;
        a[k] = a1 * h2 - a2 * h1;
    }
    let (mut p_adv, mut w_s, mut w_a, mut w_e) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..N {
        for j in 0..N {
            let k = i * N + j;
            let (dq1x, dq1y, dq2x, dq2y) = (
                d(&q1, i, j, 0),
                d(&q1, i, j, 1),
                d(&q2, i, j, 0),
                d(&q2, i, j, 1),
            );
            let [h1, h2] = h.q[k];
            p_adv -= h1 * (vx[k] * dq1x + vy[k] * dq1y) + h2 * (vx[k] * dq2x + vy[k] * dq2y);
            w_s += vx[k] * (d(&s1, i, j, 0) + d(&s2, i, j, 1))
                + vy[k] * (d(&s2, i, j, 0) - d(&s1, i, j, 1));
            w_a += vx[k] * d(&a, i, j, 1) - vy[k] * d(&a, i, j, 0);
            w_e -= vx[k] * (h1 * dq1x + h2 * dq2x) + vy[k] * (h1 * dq1y + h2 * dq2y);
        }
    }
    [
        (dot(&strain_alignment(q, v, p)), w_s),
        (dot(&corotation_strain(q, v, p)), w_a),
        (p_adv, w_e),
    ]
}

#[test]
fn each_flow_term_takes_the_work_its_conjugate_force_does() {
    let p = params(true);
    for (name, q) in [("bent", bent()), ("defect pair", defect_pair())] {
        let v = stokes_solve(&q, &p);
        for ((power, work), term) in
            pairs(&q, &v, &p)
                .into_iter()
                .zip(["align", "corotation", "advection"])
        {
            let gap = (power - work).abs() / work.abs().max(1e-30);
            assert!(
                gap < 1e-2,
                "{name} {term}: power {power:e} against work {work:e}"
            );
        }
    }
}

fn fft2(field: &[f64]) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let mut buf: Vec<Complex<f64>> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();
    for row in buf.chunks_mut(N) {
        fft.process(row);
    }
    let mut col = vec![Complex::new(0.0, 0.0); N];
    for j in 0..N {
        for i in 0..N {
            col[i] = buf[i * N + j];
        }
        fft.process(&mut col);
        for i in 0..N {
            buf[i * N + j] = col[i];
        }
    }
    buf
}

fn wavenumber(i: usize) -> f64 {
    let s = if i <= N / 2 {
        i as f64
    } else {
        i as f64 - N as f64
    };
    std::f64::consts::TAU * s / N as f64
}

#[test]
fn the_solve_is_exact_stokes_for_the_whole_passive_force() {
    // For the spectral Stokes solve, Σ_k v̂*·F̂ = η Σ_k k² |v̂|² holds to
    // rounding when F is the force the solver used. The force here is built
    // independently, so a force the solver omits or gets the sign of wrong
    // breaks the identity however small that force is. The tolerance allows
    // for the Nyquist mode, which the solver's real-part inversion drops and
    // defect cores excite: that costs 6e-7 here, and the smallest force a
    // wrong solve could omit is several per cent of the work.
    let p = params(true);
    let i1 = Complex::new(0.0, 1.0);
    for (name, q) in [("bent", bent()), ("defect pair", defect_pair())] {
        let v = stokes_solve(&q, &p);
        let h = molecular_field(&q, &p);
        let mut s1 = vec![0.0; N * N];
        let mut s2 = vec![0.0; N * N];
        let mut a = vec![0.0; N * N];
        let mut fx = vec![0.0; N * N];
        let mut fy = vec![0.0; N * N];
        let q1: Vec<f64> = q.q.iter().map(|x| x[0]).collect();
        let q2: Vec<f64> = q.q.iter().map(|x| x[1]).collect();
        for i in 0..N {
            for j in 0..N {
                let k = i * N + j;
                let [a1, a2] = q.q[k];
                let [h1, h2] = h.q[k];
                let lam_s = p.lambda * 2.0 * (a1 * a1 + a2 * a2).sqrt();
                s1[k] = -0.5 * lam_s * h1;
                s2[k] = -0.5 * lam_s * h2;
                a[k] = a1 * h2 - a2 * h1;
                fx[k] = -(h1 * d(&q1, i, j, 0) + h2 * d(&q2, i, j, 0));
                fy[k] = -(h1 * d(&q1, i, j, 1) + h2 * d(&q2, i, j, 1));
            }
        }
        let (s1h, s2h, ah, fxh, fyh) = (fft2(&s1), fft2(&s2), fft2(&a), fft2(&fx), fft2(&fy));
        let vxh = fft2(&v.v.iter().map(|x| x[0]).collect::<Vec<_>>());
        let vyh = fft2(&v.v.iter().map(|x| x[1]).collect::<Vec<_>>());
        let (mut work, mut dissipation) = (0.0, 0.0);
        for i in 0..N {
            for j in 0..N {
                let k = i * N + j;
                let (kx, ky) = (wavenumber(i), wavenumber(j));
                let fxk = i1 * kx * s1h[k] + i1 * ky * s2h[k] + i1 * ky * ah[k] + fxh[k];
                let fyk = i1 * kx * s2h[k] - i1 * ky * s1h[k] - i1 * kx * ah[k] + fyh[k];
                work += (vxh[k].conj() * fxk + vyh[k].conj() * fyk).re;
                dissipation +=
                    p.eta * (kx * kx + ky * ky) * (vxh[k].norm_sqr() + vyh[k].norm_sqr());
            }
        }
        let gap = (work - dissipation).abs() / dissipation;
        assert!(
            dissipation > 0.0 && gap < 1e-5,
            "{name}: work {work:e} against dissipation {dissipation:e}"
        );
    }
}

#[test]
fn the_wet_rhs_is_advection_corotation_and_the_aligning_source() {
    let p = params(true);
    let q = defect_pair();
    let v = stokes_solve(&q, &p);
    let wet = beris_edwards_rhs(&q, Some(&v), &p);
    let dry = beris_edwards_rhs(&q, None, &p);
    let adv = v.advect(&q);
    let cor = corotation_strain(&q, &v, &p);
    let ali = strain_alignment(&q, &v, &p);
    for k in 0..N * N {
        for al in 0..2 {
            let expect = -adv.q[k][al] + cor.q[k][al] + ali.q[k][al];
            assert!(
                (wet.q[k][al] - dry.q[k][al] - expect).abs() < 1e-12,
                "vertex {k}"
            );
        }
    }
}

#[test]
fn the_velocity_feels_every_passive_force() {
    // Stokes gives Σ v·F = η Σ |∇v|² exactly for the spectral operator; with
    // central differences on a smooth field the two agree closely, and a force
    // left out of the solve breaks that agreement.
    let p = params(true);
    let q = bent();
    let v = stokes_solve(&q, &p);
    let work: f64 = pairs(&q, &v, &p).iter().map(|(_, w)| w).sum();
    let dissipation = viscous_dissipation(&v, p.eta);
    let gap = (work - dissipation).abs() / dissipation;
    assert!(
        dissipation > 0.0 && gap < 0.03,
        "work {work:e} against dissipation {dissipation:e}"
    );
}

#[test]
fn without_backflow_a_passive_field_drives_no_flow() {
    let p = params(false);
    let v = stokes_solve(&bent(), &p);
    let max =
        v.v.iter()
            .flat_map(|a| a.iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(max < 1e-12, "flow {max} at zero activity without backflow");
}

#[test]
fn a_passive_run_with_backflow_only_loses_free_energy() {
    let mut p = params(true);
    p.dt = 0.005;
    let mut q = bent();
    let mut last = free_energy(&q, &p);
    for _ in 0..40 {
        let (next, _) = run_active_nematic_hydro(&q, &p, 25, 25);
        let f = free_energy(&next, &p);
        assert!(
            f <= last + 1e-9 * last.abs().max(1.0),
            "free energy rose from {last} to {f}"
        );
        last = f;
        q = next;
    }
}

#[test]
fn backflow_off_reproduces_the_default_run_exactly() {
    let mut p = ActiveNematicParams::default_test();
    p.nx = 32;
    p.ny = 32;
    let q = QField2D::random_perturbation(32, 32, 1.0, 0.05, 3);
    let (a, _) = run_active_nematic_hydro(&q, &p, 50, 50);
    p.backflow = false;
    let (b, _) = run_active_nematic_hydro(&q, &p, 50, 50);
    assert_eq!(a.q, b.q);
}

const M: usize = 96;

fn annihilating_pair() -> QField2D {
    let mut q = vec![[0.0; 2]; M * M];
    let (xa, xb, yc) = (40.3, 58.3, M as f64 * 0.5 + 0.3);
    for i in 0..M {
        for j in 0..M {
            let (x, y) = (i as f64, j as f64);
            let theta = 0.5 * (y - yc).atan2(x - xa) - 0.5 * (y - yc).atan2(x - xb);
            let r = ((x - xa).hypot(y - yc)).min((x - xb).hypot(y - yc));
            let s = 0.2 * (r / 2.0).tanh();
            q[i * M + j] = [s * (2.0 * theta).cos(), s * (2.0 * theta).sin()];
        }
    }
    QField2D {
        q,
        nx: M,
        ny: M,
        dx: 1.0,
    }
}

/// Mean x of the plaquettes winding +1/2 and -1/2, if both are present.
fn core_x(f: &QField2D) -> Option<(f64, f64)> {
    let ang = |i: usize, j: usize| {
        let [a, b] = f.q[i * M + j];
        b.atan2(a)
    };
    let w = |d: f64| {
        (d + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
    };
    let (mut sp, mut np, mut sm, mut nm) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            let c = (w(ang(i + 1, j) - ang(i, j))
                + w(ang(i + 1, j + 1) - ang(i + 1, j))
                + w(ang(i, j + 1) - ang(i + 1, j + 1))
                + w(ang(i, j) - ang(i, j + 1)))
                / (4.0 * std::f64::consts::PI);
            if c > 0.2 {
                sp += i as f64 + 0.5;
                np += 1.0;
            } else if c < -0.2 {
                sm += i as f64 + 0.5;
                nm += 1.0;
            }
        }
    }
    (np > 0.0 && nm > 0.0).then(|| (sp / np, sm / nm))
}

/// Distance each core travels before the pair annihilates, (+1/2, -1/2).
fn travel(backflow: bool) -> (f64, f64) {
    let mut p = ActiveNematicParams::default_test();
    p.nx = M;
    p.ny = M;
    p.zeta_eff = 0.0;
    p.a_landau = -0.3;
    p.k_r = 3.0;
    p.eta = 0.03;
    p.dt = 0.004;
    p.backflow = backflow;
    let mut q = annihilating_pair();
    let start = core_x(&q).expect("pair present at start");
    let mut last = start;
    for _ in 0..200 {
        let (next, _) = run_active_nematic_hydro(&q, &p, 100, 100);
        q = next;
        match core_x(&q) {
            Some(x) => last = x,
            None => return (last.0 - start.0, start.1 - last.1),
        }
    }
    panic!("the pair did not annihilate");
}

#[test]
fn backflow_speeds_the_plus_half_core_over_the_minus_half_in_annihilation() {
    // Toth, Denniston and Yeomans (2002): with backflow the +1/2 defect of an
    // annihilating pair moves faster than the -1/2. Without flow the two are
    // symmetric to the one-grid-unit resolution of this tracker.
    let (plus, minus) = travel(false);
    assert!(
        (plus - minus).abs() <= 1.0,
        "no backflow: +1/2 {plus}, -1/2 {minus}"
    );
    let (plus, minus) = travel(true);
    assert!(plus - minus >= 1.5, "backflow: +1/2 {plus}, -1/2 {minus}");
}
