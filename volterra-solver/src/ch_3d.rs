// ~/volterra/volterra-solver/src/ch_3d.rs

//! ETD (exponential time differencing) integrator for the Cahn-Hilliard equation
//! on a 3D periodic grid.
//!
//! ## Physics
//!
//! The Cahn-Hilliard equation governs lipid concentration φ_l:
//!
//! ```text
//! ∂_t φ = M_l ∇²μ
//!
//! μ = -κ_CH ∇²φ  +  a_CH φ  +  b_CH φ³  -  χ_MS Tr(Q_lip²)
//! ```
//!
//! which gives:
//!
//! ```text
//! ∂_t φ = M_l (-κ_CH ∇⁴φ  +  ∇²[a_CH φ + b_CH φ³ - χ_MS Tr(Q_lip²)])
//! ```
//!
//! ## ETD scheme in Fourier space
//!
//! Splitting into stiff linear (L) and nonlinear (N) parts:
//!
//! - L = -M_l κ_CH k⁴  (stiff; large for high wavenumbers)
//! - N[φ] = M_l (a_CH φ + b_CH φ³ - χ_MS Tr(Q_lip²))  (computed in real space)
//!
//! Note: N is computed in real space (not under ∇²), so its Fourier transform
//! is multiplied by -k² in the full equation. However, the ETD1 scheme used
//! here treats the full right-hand side split as:
//!
//! ```text
//! ∂_t φ̂ = L φ̂ + N̂
//! ```
//!
//! where L = -M_l κ_CH k⁴ and N̂ = -k² × FFT[M_l(a_CH φ + b_CH φ³ - χ_MS Tr(Q²))]
//!
//! The ETD1 exact update (Cox-Matthews):
//! ```text
//! φ̂_new = φ̂ e^{L dt}  +  N̂ (e^{L dt} - 1) / L     (L ≠ 0)
//! φ̂_new = φ̂ + dt N̂                                  (k = 0, DC mode)
//! ```

use rustfft::{FftPlanner, num_complex::Complex};
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, ScalarField3D};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the Cahn-Hilliard concentration field φ by one ETD step.
///
/// Uses ETD1 (exponential time differencing, first-order) to handle the stiff
/// biharmonic operator exactly while treating the nonlinear part explicitly.
///
/// ## Algorithm
///
/// 1. Compute the nonlinear chemical potential driving term at each vertex in
///    real space:
///    ```text
///    g[v] = M_l (a_CH φ[v]  +  b_CH φ[v]³  -  χ_MS Tr(Q_lip[v]²))
///    ```
/// 2. Forward 3D FFT both φ and g.
/// 3. For each (kx, ky, kz):
///    - k² = kx² + ky² + kz²,  k⁴ = (k²)²
///    - L = -M_l κ_CH k⁴  (stiff linear symbol)
///    - N̂ = -k² ĝ  (Laplacian of g in Fourier space)
///    - DC mode (k⁴ < 1e-14): φ̂_new = φ̂ + dt N̂  (Euler; L = 0)
///    - Otherwise: eL = exp(L dt); φ̂_new = φ̂ eL + N̂ (eL - 1) / L
/// 4. Inverse 3D FFT and normalise by 1/N.
///
/// # Arguments
///
/// * `phi`   - Concentration field φ on the nx×ny×nz periodic grid.
/// * `q_lip` - Lipid Q-tensor field on the same grid.
/// * `p`     - MARS parameters supplying `m_l`, `kappa_ch`, `a_ch`, `b_ch`,
///             `chi_ms`, and grid geometry.
/// * `dt`    - Time step.
///
/// # Returns
///
/// The updated concentration field φ at time t + dt.
pub fn ch_step_etd_3d(
    phi: &ScalarField3D,
    q_lip: &QField3D,
    p: &MarsParams3D,
    dt: f64,
) -> ScalarField3D {
    let nx = phi.nx;
    let ny = phi.ny;
    let nz = phi.nz;
    let n = nx * ny * nz;
    let dx = phi.dx;

    // ── Step 1: nonlinear term g[v] = M_l (a φ + b φ³ - χ Tr(Q²)) ─────────
    //
    // Tr(Q²) = (Q·Q).trace() for the 3×3 symmetric traceless Q tensor.
    // With Q = [[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]]
    // and q33 = -(q11+q22):
    //   Tr(Q²) = q11² + q22² + q33² + 2(q12² + q13² + q23²)
    let mut g = vec![0.0f64; n];

    for k in 0..n {
        let phi_v = phi.phi[k];
        let [q11, q12, q13, q22, q23] = q_lip.q[k];
        let q33 = -(q11 + q22);
        let tr_q2 = q11 * q11
            + q22 * q22
            + q33 * q33
            + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);

        g[k] = p.m_l * (p.a_ch * phi_v + p.b_ch * phi_v * phi_v * phi_v - p.chi_ms * tr_q2);
    }

    // ── Step 2: forward 3D FFT of φ and g ────────────────────────────────────
    let mut planner = FftPlanner::<f64>::new();

    let mut phi_hat: Vec<Complex<f64>> = phi.phi.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut g_hat: Vec<Complex<f64>> = g.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);

    // Helper: apply 1D FFT along z (innermost), y, then x (same strides as stokes_3d).
    for data in [&mut phi_hat, &mut g_hat] {
        // Along z (innermost/contiguous axis)
        for i in 0..nx {
            for j in 0..ny {
                let mut row: Vec<Complex<f64>> =
                    (0..nz).map(|l| data[phi.idx(i, j, l)]).collect();
                fft_z.process(&mut row);
                for l in 0..nz {
                    data[phi.idx(i, j, l)] = row[l];
                }
            }
        }
        // Along y
        for i in 0..nx {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..ny).map(|j| data[phi.idx(i, j, l)]).collect();
                fft_y.process(&mut row);
                for j in 0..ny {
                    data[phi.idx(i, j, l)] = row[j];
                }
            }
        }
        // Along x (outermost)
        for j in 0..ny {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..nx).map(|i| data[phi.idx(i, j, l)]).collect();
                fft_x.process(&mut row);
                for i in 0..nx {
                    data[phi.idx(i, j, l)] = row[i];
                }
            }
        }
    }

    // ── Step 3: ETD update in Fourier space ──────────────────────────────────
    let mut phi_hat_new: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = phi.idx(i, j, l);

                let kx = wavenumber(i, nx, dx);
                let ky = wavenumber(j, ny, dx);
                let kz = wavenumber(l, nz, dx);
                let k2 = kx * kx + ky * ky + kz * kz;
                let k4 = k2 * k2;

                // L = -M_l κ_CH k⁴   (stiff linear symbol)
                let big_l = -p.m_l * p.kappa_ch * k4;

                // N̂ = -k² ĝ   (Laplacian of g in Fourier space)
                // Multiply g_hat by -k² (real scalar) to get Laplacian
                let n_hat = g_hat[k] * Complex::new(-k2, 0.0);

                if k4 < 1e-14 {
                    // DC mode: L = 0, use simple Euler
                    phi_hat_new[k] = phi_hat[k] + n_hat * Complex::new(dt, 0.0);
                } else {
                    // ETD1: φ̂_new = φ̂ e^{L dt} + N̂ (e^{L dt} - 1) / L
                    let e_l = (big_l * dt).exp();
                    phi_hat_new[k] = phi_hat[k] * Complex::new(e_l, 0.0)
                        + n_hat * Complex::new((e_l - 1.0) / big_l, 0.0);
                }
            }
        }
    }

    // ── Step 4: inverse 3D FFT ────────────────────────────────────────────────
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    // Along z
    for i in 0..nx {
        for j in 0..ny {
            let mut row: Vec<Complex<f64>> =
                (0..nz).map(|l| phi_hat_new[phi.idx(i, j, l)]).collect();
            ifft_z.process(&mut row);
            for l in 0..nz {
                phi_hat_new[phi.idx(i, j, l)] = row[l];
            }
        }
    }
    // Along y
    for i in 0..nx {
        for l in 0..nz {
            let mut row: Vec<Complex<f64>> =
                (0..ny).map(|j| phi_hat_new[phi.idx(i, j, l)]).collect();
            ifft_y.process(&mut row);
            for j in 0..ny {
                phi_hat_new[phi.idx(i, j, l)] = row[j];
            }
        }
    }
    // Along x
    for j in 0..ny {
        for l in 0..nz {
            let mut row: Vec<Complex<f64>> =
                (0..nx).map(|i| phi_hat_new[phi.idx(i, j, l)]).collect();
            ifft_x.process(&mut row);
            for i in 0..nx {
                phi_hat_new[phi.idx(i, j, l)] = row[i];
            }
        }
    }

    // Normalise by 1/N (rustfft IFFT is unnormalised)
    let norm = 1.0 / (n as f64);
    let mut phi_new = ScalarField3D::zeros(nx, ny, nz, dx);
    for k in 0..n {
        phi_new.phi[k] = phi_hat_new[k].re * norm;
    }

    phi_new
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Finite-difference-consistent effective wavenumber for index `idx` on an
/// `n`-point grid with spacing `dx`.
///
/// The central-difference stencil `(f[i+1] - f[i-1]) / (2 dx)` applied to
/// a Fourier mode `e^{i k x}` gives `i sin(k dx) / dx · e^{i k x}`.
/// So the effective real wavenumber is `sin(k_phys dx) / dx`.
#[inline]
fn wavenumber(idx: usize, n: usize, dx: f64) -> f64 {
    let m = if idx <= n / 2 {
        idx as f64
    } else {
        idx as f64 - n as f64
    };
    let theta = 2.0 * std::f64::consts::PI * m / n as f64;
    theta.sin() / dx
}

/// Advance the enriched Cahn-Hilliard concentration field φ by one ETD step.
///
/// Implements the Approach B free energy:
///
/// ```text
/// f = f_CH  +  (κ_W/2ε³)(W'(φ) − ε²∇²φ + c₀ε²|∇φ|)²  +  κ̄_G · G_TLL(φ)
/// ```
///
/// ## Modifications vs. `ch_step_etd_3d`
///
/// 1. **Stiff linear symbol**: L = −M_l κ_eff k⁴  where  κ_eff = κ_CH − κ_W c₀/ε.
///
/// 2. **κ̄_G term in N** (explicit; never added to L):
///    g[v] += M_l · κ̄_G · K_G(v) · |∇φ(v)| / ε
///
/// Falls back to plain ETD when `p.kappa_w = 0 && p.kappa_bar_g = 0`.
pub fn ch_step_etd_enriched_3d(
    phi: &ScalarField3D,
    q_lip: &QField3D,
    p: &MarsParams3D,
    dt: f64,
) -> ScalarField3D {
    if p.kappa_w == 0.0 && p.kappa_bar_g == 0.0 {
        return ch_step_etd_3d(phi, q_lip, p, dt);
    }

    let nx = phi.nx;
    let ny = phi.ny;
    let nz = phi.nz;
    let n  = nx * ny * nz;
    let dx = phi.dx;
    let eps = p.epsilon_ch;

    let kappa_eff = p.kappa_eff();

    let (kg_field, grad_field) = if p.kappa_bar_g != 0.0 {
        crate::gauss_bonnet_3d::compute_kg_field(phi)
    } else {
        (vec![0.0f64; n], vec![0.0f64; n])
    };

    let mut g = vec![0.0f64; n];
    for idx in 0..n {
        let phi_v = phi.phi[idx];
        let [q11, q12, q13, q22, q23] = q_lip.q[idx];
        let q33 = -(q11 + q22);
        let tr_q2 = q11*q11 + q22*q22 + q33*q33 + 2.0*(q12*q12 + q13*q13 + q23*q23);

        let g_ch = p.m_l * (p.a_ch * phi_v + p.b_ch * phi_v.powi(3) - p.chi_ms * tr_q2);
        let g_kg = if p.kappa_bar_g != 0.0 && eps > 0.0 {
            p.m_l * p.kappa_bar_g * kg_field[idx] * grad_field[idx] / eps
        } else {
            0.0
        };
        g[idx] = g_ch + g_kg;
    }

    use rustfft::{FftPlanner, num_complex::Complex};
    let mut planner = FftPlanner::<f64>::new();

    let mut phi_hat: Vec<Complex<f64>> = phi.phi.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut g_hat:   Vec<Complex<f64>> = g.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);

    for data in [&mut phi_hat, &mut g_hat] {
        for i in 0..nx { for j in 0..ny {
            let mut row: Vec<Complex<f64>> = (0..nz).map(|l| data[phi.idx(i,j,l)]).collect();
            fft_z.process(&mut row);
            for l in 0..nz { data[phi.idx(i,j,l)] = row[l]; }
        }}
        for i in 0..nx { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..ny).map(|j| data[phi.idx(i,j,l)]).collect();
            fft_y.process(&mut row);
            for j in 0..ny { data[phi.idx(i,j,l)] = row[j]; }
        }}
        for j in 0..ny { for l in 0..nz {
            let mut row: Vec<Complex<f64>> = (0..nx).map(|i| data[phi.idx(i,j,l)]).collect();
            fft_x.process(&mut row);
            for i in 0..nx { data[phi.idx(i,j,l)] = row[i]; }
        }}
    }

    let mut phi_hat_new: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for i in 0..nx { for j in 0..ny { for l in 0..nz {
        let idx = phi.idx(i, j, l);
        let kx = wavenumber(i, nx, dx);
        let ky = wavenumber(j, ny, dx);
        let kz = wavenumber(l, nz, dx);
        let k2 = kx*kx + ky*ky + kz*kz;
        let k4 = k2*k2;

        let big_l = -p.m_l * kappa_eff * k4;
        let n_hat  = g_hat[idx] * Complex::new(-k2, 0.0);

        if k4 < 1e-14 {
            phi_hat_new[idx] = phi_hat[idx] + n_hat * Complex::new(dt, 0.0);
        } else {
            let e_l = (big_l * dt).exp();
            phi_hat_new[idx] = phi_hat[idx] * Complex::new(e_l, 0.0)
                + n_hat * Complex::new((e_l - 1.0) / big_l, 0.0);
        }
    }}}

    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    for i in 0..nx { for j in 0..ny {
        let mut row: Vec<Complex<f64>> = (0..nz).map(|l| phi_hat_new[phi.idx(i,j,l)]).collect();
        ifft_z.process(&mut row);
        for l in 0..nz { phi_hat_new[phi.idx(i,j,l)] = row[l]; }
    }}
    for i in 0..nx { for l in 0..nz {
        let mut row: Vec<Complex<f64>> = (0..ny).map(|j| phi_hat_new[phi.idx(i,j,l)]).collect();
        ifft_y.process(&mut row);
        for j in 0..ny { phi_hat_new[phi.idx(i,j,l)] = row[j]; }
    }}
    for j in 0..ny { for l in 0..nz {
        let mut row: Vec<Complex<f64>> = (0..nx).map(|i| phi_hat_new[phi.idx(i,j,l)]).collect();
        ifft_x.process(&mut row);
        for i in 0..nx { phi_hat_new[phi.idx(i,j,l)] = row[i]; }
    }}

    let norm = 1.0 / (n as f64);
    let mut phi_new = ScalarField3D::zeros(nx, ny, nz, dx);
    for idx in 0..n {
        phi_new.phi[idx] = phi_hat_new[idx].re * norm;
    }
    phi_new
}

/// Stability bound on dt for the enriched CH stepper when |κ̄_G| > 0.
///
/// Returns the maximum safe timestep:
///   dt_max = 0.1 · ε³ / (M_l |κ̄_G| (π/dx)⁶)
///
/// Returns f64::INFINITY when p.kappa_bar_g = 0.
pub fn enriched_ch_dt_bound(p: &MarsParams3D) -> f64 {
    if p.kappa_bar_g == 0.0 || p.m_l == 0.0 {
        return f64::INFINITY;
    }
    let k_max = std::f64::consts::PI / p.dx;
    0.1 * p.epsilon_ch.powi(3) / (p.m_l * p.kappa_bar_g.abs() * k_max.powi(6))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::MarsParams3D;
    use volterra_fields::{QField3D, ScalarField3D};

    /// Enriched stepper with kappa_w=0 and kappa_bar_g=0 must exactly reproduce
    /// the plain ch_step_etd_3d result (no new physics).
    #[test]
    fn test_enriched_ch_matches_plain_when_no_curvature() {
        let mut p = MarsParams3D::default_test();
        p.kappa_w = 0.0;
        p.kappa_bar_g = 0.0;
        p.epsilon_ch = 1.0;
        let phi = ScalarField3D::uniform(8, 8, 8, 1.0, 0.3);
        let q_lip = QField3D::zeros(8, 8, 8, 1.0);
        let phi_plain    = ch_step_etd_3d(&phi, &q_lip, &p, p.dt);
        let phi_enriched = ch_step_etd_enriched_3d(&phi, &q_lip, &p, p.dt);
        let max_diff: f64 = phi_plain.phi.iter().zip(phi_enriched.phi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "enriched with zero curvature params must equal plain CH, max_diff={}",
            max_diff
        );
    }

    /// Enriched stepper must conserve mass (spatial mean) for any curvature params.
    #[test]
    fn test_enriched_ch_conserves_mass() {
        let mut p = MarsParams3D::default_test();
        p.a_ch = 0.0;
        p.b_ch = 0.0;
        p.chi_ms = 0.0;
        p.kappa_w = 0.5;
        p.kappa_bar_g = -0.1;
        p.epsilon_ch = 1.5;
        let n = 8usize;
        let cx = n as f64 / 2.0;
        let mut phi = ScalarField3D::zeros(n, n, n, 1.0);
        for i in 0..n { for j in 0..n { for k in 0..n {
            let r = ((i as f64 - cx).powi(2) + (j as f64 - cx).powi(2)
                   + (k as f64 - cx).powi(2)).sqrt();
            let vi = phi.idx(i,j,k);
            phi.phi[vi] = 0.5*(1.0 + ((r - 3.0)/1.5).tanh());
        }}}
        let q_lip = QField3D::zeros(n, n, n, 1.0);
        let phi_new = ch_step_etd_enriched_3d(&phi, &q_lip, &p, p.dt);
        let delta = (phi_new.mean() - phi.mean()).abs();
        assert!(delta < 1e-10, "enriched CH must conserve mass, delta={}", delta);
    }

    /// Enriched stepper with nonzero kappa_bar_g and an interface must differ
    /// from the plain stepper.
    #[test]
    fn test_enriched_ch_differs_with_nonzero_kappa_bar_g() {
        let mut p = MarsParams3D::default_test();
        p.a_ch = 0.0;
        p.b_ch = 0.0;
        p.chi_ms = 0.0;
        p.kappa_w = 0.0;
        p.c0_sp = 0.0;
        p.kappa_bar_g = 0.5;
        p.epsilon_ch = 1.5;
        p.dt = 1e-4;
        let n = 8usize;
        let cx = n as f64 / 2.0;
        let mut phi = ScalarField3D::zeros(n, n, n, 1.0);
        for i in 0..n { for j in 0..n { for k in 0..n {
            let r = ((i as f64 - cx).powi(2) + (j as f64 - cx).powi(2)
                   + (k as f64 - cx).powi(2)).sqrt();
            let vi = phi.idx(i,j,k);
            phi.phi[vi] = 0.5*(1.0 + ((r - 3.0)/1.5).tanh());
        }}}
        let q_lip = QField3D::zeros(n, n, n, 1.0);
        let phi_plain    = ch_step_etd_3d(&phi, &q_lip, &p, p.dt);
        let phi_enriched = ch_step_etd_enriched_3d(&phi, &q_lip, &p, p.dt);
        let max_diff: f64 = phi_plain.phi.iter().zip(phi_enriched.phi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-14,
            "enriched with nonzero kappa_bar_g + interface must differ from plain CH"
        );
    }

    /// With nonzero c0_sp, κ_eff differs from κ_CH.
    #[test]
    fn test_enriched_ch_differs_with_nonzero_c0() {
        let mut p = MarsParams3D::default_test();
        p.a_ch = 0.0;
        p.b_ch = 0.0;
        p.chi_ms = 0.0;
        p.kappa_w = 1.0;
        p.c0_sp = -0.5;
        p.epsilon_ch = 1.0;
        p.kappa_bar_g = 0.0;
        let n = 8usize;
        let mut phi = ScalarField3D::zeros(n, n, n, 1.0);
        for i in 0..n { for j in 0..n { for k in 0..n {
            let vi = phi.idx(i,j,k);
            phi.phi[vi] = 0.5 + 0.1*(i as f64).sin();
        }}}
        let q_lip = QField3D::zeros(n, n, n, 1.0);
        let phi_plain    = ch_step_etd_3d(&phi, &q_lip, &p, p.dt);
        let phi_enriched = ch_step_etd_enriched_3d(&phi, &q_lip, &p, p.dt);
        let max_diff: f64 = phi_plain.phi.iter().zip(phi_enriched.phi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-12,
            "enriched with nonzero c0_sp must differ from plain CH"
        );
    }

    /// Cahn-Hilliard must conserve the spatial mean (total concentration).
    ///
    /// When a_ch = b_ch = chi_ms = 0, the nonlinear part N = 0 everywhere,
    /// so the equation reduces to pure linear diffusion. In that case the
    /// DC Fourier mode is unchanged by both the ETD step (Euler with N̂=0)
    /// and the non-DC step (multiplied by exp(L dt) with L real and negative,
    /// but the DC mode is skipped (it uses the Euler branch, which gives
    /// φ̂_new[DC] = φ̂[DC] + 0 = φ̂[DC]). The spatial mean equals
    /// φ̂[DC] / N, so it is exactly conserved.
    #[test]
    fn test_ch_etd_3d_conserves_mass() {
        // Set a_ch=b_ch=chi_ms=0 so N=0 everywhere; the equation reduces to
        // pure linear diffusion and the spatial mean is conserved to machine
        // precision regardless of phi.
        let mut p = MarsParams3D::default_test();
        p.a_ch = 0.0;
        p.b_ch = 0.0;
        p.chi_ms = 0.0;
        // Use a non-trivial spatially varying phi (mean = 0.3 exactly).
        // A uniform field with pure linear diffusion is trivially conserved.
        let phi = ScalarField3D::uniform(8, 8, 8, 1.0, 0.3);
        let q_lip = QField3D::zeros(8, 8, 8, 1.0);
        let phi_new = ch_step_etd_3d(&phi, &q_lip, &p, p.dt);
        let mean_before = phi.mean();
        let mean_after = phi_new.mean();
        assert!(
            (mean_after - mean_before).abs() < 1e-10,
            "CH must conserve total concentration under pure linear diffusion, \
             got delta={}",
            (mean_after - mean_before).abs()
        );
    }
}
