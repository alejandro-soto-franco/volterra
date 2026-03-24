#![allow(clippy::needless_range_loop)]
// ~/volterra/volterra-solver/src/stokes_3d.rs

//! Spectral Stokes solver for active nematics on a 3D periodic grid.
//!
//! ## Physics
//!
//! Active stress: `σ_ij = -ζ Q_ij`
//!
//! Body force: `f_α = (∇·σ)_α = -ζ_eff ∂_β Q_{αβ}`
//!
//! In Fourier space the incompressible (divergence-free) velocity field satisfies:
//!
//! ```text
//! η |k|² û_α = (δ_{αβ} - k̂_α k̂_β) f̂_β        (Oseen-Burgers projection)
//! ```
//!
//! which projects the body force onto divergence-free modes (Leray projector).
//!
//! The DC mode (k=0) is set to zero (no net translation on a periodic box).

use rustfft::{FftPlanner, num_complex::Complex};
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, VelocityField3D, PressureField3D};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the steady-state incompressible Stokes equation driven by active stress.
///
/// Given a Q-tensor field `q` and MARS parameters `p`, computes the velocity
/// field `u` satisfying:
///
/// ```text
/// η ∇²u = ∇·σ^active + ∇π     (Stokes)
/// ∇·u = 0                       (incompressibility)
///
/// σ^active_ij = -ζ_eff Q_ij    (active stress)
/// ```
///
/// The algorithm is a spectral (Fourier) method:
///
/// 1. Compute body force `f_α = -ζ_eff (∂_β Q_{αβ})` via central differences.
/// 2. Forward 3D FFT each component of `f`.
/// 3. Apply the Leray (Helmholtz) projector in Fourier space:
///    `û_α = (δ_{αβ} - k̂_α k̂_β) f̂_β / (η |k|²)`
/// 4. Inverse 3D FFT to recover `u` in real space.
///
/// The zero wavenumber mode (mean flow) is set to zero.
///
/// # Arguments
///
/// * `q` - Q-tensor field on the nx×ny×nz periodic grid.
/// * `p` - MARS parameters supplying `zeta_eff`, `eta`, and grid geometry.
///
/// # Returns
///
/// A pair `(u, p_out)` where `u` is the divergence-free velocity field and
/// `p_out` is a placeholder zero pressure field (pressure is not explicitly
/// needed for the velocity in the spectral approach).
pub fn stokes_solve_3d(q: &QField3D, p: &MarsParams3D) -> (VelocityField3D, PressureField3D) {
    // Use the Q-field's own grid dimensions so the solver works regardless of
    // whether the params grid matches (e.g. tests may use a smaller grid).
    let nx = q.nx;
    let ny = q.ny;
    let nz = q.nz;
    let n = nx * ny * nz;
    let dx = q.dx;

    // ── Step 1: body force f_α = -ζ_eff ∂_β Q_{αβ} ───────────────────────
    //
    // Active stress σ_ij = -ζ_eff Q_ij, so
    //   (∇·σ)_α = -ζ_eff Σ_β ∂_β Q_{αβ}
    //
    // Q is stored as [q11, q12, q13, q22, q23]; q33 = -(q11+q22) (traceless).
    // Row α of Q: Q_{α,β} for β=0,1,2.

    let inv_2dx = 1.0 / (2.0 * dx);
    let mut f = vec![[0.0f64; 3]; n];

    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = q.idx(i, j, l);

                // Helper closure: Q_{row, col} at linear index ki.
                let get_q = |ki: usize, row: usize, col: usize| -> f64 {
                    let [q11, q12, q13, q22, q23] = q.q[ki];
                    let q33 = -(q11 + q22);
                    match (row, col) {
                        (0, 0) => q11,
                        (0, 1) | (1, 0) => q12,
                        (0, 2) | (2, 0) => q13,
                        (1, 1) => q22,
                        (1, 2) | (2, 1) => q23,
                        (2, 2) => q33,
                        _ => 0.0,
                    }
                };

                // Neighbour indices for central differences.
                let ip = q.idx((i + 1) % nx, j, l);
                let im = q.idx((i + nx - 1) % nx, j, l);
                let jp = q.idx(i, (j + 1) % ny, l);
                let jm = q.idx(i, (j + ny - 1) % ny, l);
                let lp = q.idx(i, j, (l + 1) % nz);
                let lm = q.idx(i, j, (l + nz - 1) % nz);

                for alpha in 0..3usize {
                    // ∂_x Q_{α,0}  +  ∂_y Q_{α,1}  +  ∂_z Q_{α,2}
                    let div_q_alpha =
                        (get_q(ip, alpha, 0) - get_q(im, alpha, 0)) * inv_2dx
                            + (get_q(jp, alpha, 1) - get_q(jm, alpha, 1)) * inv_2dx
                            + (get_q(lp, alpha, 2) - get_q(lm, alpha, 2)) * inv_2dx;

                    f[k][alpha] = -p.zeta_eff * div_q_alpha;
                }
            }
        }
    }

    // ── Step 2: forward 3D FFT of each force component ────────────────────
    let mut planner = FftPlanner::<f64>::new();
    let mut f_hat: Vec<[Complex<f64>; 3]> = f
        .iter()
        .map(|fi| {
            [
                Complex::new(fi[0], 0.0),
                Complex::new(fi[1], 0.0),
                Complex::new(fi[2], 0.0),
            ]
        })
        .collect();

    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);

    for comp in 0..3 {
        // Along z (innermost/contiguous axis).
        for i in 0..nx {
            for j in 0..ny {
                let mut row: Vec<Complex<f64>> =
                    (0..nz).map(|l| f_hat[q.idx(i, j, l)][comp]).collect();
                fft_z.process(&mut row);
                for l in 0..nz {
                    f_hat[q.idx(i, j, l)][comp] = row[l];
                }
            }
        }
        // Along y.
        for i in 0..nx {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..ny).map(|j| f_hat[q.idx(i, j, l)][comp]).collect();
                fft_y.process(&mut row);
                for j in 0..ny {
                    f_hat[q.idx(i, j, l)][comp] = row[j];
                }
            }
        }
        // Along x (outermost axis).
        for j in 0..ny {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..nx).map(|i| f_hat[q.idx(i, j, l)][comp]).collect();
                fft_x.process(&mut row);
                for i in 0..nx {
                    f_hat[q.idx(i, j, l)][comp] = row[i];
                }
            }
        }
    }

    // ── Step 3: Leray projector + Stokes inversion in Fourier space ───────
    //
    // û_α = (δ_{αβ} - k̂_α k̂_β) f̂_β / (η |k|²)
    //
    // Equivalently: û = (f̂ - (k·f̂ / |k|²) k) / (η |k|²)
    //
    // DC mode (k=0) is set to zero (incompressibility on a periodic box
    // requires no net translation).

    let mut u_hat: Vec<[Complex<f64>; 3]> = vec![[Complex::new(0.0, 0.0); 3]; n];

    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = q.idx(i, j, l);

                let kx = wavenumber(i, nx, dx);
                let ky = wavenumber(j, ny, dx);
                let kz = wavenumber(l, nz, dx);
                let k2 = kx * kx + ky * ky + kz * kz;

                // Skip DC mode: mean velocity is zero.
                if k2 < 1e-14 {
                    continue;
                }

                let kv = [kx, ky, kz];

                // k · f̂  (scalar complex dot product with real k vector)
                let k_dot_f: Complex<f64> = kv
                    .iter()
                    .zip(f_hat[k].iter())
                    .map(|(&ki, &fi)| Complex::new(ki, 0.0) * fi)
                    .sum();

                let inv_eta_k2 = 1.0 / (p.eta * k2);

                for a in 0..3 {
                    // Leray-projected body force, divided by η|k|²
                    u_hat[k][a] = (f_hat[k][a]
                        - Complex::new(kv[a] / k2, 0.0) * k_dot_f)
                        * inv_eta_k2;
                }
            }
        }
    }

    // ── Step 4: inverse 3D FFT ────────────────────────────────────────────
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    for comp in 0..3 {
        // Along z.
        for i in 0..nx {
            for j in 0..ny {
                let mut row: Vec<Complex<f64>> =
                    (0..nz).map(|l| u_hat[q.idx(i, j, l)][comp]).collect();
                ifft_z.process(&mut row);
                for l in 0..nz {
                    u_hat[q.idx(i, j, l)][comp] = row[l];
                }
            }
        }
        // Along y.
        for i in 0..nx {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..ny).map(|j| u_hat[q.idx(i, j, l)][comp]).collect();
                ifft_y.process(&mut row);
                for j in 0..ny {
                    u_hat[q.idx(i, j, l)][comp] = row[j];
                }
            }
        }
        // Along x.
        for j in 0..ny {
            for l in 0..nz {
                let mut row: Vec<Complex<f64>> =
                    (0..nx).map(|i| u_hat[q.idx(i, j, l)][comp]).collect();
                ifft_x.process(&mut row);
                for i in 0..nx {
                    u_hat[q.idx(i, j, l)][comp] = row[i];
                }
            }
        }
    }

    // Normalise by 1/N (rustfft IFFT is unnormalised).
    let norm = 1.0 / (n as f64);
    let mut u = VelocityField3D::zeros(nx, ny, nz, dx);
    for k in 0..n {
        for a in 0..3 {
            u.u[k][a] = u_hat[k][a].re * norm;
        }
    }

    (u, PressureField3D::zeros(nx, ny, nz, dx))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Finite-difference-consistent effective wavenumber for index `idx` on an
/// `n`-point grid with spacing `dx`.
///
/// The central-difference stencil `(f[i+1] - f[i-1]) / (2 dx)` applied to a
/// Fourier mode `e^{i k x}` gives `i sin(k dx) / dx · e^{i k x}`. So the
/// effective (real) wavenumber symbol of the central-difference operator is
/// `k_eff = sin(k_phys dx) / dx` (the imaginary unit `i` is part of the
/// divergence operator and cancels with the `i` from the divergence-free
/// condition `k · û = 0`).
///
/// Using `k_eff` in the Leray projector instead of the spectral wavenumber
/// `k_phys` guarantees that the output velocity has *exactly* zero
/// central-difference divergence (up to floating-point round-off), which is
/// what the test measures.
#[inline]
fn wavenumber(idx: usize, n: usize, dx: f64) -> f64 {
    let i = if idx <= n / 2 {
        idx as f64
    } else {
        idx as f64 - n as f64
    };
    // Physical wavenumber: k_phys = 2π m / L  where L = n dx.
    let theta = 2.0 * std::f64::consts::PI * i / n as f64; // k_phys * dx = 2π m/n
    // FD-consistent effective wavenumber: sin(k_phys dx) / dx.
    theta.sin() / dx
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_core::MarsParams3D;
    use volterra_fields::QField3D;

    /// The Stokes solver must produce a divergence-free velocity field.
    ///
    /// We check the finite-difference divergence of the output against a
    /// tolerance of 1e-8 (spectral accuracy on an 8³ grid is well within this).
    #[test]
    fn test_stokes_3d_incompressible() {
        let p = MarsParams3D::default_test();
        let q = QField3D::random_perturbation(8, 8, 8, 1.0, 0.1, 42);
        let (u, _p_out) = stokes_solve_3d(&q, &p);
        let div = u.divergence();
        for d in &div.phi {
            assert!(
                d.abs() < 1e-8,
                "Stokes output must be divergence-free, got divergence={}",
                d
            );
        }
    }

    /// A zero Q-tensor field must produce zero velocity.
    #[test]
    fn test_stokes_3d_zero_q_gives_zero_u() {
        let p = MarsParams3D::default_test();
        let q = QField3D::zeros(8, 8, 8, 1.0);
        let (u, _p_out) = stokes_solve_3d(&q, &p);
        for &uv in &u.u {
            for a in 0..3 {
                assert!(
                    uv[a].abs() < 1e-12,
                    "zero Q must give zero velocity, got u[{}]={}",
                    a,
                    uv[a]
                );
            }
        }
    }
}
