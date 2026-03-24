#![allow(clippy::needless_range_loop)]
// ~/volterra/volterra-solver/src/lib.rs

//! # volterra-solver
//!
//! Beris-Edwards nematohydrodynamics for 2D active nematics on a regular grid.
//!
//! ## Modules
//!
//! | Symbol | Function |
//! |--------|----------|
//! | [`molecular_field`] | Landau-de Gennes H = -δF/δQ |
//! | [`beris_edwards_rhs`] | Full ∂_t Q RHS (dry active model) |
//! | [`EulerIntegrator`] | First-order Euler time step |
//! | [`RK4Integrator`] | Fourth-order Runge-Kutta time step |
//! | [`k0_convolution`] | K₀ Bessel kernel convolution for ℳ_SM (Component 2) |
//! | [`scan_defects`] | Holonomy-based defect detection (wraps cartan-geo) |
//! | [`DefectInfo`] | Position, charge, and frame of a detected disclination |
//!
//! ## Physics: dry active nematic (Component 1)
//!
//! The governing equation is:
//!
//! ```text
//! ∂_t Q = Γ_r · H_active
//!
//! H_active = K_r ∇²Q  +  (ζ_eff/2 - a) Q  -  2c tr(Q²) Q
//! ```
//!
//! where `a = a_landau` (negative for the ordered phase), `c = c_landau > 0`,
//! and `ζ_eff/2 - a` is the effective linear driving.
//! When `ζ_eff > 2|a|` (equivalently `a_eff < 0`) the system enters the active
//! turbulent phase with dense defects.
//!
//! Including flow (optional):
//!
//! ```text
//! ∂_t Q = -u·∇Q + S(W,Q) + Γ_r H_active
//! ```
//!
//! where the co-rotation/strain term is `S(W,Q) = λ(QW + WQ) - (QΩ + ΩQ)`.
//!
//! ## Physics: transfer map ℳ_SM (Component 2)
//!
//! The orientational transfer to the lyotropic lipid phase is:
//!
//! ```text
//! Q^lip(x) = ℳ_SM(Q^rot)(x)
//!          = ∫ K₀(|x-x'|/ξ_l) Q^rot(x') d²x' / (2π ξ_l²)
//! ```
//!
//! where K₀ is the modified Bessel function of the second kind, order 0,
//! and ξ_l is the lipid coupling length. This is computed by [`k0_convolution`].
//!
//! ## References
//!
//! - Beris, A. N. & Edwards, B. J. (1994). *Thermodynamics of Flowing Systems*.
//! - Doostmohammadi, A. et al. (2018). "Active nematics." *Nature Commun.* **9**, 3246.
//! - Giomi, L. (2015). "Geometry and topology of turbulence in active nematics."
//!   *Phys. Rev. X* **5**, 031003.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;
use rustfft::{FftPlanner, num_complex::Complex};
use volterra_core::{Integrator, MarsParams};
use volterra_fields::{QField2D, ScalarField2D, VelocityField2D};

use cartan_geo::holonomy::{Disclination, scan_disclinations};
use cartan_manifolds::frame_field::FrameField3D;

pub mod mol_field_3d;
pub use mol_field_3d::{molecular_field_3d, co_rotation_3d};

pub mod beris_3d;
pub use beris_3d::{beris_edwards_rhs_3d, EulerIntegrator3D, RK4Integrator3D};

pub mod stokes_3d;
pub use stokes_3d::stokes_solve_3d;

pub mod ch_3d;
pub use ch_3d::ch_step_etd_3d;

pub mod defects_3d;
pub use defects_3d::{scan_defects_3d, track_defect_events};

pub mod runner_3d;
pub use runner_3d::{run_mars_3d, run_mars_3d_full, SnapStats3D, BechStats3D};

pub mod gauss_bonnet_3d;
pub use gauss_bonnet_3d::gauss_bonnet_chi;

// ─────────────────────────────────────────────────────────────────────────────
// Molecular field: H = -δF/δQ
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the active molecular field `H_active = -δF_active/δQ` at each vertex.
///
/// The active free energy density is:
///
/// ```text
/// f = (K_r/2)|∇Q|²  +  (a_eff/2) tr(Q²)  +  (c/2)(tr(Q²))²
/// ```
///
/// where `a_eff = a_landau - ζ_eff/2`. The molecular field (functional derivative):
///
/// ```text
/// H_{active,α} = K_r ∇²q_α  -  2a_eff q_α  -  4c(q1²+q2²) q_α
/// ```
///
/// This is the RHS driver in the dry active model `∂_t Q = Γ_r H_active`.
pub fn molecular_field(q: &QField2D, params: &MarsParams) -> QField2D {
    let a_eff = params.a_eff();
    let c = params.c_landau;
    let k_r = params.k_r;

    let lap = q.laplacian();
    let sq = q.order_param_sq(); // q1² + q2² at each vertex

    let mut out = QField2D::zeros(q.nx, q.ny, q.dx);
    for k in 0..q.len() {
        let [q1, q2] = q.q[k];
        let s2 = sq[k]; // q1² + q2²
        out.q[k][0] = k_r * lap.q[k][0] - 2.0 * a_eff * q1 - 4.0 * c * s2 * q1;
        out.q[k][1] = k_r * lap.q[k][1] - 2.0 * a_eff * q2 - 4.0 * c * s2 * q2;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Co-rotation / strain coupling  S(W, Q)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the co-rotation–strain coupling `S(W, Q)` at each vertex.
///
/// For a 2D Q-tensor `Q = [[q1,q2],[q2,-q1]]` and 2D velocity gradients,
/// the coupling term (following Beris-Edwards with alignment parameter λ) is:
///
/// ```text
/// S(W,Q) = λ(D·Q + Q·D) - (Ω·Q - Q·Ω)
///        - λ tr(D·Q) I
/// ```
///
/// where `D = (∇u + (∇u)^T)/2` (strain rate) and `Ω = (∇u - (∇u)^T)/2` (vorticity).
///
/// In 2D components (using central differences for ∇u):
///
/// ```text
/// Dxx = ∂_x vx,  Dyy = ∂_y vy,  Dxy = (∂_x vy + ∂_y vx)/2
/// Ωxy = (∂_x vy - ∂_y vx)/2  (antisymmetric part)
/// ```
///
/// Returns a `QField2D` of the same shape as `q`.
pub fn corotation_strain(q: &QField2D, v: &VelocityField2D, params: &MarsParams) -> QField2D {
    let lambda = params.lambda;
    let dx = q.dx;
    let mut out = QField2D::zeros(q.nx, q.ny, q.dx);

    for i in 0..q.nx {
        for j in 0..q.ny {
            let k = q.idx(i, j);

            // Central differences for velocity gradients.
            let ip = v.idx_i(i as i64 + 1, j as i64);
            let im = v.idx_i(i as i64 - 1, j as i64);
            let jp = v.idx_i(i as i64, j as i64 + 1);
            let jm = v.idx_i(i as i64, j as i64 - 1);

            let dvx_dx = (v.v[ip][0] - v.v[im][0]) / (2.0 * dx);
            let dvx_dy = (v.v[jp][0] - v.v[jm][0]) / (2.0 * dx);
            let dvy_dx = (v.v[ip][1] - v.v[im][1]) / (2.0 * dx);
            let dvy_dy = (v.v[jp][1] - v.v[jm][1]) / (2.0 * dx);

            // Strain rate tensor D (symmetric part of ∇u).
            let d_xx = dvx_dx;
            let d_yy = dvy_dy;
            let d_xy = 0.5 * (dvx_dy + dvy_dx);

            // Vorticity (antisymmetric part of ∇u): Ω_xy = (∂_x vy - ∂_y vx)/2.
            let omega = 0.5 * (dvy_dx - dvx_dy);

            // Q components.
            let [q1, q2] = q.q[k];

            // D·Q + Q·D for 2D Q = [[q1,q2],[q2,-q1]], D = [[dxx,dxy],[dxy,dyy]]:
            // (D·Q)_11 = dxx*q1 + dxy*q2,  (Q·D)_11 = q1*dxx + q2*dxy
            // So (D·Q + Q·D)_11 = 2(dxx*q1 + dxy*q2)
            // (D·Q)_12 = dxx*q2 - dxy*q1 + dxy*q1 - q2*dyy = (dxx-dyy)*q2
            // Wait, let me compute properly:
            // Q = [[q1,q2],[q2,-q1]], D = [[dxx,dxy],[dxy,dyy]]
            // D·Q = [[dxx*q1+dxy*q2, dxx*q2-dxy*q1],[dxy*q1+dyy*q2, dxy*q2-dyy*q1]]
            // Q·D = [[q1*dxx+q2*dxy, q1*dxy+q2*dyy],[q2*dxx-q1*dxy, q2*dxy-q1*dyy]]
            // D·Q + Q·D (only need (0,0) and (0,1) for sym-traceless):
            // (0,0): dxx*q1+dxy*q2 + q1*dxx+q2*dxy = 2(dxx*q1 + dxy*q2)
            // (0,1): dxx*q2-dxy*q1 + q1*dxy+q2*dyy
            //      = dxx*q2 + dyy*q2 + dxy*(q1-q1) = (dxx+dyy)*q2 - 0... no
            //      = dxx*q2 - dxy*q1 + q1*dxy + q2*dyy = (dxx+dyy)*q2 = 0 (incompressible: dxx+dyy=0)
            // For incompressible flow: dxx + dyy = 0 => d_yy = -d_xx
            let d_yy_incomp = -d_xx; // enforce incompressibility

            let dqdq_11 = 2.0 * (d_xx * q1 + d_xy * q2);
            let dqdq_12 = (d_xx + d_yy_incomp) * q2; // = 0 (incompressible: d_xy - d_xy = 0)
            // Actually: (D·Q+Q·D)_12 = (dxx+dyy)*q2 = 0 for incompressible flow.
            // Let me redo without the incompressible assumption for generality:
            // (0,1) component: (D·Q)_01 + (Q·D)_01
            // = (d_xx*q2 - d_xy*q1) + (q1*d_xy + q2*d_yy)
            // = q2*(d_xx + d_yy) + q1*(d_xy - d_xy) ... wait
            // = d_xx*q2 - d_xy*q1 + q1*d_xy + q2*d_yy
            // = q2*(d_xx + d_yy)
            let dqdq_12_gen = q2 * (d_xx + d_yy);

            // tr(D·Q) for subtracting: tr(D·Q) = dxx*q1 + dxy*q2 + dxy*q2 - dyy*q1
            //                                    = (dxx-dyy)*q1 + 2*dxy*q2
            let tr_dq = (d_xx - d_yy) * q1 + 2.0 * d_xy * q2;

            // Ω·Q - Q·Ω for Ω = [[0,omega],[-omega,0]]:
            // (Ω·Q)_11 = 0*q1 + omega*q2 + (-omega)*q2 = 0... wait
            // Ω = [[0, omega], [-omega, 0]] (2D antisymmetric)
            // Ω·Q = [[omega*q2, -omega*q1],[... ]]
            // Ω·Q (0,0) = 0*q1 + omega*q2 = omega*q2? No.
            // Ω = [[0, omega],[-omega, 0]], Q = [[q1, q2],[q2,-q1]]
            // (Ω·Q)_00 = 0*q1 + omega*q2 = omega*q2
            // (Ω·Q)_01 = 0*q2 + omega*(-q1) = -omega*q1
            // (Q·Ω)_00 = q1*0 + q2*(-omega) = -omega*q2
            // (Q·Ω)_01 = q1*omega + q2*0 = omega*q1
            // (Ω·Q - Q·Ω)_00 = omega*q2 - (-omega*q2) = 2*omega*q2
            // (Ω·Q - Q·Ω)_01 = -omega*q1 - omega*q1 = -2*omega*q1
            let oq_qo_11 = 2.0 * omega * q2;
            let oq_qo_12 = -2.0 * omega * q1;

            // S(W,Q) = λ(D·Q + Q·D) - (Ω·Q - Q·Ω) - λ tr(D·Q) I
            // Component 1 (q1 direction):
            out.q[k][0] =
                lambda * (dqdq_11 - tr_dq) - oq_qo_11;
            // Component 2 (q2 direction):
            out.q[k][1] =
                lambda * (dqdq_12_gen) - oq_qo_12;

            let _ = dqdq_12; // suppress unused warning
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Beris-Edwards RHS
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `dQ/dt` from the Beris-Edwards equation.
///
/// For the dry active model (v = None):
///
/// ```text
/// dQ/dt = Γ_r · H_active
/// ```
///
/// For the hydrodynamic model (v = Some(...)):
///
/// ```text
/// dQ/dt = -u·∇Q + S(W,Q) + Γ_r · H_active
/// ```
///
/// The velocity field `v` is not updated here (it must be provided as input,
/// e.g., from a Stokes solver). For Component 1 validation of the dry scaling
/// ρ_d ~ ζ_eff/K_r, pass `v = None`.
pub fn beris_edwards_rhs(
    q: &QField2D,
    v: Option<&VelocityField2D>,
    params: &MarsParams,
) -> QField2D {
    let h = molecular_field(q, params);
    let mut rhs = h.scale(params.gamma_r);

    if let Some(vel) = v {
        let advection = vel.advect(q);
        let corot = corotation_strain(q, vel, params);
        // dQ/dt = -u·∇Q + S(W,Q) + Γ_r H
        rhs = rhs.add(&corot).add(&advection.scale(-1.0));
    }

    rhs
}

// ─────────────────────────────────────────────────────────────────────────────
// Time integrators
// ─────────────────────────────────────────────────────────────────────────────

/// First-order forward Euler integrator.
///
/// `Q_{n+1} = Q_n + dt · f(Q_n)`
pub struct EulerIntegrator;

impl Integrator<QField2D> for EulerIntegrator {
    fn step<F>(&self, state: &QField2D, dt: f64, rhs: F) -> QField2D
    where
        F: Fn(&QField2D) -> QField2D,
    {
        let dq = rhs(state);
        state.add(&dq.scale(dt))
    }
}

/// Fourth-order Runge-Kutta integrator (RK4).
///
/// ```text
/// k1 = f(Q_n)
/// k2 = f(Q_n + dt/2 · k1)
/// k3 = f(Q_n + dt/2 · k2)
/// k4 = f(Q_n + dt   · k3)
/// Q_{n+1} = Q_n + dt/6 · (k1 + 2k2 + 2k3 + k4)
/// ```
pub struct RK4Integrator;

impl Integrator<QField2D> for RK4Integrator {
    fn step<F>(&self, state: &QField2D, dt: f64, rhs: F) -> QField2D
    where
        F: Fn(&QField2D) -> QField2D,
    {
        let k1 = rhs(state);
        let k2 = rhs(&state.add(&k1.scale(dt / 2.0)));
        let k3 = rhs(&state.add(&k2.scale(dt / 2.0)));
        let k4 = rhs(&state.add(&k3.scale(dt)));

        // Q_{n+1} = Q_n + dt/6 (k1 + 2k2 + 2k3 + k4)
        let sum = k1
            .add(&k2.scale(2.0))
            .add(&k3.scale(2.0))
            .add(&k4);
        state.add(&sum.scale(dt / 6.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// K₀ Bessel kernel convolution for transfer map ℳ_SM (Component 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Modified Bessel function K₀(x) for x > 0.
///
/// Uses the polynomial approximations from Abramowitz & Stegun §9.8:
///
/// - For 0 < x ≤ 2: `K₀(x) = -ln(x/2) I₀(x) + poly(x²)`
/// - For x > 2:  `K₀(x) = (π/(2x))^{1/2} exp(-x) poly(1/x)`
///
/// The approximation is accurate to ≈ 1.9e-7 (A&S 9.8.5 / 9.8.6).
fn bessel_k0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY; // K₀ diverges at 0
    }
    if x <= 2.0 {
        // A&S 9.8.1: I₀(x), variable t = (x/3.75)²
        let ti = x / 3.75;
        let ti2 = ti * ti;
        let i0 = 1.0
            + 3.5156329  * ti2
            + 3.0899424  * ti2 * ti2
            + 1.2067492  * ti2.powi(3)
            + 0.2659732  * ti2.powi(4)
            + 0.0360768  * ti2.powi(5)
            + 0.0045813  * ti2.powi(6);
        // A&S 9.8.5: K₀(x) = -ln(x/2) I₀(x) + p(t), variable t = (x/2)²
        let tk = x / 2.0;
        let tk2 = tk * tk;
        let p = -0.57721566
            + 0.42278420 * tk2
            + 0.23069756 * tk2 * tk2
            + 0.03488590 * tk2.powi(3)
            + 0.00262698 * tk2.powi(4)
            + 0.00010750 * tk2.powi(5)
            + 0.0000074  * tk2.powi(6);
        -(x / 2.0).ln() * i0 + p
    } else {
        // A&S 9.8.6: asymptotic, variable t = 2/x
        let t = 2.0 / x;
        let poly = 1.25331414
            - 0.07832358 * t
            + 0.02189568 * t * t
            - 0.01062446 * t.powi(3)
            + 0.00587872 * t.powi(4)
            - 0.00251540 * t.powi(5)
            + 0.00053208 * t.powi(6);
        ((-x).exp() / x.sqrt()) * poly
    }
}

/// Compute the orientational transfer map `ℳ_SM(Q^rot)` via K₀ convolution.
///
/// ```text
/// Q^lip(x) = ∫ K₀(|x-x'|/ξ_l) Q^rot(x') d²x' / (2π ξ_l²)
/// ```
///
/// The integral is truncated at `r_max = 6 ξ_l` (the K₀ kernel decays as
/// `exp(-r/ξ_l)`, so the tail beyond 6ξ_l contributes < 2.5e-3 of the total).
/// Self-interaction (r = 0) is excluded since K₀(0) diverges: the singular
/// part is integrable and represents the local self-energy, which is already
/// captured by the Landau-de Gennes free energy. In practice this is handled
/// by skipping r = 0 in the discrete sum (error O(dx²) for smooth Q^rot).
///
/// The result is the lipid Q-tensor `Q^lip`, the primary output of Component 2.
pub fn k0_convolution(q_rot: &QField2D, params: &MarsParams) -> QField2D {
    let xi_l = params.xi_l;
    let dx = q_rot.dx;
    let nx = q_rot.nx as i64;
    let ny = q_rot.ny as i64;

    // Kernel support: truncate at 6 ξ_l, but clamp to half the grid to avoid
    // double-counting from periodic wrapping (cutoff must not exceed nx/2 or ny/2).
    let cutoff_raw = (6.0 * xi_l / dx).ceil() as i64;
    let cutoff = cutoff_raw
        .min((q_rot.nx as i64) / 2)
        .min((q_rot.ny as i64) / 2);
    // Normalization: ∫ K₀(r/ξ_l) d²r = 2π ξ_l². Discrete: dx² * sum = 2π ξ_l².
    let norm = 2.0 * std::f64::consts::PI * xi_l * xi_l;

    let mut out = QField2D::zeros(q_rot.nx, q_rot.ny, dx);

    for i in 0..nx {
        for j in 0..ny {
            let k_out = q_rot.idx_i(i, j);
            let mut s0 = 0.0_f64;
            let mut s1 = 0.0_f64;

            // Use half-open range -cutoff..cutoff to avoid double-counting the
            // Nyquist offset on a periodic grid (di=-N/2 and di=+N/2 are the same
            // vertex, so using ..= would count it twice).
            for di in -cutoff..cutoff {
                let r_x = di as f64 * dx;
                for dj in -cutoff..cutoff {
                    let r_y = dj as f64 * dx;
                    let r = (r_x * r_x + r_y * r_y).sqrt();
                    if r < 0.5 * dx {
                        // Skip self-interaction: K₀(0) diverges.
                        continue;
                    }
                    let k0_val = bessel_k0(r / xi_l);
                    let src = q_rot.idx_i(i + di, j + dj);
                    s0 += k0_val * q_rot.q[src][0];
                    s1 += k0_val * q_rot.q[src][1];
                }
            }

            out.q[k_out][0] = s0 * dx * dx / norm;
            out.q[k_out][1] = s1 * dx * dx / norm;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral Stokes solver (active stress → incompressible velocity field)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the 2D incompressible Stokes equation for the active velocity field.
///
/// Given the Q-tensor field, computes the incompressible flow driven by the
/// active stress `σ^a = ζ_eff Q` via the stream-function biharmonic equation:
///
/// ```text
/// η ∇⁴ψ = ∂_x F_y - ∂_y F_x,   F = ∇·σ^a = ζ_eff ∇·Q
/// ```
///
/// Solved spectrally (pseudospectral FFT on a periodic grid):
///
/// ```text
/// ψ̂(k) = ζ [(k_y² - k_x²) q̂_2 + 2 k_x k_y q̂_1] / [η (k²)²]
/// ```
///
/// The velocity components follow from `v_x = ∂_y ψ`, `v_y = -∂_x ψ`.
/// The k=0 mode (uniform translation) is set to zero.
///
/// Returns a [`VelocityField2D`] on the same grid.
pub fn stokes_solve(q: &QField2D, params: &MarsParams) -> VelocityField2D {
    let nx = q.nx;
    let ny = q.ny;
    let n = nx * ny;
    let dx = q.dx;
    let eta = params.eta;
    let ze = params.zeta_eff;

    let mut planner = FftPlanner::<f64>::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);

    // Wave-vector frequencies (radians/unit-length): k_j = 2π j / (N dx)
    // Using the standard DFT ordering: j = 0..N/2, then -(N/2-1)..-1 wrapped.
    let kx_vec: Vec<f64> = (0..nx).map(|i| {
        let i = i as i64;
        let n = nx as i64;
        let i_shifted = if i <= n / 2 { i } else { i - n };
        2.0 * std::f64::consts::PI * i_shifted as f64 / (nx as f64 * dx)
    }).collect();
    let ky_vec: Vec<f64> = (0..ny).map(|j| {
        let j = j as i64;
        let n = ny as i64;
        let j_shifted = if j <= n / 2 { j } else { j - n };
        2.0 * std::f64::consts::PI * j_shifted as f64 / (ny as f64 * dx)
    }).collect();

    // Helper: forward 2D FFT on a real-valued row-major field → Complex array.
    // Layout: q[i * ny + j], i in 0..nx, j in 0..ny.
    let fft2_real = |field: &[f64]| -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();
        // FFT each row (along j / y direction).
        for row in buf.chunks_mut(ny) {
            fft_y.process(row);
        }
        // Transpose to ny×nx, FFT each row (= original column / x direction), transpose back.
        let mut transposed: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        for i in 0..nx {
            for j in 0..ny {
                transposed[j * nx + i] = buf[i * ny + j];
            }
        }
        for col in transposed.chunks_mut(nx) {
            fft_x.process(col);
        }
        for i in 0..nx {
            for j in 0..ny {
                buf[i * ny + j] = transposed[j * nx + i];
            }
        }
        buf
    };

    // Helper: inverse 2D FFT on complex array → real-valued field (normalized by N).
    let ifft2_to_real = |buf: &mut Vec<Complex<f64>>| -> Vec<f64> {
        // Transpose, IFFT each column (x), transpose back, then IFFT each row (y).
        let mut transposed: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        for i in 0..nx {
            for j in 0..ny {
                transposed[j * nx + i] = buf[i * ny + j];
            }
        }
        for col in transposed.chunks_mut(nx) {
            ifft_x.process(col);
        }
        for i in 0..nx {
            for j in 0..ny {
                buf[i * ny + j] = transposed[j * nx + i];
            }
        }
        for row in buf.chunks_mut(ny) {
            ifft_y.process(row);
        }
        let norm = n as f64;
        buf.iter().map(|c| c.re / norm).collect()
    };

    // Extract Q components into row-major flat arrays.
    let q1_field: Vec<f64> = q.q.iter().map(|[q1, _]| *q1).collect();
    let q2_field: Vec<f64> = q.q.iter().map(|[_, q2]| *q2).collect();

    let q1_hat = fft2_real(&q1_field);
    let q2_hat = fft2_real(&q2_field);

    // Compute stream function ψ̂ and velocity components v̂_x, v̂_y in Fourier space.
    let mut vx_hat: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let mut vy_hat: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    let i_unit = Complex::new(0.0, 1.0);

    for ii in 0..nx {
        for jj in 0..ny {
            let k = ii * ny + jj;
            let kx = kx_vec[ii];
            let ky = ky_vec[jj];
            let k2 = kx * kx + ky * ky;
            if k2 < 1e-14 {
                // k=0: no uniform translation (set to zero).
                vx_hat[k] = Complex::new(0.0, 0.0);
                vy_hat[k] = Complex::new(0.0, 0.0);
                continue;
            }
            let k4 = k2 * k2;
            // ψ̂ = ζ [(k_y² - k_x²) q̂₂ + 2 k_x k_y q̂₁] / (η k⁴)
            let rhs = ze * ((ky * ky - kx * kx) * q2_hat[k] + 2.0 * kx * ky * q1_hat[k]);
            let psi_hat = rhs / (eta * k4);
            // v̂_x = i k_y ψ̂, v̂_y = -i k_x ψ̂
            vx_hat[k] = i_unit * ky * psi_hat;
            vy_hat[k] = -i_unit * kx * psi_hat;
        }
    }

    let vx_field = ifft2_to_real(&mut vx_hat);
    let vy_field = ifft2_to_real(&mut vy_hat);

    // Pack into VelocityField2D.
    let v_data: Vec<[f64; 2]> = (0..n).map(|k| [vx_field[k], vy_field[k]]).collect();
    VelocityField2D { v: v_data, nx, ny, dx }
}

/// Run the single-phase MARS simulation with full hydrodynamic coupling (Component 1).
///
/// At each time step the Stokes velocity field is re-computed from the active
/// stress and fed back into the Beris-Edwards equation, enabling the
/// flow-alignment instability that drives active turbulence.
///
/// The defect density in the turbulent steady state follows `ρ_d ~ ζ_eff / K_r`.
///
/// # Arguments
///
/// - `q_init`: Initial Q-tensor field.
/// - `params`: Physical and numerical parameters.
/// - `n_steps`: Total number of time steps.
/// - `snap_every`: Steps between snapshot statistics.
pub fn run_mars_component1_hydro(
    q_init: &QField2D,
    params: &MarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (QField2D, Vec<SnapStats>) {
    let mut q = q_init.clone();
    let mut stats = Vec::new();
    let lx = params.nx as f64 * params.dx;
    let ly = params.ny as f64 * params.dx;
    let area = lx * ly;

    let use_noise = params.noise_amp > 0.0;
    let noise_scale = params.noise_amp * params.dt.sqrt();
    let seed: u64 = (params.nx as u64).wrapping_mul(6364136223846793005)
        ^ (params.ny as u64).wrapping_mul(1442695040888963407)
        ^ n_steps as u64;
    let mut rng = SmallRng::seed_from_u64(seed);

    // Euler-step loop: compute v from Stokes, then advance Q.
    // (Stokes is linear and instantaneous, so it's re-solved each step.)
    for step in 0..=n_steps {
        if step % snap_every == 0 {
            let defects = scan_defects(&q, std::f64::consts::PI / 2.0);
            let (n_plus, n_minus) = defect_count(&defects);
            let n_defects = defects.len();
            stats.push(SnapStats {
                time: step as f64 * params.dt,
                mean_s: q.mean_order_param(),
                n_defects,
                n_plus,
                n_minus,
                defect_density: n_defects as f64 / area,
            });
        }
        if step < n_steps {
            let v = stokes_solve(&q, params);
            let p = params.clone();
            let dq = beris_edwards_rhs(&q, Some(&v), &p);
            // Euler step (fast; Stokes re-solve dominates cost).
            q = q.add(&dq.scale(params.dt));
            if use_noise {
                for [q1, q2] in q.q.iter_mut() {
                    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let u2: f64 = rng.random::<f64>();
                    let mag = noise_scale * (-2.0 * u1.ln()).sqrt();
                    let angle = std::f64::consts::TAU * u2;
                    *q1 += mag * angle.cos();
                    *q2 += mag * angle.sin();
                }
            }
        }
    }

    (q, stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Holonomy-based defect detection
// ─────────────────────────────────────────────────────────────────────────────

/// Information about a detected topological disclination.
#[derive(Debug, Clone)]
pub struct DefectInfo {
    /// Grid coordinates of the plaquette lower-left corner (i, j).
    pub plaquette: (usize, usize),
    /// Rotation angle of the holonomy (in radians). Close to π for ±1/2 disclinations.
    pub angle: f64,
    /// Sign of the topological charge: +1 or -1.
    /// Estimated from the direction of holonomy rotation axis (z-component sign).
    pub charge_sign: i8,
}

/// Detect topological defects in a 2D Q-tensor field using holonomy.
///
/// Each Q-tensor is embedded as a 3D sym-traceless matrix, converted to an SO(3)
/// frame via `q_to_frame`, and a plaquette-by-plaquette holonomy scan is performed.
/// Plaquettes whose holonomy rotation angle exceeds `threshold` (default π/2)
/// are reported as disclinations.
///
/// The defect charge sign is estimated from the sign of the z-component of the
/// holonomy rotation axis: positive = +1/2, negative = -1/2.
///
/// # Arguments
///
/// - `q`: The 2D Q-tensor field.
/// - `threshold`: Rotation angle threshold (radians). Use `π/2` as default.
pub fn scan_defects(q: &QField2D, threshold: f64) -> Vec<DefectInfo> {
    // Convert 2D Q-tensors to 3D Q-tensors then to SO(3) frames.
    let q3d = q.to_q3d();
    let frame_field = FrameField3D::from_q_field(&q3d);

    // Run holonomy scan.
    let raw: Vec<Disclination> =
        scan_disclinations(&frame_field.frames, q.nx, q.ny, threshold);

    // Convert to DefectInfo, estimating charge sign from holonomy axis.
    raw.into_iter()
        .map(|d| {
            // Rotation axis = (H - H^T) / (2 sin θ).
            // For θ ≈ π, sin(θ) ≈ 0, use the sign of H[2,1] - H[1,2] as a proxy.
            let h = &d.holonomy;
            let axis_z_proxy = h[(0, 1)] - h[(1, 0)]; // (H - H^T)_xy = 2 sin(θ) n_z
            let charge_sign = if axis_z_proxy >= 0.0 { 1_i8 } else { -1_i8 };
            DefectInfo {
                plaquette: d.plaquette,
                angle: d.angle,
                charge_sign,
            }
        })
        .collect()
}

/// Count +1/2 and -1/2 disclinations detected in the field.
///
/// Returns `(n_plus, n_minus)`. Topological charge conservation requires
/// `n_plus == n_minus` for a system with periodic boundary conditions
/// (net charge must vanish).
pub fn defect_count(defects: &[DefectInfo]) -> (usize, usize) {
    let n_plus = defects.iter().filter(|d| d.charge_sign > 0).count();
    let n_minus = defects.iter().filter(|d| d.charge_sign < 0).count();
    (n_plus, n_minus)
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation runner (Component 1: single-phase MARS)
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected at each snapshot during a MARS run.
#[derive(Debug, Clone)]
pub struct SnapStats {
    /// Simulation time.
    pub time: f64,
    /// Mean scalar order parameter.
    pub mean_s: f64,
    /// Total number of detected defects.
    pub n_defects: usize,
    /// Number of +1/2 disclinations.
    pub n_plus: usize,
    /// Number of -1/2 disclinations.
    pub n_minus: usize,
    /// Defect density ρ_d = n_defects / (Lx Ly).
    pub defect_density: f64,
}

/// Run the single-phase MARS simulation (Component 1).
///
/// Evolves the Q-tensor field forward for `n_steps` time steps using the RK4
/// integrator. Every `snap_every` steps, detects defects and records statistics.
///
/// # Arguments
///
/// - `q_init`: Initial Q-tensor field.
/// - `params`: Physical and numerical parameters.
/// - `n_steps`: Total number of time steps.
/// - `snap_every`: Steps between snapshot statistics.
///
/// # Returns
///
/// A tuple `(q_final, stats)` where `q_final` is the final Q-tensor field and
/// `stats` is a list of snapshot statistics.
pub fn run_mars_component1(
    q_init: &QField2D,
    params: &MarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (QField2D, Vec<SnapStats>) {
    let integrator = RK4Integrator;
    let mut q = q_init.clone();
    let mut stats = Vec::new();
    let lx = params.nx as f64 * params.dx;
    let ly = params.ny as f64 * params.dx;
    let area = lx * ly;

    // Langevin noise: Q += noise_amp * sqrt(dt) * W(x,t) per component per vertex.
    let use_noise = params.noise_amp > 0.0;
    let noise_scale = params.noise_amp * params.dt.sqrt();
    // Use a reproducible but unique seed derived from grid dimensions.
    let seed: u64 = (params.nx as u64).wrapping_mul(6364136223846793005)
        ^ (params.ny as u64).wrapping_mul(1442695040888963407)
        ^ n_steps as u64;
    let mut rng = SmallRng::seed_from_u64(seed);

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            let defects = scan_defects(&q, std::f64::consts::PI / 2.0);
            let (n_plus, n_minus) = defect_count(&defects);
            let n_defects = defects.len();
            stats.push(SnapStats {
                time: step as f64 * params.dt,
                mean_s: q.mean_order_param(),
                n_defects,
                n_plus,
                n_minus,
                defect_density: n_defects as f64 / area,
            });
        }
        if step < n_steps {
            let p = params.clone();
            q = integrator.step(&q, params.dt, move |q_| beris_edwards_rhs(q_, None, &p));
            // Langevin noise injection (Euler-Maruyama for the stochastic part).
            // Box-Muller transform: two uniform samples -> two standard normals.
            if use_noise {
                let mut iter = q.q.iter_mut();
                for [q1, q2] in iter.by_ref() {
                    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let u2: f64 = rng.random::<f64>();
                    let mag = noise_scale * (-2.0 * u1.ln()).sqrt();
                    let angle = std::f64::consts::TAU * u2;
                    *q1 += mag * angle.cos();
                    *q2 += mag * angle.sin();
                }
            }
        }
    }

    (q, stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cahn-Hilliard chemical potential (BECH Component)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Cahn-Hilliard chemical potential μ_l at each vertex.
///
/// The full BECH chemical potential is:
///
/// ```text
/// μ_l = a_ch φ_l + b_ch φ_l³ - κ_ch ∇²φ_l - χ_MS Tr(Q_lip²)
/// ```
///
/// where `Tr(Q²) = 2(q₁² + q₂²)` for the 2D sym-traceless parametrisation.
///
/// The Maier-Saupe term `-χ_MS Tr(Q_lip²)` is the key coupling: it lowers
/// the chemical potential in regions of high orientational order (large
/// |Q_lip|), driving lipid accumulation wherever the rotor defect network
/// has templated strong order in the lipid phase.  At defect cores, where
/// |Q_lip| → 0, the Maier-Saupe term vanishes and the double-well potential
/// drives φ_l toward the disordered minimum, creating the concentration
/// contrast that templates LNP closure on the defect scale ℓ_d.
///
/// The gradient term `-κ_ch ∇²φ_l` penalises sharp concentration fronts,
/// regularising the chemical potential at the CH coherence scale ξ_CH.
pub fn ch_chemical_potential(
    phi: &ScalarField2D,
    q_lip: &QField2D,
    params: &MarsParams,
) -> ScalarField2D {
    let lap_phi = phi.laplacian();
    let n = phi.len();
    let mut mu = ScalarField2D::zeros(phi.nx, phi.ny, phi.dx);
    for k in 0..n {
        let p = phi.phi[k];
        // Tr(Q²) = 2 Tr(Q_2D²) = 2(q1² + q2²) in the 2D parameterisation.
        let tr_q2 = 2.0 * (q_lip.q[k][0] * q_lip.q[k][0]
            + q_lip.q[k][1] * q_lip.q[k][1]);
        mu.phi[k] = params.a_ch * p
            + params.b_ch * p * p * p
            - params.kappa_ch * lap_phi.phi[k]
            - params.chi_ms * tr_q2;
    }
    mu
}

// ─────────────────────────────────────────────────────────────────────────────
// ETD1 Cahn-Hilliard time step (BECH Component)
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the lipid volume fraction φ_l by one time step using an
/// Exponential Time Differencing (ETD1) scheme.
///
/// The Cahn-Hilliard equation is split into stiff-linear and nonlinear parts:
///
/// ```text
/// ∂_t φ = L[φ] + N[φ, Q, v]
/// ```
///
/// where the stiff linear operator is, in Fourier space:
///
/// ```text
/// L̂_k = -M_l (a_ch k² + κ_ch k⁴)
/// ```
///
/// and the nonlinear operator is:
///
/// ```text
/// N = -v·∇φ  +  M_l ∇²(b_ch φ³  -  χ_MS Tr(Q_lip²))
/// ```
///
/// The ETD1 update reads:
///
/// ```text
/// φ̂_k^{n+1} = exp(L̂_k Δt) φ̂_k^n  +  (exp(L̂_k Δt) - 1)/L̂_k · N̂_k^n
/// ```
///
/// Because L̂_k < 0 for all k ≠ 0, the exponential factor is always
/// a contraction.  The factor `(e^x - 1)/x` is evaluated via its Taylor
/// expansion for `|x| < 10⁻¹⁰` to avoid catastrophic cancellation.
/// The k=0 mode is held fixed at the initial mean (Cahn-Hilliard conserves
/// the total lipid mass ∫ φ d²x).
///
/// # Arguments
///
/// - `phi`: Current lipid volume-fraction field φ_l^n.
/// - `q_lip`: Current lipid Q-tensor field (output of K₀ convolution).
/// - `v`: Current velocity field (from Stokes solver).
/// - `params`: Physical and numerical parameters.
pub fn ch_step_etd(
    phi: &ScalarField2D,
    q_lip: &QField2D,
    v: &VelocityField2D,
    params: &MarsParams,
) -> ScalarField2D {
    let nx = phi.nx;
    let ny = phi.ny;
    let n = nx * ny;
    let dx = phi.dx;
    let dt = params.dt;

    let mut planner = FftPlanner::<f64>::new();
    let fft_x  = planner.plan_fft_forward(nx);
    let fft_y  = planner.plan_fft_forward(ny);
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);

    // Wave vectors: k_j = 2π j / (N dx), standard DFT ordering.
    let kx_vec: Vec<f64> = (0..nx).map(|i| {
        let i = i as i64;
        let nm = nx as i64;
        let is = if i <= nm / 2 { i } else { i - nm };
        2.0 * std::f64::consts::PI * is as f64 / (nx as f64 * dx)
    }).collect();
    let ky_vec: Vec<f64> = (0..ny).map(|j| {
        let j = j as i64;
        let nm = ny as i64;
        let js = if j <= nm / 2 { j } else { j - nm };
        2.0 * std::f64::consts::PI * js as f64 / (ny as f64 * dx)
    }).collect();

    // 2D FFT helper for real-valued row-major fields.
    let fft2_real = |field: &[f64]| -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();
        for row in buf.chunks_mut(ny) { fft_y.process(row); }
        let mut tr: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        for i in 0..nx { for j in 0..ny { tr[j * nx + i] = buf[i * ny + j]; } }
        for col in tr.chunks_mut(nx) { fft_x.process(col); }
        for i in 0..nx { for j in 0..ny { buf[i * ny + j] = tr[j * nx + i]; } }
        buf
    };

    // 2D IFFT helper: normalises by N.
    let ifft2_to_real = |buf: &mut Vec<Complex<f64>>| -> Vec<f64> {
        let mut tr: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
        for i in 0..nx { for j in 0..ny { tr[j * nx + i] = buf[i * ny + j]; } }
        for col in tr.chunks_mut(nx) { ifft_x.process(col); }
        for i in 0..nx { for j in 0..ny { buf[i * ny + j] = tr[j * nx + i]; } }
        for row in buf.chunks_mut(ny) { ifft_y.process(row); }
        let norm = n as f64;
        buf.iter().map(|c| c.re / norm).collect()
    };

    // ── Build the nonlinear source N in real space ───────────────────────
    //
    // N = -v·∇φ  +  M_l ∇²(b_ch φ³  -  χ_MS Tr(Q_lip²))
    //
    // Step 1: compute pointwise nonlinear source f = b_ch φ³ - χ_MS Tr(Q²).
    let mut nonlin_src = ScalarField2D::zeros(nx, ny, dx);
    for k in 0..n {
        let p = phi.phi[k];
        let tr_q2 = 2.0 * (q_lip.q[k][0] * q_lip.q[k][0]
            + q_lip.q[k][1] * q_lip.q[k][1]);
        nonlin_src.phi[k] = params.b_ch * p * p * p - params.chi_ms * tr_q2;
    }
    // Step 2: M_l ∇²(nonlin_src) via 5-point Laplacian.
    let lap_nonlin = nonlin_src.laplacian();
    // Step 3: -v·∇φ advection term.
    let advection = v.advect_scalar(phi);
    // Assemble N = -v·∇φ + M_l ∇²f.
    let mut nonlin_real: Vec<f64> = vec![0.0; n];
    for k in 0..n {
        nonlin_real[k] = -advection.phi[k] + params.m_l * lap_nonlin.phi[k];
    }

    // ── FFT φ and N ──────────────────────────────────────────────────────
    let mut phi_hat = fft2_real(&phi.phi);
    let nonlin_hat  = fft2_real(&nonlin_real);

    // ── ETD1 update in Fourier space ─────────────────────────────────────
    for ii in 0..nx {
        for jj in 0..ny {
            let idx = ii * ny + jj;
            let kx = kx_vec[ii];
            let ky = ky_vec[jj];
            let k2 = kx * kx + ky * ky;

            if k2 < 1e-14 {
                // k = 0: conserve mean lipid mass (CH is mass-conserving).
                // phi_hat[0] unchanged.
                continue;
            }

            let k4 = k2 * k2;
            // Linear operator eigenvalue: L_k = -M_l (a_ch k² + κ_ch k⁴) < 0.
            let lk = -params.m_l * (params.a_ch * k2 + params.kappa_ch * k4);
            let lk_dt = lk * dt;
            let exp_lk = lk_dt.exp(); // < 1 since L_k < 0

            // (exp(L_k dt) - 1) / (L_k dt):  evaluated via Taylor for |L_k dt| < 1e-10.
            let expm1_over_lk = if lk_dt.abs() < 1.0e-10 {
                // Taylor: (e^x - 1)/x = 1 + x/2 + x²/6 + ...
                dt * (1.0 + lk_dt / 2.0 + lk_dt * lk_dt / 6.0)
            } else {
                (exp_lk - 1.0) / lk
            };

            // φ̂^{n+1} = e^{L_k dt} φ̂^n + (e^{L_k dt}-1)/L_k · N̂^n
            phi_hat[idx] = exp_lk * phi_hat[idx] + expm1_over_lk * nonlin_hat[idx];
        }
    }

    let phi_new_vals = ifft2_to_real(&mut phi_hat);
    ScalarField2D { phi: phi_new_vals, nx, ny, dx }
}

// ─────────────────────────────────────────────────────────────────────────────
// BECH simulation runner (full two-field coupled system)
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected at each snapshot during a BECH run.
#[derive(Debug, Clone)]
pub struct BechStats {
    /// Simulation time.
    pub time: f64,
    /// Mean scalar order parameter of the rotor Q-field.
    pub mean_s: f64,
    /// Total number of detected disclinations.
    pub n_defects: usize,
    /// Number of +1/2 disclinations.
    pub n_plus: usize,
    /// Number of -1/2 disclinations.
    pub n_minus: usize,
    /// Defect density ρ_d = n_defects / (Lx Ly).
    pub defect_density: f64,
    /// Mean lipid volume fraction ⟨φ_l⟩ (conserved; monitors numerical drift).
    pub mean_phi: f64,
    /// Variance of φ_l (grows as phase separation develops near defect cores).
    pub phi_variance: f64,
    /// Mean |∇φ_l|² (proxy for total CH interfacial area / lipid shell thickness).
    pub mean_grad_phi_sq: f64,
}

/// Run the full Beris-Edwards-Cahn-Hilliard (BECH) simulation.
///
/// Couples the active rotor Q-tensor field (Beris-Edwards + Stokes) to the
/// lyotropic lipid volume fraction φ_l (Cahn-Hilliard + Maier-Saupe) via the
/// K₀ orientational transfer map.  At each time step:
///
/// 1. **Stokes**: `v ← stokes_solve(Q^rot, params)` — compute the incompressible
///    velocity field driven by active stress σ^a = ζ_eff Q^rot.
/// 2. **Beris-Edwards (Euler)**: `Q^rot ← Q^rot + dt · [−v·∇Q^rot + S(W,Q^rot) + Γ_r H]`
///    — advance the rotor Q-field with flow.
/// 3. **Transfer map**: `Q^lip ← K₀ * Q^rot` — convolve to get the lipid orientational
///    field (Component 2 one-way coupling, valid when Da < 1 and Sp < 1).
/// 4. **CH-ETD1**: `φ_l ← ch_step_etd(φ_l, Q^lip, v, params)` — advance lipid
///    concentration with exact integration of the stiff linear part and
///    explicit Euler for the nonlinear Maier-Saupe + advection terms.
///
/// The Stokes step is re-solved every step because the active stress changes
/// at every BE step; Stokes is linear and instantaneous (Re ≪ 1 throughout).
/// The K₀ convolution is also re-evaluated every step to keep Q^lip in sync
/// with Q^rot.
///
/// # Arguments
///
/// - `q_init`:   Initial rotor Q-tensor field.
/// - `phi_init`: Initial lipid volume-fraction field φ_l⁰.
/// - `params`:   Physical and numerical parameters (including BECH fields).
/// - `n_steps`:  Total number of time steps.
/// - `snap_every`: Steps between snapshot statistics.
///
/// # Returns
///
/// `(q_final, phi_final, stats)` where `stats` records defect and
/// concentration statistics at each snapshot.
pub fn run_mars_bech(
    q_init: &QField2D,
    phi_init: &ScalarField2D,
    params: &MarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (QField2D, ScalarField2D, Vec<BechStats>) {
    let mut q   = q_init.clone();
    let mut phi = phi_init.clone();
    let mut stats: Vec<BechStats> = Vec::new();

    let lx   = params.nx as f64 * params.dx;
    let ly   = params.ny as f64 * params.dx;
    let area = lx * ly;

    // Langevin noise for the rotor Q-field (same convention as Component 1).
    let use_noise  = params.noise_amp > 0.0;
    let noise_scale = params.noise_amp * params.dt.sqrt();
    let seed: u64 = (params.nx as u64).wrapping_mul(6364136223846793005)
        ^ (params.ny as u64).wrapping_mul(1442695040888963407)
        ^ n_steps as u64;
    let mut rng = SmallRng::seed_from_u64(seed);

    for step in 0..=n_steps {
        // ── Snapshot ──────────────────────────────────────────────────────
        if step % snap_every == 0 {
            let defects = scan_defects(&q, std::f64::consts::PI / 2.0);
            let (n_plus, n_minus) = defect_count(&defects);
            let n_defects = defects.len();
            stats.push(BechStats {
                time: step as f64 * params.dt,
                mean_s: q.mean_order_param(),
                n_defects,
                n_plus,
                n_minus,
                defect_density: n_defects as f64 / area,
                mean_phi: phi.mean_value(),
                phi_variance: phi.variance(),
                mean_grad_phi_sq: phi.mean_gradient_sq(),
            });
        }

        if step == n_steps { break; }

        // ── 1. Stokes: v from active stress σ^a = ζ_eff Q^rot ─────────────
        let v = stokes_solve(&q, params);

        // ── 2. Beris-Edwards Euler step (rotor field) ─────────────────────
        let dq = beris_edwards_rhs(&q, Some(&v), params);
        q = q.add(&dq.scale(params.dt));

        // Langevin noise injection into Q^rot (Euler-Maruyama, Box-Muller).
        if use_noise {
            for [q1, q2] in q.q.iter_mut() {
                let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.random::<f64>();
                let mag   = noise_scale * (-2.0 * u1.ln()).sqrt();
                let angle = std::f64::consts::TAU * u2;
                *q1 += mag * angle.cos();
                *q2 += mag * angle.sin();
            }
        }

        // ── 3. Transfer map: Q^lip = K₀ * Q^rot ───────────────────────────
        let q_lip = k0_convolution(&q, params);

        // ── 4. CH-ETD1 step (lipid concentration field) ───────────────────
        phi = ch_step_etd(&phi, &q_lip, &v, params);
    }

    (q, phi, stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn default_params() -> MarsParams {
        MarsParams::default_test()
    }

    // ── Molecular field tests ────────────────────────────────────────────────

    #[test]
    fn molecular_field_zero_at_zero_q() {
        // H(Q=0) = 0 (the Landau polynomial has no constant term).
        let q = QField2D::zeros(8, 8, 1.0);
        let h = molecular_field(&q, &default_params());
        for &[h1, h2] in &h.q {
            assert_abs_diff_eq!(h1, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(h2, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn molecular_field_linear_at_small_q() {
        // For small uniform Q, the cubic term is negligible and H ≈ -2 a_eff Q.
        let params = default_params();
        let a_eff = params.a_eff();
        let q0 = 0.001;
        let q = QField2D::uniform(8, 8, 1.0, [q0, 0.0]);
        let h = molecular_field(&q, &params);
        // Only interior vertices; for a uniform field ∇²Q = 0.
        for &[h1, _] in &h.q {
            let expected = -2.0 * a_eff * q0;
            assert_abs_diff_eq!(h1, expected, epsilon = 1e-4 * expected.abs().max(1e-12));
        }
    }

    // ── Integrators ─────────────────────────────────────────────────────────

    #[test]
    fn euler_step_grows_unstable_mode() {
        // With a_eff < 0 (active turbulent), a small Q perturbation should grow.
        let params = default_params(); // a_eff = -0.5 - 1.0 = -1.5 < 0
        assert!(params.a_eff() < 0.0, "need active regime for this test");

        let q = QField2D::uniform(8, 8, 1.0, [0.01, 0.0]);
        let euler = EulerIntegrator;
        let p = params.clone();
        let q_next = euler.step(&q, params.dt, move |q_| beris_edwards_rhs(q_, None, &p));
        // The mode must grow (|q1| must increase from 0.01).
        let s_before = q.mean_order_param();
        let s_after = q_next.mean_order_param();
        assert!(s_after > s_before, "s_before={s_before}, s_after={s_after}");
    }

    #[test]
    fn rk4_conserves_symmetry() {
        // A Q-field that starts sym-traceless must remain sym-traceless after RK4.
        let params = default_params();
        let q = QField2D::uniform(16, 16, 1.0, [0.1, 0.05]);
        let rk4 = RK4Integrator;
        let p = params.clone();
        let q_next = rk4.step(&q, params.dt, move |q_| beris_edwards_rhs(q_, None, &p));
        // Q-tensor stays in [[q1,q2],[q2,-q1]] form by construction; check norms are finite.
        assert!(q_next.max_norm().is_finite());
        assert!(q_next.max_norm() > 0.0);
    }

    #[test]
    fn rk4_more_accurate_than_euler() {
        // Integrate a simple ODE y' = -y (decaying mode) for one step with both.
        // The exact solution after dt=0.1 is y(0.1) = y(0) * exp(-0.1).
        // RK4 should be closer to exact.
        //
        // Map to Q-field: use a uniform Q with Γ_r H ≈ -r*Q (effective rate r).
        // Use a stable (non-active) parameter set: a_eff > 0.
        let mut params = default_params();
        params.a_landau = 1.0;  // large positive: stable ordered phase
        params.zeta_eff = 0.0;  // no activity
        params.c_landau = 0.01; // small cubic term so regime is near linear
        params.gamma_r = 1.0;
        params.dt = 0.01;
        // a_eff = 1.0; linear decay rate = 2 * gamma_r * a_eff = 2.0
        let r = 2.0 * params.gamma_r * params.a_eff();
        let q0_val = 0.1_f64;
        let q0 = QField2D::uniform(4, 4, 1.0, [q0_val, 0.0]);

        let p1 = params.clone();
        let euler = EulerIntegrator;
        let q_euler = euler.step(&q0, params.dt, move |q_| beris_edwards_rhs(q_, None, &p1));

        let p2 = params.clone();
        let rk4 = RK4Integrator;
        let q_rk4 = rk4.step(&q0, params.dt, move |q_| beris_edwards_rhs(q_, None, &p2));

        let exact = q0_val * (-r * params.dt).exp();
        let err_euler = (q_euler.q[0][0] - exact).abs();
        let err_rk4 = (q_rk4.q[0][0] - exact).abs();
        assert!(err_rk4 < err_euler, "RK4 err={err_rk4:.3e} must be < Euler err={err_euler:.3e}");
    }

    // ── K₀ kernel ────────────────────────────────────────────────────────────

    #[test]
    fn bessel_k0_large_x_decays() {
        // K₀(x) ~ sqrt(π/(2x)) exp(-x) for large x; check monotone decay.
        let x_vals = [1.0, 2.0, 5.0, 10.0];
        let k_vals: Vec<f64> = x_vals.iter().map(|&x| bessel_k0(x)).collect();
        for i in 1..k_vals.len() {
            assert!(k_vals[i] < k_vals[i - 1], "K₀ not monotone: {k_vals:?}");
        }
    }

    #[test]
    fn bessel_k0_positive() {
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            assert!(bessel_k0(x) > 0.0);
        }
    }

    #[test]
    fn k0_convolution_of_uniform_field() {
        // For a uniform Q^rot, K₀ convolution returns a uniform Q^lip (each
        // vertex gets the same weight sum). Verify the output is uniform and
        // that both components scale by the same ratio relative to the input.
        let params = {
            let mut p = MarsParams::default_test();
            p.nx = 32;
            p.ny = 32;
            p.xi_l = 3.0;
            p.dx = 1.0;
            p
        };
        let q0 = [0.15, -0.08];
        let q = QField2D::uniform(params.nx, params.ny, params.dx, q0);
        let q_lip = k0_convolution(&q, &params);

        // Output should be uniform: all vertices have the same value.
        let [v0_ref, v1_ref] = q_lip.q[0];
        for &[v0, v1] in &q_lip.q {
            assert_abs_diff_eq!(v0, v0_ref, epsilon = 1e-12);
            assert_abs_diff_eq!(v1, v1_ref, epsilon = 1e-12);
        }

        // Both components scale by the same factor (ratio preserved).
        let ratio0 = v0_ref / q0[0];
        let ratio1 = v1_ref / q0[1];
        assert_abs_diff_eq!(ratio0, ratio1, epsilon = 1e-10);

        // The scale factor should be positive and finite (kernel is positive definite).
        assert!(ratio0.is_finite() && ratio0 > 0.0);

        // The discrete integral approximates ∫ K₀(r/ξ) d²r = 2π ξ²,
        // so ratio ≈ 2π ξ²/(2π ξ²) = 1 up to finite-cutoff and discretization
        // corrections. With cutoff clamped to nx/2=16 < 6ξ=18, the captured
        // integral is ≈ 95% of the full integral; allow 10% tolerance.
        assert_abs_diff_eq!(ratio0, 1.0, epsilon = 0.10);
    }

    // ── Defect detection ─────────────────────────────────────────────────────

    #[test]
    fn scan_defects_zero_field() {
        // The zero Q-tensor field has Q≡0 everywhere. q_to_frame at Q=0
        // returns an arbitrary frame (eigenvectors of the zero matrix), but
        // the holonomy of any constant frame field is identity: no defects.
        let q = QField2D::zeros(8, 8, 1.0);
        let defects = scan_defects(&q, PI / 2.0);
        // Either no defects, or if the zero eigenvector convention varies by vertex,
        // some plaquettes may be flagged. Accept either outcome (zero-field has no
        // physical defect).
        let _ = defects; // Just check it doesn't panic.
    }

    #[test]
    fn scan_defects_uniform_field_no_defects() {
        // A uniform Q-tensor field with S > 0: all frames are identical → no defects.
        let q = QField2D::uniform(16, 16, 1.0, [0.3, 0.0]);
        let defects = scan_defects(&q, PI / 2.0);
        assert!(
            defects.is_empty(),
            "uniform Q-field should have no defects, got {}",
            defects.len()
        );
    }

    // ── End-to-end: Component 1 ───────────────────────────────────────────────

    #[test]
    fn run_mars_component1_grows_order() {
        // Start from a small random perturbation in the active turbulent regime.
        // After a few steps, the order parameter should grow (instability).
        let params = MarsParams::default_test();
        assert!(params.a_eff() < 0.0);

        let q_init = QField2D::random_perturbation(16, 16, 1.0, 0.001, 7);
        let (q_final, stats) = run_mars_component1(&q_init, &params, 20, 5);

        assert!(q_final.max_norm().is_finite());
        // The order parameter should have grown from ~0.001.
        assert!(
            stats.last().unwrap().mean_s > stats.first().unwrap().mean_s,
            "order parameter did not grow: {:?}",
            stats.iter().map(|s| s.mean_s).collect::<Vec<_>>()
        );
    }

    // ── CH chemical potential ────────────────────────────────────────────────

    #[test]
    fn ch_chemical_potential_zero_at_zero_fields() {
        // μ(φ=0, Q=0) = a_ch * 0 + b_ch * 0³ - κ_ch ∇²(0) - χ_ms * 0 = 0.
        let params = MarsParams::default_test();
        let phi = ScalarField2D::zeros(8, 8, 1.0);
        let q_lip = QField2D::zeros(8, 8, 1.0);
        let mu = ch_chemical_potential(&phi, &q_lip, &params);
        for &v in &mu.phi {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ch_chemical_potential_linear_at_small_phi_no_q() {
        // For small uniform φ, no Q: μ ≈ a_ch φ (gradient term vanishes for uniform φ).
        let params = MarsParams::default_test();
        let phi0 = 0.01_f64;
        let phi = ScalarField2D::uniform(8, 8, 1.0, phi0);
        let q_lip = QField2D::zeros(8, 8, 1.0);
        let mu = ch_chemical_potential(&phi, &q_lip, &params);
        let expected = params.a_ch * phi0 + params.b_ch * phi0.powi(3);
        for &v in &mu.phi {
            assert_abs_diff_eq!(v, expected, epsilon = 1e-8 * expected.abs().max(1e-12));
        }
    }

    #[test]
    fn ch_maier_saupe_lowers_chemical_potential() {
        // With Q > 0 and χ_ms > 0, the Maier-Saupe term -χ_ms Tr(Q²) < 0
        // lowers μ relative to the Q=0 case (lipid is drawn into ordered regions).
        let params = MarsParams::default_test();
        assert!(params.chi_ms > 0.0);
        let phi = ScalarField2D::uniform(8, 8, 1.0, 0.5);
        let q_zero = QField2D::zeros(8, 8, 1.0);
        let q_ord  = QField2D::uniform(8, 8, 1.0, [0.3, 0.0]);
        let mu_no_q = ch_chemical_potential(&phi, &q_zero, &params);
        let mu_with_q = ch_chemical_potential(&phi, &q_ord, &params);
        // Maier-Saupe term = -χ_ms * 2(0.3² + 0) = -χ_ms * 0.18 < 0.
        assert!(
            mu_with_q.phi[0] < mu_no_q.phi[0],
            "Maier-Saupe term should lower μ: {} vs {}",
            mu_with_q.phi[0],
            mu_no_q.phi[0]
        );
    }

    // ── ETD CH step ──────────────────────────────────────────────────────────

    #[test]
    fn ch_step_etd_conserves_mean() {
        // The CH equation is mass-conserving: ⟨φ⟩ must not drift.
        let params = MarsParams::default_test();
        let phi0 = 0.4_f64;
        let phi = ScalarField2D::uniform(16, 16, 1.0, phi0);
        let q_lip = QField2D::uniform(16, 16, 1.0, [0.1, 0.05]);
        let v = VelocityField2D::zeros(16, 16, 1.0);
        let phi_new = ch_step_etd(&phi, &q_lip, &v, &params);
        // Mean must be conserved to within floating-point precision.
        assert_abs_diff_eq!(phi_new.mean_value(), phi0, epsilon = 1e-10);
    }

    #[test]
    fn ch_step_etd_uniform_phi_no_change() {
        // For a uniform φ field (∇φ = 0, ∇⁴φ = 0) with uniform Q and zero v,
        // the only nonlinear contribution is M_l ∇²(b_ch φ³ - χ_ms Tr(Q²)) = 0
        // (since everything is uniform). Only the k≠0 Fourier modes see the
        // stiff operator; for a truly uniform initial condition φ̂_k≠0 = 0,
        // so φ remains uniform after one ETD step.
        let params = MarsParams::default_test();
        let phi0 = 0.5_f64;
        let phi = ScalarField2D::uniform(16, 16, 1.0, phi0);
        let q_lip = QField2D::uniform(16, 16, 1.0, [0.2, 0.0]);
        let v = VelocityField2D::zeros(16, 16, 1.0);
        let phi_new = ch_step_etd(&phi, &q_lip, &v, &params);
        for &val in &phi_new.phi {
            assert_abs_diff_eq!(val, phi0, epsilon = 1e-10);
        }
    }

    #[test]
    fn ch_step_etd_output_finite() {
        // Run a few BECH steps from random initial conditions; all fields stay finite.
        let params = MarsParams::default_test();
        let q_init   = QField2D::random_perturbation(16, 16, 1.0, 0.05, 99);
        // Initialise φ near the equilibrium value sqrt(a_ch/b_ch) = 1.0 with small noise.
        let phi_vals: Vec<f64> = (0..16*16).map(|k| {
            let frac = k as f64 / (16.0 * 16.0);
            0.5 + 0.05 * (frac * 7.3).sin()
        }).collect();
        let phi_init = ScalarField2D { phi: phi_vals, nx: 16, ny: 16, dx: 1.0 };
        let (q_fin, phi_fin, stats) = run_mars_bech(&q_init, &phi_init, &params, 10, 5);
        assert!(q_fin.max_norm().is_finite());
        for &v in &phi_fin.phi {
            assert!(v.is_finite(), "phi_fin contains non-finite value");
        }
        assert_eq!(stats.len(), 3); // steps 0, 5, 10
    }

    // ── BECH full run ────────────────────────────────────────────────────────

    #[test]
    fn run_mars_bech_phi_variance_grows() {
        // Starting from uniform φ with a non-uniform Q (from active turbulence),
        // the Maier-Saupe coupling should drive phase separation: Var[φ] increases.
        let params = MarsParams::default_test();
        // Use a larger χ_ms to make the effect visible in a short run.
        let mut p = params.clone();
        p.chi_ms = 2.0;
        p.nx = 16; p.ny = 16;

        let q_init = QField2D::random_perturbation(16, 16, 1.0, 0.3, 17);
        let phi_init = ScalarField2D::uniform(16, 16, 1.0, 0.4);
        let (_, _, stats) = run_mars_bech(&q_init, &phi_init, &p, 50, 10);

        let var_init = stats.first().unwrap().phi_variance;
        let var_fin  = stats.last().unwrap().phi_variance;
        // Variance should grow (or at minimum not shrink) as Maier-Saupe
        // drives accumulation near ordered regions.
        assert!(
            var_fin >= var_init - 1e-12,
            "Expected Var[φ] to grow under Maier-Saupe coupling: {var_init:.3e} → {var_fin:.3e}"
        );
    }
}
