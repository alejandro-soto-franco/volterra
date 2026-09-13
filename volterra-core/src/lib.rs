// ~/volterra/volterra-core/src/lib.rs

//! # volterra-core
//!
//! Trait definitions, error types, and shared parameters for the volterra framework.
//!
//! ## Traits
//!
//! - [`Integrator`] -- a time-integration scheme (Euler, RK4, ...)
//!
//! ## Parameters
//!
//! - [`ActiveNematicParams`] -- dimensional parameters for the Cartesian solver
//! - [`NematicParams`] -- dimensionless parameters for the manifold engine
//!
//! ## Error type
//!
//! - [`VError`] -- unified error type for all volterra crates

/// Field types on Cartesian grids: the Q-tensor, velocity, pressure and scalar
/// fields, in two and three dimensions.
///
/// These were `volterra-fields` until they were folded in here. They are types
/// with no physics attached, which is what `volterra-core` is for, and keeping
/// them one crate away bought a dependency edge and nothing else.
pub mod fields;
pub use fields::*;

pub mod nematic_params;
pub use nematic_params::NematicParams;

pub mod sim;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Unified error type for all volterra crates.
#[derive(Debug, Error)]
pub enum VError {
    /// A field dimension does not match what was expected.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A numerical solver failed to converge.
    #[error("convergence failure after {iters} iterations (residual = {residual:.3e})")]
    ConvergenceFailure { iters: usize, residual: f64 },

    /// Invalid physical or numerical parameter.
    #[error("invalid parameter: {0}")]
    InvalidParams(String),

    /// I/O error (writing field snapshots, etc.).
    #[error("I/O error: {0}")]
    Io(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical parameters
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the single-phase active nematic
/// simulation and the coupled active nematic + concentration field system.
///
/// ## Dimensionless groups
///
/// Given these parameters the key dimensionless numbers are:
///
/// ```text
/// ℓ_d  = sqrt(K_r / zeta_eff)        defect length scale
/// Da   = Gamma_r * eta / zeta_eff     Damköhler number
/// Sp   = K_r / (Gamma_l * eta * K_l) existence condition (< 1 for coherent transfer)
/// ```
///
/// The coherent transfer window is zeta_one < zeta_eff < zeta_star.
/// Hele-Shaw wall drag for a chamber of finite depth.
///
/// Depth-averaging Stokes over a gap of height `h` with no-slip on both faces
/// leaves a drag `12 eta / h^2` on the depth-averaged velocity. Taking the curl
/// puts it on the vorticity, so the surface biharmonic gains one factor and the
/// screening length is `l_s = h / sqrt(12)`.
///
/// `None` is the default and reproduces every result computed before this type
/// existed. It is the setting in which the operator matches the one the
/// 2026-08-19 correction established.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum Screening {
    /// An unbounded-depth two-dimensional fluid, with no wall drag.
    #[default]
    None,
    /// A chamber of finite depth, with the screening length `l_s` in the same
    /// units as the mesh coordinates.
    Length(f64),
}

impl Screening {
    /// From a chamber depth `h`, in the same units as the mesh coordinates.
    pub fn from_depth(h: f64) -> Self {
        Self::Length(h / 12.0_f64.sqrt())
    }

    /// `1 / l_s^2`, which is the magnitude of the shift the Poisson solver
    /// takes, and zero where there is no screening.
    ///
    /// A non-positive length answers infinity rather than a number. Squaring
    /// alone would turn `Length(-0.5)` into the shift of `Length(0.5)` and
    /// accept it silently; a screening length is a length.
    pub fn inverse_square(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Length(l) if *l > 0.0 => 1.0 / (l * l),
            Self::Length(_) => f64::INFINITY,
        }
    }

    /// Whether this states a usable chamber depth.
    pub fn is_valid(&self) -> bool {
        match self {
            Self::None => true,
            Self::Length(l) => *l > 0.0 && l.is_finite(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveNematicParams {
    // ── Grid ──────────────────────────────────────────────────────────────
    /// Number of grid vertices in x.
    pub nx: usize,
    /// Number of grid vertices in y.
    pub ny: usize,
    /// Grid spacing (same in x and y; periodic boundary conditions assumed).
    pub dx: f64,
    /// Time step.
    pub dt: f64,

    // ── Rotor phase ────────────────────────────────────────────────────────
    /// Rotor Frank elastic constant K_r (one-constant approximation).
    pub k_r: f64,
    /// Rotor rotational viscosity Γ_r (collective relaxation rate = 1/Γ_r).
    pub gamma_r: f64,
    /// Effective activity ζ_eff = ζ₀ B₀² ω_B τ_r / (1 + (ω_B τ_r)²).
    /// Controls defect density: ρ_d ~ ζ_eff / K_r.
    pub zeta_eff: f64,
    /// Fluid viscosity η.
    pub eta: f64,
    /// Hele-Shaw wall drag from the chamber depth. Absent by default.
    #[serde(default)]
    pub screening: Screening,
    /// Landau coefficient a (< 0 for the ordered nematic without activity).
    /// Effective driving is a_eff = a - zeta_eff/2.
    pub a_landau: f64,
    /// Landau coefficient c > 0 (stabilises large |Q|).
    pub c_landau: f64,
    /// Flow alignment parameter λ (tumbling vs. flow-aligning nematics).
    /// λ = 1.0 for flow-aligning; |λ| < 1 for tumbling.
    pub lambda: f64,

    // ── Langevin noise ────────────────────────────────────────────────────
    /// RMS amplitude of the Langevin noise term added at each time step.
    ///
    /// The noise is applied as `Q += noise_amp * sqrt(dt) * W` where `W` is
    /// an i.i.d. standard Gaussian per component per grid vertex.  This models
    /// the stochastic reorientation fluctuations of the active rod ensemble.
    /// Set to 0.0 (default) to disable noise.
    pub noise_amp: f64,

    // ── Lipid phase (Component 2: one-way coupling) ───────────────────────
    /// Lipid Frank elastic constant K_l.
    pub k_l: f64,
    /// Lipid rotational viscosity Γ_l.
    pub gamma_l: f64,
    /// Lipid coupling length ξ_l (range of the K₀ response kernel).
    /// Determines the concentration-field response radius.
    pub xi_l: f64,

    // ── Cahn-Hilliard / Maier-Saupe (BECH: full two-field coupling) ───────
    /// Maier-Saupe coupling constant χ_MS (dimensionless in simulation units).
    ///
    /// Drives lipid accumulation in regions of high orientational order:
    /// f_MS = -χ_MS φ_l Tr(Q_lip²).  A positive χ_MS > 0 is required for
    /// the orientational-concentration coupling to template phase separation.
    pub chi_ms: f64,
    /// Cahn-Hilliard gradient energy coefficient κ_l (simulation units of K_r dx²).
    ///
    /// Controls the CH coherence length ξ_CH = sqrt(κ_l / a_ch) at which
    /// the gradient penalty balances the bulk driving.  At physical scale
    /// ξ_CH ~ 1--5 nm, far below ℓ_d ~ 50--200 nm, justifying the UCA as
    /// a leading-order approximation when the BECH is not run.
    pub kappa_ch: f64,
    /// CH bulk quadratic coefficient a_l > 0.
    ///
    /// In the double-well free energy F^CH = a_l φ²/2 + b_l φ⁴/4, a_l sets
    /// the curvature at the disordered minimum φ = 0.  Together with b_l it
    /// gives the equilibrium lipid fraction φ_eq = sqrt(a_l / b_l).
    pub a_ch: f64,
    /// CH bulk quartic coefficient b_l > 0.
    ///
    /// Stabilises large |φ_l| against unbounded growth.  With a_ch, gives
    /// φ_eq = sqrt(a_ch / b_ch) and interfacial width ~ ξ_CH = sqrt(κ_ch / a_ch).
    pub b_ch: f64,
    /// Cahn-Hilliard mobility M_l > 0.
    ///
    /// Sets the timescale τ_CH = ξ_CH² / (M_l a_ch) for concentration
    /// relaxation.  The stability criterion for the ETD integrator requires
    /// M_l κ_ch k_max⁴ Δt < 1 (automatically satisfied for the stiff
    /// exponential integrator; the explicit Euler limit would require
    /// Δt < 1/(M_l κ_ch k_max⁴)).
    pub m_l: f64,

    // ── Optional spatial activity ─────────────────────────────────────────
    /// Optional per-vertex active coefficient field ζ(x), row-major `i*ny + j`,
    /// length `nx*ny`. When present it overrides the scalar [`zeta_eff`] in the
    /// active stress `σ = ζ(x) Q(x)`; when `None` the scalar `zeta_eff` is used
    /// uniformly. A spatial field lets a contact-driven conversion front cross
    /// the activity threshold in space (the saddle-node of the intermittency
    /// prediction), which a single scalar cannot represent.
    ///
    /// [`zeta_eff`]: Self::zeta_eff
    #[serde(default)]
    pub zeta_field: Option<Vec<f64>>,
}

impl ActiveNematicParams {
    /// Active coefficient at vertex `i`: the spatial field value if a
    /// [`zeta_field`] is set, otherwise the scalar `zeta_eff`.
    ///
    /// [`zeta_field`]: Self::zeta_field
    pub fn zeta_at(&self, i: usize) -> f64 {
        match &self.zeta_field {
            Some(f) => f[i],
            None => self.zeta_eff,
        }
    }

    /// Defect length scale ℓ_d = sqrt(K_r / ζ_eff).
    ///
    /// This equals the mean rotor defect spacing.
    pub fn defect_length(&self) -> f64 {
        (self.k_r / self.zeta_eff).sqrt()
    }

    /// Dimensionless existence condition Π = K_r / (Γ_l η K_l).
    ///
    /// Coherent transfer requires Π < 1.
    pub fn pi_number(&self) -> f64 {
        self.k_r / (self.gamma_l * self.eta * self.k_l)
    }

    /// Effective Landau parameter a_eff = a_landau - zeta_eff / 2.
    ///
    /// When a_eff < 0 the system is in the active turbulent (defect-laden) phase.
    pub fn a_eff(&self) -> f64 {
        self.a_landau - self.zeta_eff / 2.0
    }

    /// Cahn-Hilliard coherence length ξ_CH = sqrt(κ_ch / a_ch).
    ///
    /// At physical scale ξ_CH ~ 1--5 nm, far below the defect length
    /// ℓ_d ~ 50--200 nm, which justifies the Uniform Concentration
    /// Approximation (UCA) as a leading-order limit of the full BECH.
    pub fn ch_coherence_length(&self) -> f64 {
        (self.kappa_ch / self.a_ch).sqrt()
    }

    /// Equilibrium lipid fraction φ_eq = sqrt(a_ch / b_ch).
    ///
    /// The double-well free energy F^CH = a_ch φ²/2 + b_ch φ⁴/4 has minima
    /// at φ = 0 and φ = φ_eq; the system phase-separates toward φ_eq in
    /// regions of strong Maier-Saupe coupling.
    pub fn phi_eq(&self) -> f64 {
        (self.a_ch / self.b_ch).sqrt()
    }

    /// Validate that parameters are physically reasonable.
    pub fn validate(&self) -> Result<(), VError> {
        if self.nx < 2 {
            return Err(VError::InvalidParams("nx must be >= 2".into()));
        }
        if self.ny < 2 {
            return Err(VError::InvalidParams("ny must be >= 2".into()));
        }
        if self.dx <= 0.0 {
            return Err(VError::InvalidParams("dx must be positive".into()));
        }
        if self.dt <= 0.0 {
            return Err(VError::InvalidParams("dt must be positive".into()));
        }
        if self.k_r <= 0.0 {
            return Err(VError::InvalidParams("k_r must be positive".into()));
        }
        if self.gamma_r <= 0.0 {
            return Err(VError::InvalidParams("gamma_r must be positive".into()));
        }
        if self.zeta_eff < 0.0 {
            return Err(VError::InvalidParams(
                "zeta_eff must be non-negative".into(),
            ));
        }
        if self.eta <= 0.0 {
            return Err(VError::InvalidParams("eta must be positive".into()));
        }
        if self.c_landau <= 0.0 {
            return Err(VError::InvalidParams("c_landau must be positive".into()));
        }
        if self.xi_l <= 0.0 {
            return Err(VError::InvalidParams("xi_l must be positive".into()));
        }
        if self.noise_amp < 0.0 {
            return Err(VError::InvalidParams(
                "noise_amp must be non-negative".into(),
            ));
        }
        if self.chi_ms < 0.0 {
            return Err(VError::InvalidParams("chi_ms must be non-negative".into()));
        }
        if self.kappa_ch <= 0.0 {
            return Err(VError::InvalidParams("kappa_ch must be positive".into()));
        }
        if self.a_ch <= 0.0 {
            return Err(VError::InvalidParams("a_ch must be positive".into()));
        }
        if self.b_ch <= 0.0 {
            return Err(VError::InvalidParams("b_ch must be positive".into()));
        }
        if self.m_l <= 0.0 {
            return Err(VError::InvalidParams("m_l must be positive".into()));
        }
        if let Some(f) = &self.zeta_field {
            if f.len() != self.nx * self.ny {
                return Err(VError::InvalidParams(format!(
                    "zeta_field length {} must equal nx*ny = {}",
                    f.len(),
                    self.nx * self.ny
                )));
            }
        }
        if !self.screening.is_valid() {
            return Err(VError::InvalidParams(format!(
                "screening length must be positive and finite, got {:?}",
                self.screening
            )));
        }
        Ok(())
    }

    /// Construct a minimal default parameter set useful for testing.
    ///
    /// Grid: 64x64, dx=1.0, dt=0.01.
    /// Physics: K_r=1, Γ_r=1, ζ_eff=2 (active turbulent), η=1.
    ///
    /// CH parameters are set to physically plausible dimensionless values:
    /// χ_MS=0.5, κ_ch=1.0, a_ch=1.0, b_ch=1.0 (φ_eq = 1.0), M_l=0.1.
    /// The CH coherence length ξ_CH = sqrt(κ_ch/a_ch) = 1.0 (one grid cell),
    /// which is << ℓ_d = sqrt(K_r/ζ_eff) = 0.71 in the default active state,
    /// consistent with the UCA scale-separation argument.
    pub fn default_test() -> Self {
        Self {
            nx: 64,
            ny: 64,
            dx: 1.0,
            dt: 0.01,
            k_r: 1.0,
            gamma_r: 1.0,
            zeta_eff: 2.0,
            eta: 1.0,
            screening: Screening::None,
            a_landau: -0.5,
            c_landau: 4.5,
            lambda: 0.7,
            noise_amp: 0.0,
            k_l: 0.5,
            gamma_l: 1.0,
            xi_l: 5.0,
            chi_ms: 0.5,
            kappa_ch: 1.0,
            a_ch: 1.0,
            b_ch: 1.0,
            m_l: 0.1,
            zeta_field: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the 3D active nematic + concentration field simulation.
///
/// ## Symbol conventions
///
/// `lambda` in this struct is the flow-alignment parameter xi from Jeffery orbit
/// theory (eq. xi_flow in the paper): xi = (r^2-1)/(r^2+1). Named `lambda` in
/// the struct to match the 2D ActiveNematicParams convention; used as `xi` in all physics docs.
///
/// `chi_a` encodes mu_0 * Delta_chi / 2 (SI). The magnetic torque molecular field
/// H_mag = chi_a * b0^2 * [...]. Do NOT multiply by gamma_r inside molecular_field_3d;
/// the single Gamma_r multiplication occurs in beris_edwards_rhs_3d.
///
/// `epsilon_a`, `e0` and `omega_e` are the electric counterpart, and enter the
/// molecular field the same way: both fields couple quadratically to a
/// direction, so both contribute a traceless rank-two term
/// `coefficient * amplitude^2 * [d (x) d - I/3]`. `epsilon_a` encodes
/// `eps_0 * Delta_eps / 2`, matching how `chi_a` encodes `mu_0 * Delta_chi / 2`.
/// A zero amplitude removes the term identically, so a run that sets neither
/// field is unchanged to the last bit.
///
/// `omega_b` and `omega_e` rotate their field in the xy plane; zero leaves it
/// static along x. The rotating case is the MARS actuation, where a suspension
/// of magnetic rods is driven at `omega_b` past its rotational relaxation rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveNematicParams3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub k_r: f64,
    pub gamma_r: f64,
    pub zeta_eff: f64,
    pub eta: f64,
    pub a_landau: f64,
    pub c_landau: f64,
    pub b_landau: f64,
    pub lambda: f64,
    pub noise_amp: f64,
    pub chi_a: f64,
    pub b0: f64,
    pub omega_b: f64,
    /// Dielectric anisotropy, encoding `eps_0 * Delta_eps / 2`.
    pub epsilon_a: f64,
    /// Electric field amplitude. Zero removes the electric term identically.
    pub e0: f64,
    /// Electric field rotation rate in the xy plane; zero holds it along x.
    pub omega_e: f64,
    pub k_l: f64,
    pub gamma_l: f64,
    pub xi_l: f64,
    pub chi_ms: f64,
    pub kappa_ch: f64,
    pub a_ch: f64,
    pub b_ch: f64,
    pub m_l: f64,

    /// Spontaneous curvature c₀ [m⁻¹]. Negative for QII-forming lipids.
    /// Modifies the interface bending penalty: κ_eff = κ_CH - κ_W · c₀/ε_CH.
    /// Typical value: -1/ξ_CH ≈ -3.3e8 m⁻¹ for ξ_CH = 3 nm.
    /// Set to 0.0 to recover the plain CH free energy.
    #[serde(default)]
    pub c0_sp: f64,

    /// Curvature penalty coefficient κ_W [J/m³] ≥ 0.
    /// Scales the Willmore-like square term in the enriched CH free energy.
    /// Set to 0.0 to disable the curvature coupling.
    #[serde(default)]
    pub kappa_w: f64,

    /// Gaussian curvature modulus κ̄_G [J/m]. May be negative.
    /// Treated EXPLICITLY in the nonlinear ETD part (never added to L).
    /// Timestep bound when |κ̄_G| > 0: dt < C ε³ / (M_l |κ̄_G| k_max⁶).
    /// Set to 0.0 to disable Gaussian curvature coupling.
    #[serde(default)]
    pub kappa_bar_g: f64,

    /// Interface half-width ε_CH \[m\] > 0.
    /// Sets the gradient interface scale for the enriched free energy.
    /// Rule of thumb: ε_CH = dx (one grid spacing) for unit tests;
    /// physical value ≈ ξ_CH = 3 nm for a 1 nm grid.
    #[serde(default = "default_epsilon_ch")]
    pub epsilon_ch: f64,

    /// Fraction of the interior peak of the disclination density taken as the
    /// isosurface the lines are read off.
    ///
    /// `s` scales as the square of a Q gradient, so it depends on the field's
    /// normalisation and on the grid spacing, and an absolute value transfers
    /// between runs no better than a Reynolds number does between fluids. A
    /// fraction of the peak transfers, and the value used is recorded in every
    /// snapshot's statistics so a run states its own threshold.
    ///
    /// The reading is relative, so a field with no disclination in it still
    /// returns whatever its largest gradient is. Read the count alongside the
    /// threshold, which falls by orders of magnitude when the defects go.
    #[serde(default = "default_disclination_threshold_fraction")]
    pub disclination_threshold_fraction: f64,

    /// Absolute floor on that threshold, below which the answer is that there
    /// are no disclinations.
    ///
    /// The fraction alone reports lines in the noise of a field that has none,
    /// since `s` has a largest value wherever the field is. A 32^3 dry run
    /// watched its threshold fall from 6.3e-3 to 3.6e-9 as its defects
    /// annihilated, while the order parameter sat at 0.499 and the line count
    /// stayed near 38. The threshold that is used is the larger of this and the
    /// fraction of the peak.
    ///
    /// `None`, the default, scales one from the field's own equilibrium and the
    /// grid spacing; see [`disclination_floor`](Self::disclination_floor).
    /// `Some(0.0)` leaves the relative rule alone.
    #[serde(default)]
    pub disclination_threshold_floor: Option<f64>,
}

/// The default floor, as a multiple of `(q_eq / dx)^2`.
///
/// A resolved `+1/2` core reads `0.647 (q_eq / dx)^2`, measured, and the same
/// multiple at every spacing, since `s` is quadratic in a Q gradient and a core
/// turns the director through a fixed angle over a lattice spacing. An ordered
/// field's numerical noise sat at `1.8e-7` of that scale. This coefficient is
/// near the middle of the two in the logarithm: about 2.8 decades below a core
/// and 3.8 above the noise.
pub const DEFAULT_DISCLINATION_FLOOR_COEFFICIENT: f64 = 1e-3;

fn default_disclination_threshold_fraction() -> f64 {
    0.25
}

fn default_epsilon_ch() -> f64 {
    1.0
}

impl ActiveNematicParams3D {
    /// Defect length scale ℓ_d = sqrt(K_r / ζ_eff).
    pub fn defect_length(&self) -> f64 {
        (self.k_r / self.zeta_eff).sqrt()
    }

    /// Dimensionless existence condition Π = K_r / (Γ_l η K_l).
    pub fn pi_number(&self) -> f64 {
        self.k_r / (self.gamma_l * self.eta * self.k_l)
    }

    /// Effective Landau parameter a_eff = a_landau - zeta_eff / 2.
    pub fn a_eff(&self) -> f64 {
        self.a_landau - self.zeta_eff / 2.0
    }

    /// The equilibrium scalar order parameter, the positive root of
    /// `6a + 3b q + 8c q^2 = 0` in the `Q = q (nn - I/3)` convention.
    ///
    /// Zero where the quartic has no ordered minimum, which is the isotropic
    /// state and has no disclination in it to find.
    pub fn equilibrium_q(&self) -> f64 {
        let disc = 9.0 * self.b_landau * self.b_landau - 192.0 * self.a_landau * self.c_landau;
        if disc < 0.0 || self.c_landau == 0.0 {
            return 0.0;
        }
        (-3.0 * self.b_landau + disc.sqrt()) / (16.0 * self.c_landau)
    }

    /// The floor on the disclination threshold this run should use.
    ///
    /// An explicit [`disclination_threshold_floor`] where one is set, and
    /// otherwise [`DEFAULT_DISCLINATION_FLOOR_COEFFICIENT`] times
    /// `(q_eq / dx)^2`. Scaling it that way is what makes the default mean the
    /// same thing on a different grid or at a different normalisation, where a
    /// bare number would silently mean something else.
    ///
    /// [`disclination_threshold_floor`]: Self::disclination_threshold_floor
    pub fn disclination_floor(&self) -> f64 {
        self.disclination_threshold_floor.unwrap_or_else(|| {
            DEFAULT_DISCLINATION_FLOOR_COEFFICIENT * (self.equilibrium_q() / self.dx).powi(2)
        })
    }

    /// Cahn-Hilliard coherence length ξ_CH = sqrt(κ_ch / a_ch).
    pub fn ch_coherence_length(&self) -> f64 {
        (self.kappa_ch / self.a_ch).sqrt()
    }

    /// Equilibrium lipid fraction φ_eq = sqrt(a_ch / b_ch).
    pub fn phi_eq(&self) -> f64 {
        (self.a_ch / self.b_ch).sqrt()
    }

    /// Effective bending stiffness κ_eff = κ_CH - κ_W · c₀_sp / ε_CH.
    ///
    /// For c₀_sp < 0 (QII-forming lipids), κ_eff > κ_CH (stiffer interface).
    /// This value replaces κ_CH as the stiff linear coefficient in the
    /// enriched ETD step.
    ///
    /// Returns κ_CH unchanged when kappa_w = 0 or epsilon_ch is not yet set.
    pub fn kappa_eff(&self) -> f64 {
        if self.epsilon_ch > 0.0 {
            self.kappa_ch - self.kappa_w * self.c0_sp / self.epsilon_ch
        } else {
            self.kappa_ch
        }
    }

    /// Validate that parameters are physically reasonable.
    pub fn validate(&self) -> Result<(), VError> {
        if self.nx < 2 {
            return Err(VError::InvalidParams("nx must be >= 2".into()));
        }
        if self.ny < 2 {
            return Err(VError::InvalidParams("ny must be >= 2".into()));
        }
        if self.nz < 2 {
            return Err(VError::InvalidParams("nz must be >= 2".into()));
        }
        if self.dx <= 0.0 {
            return Err(VError::InvalidParams("dx must be positive".into()));
        }
        if self.dt <= 0.0 {
            return Err(VError::InvalidParams("dt must be positive".into()));
        }
        if self.k_r <= 0.0 {
            return Err(VError::InvalidParams("k_r must be positive".into()));
        }
        if self.gamma_r <= 0.0 {
            return Err(VError::InvalidParams("gamma_r must be positive".into()));
        }
        if self.zeta_eff < 0.0 {
            return Err(VError::InvalidParams(
                "zeta_eff must be non-negative".into(),
            ));
        }
        if self.eta <= 0.0 {
            return Err(VError::InvalidParams("eta must be positive".into()));
        }
        if self.c_landau <= 0.0 {
            return Err(VError::InvalidParams("c_landau must be positive".into()));
        }
        if self.noise_amp < 0.0 {
            return Err(VError::InvalidParams(
                "noise_amp must be non-negative".into(),
            ));
        }
        if self.chi_a < 0.0 {
            return Err(VError::InvalidParams("chi_a must be non-negative".into()));
        }
        if self.b0 < 0.0 {
            return Err(VError::InvalidParams("b0 must be non-negative".into()));
        }
        if self.e0 < 0.0 {
            return Err(VError::InvalidParams("e0 must be non-negative".into()));
        }
        if self.k_l <= 0.0 {
            return Err(VError::InvalidParams("k_l must be positive".into()));
        }
        if self.gamma_l <= 0.0 {
            return Err(VError::InvalidParams("gamma_l must be positive".into()));
        }
        if self.xi_l <= 0.0 {
            return Err(VError::InvalidParams("xi_l must be positive".into()));
        }
        if self.chi_ms < 0.0 {
            return Err(VError::InvalidParams("chi_ms must be non-negative".into()));
        }
        if self.kappa_ch <= 0.0 {
            return Err(VError::InvalidParams("kappa_ch must be positive".into()));
        }
        if self.a_ch <= 0.0 {
            return Err(VError::InvalidParams("a_ch must be positive".into()));
        }
        if self.b_ch <= 0.0 {
            return Err(VError::InvalidParams("b_ch must be positive".into()));
        }
        if self.m_l <= 0.0 {
            return Err(VError::InvalidParams("m_l must be positive".into()));
        }
        if self.kappa_w < 0.0 {
            return Err(VError::InvalidParams("kappa_w must be non-negative".into()));
        }
        if self.epsilon_ch <= 0.0 {
            return Err(VError::InvalidParams("epsilon_ch must be positive".into()));
        }
        // κ̄_G stability bound: dt < 0.1 ε³ / (M_l |κ̄_G| k_max⁶)
        // where k_max = π/dx. Violated timesteps will cause blow-up.
        if self.kappa_bar_g != 0.0 && self.m_l > 0.0 && self.epsilon_ch > 0.0 {
            let k_max = std::f64::consts::PI / self.dx;
            let dt_max =
                0.1 * self.epsilon_ch.powi(3) / (self.m_l * self.kappa_bar_g.abs() * k_max.powi(6));
            if self.dt > dt_max {
                return Err(VError::InvalidParams(format!(
                    "dt={:.3e} exceeds κ̄_G ETD stability bound dt_max={:.3e}; \
                     reduce dt or reduce |κ̄_G|",
                    self.dt, dt_max
                )));
            }
        }
        Ok(())
    }

    /// Default parameter set for testing: 16x16x16 grid, active turbulent phase.
    pub fn default_test() -> Self {
        Self {
            nx: 16,
            ny: 16,
            nz: 16,
            dx: 1.0,
            dt: 0.01,
            k_r: 1.0,
            gamma_r: 1.0,
            zeta_eff: 2.0,
            eta: 1.0,
            a_landau: -0.5,
            c_landau: 4.5,
            b_landau: 0.0,
            lambda: 0.95,
            noise_amp: 0.0,
            chi_a: 0.0,
            b0: 1.0,
            omega_b: 1.0,
            epsilon_a: 0.0,
            e0: 0.0,
            omega_e: 0.0,
            k_l: 0.5,
            gamma_l: 1.0,
            xi_l: 5.0,
            chi_ms: 0.5,
            kappa_ch: 1.0,
            a_ch: 1.0,
            b_ch: 1.0,
            m_l: 0.1,
            c0_sp: 0.0,
            kappa_w: 0.0,
            kappa_bar_g: 0.0,
            epsilon_ch: 1.0, // = dx for unit tests
            disclination_threshold_fraction: 0.25,
            disclination_threshold_floor: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integrator trait
// ─────────────────────────────────────────────────────────────────────────────

/// A time integration scheme for the Beris-Edwards equation.
///
/// The generic parameter `S` is the full simulation state (e.g., a struct
/// containing the Q-tensor field and any other evolving quantities).
pub trait Integrator<S> {
    /// Advance `state` by one time step `dt`, given the RHS function `rhs`.
    ///
    /// `rhs(state) -> dstate/dt` should return the time derivative in the
    /// same representation as `state`.
    fn step<F>(&self, state: &S, dt: f64, rhs: F) -> S
    where
        F: Fn(&S) -> S;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_3d {
    use super::*;

    #[test]
    fn test_params_3d_validate_ok() {
        let p = ActiveNematicParams3D::default_test();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_params_3d_defect_length() {
        let p = ActiveNematicParams3D::default_test();
        let ld = p.defect_length();
        assert!(ld > 0.0, "defect_length must be positive");
        // ld = sqrt(k_r / zeta_eff) = sqrt(1/2) ~ 0.707
        assert!((ld - (0.5f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_params_3d_invalid_nz() {
        let mut p = ActiveNematicParams3D::default_test();
        p.nz = 0;
        assert!(p.validate().is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_test_params_valid() {
        let p = ActiveNematicParams::default_test();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn defect_length_correct() {
        let mut p = ActiveNematicParams::default_test();
        p.k_r = 4.0;
        p.zeta_eff = 1.0;
        assert!((p.defect_length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_eff_correct() {
        let mut p = ActiveNematicParams::default_test();
        p.a_landau = -1.0;
        p.zeta_eff = 3.0;
        // a_eff = -1.0 - 1.5 = -2.5
        assert!((p.a_eff() - (-2.5)).abs() < 1e-12);
    }

    #[test]
    fn screening_is_absent_by_default_and_hele_shaw_when_set() {
        assert_eq!(Screening::default(), Screening::None);
        assert_eq!(Screening::None.inverse_square(), 0.0);
        assert_eq!(
            ActiveNematicParams::default_test().screening,
            Screening::None
        );

        // `l_s = h / sqrt(12)`, so `1 / l_s^2 = 12 / h^2`.
        let h = 50e-3_f64;
        let want = 12.0 / (h * h);
        let got = Screening::from_depth(h).inverse_square();
        assert!((got - want).abs() / want < 1e-12, "got {got}, want {want}");

        // A negative length squares to look like a positive one, so it has to
        // be rejected by value rather than by its square.
        assert!(!Screening::Length(-0.5).is_valid());
        assert!(!Screening::Length(0.0).is_valid());
        assert!(!Screening::Length(f64::INFINITY).is_valid());
        assert!(Screening::Length(0.5).is_valid());
        assert_eq!(Screening::Length(-0.5).inverse_square(), f64::INFINITY);
        let mut bad = ActiveNematicParams::default_test();
        bad.screening = Screening::Length(-0.5);
        assert!(
            bad.validate().is_err(),
            "validate must reject a negative screening length"
        );
        bad.screening = Screening::Length(0.5);
        assert!(bad.validate().is_ok());

        // A params file written before this field existed still loads. The case
        // is built by serialising the current struct and removing the one key,
        // so it states the property without restating a field list that would
        // rot beside the struct.
        let mut v: serde_json::Value =
            serde_json::to_value(ActiveNematicParams::default_test()).unwrap();
        v.as_object_mut()
            .unwrap()
            .remove("screening")
            .expect("the key is present to remove");
        let back: ActiveNematicParams = serde_json::from_value(v).unwrap();
        assert_eq!(back.screening, Screening::None);
    }

    #[test]
    fn invalid_params_caught() {
        let mut p = ActiveNematicParams::default_test();
        p.dx = -1.0;
        assert!(p.validate().is_err());
        p.dx = 1.0;
        p.k_r = 0.0;
        assert!(p.validate().is_err());
    }
}

#[cfg(test)]
mod tests_approach_b {
    use super::*;

    #[test]
    fn test_new_params_present() {
        let p = ActiveNematicParams3D::default_test();
        assert!(p.kappa_w >= 0.0);
        assert!(p.epsilon_ch > 0.0);
    }

    #[test]
    fn test_validate_rejects_negative_kappa_w() {
        let mut p = ActiveNematicParams3D::default_test();
        p.kappa_w = -1.0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_epsilon_ch() {
        let mut p = ActiveNematicParams3D::default_test();
        p.epsilon_ch = 0.0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_kappa_eff() {
        let mut p = ActiveNematicParams3D::default_test();
        p.kappa_ch = 1.0;
        p.kappa_w = 2.0;
        p.c0_sp = -0.5;
        p.epsilon_ch = 1.0;
        // κ_eff = κ_CH - κ_W * c0_sp / ε_CH = 1.0 - 2.0*(-0.5)/1.0 = 2.0
        assert!((p.kappa_eff() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_validate_rejects_dt_violating_kappa_bar_bound() {
        let mut p = ActiveNematicParams3D::default_test();
        p.kappa_bar_g = 1e4; // large |κ̄_G| → tiny dt_max
        p.dt = 1.0; // grossly too large
        p.epsilon_ch = 1.0;
        assert!(
            p.validate().is_err(),
            "large kappa_bar_g + large dt must fail validate"
        );
    }

    #[test]
    fn test_validate_accepts_safe_dt_for_kappa_bar() {
        let mut p = ActiveNematicParams3D::default_test();
        p.kappa_bar_g = 0.0; // disabled
        assert!(p.validate().is_ok(), "kappa_bar_g=0 must always pass");
    }
}
