#![allow(clippy::needless_range_loop)]
// Indexed loops are the clearer form in stencil code, where one index reads
// several arrays at once. `volterra-solver` had this allow at its crate
// root, and the physics that moved here came with it.

//! `volterra-fd`: the finite-difference discretisation of Beris-Edwards with a
//! relaxation Stokes solve, on confined Cartesian domains.
//!
//! Named for the method rather than a dimension, alongside `volterra-dec`,
//! which is the discrete-exterior-calculus discretisation of the same physics.
//! What is implemented here today is two-dimensional; the three-dimensional
//! finite-difference kernels live in `volterra-solver` for now
//! (`mol_field_3d`), and belong here if those are ever consolidated.
//!
//! This crate is a complete Rust port of the solver in
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`. Every kernel matches the
//! Python numba reference to rounding (the golden-data tests under `tests/ref/`
//! are the contract). It provides the simulation parameters, the confinement
//! boundary constructions, the physics kernels, and a hardened run harness.

pub mod boundary;
pub use boundary::{
    Boundary, Epitrochoid, cardioid_boundary, circular_boundary, epitrochoid_boundary,
    nephroid_boundary, periodic_boundary, trefoiloid_boundary,
};

pub mod index;

pub mod par_gate;

pub mod regime;
pub use regime::{Regime, RegimeConstants, classify, melted_fraction, topological_defects};
pub mod ops;
pub mod nematic;
pub mod stokes;
pub mod bc;
pub mod step;

pub mod error;
pub use error::{FdError, FdResult};

pub mod guard;

pub mod sim_step;
pub mod output;

pub mod ic;
pub use ic::{SeededDefect, mitchell_figure_2a, mitchell_four_defect, seeded_q};

pub mod locking;
pub use locking::{Locking, RotationRates, add_fracture_switch, rotation_rates};

pub mod stretching;
pub use stretching::{MaterialLine, StretchFit};

/// Simulation parameters, matching the Python solver's constants.
///
/// Constructor: [`Params::new`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Params {
    // Grid
    pub lx: usize,
    pub ly: usize,

    // Time stepping
    pub dt: f64,

    // Frank elastic constant  (Python: k_elastic = 2^14)
    pub k_elastic: f64,

    // Rotational viscosity  (Python: gamma)
    pub gamma: f64,

    // Shear viscosity  (Python: eta = sqrt(10 * k_elastic))
    pub eta: f64,

    // Fluid density  (Python: rho)
    pub rho: f64,

    // Flow-alignment coupling  (Python: chi)
    pub chi: f64,

    // Active stress magnitude  (Python: zeta = k_elastic / als^2)
    pub zeta: f64,

    // Landau free-energy coefficients
    // a_landau = -c_landau  (Python convention)
    pub a_landau: f64,
    pub c_landau: f64,

    // Equilibrium scalar order parameter  (Python: S0 = sqrt(2))
    pub s0: f64,

    // Flow-alignment parameter  (Python: lambda)
    pub lambda: f64,

    // Maximum pressure-Poisson iterations per step (negative = uncapped, code-truth default)
    pub max_p_iters: i64,

    // Total topological charge q the boundary condition imposes on the
    // interior (Python: net_charge in apply_Q_boundary_conditions).
    // Defaults to 1.0, matching the crate's original hardcoded value.
    pub net_charge: f64,

    /// Enhanced nematic locking (arXiv:2506.20996). `None` is the standard
    /// Beris-Edwards model and is what every constructor here returns, so a run
    /// made before this field existed is reproduced bit for bit. See
    /// [`crate::locking`].
    #[serde(default)]
    pub locking: Option<Locking>,

    /// Which elastic stress enters the Navier-Stokes force.
    ///
    /// `#[serde(default)]` and [`StressModel::Full`] everywhere, so a run made
    /// before this field existed is reproduced bit for bit.
    #[serde(default)]
    pub stress: StressModel,
}

/// The elastic stress the force density is taken from.
///
/// The two papers this crate reproduces use different stresses, and each says so
/// in its own equations.
///
/// Klein, Soto Franco, Mitchell and Beller, "Chaos-generating periodic orbits of
/// topological defects in confined active nematics", Eq. (11), take
///
/// ```text
/// F_i = d_j [ -H_ij - zeta Q_ij + [Q, H]_ij + 2 Tr(QH) Q_ij - K d_i Q_kl d_j Q_kl ],
/// ```
///
/// the full Beris-Edwards stress. That is `flow-solver.py`'s force and therefore
/// this crate's, and it is [`StressModel::Full`].
///
/// Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing active
/// nematics", Phys. Rev. E 109, 014606 (2024), take `Pi = Pi_E + Pi_A` with
/// `Pi_E = -lambda H + [Q, H]` and `Pi_A = -alpha Q`, "as given in Ref. 10",
/// which is Giomi, Phys. Rev. X 5, 031003 (2015). Neither the `2 Tr(QH) Q` term
/// nor the Ericksen distortion stress appears, and Giomi states why: the
/// Ericksen stress "has been neglected because of higher order in the
/// derivatives of Q_ij compared to sigma^e_ij", a simplification "known not to
/// have appreciable consequences in the fluid mechanics of two-dimensional
/// active nematics". That is [`StressModel::Giomi`].
///
/// Part of the Ericksen divergence is a gradient the pressure absorbs on an
/// incompressible flow and part is not, so the two are different forces on the
/// defects, and which one a run uses is a modelling choice rather than a detail.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StressModel {
    /// `-lambda H - zeta Q + [Q, H]` plus the Ericksen and `2 Tr(QH) Q` terms.
    #[default]
    Full,
    /// `-lambda H - zeta Q + [Q, H]` alone, as Giomi and Mitchell et al. write it.
    Giomi,
}

impl Params {
    /// Construct from the four free parameters used in the Python notebooks.
    ///
    /// - `lx`          : grid width  (pixels)
    /// - `als`         : active length scale  (pixels); determines `zeta`
    /// - `ncl`         : nematic correlation length  (pixels); determines `a/c_landau`
    /// - `dt`          : time step
    /// - `max_p_iters` : max pressure-Poisson inner iterations
    ///
    /// Derived constants follow the Python solver conventions:
    /// ```text
    /// k_elastic = 2^14
    /// gamma     = 100
    /// eta       = sqrt(10 * k_elastic)
    /// rho       = 1
    /// chi       = 1
    /// zeta      = k_elastic / als^2
    /// c_landau  = k_elastic / ncl^2
    /// a_landau  = -c_landau
    /// s0        = sqrt(2)
    /// lambda    = flow-alignment (code-truth flow-solver.py / fsn.py: λ = 1)
    /// ```
    ///
    /// `als` and `ncl` are the active and nematic-coherence length scales in
    /// lattice units, as floats, since the production sweep uses fractional values, e.g.
    /// the silver braid at als=2.8, ncl=4.8). The map is the code-truth one:
    /// `ζ = K/als²`, `C = K/ncl²`, `A = −C` (flow-solver.py:1478-1481).
    pub fn new(lx: usize, als: f64, ncl: f64, lambda: f64, dt: f64, max_p_iters: i64) -> Self {
        let k_elastic = 2_f64.powi(14);
        let gamma = 100.0_f64;
        let eta = (10.0 * k_elastic).sqrt();
        let rho = 1.0_f64;
        let chi = 1.0_f64;
        let zeta = k_elastic / als.powi(2);
        let c_landau = k_elastic / ncl.powi(2);
        let a_landau = -c_landau;
        // flow-solver.py:1481: `S0 = np.sqrt(-2 * A / C)`. Since A=−C this is
        // sqrt(2), matching the paper's stated S0 (Klein et al., arXiv:2503.10880,
        // p. 1). A previous version of this formula omitted the factor of 2
        // (giving S0=1 instead of sqrt(2)); the bit-for-bit concurrence tests in
        // tests/step.rs never exercised this path, since they set s0 directly
        // from a hardcoded SQRT_2 constant rather than through Params::new.
        let s0 = (-2.0 * a_landau / c_landau).sqrt();
        let ly = lx; // square grid assumed (can override via field)

        Params {
            lx,
            ly,
            dt,
            k_elastic,
            gamma,
            eta,
            rho,
            chi,
            zeta,
            a_landau,
            c_landau,
            s0,
            lambda,
            max_p_iters,
            net_charge: 1.0,
            locking: None,
            stress: StressModel::Full,
        }
    }

    /// Override the boundary's total topological charge `q`. Defaults to
    /// `1.0` (the crate's original hardcoded value, corresponding to the
    /// nephroid production run's `net_charge = 2/2`).
    pub fn with_net_charge(mut self, net_charge: f64) -> Self {
        self.net_charge = net_charge;
        self
    }

    /// Switch on enhanced nematic locking (arXiv:2506.20996).
    pub fn with_locking(mut self, locking: Locking) -> Self {
        self.locking = Some(locking);
        self
    }

    /// Select the elastic stress the force density is taken from.
    ///
    /// [`StressModel::Giomi`] drops the Ericksen and `2 Tr(QH) Q` terms, which
    /// is the stress Mitchell et al. and Giomi write. Everything else is
    /// unchanged.
    pub fn with_stress(mut self, stress: StressModel) -> Self {
        self.stress = stress;
        self
    }

    /// Build from the dimensionless groups Mitchell et al. state their runs in.
    ///
    /// `k_elastic` sets the remaining scale, which is free: the dynamics depend
    /// on `k_elastic` only through `dt`, since every group in [`Dimensionless`]
    /// is invariant under a rescaling of `K` at fixed lattice spacing.
    /// [`Dimensionless::MITCHELL_K`] is the value that reproduces the reference's
    /// own stated constants digit for digit.
    ///
    /// The domain is a torus, so [`Params::net_charge`] is never read and is set
    /// to zero.
    pub fn from_dimensionless(lx: usize, ly: usize, d: Dimensionless, k_elastic: f64,
                              dt: f64, max_p_iters: i64) -> Self {
        let rho = 1.0_f64;
        // Re = K / (rho nu^2).
        let eta = (k_elastic / (rho * d.re)).sqrt();
        // gamma_tilde = gamma nu / K.
        let gamma = d.gamma_tilde * k_elastic / eta;
        let zeta = k_elastic / d.ell_a.powi(2);
        let c_landau = d.c_tilde * zeta;
        // S_eq = sqrt(-2 A / C).
        let a_landau = -0.5 * c_landau * d.s_eq.powi(2);

        Params {
            lx,
            ly,
            dt,
            k_elastic,
            gamma,
            eta,
            rho,
            chi: 1.0,
            zeta,
            a_landau,
            c_landau,
            s0: d.s_eq,
            lambda: d.lambda,
            max_p_iters,
            net_charge: 0.0,
            locking: None,
            stress: StressModel::Full,
        }
    }

    /// Active length `sqrt(K / zeta)`, in lattice units.
    pub fn active_length(&self) -> f64 {
        (self.k_elastic / self.zeta).sqrt()
    }

    /// Nematic coherence length `sqrt(K / C)`, in lattice units.
    pub fn coherence_length(&self) -> f64 {
        (self.k_elastic / self.c_landau).sqrt()
    }

    /// Active time scale `t_a = K / (zeta nu)`, in units of the integration time.
    ///
    /// Mitchell et al. (2024) report a dimensionless topological entropy
    /// `h_tilde = h t_a`; this is the factor.
    pub fn active_time(&self) -> f64 {
        self.k_elastic / (self.zeta * self.eta)
    }
}

/// The dimensionless groups of Mitchell, Sabbir, Geumhan, Smith, Klein and
/// Beller, "Maximally mixing active nematics", Phys. Rev. E 109, 014606 (2024).
///
/// The paper states that its dynamics are determined by five numbers: the
/// flow-alignment parameter, the Reynolds number `Re = K / (rho nu^2)`, a
/// dimensionless rotational viscosity `gamma_tilde = gamma nu / K`, a
/// Landau-de Gennes parameter `C_tilde = C / zeta = (ell_a / ell_n)^2`, and the
/// confinement ratio `ell_a / L`, the last of which is set by the grid size
/// rather than here.
///
/// `s_eq` is the sixth thing a reader has to supply, because the two papers
/// modelled here fix it differently: Mitchell et al. (2024) take `C = -A` and
/// so `S_eq = sqrt(2)`, while arXiv:2506.20996 takes `C = -2A` and `S_eq = 1`.
/// The groups above are the same in both.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Dimensionless {
    /// Active length `ell_a = sqrt(K / zeta)`, in lattice units.
    pub ell_a: f64,
    /// Reynolds number `K / (rho nu^2)`.
    pub re: f64,
    /// Dimensionless rotational viscosity `gamma nu / K`.
    pub gamma_tilde: f64,
    /// `C / zeta = (ell_a / ell_n)^2`.
    pub c_tilde: f64,
    /// Flow-alignment parameter. Enhanced nematic locking requires `1`.
    pub lambda: f64,
    /// Equilibrium scalar order parameter `sqrt(-2 A / C)`.
    pub s_eq: f64,
}

impl Dimensionless {
    /// The elastic constant that recovers the reference's own stated constants.
    ///
    /// arXiv:2506.20996 lists `gamma = 5 * 256`, `C = 256^2`, `K = 256^2`,
    /// `eta = 2560` and `zeta = (256/3)^2`. Feeding `K = 256^2` and
    /// [`Dimensionless::mitchell`] at `ell_a = 3` into
    /// [`Params::from_dimensionless`] returns exactly those five numbers.
    pub const MITCHELL_K: f64 = 65536.0;

    /// Mitchell et al. (2024): `lambda = 1`, `Re = 0.01`, `gamma_tilde = 50`,
    /// `C_tilde = 9`, `C = -A` so `S_eq = sqrt(2)`.
    pub fn mitchell(ell_a: f64) -> Self {
        Self {
            ell_a,
            re: 0.01,
            gamma_tilde: 50.0,
            c_tilde: 9.0,
            lambda: 1.0,
            s_eq: std::f64::consts::SQRT_2,
        }
    }

    /// arXiv:2506.20996: the same groups, with `C = -2A` so `S_eq = 1`, which is
    /// the convention its switch width `sigma = 0.2` is quoted in.
    pub fn nematic_locking(ell_a: f64) -> Self {
        Self { s_eq: 1.0, ..Self::mitchell(ell_a) }
    }

    /// Nematic coherence length implied by `C_tilde`, in lattice units.
    pub fn ell_n(&self) -> f64 {
        self.ell_a / self.c_tilde.sqrt()
    }
}

// The finite-difference physics that was `volterra-solver` until that crate was
// dissolved. It is all finite difference on Cartesian grids, in two dimensions
// and three, which is what this crate is named for.

pub mod cartesian_2d;
pub use cartesian_2d::*;

pub mod mol_field_3d;
pub use mol_field_3d::{molecular_field_3d, molecular_field_3d_par, molecular_field_3d_par_into,
                       euler_step_fused_par, co_rotation_3d};

pub mod beris_3d;
pub use beris_3d::{beris_edwards_rhs_3d, beris_edwards_rhs_3d_par_dry,
                   beris_edwards_rhs_3d_par_dry_into, euler_step_par};

pub mod fire;
pub use fire::{fire_minimize_3d_par, fire_step_3d_par, force_max_metric, FireParams, FireState};

pub mod stokes_3d;
pub use stokes_3d::stokes_solve_3d;

pub mod ch_3d;
pub use ch_3d::ch_step_etd_3d;

pub mod defects_3d;
pub use defects_3d::{scan_defects_3d, track_defect_events};

pub mod confinement_3d;
pub use confinement_3d::{
    ConfinedLdg, LdgFromChi, PhaseField3D, activity_number, anchoring_molecular_field,
    molecular_field_confined_3d, relax_step_confined_3d,
};

pub mod gauss_bonnet_3d;
pub use gauss_bonnet_3d::gauss_bonnet_chi;

pub mod runner_3d;
pub use runner_3d::{run_dry_active_nematic_3d, run_bech_3d, SnapStats3D, BechStats3D};

pub mod sim_impls;
