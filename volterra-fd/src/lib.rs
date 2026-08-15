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
//! are the contract). It provides the simulation parameters, the nephroid
//! boundary construction, the physics kernels, and a hardened run harness.

pub mod boundary;
pub use boundary::{circular_boundary, nephroid_boundary, Boundary};

pub mod index;

pub mod par_gate;
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
        }
    }

    /// Override the boundary's total topological charge `q`. Defaults to
    /// `1.0` (the crate's original hardcoded value, corresponding to the
    /// nephroid production run's `net_charge = 2/2`).
    pub fn with_net_charge(mut self, net_charge: f64) -> Self {
        self.net_charge = net_charge;
        self
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
