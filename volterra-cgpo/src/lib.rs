//! `volterra-cgpo`: scaffold and boundary for the nephroid-confined CGPO solver.
//!
//! This crate is the foundation for a Rust port of the Beris–Edwards +
//! relaxation-Stokes finite-difference solver from
//! `~/Chaos-Generating-Periodic-Orbits/flow-solver.py`.
//! Task A provides the simulation parameters and the nephroid boundary
//! construction; later tasks add the physics kernels.

pub mod boundary;
pub use boundary::{nephroid_boundary, Boundary};

pub mod index;

pub mod par_gate;
pub mod ops;
pub mod nematic;
pub mod stokes;
pub mod bc;
pub mod step;

pub mod error;
pub use error::{CgpoError, CgpoResult};

/// Simulation parameters, matching the Python solver's constants.
///
/// Constructor: [`Params::new`].
#[derive(Debug, Clone)]
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
    /// lattice units (floats — the production sweep uses fractional values, e.g.
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
        // code-truth: S₀ = √(−A/C) (flow-solver.py / fsn.py). Since A=−C this is 1.
        // (Sets only the IC amplitude; the field relaxes to the A,C equilibrium.)
        let s0 = (-a_landau / c_landau).sqrt();
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
        }
    }
}
