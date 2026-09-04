//! One parameter set for the confined active nematic, in whichever form is to hand.
//!
//! Three parameterisations are in use across this workspace and the literature,
//! and they describe the same physics:
//!
//! **Dimensional Landau-de Gennes**, which is what the kernels consume and what
//! `flow-solver.py` sets: a Frank constant `K`, Landau coefficients `A < 0` and
//! `C > 0`, a rotational viscosity `gamma`, a flow-alignment `lambda`, an activity
//! `zeta`, a kinematic viscosity `eta` and a density `rho`.
//!
//! **The length-scale form**, which specifies the same thing through two lengths
//! in lattice units, `als` and `ncl`, with everything else fixed:
//!
//! ```text
//! K = 2^14   gamma = 100   Re = 0.1   eta = sqrt(K / Re)   rho = 1
//! zeta = K / als^2         C = K / ncl^2       A = -C
//! ```
//!
//! so `als = sqrt(K / zeta)` is the active length and `ncl = sqrt(K / C)` the
//! nematic coherence length, both in lattice sites.
//!
//! **The nondimensional form** used by `active_nematic_engine`, which measures
//! those two lengths against the domain instead of against the lattice:
//!
//! ```text
//! Pe = (L / als)^2 = zeta L^2 / K        epsilon = ncl / L
//! ```
//!
//! The three are the same two numbers in different units, so there is no reason to
//! carry more than one set of state. [`NematicParams`] stores the dimensional set,
//! which is what every kernel reads, and every other form is a constructor into it
//! or an accessor out of it. Nothing is converted per step.
//!
//! ## The two conventions for the cubic term
//!
//! The reference's molecular field is
//!
//! ```text
//! H = K grad^2 Q - (A + C Tr(Q^2)) Q,     Tr(Q^2) = 2 (Qxx^2 + Qxy^2)
//! ```
//!
//! while [`crate::molecular_field_dec`] takes `ActiveNematicParams` and computes
//!
//! ```text
//! H = -K Delta_L Q - a_eff Q - 2 c_landau Tr(Q^2) Q,   a_eff = a_landau - zeta/2
//! ```
//!
//! Two differences, and both bite. The cubic coefficient is `2 c_landau` against
//! the reference's `C`, so `c_landau = C / 2`. The rotor model also folds half the
//! activity into the linear term, where the reference puts activity in the stress
//! alone. [`NematicParams::to_rotor_convention`] performs both corrections, so a
//! caller cannot get them wrong by hand.

use serde::{Deserialize, Serialize};

/// How a parameter set was specified, kept for provenance and for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Parameterisation {
    /// The two lattice lengths at a given grid side.
    LengthScales { active_length: f64, coherence_length: f64, resolution: usize },
    /// Activity and core size measured against the domain.
    Nondimensional { pe: f64, epsilon: f64, domain: f64 },
    /// The dimensional constants given directly.
    Direct,
}

/// The dimensional Landau-de Gennes and Beris-Edwards constants.
///
/// Every field is in the units the kernels use. Construct with
/// [`Self::from_length_scales`],
/// [`Self::nondimensional`] or [`Self::direct`], and read the derived groups off
/// the accessors rather than recomputing them at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NematicParams {
    /// Frank elastic constant, `L` in the Landau-de Gennes free energy.
    pub k_frank: f64,
    /// Landau quadratic coefficient, negative for an ordered nematic.
    pub a_landau: f64,
    /// Landau quartic coefficient, positive so large `|Q|` is penalised.
    pub c_landau: f64,
    /// Rotational viscosity. The Q relaxation carries `1 / gamma`.
    pub gamma: f64,
    /// Flow alignment: 1 is flow-aligning, 0 co-rotational.
    pub lambda: f64,
    /// Activity. Enters the stress as `-zeta Q`, and nothing else.
    pub zeta: f64,
    /// Kinematic viscosity.
    pub eta: f64,
    /// Density.
    pub rho: f64,
    /// Timestep.
    pub dt: f64,
    /// Which form this was given in.
    pub source: Parameterisation,
}

/// The fixed dimensional constants of the length-scale parameterisation.
pub mod constants {
    /// Frank constant, `K = 2^14`.
    pub const K: f64 = 16384.0;
    /// Rotational viscosity.
    pub const GAMMA: f64 = 100.0;
    /// Reynolds number used only to set the viscosity.
    pub const RE: f64 = 0.1;
    /// Density.
    pub const RHO: f64 = 1.0;
    /// Flow alignment.
    pub const LAMBDA: f64 = 1.0;
    /// Timestep, at `resolution = 1`.
    pub const DT: f64 = 1e-4;
    /// Snapshot cadence, `10 * jit_loops`.
    pub const SAVE_EVERY: usize = 1000;
}

impl NematicParams {
    /// The length-scale parameterisation: two lattice lengths, everything else
    /// fixed.
    ///
    /// `lx` enters only as provenance and for the nondimensional accessors; the
    /// dimensional constants do not depend on it, which is why two runs at
    /// different resolutions and the same `als` and `ncl` are the same physics at
    /// different discretisations rather than different physics.
    pub fn from_length_scales(als: f64, ncl: f64, lx: usize) -> Self {
        let c = constants::K / (ncl * ncl);
        Self {
            k_frank: constants::K,
            a_landau: -c,
            c_landau: c,
            gamma: constants::GAMMA,
            lambda: constants::LAMBDA,
            zeta: constants::K / (als * als),
            eta: (constants::K / constants::RE).sqrt(),
            rho: constants::RHO,
            dt: constants::DT,
            source: Parameterisation::LengthScales {
                active_length: als,
                coherence_length: ncl,
                resolution: lx,
            },
        }
    }

    /// The nondimensional form: activity and core size against the domain.
    ///
    /// `pe = (domain / als)^2` and `epsilon = ncl / domain`, so this is the
    /// length-scale pair rescaled. The remaining constants take the values above,
    /// since the nondimensional groups do not fix them and a comparison wants them
    /// equal.
    pub fn nondimensional(pe: f64, epsilon: f64, domain: f64) -> Self {
        let als = domain / pe.sqrt();
        let ncl = epsilon * domain;
        let mut p = Self::from_length_scales(als, ncl, domain.round() as usize);
        p.source = Parameterisation::Nondimensional { pe, epsilon, domain };
        p
    }

    /// The dimensional constants given directly.
    #[allow(clippy::too_many_arguments)]
    pub fn direct(
        k_frank: f64,
        a_landau: f64,
        c_landau: f64,
        gamma: f64,
        lambda: f64,
        zeta: f64,
        eta: f64,
        rho: f64,
        dt: f64,
    ) -> Self {
        Self {
            k_frank,
            a_landau,
            c_landau,
            gamma,
            lambda,
            zeta,
            eta,
            rho,
            dt,
            source: Parameterisation::Direct,
        }
    }

    /// Equilibrium scalar order parameter, `sqrt(-2 A / C)`, which is `sqrt(2)`
    /// whenever `A = -C`, as the reference sets it.
    pub fn s0(&self) -> f64 {
        (-2.0 * self.a_landau / self.c_landau).sqrt()
    }

    /// Nematic coherence length, `sqrt(K / C)`, the `ncl` above.
    pub fn coherence_length(&self) -> f64 {
        (self.k_frank / self.c_landau).sqrt()
    }

    /// Active length, `sqrt(K / zeta)`, the `als` above. Infinite when passive.
    pub fn active_length(&self) -> f64 {
        if self.zeta == 0.0 {
            f64::INFINITY
        } else {
            (self.k_frank / self.zeta.abs()).sqrt()
        }
    }

    /// Reynolds number implied by the viscosity, `K / eta^2`.
    pub fn reynolds(&self) -> f64 {
        self.k_frank / (self.eta * self.eta)
    }

    /// Activity against the domain, `zeta L^2 / K = (L / als)^2`.
    pub fn peclet(&self, domain: f64) -> f64 {
        self.zeta * domain * domain / self.k_frank
    }

    /// Core size against the domain, `ncl / L`.
    pub fn epsilon(&self, domain: f64) -> f64 {
        self.coherence_length() / domain
    }

    /// The paper's own dimensionless lengths, both divided by the square root of
    /// the confined area rather than by the grid side.
    ///
    /// The plotting script behind Fig 7 uses `sqrt(A_sys) = 0.764031 L` for the
    /// nephroid, so passing the interior cell count of the run is what reproduces
    /// its axes; passing the mesh's own area does the same for a mesh.
    pub fn paper_lengths(&self, area: f64) -> (f64, f64) {
        let s = area.sqrt();
        (self.active_length() / s, self.coherence_length() / s)
    }

    /// The same physics with the activity removed, for a passive relaxation.
    ///
    /// The reference does this too, by setting `consts_dict["zeta"] = 0`, and it is
    /// the cheapest configuration to compare two discretisations in: no flow, so
    /// no Stokes solve and no advection, and the equilibrium is set by the
    /// anchoring and the elasticity alone.
    pub fn passive(&self) -> Self {
        Self { zeta: 0.0, ..*self }
    }

    /// Diffusive stability limit for an explicit step of the Q relaxation.
    ///
    /// The elastic term relaxes at `K / gamma` per unit area, so an explicit step
    /// on a mesh of smallest edge `h` needs `dt < gamma h^2 / (4 K)`. At the reference's
    /// constants and `h = 1` that is `1.5e-3`, and his `dt = 1e-4` sits a factor of
    /// fifteen inside it. A graded mesh has a much smaller `h` at the cusp, so this
    /// is the number that decides whether the same `dt` is usable there.
    pub fn q_diffusive_dt_limit(&self, h_min: f64) -> f64 {
        self.gamma * h_min * h_min / (4.0 * self.k_frank)
    }

    /// Viscous stability limit for an explicit step of the velocity update.
    pub fn viscous_dt_limit(&self, h_min: f64) -> f64 {
        h_min * h_min / (4.0 * self.eta)
    }

    /// The rotor model's `(a_eff, c)` pair that reproduces this molecular field.
    ///
    /// [`crate::molecular_field_dec`] computes
    /// `H = -K Delta_L Q - a_eff Q - 2 c Tr(Q^2) Q` with
    /// `a_eff = a_landau - zeta_eff / 2`, against the reference's
    /// `H = K grad^2 Q - (A + C Tr(Q^2)) Q`. Matching them needs `c = C / 2` and
    /// `a_eff = A`, and since that function derives `a_eff` from its own fields the
    /// activity has to be handed to it as zero and applied through the stress
    /// instead, which is where the reference puts activity anyway.
    ///
    /// Returns `(a_landau_for_rotor, c_landau_for_rotor, zeta_eff_for_rotor)`.
    pub fn to_rotor_convention(&self) -> (f64, f64, f64) {
        (self.a_landau, self.c_landau / 2.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_scales_reproduce_the_reference_constants() {
        // The values `flow-solver.py` derives at als = 2.8, ncl = 4.6, which is
        // the stable-golden point of the supplementary videos.
        let p = NematicParams::from_length_scales(2.8, 4.6, 200);
        assert!((p.k_frank - 16384.0).abs() < 1e-9);
        assert!((p.zeta - 16384.0 / (2.8 * 2.8)).abs() < 1e-9);
        assert!((p.c_landau - 16384.0 / (4.6 * 4.6)).abs() < 1e-9);
        assert!((p.a_landau + p.c_landau).abs() < 1e-12, "A = -C");
        assert!((p.eta - (16384.0f64 / 0.1).sqrt()).abs() < 1e-9);
        assert!((p.s0() - 2.0f64.sqrt()).abs() < 1e-12);
        assert!((p.reynolds() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn the_two_lengths_round_trip() {
        for (als, ncl) in [(1.0, 9.0), (1.5, 2.0), (2.8, 4.6), (2.4, 4.0)] {
            let p = NematicParams::from_length_scales(als, ncl, 100);
            assert!((p.active_length() - als).abs() < 1e-9, "als");
            assert!((p.coherence_length() - ncl).abs() < 1e-9, "ncl");
        }
    }

    #[test]
    fn nondimensional_and_length_scales_agree() {
        // The nondimensional pair is the length-scale pair measured against the domain, so
        // going out and back has to land on the same dimensional constants.
        let domain = 100.0;
        for (als, ncl) in [(1.0, 9.0), (1.5, 2.0), (2.8, 4.6)] {
            let a = NematicParams::from_length_scales(als, ncl, domain as usize);
            let pe = a.peclet(domain);
            let eps = a.epsilon(domain);
            let b = NematicParams::nondimensional(pe, eps, domain);
            assert!((a.zeta - b.zeta).abs() < 1e-6 * a.zeta, "zeta at als {als}");
            assert!(
                (a.c_landau - b.c_landau).abs() < 1e-6 * a.c_landau,
                "C at ncl {ncl}"
            );
            assert!((a.a_landau - b.a_landau).abs() < 1e-6 * a.c_landau, "A");
            assert!((a.k_frank - b.k_frank).abs() < 1e-9, "K");
        }
    }

    #[test]
    fn paper_lengths_match_the_plotting_script() {
        // Fig 7's axes are als and ncl divided by sqrt(A_sys) = 0.764031 L for the
        // nephroid, which is the factor its plotting script hard-codes.
        let l = 100.0;
        let area = (0.764031 * l) * (0.764031 * l);
        let p = NematicParams::from_length_scales(1.0, 9.0, 100);
        let (ea, ec) = p.paper_lengths(area);
        assert!((ea - 1.0 / (0.764031 * l)).abs() < 1e-12);
        assert!((ec - 9.0 / (0.764031 * l)).abs() < 1e-12);
    }

    #[test]
    fn passive_removes_only_the_activity() {
        let p = NematicParams::from_length_scales(2.8, 4.6, 200);
        let q = p.passive();
        assert_eq!(q.zeta, 0.0);
        assert!(q.active_length().is_infinite());
        assert!((q.c_landau - p.c_landau).abs() < 1e-12);
        assert!((q.k_frank - p.k_frank).abs() < 1e-12);
    }

    #[test]
    fn rotor_convention_halves_the_cubic_and_drops_the_activity_shift() {
        let p = NematicParams::from_length_scales(2.8, 4.6, 200);
        let (a, c, z) = p.to_rotor_convention();
        assert!((a - p.a_landau).abs() < 1e-12);
        assert!((2.0 * c - p.c_landau).abs() < 1e-12, "2c must equal C");
        assert_eq!(z, 0.0, "activity belongs in the stress, not the linear term");
    }

    #[test]
    fn stability_limits_bracket_the_reference_timestep() {
        // At h = 1 the reference dt sits inside both limits; the point of reporting them
        // is that a graded mesh has h far below 1 at the cusp.
        let p = NematicParams::from_length_scales(2.8, 4.6, 200);
        assert!(p.dt < p.q_diffusive_dt_limit(1.0));
        assert!(p.dt < p.viscous_dt_limit(1.0));
        // And at the cusp scale of a d = 0.99 mesh the diffusive limit is far
        // below it, so an implicit step is needed there rather than optional.
        assert!(p.q_diffusive_dt_limit(1e-3) < p.dt);
    }
}
