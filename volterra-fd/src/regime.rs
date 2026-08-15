//! Which dynamical regime a confined active nematic falls into, without
//! running it.
//!
//! A run of `volterra-fd` on an epitrochoid costs tens of minutes; deciding
//! afterwards that its parameters put it in the melted regime is an expensive
//! way to learn that. Everything here is closed form.
//!
//! The two inputs are the dimensionless lengths arXiv:2503.10880 reports: the
//! active length `ell_a = sqrt(K / zeta)` and the coherence length
//! `ell_c = sqrt(K / |A|)`, each divided by `sqrt(A_sys)`, the square root of
//! the confined area in lattice sites. [`Boundary::sqrt_area`] supplies the
//! divisor and the `fd` driver takes the dimensionless values directly through
//! `FD_ELL_A` and `FD_ELL_C`.
//!
//! # What is derived and what is fitted
//!
//! The defect count and the cusp geometry are exact. The three regime
//! boundaries have forms fixed by dimensional analysis and four coefficients
//! fitted against the 250 classified points of that paper's Fig. 7, which
//! agree with its classification on 72% of them: essentially all of its
//! braiding points and about half of its turbulent ones. The bias is towards
//! calling a point braiding, so a prediction of braiding is worth checking and
//! a prediction of melted or arrested is worth believing.
//!
//! [`Boundary::sqrt_area`]: crate::boundary::Boundary::sqrt_area

use crate::boundary::Epitrochoid;

/// Disclinations a confinement requires, before activity does anything.
///
/// Tangential anchoring on a simply connected domain fixes the interior charge
/// at `+1` whatever the shape, so with `k` cusps each pinning a `-1/2` the
/// mobile population follows from `n_plus/2 - k/2 = 1`:
///
/// ```text
/// n_plus = 2 + k,   n_minus = k
/// ```
///
/// Three and one for the cardioid, four and two for the nephroid. None of it is
/// imposed: the boundary condition carries winding number one, the same as a
/// disk's, and the `-1/2` defects are pinned dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefectPopulation {
    /// Mobile `+1/2` disclinations.
    pub positive: usize,
    /// `-1/2` disclinations pinned at the cusps.
    pub negative: usize,
}

/// The population topology requires of an epitrochoid.
pub fn topological_defects(epi: &Epitrochoid) -> DefectPopulation {
    let k = epi.cusps().round() as usize;
    DefectPopulation {
        positive: 2 + k,
        negative: k,
    }
}

/// Radius of curvature at a cusp, in the same units as `r`.
///
/// From Eq. SI.6 with `m = 2q - 1` and `eps = 1 - d`:
///
/// ```text
/// R_cusp = (r / 2q) m eps^2 / (m - 1)
/// ```
///
/// At the paper's `d = 0.99` this is 0.0066 lattice spacings for a cardioid at
/// `r = 99` and 0.0018 for a nephroid at `r = 49`, four orders of magnitude
/// below any coherence length a lattice can resolve. The regularisation is
/// therefore invisible to the physics: the cusps are sharp at every usable
/// `ell_c`, which is why they pin a `-1/2` at every parameter point rather than
/// only at sharp ones.
pub fn cusp_radius(epi: &Epitrochoid, r: f64) -> f64 {
    let m = 2.0 * epi.q - 1.0;
    if m <= 1.0 {
        return f64::INFINITY; // a disk has no cusp
    }
    let a = r / (2.0 * epi.q);
    a * m * (1.0 - epi.d).powi(2) / (m - 1.0)
}

/// Coefficients of the regime model, fitted to arXiv:2503.10880 Fig. 7.
#[derive(Debug, Clone, Copy)]
pub struct RegimeConstants {
    /// Defect count activity alone sustains is `c_activity / ell_a^2`. Activity
    /// and elasticity make one length between them, so the areal density can
    /// only go as `ell_a^-2`.
    pub c_activity: f64,
    /// Melted area per core, in units of `pi ell_c^2`. Above one because the
    /// order parameter is depressed well past the nominal core radius.
    pub c_core: f64,
    /// Least motility number `ell_c / ell_a^2` at which defects move.
    pub m_arrest: f64,
    /// Cores the domain holds is `c_room / ell_c^2`. Nucleation needs room for
    /// two more, which is what puts turbulence at small `ell_c`.
    pub c_room: f64,
}

impl Default for RegimeConstants {
    fn default() -> Self {
        Self {
            // c_core is fitted directly to order parameters measured over
            // seven runs of this solver spanning ell_c from 0.011 to 0.118.
            // The other three are fitted to Fig. 7 with it held: left free
            // against that weaker constraint c_core drifts to 2.5 and then
            // over-predicts the measured melting by half, at no gain in
            // classification accuracy.
            c_activity: 1.585e-3,
            c_core: 1.65,
            m_arrest: 35.0,
            c_room: 1.995e-2,
        }
    }
}

/// What a confined active nematic does at a given pair of lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Nematic order is gone over enough of the domain that a defect position
    /// is not a sharp quantity.
    Melted,
    /// The defects sit at their elastic equilibrium and do not move.
    Arrested,
    /// Activity nucleates pairs, so the defect count is not constant and no
    /// braid closes.
    Turbulent,
    /// A fixed set of mobile `+1/2` defects, which is what braids.
    Braiding,
}

/// The full prediction at one parameter point.
#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub regime: Regime,
    /// Mobile `+1/2` defects expected.
    pub positive: usize,
    /// Fraction of the interior below half the equilibrium order.
    pub melted_fraction: f64,
    /// Motility number `ell_c / ell_a^2`.
    pub motility: f64,
    /// Distance to the nearest regime boundary as a fraction of that boundary,
    /// positive only inside the braiding regime.
    pub margin: f64,
}

/// Fraction of the interior with its nematic order destroyed.
///
/// Cores laid down at random cover `1 - exp(-N pi ell_c^2)` rather than
/// `N pi ell_c^2`, since they overlap. At the coherence lengths of Fig. 7 that
/// is the difference between predicting 58% and predicting 44%, against a
/// measured 42%.
///
/// `N` counts the mobile defects, the pinned ones, and one melted patch per
/// cusp: a cusp whose radius of curvature is four orders of magnitude below
/// `ell_c` melts a region whether or not a defect sits in it.
pub fn melted_fraction(epi: &Epitrochoid, ell_c: f64, positive: usize, c: &RegimeConstants) -> f64 {
    let k = epi.cusps().round() as usize;
    let n = (positive + 2 * k) as f64;
    1.0 - (-c.c_core * n * std::f64::consts::PI * ell_c * ell_c).exp()
}

/// Predict the regime at `(ell_a, ell_c)`, both dimensionless.
///
/// The order of the tests follows the physics: a field with no order left has
/// no defects to move, so melting is checked before motion, and defects pinned
/// in place cannot nucleate more, so arrest is checked before nucleation.
pub fn classify(epi: &Epitrochoid, ell_a: f64, ell_c: f64, c: &RegimeConstants) -> Prediction {
    let floor = topological_defects(epi).positive;
    let wanted = c.c_activity / (ell_a * ell_a);
    let positive = if wanted > floor as f64 {
        wanted.round() as usize
    } else {
        floor
    };
    let melted = melted_fraction(epi, ell_c, positive, c);
    let motility = ell_c / (ell_a * ell_a);
    let room = c.c_room / (ell_c * ell_c);

    let regime = if melted > 0.5 {
        Regime::Melted
    } else if motility < c.m_arrest {
        Regime::Arrested
    } else if wanted > floor as f64 + 0.5 && room > floor as f64 + 2.0 {
        Regime::Turbulent
    } else {
        Regime::Braiding
    };

    let to_melt = 1.0 - melted / 0.5;
    let to_arrest = motility / c.m_arrest - 1.0;
    // Turbulence needs both activity wanting more defects and room for them, so
    // the distance from it is the larger of the two slacks.
    let to_turbulent = (1.0 - wanted / (floor as f64 + 0.5)).max(1.0 - room / (floor as f64 + 2.0));

    Prediction {
        regime,
        positive: if regime == Regime::Braiding { floor } else { positive },
        melted_fraction: melted,
        motility,
        margin: to_melt.min(to_arrest).min(to_turbulent),
    }
}

/// The braid a given number of mobile `+1/2` defects adopts in confinement.
///
/// arXiv:2503.10880 finds the golden braid generic to three defects and the
/// silver to four, and at six a Ceilidh dance, the six-defect member of the
/// same family. Five braids nothing periodic.
pub fn braid_of(positive: usize) -> Option<(&'static str, f64)> {
    match positive {
        3 => Some(("golden", crate::regime::GOLDEN_ENTROPY)),
        4 => Some(("silver", crate::regime::SILVER_ENTROPY)),
        6 => Some(("ceilidh", crate::regime::SILVER_ENTROPY)),
        _ => None,
    }
}

/// `2 log phi`, the golden braid's topological entropy.
pub const GOLDEN_ENTROPY: f64 = 0.962_423_650_119_205_8;
/// `log(3 + 2 sqrt 2)`, the silver braid's topological entropy.
pub const SILVER_ENTROPY: f64 = 1.762_747_174_039_086;

#[cfg(test)]
mod tests {
    use super::*;

    fn epi(q: f64) -> Epitrochoid {
        Epitrochoid::new(q)
    }

    /// The population is what the paper observes for each geometry.
    #[test]
    fn topological_counts_match_the_paper() {
        assert_eq!(
            topological_defects(&epi(1.5)),
            DefectPopulation { positive: 3, negative: 1 }
        );
        assert_eq!(
            topological_defects(&epi(2.0)),
            DefectPopulation { positive: 4, negative: 2 }
        );
        assert_eq!(
            topological_defects(&epi(2.5)),
            DefectPopulation { positive: 5, negative: 3 }
        );
    }

    /// The cusps are sharp against any coherence length a lattice can resolve.
    ///
    /// This is what licenses treating the pinning as unconditional. A cusp
    /// radius even a tenth of a lattice spacing would leave it in question.
    #[test]
    fn cusps_are_sharp_at_the_papers_regularisation() {
        assert!(cusp_radius(&epi(1.5), 99.0) < 0.01);
        assert!(cusp_radius(&epi(2.0), 49.0) < 0.01);
        // A disk has none.
        assert!(cusp_radius(&epi(1.0), 99.0).is_infinite());
    }

    /// Melting predicted at the paper's two snapshot points against measured.
    ///
    /// The measurements are of this solver's own runs: 17% of the cardioid's
    /// interior and 42% of the nephroid's below half the equilibrium order.
    #[test]
    fn melting_matches_the_measured_runs() {
        let c = RegimeConstants::default();
        let cardioid = melted_fraction(&epi(1.5), 0.0903, 3, &c);
        let nephroid = melted_fraction(&epi(2.0), 0.1178, 4, &c);
        assert!(
            (cardioid - 0.17).abs() < 0.12,
            "cardioid predicted {cardioid:.3}, measured 0.17"
        );
        assert!(
            (nephroid - 0.42).abs() < 0.15,
            "nephroid predicted {nephroid:.3}, measured 0.42"
        );
        assert!(nephroid > cardioid, "the nephroid point is the more melted");
    }

    /// The nephroid at the paper's own point comes out arrested or melted,
    /// which is what four seeds of this solver do there.
    #[test]
    fn the_published_nephroid_point_is_not_predicted_to_braid() {
        let p = classify(&epi(2.0), 0.0131, 0.1178, &RegimeConstants::default());
        assert_ne!(p.regime, Regime::Braiding, "got {:?}", p.regime);
    }

    /// Lowering the coherence length at fixed activity un-melts the domain.
    #[test]
    fn melting_falls_with_the_coherence_length() {
        let c = RegimeConstants::default();
        let hi = melted_fraction(&epi(2.0), 0.1178, 4, &c);
        let lo = melted_fraction(&epi(2.0), 0.0393, 4, &c);
        assert!(lo < hi / 3.0, "expected a large fall, got {lo:.3} from {hi:.3}");
    }

    /// Each geometry's braid follows from its cusp count alone.
    #[test]
    fn braid_follows_from_the_geometry() {
        assert_eq!(braid_of(topological_defects(&epi(1.5)).positive).unwrap().0, "golden");
        assert_eq!(braid_of(topological_defects(&epi(2.0)).positive).unwrap().0, "silver");
        assert!(braid_of(topological_defects(&epi(2.5)).positive).is_none());
    }

    /// The margin is positive only where the prediction is braiding.
    #[test]
    fn margin_is_positive_exactly_when_braiding() {
        let c = RegimeConstants::default();
        for &(q, la, lc) in &[
            (1.5, 0.0278, 0.0625),
            (2.0, 0.0229, 0.0393),
            (2.0, 0.0131, 0.1178),
            (1.5, 0.0069, 0.0300),
            (1.5, 0.0556, 0.1000),
        ] {
            let p = classify(&epi(q), la, lc, &c);
            assert_eq!(
                p.margin > 0.0,
                p.regime == Regime::Braiding,
                "q={q} la={la} lc={lc} gave {:?} with margin {:.3}",
                p.regime,
                p.margin
            );
        }
    }
}
