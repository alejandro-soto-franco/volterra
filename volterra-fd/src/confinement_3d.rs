//! Phase-field confinement of a 3D nematic, with tangential anchoring on the
//! confining surface.
//!
//! The geometry Head, Digregorio, Marenduzzo, Pagonabarraga, Beller and Negro
//! (arXiv:2607.10234) confine their active nematic in: a liquid-crystal double
//! emulsion, where a scalar phase field `phi` marks the interior of a cylinder
//! and the nematic lives where `phi = phi_0`. The confining surface is a level
//! set of `phi` rather than a meshed boundary, so a curved wall needs no
//! boundary geometry at all, and anchoring on it is a term in the free energy:
//!
//! ```text
//! f_anchor = W (d_alpha phi) Q_{alpha beta} (d_beta phi)
//! ```
//!
//! which is minimised, for `W > 0`, by a director perpendicular to `grad phi`,
//! and so tangential to the surface.
//!
//! # Why the phase field is static here
//!
//! `phi` obeys a Cahn-Hilliard equation in general. The reference's own
//! protocol evolves it for `10^4` steps to relax the interface and then
//! **freezes it**, initialises `Q` at random, relaxes that, and only then
//! switches on activity. Nothing after the freeze moves `phi`. So reproducing
//! that protocol needs the equilibrium profile and not the equation that
//! reaches it, which [`PhaseField3D::capped_cylinder`] writes down in closed
//! form. `crate::ch_3d` carries a Cahn-Hilliard integrator for a different free
//! energy (a lipid field coupled to `Tr(Q^2)`), and is not what this uses.
//!
//! # Free-energy convention
//!
//! The reference writes the bulk nematic energy with a single coupling
//! `chi(phi)`,
//!
//! ```text
//! (A0/2)(1 - chi/3) Q_ab Q_ab  -  (A0 chi/3) Q_ab Q_bc Q_ca  +  (A0 chi/4) (Q_ab Q_ab)^2
//! ```
//!
//! which is the `a Tr(Q^2) + b Tr(Q^3) + c (Tr Q^2)^2` volterra already carries,
//! at
//!
//! ```text
//! a = (A0/2)(1 - chi/3),   b = -A0 chi/3,   c = A0 chi/4.
//! ```
//!
//! [`LdgFromChi`] performs that mapping. At the reference's own `A0 = 0.5` and
//! `chi = 3.2` it puts the equilibrium scalar order parameter at `0.556186`,
//! against the `0.556` the reference states.

use rayon::prelude::*;
use volterra_core::QField3D;

/// A scalar phase field on the same grid as a [`QField3D`].
///
/// `phi` runs from `0` outside the confinement to `phi_0` inside, across an
/// interface of width `xi`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseField3D {
    /// Value at each vertex, indexed `((i * ny) + j) * nz + l`.
    pub phi: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
}

impl PhaseField3D {
    /// Linear index for vertex `(i, j, l)`, matching `QField3D::idx`.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, l: usize) -> usize {
        ((i % self.nx) * self.ny + (j % self.ny)) * self.nz + (l % self.nz)
    }

    /// The equilibrium interface width `xi = sqrt(2k/a)` of the double well
    /// `(a/4) phi^2 (phi - phi_0)^2` with gradient cost `(k/2) |grad phi|^2`.
    ///
    /// The stationary profile of that functional is
    /// `phi(d) = (phi_0/2) (1 - tanh(d / w))` with `w = (2/phi_0) sqrt(2k/a)`,
    /// which is `sqrt(2k/a)` exactly at the reference's `phi_0 = 2`.
    pub fn interface_width(a: f64, k: f64) -> f64 {
        (2.0 * k / a).sqrt()
    }

    /// Surface tension `sigma = sqrt(8ak/9)` of the same double well.
    pub fn surface_tension(a: f64, k: f64) -> f64 {
        (8.0 * a * k / 9.0).sqrt()
    }

    /// A cylinder of radius `radius` and length `length`, axis along z, centred
    /// in the box, with flat endcaps and a `tanh` interface of width `xi`.
    ///
    /// `phi` is `phi_0` inside and `0` outside. Lengths are in grid units.
    // A grid, a geometry and an interface width, each independent of the others.
    #[allow(clippy::too_many_arguments)]
    pub fn capped_cylinder(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        radius: f64,
        length: f64,
        phi_0: f64,
        xi: f64,
    ) -> Self {
        assert!(xi > 0.0, "interface width must be positive");
        let (cx, cy, cz) = (
            (nx as f64 - 1.0) / 2.0,
            (ny as f64 - 1.0) / 2.0,
            (nz as f64 - 1.0) / 2.0,
        );
        let half = length / 2.0;
        let mut phi = vec![0.0; nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for l in 0..nz {
                    let x = (i as f64 - cx) * dx;
                    let y = (j as f64 - cy) * dx;
                    let z = (l as f64 - cz) * dx;
                    // Signed distance to a capped cylinder, positive outside.
                    let d_radial = (x * x + y * y).sqrt() - radius;
                    let d_axial = z.abs() - half;
                    let d = if d_radial > 0.0 && d_axial > 0.0 {
                        (d_radial * d_radial + d_axial * d_axial).sqrt()
                    } else {
                        d_radial.max(d_axial)
                    };
                    phi[((i * ny) + j) * nz + l] = 0.5 * phi_0 * (1.0 - (d / xi).tanh());
                }
            }
        }
        Self { phi, nx, ny, nz, dx }
    }

    /// Central-difference gradient at every vertex, periodic in each direction.
    pub fn gradient(&self) -> Vec<[f64; 3]> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let inv_2dx = 1.0 / (2.0 * self.dx);
        let mut out = vec![[0.0; 3]; self.phi.len()];
        for i in 0..nx {
            for j in 0..ny {
                for l in 0..nz {
                    let k = self.idx(i, j, l);
                    out[k] = [
                        (self.phi[self.idx((i + 1) % nx, j, l)]
                            - self.phi[self.idx((i + nx - 1) % nx, j, l)])
                            * inv_2dx,
                        (self.phi[self.idx(i, (j + 1) % ny, l)]
                            - self.phi[self.idx(i, (j + ny - 1) % ny, l)])
                            * inv_2dx,
                        (self.phi[self.idx(i, j, (l + 1) % nz)]
                            - self.phi[self.idx(i, j, (l + nz - 1) % nz)])
                            * inv_2dx,
                    ];
                }
            }
        }
        out
    }
}

/// The reference's `(A0, chi)` bulk parameters, in volterra's `(a, b, c)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LdgFromChi {
    pub a_landau: f64,
    pub b_landau: f64,
    pub c_landau: f64,
}

impl LdgFromChi {
    /// Map `(A0, chi)` onto `a Tr(Q^2) + b Tr(Q^3) + c (Tr Q^2)^2`.
    pub fn new(a0: f64, chi: f64) -> Self {
        Self {
            a_landau: 0.5 * a0 * (1.0 - chi / 3.0),
            b_landau: -a0 * chi / 3.0,
            c_landau: a0 * chi / 4.0,
        }
    }

    /// The reference's `chi(phi) = chi_0 + chi_s phi`.
    pub fn chi(chi_0: f64, chi_s: f64, phi: f64) -> f64 {
        chi_0 + chi_s * phi
    }

    /// The positive root of `6a + 3b q + 8c q^2 = 0`: the equilibrium scalar
    /// order parameter in the `Q = q (nn - I/3)` convention.
    ///
    /// The same equilibrium condition open-Qmin uses, and the same one
    /// `crate::mol_field_3d` is derived against.
    pub fn equilibrium_q(&self) -> f64 {
        let (a, b, c) = (self.a_landau, self.b_landau, self.c_landau);
        let disc = 9.0 * b * b - 192.0 * a * c;
        if disc < 0.0 || c == 0.0 {
            return 0.0;
        }
        (-3.0 * b + disc.sqrt()) / (16.0 * c)
    }
}

/// The anchoring contribution to the molecular field.
///
/// From `f = W (d_a phi) Q_ab (d_b phi)`, the variation is
/// `df/dQ_ab = W (d_a phi)(d_b phi)`, and `H = -df/dQ + (I/3) Tr(df/dQ)` makes
/// the contribution the traceless part of `-W grad phi (x) grad phi`:
///
/// ```text
/// H^anchor_ab = -W [ (d_a phi)(d_b phi) - (delta_ab / 3) |grad phi|^2 ].
/// ```
///
/// Returned in the same five-component packing as [`QField3D`],
/// `[H11, H12, H13, H22, H23]`, to be added to the bulk and elastic molecular
/// field.
pub fn anchoring_molecular_field(phase: &PhaseField3D, w: f64) -> QField3D {
    let grad = phase.gradient();
    let mut out = QField3D::zeros(phase.nx, phase.ny, phase.nz, phase.dx);
    for (k, g) in grad.iter().enumerate() {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        let t = g2 / 3.0;
        out.q[k] = [
            -w * (g[0] * g[0] - t),
            -w * (g[0] * g[1]),
            -w * (g[0] * g[2]),
            -w * (g[1] * g[1] - t),
            -w * (g[1] * g[2]),
        ];
    }
    out
}

/// The reference's confined free energy, as its own constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfinedLdg {
    /// `A0`, the overall scale of the bulk nematic energy.
    pub a0: f64,
    /// `chi_0` and `chi_s` in `chi(phi) = chi_0 + chi_s phi`.
    pub chi_0: f64,
    pub chi_s: f64,
    /// Elastic constant `kappa`, one-constant approximation.
    pub kappa: f64,
    /// Anchoring strength `W`, tangential when positive.
    pub w_anchor: f64,
    /// Rotational diffusion constant `Gamma`.
    pub gamma: f64,
}

impl Default for ConfinedLdg {
    /// The constants stated in arXiv:2607.10234's Methods.
    fn default() -> Self {
        Self {
            a0: 0.5,
            chi_0: 2.4,
            chi_s: 0.4,
            kappa: 0.04,
            w_anchor: 0.2,
            gamma: 0.3,
        }
    }
}

/// The molecular field of a nematic confined by a phase field.
///
/// `H = kappa lap Q - 2a Q - 4c Tr(Q^2) Q - 3b [Q^2 - Tr(Q^2)/3 I] + H_anchor`,
/// with `(a, b, c)` evaluated per site from the local `chi(phi)`, so the
/// exterior sits below `chi_cr` and relaxes to `Q = 0` while the interior orders.
/// The bulk terms are the same ones [`crate::mol_field_3d`] carries, written
/// against its `f = a Tr(Q^2) + b Tr(Q^3) + c (Tr Q^2)^2` convention, with the
/// coefficients allowed to vary in space rather than fixed for the field.
///
/// `Gamma` is not applied here, matching `mol_field_3d`'s own split.
pub fn molecular_field_confined_3d(
    q: &QField3D,
    phase: &PhaseField3D,
    p: &ConfinedLdg,
) -> QField3D {
    assert_eq!(q.q.len(), phase.phi.len(), "Q and phi must share a grid");
    let lap = q.laplacian();
    let anchor = anchoring_molecular_field(phase, p.w_anchor);
    let mut out = QField3D::zeros(q.nx, q.ny, q.nz, q.dx);

    out.q.par_iter_mut().enumerate().for_each(|(k, out_k)| {
        let [q11, q12, q13, q22, q23] = q.q[k];
        let q33 = -(q11 + q22);
        let tr_q2 =
            q11 * q11 + q22 * q22 + q33 * q33 + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);

        let chi = LdgFromChi::chi(p.chi_0, p.chi_s, phase.phi[k]);
        let ldg = LdgFromChi::new(p.a0, chi);
        let (a, b, c) = (ldg.a_landau, ldg.b_landau, ldg.c_landau);

        // Q^2, needed only for the cubic term.
        let qq11 = q11 * q11 + q12 * q12 + q13 * q13;
        let qq12 = q11 * q12 + q12 * q22 + q13 * q23;
        let qq13 = q11 * q13 + q12 * q23 + q13 * q33;
        let qq22 = q12 * q12 + q22 * q22 + q23 * q23;
        let qq23 = q12 * q13 + q22 * q23 + q23 * q33;

        let lin = -2.0 * a - 4.0 * c * tr_q2;
        *out_k = [
            p.kappa * lap.q[k][0] + lin * q11 - 3.0 * b * qq11 + b * tr_q2 + anchor.q[k][0],
            p.kappa * lap.q[k][1] + lin * q12 - 3.0 * b * qq12 + anchor.q[k][1],
            p.kappa * lap.q[k][2] + lin * q13 - 3.0 * b * qq13 + anchor.q[k][2],
            p.kappa * lap.q[k][3] + lin * q22 - 3.0 * b * qq22 + b * tr_q2 + anchor.q[k][3],
            p.kappa * lap.q[k][4] + lin * q23 - 3.0 * b * qq23 + anchor.q[k][4],
        ];
    });
    out
}

/// One explicit relaxation step of the confined nematic, with no flow.
///
/// `Q <- Q + dt Gamma H`. The reference relaxes `Q` this way to its free-energy
/// minimum on a frozen `phi` before any activity is switched on, which is the
/// stage this reproduces.
pub fn relax_step_confined_3d(
    q: &mut QField3D,
    phase: &PhaseField3D,
    p: &ConfinedLdg,
    dt: f64,
) {
    let h = molecular_field_confined_3d(q, phase, p);
    let step = dt * p.gamma;
    q.q.par_iter_mut().zip(h.q.par_iter()).for_each(|(qk, hk)| {
        for c in 0..5 {
            qk[c] += step * hk[c];
        }
    });
}

/// The activity number `A = R / sqrt(K / zeta)`.
///
/// The ratio of the confining radius to the active length `sqrt(K/zeta)`, and
/// the parameter the reference's whole regime map is drawn against. `zeta <= 0`
/// returns `0`, the passive case having no active length.
pub fn activity_number(radius: f64, elastic_k: f64, zeta: f64) -> f64 {
    if zeta <= 0.0 {
        return 0.0;
    }
    radius / (elastic_k / zeta).sqrt()
}

#[cfg(test)]
mod confinement_tests {
    use super::*;

    // The reference's own constants (arXiv:2607.10234, Methods).
    const A_PHI: f64 = 0.01;
    const K_PHI: f64 = 0.14;
    const PHI_0: f64 = 2.0;
    const A0: f64 = 0.5;
    const CHI_IN: f64 = 3.2;
    const W_ANCHOR: f64 = 0.2;

    #[test]
    fn chi_mapping_reproduces_the_reference_order_parameter() {
        let ldg = LdgFromChi::new(A0, CHI_IN);
        let q = ldg.equilibrium_q();
        assert!(
            (q - 0.556).abs() < 5e-4,
            "equilibrium q = {q}, the reference states 0.556"
        );
    }

    #[test]
    fn chi_inside_the_cylinder_matches_the_reference() {
        // chi_0 = 2.4, chi_s = 0.4, phi = phi_0 = 2 gives chi = 3.2.
        assert!((LdgFromChi::chi(2.4, 0.4, PHI_0) - 3.2).abs() < 1e-12);
        // Outside, phi = 0 leaves chi = 2.4, below the chi_cr = 2.7 the
        // reference gives for the isotropic-to-nematic transition, so the
        // exterior is isotropic.
        assert!(LdgFromChi::chi(2.4, 0.4, 0.0) < 2.7);
    }

    #[test]
    fn isotropic_exterior_has_no_ordered_equilibrium() {
        // Below chi_cr the quadratic coefficient is positive and q = 0 is the
        // only minimum, so the mapping must not report an ordered state there.
        let outside = LdgFromChi::new(A0, 2.4);
        assert!(outside.a_landau > 0.0, "exterior should be isotropic");
    }

    #[test]
    fn interface_width_and_tension_match_the_reference_formulas() {
        let xi = PhaseField3D::interface_width(A_PHI, K_PHI);
        assert!((xi - (2.0 * K_PHI / A_PHI).sqrt()).abs() < 1e-12);
        let sigma = PhaseField3D::surface_tension(A_PHI, K_PHI);
        assert!((sigma - (8.0 * A_PHI * K_PHI / 9.0).sqrt()).abs() < 1e-12);
    }

    // An odd grid puts the box centre on a lattice point, so the cylinder axis
    // and the +x wall midpoint are sites rather than half-cell offsets.
    const N: usize = 49;
    const C: usize = 24;

    #[test]
    fn cylinder_is_full_inside_and_empty_outside() {
        let f = PhaseField3D::capped_cylinder(N, N, N, 1.0, 10.0, 24.0, PHI_0, 2.0);
        // The interface is diffuse, so the interior approaches phi_0 as tanh
        // rather than reaching it: at five interface widths from the wall the
        // shortfall is 1 - tanh(5), about 9e-5.
        let centre = f.phi[f.idx(C, C, C)];
        assert!(
            (centre - PHI_0).abs() < 2e-4,
            "centre reads {centre}, expected phi_0 to within the tanh tail"
        );
        assert!(f.phi[f.idx(1, 1, 1)] < 1e-6, "a far corner is not empty");
        // At the wall itself the distance is zero and tanh with it, so phi sits
        // at exactly half its interior value.
        let wall = f.phi[f.idx(C + 10, C, C)];
        assert!(
            (wall - 0.5 * PHI_0).abs() < 1e-12,
            "the interface midpoint reads {wall}"
        );
    }

    #[test]
    fn cylinder_endcaps_confine_along_the_axis() {
        let f = PhaseField3D::capped_cylinder(N, N, N, 1.0, 18.0, 20.0, PHI_0, 2.0);
        // On the axis, inside the half-length.
        assert!(f.phi[f.idx(C, C, C)] > 1.9);
        // On the axis, past the endcap.
        assert!(f.phi[f.idx(C, C, C + 18)] < 0.1);
    }

    #[test]
    fn the_gradient_points_out_of_the_cylinder() {
        let f = PhaseField3D::capped_cylinder(N, N, N, 1.0, 10.0, 24.0, PHI_0, 2.0);
        let g = f.gradient();
        // On the +x wall, phi falls with x, so the gradient points in -x, and
        // the other two components vanish by symmetry.
        let wall = g[f.idx(C + 10, C, C)];
        assert!(wall[0] < 0.0, "gradient {wall:?} does not fall outward");
        assert!(wall[1].abs() < 1e-15 && wall[2].abs() < 1e-15, "{wall:?}");
        // Deep inside phi is flat to the tanh tail, four orders below the wall.
        let inside = g[f.idx(C, C, C)];
        let mag = inside.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let at_wall = wall.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            mag < 1e-3 * at_wall,
            "the interior is not flat: {mag} against {at_wall} at the wall"
        );
    }

    #[test]
    fn anchoring_field_is_traceless_and_symmetric_by_construction() {
        let n = 32;
        let f = PhaseField3D::capped_cylinder(n, n, n, 1.0, 8.0, 16.0, PHI_0, 2.0);
        let h = anchoring_molecular_field(&f, W_ANCHOR);
        for k in 0..h.q.len() {
            let [h11, _, _, h22, _] = h.q[k];
            let h33 = -(h11 + h22);
            assert!(
                (h11 + h22 + h33).abs() < 1e-12,
                "site {k} has a non-traceless anchoring field"
            );
        }
    }

    #[test]
    fn anchoring_acts_at_the_wall_and_not_in_the_bulk() {
        let f = PhaseField3D::capped_cylinder(N, N, N, 1.0, 10.0, 24.0, PHI_0, 2.0);
        let h = anchoring_molecular_field(&f, W_ANCHOR);
        let mag = |k: usize| h.q[k].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let bulk = mag(f.idx(C, C, C));
        let wall = mag(f.idx(C + 10, C, C));
        // The anchoring field goes as |grad phi|^2, so the tanh tail leaves it
        // nonzero in the bulk but many orders below its value at the wall.
        assert!(wall > 0.0, "no anchoring at the wall");
        assert!(
            bulk < 1e-5 * wall,
            "anchoring acts in the bulk: {bulk} against {wall} at the wall"
        );
    }

    #[test]
    fn anchoring_drives_the_director_tangential() {
        // On the +x wall of a cylinder the surface normal is x, so the
        // anchoring energy must prefer a director in the yz plane. Compare the
        // energy density W (grad phi) . Q . (grad phi) for a director along the
        // normal against one along the axis.
        let n = 32;
        let f = PhaseField3D::capped_cylinder(n, n, n, 1.0, 8.0, 16.0, PHI_0, 2.0);
        let g = f.gradient()[f.idx(n / 2 + 8, n / 2, n / 2)];

        let energy = |dir: [f64; 3]| {
            let q_mag = 0.556;
            let t = 1.0 / 3.0;
            // (grad phi) . Q . (grad phi) with Q = q (nn - I/3).
            let n_dot_g = dir[0] * g[0] + dir[1] * g[1] + dir[2] * g[2];
            let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            W_ANCHOR * q_mag * (n_dot_g * n_dot_g - t * g2)
        };

        let normal = energy([1.0, 0.0, 0.0]);
        let tangential = energy([0.0, 0.0, 1.0]);
        assert!(
            tangential < normal,
            "tangential energy {tangential} is not below normal {normal}"
        );
    }

    #[test]
    fn anchoring_field_scales_linearly_with_strength() {
        let n = 24;
        let f = PhaseField3D::capped_cylinder(n, n, n, 1.0, 6.0, 12.0, PHI_0, 2.0);
        let one = anchoring_molecular_field(&f, 1.0);
        let two = anchoring_molecular_field(&f, 2.0);
        let worst = one
            .q
            .iter()
            .zip(&two.q)
            .flat_map(|(a, b)| (0..5).map(move |c| (2.0 * a[c] - b[c]).abs()))
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-15, "not linear in W: {worst}");
    }

    #[test]
    fn relaxation_orders_the_interior_and_leaves_the_exterior_isotropic() {
        // Inside the cylinder chi exceeds chi_cr and the nematic orders;
        // outside it falls below and Q relaxes to zero.
        let n = 32;
        let p = ConfinedLdg::default();
        let phase = PhaseField3D::capped_cylinder(n, n, n, 1.0, 8.0, 20.0, PHI_0, 2.0);
        let mut q = QField3D::random_perturbation(n, n, n, 1.0, 0.05, 11);

        for _ in 0..4000 {
            relax_step_confined_3d(&mut q, &phase, &p, 0.05);
        }

        let order = |k: usize| {
            let [q11, q12, q13, q22, q23] = q.q[k];
            let q33 = -(q11 + q22);
            (q11 * q11 + q22 * q22 + q33 * q33
                + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23))
                .sqrt()
        };
        let inside = order(phase.idx(16, 16, 16));
        let outside = order(phase.idx(1, 1, 1));
        assert!(
            inside > 10.0 * outside.max(1e-12),
            "interior order {inside} is not above exterior {outside}"
        );
        assert!(inside.is_finite() && inside > 0.1, "interior did not order");
    }

    #[test]
    fn relaxation_lowers_the_free_energy_monotonically() {
        // A gradient flow on a frozen phi must not increase the energy it
        // descends, which is the check that the molecular field is the
        // variation of the energy it claims to be.
        let n = 24;
        let p = ConfinedLdg::default();
        let phase = PhaseField3D::capped_cylinder(n, n, n, 1.0, 6.0, 14.0, PHI_0, 2.0);
        let mut q = QField3D::random_perturbation(n, n, n, 1.0, 0.05, 3);

        let energy = |q: &QField3D| {
            let mut e = 0.0;
            let grad_phi = phase.gradient();
            for k in 0..q.q.len() {
                let [q11, q12, q13, q22, q23] = q.q[k];
                let q33 = -(q11 + q22);
                let tr_q2 = q11 * q11 + q22 * q22 + q33 * q33
                    + 2.0 * (q12 * q12 + q13 * q13 + q23 * q23);
                let tr_q3 = {
                    let m = [
                        [q11, q12, q13],
                        [q12, q22, q23],
                        [q13, q23, q33],
                    ];
                    let mut t = 0.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            for l in 0..3 {
                                t += m[i][j] * m[j][l] * m[l][i];
                            }
                        }
                    }
                    t
                };
                let chi = LdgFromChi::chi(p.chi_0, p.chi_s, phase.phi[k]);
                let ldg = LdgFromChi::new(p.a0, chi);
                e += ldg.a_landau * tr_q2 + ldg.b_landau * tr_q3 + ldg.c_landau * tr_q2 * tr_q2;
                let g = grad_phi[k];
                e += p.w_anchor
                    * (g[0] * (q11 * g[0] + q12 * g[1] + q13 * g[2])
                        + g[1] * (q12 * g[0] + q22 * g[1] + q23 * g[2])
                        + g[2] * (q13 * g[0] + q23 * g[1] + q33 * g[2]));
            }
            // Elastic term, one-constant, same 6-point stencil as the field.
            let lap = q.laplacian();
            for k in 0..q.q.len() {
                let mut d = 0.0;
                for c in 0..5 {
                    d += q.q[k][c] * lap.q[k][c];
                }
                // integrate by parts: (kappa/2)|grad Q|^2 -> -(kappa/2) Q.lap Q
                e -= 0.5 * p.kappa * (2.0 * d + q.q[k][0] * lap.q[k][3]);
            }
            e
        };

        let mut previous = energy(&q);
        for _ in 0..40 {
            for _ in 0..20 {
                relax_step_confined_3d(&mut q, &phase, &p, 0.02);
            }
            let now = energy(&q);
            assert!(
                now <= previous + 1e-9 * previous.abs().max(1.0),
                "energy rose from {previous} to {now}"
            );
            previous = now;
        }
    }

    #[test]
    fn activity_number_is_the_radius_over_the_active_length() {
        // A = R / sqrt(K/zeta). At K = 0.04 the reference's own elastic
        // constant, a radius of 20 and zeta chosen to put the active length at
        // 2 gives A = 10.
        let k = 0.04;
        let zeta = k / 4.0; // active length sqrt(K/zeta) = 2
        assert!((activity_number(20.0, k, zeta) - 10.0).abs() < 1e-12);
        // Passive has no active length and no activity number.
        assert_eq!(activity_number(20.0, k, 0.0), 0.0);
    }
}
