//! Helfrich bending energy and forces on triangle meshes.
//!
//! The Helfrich energy is:
//!
//!   E = integral( kb/2 * (H - H0)^2 + kg * K ) dA
//!
//! where H is mean curvature, K is Gaussian curvature, kb is bending rigidity,
//! kg is Gaussian curvature modulus, and H0 is spontaneous curvature.
//!
//! ## References
//!
//! - Helfrich, W. "Elastic properties of lipid bilayers." Z. Naturforsch C, 1973.
//! - Zhu, Lee, Rangamani. "Mem3DG." Biophysical Reports, 2022.

use cartan_core::Manifold;

use crate::domain::DecDomain;

/// Parameters for the Helfrich bending energy.
pub struct HelfrichParams {
    /// Bending rigidity (kb), units of energy.
    pub kb: f64,
    /// Gaussian curvature modulus (kg), units of energy.
    pub kg: f64,
    /// Spontaneous curvature at each vertex, one per vertex.
    pub h0: Vec<f64>,
}

/// Compute the total Helfrich bending energy over the mesh.
///
/// The discrete energy sums over all vertices:
///
///   E = sum_v A_v * [ kb/2 * (H_v - H0_v)^2 + kg * K_v ]
///
/// where A_v is the dual cell area, H_v is the discrete mean curvature,
/// K_v is the discrete Gaussian curvature, and H0_v is the spontaneous
/// curvature at vertex v.
pub fn helfrich_energy<M: Manifold>(domain: &DecDomain<M>, params: &HelfrichParams) -> f64 {
    let nv = domain.n_vertices();
    let mut energy = 0.0;
    for v in 0..nv {
        let a_v = domain.dual_areas[v];
        let h_v = domain.mean_curvatures[v];
        let k_v = domain.gaussian_curvatures[v];
        let h0_v = params.h0[v];
        let dh = h_v - h0_v;
        energy += a_v * (0.5 * params.kb * dh * dh + params.kg * k_v);
    }
    energy
}

/// Compute the Helfrich force per vertex (negative gradient of the energy).
///
/// For a discrete membrane, the force on vertex v is:
///
///   F_v = -dE/dx_v
///
/// The discrete gradient involves the shape operator and Laplace-Beltrami.
/// The current implementation returns zero forces as a placeholder; the full
/// analytical gradient (which requires differentiating the discrete mean
/// curvature with respect to vertex positions) will be added as a follow-up.
///
/// Returns one tangent vector per vertex.
pub fn helfrich_forces<M: Manifold>(
    domain: &DecDomain<M>,
    params: &HelfrichParams,
) -> Vec<M::Tangent> {
    let _ = params; // will be used by the analytical gradient
    let nv = domain.n_vertices();
    (0..nv)
        .map(|v| domain.manifold.zero_tangent(&domain.mesh.vertices[v]))
        .collect()
}
