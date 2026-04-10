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
//! - Mem3DG. Biophysical Reports, 2022.

use cartan_core::Manifold;
use nalgebra::SVector;

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
/// The normal component of the shape derivative gives:
///
///   F_v = -kb * A_v * (Delta_s(H) + 2(H - H0)(H^2 - K)) * n_v
///
/// where Delta_s(H) is the Laplacian of mean curvature (precomputed in
/// `domain.laplacian_mean_curvatures`), and n_v is the vertex normal.
///
/// Requires `domain.vertex_normals` and `domain.laplacian_mean_curvatures`
/// to be filled before calling. Returns one tangent vector per vertex.
pub fn helfrich_forces<M: Manifold<Point = SVector<f64, 3>, Tangent = SVector<f64, 3>>>(
    domain: &DecDomain<M>,
    params: &HelfrichParams,
) -> Vec<M::Tangent> {
    let nv = domain.n_vertices();
    let h = &domain.mean_curvatures;
    let k = &domain.gaussian_curvatures;
    let normals = &domain.vertex_normals;
    let lap_h = &domain.laplacian_mean_curvatures;

    (0..nv)
        .map(|v| {
            let h0_v = params.h0[v];
            let dh = h[v] - h0_v;
            // Shape equation: f_n = -kb * A_v * (lap_H + 2 * (H - H0) * (H^2 - K))
            let f_scalar = -params.kb * domain.dual_areas[v]
                * (lap_h[v] + 2.0 * dh * (h[v] * h[v] - k[v]));
            let n = normals[v];
            SVector::<f64, 3>::new(f_scalar * n[0], f_scalar * n[1], f_scalar * n[2])
        })
        .collect()
}
