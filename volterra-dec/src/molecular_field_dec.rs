//! Molecular field H = -dF/dQ for active nematics on a DEC mesh.
//!
//! Uses the Lichnerowicz Laplacian from cartan-dec with an optional
//! curvature correction callback for curved surfaces.
//!
//! ## Molecular field
//!
//! ```text
//! H = K_frank * Delta_L Q  +  (zeta_eff/2 - a_landau) Q  -  2 c_landau Tr(Q^2) Q
//! ```
//!
//! where Delta_L is the Lichnerowicz Laplacian (connection Laplacian on
//! symmetric 2-tensors). On flat meshes (K = 0), Delta_L reduces to the
//! component-wise scalar Laplace-Beltrami.

use cartan_core::Manifold;
use cartan_dec::Operators;
use volterra_core::ActiveNematicParams;

use crate::QFieldDec;

/// Compute the molecular field H at each vertex on a DEC mesh.
///
/// `curvature_correction`: optional per-vertex Weitzenboeck endomorphism for the
/// Lichnerowicz Laplacian on curved surfaces. Pass `None` for flat meshes.
/// For a surface of constant Gaussian curvature K, pass
/// `Some(&constant_curvature_2d(K))` from [`crate::curvature_correction`].
pub fn molecular_field_dec<M: Manifold>(
    q: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> QFieldDec {
    let nv = q.n_vertices;
    let k_frank = params.k_r;
    let a_eff = params.a_eff(); // a_landau - zeta_eff/2
    let c = params.c_landau;

    // Elastic term: K_frank * Delta_L Q
    let q_layout = q.to_lichnerowicz_layout();
    let lap_q = ops.apply_lichnerowicz_laplacian(&q_layout, curvature_correction);
    let lap_field = QFieldDec::from_lichnerowicz_layout(&lap_q);

    // Bulk LdG terms.
    // H = K * lap(Q) - a_eff * Q - 2c * Tr(Q^2) * Q
    //   = K * lap(Q) + (zeta_eff/2 - a_landau) * Q - 2c * Tr(Q^2) * Q
    let tr_q2 = q.trace_q_squared();

    let bulk_linear = -a_eff;
    let mut h = QFieldDec::zeros(nv);
    for (i, &tr) in tr_q2.iter().enumerate() {
        let bulk = bulk_linear - 2.0 * c * tr;
        h.q1[i] = k_frank * lap_field.q1[i] + bulk * q.q1[i];
        h.q2[i] = k_frank * lap_field.q2[i] + bulk * q.q2[i];
    }

    h
}
