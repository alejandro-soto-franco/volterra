//! Boundary conditions for confined active nematics on DEC meshes.
//!
//! ## Strong planar anchoring (Dirichlet on Q)
//!
//! At each boundary vertex, Q is fixed to the anchoring value:
//!   Q_ij = S_0 (n_i n_j - delta_ij / 2)
//!
//! where n is the prescribed anchoring direction and S_0 is the
//! equilibrium scalar order parameter.
//!
//! ## No-slip (Dirichlet on stream function)
//!
//! For the Stokes solver: psi = 0 at boundary vertices.

use crate::epitrochoid::ConfinedMesh;
use crate::QFieldDec;

/// Apply strong planar anchoring at boundary vertices.
///
/// Sets Q at each boundary vertex to Q_ij = s0 * (n_i n_j - delta_ij / 2),
/// where n is the anchoring direction from the `ConfinedMesh`.
///
/// In the traceless representation (q1 = Q_xx, q2 = Q_xy):
///   q1 = s0 * (n_x^2 - 1/2) = s0/2 * (2 n_x^2 - 1) = s0/2 * cos(2*theta)
///   q2 = s0 * n_x * n_y     = s0/2 * sin(2*theta)
///
/// where theta = atan2(n_y, n_x).
pub fn apply_strong_anchoring(q: &mut QFieldDec, confined: &ConfinedMesh, s0: f64) {
    for (idx, &bv) in confined.boundary_vertices.iter().enumerate() {
        let [nx, ny] = confined.anchoring_directions[idx];
        // Q_xx = s0 * (n_x^2 - 1/2), Q_xy = s0 * n_x * n_y
        q.q1[bv] = s0 * (nx * nx - 0.5);
        q.q2[bv] = s0 * nx * ny;
    }
}

/// Enforce strong anchoring after each time step.
///
/// This is called after the RK4 step to reset boundary vertices to
/// their prescribed values (overwriting any drift from the time integration).
pub fn enforce_anchoring(q: &mut QFieldDec, confined: &ConfinedMesh, s0: f64) {
    apply_strong_anchoring(q, confined, s0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epitrochoid::disk_mesh;

    #[test]
    fn anchoring_sets_boundary_q() {
        let cm = disk_mesh(5.0, 1.5, 32, 1.0);
        let nv = cm.mesh.n_vertices();
        let mut q = QFieldDec::zeros(nv);
        let s0 = 1.0;

        apply_strong_anchoring(&mut q, &cm, s0);

        // Boundary vertices should have non-zero Q.
        for &bv in &cm.boundary_vertices {
            let s = 2.0 * (q.q1[bv].powi(2) + q.q2[bv].powi(2)).sqrt();
            assert!(
                (s - s0).abs() < 1e-10,
                "boundary vertex {bv}: S = {s}, expected {s0}"
            );
        }
    }

    #[test]
    fn anchoring_preserves_interior() {
        let cm = disk_mesh(5.0, 1.5, 32, 1.0);
        let nv = cm.mesh.n_vertices();
        let mut q = QFieldDec::uniform(nv, 0.1, 0.2);

        apply_strong_anchoring(&mut q, &cm, 1.0);

        // Interior vertices (index >= 32) should be unchanged.
        for i in 32..nv {
            assert!(
                (q.q1[i] - 0.1).abs() < 1e-14,
                "interior vertex {i}: q1 = {}, expected 0.1",
                q.q1[i]
            );
        }
    }
}
