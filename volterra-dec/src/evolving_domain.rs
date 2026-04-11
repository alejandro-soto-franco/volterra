//! EvolvingDomain: DecDomain extended with mesh motion and operator re-assembly.
//!
//! When vertex positions change (due to shape evolution or prescribed deformation),
//! ALL geometric quantities must update: Hodge stars, Laplacian, dual areas,
//! edge lengths, curvatures, normals, and connection transport matrices.
//!
//! The `deform` method moves vertices and rebuilds everything. This is the
//! expensive O(n_vertices) path; incremental updates are a future optimisation.

use nalgebra::SVector;

use cartan_core::bundle::{CovLaplacian, EdgeTransport2D};
use cartan_core::Manifold;
use cartan_dec::{DecError, Operators};

use crate::domain::DecDomain;

/// DecDomain extended with mesh motion and automatic operator rebuild.
///
/// Wraps a `DecDomain` and adds:
/// - Discrete Levi-Civita connection (`EdgeTransport2D` for 2-manifolds)
/// - Covariant Laplacian stencil
/// - Automatic re-assembly after vertex displacement
pub struct EvolvingDomain<M: Manifold> {
    /// The underlying static domain (mesh + operators + geometry).
    pub domain: DecDomain<M>,
    /// Discrete Levi-Civita connection (SO(2) transport per edge).
    pub transport: EdgeTransport2D,
    /// Covariant Laplacian stencil (cotangent weights + dual areas + adjacency).
    pub cov_lap: CovLaplacian,
    /// Step counter (number of deformations applied).
    pub deform_count: usize,
}

impl<M> EvolvingDomain<M>
where
    M: Manifold<Point = SVector<f64, 3>, Tangent = SVector<f64, 3>>,
{
    /// Build an evolving domain from a mesh and manifold.
    ///
    /// Assembles all DEC operators AND the discrete Levi-Civita connection.
    pub fn new(mesh: cartan_dec::Mesh<M, 3, 2>, manifold: M) -> Result<Self, DecError> {
        let domain = DecDomain::new(mesh, manifold)?;
        let transport = cartan_dec::levi_civita_2d(&domain.mesh, &domain.manifold);

        let star1: Vec<f64> = (0..domain.n_edges())
            .map(|i| domain.ops.hodge.star1()[i])
            .collect();
        let cov_lap = CovLaplacian::new(
            domain.n_vertices(),
            &transport.edges,
            &star1,
            &domain.dual_areas,
        );

        Ok(Self {
            domain,
            transport,
            cov_lap,
            deform_count: 0,
        })
    }

    /// Move vertex positions and rebuild ALL derived quantities.
    ///
    /// This is the expensive path: O(n_vertices) recomputation of Hodge stars,
    /// Laplacian, dual areas, edge lengths, connection angles, and covariant
    /// Laplacian stencil. Call once per timestep after shape evolution.
    pub fn deform(&mut self, new_positions: &[SVector<f64, 3>]) -> Result<(), DecError> {
        let nv = self.domain.n_vertices();
        assert_eq!(new_positions.len(), nv, "position count mismatch");

        // Update vertex positions.
        for (i, pos) in new_positions.iter().enumerate() {
            self.domain.mesh.vertices[i] = *pos;
        }

        // Rebuild DEC operators from the new geometry.
        self.domain.ops = Operators::from_mesh_generic(&self.domain.mesh, &self.domain.manifold)?;

        // Rebuild dual areas and edge lengths.
        let star0 = self.domain.ops.hodge.star0();
        for i in 0..nv {
            self.domain.dual_areas[i] = star0[i];
        }
        let ne = self.domain.n_edges();
        for e in 0..ne {
            self.domain.edge_lengths[e] = self.domain.mesh.boundary_volume(&self.domain.manifold, e);
        }

        // Rebuild Levi-Civita connection.
        self.transport = cartan_dec::levi_civita_2d(&self.domain.mesh, &self.domain.manifold);

        // Rebuild covariant Laplacian stencil.
        let star1: Vec<f64> = (0..ne)
            .map(|i| self.domain.ops.hodge.star1()[i])
            .collect();
        self.cov_lap = CovLaplacian::new(
            nv,
            &self.transport.edges,
            &star1,
            &self.domain.dual_areas,
        );

        self.deform_count += 1;
        Ok(())
    }

    /// Displace vertices along their normals by `v_n[i] * dt`.
    ///
    /// Convenience method for shape evolution: applies normal displacement,
    /// projects back to the manifold, and rebuilds operators.
    ///
    /// `v_n` is the normal velocity at each vertex (scalar, positive = outward).
    /// `normals` are the unit outward normals at each vertex.
    pub fn displace_normal(
        &mut self,
        v_n: &[f64],
        normals: &[[f64; 3]],
        dt: f64,
    ) -> Result<(), DecError> {
        let nv = self.domain.n_vertices();
        assert_eq!(v_n.len(), nv);
        assert_eq!(normals.len(), nv);

        let new_positions: Vec<SVector<f64, 3>> = (0..nv)
            .map(|i| {
                let p = &self.domain.mesh.vertices[i];
                let displacement = SVector::from([
                    v_n[i] * normals[i][0] * dt,
                    v_n[i] * normals[i][1] * dt,
                    v_n[i] * normals[i][2] * dt,
                ]);
                // Move and project back to manifold.
                let moved = p + displacement;
                self.domain.manifold.project_point(&moved)
            })
            .collect();

        self.deform(&new_positions)
    }

    /// Convenience: number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.domain.n_vertices()
    }

    /// Convenience: number of edges.
    pub fn n_edges(&self) -> usize {
        self.domain.n_edges()
    }
}
