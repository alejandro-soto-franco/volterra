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

    /// Recompute vertex normals, mean curvature, Gaussian curvature,
    /// and Laplacian of mean curvature from current mesh geometry.
    ///
    /// Call after `deform()` if you need curvature data for the shape equation.
    pub fn recompute_curvatures(&mut self) {
        let nv = self.domain.n_vertices();
        let verts = &self.domain.mesh.vertices;
        let tris = &self.domain.mesh.simplices;
        let boundaries = &self.domain.mesh.boundaries;

        // 1. Vertex normals (area-weighted face normals).
        let mut normals = vec![[0.0_f64; 3]; nv];
        for tri in tris {
            let p0 = verts[tri[0]];
            let p1 = verts[tri[1]];
            let p2 = verts[tri[2]];
            let e01 = p1 - p0;
            let e02 = p2 - p0;
            let fn_vec = [
                e01[1] * e02[2] - e01[2] * e02[1],
                e01[2] * e02[0] - e01[0] * e02[2],
                e01[0] * e02[1] - e01[1] * e02[0],
            ];
            for &idx in tri {
                for d in 0..3 { normals[idx][d] += fn_vec[d]; }
            }
        }
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-14 {
                n[0] /= len; n[1] /= len; n[2] /= len;
            }
        }
        self.domain.vertex_normals = normals;

        // 2. Gaussian curvature (angle defect / dual area).
        let mut angle_sum = vec![0.0_f64; nv];
        for tri in tris {
            for local in 0..3 {
                let vi = tri[local];
                let vj = tri[(local + 1) % 3];
                let vk = tri[(local + 2) % 3];
                let pi = verts[vi];
                let pj = verts[vj];
                let pk = verts[vk];
                let a = pj - pi;
                let b = pk - pi;
                let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
                if na > 1e-30 && nb > 1e-30 {
                    let cos_a = (dot / (na * nb)).clamp(-1.0, 1.0);
                    angle_sum[vi] += cos_a.acos();
                }
            }
        }
        for v in 0..nv {
            let a = self.domain.dual_areas[v];
            if a > 1e-30 {
                self.domain.gaussian_curvatures[v] =
                    (2.0 * std::f64::consts::PI - angle_sum[v]) / a;
            }
        }

        // 3. Mean curvature (cotangent formula: H = (1/2A) * sum_e w_e * (xj - xi) . n).
        let star1 = self.domain.ops.hodge.star1();
        let mut h_sum = vec![0.0_f64; nv];
        for (e, &[v0, v1]) in boundaries.iter().enumerate() {
            let w = star1[e];
            let edge = verts[v1] - verts[v0];
            // Mean curvature normal: contribution from this edge.
            // H_v0 += w * (edge . n_v0) / (2 * A_v0)
            let dot_v0 = edge[0] * self.domain.vertex_normals[v0][0]
                + edge[1] * self.domain.vertex_normals[v0][1]
                + edge[2] * self.domain.vertex_normals[v0][2];
            let dot_v1 = -(edge[0] * self.domain.vertex_normals[v1][0]
                + edge[1] * self.domain.vertex_normals[v1][1]
                + edge[2] * self.domain.vertex_normals[v1][2]);
            h_sum[v0] += w * dot_v0;
            h_sum[v1] += w * dot_v1;
        }
        for v in 0..nv {
            let a = self.domain.dual_areas[v];
            if a > 1e-30 {
                self.domain.mean_curvatures[v] = h_sum[v] / (2.0 * a);
            }
        }

        // 4. Laplacian of mean curvature (apply scalar Laplace-Beltrami to H).
        let h_vec = nalgebra::DVector::from_column_slice(&self.domain.mean_curvatures);
        let lap_h = self.domain.ops.apply_laplace_beltrami(&h_vec);
        for v in 0..nv {
            self.domain.laplacian_mean_curvatures[v] = lap_h[v];
        }
    }

    /// Compute the normal velocity from the shape equation.
    ///
    /// v_n = (1/eta_s) * (-kb * (lap_H + 2(H-H0)(H^2-K)) + tension * H)
    ///
    /// Requires `recompute_curvatures()` to have been called first.
    pub fn shape_velocity(
        &self,
        kb: f64,
        h0: &[f64],
        tension: f64,
        eta_surface: f64,
    ) -> Vec<f64> {
        let nv = self.domain.n_vertices();
        let h = &self.domain.mean_curvatures;
        let k = &self.domain.gaussian_curvatures;
        let lap_h = &self.domain.laplacian_mean_curvatures;

        (0..nv)
            .map(|v| {
                let dh = h[v] - h0[v];
                let helfrich_force = -kb * (lap_h[v] + 2.0 * dh * (h[v] * h[v] - k[v]));
                let tension_force = tension * h[v];
                (helfrich_force + tension_force) / eta_surface
            })
            .collect()
    }

    /// Compute the v_n correction to the Q-tensor material derivative.
    ///
    /// For U1Spin2 on a 2-manifold: (bQ + Qb)_traceless = 2H * Q.
    /// The correction to dQ/dt is: v_n * 2H * Q at each vertex.
    ///
    /// Returns delta_q1, delta_q2 to be added to the RHS of the Q evolution.
    pub fn vn_correction(
        &self,
        v_n: &[f64],
        q1: &[f64],
        q2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let nv = self.domain.n_vertices();
        let h = &self.domain.mean_curvatures;

        let dq1: Vec<f64> = (0..nv)
            .map(|v| v_n[v] * 2.0 * h[v] * q1[v])
            .collect();
        let dq2: Vec<f64> = (0..nv)
            .map(|v| v_n[v] * 2.0 * h[v] * q2[v])
            .collect();

        (dq1, dq2)
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
