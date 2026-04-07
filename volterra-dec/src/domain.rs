//! DecDomain: the central mesh-operator bundle for DEC simulations.
//!
//! Bundles a triangle mesh with its precomputed DEC operators (exterior
//! derivative, Hodge star, Laplacian) and derived geometric quantities
//! (dual cell areas, edge lengths, mean and Gaussian curvature).

use cartan_core::Manifold;
use cartan_dec::{DecError, Mesh, Operators};

/// The assembled discrete domain: mesh + precomputed operators.
///
/// `DecDomain` bundles a triangle mesh with its DEC operators (exterior
/// derivative, Hodge star, Laplacian) and derived geometric quantities
/// (dual cell areas, edge lengths, mean/Gaussian curvature).
pub struct DecDomain<M: Manifold> {
    /// The triangle mesh.
    pub mesh: Mesh<M, 3, 2>,
    /// The manifold on which the mesh lives.
    pub manifold: M,
    /// Assembled DEC operators (Laplace-Beltrami, exterior derivative, Hodge star).
    pub ops: Operators<M, 3, 2>,
    /// Dual cell areas (star0 diagonal), one per vertex.
    pub dual_areas: Vec<f64>,
    /// Edge lengths, one per boundary (edge).
    pub edge_lengths: Vec<f64>,
    /// Mean curvature at each vertex (initially zero, filled by caller).
    pub mean_curvatures: Vec<f64>,
    /// Gaussian curvature at each vertex (initially zero, filled by caller).
    pub gaussian_curvatures: Vec<f64>,
}

impl<M: Manifold> DecDomain<M> {
    /// Build a `DecDomain` from a mesh and manifold.
    ///
    /// Assembles all DEC operators (exterior derivative, Hodge star,
    /// Laplace-Beltrami) and extracts dual cell areas and edge lengths.
    /// Curvature arrays are initialised to zero; the caller fills them
    /// via a separate curvature estimation pass.
    pub fn new(mesh: Mesh<M, 3, 2>, manifold: M) -> Result<Self, DecError> {
        let ops = Operators::from_mesh_generic(&mesh, &manifold)?;
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();

        // Extract dual cell areas from the star0 diagonal.
        let star0 = ops.hodge.star0();
        let dual_areas: Vec<f64> = (0..nv).map(|i| star0[i]).collect();

        // Compute primal edge lengths via the manifold metric.
        let edge_lengths: Vec<f64> = (0..ne)
            .map(|e| mesh.boundary_volume(&manifold, e))
            .collect();

        Ok(Self {
            mesh,
            manifold,
            ops,
            dual_areas,
            edge_lengths,
            mean_curvatures: vec![0.0; nv],
            gaussian_curvatures: vec![0.0; nv],
        })
    }

    /// Number of vertices in the mesh.
    pub fn n_vertices(&self) -> usize {
        self.mesh.n_vertices()
    }

    /// Number of edges (boundary faces) in the mesh.
    pub fn n_edges(&self) -> usize {
        self.mesh.n_boundaries()
    }

    /// Number of triangles (top-level simplices) in the mesh.
    pub fn n_faces(&self) -> usize {
        self.mesh.n_simplices()
    }
}
