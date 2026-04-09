//! Curvature correction callbacks for the Lichnerowicz Laplacian on 2-manifolds.
//!
//! The Weitzenboeck endomorphism for a traceless symmetric rank-2 tensor
//! on a 2D surface with Gaussian curvature K is W = -4K Id (acting on the
//! 3-component [Q_xx, Q_xy, Q_yy] representation).
//!
//! Verified via SymPy for constant-curvature n-manifolds with n = 2, 3, 4, 5:
//! the general formula is W = -2n kappa h. For n = 2, this gives -4K h.
//!
//! These callbacks are passed to
//! [`cartan_dec::Operators::apply_lichnerowicz_laplacian`].

/// Return a curvature correction callback for a surface of constant
/// Gaussian curvature K.
///
/// The callback returns a 3x3 diagonal matrix with all entries equal to -4K,
/// corresponding to the Weitzenboeck endomorphism for traceless symmetric
/// 2-tensors on a 2D surface.
pub fn constant_curvature_2d(kappa: f64) -> impl Fn(usize) -> [[f64; 3]; 3] {
    let w = -4.0 * kappa;
    move |_vertex: usize| [[w, 0.0, 0.0], [0.0, w, 0.0], [0.0, 0.0, w]]
}

/// Return a curvature correction callback using per-vertex Gaussian curvature.
///
/// `gaussian_curvatures` is a vector of K(v) values, one per vertex.
/// The Weitzenboeck endomorphism at vertex v is -4 K(v) Id.
pub fn variable_curvature_2d(gaussian_curvatures: Vec<f64>) -> impl Fn(usize) -> [[f64; 3]; 3] {
    move |vertex: usize| {
        let w = -4.0 * gaussian_curvatures[vertex];
        [[w, 0.0, 0.0], [0.0, w, 0.0], [0.0, 0.0, w]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_curvature_flat() {
        let cb = constant_curvature_2d(0.0);
        let m = cb(0);
        for i in 0..3 {
            for j in 0..3 {
                assert!(m[i][j].abs() < 1e-15, "flat space correction should be zero");
            }
        }
    }

    #[test]
    fn constant_curvature_sphere() {
        let kappa = 1.0; // unit sphere
        let cb = constant_curvature_2d(kappa);
        let m = cb(0);
        assert!((m[0][0] - (-4.0)).abs() < 1e-15);
        assert!((m[1][1] - (-4.0)).abs() < 1e-15);
        assert!((m[2][2] - (-4.0)).abs() < 1e-15);
        assert!(m[0][1].abs() < 1e-15);
    }

    #[test]
    fn variable_curvature_per_vertex() {
        let curvatures = vec![0.0, 1.0, -0.5];
        let cb = variable_curvature_2d(curvatures);

        let m0 = cb(0);
        assert!(m0[0][0].abs() < 1e-15, "vertex 0: K=0, correction should be 0");

        let m1 = cb(1);
        assert!((m1[0][0] - (-4.0)).abs() < 1e-15, "vertex 1: K=1, correction should be -4");

        let m2 = cb(2);
        assert!((m2[0][0] - 2.0).abs() < 1e-15, "vertex 2: K=-0.5, correction should be +2");
    }
}
