// ~/volterra/volterra-solver/src/gauss_bonnet_3d.rs

//! Gauss-Bonnet Euler characteristic observable for the phi_l isosurface.
//!
//! ## Physics
//!
//! The Euler characteristic of the phi_l = 1/2 isosurface is computed via the
//! co-area formula (Federer 1959; Evans-Gariepy §3.4):
//!
//! ```text
//! chi_phi = (1/2pi) integral_V K_G(phi) |grad(phi)|  dV
//! ```
//!
//! Note: no epsilon factor in the integrand. For a tanh profile phi = 0.5*(1 + tanh((r-R)/eps)),
//! the normal integral of |grad phi| over the interface equals 1 (total variation of phi),
//! so the epsilon dependence cancels exactly. The 1/2pi prefactor comes from Gauss-Bonnet:
//! integral_S K_G dA = 2*pi*chi.
//!
//! where K_G(phi) is the Gaussian curvature of the level set, computed via the
//! bordered Hessian of the scalar field F(x) = phi(x) - 1/2:
//!
//! ```text
//!        ( 0     phi_x   phi_y   phi_z  )
//!        ( phi_x   phi_xx  phi_xy  phi_xz )
//! K_G = -det (             )  /  |grad(phi)|^4
//!        ( phi_y   phi_xy  phi_yy  phi_yz )
//!        ( phi_z   phi_xz  phi_yz  phi_zz )
//! ```
//!
//! All partial derivatives are approximated with second-order central finite
//! differences on the regular grid. The integral is approximated by a
//! sum over all grid vertices weighted by dx^3.
//!
//! ## Verification
//!
//! For a tanh-profile sphere of radius R with interface width epsilon, the discrete
//! sum converges to chi = 2.0 as the grid is refined (see unit tests).
//!
//! ## Gradient threshold
//!
//! Far from the interface |grad(phi)| approx 0. We set K_G = 0 at vertices where
//! |grad(phi)| < GRAD_THRESHOLD to avoid division by nearly-zero values.

use volterra_fields::ScalarField3D;

/// Minimum |grad(phi)| below which K_G is set to zero (far from the interface).
const GRAD_THRESHOLD: f64 = 1e-6;

/// Compute the Gauss-Bonnet Euler characteristic of the phi_l isosurface.
///
/// Uses the co-area formula chi_phi = (1/2pi) sum_v K_G(v) |grad(phi)(v)| dx^3.
/// All derivatives are second-order central differences; periodic BCs.
///
/// The epsilon parameter is retained for API compatibility with callers that have it
/// available from MarsParams3D.epsilon_ch, but the integral does not depend on it:
/// for a tanh(0->1) profile the normal integral of |grad phi| equals 1 exactly.
///
/// # Arguments
///
/// * `phi`      - Lipid concentration field phi_l on a periodic nx x ny x nz grid.
/// * `_epsilon` - Unused (interface half-width retained for API compatibility).
///
/// # Returns
///
/// chi_phi as a real number. For a topological sphere chi = 2; for a gyroid of genus 5,
/// chi = -8. Returns 0.0 for uniform fields (|grad(phi)| < GRAD_THRESHOLD everywhere).
pub fn gauss_bonnet_chi(phi: &ScalarField3D, _epsilon: f64) -> f64 {
    let nx = phi.nx;
    let ny = phi.ny;
    let nz = phi.nz;
    let dx = phi.dx;

    let mut chi_sum = 0.0_f64;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // First derivatives (central FD, periodic)
                let ip = (i + 1) % nx;
                let im = (i + nx - 1) % nx;
                let jp = (j + 1) % ny;
                let jm = (j + ny - 1) % ny;
                let kp = (k + 1) % nz;
                let km = (k + nz - 1) % nz;

                let phi_x = (phi.phi[phi.idx(ip, j,  k )] - phi.phi[phi.idx(im, j,  k )]) / (2.0*dx);
                let phi_y = (phi.phi[phi.idx(i,  jp, k )] - phi.phi[phi.idx(i,  jm, k )]) / (2.0*dx);
                let phi_z = (phi.phi[phi.idx(i,  j,  kp)] - phi.phi[phi.idx(i,  j,  km)]) / (2.0*dx);

                let grad2 = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z;
                let grad_mag = grad2.sqrt();

                // Skip vertices far from the interface
                if grad_mag < GRAD_THRESHOLD {
                    continue;
                }

                // Second derivatives (central FD)
                let phi_c  = phi.phi[phi.idx(i, j, k)];

                let phi_xx = (phi.phi[phi.idx(ip, j,  k )] - 2.0*phi_c + phi.phi[phi.idx(im, j,  k )]) / (dx*dx);
                let phi_yy = (phi.phi[phi.idx(i,  jp, k )] - 2.0*phi_c + phi.phi[phi.idx(i,  jm, k )]) / (dx*dx);
                let phi_zz = (phi.phi[phi.idx(i,  j,  kp)] - 2.0*phi_c + phi.phi[phi.idx(i,  j,  km)]) / (dx*dx);

                // Mixed partials via 4-point stencil
                let phi_xy = (phi.phi[phi.idx(ip, jp, k)] - phi.phi[phi.idx(ip, jm, k)]
                            - phi.phi[phi.idx(im, jp, k)] + phi.phi[phi.idx(im, jm, k)])
                            / (4.0*dx*dx);
                let phi_xz = (phi.phi[phi.idx(ip, j, kp)] - phi.phi[phi.idx(ip, j, km)]
                            - phi.phi[phi.idx(im, j, kp)] + phi.phi[phi.idx(im, j, km)])
                            / (4.0*dx*dx);
                let phi_yz = (phi.phi[phi.idx(i, jp, kp)] - phi.phi[phi.idx(i, jp, km)]
                            - phi.phi[phi.idx(i, jm, kp)] + phi.phi[phi.idx(i, jm, km)])
                            / (4.0*dx*dx);

                // Bordered Hessian determinant
                //
                // det([[0,   px,  py,  pz ],
                //      [px,  pxx, pxy, pxz],
                //      [py,  pxy, pyy, pyz],
                //      [pz,  pxz, pyz, pzz]])
                //
                // Expand along the first row:
                // = 0*M00 - px*M01 + py*M02 - pz*M03
                // where M0j are the 3x3 minors of the first row.

                // M01 = det([[px,  pxy, pxz],
                //            [py,  pyy, pyz],
                //            [pz,  pyz, pzz]])
                let m01 = phi_x * (phi_yy*phi_zz - phi_yz*phi_yz)
                        - phi_xy * (phi_y*phi_zz - phi_yz*phi_z)
                        + phi_xz * (phi_y*phi_yz - phi_yy*phi_z);

                // M02 = det([[px,  pxx, pxz],
                //            [py,  pxy, pyz],
                //            [pz,  pxz, pzz]])
                let m02 = phi_x * (phi_xy*phi_zz - phi_yz*phi_xz)
                        - phi_xx * (phi_y*phi_zz - phi_yz*phi_z)
                        + phi_xz * (phi_y*phi_xz - phi_xy*phi_z);

                // M03 = det([[px,  pxx, pxy],
                //            [py,  pxy, pyy],
                //            [pz,  pxz, pyz]])
                let m03 = phi_x * (phi_xy*phi_yz - phi_yy*phi_xz)
                        - phi_xx * (phi_y*phi_yz - phi_yy*phi_z)
                        + phi_xy * (phi_y*phi_xz - phi_xy*phi_z);

                let bh_det = -phi_x * m01 + phi_y * m02 - phi_z * m03;

                // K_G = -det(bordered Hessian) / |grad(phi)|^4
                let k_g = -bh_det / (grad2 * grad2);

                // Co-area weight: K_G |grad(phi)| dx^3
                // The co-area formula gives: (1/2pi) * integral K_G |grad phi| dV = chi.
                // (No epsilon factor: integral |grad phi| dr = 1 for tanh 0->1 profile.)
                chi_sum += k_g * grad_mag * dx * dx * dx;
            }
        }
    }

    chi_sum / (2.0 * std::f64::consts::PI)
}


// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use volterra_fields::ScalarField3D;

    #[test]
    fn test_uniform_field_chi_zero() {
        let phi = ScalarField3D::uniform(8, 8, 8, 1.0, 0.5);
        let chi = gauss_bonnet_chi(&phi, 1.0);
        assert!(chi.abs() < 1e-10, "uniform phi must give chi=0, got {}", chi);
    }

    #[test]
    fn test_sphere_chi_approx_two() {
        let n = 64usize;
        let dx = 1.0_f64;
        let r0 = 16.0_f64;
        let epsilon = 3.0_f64;
        let cx = n as f64 / 2.0;

        let mut phi = ScalarField3D::zeros(n, n, n, dx);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 - cx;
                    let y = j as f64 - cx;
                    let z = k as f64 - cx;
                    let r = (x*x + y*y + z*z).sqrt();
                    let idx = phi.idx(i, j, k);
                    phi.phi[idx] = 0.5 * (1.0 + ((r - r0) / epsilon).tanh());
                }
            }
        }

        let chi = gauss_bonnet_chi(&phi, epsilon);
        assert!(
            (chi - 2.0).abs() < 0.5,
            "sphere shell chi expected approx 2.0, got {:.4}",
            chi
        );
    }
}
