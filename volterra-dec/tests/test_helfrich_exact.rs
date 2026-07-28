//! Exact discrete Helfrich gradient: the defining property and the sphere regression.
//!
//! The governing requirement is that the returned force be the exact gradient of
//! the discrete energy with respect to vertex positions. Once that holds, a
//! discrete equilibrium is a true critical point at every resolution, and the
//! flow dissipates energy by construction.
//!
//! The companion regression pins the failure this replaces: `shape_velocity`
//! builds its force by applying the DEC Laplacian to a reconstructed pointwise
//! mean curvature, whose per-vertex error is O(h) and non-smooth, so the h^-2
//! amplification of the Laplacian makes the spurious velocity on an exact unit
//! sphere DIVERGE under refinement (rms 1.2 -> 20.4, max 1.8 -> 559 over
//! icosphere levels 1..5).

use nalgebra::Vector3;
use volterra_dec::bending::{BendingParams, bending_energy, bending_gradient};
use volterra_dec::mesh_gen::icosphere;

/// Unit-sphere mesh as raw positions plus triangles.
fn sphere_mesh(level: usize) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    let m = icosphere(level);
    (m.vertices.clone(), m.simplices.clone())
}

/// Deterministic smooth perturbation, so the test mesh is a generic surface
/// rather than one sitting on a symmetry orbit.
fn perturb(verts: &mut [Vector3<f64>], eps: f64) {
    for (i, v) in verts.iter_mut().enumerate() {
        let t = i as f64;
        *v += eps
            * Vector3::new(
                (3.0 * t).sin(),
                (5.0 * t + 1.0).sin(),
                (7.0 * t + 2.0).sin(),
            );
    }
}

/// Central-difference gradient of `bending_energy`, the oracle for the analytic one.
fn fd_gradient(
    verts: &[Vector3<f64>],
    tris: &[[usize; 3]],
    params: &BendingParams,
    h: f64,
) -> Vec<Vector3<f64>> {
    let mut g = vec![Vector3::zeros(); verts.len()];
    let mut work = verts.to_vec();
    for i in 0..verts.len() {
        for c in 0..3 {
            let orig = work[i][c];
            work[i][c] = orig + h;
            let ep = bending_energy(&work, tris, params);
            work[i][c] = orig - h;
            let em = bending_energy(&work, tris, params);
            work[i][c] = orig;
            g[i][c] = (ep - em) / (2.0 * h);
        }
    }
    g
}

#[test]
fn bending_gradient_matches_finite_differences() {
    let (mut verts, tris) = sphere_mesh(2);
    perturb(&mut verts, 0.05);
    let params = BendingParams {
        kappa: 1.0,
        kappa_bar: 0.0,
        h0: vec![0.0; verts.len()],
        tension: 0.0,
    };

    let analytic = bending_gradient(&verts, &tris, &params);
    let numeric = fd_gradient(&verts, &tris, &params, 1e-6);

    let scale = numeric
        .iter()
        .fold(0.0_f64, |a, g| a.max(g.norm()))
        .max(1e-30);
    let err = analytic
        .iter()
        .zip(&numeric)
        .fold(0.0_f64, |a, (x, y)| a.max((x - y).norm()));

    assert!(
        err / scale < 1e-6,
        "analytic gradient must match central differences: rel err {:.3e} (scale {scale:.3e})",
        err / scale
    );
    assert!(
        scale > 1e-3,
        "oracle gradient is degenerate ({scale:.3e}); the test would pass vacuously"
    );
}

/// Root-mean-square bending gradient on an exact unit sphere, where the
/// continuum answer is identically zero.
fn sphere_residual(level: usize) -> f64 {
    let (verts, tris) = sphere_mesh(level);
    let params = BendingParams {
        kappa: 1.0,
        kappa_bar: 0.0,
        h0: vec![0.0; verts.len()],
        tension: 0.0,
    };
    let g = bending_gradient(&verts, &tris, &params);
    (g.iter().map(|v| v.norm_squared()).sum::<f64>() / g.len() as f64).sqrt()
}

#[test]
fn bending_residual_converges_under_refinement() {
    // The failure this replaces diverges here: shape_velocity gives rms
    // 1.2, 2.6, 5.1, 10.2, 20.4 over levels 1..5. An exact gradient of a
    // consistent discrete energy must go the other way.
    let r: Vec<f64> = (1..=4).map(sphere_residual).collect();
    for (i, w) in r.windows(2).enumerate() {
        assert!(
            w[1] < w[0],
            "residual must fall under refinement: level {} -> {} gave {:.4e} -> {:.4e} (all: {:?})",
            i + 1,
            i + 2,
            w[0],
            w[1],
            r
        );
    }
}

#[test]
fn gradient_descent_dissipates_energy_monotonically() {
    // The payoff of an exact gradient: descent on the discrete energy cannot
    // increase it. The old force is not the gradient of anything, so it carries
    // no such guarantee.
    let (mut verts, tris) = sphere_mesh(3);
    perturb(&mut verts, 0.03);
    let params = BendingParams {
        kappa: 1.0,
        kappa_bar: 0.0,
        h0: vec![0.0; verts.len()],
        tension: 0.0,
    };

    let residual = |v: &[Vector3<f64>]| {
        let g = bending_gradient(v, &tris, &params);
        (g.iter().map(|x| x.norm_squared()).sum::<f64>() / g.len() as f64).sqrt()
    };

    let e0 = bending_energy(&verts, &tris, &params);
    let r0 = residual(&verts);
    let mut prev = e0;
    let dt = 2e-4;

    for step in 0..200 {
        let g = bending_gradient(&verts, &tris, &params);
        for (v, gi) in verts.iter_mut().zip(&g) {
            *v -= dt * gi;
        }
        let e = bending_energy(&verts, &tris, &params);
        assert!(
            e <= prev + 1e-12,
            "energy rose at step {step}: {prev:.12} -> {e:.12}"
        );
        prev = e;
    }

    let r1 = residual(&verts);
    assert!(
        r1 < r0,
        "descent should reduce the residual: {r0:.4e} -> {r1:.4e}"
    );
    assert!(
        prev < e0,
        "descent should reduce the energy: {e0:.6} -> {prev:.6}"
    );
}

#[test]
fn saddle_splay_term_has_zero_gradient() {
    // sum_v (2 pi - angle defect) = 2 pi chi is an exact combinatorial identity
    // on a closed mesh, so kappa_bar can do no work at fixed topology.
    let (mut verts, tris) = sphere_mesh(2);
    perturb(&mut verts, 0.05);
    let params = BendingParams {
        kappa: 0.0,
        kappa_bar: 1.0,
        h0: vec![0.0; verts.len()],
        tension: 0.0,
    };
    let g = bending_gradient(&verts, &tris, &params);
    let max = g.iter().fold(0.0_f64, |a, v| a.max(v.norm()));
    assert!(
        max < 1e-12,
        "saddle-splay gradient must vanish, got {max:.3e}"
    );

    // and the energy itself is 2 pi kappa_bar chi, with chi = 2 for a sphere
    let e = bending_energy(&verts, &tris, &params);
    let expected = 2.0 * std::f64::consts::PI * 2.0;
    assert!(
        (e - expected).abs() < 1e-10,
        "saddle-splay energy should be 2 pi chi = {expected:.6}, got {e:.6}"
    );
}

#[test]
fn unit_sphere_bending_energy_is_two_pi_kappa() {
    // E = (kappa/2) * H^2 * area = (kappa/2) * 1 * 4 pi = 2 pi kappa.
    let (verts, tris) = sphere_mesh(4);
    let params = BendingParams {
        kappa: 1.0,
        kappa_bar: 0.0,
        h0: vec![0.0; verts.len()],
        tension: 0.0,
    };
    let e = bending_energy(&verts, &tris, &params);
    let expected = 2.0 * std::f64::consts::PI;
    assert!(
        (e - expected).abs() / expected < 2e-3,
        "unit-sphere bending energy should be 2 pi = {expected:.6}, got {e:.6}"
    );
}

#[test]
fn spontaneous_curvature_sign_is_outward_positive() {
    // On an outward-oriented unit sphere H = +1, so H0 = +1 costs nothing and
    // H0 = -1 costs (kappa/2)(2)^2 (4 pi) = 8 pi kappa. The reverse ordering
    // would mean a positive H0 bends towards the inward normal.
    let (verts, tris) = sphere_mesh(4);
    let nv = verts.len();
    let matched = bending_energy(
        &verts,
        &tris,
        &BendingParams {
            kappa: 1.0,
            kappa_bar: 0.0,
            h0: vec![1.0; nv],
            tension: 0.0,
        },
    );
    let opposed = bending_energy(
        &verts,
        &tris,
        &BendingParams {
            kappa: 1.0,
            kappa_bar: 0.0,
            h0: vec![-1.0; nv],
            tension: 0.0,
        },
    );
    assert!(
        matched < 1e-2,
        "H0 = +1 should match an outward unit sphere, got E = {matched:.4e}"
    );
    let expected = 8.0 * std::f64::consts::PI;
    assert!(
        (opposed - expected).abs() / expected < 2e-3,
        "H0 = -1 should cost 8 pi = {expected:.4}, got {opposed:.4}"
    );
}

#[test]
fn tension_gradient_shrinks_a_sphere() {
    // Descent moves along -grad, so a sphere under tension alone must contract:
    // grad . x_hat > 0 at every vertex.
    let (verts, tris) = sphere_mesh(3);
    let params = BendingParams {
        kappa: 0.0,
        kappa_bar: 0.0,
        h0: vec![0.0; verts.len()],
        tension: 1.0,
    };
    let g = bending_gradient(&verts, &tris, &params);
    for (i, (gi, xi)) in g.iter().zip(&verts).enumerate() {
        let radial = gi.dot(&xi.normalize());
        assert!(
            radial > 0.0,
            "vertex {i}: tension gradient must point outward so descent shrinks, got {radial:.4e}"
        );
    }
}
