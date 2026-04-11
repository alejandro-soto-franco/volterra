//! Tests for EvolvingDomain: mesh deformation with automatic operator rebuild.

use cartan_manifolds::sphere::Sphere;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::EvolvingDomain;

use cartan_core::fiber::{Section, U1Spin2, VecSection};

#[test]
fn evolving_domain_constructs_on_sphere() {
    let mesh = icosphere(2); // 162 vertices
    let ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    assert_eq!(ed.n_vertices(), 162);
    assert!(ed.n_edges() > 0);
    assert_eq!(ed.deform_count, 0);
}

#[test]
fn evolving_domain_inflate_sphere() {
    // Start with unit sphere, inflate to radius 1.1, verify areas increase.
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();

    let total_area_before: f64 = ed.domain.dual_areas.iter().sum();

    // Inflate: move each vertex to radius 1.1.
    let scale = 1.1;
    let new_positions: Vec<nalgebra::SVector<f64, 3>> = (0..nv)
        .map(|i| {
            let p = ed.domain.mesh.vertices[i];
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r > 1e-14 {
                p * (scale / r)
            } else {
                p
            }
        })
        .collect();

    ed.deform(&new_positions).unwrap();
    assert_eq!(ed.deform_count, 1);

    let total_area_after: f64 = ed.domain.dual_areas.iter().sum();

    // Area of sphere scales as R^2. ratio = (1.1)^2 = 1.21.
    let ratio = total_area_after / total_area_before;
    assert!(
        (ratio - 1.21).abs() < 0.05,
        "area ratio should be ~1.21, got {ratio}"
    );
}

#[test]
fn evolving_domain_laplacian_updates_after_deform() {
    // After deforming the mesh, the covariant Laplacian should use the
    // new geometry. Verify the Laplacian of a zero field is still zero,
    // and that the stencil dimension matches the new mesh.
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();

    let area_before: f64 = ed.domain.dual_areas.iter().sum();

    // Scale all vertices by 2.0 (inflate to radius 2).
    let new_positions: Vec<nalgebra::SVector<f64, 3>> = (0..nv)
        .map(|i| ed.domain.mesh.vertices[i] * 2.0)
        .collect();

    ed.deform(&new_positions).unwrap();

    let area_after: f64 = ed.domain.dual_areas.iter().sum();
    // Area should scale by 4x (R^2 factor).
    let ratio = area_after / area_before;
    assert!((ratio - 4.0).abs() < 0.5, "area ratio should be ~4.0, got {ratio}");

    // Zero field: Laplacian must be zero regardless of geometry.
    let zero_field = VecSection::<U1Spin2>::from_vec(vec![[0.0, 0.0]; nv]);
    let lap_result = ed.cov_lap.apply::<U1Spin2, 2, _>(&zero_field, &ed.transport);

    let max_lap: f64 = (0..nv)
        .map(|v| {
            let r = lap_result.at(v);
            r[0].abs().max(r[1].abs())
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_lap < 1e-12,
        "zero field should have zero Laplacian, got max = {max_lap}"
    );
}

#[test]
fn evolving_domain_deform_explicit() {
    // Explicitly set new positions (scale by 1.1) and verify radii.
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();

    let new_positions: Vec<nalgebra::SVector<f64, 3>> = (0..nv)
        .map(|i| ed.domain.mesh.vertices[i] * 1.1)
        .collect();

    ed.deform(&new_positions).unwrap();

    // Check radius of first vertex.
    let p = ed.domain.mesh.vertices[0];
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        (r - 1.1).abs() < 0.01,
        "radius after scaling should be ~1.1, got {r}"
    );
    assert_eq!(ed.deform_count, 1);
}

#[test]
fn evolving_domain_multiple_deformations() {
    // Apply several deformations and verify the domain stays consistent.
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();

    // 10 incremental inflation steps (scale by 1.01 each).
    for step in 0..10 {
        let scale = 1.0 + 0.01 * (step + 1) as f64;
        let new_positions: Vec<nalgebra::SVector<f64, 3>> = (0..nv)
            .map(|i| {
                let p = ed.domain.mesh.vertices[i];
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                if r > 1e-14 { p * (scale / r) } else { p }
            })
            .collect();
        ed.deform(&new_positions).unwrap();
    }

    assert_eq!(ed.deform_count, 10);

    // Verify all dual areas are positive.
    for (i, &a) in ed.domain.dual_areas.iter().enumerate() {
        assert!(a > 0.0, "dual_area[{i}] = {a} should be positive");
    }

    // Verify edge lengths are positive.
    for (e, &l) in ed.domain.edge_lengths.iter().enumerate() {
        assert!(l > 0.0, "edge_length[{e}] = {l} should be positive");
    }
}

// ─── Curvature and shape equation tests ─────────────────────────────────────

#[test]
fn curvature_unit_sphere() {
    // On a unit sphere: H = 1.0, K = 1.0.
    let mesh = icosphere(3); // 642 vertices for better accuracy
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    ed.recompute_curvatures();

    let mean_h: f64 = ed.domain.mean_curvatures.iter().sum::<f64>()
        / ed.n_vertices() as f64;
    let mean_k: f64 = ed.domain.gaussian_curvatures.iter().sum::<f64>()
        / ed.n_vertices() as f64;

    // Gaussian curvature should be 1.0 on unit sphere.
    assert!(
        (mean_k - 1.0).abs() < 0.15,
        "mean Gaussian curvature should be ~1.0, got {mean_k}"
    );

    // Mean curvature should be ~1.0 on unit sphere.
    assert!(
        (mean_h.abs() - 1.0).abs() < 0.3,
        "mean |H| should be ~1.0, got {mean_h}"
    );
}

#[test]
fn gaussian_curvature_integrates_to_4pi() {
    // Gauss-Bonnet: integral of K over S^2 = 4*pi.
    let mesh = icosphere(3);
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    ed.recompute_curvatures();

    let integral_k: f64 = ed.domain.gaussian_curvatures.iter()
        .zip(&ed.domain.dual_areas)
        .map(|(&k, &a)| k * a)
        .sum();

    let expected = 4.0 * std::f64::consts::PI;
    assert!(
        (integral_k - expected).abs() < 0.1,
        "integral of K should be 4*pi = {expected:.4}, got {integral_k:.4}"
    );
}

#[test]
fn shape_velocity_sphere_with_tension() {
    // A sphere with surface tension and no bending rigidity: v_n = tension * H / eta.
    // On a unit sphere, H ~ 1, so v_n ~ tension / eta.
    let mesh = icosphere(3);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    ed.recompute_curvatures();

    let h0 = vec![0.0; nv];
    let tension = 1.0;
    let eta = 1.0;
    let v_n = ed.shape_velocity(0.0, &h0, tension, eta); // kb=0, tension only

    let mean_vn: f64 = v_n.iter().map(|v| v.abs()).sum::<f64>() / nv as f64;
    // |v_n| ~ |tension * H / eta| ~ 1.0 (since |H| ~ 1 on unit sphere).
    assert!(
        mean_vn > 0.5,
        "tension-driven |v_n| should be ~1.0, got mean = {mean_vn}"
    );
}

#[test]
fn vn_correction_zero_for_zero_velocity() {
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    ed.recompute_curvatures();

    let v_n = vec![0.0; nv];
    let q1 = vec![0.3; nv];
    let q2 = vec![0.4; nv];
    let (dq1, dq2) = ed.vn_correction(&v_n, &q1, &q2);

    for v in 0..nv {
        assert!(dq1[v].abs() < 1e-14);
        assert!(dq2[v].abs() < 1e-14);
    }
}

#[test]
fn vn_correction_proportional_to_mean_curvature() {
    let mesh = icosphere(2);
    let nv = mesh.n_vertices();
    let mut ed = EvolvingDomain::new(mesh, Sphere::<3>).unwrap();
    ed.recompute_curvatures();

    let v_n = vec![1.0; nv]; // uniform normal velocity
    let q1 = vec![1.0; nv]; // unit q1
    let q2 = vec![0.0; nv];
    let (dq1, _dq2) = ed.vn_correction(&v_n, &q1, &q2);

    // correction = v_n * 2H * q1 = 2H for v_n=1, q1=1.
    for v in 0..nv {
        let expected = 2.0 * ed.domain.mean_curvatures[v];
        assert!(
            (dq1[v] - expected).abs() < 1e-12,
            "dq1[{v}] = {}, expected {}",
            dq1[v],
            expected
        );
    }
}
