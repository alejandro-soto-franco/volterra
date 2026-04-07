//! Integration tests for the volterra-dec crate.

use cartan_dec::FlatMesh;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::SVector;

use volterra_dec::helfrich::{helfrich_energy, HelfrichParams};
use volterra_dec::variational::{baoab_ba_step, compute_dt, kinetic_energy};
use volterra_dec::DecDomain;

#[test]
fn dec_domain_construction() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(4);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    assert!(domain.n_vertices() > 0);
    assert!(domain.n_edges() > 0);
    assert!(domain.n_faces() > 0);
    assert_eq!(domain.dual_areas.len(), domain.n_vertices());
    assert_eq!(domain.edge_lengths.len(), domain.n_edges());
}

#[test]
fn dual_areas_sum_to_total_area() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(8);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let total: f64 = domain.dual_areas.iter().sum();
    // Unit square has area 1.0.
    assert!(
        (total - 1.0).abs() < 1e-12,
        "dual areas should sum to 1.0, got {total}"
    );
}

#[test]
fn helfrich_energy_zero_for_flat_mesh() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(4);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    let params = HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![0.0; nv],
    };

    let energy = helfrich_energy(&domain, &params);
    assert!(
        energy.abs() < 1e-12,
        "flat mesh should have zero Helfrich energy, got {energy}"
    );
}

#[test]
fn helfrich_energy_with_spontaneous_curvature() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(4);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    // Non-zero spontaneous curvature on a flat mesh produces positive energy.
    let params = HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![1.0; nv],
    };

    let energy = helfrich_energy(&domain, &params);
    assert!(
        energy > 0.0,
        "flat mesh with H0 != 0 should have positive Helfrich energy"
    );
}

#[test]
fn baoab_step_preserves_vertex_count() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(3);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    let params = HelfrichParams {
        kb: 1.0,
        kg: 0.0,
        h0: vec![0.0; nv],
    };

    let mut positions = domain.mesh.vertices.clone();
    let mut momenta: Vec<SVector<f64, 2>> = vec![SVector::zeros(); nv];
    let masses = domain.dual_areas.clone();

    baoab_ba_step(
        &manifold,
        &mut positions,
        &mut momenta,
        &masses,
        &params,
        &domain,
        0.001,
    );

    assert_eq!(positions.len(), nv);
    assert_eq!(momenta.len(), nv);
}

#[test]
fn kinetic_energy_zero_at_rest() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(3);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    let momenta: Vec<SVector<f64, 2>> = vec![SVector::zeros(); nv];
    let masses: Vec<f64> = vec![1.0; nv];

    let ke = kinetic_energy(&manifold, &domain.mesh.vertices, &momenta, &masses);
    assert!(ke.abs() < 1e-15, "kinetic energy at rest should be zero");
}

#[test]
fn kinetic_energy_positive_with_momentum() {
    let manifold = Euclidean::<2>;
    let mesh = FlatMesh::unit_square_grid(3);
    let domain = DecDomain::new(mesh, manifold).unwrap();
    let nv = domain.n_vertices();

    let momenta: Vec<SVector<f64, 2>> = vec![SVector::<f64, 2>::new(1.0, 0.0); nv];
    let masses: Vec<f64> = vec![1.0; nv];

    let ke = kinetic_energy(&manifold, &domain.mesh.vertices, &momenta, &masses);
    assert!(ke > 0.0, "kinetic energy with nonzero momenta should be positive");
}

#[test]
fn compute_dt_respects_cap() {
    let dt = compute_dt(0.1, 1.0, 0.005, 0.5, 0.5);
    assert!(dt <= 0.005, "dt should not exceed dt_max");
    assert!(dt > 0.0, "dt should be positive");
}

#[test]
fn compute_dt_with_zero_force() {
    let dt = compute_dt(0.1, 0.0, 1.0, 0.5, 0.5);
    // With zero force, dt_force = MAX, so dt = min(dt_max, dt_diff).
    let dt_diff = 0.5 * 0.01;
    assert!(
        (dt - dt_diff).abs() < 1e-15,
        "with zero force, dt should equal dt_diff = {dt_diff}, got {dt}"
    );
}
