//! The external field reaching the planar molecular field on a DEC mesh.
//!
//! `ActiveNematicParams::field_term` is tested on its own in volterra-core.
//! What is tested here is that it arrives, unscaled and at every vertex, and
//! that a run setting no field is unchanged to the last bit.

use cartan_dec::Operators;
use cartan_dec::mesh::FlatMesh;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::{QField, molecular_field_dec};

fn setup() -> (FlatMesh, Euclidean<2>) {
    (FlatMesh::unit_square_grid(12), Euclidean::<2>)
}

#[test]
fn no_field_leaves_the_molecular_field_untouched() {
    let (mesh, manifold) = setup();
    let ops = Operators::from_mesh(&mesh, &manifold);
    let q = QField::uniform(mesh.n_vertices(), 0.21, -0.07);

    let base = ActiveNematicParams::default_test();
    let mut coefficient_only = base.clone();
    coefficient_only.chi_a = 4.0; // a coefficient with no field behind it

    let a = molecular_field_dec(&q, &base, &ops, None, 0.9);
    let b = molecular_field_dec(&q, &coefficient_only, &ops, None, 0.9);
    for (x, y) in
        a.q1.iter()
            .zip(b.q1.iter())
            .chain(a.q2.iter().zip(b.q2.iter()))
    {
        assert_eq!(x, y, "a coefficient with zero amplitude moved the field");
    }
}

#[test]
fn the_field_term_arrives_unscaled_at_every_vertex() {
    let (mesh, manifold) = setup();
    let ops = Operators::from_mesh(&mesh, &manifold);
    let q = QField::uniform(mesh.n_vertices(), 0.21, -0.07);

    let off = ActiveNematicParams::default_test();
    let mut on = off.clone();
    on.chi_a = 0.8;
    on.b0 = 1.5;
    on.omega_b = 2.3;

    let t = 0.37;
    let without = molecular_field_dec(&q, &off, &ops, None, t);
    let with = molecular_field_dec(&q, &on, &ops, None, t);
    let [e1, e2] = on.field_term(t);
    assert!(
        e1.abs() > 1e-6,
        "the fixture must set a term worth measuring"
    );

    for v in 0..mesh.n_vertices() {
        assert!(
            (with.q1[v] - without.q1[v] - e1).abs() < 1e-12,
            "vertex {v}: q1 moved by {}, expected {e1}",
            with.q1[v] - without.q1[v]
        );
        assert!(
            (with.q2[v] - without.q2[v] - e2).abs() < 1e-12,
            "vertex {v}: q2 moved by {}, expected {e2}",
            with.q2[v] - without.q2[v]
        );
    }
}

#[test]
fn a_rotating_drive_moves_the_field_between_times() {
    let (mesh, manifold) = setup();
    let ops = Operators::from_mesh(&mesh, &manifold);
    let q = QField::uniform(mesh.n_vertices(), 0.1, 0.0);
    let mut p = ActiveNematicParams::default_test();
    p.chi_a = 0.8;
    p.b0 = 1.5;
    p.omega_b = 1.0;

    let early = molecular_field_dec(&q, &p, &ops, None, 0.0);
    let later = molecular_field_dec(&q, &p, &ops, None, std::f64::consts::FRAC_PI_4);
    let moved = (early.q1[0] - later.q1[0]).abs() + (early.q2[0] - later.q2[0]).abs();
    assert!(
        moved > 1e-6,
        "a rotating drive must move the molecular field with time, moved {moved}"
    );
}

#[test]
fn a_static_drive_does_not_move_the_field_between_times() {
    let (mesh, manifold) = setup();
    let ops = Operators::from_mesh(&mesh, &manifold);
    let q = QField::uniform(mesh.n_vertices(), 0.1, 0.0);
    let mut p = ActiveNematicParams::default_test();
    p.chi_a = 0.8;
    p.b0 = 1.5;
    p.omega_b = 0.0;

    let early = molecular_field_dec(&q, &p, &ops, None, 0.0);
    let later = molecular_field_dec(&q, &p, &ops, None, 11.0);
    for (x, y) in early
        .q1
        .iter()
        .zip(later.q1.iter())
        .chain(early.q2.iter().zip(later.q2.iter()))
    {
        assert_eq!(x, y, "a static drive made the field depend on the time");
    }
}
