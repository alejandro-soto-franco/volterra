use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::{molecular_field_dec, QFieldDec};

#[test]
fn zeros_has_zero_order() {
    let q = QFieldDec::zeros(10);
    assert_eq!(q.n_vertices, 10);
    assert!(q.mean_order_param() == 0.0);
}

#[test]
fn uniform_order_parameter() {
    let q = QFieldDec::uniform(10, 0.3, 0.0);
    let s = q.mean_order_param();
    // S = 2 * |q1| = 0.6
    assert!((s - 0.6).abs() < 1e-12, "expected 0.6, got {s}");
}

#[test]
fn lichnerowicz_layout_roundtrip() {
    let q = QFieldDec::random_perturbation(20, 0.1, 42);
    let v = q.to_lichnerowicz_layout();
    assert_eq!(v.len(), 60); // 3 * 20
    let q2 = QFieldDec::from_lichnerowicz_layout(&v);

    let diff: f64 = q
        .q1
        .iter()
        .zip(&q2.q1)
        .chain(q.q2.iter().zip(&q2.q2))
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff < 1e-14, "roundtrip failed: diff = {diff}");
}

#[test]
fn lichnerowicz_layout_traceless() {
    let q = QFieldDec::random_perturbation(10, 0.5, 99);
    let v = q.to_lichnerowicz_layout();
    let nv = q.n_vertices;
    // Q_xx + Q_yy should be zero (traceless)
    for i in 0..nv {
        let qxx = v[i];
        let qyy = v[2 * nv + i];
        assert!(
            (qxx + qyy).abs() < 1e-14,
            "vertex {i}: Q_xx + Q_yy = {} (not traceless)",
            qxx + qyy
        );
    }
}

#[test]
fn add_and_scale() {
    let a = QFieldDec::uniform(5, 1.0, 0.0);
    let b = QFieldDec::uniform(5, 0.0, 1.0);
    let c = a.add(&b.scale(2.0));
    assert!((c.q1[0] - 1.0).abs() < 1e-14);
    assert!((c.q2[0] - 2.0).abs() < 1e-14);
}

#[test]
fn trace_q_squared_value() {
    let q = QFieldDec::uniform(5, 0.3, 0.4);
    let tr = q.trace_q_squared();
    let expected = 2.0 * (0.3_f64.powi(2) + 0.4_f64.powi(2));
    assert!(
        (tr[0] - expected).abs() < 1e-14,
        "expected {expected}, got {}",
        tr[0]
    );
}

// ── Molecular field tests ────────────────────────────────────────────────────

#[test]
fn molecular_field_zero_at_zero_q() {
    let mesh = FlatMesh::unit_square_grid(8);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);
    let params = ActiveNematicParams::default_test();
    let q = QFieldDec::zeros(mesh.n_vertices());

    let h = molecular_field_dec(&q, &params, &ops, None);
    let h_norm: f64 = h.q1.iter().chain(&h.q2).map(|x| x.abs()).sum();
    assert!(h_norm < 1e-12, "H(Q=0) should vanish, got norm = {h_norm}");
}

#[test]
fn molecular_field_uniform_q_no_laplacian() {
    // For a uniform Q on a flat mesh, the Laplacian vanishes.
    // H should be purely the bulk term: (-a_eff - 2c Tr(Q^2)) Q
    let mesh = FlatMesh::unit_square_grid(16);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);
    let params = ActiveNematicParams::default_test();
    let a_eff = params.a_eff();
    let _c = params.c_landau;

    let q0 = 0.001;
    let q = QFieldDec::uniform(mesh.n_vertices(), q0, 0.0);
    let h = molecular_field_dec(&q, &params, &ops, None);

    // At small Q, cubic term is negligible:
    // H_q1 approx -a_eff * q0 = (zeta_eff/2 - a_landau) * q0
    let expected = -a_eff * q0;
    // Check a few interior vertices (boundary may have stencil edge effects)
    let nv = mesh.n_vertices();
    let n_side = 17; // unit_square_grid(16) has 17 vertices per side
    let mut max_err = 0.0_f64;
    for i in 2..(n_side - 2) {
        for j in 2..(n_side - 2) {
            let idx = j * n_side + i;
            if idx < nv {
                let err = (h.q1[idx] - expected).abs();
                max_err = max_err.max(err);
            }
        }
    }
    assert!(
        max_err < 0.01 * expected.abs(),
        "interior H_q1 should be close to {expected}, max_err = {max_err}"
    );
}
