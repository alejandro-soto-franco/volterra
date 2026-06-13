//! Tests for the optional spatial active-coefficient field zeta(x).
//!
//! The active stress is sigma = zeta(x) Q(x). With a scalar zeta_eff this is a
//! uniform multiply; with a per-vertex zeta_field it is a real-space product.
//! A uniform field must reproduce the scalar result exactly (FFT linearity); a
//! spatially modulated field must change the driven flow.

use volterra_core::ActiveNematicParams;
use volterra_fields::QField2D;
use volterra_solver::stokes_solve;

fn perturbed_q(nx: usize, ny: usize, dx: f64) -> QField2D {
    let mut q = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 / nx as f64;
            let y = j as f64 / ny as f64;
            let a = 0.3 * (2.0 * std::f64::consts::PI * x).sin();
            let b = 0.3 * (2.0 * std::f64::consts::PI * y).cos();
            q.push([a, b]);
        }
    }
    QField2D { q, nx, ny, dx }
}

#[test]
fn uniform_zeta_field_matches_scalar() {
    let (nx, ny, dx) = (16, 16, 1.0);
    let mut p = ActiveNematicParams::default_test();
    p.nx = nx;
    p.ny = ny;
    p.dx = dx;
    let q = perturbed_q(nx, ny, dx);

    let v_scalar = stokes_solve(&q, &p);

    let mut p_field = p.clone();
    p_field.zeta_field = Some(vec![p.zeta_eff; nx * ny]);
    let v_field = stokes_solve(&q, &p_field);

    for (a, b) in v_scalar.v.iter().zip(v_field.v.iter()) {
        assert!(
            (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9,
            "uniform zeta_field must reproduce the scalar flow: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn spatial_zeta_field_changes_flow() {
    let (nx, ny, dx) = (16, 16, 1.0);
    let mut p = ActiveNematicParams::default_test();
    p.nx = nx;
    p.ny = ny;
    p.dx = dx;
    let q = perturbed_q(nx, ny, dx);

    let v_scalar = stokes_solve(&q, &p);

    // Active in the left half, quiescent in the right half.
    let mut field = vec![0.0_f64; nx * ny];
    for i in 0..nx / 2 {
        for j in 0..ny {
            field[i * ny + j] = p.zeta_eff;
        }
    }
    let mut p_field = p.clone();
    p_field.zeta_field = Some(field);
    let v_field = stokes_solve(&q, &p_field);

    let diff: f64 = v_scalar
        .v
        .iter()
        .zip(v_field.v.iter())
        .map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
        .sum();
    assert!(diff > 1e-6, "spatial zeta should change the flow, diff={diff}");
}
