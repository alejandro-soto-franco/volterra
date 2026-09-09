//! Hydrodynamics in three dimensions: the flow a nematic's own activity drives.
//!
//! The wet runner adds a Stokes solve to the dry one and advects `Q` by the
//! result, so the disclination lines move in a flow they themselves generate.
//! Three things follow that a test can hold it to.
//!
//! With the activity switched off the fluid is at rest and the wet path has to
//! reproduce the dry one exactly, which is what separates a coupling from a
//! perturbation. With it on, the velocity is divergence free, since the solver
//! applies the Leray projector in Fourier space and incompressibility is
//! therefore exact rather than approximate. And the flow has to do something:
//! the field a wet run reaches has to differ from the dry one's.

use volterra_core::{ActiveNematicParams3D, QField3D};
use volterra_fd::runner_3d::{run_dry_active_nematic_3d, run_wet_active_nematic_3d};

fn uniaxial(n: [f64; 3], q: f64) -> [f64; 5] {
    let t = 1.0 / 3.0;
    [
        q * (n[0] * n[0] - t),
        q * (n[0] * n[1]),
        q * (n[0] * n[2]),
        q * (n[1] * n[1] - t),
        q * (n[1] * n[2]),
    ]
}

/// A `+1/2` disclination loop, which is the texture the flow acts on.
fn seeded_loop(n: usize, r: f64, q_eq: f64) -> QField3D {
    let c = (n as f64 - 1.0) / 2.0;
    let mut field = QField3D::zeros(n, n, n, 1.0);
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, l as f64 - c);
                let rho = (x * x + y * y).sqrt().max(1e-9);
                let half = 0.5 * z.atan2(rho - r);
                let dir = [half.cos() * x / rho, half.cos() * y / rho, half.sin()];
                field.q[((i * n) + j) * n + l] = uniaxial(dir, q_eq);
            }
        }
    }
    field
}

fn params(n: usize) -> ActiveNematicParams3D {
    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("volterra_wet_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn without_activity_the_fluid_stays_at_rest() {
    let n = 16;
    let mut p = params(n);
    p.zeta_eff = 0.0;
    let q0 = seeded_loop(n, 5.0, p.equilibrium_q());

    let (_, vel, stats) = run_wet_active_nematic_3d(&q0, &p, 4, 4, &scratch("rest"), false);
    let fastest = vel
        .u
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .fold(0.0_f64, f64::max);

    assert!(fastest < 1e-12, "an inactive nematic drove a flow of {fastest:.3e}");
    assert!(stats.last().unwrap().max_speed < 1e-12);
}

#[test]
fn without_activity_the_wet_path_reproduces_the_dry_one() {
    // The check that the coupling is a coupling: with no flow to couple to, the
    // wet path has to land where the dry one does.
    //
    // Not bit for bit, and the reason is worth stating. A zero velocity still
    // goes through the advection and co-rotation terms, which are mathematically
    // zero and computed anyway, so the wet path sums the same quantities in a
    // different order. The difference that leaves is a few units in the last
    // place, twelve orders below the field itself, and an exact-equality
    // assertion here would be testing the order of the arithmetic rather than
    // the physics.
    let n = 16;
    let mut p = params(n);
    p.zeta_eff = 0.0;
    let q0 = seeded_loop(n, 5.0, p.equilibrium_q());

    let (dry, _) = run_dry_active_nematic_3d(&q0, &p, 6, 6, &scratch("dry"), false);
    let (wet, _, _) = run_wet_active_nematic_3d(&q0, &p, 6, 6, &scratch("wetdry"), false);

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (a, b) in dry.q.iter().zip(wet.q.iter()) {
        for c in 0..5 {
            worst = worst.max((a[c] - b[c]).abs());
            scale = scale.max(a[c].abs());
        }
    }
    assert!(
        worst < 1e-13 * scale,
        "the two paths parted by {worst:.3e} against a field of {scale:.3e}"
    );
}

#[test]
fn the_active_flow_is_divergence_free() {
    // The Leray projector removes the compressible part in Fourier space, so
    // this is exact to the accuracy of the difference stencil that measures it,
    // rather than a tolerance the solver was tuned to meet.
    let n = 24;
    let p = params(n);
    let q0 = seeded_loop(n, 7.0, p.equilibrium_q());
    let (_, vel, _) = run_wet_active_nematic_3d(&q0, &p, 2, 2, &scratch("div"), false);

    let at = |i: usize, j: usize, l: usize| vel.u[((i % n) * n + (j % n)) * n + (l % n)];
    let mut worst: f64 = 0.0;
    let mut speed: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let d = (at((i + 1) % n, j, l)[0] - at((i + n - 1) % n, j, l)[0]
                    + at(i, (j + 1) % n, l)[1]
                    - at(i, (j + n - 1) % n, l)[1]
                    + at(i, j, (l + 1) % n)[2]
                    - at(i, j, (l + n - 1) % n)[2])
                    / (2.0 * p.dx);
                worst = worst.max(d.abs());
                let u = at(i, j, l);
                speed = speed.max((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt());
            }
        }
    }
    assert!(speed > 1e-6, "the active flow was too weak to test, {speed:.3e}");
    assert!(
        worst < 1e-9 * speed / p.dx,
        "divergence {worst:.3e} against a speed of {speed:.3e}"
    );
}

#[test]
fn activity_drives_a_flow_that_changes_the_field() {
    let n = 24;
    let p = params(n);
    let q0 = seeded_loop(n, 7.0, p.equilibrium_q());

    let (dry, _) = run_dry_active_nematic_3d(&q0, &p, 40, 40, &scratch("dry2"), false);
    let (wet, vel, stats) = run_wet_active_nematic_3d(&q0, &p, 40, 40, &scratch("wet2"), false);

    let fastest = vel
        .u
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(fastest > 1e-6, "activity drove no flow, {fastest:.3e}");

    let drift: f64 = dry
        .q
        .iter()
        .zip(wet.q.iter())
        .map(|(a, b)| (0..5).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    assert!(drift > 1e-8, "the flow left the field unchanged, drift {drift:.3e}");
    assert!(stats.last().unwrap().max_speed > 0.0);
}
