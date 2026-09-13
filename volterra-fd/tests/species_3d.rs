//! A transported composition field, which nucleation needs and nothing has.
//!
//! Supersaturation is a concentration, so solute lost by the scheme is
//! nucleation lost by the model, silently. Conservation is therefore the first
//! property to pin: in flux form the interior fluxes cancel pairwise and the
//! total is conserved to round-off, which is an exact statement rather than a
//! tolerance.
//!
//! The other three properties are that pure diffusion spreads a Gaussian at the
//! analytic rate, that a monotone profile stays monotone across a front, and
//! that a uniform flow translates a profile without inventing or destroying any
//! of it.

use volterra_core::{SpeciesField3D, VelocityField3D};
use volterra_fd::species_3d::{advect_diffuse_species, batchelor_scale, species_step_limit};

fn uniform_flow(n: usize, dx: f64, u: [f64; 3]) -> VelocityField3D {
    let mut v = VelocityField3D::zeros(n, n, n, dx);
    for k in 0..v.u.len() {
        v.u[k] = u;
    }
    v
}

/// A blob at the centre, which is what the schemes are asked to move and spread.
fn blob(n: usize, dx: f64, width: f64, species: usize) -> SpeciesField3D {
    let mut f = SpeciesField3D::zeros(species, n, n, n, dx);
    let c = (n as f64 - 1.0) / 2.0;
    for s in 0..species {
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let r2 =
                        ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (l as f64 - c).powi(2))
                            * dx
                            * dx;
                    f.c[s][((i * n) + j) * n + l] =
                        (1.0 + s as f64) * (-r2 / (2.0 * width * width)).exp();
                }
            }
        }
    }
    f
}

#[test]
fn advection_conserves_every_species_to_round_off() {
    // The property the whole design rests on. A divergence-free flow moves
    // solute and creates none, and in flux form that is exact rather than
    // approximate, since each interior face appears twice with opposite sign.
    let n = 24;
    let dx = 0.5;
    let mut f = blob(n, dx, 2.0, 3);
    f.diffusivity = vec![0.0; 3];
    let vel = uniform_flow(n, dx, [0.7, -0.4, 0.25]);

    let before: Vec<f64> = f.c.iter().map(|c| c.iter().sum()).collect();
    for _ in 0..40 {
        f = advect_diffuse_species(&f, &vel, 0.05);
    }
    let after: Vec<f64> = f.c.iter().map(|c| c.iter().sum()).collect();

    for (s, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10 * a.abs(),
            "species {s} went from {a} to {b}, a relative change of {}",
            (a - b).abs() / a.abs()
        );
    }
}

#[test]
fn diffusion_conserves_and_spreads_at_the_analytic_rate() {
    // A Gaussian of width w under pure diffusion has width sqrt(w^2 + 2 D t) in
    // each direction, which is a closed-form answer to check the coefficient
    // against rather than a shape to eyeball.
    let n = 48;
    let dx = 0.5;
    let d = 0.05;
    let w0 = 2.0;
    let mut f = blob(n, dx, w0, 1);
    f.diffusivity = vec![d];
    let vel = VelocityField3D::zeros(n, n, n, dx);

    let mass_before: f64 = f.c[0].iter().sum();
    let dt = 0.2 * dx * dx / (6.0 * d);
    let steps = 60;
    for _ in 0..steps {
        f = advect_diffuse_species(&f, &vel, dt);
    }
    let mass_after: f64 = f.c[0].iter().sum();
    assert!(
        (mass_before - mass_after).abs() < 1e-10 * mass_before,
        "diffusion changed the mass from {mass_before} to {mass_after}"
    );

    // Second moment about the centre, which is the width the analytic law names.
    let c = (n as f64 - 1.0) / 2.0;
    let mut m0 = 0.0;
    let mut m2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let v = f.c[0][((i * n) + j) * n + l];
                let x = (i as f64 - c) * dx;
                m0 += v;
                m2 += v * x * x;
            }
        }
    }
    let measured = (m2 / m0).sqrt();
    let expected = (w0 * w0 + 2.0 * d * dt * steps as f64).sqrt();
    assert!(
        (measured - expected).abs() < 0.05 * expected,
        "width {measured} against the analytic {expected}"
    );
}

#[test]
fn a_front_stays_between_its_own_bounds() {
    // What the limiter is for. An unlimited second-order scheme
    // overshoots at a step, and a negative concentration is not a small error in
    // a nucleation model: it is a rate that does not exist.
    let n = 32;
    let dx = 1.0;
    let mut f = SpeciesField3D::zeros(1, n, n, n, dx);
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                f.c[0][((i * n) + j) * n + l] = if i < n / 2 { 1.0 } else { 0.0 };
            }
        }
    }
    f.diffusivity = vec![0.0];
    let vel = uniform_flow(n, dx, [0.6, 0.0, 0.0]);

    for _ in 0..30 {
        f = advect_diffuse_species(&f, &vel, 0.4);
        let lo = f.c[0].iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = f.c[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(lo > -1e-12, "undershoot to {lo}");
        assert!(hi < 1.0 + 1e-12, "overshoot to {hi}");
    }
}

#[test]
fn a_uniform_flow_moves_the_blob_where_it_should() {
    let n = 40;
    let dx = 1.0;
    let mut f = blob(n, dx, 3.0, 1);
    f.diffusivity = vec![0.0];
    let u = [0.5, 0.0, 0.0];
    let vel = uniform_flow(n, dx, u);

    let centroid = |f: &SpeciesField3D| {
        let mut m0 = 0.0;
        let mut m1 = 0.0;
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let v = f.c[0][((i * n) + j) * n + l];
                    m0 += v;
                    m1 += v * i as f64 * dx;
                }
            }
        }
        m1 / m0
    };

    let start = centroid(&f);
    let dt = 0.5;
    let steps = 20;
    for _ in 0..steps {
        f = advect_diffuse_species(&f, &vel, dt);
    }
    let moved = centroid(&f) - start;
    let expected = u[0] * dt * steps as f64;
    assert!(
        (moved - expected).abs() < 0.05 * expected,
        "the blob moved {moved} against an expected {expected}"
    );
}

#[test]
fn the_step_limit_and_the_batchelor_scale_are_reported() {
    // Neither is enforced, and both are what a caller needs to know. A liquid's
    // Schmidt number is around a thousand, so the scalar has structure some
    // thirty times finer than the flow, and no grid affordable in three
    // dimensions resolves it. The number is reported so a run states whether it
    // could.
    let dx = 0.5;
    let limit = species_step_limit(&[0.05, 0.2], 1.0, dx);
    assert!(limit > 0.0 && limit.is_finite());
    // The tighter of the two species binds.
    assert!(limit <= dx * dx / (6.0 * 0.2) + 1e-12);

    let eta = 1.0;
    assert!((batchelor_scale(eta, 1e3) - eta / (1e3_f64).sqrt()).abs() < 1e-12);
    assert!(batchelor_scale(eta, 1.0) == eta);
}
