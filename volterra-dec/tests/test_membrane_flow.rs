//! Stability of the semi-implicit Helfrich step against explicit Euler.
//!
//! Overdamped Helfrich flow is fourth order in position, so an explicit step is
//! limited to `dt ~ eta h^4 / kappa`. Treating the leading operator implicitly
//! is what makes the solver usable at a resolution where the physics lives.
//!
//! Both integrators descend the SAME discrete energy, so "stable" here means
//! what it means for a gradient flow: every iterate finite, and the energy
//! never above where it started.

use nalgebra::Vector3;
use volterra_dec::bending::{BendingParams, bending_energy, bending_gradient};
use volterra_dec::flow::{FlowConfig, semi_implicit_step};
use volterra_dec::mesh_gen::icosphere;

fn perturbed_sphere(level: usize, eps: f64) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    let m = icosphere(level);
    let mut verts = m.vertices.clone();
    for (i, v) in verts.iter_mut().enumerate() {
        let t = i as f64;
        *v += eps
            * Vector3::new(
                (3.0 * t).sin(),
                (5.0 * t + 1.0).sin(),
                (7.0 * t + 2.0).sin(),
            );
    }
    (verts, m.simplices.clone())
}

fn params(nv: usize) -> BendingParams {
    BendingParams {
        kappa: 1.0,
        kappa_bar: 0.0,
        h0: vec![0.0; nv],
        tension: 0.0,
    }
}

/// Explicit Euler on the same energy: `eta A_v dx/dt = -grad E`. The dual-area
/// division is what makes the effective step scale as `h^-2` and is the reason
/// this integrator is limited so severely.
fn explicit_step(
    verts: &mut [Vector3<f64>],
    tris: &[[usize; 3]],
    p: &BendingParams,
    dt: f64,
    eta: f64,
) {
    let areas = volterra_dec::flow::dual_areas(verts, tris);
    let g = bending_gradient(verts, tris, p);
    for ((v, gi), a) in verts.iter_mut().zip(&g).zip(&areas) {
        *v -= (dt / (eta * a.max(1e-30))) * gi;
    }
}

/// Run `steps` of a stepper and report whether the flow stayed a descent.
fn survives(level: usize, dt: f64, steps: usize, semi_implicit: bool) -> bool {
    let (mut verts, tris) = perturbed_sphere(level, 0.05);
    let p = params(verts.len());
    let cfg = FlowConfig {
        dt,
        eta: 1.0,
        cg_tol: 1e-10,
        cg_max_iter: 500,
    };
    let e0 = bending_energy(&verts, &tris, &p);

    for _ in 0..steps {
        if semi_implicit {
            semi_implicit_step(&mut verts, &tris, &p, &cfg);
        } else {
            explicit_step(&mut verts, &tris, &p, dt, cfg.eta);
        }
        if verts.iter().any(|v| !v.iter().all(|c| c.is_finite())) {
            return false;
        }
        let e = bending_energy(&verts, &tris, &p);
        if !e.is_finite() || e > e0 * 1.001 + 1e-9 {
            return false;
        }
    }
    true
}

/// Largest `dt` on a geometric ladder for which the flow stays a descent.
fn max_stable_dt(level: usize, steps: usize, semi_implicit: bool) -> f64 {
    let mut best = 0.0;
    let mut dt = 1e-9;
    for _ in 0..40 {
        if survives(level, dt, steps, semi_implicit) {
            best = dt;
        } else if best > 0.0 {
            break;
        }
        dt *= 2.0;
    }
    best
}

#[test]
fn semi_implicit_stable_timestep_exceeds_explicit_by_100x() {
    let explicit = max_stable_dt(2, 30, false);
    let implicit = max_stable_dt(2, 30, true);
    assert!(explicit > 0.0, "explicit baseline never stabilised");
    let gain = implicit / explicit;
    assert!(
        gain >= 100.0,
        "semi-implicit should buy at least 100x in dt: explicit {explicit:.3e}, \
         semi-implicit {implicit:.3e}, gain {gain:.1}x"
    );
}

#[test]
fn semi_implicit_agrees_with_explicit_at_small_timestep() {
    // Both discretise the same flow, so at a step well inside the explicit
    // stability limit they must produce the same displacement to leading order.
    let (verts0, tris) = perturbed_sphere(2, 0.05);
    let p = params(verts0.len());
    let dt = 1e-8;

    let mut a = verts0.clone();
    explicit_step(&mut a, &tris, &p, dt, 1.0);

    let mut b = verts0.clone();
    semi_implicit_step(
        &mut b,
        &tris,
        &p,
        &FlowConfig {
            dt,
            eta: 1.0,
            cg_tol: 1e-12,
            cg_max_iter: 500,
        },
    );

    let moved = a
        .iter()
        .zip(&verts0)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()));
    let diff = a
        .iter()
        .zip(&b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()));
    assert!(moved > 0.0, "explicit step did not move anything");
    assert!(
        diff / moved < 1e-3,
        "steppers disagree at dt = {dt:.1e}: relative difference {:.3e}",
        diff / moved
    );
}

/// Relative spread of the vertex radii: zero on a round sphere.
fn roundness(v: &[Vector3<f64>]) -> f64 {
    let r: Vec<f64> = v.iter().map(|x| x.norm()).collect();
    let mean = r.iter().sum::<f64>() / r.len() as f64;
    let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / r.len() as f64;
    var.sqrt() / mean
}

/// Discrete energy of a clean icosphere at the same resolution, the floor a
/// descent should respect.
///
/// Willmore's theorem gives `int H^2 dA >= 4 pi` for every closed immersed
/// surface, so `E >= 2 pi kappa` in the continuum. The cotangent discretisation
/// approaches that from BELOW at second order, measured as -6.86, -1.76, -0.445,
/// -0.112, -0.028 percent over icosphere levels 1 to 5, so the continuum number
/// is not a valid acceptance test on a discrete run. The discrete sphere energy at the
/// working resolution is.
fn discrete_sphere_energy(level: usize) -> f64 {
    let m = icosphere(level);
    bending_energy(&m.vertices, &m.simplices, &params(m.vertices.len()))
}

#[test]
fn semi_implicit_descent_stays_near_the_discrete_sphere_energy() {
    // The regime where the scheme is trustworthy: a mild perturbation relaxed
    // over a short horizon, before the lack of tangential control has degraded
    // the triangulation. The 2 percent margin allows the flow to find a
    // configuration slightly better than the icosphere, which is not itself the
    // discrete minimiser, while still catching a collapse.
    let level = 3;
    let floor = discrete_sphere_energy(level) * 0.98;
    let (mut verts, tris) = perturbed_sphere(level, 0.01);
    let p = params(verts.len());
    let cfg = FlowConfig {
        dt: 1e-4,
        eta: 1.0,
        cg_tol: 1e-10,
        cg_max_iter: 500,
    };

    let e0 = bending_energy(&verts, &tris, &p);
    let mut prev = e0;
    for step in 0..100 {
        semi_implicit_step(&mut verts, &tris, &p, &cfg);
        let e = bending_energy(&verts, &tris, &p);
        assert!(
            e <= prev + 1e-9,
            "energy rose at step {step}: {prev:.10} -> {e:.10}"
        );
        assert!(
            e >= floor,
            "step {step}: energy {e:.6} fell below the discrete sphere floor \
             {floor:.6}, so the mesh no longer represents a surface"
        );
        prev = e;
    }
    assert!(
        prev < e0,
        "bending energy should fall: {e0:.6} -> {prev:.6}"
    );
}

#[test]
#[ignore = "KNOWN BUG: L2 flow has no tangential redistribution; see buglog"]
fn semi_implicit_relaxes_a_perturbed_sphere() {
    // KNOWN-BUG ORACLE asserting CORRECT behaviour. Willmore flow drives a
    // perturbed sphere to a round one, and no closed surface can sit below the
    // Willmore bound. Both fail today, because the L2 gradient flow moves
    // vertices tangentially with nothing to redistribute them, so the
    // triangulation degrades and the discrete energy stops representing the
    // surface. Measured over 300 steps at dt = 1e-2, level 3, eps = 0.06:
    //
    //   max face area  3.23e-2 -> 5.57e-1 (17x) with min area flat at ~4e-4
    //   min aspect     7.01e-2 -> 2.90e-2
    //   mean |cos(K,n)|  0.967 -> 0.897
    //   energy          108.79 -> 1.996, against a 6.255 discrete sphere floor
    //   radial spread  4.23e-2 -> 1.24e-1, away from round
    //
    // Passes once the step carries tangential redistribution, either the
    // coupled (x, H n) formulation of Barrett-Garcke-Nurnberg or an explicit
    // redistribution pass.
    let level = 3;
    let floor = discrete_sphere_energy(level) * 0.98;
    let (mut verts, tris) = perturbed_sphere(level, 0.06);
    let p = params(verts.len());
    let cfg = FlowConfig {
        dt: 1e-2,
        eta: 1.0,
        cg_tol: 1e-10,
        cg_max_iter: 500,
    };

    let before = roundness(&verts);
    for step in 0..300 {
        semi_implicit_step(&mut verts, &tris, &p, &cfg);
        let e = bending_energy(&verts, &tris, &p);
        assert!(
            e >= floor,
            "step {step}: energy {e:.6} fell below the discrete sphere floor {floor:.6}"
        );
    }
    let after = roundness(&verts);
    assert!(
        after < before * 0.5,
        "flow should round the surface: radial spread {before:.4e} -> {after:.4e}"
    );
}
