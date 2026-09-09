//! Curvature of a disclination loop, and of the tube around it.
//!
//! A disclination loop is a curve and it is also the axis of an `s` isosurface,
//! so it has two curvatures and both are reported. The tests below fix each
//! against a shape whose answer is known in closed form: a circle and a helix
//! for the line, a sphere and a cylinder for the surface.
//!
//! The line curvature is a second derivative, so it is only as good as the
//! positions it differentiates. Supra-threshold voxels form a tube several
//! voxels across and a walk through them zigzags at the lattice scale, which is
//! why the sites are thinned to the ridge and refined to sub-voxel accuracy
//! before any of this is applied.

use std::f64::consts::PI;

use volterra_braid::disclination::{
    disclination_lines, disclination_lines_at_fraction, disclination_magnitude, disclination_sites,
    frenet, level_set_curvature,
};

// ─────────────────────────────────────────────────────────────────────────────
// Line curvature, against curves whose Frenet apparatus is known
// ─────────────────────────────────────────────────────────────────────────────

/// `n` points of a circle of radius `r` in the plane `z = 0`.
fn circle(r: f64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|k| {
            let t = 2.0 * PI * k as f64 / n as f64;
            [r * t.cos(), r * t.sin(), 0.0]
        })
        .collect()
}

#[test]
fn a_circle_has_curvature_one_over_its_radius() {
    let r = 7.0;
    let f = frenet(&circle(r, 200), true);

    for (i, &k) in f.curvatures.iter().enumerate() {
        assert!(
            (k - 1.0 / r).abs() < 1e-3,
            "site {i} read curvature {k}, expected {}",
            1.0 / r
        );
    }
}

#[test]
fn a_plane_curve_has_no_torsion() {
    let f = frenet(&circle(7.0, 200), true);

    for (i, &t) in f.torsions.iter().enumerate() {
        assert!(t.abs() < 1e-6, "site {i} read torsion {t} on a plane curve");
    }
}

#[test]
fn a_straight_line_has_no_curvature() {
    let pts: Vec<[f64; 3]> = (0..40).map(|k| [0.3 * k as f64, 0.0, 0.0]).collect();
    let f = frenet(&pts, false);

    for (i, &k) in f.curvatures.iter().enumerate() {
        assert!(k.abs() < 1e-9, "site {i} read curvature {k} on a straight line");
    }
}

/// `n` points of the helix `r(t) = (a cos t, a sin t, b t)` over three turns.
fn helix(a: f64, b: f64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|k| {
            let t = 6.0 * PI * k as f64 / n as f64;
            [a * t.cos(), a * t.sin(), b * t]
        })
        .collect()
}

/// Worst curvature and torsion error over the interior, where the window is
/// centred rather than one-sided.
fn helix_error(a: f64, b: f64, n: usize) -> (f64, f64) {
    let f = frenet(&helix(a, b, n), false);
    let (kappa, tau) = (a / (a * a + b * b), b / (a * a + b * b));
    let mut worst = (0.0f64, 0.0f64);
    for i in 5..n - 5 {
        worst.0 = worst.0.max((f.curvatures[i] - kappa).abs());
        worst.1 = worst.1.max((f.torsions[i] - tau).abs());
    }
    worst
}

#[test]
fn a_helix_returns_its_analytic_curvature_and_torsion() {
    // r(t) = (a cos t, a sin t, b t) has kappa = a / (a^2 + b^2) and
    // tau = b / (a^2 + b^2), both constant along it.
    let (a, b) = (3.0, 2.0);
    let (kappa_err, tau_err) = helix_error(a, b, 1600);

    assert!(kappa_err < 5e-5, "worst curvature error {kappa_err}");
    assert!(tau_err < 1e-5, "worst torsion error {tau_err}");
}

#[test]
fn the_helixs_error_is_truncation_of_the_fit_and_refines_at_second_order() {
    // The residual at a given sampling is the cubic's truncation over its own
    // window, which is second order in the spacing for both quantities: a
    // cubic reproduces three derivatives exactly and the quartic term it drops
    // enters the second derivative at h^2. A method error would sit at the same
    // size on both grids instead, which is what this separates.
    let (a, b) = (3.0, 2.0);
    let coarse = helix_error(a, b, 400);
    let fine = helix_error(a, b, 800);

    for (name, c, f) in [
        ("curvature", coarse.0, fine.0),
        ("torsion", coarse.1, fine.1),
    ] {
        let ratio = c / f;
        assert!(
            (3.5..4.5).contains(&ratio),
            "{name} error went {c} to {f} on halving the spacing, a ratio of {ratio} against the 4 second order gives"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Surface curvature, against level sets whose curvature is known
// ─────────────────────────────────────────────────────────────────────────────

/// Sample `f` on an `n^3` grid of spacing `dx`, centred on the box.
fn sample<F: Fn(f64, f64, f64) -> f64>(n: usize, dx: f64, f: F) -> Vec<f64> {
    let c = (n as f64 - 1.0) / 2.0;
    let mut out = vec![0.0; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (
                    (i as f64 - c) * dx,
                    (j as f64 - c) * dx,
                    (l as f64 - c) * dx,
                );
                out[((i * n) + j) * n + l] = f(x, y, z);
            }
        }
    }
    out
}

#[test]
fn a_sphere_reads_mean_curvature_one_over_r_and_gaussian_one_over_r_squared() {
    // The level sets of x^2 + y^2 + z^2 are spheres. At radius r the mean
    // curvature is 1/r in magnitude and the Gaussian curvature is 1/r^2.
    let (n, dx) = (65, 0.25);
    let field = sample(n, dx, |x, y, z| x * x + y * y + z * z);
    let c = (n - 1) / 2;

    // A site four voxels off centre sits at r = 1.0.
    let r = 4.0 * dx;
    let k = level_set_curvature(&field, n, n, n, dx, (c + 4, c, c));

    assert!(
        (k.mean.abs() - 1.0 / r).abs() < 1e-3,
        "mean curvature {} at radius {r}, expected magnitude {}",
        k.mean,
        1.0 / r
    );
    assert!(
        (k.gaussian - 1.0 / (r * r)).abs() < 1e-3,
        "Gaussian curvature {}, expected {}",
        k.gaussian,
        1.0 / (r * r)
    );
}

#[test]
fn a_cylinder_reads_half_the_spheres_mean_curvature_and_no_gaussian_curvature() {
    // The level sets of x^2 + y^2 are cylinders: one principal curvature 1/r,
    // the other zero, so the mean is 1/(2r) and the Gaussian vanishes. This is
    // the shape of the tube around a straight disclination.
    let (n, dx) = (65, 0.25);
    let field = sample(n, dx, |x, y, _| x * x + y * y);
    let c = (n - 1) / 2;

    let r = 4.0 * dx;
    let k = level_set_curvature(&field, n, n, n, dx, (c + 4, c, c));

    assert!(
        (k.mean.abs() - 0.5 / r).abs() < 1e-3,
        "mean curvature {} at radius {r}, expected magnitude {}",
        k.mean,
        0.5 / r
    );
    assert!(
        k.gaussian.abs() < 1e-3,
        "Gaussian curvature {} on a cylinder, expected zero",
        k.gaussian
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test fields
// ─────────────────────────────────────────────────────────────────────────────

/// `Q = q (nn - I/3)`, the convention the 3D papers use.
fn uniaxial(n: [f64; 3], q_mag: f64) -> [f64; 5] {
    let t = 1.0 / 3.0;
    [
        q_mag * (n[0] * n[0] - t),
        q_mag * (n[0] * n[1]),
        q_mag * (n[0] * n[2]),
        q_mag * (n[1] * n[1] - t),
        q_mag * (n[1] * n[2]),
    ]
}

/// A `+1/2` wedge line running the full depth in z, cored at `(cx, cy)`.
fn wedge_at(n: usize, cx: f64, cy: f64) -> Vec<[f64; 5]> {
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let theta = 0.5 * (j as f64 - cy).atan2(i as f64 - cx);
                q[((i * n) + j) * n + l] = uniaxial([theta.cos(), theta.sin(), 0.0], 0.556);
            }
        }
    }
    q
}

/// A `+1/2` disclination loop of radius `r` in the mid-plane, on the box axis.
///
/// Around the core the director winds by `pi` in the meridional plane, which is
/// the loop's defining texture: at meridional angle `psi` measured from the
/// outward radial direction, the director is
/// `cos(psi/2) e_rho + sin(psi/2) e_z`.
fn loop_field(n: usize, r: f64) -> Vec<[f64; 5]> {
    let c = (n as f64 - 1.0) / 2.0;
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, l as f64 - c);
                let rho = (x * x + y * y).sqrt();
                let half = 0.5 * z.atan2(rho - r);
                let e_rho = if rho > 1e-9 { [x / rho, y / rho] } else { [1.0, 0.0] };
                let dir = [half.cos() * e_rho[0], half.cos() * e_rho[1], half.sin()];
                q[((i * n) + j) * n + l] = uniaxial(dir, 0.556);
            }
        }
    }
    q
}

/// The peak of `s` away from the faces, and a fraction of it.
///
/// The derivative stencil wraps, so the box faces read a seam that has nothing
/// to do with a disclination. Every threshold below is taken from the interior
/// peak for that reason, which is also how a real run sets one.
fn interior_threshold(q: &[[f64; 5]], n: usize, frac: f64) -> f64 {
    let s = disclination_magnitude(q, n, n, n, 1.0);
    let mut peak = 0.0_f64;
    for i in 2..n - 2 {
        for j in 2..n - 2 {
            for l in 2..n - 2 {
                peak = peak.max(s[((i * n) + j) * n + l]);
            }
        }
    }
    frac * peak
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-voxel cores
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_core_placed_between_voxels_is_found_between_voxels() {
    // The core sits at x = 9.37, which no voxel index can represent. Every site
    // on the ridge should land within a quarter of a voxel of it; the raw index
    // can only ever answer 9 or 10.
    let n = 32;
    let (cx, cy) = (9.37_f64, 15.5_f64);
    let q = wedge_at(n, cx, cy);
    let cut = interior_threshold(&q, n, 0.25);
    let sites = disclination_sites(&q, n, n, n, 1.0, cut);

    let interior: Vec<_> = sites
        .iter()
        .filter(|s| (2..n - 2).contains(&s.ijl.0) && (2..n - 2).contains(&s.ijl.1))
        .collect();
    assert!(!interior.is_empty(), "no interior sites found");

    for s in &interior {
        assert!(
            (s.pos[0] - cx).abs() < 0.25,
            "site at voxel {:?} refined to x = {}, core at {cx}",
            s.ijl,
            s.pos[0]
        );
        assert!(
            (s.pos[1] - cy).abs() < 0.25,
            "site at voxel {:?} refined to y = {}, core at {cy}",
            s.ijl,
            s.pos[1]
        );
    }
}

#[test]
fn the_ridge_keeps_one_site_per_slice_of_a_straight_line() {
    // A supra-threshold tube is several voxels across. Thinning to the ridge is
    // what makes the site list a curve, and a straight line along z has exactly
    // one ridge point in each of its slices.
    let n = 32;
    let q = wedge_at(n, 15.5, 15.5);
    let cut = interior_threshold(&q, n, 0.25);
    let sites = disclination_sites(&q, n, n, n, 1.0, cut);

    for l in 2..n - 2 {
        let in_slice = sites
            .iter()
            .filter(|s| {
                s.ijl.2 == l && (2..n - 2).contains(&s.ijl.0) && (2..n - 2).contains(&s.ijl.1)
            })
            .count();
        assert_eq!(in_slice, 1, "slice {l} kept {in_slice} sites, expected 1");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The two curvatures of an actual loop
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_disclination_loop_closes_and_reads_the_curvature_of_its_radius() {
    let (n, r) = (48, 12.0);
    let q = loop_field(n, r);
    let cut = interior_threshold(&q, n, 0.25);

    let lines = disclination_lines(&q, n, n, n, 1.0, cut);
    let ring = lines
        .iter()
        .find(|c| c.is_loop)
        .expect("no closed curve was assembled");

    assert!(
        (ring.length - 2.0 * PI * r).abs() < 0.1 * 2.0 * PI * r,
        "contour length {} against a circumference of {}",
        ring.length,
        2.0 * PI * r
    );
    assert!(
        (ring.mean_curvature - 1.0 / r).abs() < 0.2 / r,
        "line curvature {} against 1/r = {}",
        ring.mean_curvature,
        1.0 / r
    );
}

#[test]
fn the_tube_around_a_loop_is_curved_across_it_and_along_it() {
    // The `s` isosurface around a loop is a torus. Across the tube it curves on
    // the tube radius, which is far tighter than the loop radius, so the mean
    // curvature of the surface is much the larger of the two numbers.
    let (n, r) = (48, 12.0);
    let q = loop_field(n, r);
    let cut = interior_threshold(&q, n, 0.25);

    let lines = disclination_lines(&q, n, n, n, 1.0, cut);
    let ring = lines
        .iter()
        .find(|c| c.is_loop)
        .expect("no closed curve was assembled");

    assert!(
        ring.surface_mean_curvature.abs() > ring.mean_curvature,
        "surface mean curvature {} should exceed the line's {}",
        ring.surface_mean_curvature,
        ring.mean_curvature
    );
}

#[test]
fn the_tube_around_a_straight_line_reads_one_over_twice_its_radius() {
    // Around a straight disclination the isosurface is a cylinder, so its mean
    // curvature is 1/(2R) at the tube radius R. The sign is positive, since `s`
    // increases towards the core and the surface normal follows the gradient
    // inward: the tube curves away from the direction the normal points.
    let n = 40;
    let (cx, cy) = (19.5_f64, 19.5_f64);
    let q = wedge_at(n, cx, cy);
    let cut = interior_threshold(&q, n, 0.25);

    // The tube radius, measured rather than assumed: the mean distance from the
    // core of the voxels making up the threshold's inner shell.
    let s = disclination_magnitude(&q, n, n, n, 1.0);
    let at = |i: usize, j: usize, l: usize| s[((i * n) + j) * n + l];
    let (mut radius_sum, mut count) = (0.0_f64, 0usize);
    for i in 2..n - 2 {
        for j in 2..n - 2 {
            let l = n / 2;
            if at(i, j, l) <= cut {
                continue;
            }
            let shell = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                .into_iter()
                .any(|(a, b)| at(a, b, l) <= cut);
            if shell {
                radius_sum += ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)).sqrt();
                count += 1;
            }
        }
    }
    assert!(count > 0, "the threshold's shell was empty");
    let tube_radius = radius_sum / count as f64;

    let lines = disclination_lines(&q, n, n, n, 1.0, cut);
    let line = lines.first().expect("the straight line was not found");

    let expected = 1.0 / (2.0 * tube_radius);
    assert!(
        line.surface_mean_curvature > 0.0,
        "surface mean curvature {} should be positive around a core",
        line.surface_mean_curvature
    );
    assert!(
        (line.surface_mean_curvature - expected).abs() < 0.3 * expected,
        "surface mean curvature {} against 1/(2R) = {expected} at R = {tube_radius}",
        line.surface_mean_curvature
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// A field with no disclination in it
// ─────────────────────────────────────────────────────────────────────────────

/// A uniformly ordered field with a little numerical noise on it, which is what
/// a run looks like once its defects have annihilated.
fn ordered_with_noise(n: usize, noise: f64) -> Vec<[f64; 5]> {
    let mut q = vec![[0.0; 5]; n * n * n];
    let mut state = 0x2545F491_4F6CDD1Du64;
    for k in 0..q.len() {
        // xorshift, so the noise is reproducible without a dependency
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        let dir = [1.0, noise * r, noise * r];
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        q[k] = uniaxial([dir[0] / norm, dir[1] / norm, dir[2] / norm], 0.556);
    }
    q
}

#[test]
fn a_relative_threshold_alone_finds_lines_in_an_ordered_field() {
    // The failure a floor exists to stop. `s` has a largest value wherever the
    // field is, defects or none, so a threshold taken purely as a fraction of it
    // reports lines in the noise of a field that has none. A run reads this as a
    // threshold collapsing by orders of magnitude while the order parameter sits
    // at its equilibrium.
    let n = 24;
    let q = ordered_with_noise(n, 1e-6);
    let (lines, threshold) = disclination_lines_at_fraction(&q, n, n, n, 1.0, 0.25, 0.0);

    assert!(threshold < 1e-10, "an ordered field read a threshold of {threshold}");
    assert!(
        !lines.is_empty(),
        "the relative rule found nothing, so this test no longer shows the failure"
    );
}

#[test]
fn a_floor_keeps_an_ordered_field_empty() {
    let n = 24;
    let q = ordered_with_noise(n, 1e-6);
    let (lines, threshold) = disclination_lines_at_fraction(&q, n, n, n, 1.0, 0.25, 1e-6);

    assert_eq!(threshold, 1e-6, "the floor should have bound");
    assert!(
        lines.is_empty(),
        "an ordered field returned {} lines above the floor",
        lines.len()
    );
}

#[test]
fn a_floor_below_the_peak_leaves_a_real_line_alone() {
    // The floor must not cost a detection that the relative rule would make.
    let n = 32;
    let q = wedge_at(n, 15.5, 15.5);
    let relative = interior_threshold(&q, n, 0.25);

    let (lines, threshold) = disclination_lines_at_fraction(&q, n, n, n, 1.0, 0.25, 1e-6);
    assert_eq!(
        threshold, relative,
        "the relative rule should still bind on a field with a real core"
    );
    assert!(!lines.is_empty(), "the wedge line was lost to the floor");
}

#[test]
fn the_density_at_a_resolved_core_scales_as_the_order_over_the_spacing_squared() {
    // What makes a threshold transferable. `s` is quadratic in a Q gradient, and
    // a `+1/2` core turns the director through `pi/2` over a lattice spacing, so
    // the density at a resolved core is a fixed multiple of `(q/dx)^2` whatever
    // the spacing is. A floor written in those units therefore means the same
    // thing on every grid, where a bare number does not.
    let (n, q_mag) = (40, 0.556_f64);
    let mut ratios = Vec::new();
    for dx in [0.5_f64, 1.0, 2.0] {
        let q = wedge_at(n, 19.5, 19.5);
        let s = disclination_magnitude(&q, n, n, n, dx);
        let mut peak = 0.0_f64;
        for i in 2..n - 2 {
            for j in 2..n - 2 {
                for l in 2..n - 2 {
                    peak = peak.max(s[((i * n) + j) * n + l]);
                }
            }
        }
        ratios.push(peak / (q_mag / dx).powi(2));
    }
    for r in &ratios {
        assert!(
            (r - ratios[0]).abs() < 1e-9,
            "the ratio moved with the spacing: {ratios:?}"
        );
    }
    assert!(
        (ratios[0] - 0.647).abs() < 0.01,
        "a resolved core read {} of (q/dx)^2, measured at 0.647",
        ratios[0]
    );
}
