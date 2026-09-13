//! The domain abstraction against the lattice code it generalises.
//!
//! [`CartesianDomain`] exists to be the first implementation of [`Domain3`], so
//! what these assert is that it changes nothing: the generic path has to
//! reproduce the lattice path exactly, not merely closely. A tolerance here
//! would hide the sign or stride error that such a rewrite invites, so these
//! compare bit for bit.

use volterra_braid::disclination::{decompose, disclination_density, level_set_curvature};
use volterra_braid::domain::{CartesianDomain, Domain3, disclination_density_on};

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

/// A twisted field with structure in all three directions, so a stride error
/// cannot pass by symmetry.
fn twisted(n: usize) -> Vec<[f64; 5]> {
    let c = n as f64 / 2.0 - 0.5;
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, l as f64 - c);
                let theta = 0.5 * y.atan2(x) + 0.17 * z;
                let tilt = 0.3 * (0.11 * x).sin();
                let dir = [
                    theta.cos() * tilt.cos(),
                    theta.sin() * tilt.cos(),
                    tilt.sin(),
                ];
                q[((i * n) + j) * n + l] = uniaxial(dir, 0.556);
            }
        }
    }
    q
}

#[test]
fn the_generic_density_reproduces_the_lattice_one_bit_for_bit() {
    let n = 20;
    let dx = 0.37;
    let q = twisted(n);
    let domain = CartesianDomain::new(n, n, n, dx);

    let lattice = disclination_density(&q, n, n, n, dx);
    let generic = disclination_density_on(&domain, &q);

    assert_eq!(lattice.len(), generic.len());
    for (site, (a, b)) in lattice.iter().zip(generic.iter()).enumerate() {
        for c in 0..9 {
            assert_eq!(
                a[c].to_bits(),
                b[c].to_bits(),
                "site {site} component {c}: {} against {}",
                a[c],
                b[c]
            );
        }
    }
}

#[test]
fn the_generic_density_gives_the_same_winding_character() {
    // The density agreeing entry by entry already implies this, but the
    // factorisation is what every consumer actually reads, so it is asserted
    // where a reader will look for it.
    let n = 16;
    let q = twisted(n);
    let domain = CartesianDomain::new(n, n, n, 1.0);
    for (a, b) in disclination_density(&q, n, n, n, 1.0)
        .iter()
        .zip(disclination_density_on(&domain, &q).iter())
    {
        let (x, y) = (decompose(a), decompose(b));
        assert_eq!(x.s.to_bits(), y.s.to_bits());
        assert_eq!(x.cos_beta.to_bits(), y.cos_beta.to_bits());
    }
}

#[test]
fn the_domains_scalar_derivatives_match_the_level_set_stencil() {
    // `level_set_curvature` clamps at the faces rather than wrapping, and the
    // domain has to agree, since a level set reaching a wall is a real feature
    // of a confined run.
    let n = 18;
    let dx = 0.25;
    let c = (n as f64 - 1.0) / 2.0;
    let mut field = vec![0.0; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (
                    (i as f64 - c) * dx,
                    (j as f64 - c) * dx,
                    (l as f64 - c) * dx,
                );
                field[((i * n) + j) * n + l] = x * x + 2.0 * y * y + 0.5 * z * z + 0.3 * x * y;
            }
        }
    }
    let domain = CartesianDomain::new(n, n, n, dx);

    // A site well inside, and one on a face where the clamping matters.
    for (i, j, l) in [(9, 9, 9), (0, 7, 11), (n - 1, n - 1, 3)] {
        let site = (i * n + j) * n + l;
        let (g, h) = domain.scalar_derivatives(&field, site);
        let k = level_set_curvature(&field, n, n, n, dx, (i, j, l));

        // Rebuild the curvature from the domain's derivatives and compare.
        let norm = g.norm();
        let (a, b, cc) = (h[(0, 0)], h[(0, 1)], h[(0, 2)]);
        let (d, e, f) = (h[(1, 1)], h[(1, 2)], h[(2, 2)]);
        let adj = nalgebra::Matrix3::new(
            d * f - e * e,
            cc * e - b * f,
            b * e - cc * d,
            cc * e - b * f,
            a * f - cc * cc,
            b * cc - a * e,
            b * e - cc * d,
            b * cc - a * e,
            a * d - b * b,
        );
        let gaussian = g.dot(&(adj * g)) / norm.powi(4);
        let mean = (g.dot(&(h * g)) - norm * norm * h.trace()) / (2.0 * norm.powi(3));

        assert!(
            (mean - k.mean).abs() < 1e-12,
            "mean at {i},{j},{l}: {mean} against {}",
            k.mean
        );
        assert!(
            (gaussian - k.gaussian).abs() < 1e-12,
            "gaussian at {i},{j},{l}: {gaussian} against {}",
            k.gaussian
        );
    }
}

#[test]
fn the_lattice_offers_twenty_six_neighbours_inside_and_fewer_at_a_corner() {
    let n = 6;
    let domain = CartesianDomain::new(n, n, n, 1.0);
    let mut out = Vec::new();

    domain.neighbours((2 * n + 2) * n + 2, &mut out);
    assert_eq!(out.len(), 26, "an interior site has 26 neighbours");

    domain.neighbours(0, &mut out);
    assert_eq!(out.len(), 7, "a corner has 7");

    // Adjacency is symmetric, which the curve assembly relies on.
    let site = (1 * n + 2) * n + 3;
    domain.neighbours(site, &mut out);
    let mut back = Vec::new();
    for &m in &out {
        domain.neighbours(m, &mut back);
        assert!(
            back.contains(&site),
            "{m} adjoins {site} but not the reverse"
        );
    }
}

#[test]
fn positions_and_sampling_agree_at_the_sites_themselves() {
    let n = 8;
    let dx = 0.6;
    let domain = CartesianDomain::new(n, n, n, dx);
    let field: Vec<f64> = (0..n * n * n).map(|k| (k as f64 * 0.37).sin()).collect();

    for site in [0usize, 5, 100, n * n * n - 1] {
        let p = domain.position(site);
        let sampled = domain.sample(&field, p);
        assert!(
            (sampled - field[site]).abs() < 1e-12,
            "sampling at site {site} gave {sampled} against {}",
            field[site]
        );
    }
}
