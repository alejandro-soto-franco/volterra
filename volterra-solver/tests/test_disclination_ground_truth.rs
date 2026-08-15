//! The 3D disclination detectors against fields whose answer is known exactly.
//!
//! volterra reaches a disclination line two ways: `defects_3d::scan_defects_3d`,
//! a holonomy over each voxel face, and `volterra_braid::disclination`, the
//! disclination density tensor. They share no code and no intermediate
//! quantity, so a field with a known answer separates them.
//!
//! The field used here is a `+1/2` and a `-1/2` straight line, parallel, both
//! running the full depth of the box. Superposing the two windings makes the
//! director single-valued far from either core, so the box faces carry no jump
//! and neither detector is being asked about an artefact of the test field.
//!
//! Measured on that field at `32^3` (see `docs/REPLICATION.md`): the density
//! tensor puts both cores within a hundredth of a lattice unit of where they
//! were placed and follows each line the full depth of the box, while the
//! holonomy path returns one line of four vertices covering two of the
//! thirty-two slices. The assertions below are on the tensor, which is what the
//! confined-cylinder work uses.

use volterra_braid::disclination::{decompose, disclination_density, disclination_lines};
use volterra_fields::QField3D;

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

/// Two parallel straight lines of opposite winding at `x = x1` and `x = x2`,
/// both at the mid-plane in y, both running the full depth in z.
fn two_lines(n: usize, x1: f64, x2: f64) -> QField3D {
    let cy = (n as f64 - 1.0) / 2.0;
    let mut q = QField3D::zeros(n, n, n, 1.0);
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y) = (i as f64, j as f64 - cy);
                let theta = 0.5 * y.atan2(x - x1) - 0.5 * y.atan2(x - x2);
                let k = q.idx(i, j, l);
                q.q[k] = uniaxial([theta.cos(), theta.sin(), 0.0], 0.556);
            }
        }
    }
    q
}

/// Interior peak of the density, and the threshold taken from it.
fn threshold(q: &QField3D, frac: f64) -> f64 {
    let n = q.nx;
    let d = disclination_density(&q.q, n, n, n, 1.0);
    let mut peak = 0.0_f64;
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for l in 1..n - 1 {
                peak = peak.max(decompose(&d[((i * n) + j) * n + l]).s);
            }
        }
    }
    frac * peak
}

#[test]
fn the_tensor_finds_both_cores_where_they_were_placed() {
    let n = 32;
    let (x1, x2) = (9.5_f64, 21.5_f64);
    let q = two_lines(n, x1, x2);
    let d = disclination_density(&q.q, n, n, n, 1.0);
    let cut = threshold(&q, 0.5);

    let mid = n / 2;
    let (mut left, mut right) = ((0.0, 0.0, 0usize), (0.0, 0.0, 0usize));
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            if decompose(&d[((i * n) + j) * n + mid]).s > cut {
                let t = if (i as f64) < 0.5 * (x1 + x2) {
                    &mut left
                } else {
                    &mut right
                };
                t.0 += i as f64;
                t.1 += j as f64;
                t.2 += 1;
            }
        }
    }
    assert!(left.2 > 0 && right.2 > 0, "one of the two cores was missed");

    let lx = left.0 / left.2 as f64;
    let rx = right.0 / right.2 as f64;
    assert!((lx - x1).abs() < 0.1, "left core at {lx}, placed at {x1}");
    assert!((rx - x2).abs() < 0.1, "right core at {rx}, placed at {x2}");
}

#[test]
fn the_tensor_follows_both_lines_the_full_depth_of_the_box() {
    let n = 32;
    let q = two_lines(n, 9.5, 21.5);
    let cut = threshold(&q, 0.5);

    // The lines run the full depth, through the z faces, so only x and y are
    // filtered: a line reaching l = 0 and l = n - 1 is the expected result, not
    // an edge artefact.
    let lines: Vec<_> = disclination_lines(&q.q, n, n, n, 1.0, cut)
        .into_iter()
        .filter(|c| {
            c.sites
                .iter()
                .all(|s| s.ijl.0 > 0 && s.ijl.0 < n - 1 && s.ijl.1 > 0 && s.ijl.1 < n - 1)
        })
        .collect();
    assert_eq!(lines.len(), 2, "expected two lines, got {}", lines.len());

    for c in &lines {
        // Each line runs the full depth, so its contour length reaches it.
        assert!(
            c.length >= (n - 1) as f64,
            "line of length {} in a box {n} deep",
            c.length
        );
        let mut slices: Vec<usize> = c.sites.iter().map(|s| s.ijl.2).collect();
        slices.sort_unstable();
        slices.dedup();
        assert_eq!(
            slices.len(),
            n,
            "line covers {} of {n} slices",
            slices.len()
        );
    }
}

#[test]
fn the_two_lines_carry_opposite_winding_character() {
    let n = 32;
    let q = two_lines(n, 9.5, 21.5);
    let cut = threshold(&q, 0.5);
    let mut lines: Vec<_> = disclination_lines(&q.q, n, n, n, 1.0, cut)
        .into_iter()
        .filter(|c| {
            c.sites
                .iter()
                .all(|s| s.ijl.0 > 0 && s.ijl.0 < n - 1 && s.ijl.1 > 0 && s.ijl.1 < n - 1)
        })
        .collect();
    assert_eq!(lines.len(), 2);
    lines.sort_by(|a, b| {
        let mx = |c: &volterra_braid::disclination::DisclinationCurve| {
            c.sites.iter().map(|s| s.ijl.0 as f64).sum::<f64>() / c.sites.len() as f64
        };
        mx(a).total_cmp(&mx(b))
    });
    assert!(
        lines[0].mean_cos_beta * lines[1].mean_cos_beta < 0.0,
        "the two lines gave cos(beta) {} and {}; opposite winding should differ in sign",
        lines[0].mean_cos_beta,
        lines[1].mean_cos_beta
    );
}

#[test]
fn a_uniform_field_carries_no_disclination() {
    let n = 16;
    let mut q = QField3D::zeros(n, n, n, 1.0);
    for k in 0..q.q.len() {
        q.q[k] = uniaxial([0.0, 0.0, 1.0], 0.556);
    }
    let d = disclination_density(&q.q, n, n, n, 1.0);
    let worst = d.iter().map(|x| decompose(x).s).fold(0.0_f64, f64::max);
    assert!(worst < 1e-15, "uniform field read {worst}");
    assert!(disclination_lines(&q.q, n, n, n, 1.0, 1e-12).is_empty());
}
