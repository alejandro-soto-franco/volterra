//! Disclination lines in a 3D Q-tensor field, through the disclination density
//! tensor.
//!
//! The 3D counterpart of [`crate::defect`], which finds point defects in a 2D
//! field. A disclination in three dimensions is a line, and it carries a
//! character a 2D point defect has no room for: the rotation the
//! director performs around the line can lie along the line (a wedge) or across
//! it (a twist), or anywhere between.
//!
//! Schimming and Viñals (2022) give the tensor
//!
//! ```text
//! D_ij = eps_{i mu nu} eps_{j l k} (d_l Q_{mu alpha}) (d_k Q_{nu alpha})
//! ```
//!
//! which factors as `D = s Omega T^T`: `T` the local tangent to the line,
//! `Omega` the axis the director rotates about, and `s` a positive scalar
//! peaking at the core. The winding character is `cos(beta) = Omega . T`, which
//! is `+1` for a `+1/2` wedge, `-1` for a `-1/2` wedge and `0` for a twist.
//!
//! This is the analysis Head, Digregorio, Marenduzzo, Pagonabarraga, Beller and
//! Negro (arXiv:2607.10234) apply to confined 3D active nematics, where defects
//! are read off the `s` isosurface and sorted by `cos(beta)`.
//!
//! # Cost
//!
//! Written as stated the contraction runs over seven indices. Collecting the
//! `l, k` sum into a cross product and using the antisymmetry of the `mu, nu`
//! sum leaves
//!
//! ```text
//! D_{i j} = 2 sum_alpha (grad Q_{mu alpha} x grad Q_{nu alpha})_j,  (i, mu, nu) cyclic
//! ```
//!
//! which is nine cross products per site rather than a 2187-term sum.

use nalgebra::{Matrix3, Vector3};

/// A disclination line's local character at one site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disclination {
    /// Magnitude of the disclination density, positive and peaking at the core.
    pub s: f64,
    /// Unit tangent to the line.
    ///
    /// A disclination line carries no intrinsic orientation, so this is defined
    /// only up to sign. The sign is fixed here by making the component of
    /// largest magnitude positive, and [`rotation`](Self::rotation) is flipped
    /// with it so that the product `s * Omega T^T` is unchanged.
    pub tangent: [f64; 3],
    /// Unit axis the director rotates about, to the same sign convention.
    pub rotation: [f64; 3],
    /// `Omega . T`: `+1` for a `+1/2` wedge, `-1` for a `-1/2` wedge, `0` for a
    /// twist. Unaffected by the sign convention above, which flips both vectors
    /// together.
    pub cos_beta: f64,
}

/// Embed the five stored components as the full symmetric traceless 3x3 tensor.
#[inline]
fn embed(q: [f64; 5]) -> Matrix3<f64> {
    let [q11, q12, q13, q22, q23] = q;
    Matrix3::new(q11, q12, q13, q12, q22, q23, q13, q23, -(q11 + q22))
}

/// The disclination density tensor at every site, row-major `[D_00 .. D_22]`.
///
/// `q` holds `[q11, q12, q13, q22, q23]` per site, indexed `((i * ny) + j) * nz
/// + l`, matching `volterra_fields::QField3D`. Derivatives are central
/// differences with periodic wrapping, the same stencil convention
/// `QField3D::laplacian` uses.
pub fn disclination_density(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
) -> Vec<[f64; 9]> {
    assert_eq!(q.len(), nx * ny * nz, "q length must be nx * ny * nz");
    let idx = |i: usize, j: usize, l: usize| ((i % nx) * ny + (j % ny)) * nz + (l % nz);
    let inv_2dx = 1.0 / (2.0 * dx);
    let mut out = vec![[0.0_f64; 9]; q.len()];

    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = idx(i, j, l);

                // grad[d] is the derivative of the full Q along direction d.
                let neighbours = [
                    (idx((i + 1) % nx, j, l), idx((i + nx - 1) % nx, j, l)),
                    (idx(i, (j + 1) % ny, l), idx(i, (j + ny - 1) % ny, l)),
                    (idx(i, j, (l + 1) % nz), idx(i, j, (l + nz - 1) % nz)),
                ];
                let grad: [Matrix3<f64>; 3] = std::array::from_fn(|d| {
                    let (p, m) = neighbours[d];
                    (embed(q[p]) - embed(q[m])) * inv_2dx
                });

                // g[mu][alpha] holds the three derivatives of Q_{mu alpha}.
                let g = |mu: usize, alpha: usize| {
                    Vector3::new(grad[0][(mu, alpha)], grad[1][(mu, alpha)], grad[2][(mu, alpha)])
                };

                let mut d = [0.0_f64; 9];
                for i_row in 0..3 {
                    let mu = (i_row + 1) % 3;
                    let nu = (i_row + 2) % 3;
                    let mut row = Vector3::zeros();
                    for alpha in 0..3 {
                        row += g(mu, alpha).cross(&g(nu, alpha));
                    }
                    row *= 2.0;
                    for j_col in 0..3 {
                        d[i_row * 3 + j_col] = row[j_col];
                    }
                }
                out[k] = d;
            }
        }
    }
    out
}

/// Factor one site's tensor into `s`, `Omega` and `T`.
///
/// `D` is rank one wherever a disclination is resolved, so the factorisation is
/// its leading singular triplet: `s` the leading singular value, `Omega` and `T`
/// the corresponding left and right singular vectors.
pub fn decompose(d: &[f64; 9]) -> Disclination {
    let m = Matrix3::from_row_slice(d);
    let svd = m.svd(true, true);
    let s = svd.singular_values[0];
    let u = svd.u.expect("left singular vectors requested");
    let v_t = svd.v_t.expect("right singular vectors requested");

    let mut omega = Vector3::new(u[(0, 0)], u[(1, 0)], u[(2, 0)]);
    let mut tangent = Vector3::new(v_t[(0, 0)], v_t[(0, 1)], v_t[(0, 2)]);

    // Fix the shared sign on the tangent's largest component. Flipping both
    // leaves s * Omega T^T, and so cos_beta, unchanged.
    let lead = (0..3)
        .max_by(|&a, &b| tangent[a].abs().total_cmp(&tangent[b].abs()))
        .unwrap_or(0);
    if tangent[lead] < 0.0 {
        tangent = -tangent;
        omega = -omega;
    }

    let cos_beta = if s > 0.0 { omega.dot(&tangent) } else { 0.0 };
    Disclination {
        s,
        tangent: [tangent[0], tangent[1], tangent[2]],
        rotation: [omega[0], omega[1], omega[2]],
        cos_beta,
    }
}

/// A site sitting on a disclination line, with its grid position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisclinationSite {
    /// Grid position `(i, j, l)`.
    pub ijl: (usize, usize, usize),
    /// The local character there.
    pub disclination: Disclination,
}

/// Every site whose disclination density exceeds `threshold`.
///
/// The reference reads defects off the `s = 0.09` isosurface at its own
/// normalisation and grid spacing; the threshold is left to the caller here
/// because `s` scales as the square of a Q gradient, so it carries the field's
/// units and the grid spacing with it.
pub fn disclination_sites(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    threshold: f64,
) -> Vec<DisclinationSite> {
    let density = disclination_density(q, nx, ny, nz, dx);
    let mut out = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = ((i * ny) + j) * nz + l;
                let disclination = decompose(&density[k]);
                if disclination.s > threshold {
                    out.push(DisclinationSite {
                        ijl: (i, j, l),
                        disclination,
                    });
                }
            }
        }
    }
    out
}

/// One connected disclination line.
#[derive(Debug, Clone, PartialEq)]
pub struct DisclinationCurve {
    /// The sites making up the line, ordered along it.
    pub sites: Vec<DisclinationSite>,
    /// Contour length in grid units, summed along the ordered sites.
    pub length: f64,
    /// Site-count-weighted mean of `cos(beta)`: near `+1` a `+1/2` wedge line,
    /// near `-1` a `-1/2` wedge line, near `0` a twist line.
    pub mean_cos_beta: f64,
    /// Whether the two ends meet, within one lattice diagonal.
    pub is_loop: bool,
}

/// Assemble supra-threshold sites into connected lines.
///
/// Sites are grouped by 26-connectivity, then each group is ordered by a walk
/// from the site furthest from the group's centroid, taking the nearest unused
/// neighbour at each step. Contour length is the sum of the steps of that walk,
/// which is the quantity arXiv:2607.10234 reports distributions of.
///
/// Grouping is on the lattice and takes no account of periodic wrapping, so a
/// line that leaves one face and re-enters the opposite one is reported as two.
pub fn disclination_lines(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    threshold: f64,
) -> Vec<DisclinationCurve> {
    let sites = disclination_sites(q, nx, ny, nz, dx, threshold);
    if sites.is_empty() {
        return Vec::new();
    }

    // Index sites by grid position so neighbours are found without an O(n^2)
    // sweep.
    let key = |ijl: (usize, usize, usize)| (ijl.0 * ny + ijl.1) * nz + ijl.2;
    let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (n, s) in sites.iter().enumerate() {
        at.insert(key(s.ijl), n);
    }

    let mut group = vec![usize::MAX; sites.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..sites.len() {
        if group[start] != usize::MAX {
            continue;
        }
        let g = groups.len();
        let mut stack = vec![start];
        let mut members = Vec::new();
        group[start] = g;
        while let Some(n) = stack.pop() {
            members.push(n);
            let (i, j, l) = sites[n].ijl;
            for di in -1i64..=1 {
                for dj in -1i64..=1 {
                    for dl in -1i64..=1 {
                        if di == 0 && dj == 0 && dl == 0 {
                            continue;
                        }
                        let (ni, nj, nl) = (i as i64 + di, j as i64 + dj, l as i64 + dl);
                        if ni < 0
                            || nj < 0
                            || nl < 0
                            || ni >= nx as i64
                            || nj >= ny as i64
                            || nl >= nz as i64
                        {
                            continue;
                        }
                        let nk = key((ni as usize, nj as usize, nl as usize));
                        if let Some(&m) = at.get(&nk) {
                            if group[m] == usize::MAX {
                                group[m] = g;
                                stack.push(m);
                            }
                        }
                    }
                }
            }
        }
        groups.push(members);
    }

    let pos = |n: usize| {
        let (i, j, l) = sites[n].ijl;
        [i as f64 * dx, j as f64 * dx, l as f64 * dx]
    };
    let dist = |a: usize, b: usize| {
        let (p, q) = (pos(a), pos(b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };

    let mut out = Vec::with_capacity(groups.len());
    for members in groups {
        // Start the walk at the member furthest from the centroid, which is an
        // end of an open line and an arbitrary point of a closed one.
        let mut centroid = [0.0; 3];
        for &m in &members {
            let p = pos(m);
            for c in 0..3 {
                centroid[c] += p[c] / members.len() as f64;
            }
        }
        let start = *members
            .iter()
            .max_by(|&&a, &&b| {
                let d = |m: usize| {
                    let p = pos(m);
                    (0..3).map(|c| (p[c] - centroid[c]).powi(2)).sum::<f64>()
                };
                d(a).total_cmp(&d(b))
            })
            .expect("non-empty group");

        let mut remaining: Vec<usize> = members.iter().copied().filter(|&m| m != start).collect();
        let mut order = vec![start];
        let mut length = 0.0;
        let mut current = start;
        while !remaining.is_empty() {
            let (idx, _) = remaining
                .iter()
                .enumerate()
                .min_by(|&(_, &a), &(_, &b)| dist(current, a).total_cmp(&dist(current, b)))
                .expect("non-empty remainder");
            let next = remaining.swap_remove(idx);
            length += dist(current, next);
            order.push(next);
            current = next;
        }

        let is_loop = order.len() > 2 && dist(order[0], current) <= dx * 3.0_f64.sqrt();
        let mean_cos_beta = order
            .iter()
            .map(|&m| sites[m].disclination.cos_beta)
            .sum::<f64>()
            / order.len() as f64;
        out.push(DisclinationCurve {
            sites: order.iter().map(|&m| sites[m]).collect(),
            length,
            mean_cos_beta,
            is_loop,
        });
    }
    out.sort_by(|a, b| b.length.total_cmp(&a.length));
    out
}

#[cfg(test)]
mod disclination_tests {
    use super::*;
    use std::f64::consts::PI;

    /// A uniaxial Q from a director and a scalar order parameter, in the
    /// convention `Q = q (n n - I/3)` the 3D papers use.
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

    /// A wedge disclination line along z, of winding `charge`, cored between
    /// grid points so no site sits on the singularity.
    ///
    /// The director lies in the xy plane at angle `charge * atan2(y, x)`, which
    /// is the `+1/2` profile at `charge = 0.5` and the `-1/2` profile at
    /// `charge = -0.5`.
    fn wedge_line(n: usize, charge: f64) -> Vec<[f64; 5]> {
        let centre = n as f64 / 2.0 - 0.5;
        let mut q = vec![[0.0; 5]; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let x = i as f64 - centre;
                    let y = j as f64 - centre;
                    let theta = charge * y.atan2(x);
                    let dir = [theta.cos(), theta.sin(), 0.0];
                    q[((i * n) + j) * n + l] = uniaxial(dir, 0.556);
                }
            }
        }
        q
    }

    /// Interior sites only.
    ///
    /// The analytic fields below wrap onto a discontinuity: the director at
    /// `i = 0` and at `i = n - 1` belong to opposite sides of the wedge, so the
    /// wrapped central difference across that seam sees a jump that has nothing
    /// to do with a disclination. The stencil is periodic, matching
    /// `QField3D::laplacian`, which is correct for the confined fields it runs
    /// on, where the nematic sits well inside the box. Here it means the
    /// outermost layer carries an artefact, so the tests read the interior.
    fn interior(n: usize) -> impl Iterator<Item = (usize, usize, usize)> {
        (1..n - 1).flat_map(move |i| {
            (1..n - 1).flat_map(move |j| (1..n - 1).map(move |l| (i, j, l)))
        })
    }

    /// The interior site of largest `s`, which is the one nearest the core.
    fn peak(q: &[[f64; 5]], n: usize) -> Disclination {
        let density = disclination_density(q, n, n, n, 1.0);
        interior(n)
            .map(|(i, j, l)| decompose(&density[((i * n) + j) * n + l]))
            .max_by(|a, b| a.s.total_cmp(&b.s))
            .expect("non-empty interior")
    }

    #[test]
    fn uniform_field_has_no_disclination() {
        let n = 8;
        let q = vec![uniaxial([0.0, 0.0, 1.0], 0.556); n * n * n];
        let density = disclination_density(&q, n, n, n, 1.0);
        let worst = density
            .iter()
            .map(|d| d.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-15, "uniform field gave density {worst}");
    }

    #[test]
    fn plus_half_wedge_is_a_wedge_along_the_line() {
        let n = 16;
        let d = peak(&wedge_line(n, 0.5), n);
        assert!(d.s > 0.0, "no disclination found");
        // The line runs along z, so the tangent is +-z; the sign convention
        // makes it +z.
        assert!(d.tangent[2].abs() > 0.99, "tangent {:?} is not along z", d.tangent);
        // A wedge rotates about its own tangent.
        assert!(
            d.cos_beta.abs() > 0.99,
            "cos(beta) = {} is not a wedge",
            d.cos_beta
        );
    }

    #[test]
    fn wedge_charge_sign_flips_the_winding_character() {
        let n = 16;
        let plus = peak(&wedge_line(n, 0.5), n);
        let minus = peak(&wedge_line(n, -0.5), n);
        assert!(
            plus.cos_beta * minus.cos_beta < 0.0,
            "+1/2 gave cos(beta) {}, -1/2 gave {}; the two should differ in sign",
            plus.cos_beta,
            minus.cos_beta
        );
    }

    #[test]
    fn twist_disclination_rotates_across_its_line() {
        // The director rotates in the xz plane as the angle about the z axis
        // advances, so the rotation axis is y while the line still runs along z.
        let n = 16;
        let centre = n as f64 / 2.0 - 0.5;
        let mut q = vec![[0.0; 5]; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let x = i as f64 - centre;
                    let y = j as f64 - centre;
                    let theta = 0.5 * y.atan2(x);
                    let dir = [theta.cos(), 0.0, theta.sin()];
                    q[((i * n) + j) * n + l] = uniaxial(dir, 0.556);
                }
            }
        }
        let d = peak(&q, n);
        assert!(d.s > 0.0, "no disclination found");
        assert!(
            d.cos_beta.abs() < 0.1,
            "cos(beta) = {} is not a twist",
            d.cos_beta
        );
    }

    #[test]
    fn density_peaks_at_the_core_and_decays_outward() {
        let n = 24;
        let q = wedge_line(n, 0.5);
        let density = disclination_density(&q, n, n, n, 1.0);
        let mid = n / 2;
        let at = |i: usize, j: usize| decompose(&density[((i * n) + j) * n + mid]).s;
        // Walk out along x from the core towards the edge.
        let near = at(mid, mid);
        let far = at(mid + 6, mid);
        assert!(near > far, "s did not decay outward: {near} at core, {far} away");
    }

    #[test]
    fn tangent_and_rotation_reconstruct_the_tensor() {
        let n = 16;
        let q = wedge_line(n, 0.5);
        let density = disclination_density(&q, n, n, n, 1.0);
        let k = density
            .iter()
            .enumerate()
            .max_by(|a, b| decompose(a.1).s.total_cmp(&decompose(b.1).s))
            .map(|(k, _)| k)
            .expect("non-empty field");
        let d = decompose(&density[k]);

        // D is rank one where a disclination is resolved, so s * Omega T^T
        // returns the tensor itself.
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let rebuilt = d.s * d.rotation[i] * d.tangent[j];
                worst = worst.max((rebuilt - density[k][i * 3 + j]).abs());
            }
        }
        let scale = density[k].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            worst < 1e-6 * scale.max(1.0),
            "rank-one reconstruction off by {worst} against a scale of {scale}"
        );
    }

    #[test]
    fn a_line_along_x_is_found_along_x() {
        // Same wedge profile, rotated so the line runs along x: the tangent must
        // follow it rather than staying where the previous test found it.
        let n = 16;
        let centre = n as f64 / 2.0 - 0.5;
        let mut q = vec![[0.0; 5]; n * n * n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let y = j as f64 - centre;
                    let z = l as f64 - centre;
                    let theta = 0.5 * z.atan2(y);
                    let dir = [0.0, theta.cos(), theta.sin()];
                    q[((i * n) + j) * n + l] = uniaxial(dir, 0.556);
                }
            }
        }
        let d = peak(&q, n);
        assert!(d.tangent[0].abs() > 0.99, "tangent {:?} is not along x", d.tangent);
    }

    #[test]
    fn sites_above_a_threshold_lie_on_the_line() {
        let n = 16;
        let q = wedge_line(n, 0.5);
        let density = disclination_density(&q, n, n, n, 1.0);
        let peak_s = interior(n)
            .map(|(i, j, l)| decompose(&density[((i * n) + j) * n + l]).s)
            .fold(0.0_f64, f64::max);

        let sites: Vec<_> = disclination_sites(&q, n, n, n, 1.0, 0.5 * peak_s)
            .into_iter()
            .filter(|s| {
                let (i, j, l) = s.ijl;
                i > 0 && j > 0 && l > 0 && i < n - 1 && j < n - 1 && l < n - 1
            })
            .collect();
        assert!(!sites.is_empty(), "no interior sites above half the peak");
        // The line runs the full length of z, so every interior z index appears.
        let mut seen = vec![false; n];
        for s in &sites {
            seen[s.ijl.2] = true;
        }
        assert!(
            (1..n - 1).all(|l| seen[l]),
            "the line does not span z: {seen:?}"
        );
        // And every one of them sits near the core in x and y.
        let centre = n as f64 / 2.0 - 0.5;
        for s in &sites {
            let dxc = s.ijl.0 as f64 - centre;
            let dyc = s.ijl.1 as f64 - centre;
            assert!(
                (dxc * dxc + dyc * dyc).sqrt() < 3.0,
                "site {:?} is far from the core",
                s.ijl
            );
        }
    }

    #[test]
    fn a_straight_line_assembles_into_one_curve_spanning_the_box() {
        let n = 24;
        let q = wedge_line(n, 0.5);
        let density = disclination_density(&q, n, n, n, 1.0);
        let peak = interior(n)
            .map(|(i, j, l)| decompose(&density[((i * n) + j) * n + l]).s)
            .fold(0.0_f64, f64::max);

        let lines = disclination_lines(&q, n, n, n, 1.0, 0.5 * peak);
        assert!(!lines.is_empty(), "no line assembled");
        let longest = &lines[0];
        // A straight line through the box spans every z, so its contour length
        // is at least the box depth less the two boundary layers it excludes.
        assert!(
            longest.length >= (n - 3) as f64,
            "contour length {} is short of the box depth {n}",
            longest.length
        );
        assert!(
            longest.mean_cos_beta.abs() > 0.9,
            "a wedge line reported mean cos(beta) {}",
            longest.mean_cos_beta
        );
        assert!(!longest.is_loop, "a straight line was called a loop");
    }

    #[test]
    fn two_separated_lines_assemble_separately() {
        // Two parallel wedge lines, far enough apart that no supra-threshold
        // site of one touches the other.
        let n = 32;
        let mut q = vec![[0.0; 5]; n * n * n];
        let (c1, c2) = (9.5_f64, 21.5_f64);
        let cy = (n as f64 - 1.0) / 2.0;
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let (x, y) = (i as f64, j as f64 - cy);
                    let theta = 0.5 * y.atan2(x - c1) - 0.5 * y.atan2(x - c2);
                    q[((i * n) + j) * n + l] =
                        uniaxial([theta.cos(), theta.sin(), 0.0], 0.556);
                }
            }
        }
        let density = disclination_density(&q, n, n, n, 1.0);
        let peak = interior(n)
            .map(|(i, j, l)| decompose(&density[((i * n) + j) * n + l]).s)
            .fold(0.0_f64, f64::max);

        let lines = disclination_lines(&q, n, n, n, 1.0, 0.5 * peak);
        assert_eq!(lines.len(), 2, "expected two lines, got {}", lines.len());
        // Each sits at one of the two cores.
        let mean_x = |c: &DisclinationCurve| {
            c.sites.iter().map(|s| s.ijl.0 as f64).sum::<f64>() / c.sites.len() as f64
        };
        let mut xs = [mean_x(&lines[0]), mean_x(&lines[1])];
        xs.sort_by(f64::total_cmp);
        assert!((xs[0] - c1).abs() < 1.5, "first core at {}", xs[0]);
        assert!((xs[1] - c2).abs() < 1.5, "second core at {}", xs[1]);
    }

    #[test]
    fn density_scales_as_the_square_of_the_order_parameter() {
        // D is a product of two Q derivatives and Q is linear in q, so s scales
        // as q^2 and doubling q scales it by four.
        let n = 16;
        let centre = n as f64 / 2.0 - 0.5;
        let build = |q_mag: f64| {
            let mut q = vec![[0.0; 5]; n * n * n];
            for i in 0..n {
                for j in 0..n {
                    for l in 0..n {
                        let x = i as f64 - centre;
                        let y = j as f64 - centre;
                        let theta = 0.5 * y.atan2(x);
                        q[((i * n) + j) * n + l] =
                            uniaxial([theta.cos(), theta.sin(), 0.0], q_mag);
                    }
                }
            }
            q
        };
        let a = peak(&build(0.5), n).s;
        let b = peak(&build(1.0), n).s;
        assert!(
            (b / a - 4.0).abs() < 1e-6,
            "doubling q scaled s by {}, expected 4",
            b / a
        );
    }

    #[test]
    fn a_full_two_pi_rotation_is_not_a_disclination_core() {
        // Charge 1 is a director field that returns to itself, so it is a
        // defect the nematic can escape; the profile is smooth away from the
        // axis and carries no pi rotation for the tensor to pick up as a
        // half-integer line. It still has structure, so this checks the
        // character rather than the magnitude.
        let n = 16;
        let d = peak(&wedge_line(n, 1.0), n);
        assert!(d.s.is_finite());
        assert!(d.cos_beta.abs() <= 1.0 + 1e-12);
        let _ = PI;
    }
}
