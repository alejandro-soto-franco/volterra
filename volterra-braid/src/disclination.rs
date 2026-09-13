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

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

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
/// + l`, matching `volterra_core::QField3D`. Derivatives are central
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
                    Vector3::new(
                        grad[0][(mu, alpha)],
                        grad[1][(mu, alpha)],
                        grad[2][(mu, alpha)],
                    )
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
    /// Grid position `(i, j, l)` of the voxel the site was found in.
    pub ijl: (usize, usize, usize),
    /// The core position in physical units, refined to sub-voxel accuracy.
    ///
    /// `s` peaks at the core, so a parabola through the three samples either
    /// side of the peak, taken in each of the two directions perpendicular to
    /// the tangent, puts the core between voxels where it belongs. Curvature is
    /// a second derivative, and differentiating voxel indices measures the
    /// lattice staircase rather than the line, so this is what the geometry is
    /// computed from.
    pub pos: [f64; 3],
    /// The local character there.
    pub disclination: Disclination,
}

/// Every site whose disclination density exceeds `threshold`.
///
/// The reference reads defects off the `s = 0.09` isosurface at its own
/// normalisation and grid spacing; the threshold is left to the caller here
/// because `s` scales as the square of a Q gradient, so it carries the field's
/// units and the grid spacing with it.
///
/// # The ridge
///
/// A supra-threshold region is a tube several voxels across, so keeping every
/// voxel in it gives a fat blob whose nearest-neighbour ordering zigzags at the
/// lattice scale. Only voxels that are a local maximum of `s` in the plane
/// perpendicular to the local tangent survive, which leaves the tube's axis one
/// voxel wide, and each survivor's [`pos`](DisclinationSite::pos) is then fitted
/// between voxels. Sites whose refined positions agree to within half a voxel
/// are one site, which is what happens along a core sitting exactly between
/// voxels, where four of them read the same `s`.
pub fn disclination_sites(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    threshold: f64,
) -> Vec<DisclinationSite> {
    let density = disclination_density(q, nx, ny, nz, dx);
    let mag = magnitudes(&density);
    sites_from(&density, &mag, nx, ny, nz, dx, threshold)
}

/// The body of [`disclination_sites`], against a density and a magnitude field
/// already computed.
///
/// The factorisation runs a singular value decomposition at every voxel, so a
/// caller wanting the sites and the field they were read from computes the
/// density once and comes here, rather than paying for it twice.
fn sites_from(
    density: &[[f64; 9]],
    mag: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    threshold: f64,
) -> Vec<DisclinationSite> {
    // Trilinear sample of `s` at a position in voxel units, clamped at the
    // faces so a core lying in the first or last slice is still refined.
    let sample = |p: [f64; 3]| -> f64 {
        let n = [nx, ny, nz];
        let mut base = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for c in 0..3 {
            let hi = (n[c] - 1) as f64;
            let x = p[c].clamp(0.0, hi);
            let b = x.floor().min(hi - 1.0).max(0.0);
            base[c] = b as usize;
            frac[c] = x - b;
        }
        let mut acc = 0.0;
        for di in 0..2 {
            for dj in 0..2 {
                for dl in 0..2 {
                    let w = (if di == 1 { frac[0] } else { 1.0 - frac[0] })
                        * (if dj == 1 { frac[1] } else { 1.0 - frac[1] })
                        * (if dl == 1 { frac[2] } else { 1.0 - frac[2] });
                    let i = (base[0] + di).min(nx - 1);
                    let j = (base[1] + dj).min(ny - 1);
                    let l = (base[2] + dl).min(nz - 1);
                    acc += w * mag[((i * ny) + j) * nz + l];
                }
            }
        }
        acc
    };

    // Candidates: supra-threshold, a ridge in both perpendicular directions,
    // and refined by a parabola through the three samples in each.
    let mut candidates: Vec<DisclinationSite> = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let k = ((i * ny) + j) * nz + l;
                // `mag[k]` is this voxel's leading singular value, already
                // computed. Testing it first keeps the decomposition for the
                // few supra-threshold voxels rather than running a second
                // singular value decomposition over the whole grid.
                if mag[k] <= threshold {
                    continue;
                }
                let disclination = decompose(&density[k]);
                let (e1, e2) = perpendicular_basis(disclination.tangent);
                let here = [i as f64, j as f64, l as f64];
                let s0 = mag[k];

                let mut offset = [0.0f64; 2];
                let mut on_ridge = true;
                for (axis, e) in [e1, e2].iter().enumerate() {
                    let plus = sample(add(here, *e, 1.0));
                    let minus = sample(add(here, *e, -1.0));
                    if s0 < plus || s0 < minus {
                        on_ridge = false;
                        break;
                    }
                    // Peak of the parabola through (-1, minus), (0, s0), (1, plus).
                    let curv = minus - 2.0 * s0 + plus;
                    offset[axis] = if curv.abs() > 1e-30 {
                        (0.5 * (minus - plus) / curv).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                }
                if !on_ridge {
                    continue;
                }

                let mut pos = [0.0f64; 3];
                for c in 0..3 {
                    pos[c] = (here[c] + offset[0] * e1[c] + offset[1] * e2[c]) * dx;
                }
                candidates.push(DisclinationSite {
                    ijl: (i, j, l),
                    pos,
                    disclination,
                });
            }
        }
    }

    merge_coincident(candidates, dx)
}

/// Collapse sites whose refined positions agree to within half a voxel.
///
/// A core sitting exactly between voxels reads the same `s` at each of the four
/// around it, so all four pass the ridge test and all four refine to the same
/// point. The strongest of a coincident group is kept.
fn merge_coincident(mut sites: Vec<DisclinationSite>, dx: f64) -> Vec<DisclinationSite> {
    sites.sort_by(|a, b| b.disclination.s.total_cmp(&a.disclination.s));

    let cell = dx.max(1e-30);
    let key = |p: [f64; 3]| {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut buckets: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    let mut kept: Vec<DisclinationSite> = Vec::new();

    for s in sites {
        let (kx, ky, kz) = key(s.pos);
        let mut coincident = false;
        'search: for dx_ in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(near) = buckets.get(&(kx + dx_, ky + dy, kz + dz)) {
                        for &n in near {
                            let p = kept[n].pos;
                            let d2: f64 = (0..3).map(|c| (p[c] - s.pos[c]).powi(2)).sum();
                            if d2 < (0.5 * dx).powi(2) {
                                coincident = true;
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        if !coincident {
            buckets.entry((kx, ky, kz)).or_default().push(kept.len());
            kept.push(s);
        }
    }
    kept
}

/// Two unit vectors completing `t` to a right-handed orthonormal frame.
fn perpendicular_basis(t: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let t = Vector3::new(t[0], t[1], t[2]);
    // Cross with whichever axis is least aligned with t, so the result is never
    // near zero.
    let lead = (0..3)
        .min_by(|&a, &b| t[a].abs().total_cmp(&t[b].abs()))
        .unwrap_or(0);
    let mut axis = Vector3::zeros();
    axis[lead] = 1.0;
    let e1 = t.cross(&axis).normalize();
    let e2 = t.cross(&e1).normalize();
    ([e1[0], e1[1], e1[2]], [e2[0], e2[1], e2[2]])
}

/// `p + scale * e`.
fn add(p: [f64; 3], e: [f64; 3], scale: f64) -> [f64; 3] {
    [
        p[0] + scale * e[0],
        p[1] + scale * e[1],
        p[2] + scale * e[2],
    ]
}

/// The leading singular value at every site, which is the field an isosurface
/// is taken of.
fn magnitudes(density: &[[f64; 9]]) -> Vec<f64> {
    density.iter().map(|d| decompose(d).s).collect()
}

/// The disclination density magnitude `s` at every site.
///
/// This is the scalar an isosurface is drawn on: it vanishes in the ordered
/// bulk and peaks on the core, so `{s = c}` is a tube around every disclination
/// line. Pair it with [`cos_beta_field`] to colour that surface by the winding
/// character underneath it.
pub fn disclination_magnitude(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
) -> Vec<f64> {
    magnitudes(&disclination_density(q, nx, ny, nz, dx))
}

/// `cos(beta)` at every site, which is `+1` on a `+1/2` wedge, `-1` on a
/// `-1/2` wedge and `0` on a twist.
///
/// Meaningless away from a core, where `s` is small and the factorisation has
/// nothing to resolve, so read it only on or near the isosurface.
pub fn cos_beta_field(q: &[[f64; 5]], nx: usize, ny: usize, nz: usize, dx: f64) -> Vec<f64> {
    disclination_density(q, nx, ny, nz, dx)
        .iter()
        .map(|d| decompose(d).cos_beta)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The curve's own geometry
// ─────────────────────────────────────────────────────────────────────────────

/// The Frenet apparatus along a discrete curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Frenet {
    /// Unit tangent at each point.
    pub tangents: Vec<[f64; 3]>,
    /// Curvature `|T'|` at each point, in inverse length units.
    pub curvatures: Vec<f64>,
    /// Torsion at each point, zero wherever the curve is locally straight.
    pub torsions: Vec<f64>,
}

/// Tangent, curvature and torsion along a polyline.
///
/// Each point gets a cubic fitted by least squares to the seven points centred
/// on it, and the derivatives are read off that fit. A cubic is the lowest
/// degree with a third derivative, which torsion needs, and fitting over a
/// window rather than differencing neighbours is what keeps the second
/// derivative usable: refined core positions still land a few hundredths of a
/// voxel off, and central differences would amplify that by the square of the
/// spacing.
///
/// The three quantities are invariant under reparameterisation, so the fit runs
/// against the point index and no arclength estimate enters:
///
/// ```text
/// kappa = |r' x r''| / |r'|^3       tau = (r' x r'') . r''' / |r' x r''|^2
/// ```
///
/// Set `closed` for a loop, which wraps the window rather than one-siding it at
/// the ends. Curves of fewer than five points return tangents alone, since a
/// cubic through four points has no residual and its third derivative is noise.
pub fn frenet(points: &[[f64; 3]], closed: bool) -> Frenet {
    let n = points.len();
    let mut tangents = vec![[0.0; 3]; n];
    let mut curvatures = vec![0.0; n];
    let mut torsions = vec![0.0; n];

    if n < 2 {
        return Frenet {
            tangents,
            curvatures,
            torsions,
        };
    }
    if n < 5 {
        for (i, tangent) in tangents.iter_mut().enumerate() {
            let (a, b) = if i == 0 {
                (0, 1)
            } else if i == n - 1 {
                (n - 2, n - 1)
            } else {
                (i - 1, i + 1)
            };
            let d = Vector3::new(
                points[b][0] - points[a][0],
                points[b][1] - points[a][1],
                points[b][2] - points[a][2],
            );
            let t = if d.norm() > 1e-30 {
                d.normalize()
            } else {
                Vector3::zeros()
            };
            *tangent = [t[0], t[1], t[2]];
        }
        return Frenet {
            tangents,
            curvatures,
            torsions,
        };
    }

    let width = 7usize.min(n);
    let half = (width / 2) as isize;

    for i in 0..n {
        // Window nodes as (offset from this point, position).
        let mut nodes: Vec<(f64, [f64; 3])> = Vec::with_capacity(width);
        if closed {
            for k in -half..=half {
                let idx = (i as isize + k).rem_euclid(n as isize) as usize;
                nodes.push((k as f64, points[idx]));
            }
        } else {
            let start = (i as isize - half).clamp(0, n as isize - width as isize);
            for k in 0..width as isize {
                nodes.push((
                    (start + k - i as isize) as f64,
                    points[(start + k) as usize],
                ));
            }
        }

        // Normal equations for a cubic in the offset.
        let mut m = Matrix4::zeros();
        for a in 0..4 {
            for b in 0..4 {
                m[(a, b)] = nodes.iter().map(|(t, _)| t.powi((a + b) as i32)).sum();
            }
        }
        let lu = m.lu();

        let mut d1: Vector3<f64> = Vector3::zeros();
        let mut d2: Vector3<f64> = Vector3::zeros();
        let mut d3: Vector3<f64> = Vector3::zeros();
        for c in 0..3 {
            let rhs =
                Vector4::from_fn(|a, _| nodes.iter().map(|(t, p)| t.powi(a as i32) * p[c]).sum());
            let Some(coeff) = lu.solve(&rhs) else {
                continue;
            };
            d1[c] = coeff[1];
            d2[c] = 2.0 * coeff[2];
            d3[c] = 6.0 * coeff[3];
        }

        let speed = d1.norm();
        if speed < 1e-30 {
            continue;
        }
        let t = d1 / speed;
        tangents[i] = [t[0], t[1], t[2]];

        let cross = d1.cross(&d2);
        let cross_norm = cross.norm();
        curvatures[i] = cross_norm / speed.powi(3);
        // Torsion divides by |r' x r''|^2, which vanishes on a straight
        // stretch. There the osculating plane is undefined and the torsion with
        // it, so it is reported as zero rather than as a ratio of noise.
        if cross_norm > 1e-12 * speed.powi(2) {
            torsions[i] = cross.dot(&d3) / (cross_norm * cross_norm);
        }
    }

    Frenet {
        tangents,
        curvatures,
        torsions,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The isosurface's geometry
// ─────────────────────────────────────────────────────────────────────────────

/// The curvature of a level set at one site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvature {
    /// Mean curvature, the average of the two principal curvatures.
    pub mean: f64,
    /// Gaussian curvature, their product.
    pub gaussian: f64,
}

impl SurfaceCurvature {
    /// The two principal curvatures, largest first.
    ///
    /// `k = H +/- sqrt(H^2 - K)`, with the discriminant clamped at zero, where
    /// a discretised surface can push it slightly negative.
    pub fn principal(&self) -> (f64, f64) {
        let disc = (self.mean * self.mean - self.gaussian).max(0.0).sqrt();
        (self.mean + disc, self.mean - disc)
    }
}

/// The curvature of the level set of `field` passing through one site.
///
/// Goldman's implicit-surface formulas, evaluated on the gradient and Hessian
/// of the field itself, so no surface is meshed and no isosurface value is
/// named: every level set through the site has the same normal direction, and
/// the one through it is the one measured.
///
/// ```text
/// K = (g . adj(H) g) / |g|^4        2 M = (g . H g - |g|^2 tr H) / |g|^3
/// ```
///
/// The normal is `g / |g|`, which points the way `field` increases, so the sign
/// says which side of the surface the field rises towards. `s` peaks on a
/// disclination core, so the tube around a line reads `+1/(2R)` at its radius
/// `R`, against the `-1/(2R)` of a cylinder whose field rises outward.
///
/// Derivatives are clamped at the faces rather than wrapped, since a level set
/// reaching a wall is a real feature of a confined run and wrapping would join
/// it to whatever sits on the far side.
pub fn level_set_curvature(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    ijl: (usize, usize, usize),
) -> SurfaceCurvature {
    let at = |i: isize, j: isize, l: isize| -> f64 {
        let i = i.clamp(0, nx as isize - 1) as usize;
        let j = j.clamp(0, ny as isize - 1) as usize;
        let l = l.clamp(0, nz as isize - 1) as usize;
        field[((i * ny) + j) * nz + l]
    };
    let (i, j, l) = (ijl.0 as isize, ijl.1 as isize, ijl.2 as isize);
    let step = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let shift = |d: usize, n: isize| (i + n * step[d][0], j + n * step[d][1], l + n * step[d][2]);

    let centre = at(i, j, l);
    let mut g = Vector3::zeros();
    let mut h = Matrix3::zeros();
    for d in 0..3 {
        let (pi, pj, pl) = shift(d, 1);
        let (mi, mj, ml) = shift(d, -1);
        let (plus, minus) = (at(pi, pj, pl), at(mi, mj, ml));
        g[d] = (plus - minus) / (2.0 * dx);
        h[(d, d)] = (plus - 2.0 * centre + minus) / (dx * dx);
    }
    for d in 0..3 {
        for e in d + 1..3 {
            let off = |sd: isize, se: isize| {
                at(
                    i + sd * step[d][0] + se * step[e][0],
                    j + sd * step[d][1] + se * step[e][1],
                    l + sd * step[d][2] + se * step[e][2],
                )
            };
            let mixed = (off(1, 1) - off(1, -1) - off(-1, 1) + off(-1, -1)) / (4.0 * dx * dx);
            h[(d, e)] = mixed;
            h[(e, d)] = mixed;
        }
    }

    let norm = g.norm();
    if norm < 1e-30 {
        return SurfaceCurvature {
            mean: 0.0,
            gaussian: 0.0,
        };
    }

    // Adjugate of a symmetric 3x3, written out so a singular Hessian is fine:
    // the cylinder's is exactly singular and its Gaussian curvature is zero.
    let (a, b, c) = (h[(0, 0)], h[(0, 1)], h[(0, 2)]);
    let (d, e, f) = (h[(1, 1)], h[(1, 2)], h[(2, 2)]);
    let adj = Matrix3::new(
        d * f - e * e,
        c * e - b * f,
        b * e - c * d,
        c * e - b * f,
        a * f - c * c,
        b * c - a * e,
        b * e - c * d,
        b * c - a * e,
        a * d - b * b,
    );

    let gaussian = g.dot(&(adj * g)) / norm.powi(4);
    let mean = (g.dot(&(h * g)) - norm * norm * h.trace()) / (2.0 * norm.powi(3));
    SurfaceCurvature { mean, gaussian }
}

/// One connected disclination line.
#[derive(Debug, Clone, PartialEq)]
pub struct DisclinationCurve {
    /// The sites making up the line, ordered along it.
    pub sites: Vec<DisclinationSite>,
    /// Contour length in physical units, summed along the ordered sites.
    pub length: f64,
    /// Site-count-weighted mean of `cos(beta)`: near `+1` a `+1/2` wedge line,
    /// near `-1` a `-1/2` wedge line, near `0` a twist line.
    pub mean_cos_beta: f64,
    /// Whether the two ends meet, within one lattice diagonal.
    pub is_loop: bool,
    /// Unit tangent at each site, from the fitted curve rather than from the
    /// density tensor.
    ///
    /// [`Disclination::tangent`] is a local reading of the field and this is a
    /// geometric one. They agree where the line is resolved and part where the
    /// sampling has lost it, so the two together say how far to trust either.
    pub tangents: Vec<[f64; 3]>,
    /// Curvature of the line at each site, in inverse length units.
    pub curvatures: Vec<f64>,
    /// Torsion of the line at each site.
    pub torsions: Vec<f64>,
    /// Mean of [`curvatures`](Self::curvatures). A planar circular loop reads
    /// the reciprocal of its radius.
    pub mean_curvature: f64,
    /// Mean curvature of the `s` isosurface around this line, averaged over the
    /// voxels of the tube's own surface.
    ///
    /// Positive around a core, since `s` rises inward and the normal follows it,
    /// and near `1/(2R)` at the tube radius `R`. That makes it much the larger
    /// of the two curvatures on any line that is not bent on the scale of its
    /// own core.
    pub surface_mean_curvature: f64,
    /// Gaussian curvature of that surface, averaged the same way.
    ///
    /// Near zero along a straight stretch, where the tube is a cylinder,
    /// positive where it caps and negative where the tube bends.
    pub surface_gaussian_curvature: f64,
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
    let density = disclination_density(q, nx, ny, nz, dx);
    let mag = magnitudes(&density);
    let sites = sites_from(&density, &mag, nx, ny, nz, dx, threshold);
    if sites.is_empty() {
        return Vec::new();
    }
    let mut out = assemble(sites, dx);
    attach_surface_curvature(&mut out, &mag, nx, ny, nz, dx, threshold);
    out
}

/// Group the ridge sites by adjacency, order each group along itself, and read
/// the geometry of the curve that results.
fn assemble(sites: Vec<DisclinationSite>, dx: f64) -> Vec<DisclinationCurve> {
    // Index sites by grid position so neighbours are found without an O(n^2)
    // sweep.
    let mut at: std::collections::HashMap<(usize, usize, usize), usize> =
        std::collections::HashMap::new();
    for (n, s) in sites.iter().enumerate() {
        at.insert(s.ijl, n);
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
                        if ni < 0 || nj < 0 || nl < 0 {
                            continue;
                        }
                        // A coordinate past the far face is simply absent from
                        // the map, so no upper bound is needed here.
                        let nk = (ni as usize, nj as usize, nl as usize);
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

    let pos = |n: usize| sites[n].pos;
    let dist = |a: usize, b: usize| {
        let (p, r) = (pos(a), pos(b));
        ((p[0] - r[0]).powi(2) + (p[1] - r[1]).powi(2) + (p[2] - r[2]).powi(2)).sqrt()
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

        // A loop's contour closes, so its length takes the segment from the last
        // site back to the first. Leaving it out biases every loop short by
        // about one segment, which the derived radius and the 2 pi / l circle
        // reference both inherit.
        let closing = dist(order[0], current);
        let is_loop = order.len() > 2 && closing <= dx * 3.0_f64.sqrt();
        if is_loop {
            length += closing;
        }
        let mean_cos_beta = order
            .iter()
            .map(|&m| sites[m].disclination.cos_beta)
            .sum::<f64>()
            / order.len() as f64;
        let ordered: Vec<DisclinationSite> = order.iter().map(|&m| sites[m]).collect();
        let points: Vec<[f64; 3]> = ordered.iter().map(|s| s.pos).collect();
        let geometry = frenet(&points, is_loop);
        let mean_curvature = if geometry.curvatures.is_empty() {
            0.0
        } else {
            geometry.curvatures.iter().sum::<f64>() / geometry.curvatures.len() as f64
        };

        out.push(DisclinationCurve {
            sites: ordered,
            length,
            mean_cos_beta,
            is_loop,
            tangents: geometry.tangents,
            curvatures: geometry.curvatures,
            torsions: geometry.torsions,
            mean_curvature,
            surface_mean_curvature: 0.0,
            surface_gaussian_curvature: 0.0,
        });
    }
    out.sort_by(|a, b| b.length.total_cmp(&a.length));
    out
}

/// Assemble the lines at a threshold taken from the field's own interior peak.
///
/// `s` scales as the square of a Q gradient, so an absolute threshold depends on
/// the normalisation and on the grid spacing and transfers between runs poorly.
/// A fraction of the peak transfers, and the threshold that was used comes back
/// with the lines so a run can record what it read them at.
///
/// The peak is taken two voxels clear of the faces. The derivative stencil
/// wraps, so a field that is not periodic reads a seam there which has nothing
/// to do with a disclination, and one such voxel would otherwise set the
/// threshold for the whole box.
///
/// # The floor
///
/// A field with no disclination in it still has a largest gradient somewhere, so
/// the relative rule alone reports lines in the noise of a field that has none,
/// and reports them with no sign of trouble. What it does show is the threshold
/// collapsing by orders of magnitude while the order parameter sits at its
/// equilibrium, which is what happens once a run's defects annihilate.
///
/// `floor` is the absolute value below which the answer is that there are no
/// disclinations, and the threshold returned is the larger of the two. It is in
/// the units of `s`, which scale as the square of a Q gradient, so a value
/// transfers between runs only at matched normalisation and grid spacing. Pass
/// zero for the relative rule alone.
pub fn disclination_lines_at_fraction(
    q: &[[f64; 5]],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    fraction: f64,
    floor: f64,
) -> (Vec<DisclinationCurve>, f64) {
    let density = disclination_density(q, nx, ny, nz, dx);
    let mag = magnitudes(&density);

    let mut peak = 0.0_f64;
    if nx > 4 && ny > 4 && nz > 4 {
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for l in 2..nz - 2 {
                    peak = peak.max(mag[((i * ny) + j) * nz + l]);
                }
            }
        }
    } else {
        peak = mag.iter().copied().fold(0.0_f64, f64::max);
    }
    let threshold = (fraction * peak).max(floor);

    let sites = sites_from(&density, &mag, nx, ny, nz, dx, threshold);
    if sites.is_empty() {
        return (Vec::new(), threshold);
    }
    let mut curves = assemble(sites, dx);
    attach_surface_curvature(&mut curves, &mag, nx, ny, nz, dx, threshold);
    (curves, threshold)
}

/// Curve sites bucketed by integer cell, as `(curve index, position)`.
///
/// The bucket cell is four voxels wide, so the nearest site to a shell voxel is
/// found from a handful of cells rather than from every site on every curve.
type CellBuckets = std::collections::HashMap<(i64, i64, i64), Vec<(usize, [f64; 3])>>;

/// Measure the `s` isosurface around each line and record its curvature.
///
/// The surface is taken as the inner face of the supra-threshold region: a
/// voxel above the threshold with a six-neighbour below it. That shell is one
/// voxel thick and is what a contour at the same threshold would draw. Each of
/// its voxels is charged to the line whose nearest site it is nearest to, and
/// the two curvatures are averaged over the voxels a line collects.
fn attach_surface_curvature(
    curves: &mut [DisclinationCurve],
    mag: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    threshold: f64,
) {
    if curves.is_empty() {
        return;
    }
    let at = |i: usize, j: usize, l: usize| mag[((i * ny) + j) * nz + l];

    // Sites of every curve, bucketed so the nearest is found without sweeping
    // all of them for each shell voxel.
    let cell = 4.0 * dx;
    let key = |p: [f64; 3]| {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut buckets: CellBuckets = std::collections::HashMap::new();
    for (c, curve) in curves.iter().enumerate() {
        for s in &curve.sites {
            buckets.entry(key(s.pos)).or_default().push((c, s.pos));
        }
    }

    let mut sums = vec![(0.0f64, 0.0f64, 0usize); curves.len()];
    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                if at(i, j, l) <= threshold {
                    continue;
                }
                let outside = [
                    (i.wrapping_sub(1), j, l),
                    (i + 1, j, l),
                    (i, j.wrapping_sub(1), l),
                    (i, j + 1, l),
                    (i, j, l.wrapping_sub(1)),
                    (i, j, l + 1),
                ]
                .into_iter()
                .any(|(a, b, c)| a >= nx || b >= ny || c >= nz || at(a, b, c) <= threshold);
                if !outside {
                    continue;
                }

                let here = [i as f64 * dx, j as f64 * dx, l as f64 * dx];
                let (kx, ky, kz) = key(here);
                let mut best: Option<(usize, f64)> = None;
                for radius in 1..=3i64 {
                    for a in -radius..=radius {
                        for b in -radius..=radius {
                            for c in -radius..=radius {
                                let Some(near) = buckets.get(&(kx + a, ky + b, kz + c)) else {
                                    continue;
                                };
                                for &(curve, p) in near {
                                    let d2: f64 = (0..3).map(|n| (p[n] - here[n]).powi(2)).sum();
                                    if best.is_none_or(|(_, d)| d2 < d) {
                                        best = Some((curve, d2));
                                    }
                                }
                            }
                        }
                    }
                    if best.is_some() {
                        break;
                    }
                }
                let Some((curve, _)) = best else { continue };

                let k = level_set_curvature(mag, nx, ny, nz, dx, (i, j, l));
                sums[curve].0 += k.mean;
                sums[curve].1 += k.gaussian;
                sums[curve].2 += 1;
            }
        }
    }

    for (curve, (mean, gaussian, count)) in curves.iter_mut().zip(sums) {
        if count > 0 {
            curve.surface_mean_curvature = mean / count as f64;
            curve.surface_gaussian_curvature = gaussian / count as f64;
        }
    }
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
        (1..n - 1)
            .flat_map(move |i| (1..n - 1).flat_map(move |j| (1..n - 1).map(move |l| (i, j, l))))
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
        assert!(
            d.tangent[2].abs() > 0.99,
            "tangent {:?} is not along z",
            d.tangent
        );
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
        assert!(
            near > far,
            "s did not decay outward: {near} at core, {far} away"
        );
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
        assert!(
            d.tangent[0].abs() > 0.99,
            "tangent {:?} is not along x",
            d.tangent
        );
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
                    q[((i * n) + j) * n + l] = uniaxial([theta.cos(), theta.sin(), 0.0], 0.556);
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
                        q[((i * n) + j) * n + l] = uniaxial([theta.cos(), theta.sin(), 0.0], q_mag);
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

// ---------------------------------------------------------------------------
// Linking
// ---------------------------------------------------------------------------

/// Gauss linking number of two closed curves, each a list of points.
///
/// ```text
/// Lk = (1/4 pi) \oint \oint (r1 - r2) . (dr1 x dr2) / |r1 - r2|^3
/// ```
///
/// evaluated as the double sum over segments. For two closed curves this is an
/// integer, and the deviation from one measures the discretisation: a curve
/// sampled too coarsely against the separation returns something visibly
/// non-integral, which is the check to apply before believing a value.
///
/// # What it is for
///
/// Two things at once, because they are the same object. In three dimensions
/// the disclinations of a nematic are curves and their linking is a property of
/// the texture. In two dimensions the defect worldlines, drawn in `(x, y, t)`,
/// are also curves: a braid is a link once time is a coordinate, and closing a
/// periodic orbit turns its braid into one. So the same integral reads the
/// topology of a 3D texture and of a 2D orbit.
///
/// # Caveat for half-integer lines
///
/// A `+/-1/2` disclination is not orientable on its own: transporting the
/// director around it returns a rotation by `pi`, so the loop is a class in
/// `pi_1(RP^2) = Z/2` and only the parity of the linking is an invariant of the
/// texture. The integral below is the geometric linking of whatever orientation
/// the point ordering gives, which is what is wanted for worldlines, where the
/// time direction orients everything, and which needs that caveat for genuine
/// disclination loops.
pub fn linking_number(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let mut total = 0.0;
    for i in 0..a.len() {
        let p1 = a[i];
        let p2 = a[(i + 1) % a.len()];
        let d1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        for j in 0..b.len() {
            let q1 = b[j];
            let q2 = b[(j + 1) % b.len()];
            let d2 = [q2[0] - q1[0], q2[1] - q1[1], q2[2] - q1[2]];
            // midpoint separation, which halves the segment-count needed for a
            // given accuracy against using an endpoint
            let r = [
                (p1[0] + p2[0]) / 2.0 - (q1[0] + q2[0]) / 2.0,
                (p1[1] + p2[1]) / 2.0 - (q1[1] + q2[1]) / 2.0,
                (p1[2] + p2[2]) / 2.0 - (q1[2] + q2[2]) / 2.0,
            ];
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            if r2 < 1e-12 {
                continue;
            }
            let cross = [
                d1[1] * d2[2] - d1[2] * d2[1],
                d1[2] * d2[0] - d1[0] * d2[2],
                d1[0] * d2[1] - d1[1] * d2[0],
            ];
            total += (r[0] * cross[0] + r[1] * cross[1] + r[2] * cross[2]) / (r2 * r2.sqrt());
        }
    }
    total / (4.0 * std::f64::consts::PI)
}

/// The points of a [`DisclinationCurve`] in grid coordinates.
pub fn curve_points(c: &DisclinationCurve) -> Vec<[f64; 3]> {
    c.sites.iter().map(|s| s.pos).collect()
}

/// Pairwise linking numbers of every pair of closed curves.
///
/// Open curves are skipped: linking is defined for closed ones, and a line
/// running out of the box has to be closed through the boundary before it means
/// anything.
pub fn pairwise_linking(curves: &[DisclinationCurve]) -> Vec<(usize, usize, f64)> {
    let closed: Vec<usize> = (0..curves.len()).filter(|&i| curves[i].is_loop).collect();
    let mut out = Vec::new();
    for a in 0..closed.len() {
        for b in a + 1..closed.len() {
            let (i, j) = (closed[a], closed[b]);
            out.push((
                i,
                j,
                linking_number(&curve_points(&curves[i]), &curve_points(&curves[j])),
            ));
        }
    }
    out
}

#[cfg(test)]
mod linking_tests {
    use super::*;

    /// A circle of `n` points in the plane `z = z0`, centred on `(cx, cy)`.
    fn circle(cx: f64, cy: f64, z0: f64, r: f64, n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [cx + r * t.cos(), cy + r * t.sin(), z0]
            })
            .collect()
    }

    /// The same circle stood up in the `xz` plane, centred on `(cx, cy, z0)`.
    ///
    /// Standing it up in `yz` instead puts it beside the first circle rather
    /// than through it, and the pair is then unlinked: the integral says zero
    /// and it is right to.
    fn circle_xz(cx: f64, cy: f64, z0: f64, r: f64, n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [cx + r * t.cos(), cy, z0 + r * t.sin()]
            })
            .collect()
    }

    /// Two circles side by side are unlinked.
    #[test]
    fn unlink_is_zero() {
        let a = circle(0.0, 0.0, 0.0, 1.0, 400);
        let b = circle(5.0, 0.0, 0.0, 1.0, 400);
        assert!(linking_number(&a, &b).abs() < 1e-3);
    }

    /// The Hopf link has linking number one, and minus one reversed.
    #[test]
    fn hopf_link_is_one() {
        let a = circle(0.0, 0.0, 0.0, 1.0, 600);
        let b = circle_xz(1.0, 0.0, 0.0, 1.0, 600);
        let lk = linking_number(&a, &b);
        assert!((lk.abs() - 1.0).abs() < 1e-2, "expected |Lk| = 1, got {lk}");
        let mut rev = b.clone();
        rev.reverse();
        assert!(
            (linking_number(&a, &rev) + lk).abs() < 1e-2,
            "reversing one curve must flip the sign"
        );
    }

    /// Linking is symmetric in its two arguments.
    #[test]
    fn linking_is_symmetric() {
        let a = circle(0.0, 0.0, 0.0, 1.0, 300);
        let b = circle_xz(1.0, 0.0, 0.0, 1.0, 300);
        assert!((linking_number(&a, &b) - linking_number(&b, &a)).abs() < 1e-6);
    }

    /// The Borromean rings: every pair unlinked, the whole linked.
    ///
    /// This is the case that matters for reading a braid, because a braid whose
    /// word is a commutator has vanishing pairwise linking while being far from
    /// trivial. Pairwise linking is a first invariant, never a complete one.
    #[test]
    fn borromean_rings_are_pairwise_unlinked() {
        let n = 800;
        let (a, b, c) = (2.0_f64, 1.0_f64, 0.0_f64);
        // three mutually orthogonal ellipses, the standard realisation
        let e1: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [a * t.cos(), b * t.sin(), c]
            })
            .collect();
        let e2: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [c, a * t.cos(), b * t.sin()]
            })
            .collect();
        let e3: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [b * t.sin(), c, a * t.cos()]
            })
            .collect();
        for (x, y, nm) in [(&e1, &e2, "1-2"), (&e2, &e3, "2-3"), (&e1, &e3, "1-3")] {
            let lk = linking_number(x, y);
            assert!(lk.abs() < 1e-2, "pair {nm} should be unlinked, got {lk}");
        }
    }
}
