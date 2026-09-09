//! The domain a 3D field lives on, so the disclination machinery stops assuming
//! a lattice.
//!
//! [`disclination_density`](crate::disclination::disclination_density) and
//! everything after it work on a uniform Cartesian grid: three index strides, a
//! central difference and 26-connectivity. None of that is intrinsic to the
//! measurement. What the detector actually needs is a way to differentiate a Q
//! field, a way to differentiate a scalar, a way to sample one between sites,
//! and a notion of which sites adjoin which.
//!
//! [`Domain3`] names exactly those four, so the same detector reads a lattice, a
//! tetrahedral mesh of a confined region, or a curved 3-manifold whose
//! derivatives are covariant. [`CartesianDomain`] is the first implementation
//! and reproduces the existing stencils exactly, which is what
//! `tests/domain.rs` asserts.
//!
//! # The metric
//!
//! [`Domain3::q_gradients`] returns derivatives along three orthonormal
//! directions, so a curved implementation resolves its own frame and connection
//! and hands back components a flat consumer can use. That is what keeps the
//! disclination density tensor, which contracts two Levi-Civita symbols against
//! a pair of Q gradients, correct on a curved domain without the caller
//! knowing.

use nalgebra::{Matrix3, Vector3};

/// A domain with a Q-tensor field at a finite set of sites.
pub trait Domain3 {
    /// How many sites the field has.
    fn len(&self) -> usize;

    /// Whether the domain has no sites at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The position of a site, in the domain's length units.
    fn position(&self, site: usize) -> [f64; 3];

    /// The derivative of the Q field along each of three orthonormal directions.
    ///
    /// Returned as full symmetric traceless tensors rather than the five stored
    /// components, since every consumer embeds them anyway.
    fn q_gradients(&self, q: &[[f64; 5]], site: usize) -> [Matrix3<f64>; 3];

    /// The gradient and Hessian of a scalar field at a site.
    ///
    /// The pair is what an implicit surface's curvature needs, and computing
    /// them together lets an implementation share a stencil or a local fit.
    fn scalar_derivatives(&self, f: &[f64], site: usize) -> (Vector3<f64>, Matrix3<f64>);

    /// A scalar field sampled at an arbitrary position.
    ///
    /// Used to find a ridge between sites and to fit a core to sub-site
    /// accuracy, so it has to interpolate rather than round to the nearest site.
    fn sample(&self, f: &[f64], at: [f64; 3]) -> f64;

    /// The sites adjoining this one, appended to `out`, which is cleared first.
    ///
    /// Adjacency defines what "connected" means when sites are assembled into a
    /// line, so a lattice offers its 26 neighbours and a mesh its edge
    /// neighbours.
    fn neighbours(&self, site: usize, out: &mut Vec<usize>);
}

/// A uniform Cartesian lattice, which is what the detector assumed all along.
///
/// Derivatives are central differences and the stencil wraps, matching
/// `QField3D::laplacian` and the existing
/// [`disclination_density`](crate::disclination::disclination_density) exactly.
/// Scalar derivatives clamp at the faces instead, since a level set reaching a
/// wall is a real feature of a confined run and wrapping would join it to
/// whatever sits opposite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianDomain {
    /// Sites along x, y and z.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Uniform spacing.
    pub dx: f64,
}

impl CartesianDomain {
    /// A lattice of the given extent and spacing.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { nx, ny, nz, dx }
    }

    /// Flat index of a lattice position, wrapping.
    #[inline]
    pub fn index(&self, i: usize, j: usize, l: usize) -> usize {
        ((i % self.nx) * self.ny + (j % self.ny)) * self.nz + (l % self.nz)
    }

    /// The lattice position of a flat index.
    #[inline]
    pub fn coordinates(&self, site: usize) -> (usize, usize, usize) {
        let l = site % self.nz;
        let j = (site / self.nz) % self.ny;
        let i = site / (self.nz * self.ny);
        (i, j, l)
    }
}

/// Embed the five stored components as the full symmetric traceless tensor.
#[inline]
fn embed(q: [f64; 5]) -> Matrix3<f64> {
    let [q11, q12, q13, q22, q23] = q;
    Matrix3::new(q11, q12, q13, q12, q22, q23, q13, q23, -(q11 + q22))
}

impl Domain3 for CartesianDomain {
    fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn position(&self, site: usize) -> [f64; 3] {
        let (i, j, l) = self.coordinates(site);
        [i as f64 * self.dx, j as f64 * self.dx, l as f64 * self.dx]
    }

    fn q_gradients(&self, q: &[[f64; 5]], site: usize) -> [Matrix3<f64>; 3] {
        let (i, j, l) = self.coordinates(site);
        let inv = 1.0 / (2.0 * self.dx);
        let pairs = [
            (
                self.index((i + 1) % self.nx, j, l),
                self.index((i + self.nx - 1) % self.nx, j, l),
            ),
            (
                self.index(i, (j + 1) % self.ny, l),
                self.index(i, (j + self.ny - 1) % self.ny, l),
            ),
            (
                self.index(i, j, (l + 1) % self.nz),
                self.index(i, j, (l + self.nz - 1) % self.nz),
            ),
        ];
        std::array::from_fn(|d| {
            let (p, m) = pairs[d];
            (embed(q[p]) - embed(q[m])) * inv
        })
    }

    fn scalar_derivatives(&self, f: &[f64], site: usize) -> (Vector3<f64>, Matrix3<f64>) {
        let (i, j, l) = self.coordinates(site);
        let n = [self.nx as isize, self.ny as isize, self.nz as isize];
        let at = |a: isize, b: isize, c: isize| -> f64 {
            let a = a.clamp(0, n[0] - 1) as usize;
            let b = b.clamp(0, n[1] - 1) as usize;
            let c = c.clamp(0, n[2] - 1) as usize;
            f[(a * self.ny + b) * self.nz + c]
        };
        let (i, j, l) = (i as isize, j as isize, l as isize);
        let step = [[1isize, 0, 0], [0, 1, 0], [0, 0, 1]];
        let centre = at(i, j, l);

        let mut g = Vector3::zeros();
        let mut h = Matrix3::zeros();
        for d in 0..3 {
            let s = step[d];
            let plus = at(i + s[0], j + s[1], l + s[2]);
            let minus = at(i - s[0], j - s[1], l - s[2]);
            g[d] = (plus - minus) / (2.0 * self.dx);
            h[(d, d)] = (plus - 2.0 * centre + minus) / (self.dx * self.dx);
        }
        for d in 0..3 {
            for e in d + 1..3 {
                let (u, v) = (step[d], step[e]);
                let off = |sd: isize, se: isize| {
                    at(
                        i + sd * u[0] + se * v[0],
                        j + sd * u[1] + se * v[1],
                        l + sd * u[2] + se * v[2],
                    )
                };
                let mixed =
                    (off(1, 1) - off(1, -1) - off(-1, 1) + off(-1, -1)) / (4.0 * self.dx * self.dx);
                h[(d, e)] = mixed;
                h[(e, d)] = mixed;
            }
        }
        (g, h)
    }

    fn sample(&self, f: &[f64], at: [f64; 3]) -> f64 {
        let n = [self.nx, self.ny, self.nz];
        let mut base = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for c in 0..3 {
            let hi = (n[c] - 1) as f64;
            let x = (at[c] / self.dx).clamp(0.0, hi);
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
                    let i = (base[0] + di).min(self.nx - 1);
                    let j = (base[1] + dj).min(self.ny - 1);
                    let l = (base[2] + dl).min(self.nz - 1);
                    acc += w * f[(i * self.ny + j) * self.nz + l];
                }
            }
        }
        acc
    }

    fn neighbours(&self, site: usize, out: &mut Vec<usize>) {
        out.clear();
        let (i, j, l) = self.coordinates(site);
        let (i, j, l) = (i as i64, j as i64, l as i64);
        for di in -1i64..=1 {
            for dj in -1i64..=1 {
                for dl in -1i64..=1 {
                    if di == 0 && dj == 0 && dl == 0 {
                        continue;
                    }
                    let (a, b, c) = (i + di, j + dj, l + dl);
                    if a < 0
                        || b < 0
                        || c < 0
                        || a >= self.nx as i64
                        || b >= self.ny as i64
                        || c >= self.nz as i64
                    {
                        continue;
                    }
                    out.push(self.index(a as usize, b as usize, c as usize));
                }
            }
        }
    }
}

/// The disclination density tensor at every site of any domain.
///
/// The domain-generic counterpart of
/// [`disclination_density`](crate::disclination::disclination_density), which it
/// reproduces exactly on a [`CartesianDomain`]. The contraction is the same
/// nine cross products per site; only the derivatives come from elsewhere.
pub fn disclination_density_on<D: Domain3 + ?Sized>(
    domain: &D,
    q: &[[f64; 5]],
) -> Vec<[f64; 9]> {
    assert_eq!(q.len(), domain.len(), "the field and the domain disagree in size");
    let mut out = vec![[0.0_f64; 9]; q.len()];
    for site in 0..domain.len() {
        let grad = domain.q_gradients(q, site);
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
        out[site] = d;
    }
    out
}
