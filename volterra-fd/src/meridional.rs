//! Axisymmetric Q-tensor relaxation in the meridional half-plane.
//!
//! The dry passive model `dQ/dt = Gamma (K lap Q - a Q - 2c Tr(Q^2) Q - 3b
//! (Q^2)_0)` uses a Laplacian that acts on each Cartesian component separately
//! and a bulk term that acts pointwise. Both commute with rotations of space
//! that leave the components alone. A field whose Cartesian components depend
//! only on the cylindrical radius `rho` and the height `z` therefore stays so,
//! and the problem reduces exactly to the half-plane `rho >= 0`, with
//!
//! ```text
//! lap f = (1/rho) d_rho (rho d_rho f) + d_z^2 f.
//! ```
//!
//! Such fields include every texture whose director turns about a fixed axis
//! by an angle that depends on `(rho, z)` alone, among them the disclination
//! loops seeded by [`seed_solid_angle_ring`].
//!
//! The grid is cell centred: `rho_i = (i + 1/2) h` and `z_l = (l - nz/2 + 1/2)
//! h`. The Laplacian is the flux form, whose radial flux vanishes on the axis
//! (`rho_{-1/2} = 0`), and every outer face carries zero flux. It is the exact
//! variation of the discrete energy [`MeridionalQ::free_energy`], so an
//! explicit step below the stability limit never raises the energy.

use rayon::prelude::*;
use volterra_core::ActiveNematicParams3D;

use crate::mol_field_3d::cubic_bulk_term;

/// A Q field on the meridional half-plane, five components per cell,
/// `[q11, q12, q13, q22, q23]`, indexed `i * nz + l`.
#[derive(Debug, Clone)]
pub struct MeridionalQ {
    pub nr: usize,
    pub nz: usize,
    pub h: f64,
    pub q: Vec<[f64; 5]>,
}

/// `Tr(A B)` for two symmetric traceless matrices in the five-component layout.
#[inline]
pub fn tr_prod(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let a33 = -(a[0] + a[3]);
    let b33 = -(b[0] + b[3]);
    a[0] * b[0] + a[3] * b[3] + a33 * b33 + 2.0 * (a[1] * b[1] + a[2] * b[2] + a[4] * b[4])
}

/// The bulk molecular field `-a Q - 2c Tr(Q^2) Q - 3b (Q^2)_0`, with `a` the
/// effective Landau coefficient, as `mol_field_3d` computes it.
#[inline]
pub fn bulk_field(v: &[f64; 5], p: &ActiveNematicParams3D) -> [f64; 5] {
    let [q11, q12, q13, q22, q23] = *v;
    let q33 = -(q11 + q22);
    let t2 = tr_prod(v, v);
    let cubic = cubic_bulk_term(q11, q12, q13, q22, q23, q33, t2, p.b_landau);
    let lin = -p.a_eff() - 2.0 * p.c_landau * t2;
    std::array::from_fn(|m| lin * v[m] + cubic[m])
}

/// The bulk energy density `(a/2) Tr Q^2 + b Tr Q^3 + (c/2)(Tr Q^2)^2` whose
/// variation is [`bulk_field`].
#[inline]
pub fn bulk_energy(v: &[f64; 5], p: &ActiveNematicParams3D) -> f64 {
    let [q11, q12, q13, q22, q23] = *v;
    let q33 = -(q11 + q22);
    let m = [[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]];
    let mut t3 = 0.0;
    for (a, row) in m.iter().enumerate() {
        for (b, &mab) in row.iter().enumerate() {
            for (c, &mbc) in m[b].iter().enumerate() {
                t3 += mab * mbc * m[c][a];
            }
        }
    }
    let t2 = tr_prod(v, v);
    0.5 * p.a_eff() * t2 + p.b_landau * t3 + 0.5 * p.c_landau * t2 * t2
}

impl MeridionalQ {
    pub fn zeros(nr: usize, nz: usize, h: f64) -> Self {
        Self {
            nr,
            nz,
            h,
            q: vec![[0.0; 5]; nr * nz],
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, l: usize) -> usize {
        i * self.nz + l
    }

    #[inline]
    pub fn rho(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.h
    }

    #[inline]
    pub fn z(&self, l: usize) -> f64 {
        (l as f64 - self.nz as f64 / 2.0 + 0.5) * self.h
    }

    /// The axisymmetric Laplacian of every component, in flux form with zero
    /// flux through the axis and through every outer face.
    pub fn laplacian_into(&self, out: &mut [[f64; 5]]) {
        let (nr, nz, h) = (self.nr, self.nz, self.h);
        let q = &self.q;
        out.par_chunks_mut(nz).enumerate().for_each(|(i, row)| {
            let rho = (i as f64 + 0.5) * h;
            let w_out = if i + 1 < nr {
                (i as f64 + 1.0) * h / rho
            } else {
                0.0
            };
            let w_in = i as f64 * h / rho;
            for (l, slot) in row.iter_mut().enumerate() {
                let c = &q[i * nz + l];
                let mut acc = [0.0; 5];
                if i + 1 < nr {
                    let o = &q[(i + 1) * nz + l];
                    for m in 0..5 {
                        acc[m] += w_out * (o[m] - c[m]);
                    }
                }
                if i > 0 {
                    let o = &q[(i - 1) * nz + l];
                    for m in 0..5 {
                        acc[m] -= w_in * (c[m] - o[m]);
                    }
                }
                if l + 1 < nz {
                    let o = &q[i * nz + l + 1];
                    for m in 0..5 {
                        acc[m] += o[m] - c[m];
                    }
                }
                if l > 0 {
                    let o = &q[i * nz + l - 1];
                    for m in 0..5 {
                        acc[m] -= c[m] - o[m];
                    }
                }
                *slot = acc.map(|a| a / (h * h));
            }
        });
    }

    /// The rate `Gamma H` at every cell, with `H = K lap Q + bulk_field`.
    pub fn rate_into(&self, p: &ActiveNematicParams3D, out: &mut [[f64; 5]]) {
        self.laplacian_into(out);
        out.par_iter_mut()
            .zip(self.q.par_iter())
            .for_each(|(r, v)| {
                let b = bulk_field(v, p);
                for m in 0..5 {
                    r[m] = p.gamma_r * (p.k_r * r[m] + b[m]);
                }
            });
    }

    /// One explicit Euler step of length `p.dt`, using `scratch` for the rate.
    pub fn step(&mut self, p: &ActiveNematicParams3D, scratch: &mut [[f64; 5]]) {
        self.rate_into(p, scratch);
        let dt = p.dt;
        self.q
            .par_iter_mut()
            .zip(scratch.par_iter())
            .for_each(|(v, r)| {
                for m in 0..5 {
                    v[m] += dt * r[m];
                }
            });
    }

    /// The discrete free energy above the uniform state of density `f_eq`,
    /// `sum_faces K pi rho_face |dQ|^2 + sum_cells 2 pi rho h^2 (f_b - f_eq)`.
    /// Its gradient with respect to the cell values, divided by the cell
    /// volume `2 pi rho h^2`, is `-K lap Q - bulk_field`.
    pub fn free_energy(&self, p: &ActiveNematicParams3D, f_eq: f64) -> f64 {
        let (nr, nz, h) = (self.nr, self.nz, self.h);
        let pi = std::f64::consts::PI;
        (0..nr)
            .into_par_iter()
            .map(|i| {
                let rho = (i as f64 + 0.5) * h;
                let mut e = 0.0;
                for l in 0..nz {
                    let c = &self.q[i * nz + l];
                    e += 2.0 * pi * rho * h * h * (bulk_energy(c, p) - f_eq);
                    if i + 1 < nr {
                        let o = &self.q[(i + 1) * nz + l];
                        let d: [f64; 5] = std::array::from_fn(|m| o[m] - c[m]);
                        e += p.k_r * pi * (i as f64 + 1.0) * h * tr_prod(&d, &d);
                    }
                    if l + 1 < nz {
                        let o = &self.q[i * nz + l + 1];
                        let d: [f64; 5] = std::array::from_fn(|m| o[m] - c[m]);
                        e += p.k_r * pi * rho * tr_prod(&d, &d);
                    }
                }
                e
            })
            .sum()
    }

    /// The disclination core: the cell of least `Tr Q^2` off the axis, refined
    /// by a parabola through its neighbours along `rho` and along `z`. Returns
    /// `(rho, z)`.
    pub fn core(&self) -> (f64, f64) {
        let (nr, nz) = (self.nr, self.nz);
        let s2 = |i: usize, l: usize| {
            let v = &self.q[i * nz + l];
            tr_prod(v, v)
        };
        let (mut best, mut bi, mut bl) = (f64::INFINITY, 1, 1);
        for i in 1..nr - 1 {
            for l in 1..nz - 1 {
                let v = s2(i, l);
                if v < best {
                    (best, bi, bl) = (v, i, l);
                }
            }
        }
        let vertex = |m: f64, c: f64, p: f64| {
            let den = m - 2.0 * c + p;
            if den.abs() < 1e-300 {
                0.0
            } else {
                0.5 * (m - p) / den
            }
        };
        let di = vertex(s2(bi - 1, bl), s2(bi, bl), s2(bi + 1, bl));
        let dl = vertex(s2(bi, bl - 1), s2(bi, bl), s2(bi, bl + 1));
        (
            (bi as f64 + 0.5 + di) * self.h,
            (bl as f64 - nz as f64 / 2.0 + 0.5 + dl) * self.h,
        )
    }
}

/// Solid angle subtended at `(rho, z)` by the disc of radius `r` in `z = 0`,
/// signed by `z`. A quadrature over directions from the foot of the point:
/// all directions when the foot lies inside the disc, the chord directions
/// otherwise.
pub fn disc_solid_angle(rho: f64, z: f64, r: f64, quad: usize) -> f64 {
    let az = z.abs();
    let s = z.signum();
    if rho < r {
        let h = 2.0 * std::f64::consts::PI / quad as f64;
        let mut acc = 0.0;
        for m in 0..quad {
            let phi = (m as f64 + 0.5) * h;
            let reach = -rho * phi.cos() + (r * r - rho * rho * phi.sin().powi(2)).sqrt();
            acc += 1.0 - az / (az * az + reach * reach).sqrt();
        }
        s * acc * h
    } else {
        let phi_max = (r / rho).asin();
        let h = 2.0 * phi_max / quad as f64;
        let mut acc = 0.0;
        for m in 0..quad {
            let phi = -phi_max + (m as f64 + 0.5) * h;
            let half = (r * r - rho * rho * phi.sin().powi(2)).max(0.0).sqrt();
            let near = rho * phi.cos() - half;
            let far = rho * phi.cos() + half;
            acc += az / (az * az + near * near).sqrt() - az / (az * az + far * far).sqrt();
        }
        s * acc * h
    }
}

/// `Q = q (nn - I/3)` in the five-component layout.
pub fn uniaxial(n: [f64; 3], q: f64) -> [f64; 5] {
    let t = 1.0 / 3.0;
    [
        q * (n[0] * n[0] - t),
        q * (n[0] * n[1]),
        q * (n[0] * n[2]),
        q * (n[1] * n[1] - t),
        q * (n[1] * n[2]),
    ]
}

/// A disclination loop of radius `r` in `z = 0`: the director `e_x` turned
/// about `e_y` by a quarter of the solid angle of the loop's disc, so that it
/// turns by `pi` round the core and is uniform far away. The order is
/// `q_eq tanh(d / core)` at distance `d` from the core line.
pub fn seed_solid_angle_ring(field: &mut MeridionalQ, r: f64, q_eq: f64, core: f64) {
    let (nz, h) = (field.nz, field.h);
    field.q.par_chunks_mut(nz).enumerate().for_each(|(i, row)| {
        let rho = (i as f64 + 0.5) * h;
        for (l, slot) in row.iter_mut().enumerate() {
            let z = (l as f64 - nz as f64 / 2.0 + 0.5) * h;
            let psi = 0.25 * disc_solid_angle(rho, z, r, 2048);
            let d = ((rho - r).powi(2) + z * z).sqrt();
            *slot = uniaxial([psi.cos(), 0.0, psi.sin()], q_eq * (d / core).tanh());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ActiveNematicParams3D {
        let mut p = ActiveNematicParams3D::default_test();
        p.zeta_eff = 0.0;
        p.noise_amp = 0.0;
        p.k_r = 4.0;
        p.b_landau = -1.5;
        p.dt = 0.02;
        p
    }

    #[test]
    fn laplacian_of_rho_squared_plus_z_squared_is_six_in_the_interior() {
        let (nr, nz, h) = (20, 24, 0.7);
        let mut f = MeridionalQ::zeros(nr, nz, h);
        for i in 0..nr {
            for l in 0..nz {
                let v = f.rho(i).powi(2) + f.z(l).powi(2);
                let k = f.idx(i, l);
                f.q[k] = [v, 0.0, 0.0, 0.0, 0.0];
            }
        }
        let mut out = vec![[0.0; 5]; nr * nz];
        f.laplacian_into(&mut out);
        // Every cell off the outer faces, the axis cell included.
        for i in 0..nr - 1 {
            for l in 1..nz - 1 {
                let got = out[f.idx(i, l)][0];
                assert!((got - 6.0).abs() < 1e-9, "cell ({i},{l}): {got}");
            }
        }
    }

    #[test]
    fn laplacian_is_the_variation_of_the_gradient_energy() {
        // E_grad = sum_faces K pi rho_face |dQ|^2; dE/dQ_c = -2 pi rho_c h^2 K lap Q_c.
        let mut p = params();
        p.a_landau = 0.0;
        p.b_landau = 0.0;
        p.c_landau = 0.0;
        let (nr, nz, h) = (7, 6, 0.9);
        let mut f = MeridionalQ::zeros(nr, nz, h);
        for (k, v) in f.q.iter_mut().enumerate() {
            let x = k as f64;
            *v = [
                (0.3 * x).sin(),
                (0.7 * x).cos(),
                0.1 * x,
                (0.2 * x).sin(),
                -0.05 * x,
            ];
        }
        let mut lap = vec![[0.0; 5]; nr * nz];
        f.laplacian_into(&mut lap);
        let (i, l) = (0, 3);
        let k = f.idx(i, l);
        let eps = 1e-6;
        for m in [0, 1, 4] {
            let mut up = f.clone();
            up.q[k][m] += eps;
            let mut dn = f.clone();
            dn.q[k][m] -= eps;
            let de = (up.free_energy(&p, 0.0) - dn.free_energy(&p, 0.0)) / (2.0 * eps);
            // The off-diagonal components appear twice in tr_prod.
            let weight = if m == 0 || m == 3 { 1.0 } else { 2.0 };
            let vol = 2.0 * std::f64::consts::PI * f.rho(i) * h * h;
            let expect = -weight * vol * p.k_r * lap[k][m]
                - if m == 0 {
                    vol * p.k_r * (lap[k][0] + lap[k][3])
                } else {
                    0.0
                };
            // For q11 the q33 = -(q11 + q22) dependence adds the trace term.
            assert!(
                (de - expect).abs() < 1e-5 * (1.0 + expect.abs()),
                "m = {m}: {de} vs {expect}"
            );
        }
    }

    #[test]
    fn relaxation_never_raises_the_energy() {
        let p = params();
        let q_eq = p.equilibrium_q();
        let f_eq = bulk_energy(&uniaxial([1.0, 0.0, 0.0], q_eq), &p);
        let mut f = MeridionalQ::zeros(40, 60, 1.0);
        seed_solid_angle_ring(&mut f, 12.0, q_eq, 2.0);
        let mut scratch = vec![[0.0; 5]; f.q.len()];
        let mut e = f.free_energy(&p, f_eq);
        for _ in 0..200 {
            f.step(&p, &mut scratch);
            let next = f.free_energy(&p, f_eq);
            assert!(next <= e + 1e-9 * e.abs(), "energy rose: {e} -> {next}");
            e = next;
        }
    }

    #[test]
    fn solid_angle_jumps_by_four_pi_across_the_disc_and_decays_outside() {
        let r = 5.0;
        let up = disc_solid_angle(2.0, 1e-6, r, 4096);
        let dn = disc_solid_angle(2.0, -1e-6, r, 4096);
        assert!((up - dn - 4.0 * std::f64::consts::PI).abs() < 1e-3);
        // On the axis: 2 pi (1 - z / sqrt(z^2 + r^2)).
        let z = 3.0;
        let axis = 2.0 * std::f64::consts::PI * (1.0 - z / (z * z + r * r).sqrt());
        assert!((disc_solid_angle(1e-9, z, r, 4096) - axis).abs() < 1e-6);
        assert!(disc_solid_angle(50.0, 10.0, r, 4096).abs() < 0.01);
    }
}
