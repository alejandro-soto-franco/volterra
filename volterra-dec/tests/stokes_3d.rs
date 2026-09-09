//! The validation ladder for the bounded three-dimensional Stokes solver.
//!
//! Each rung is a separate test and each was shown to fail against a
//! deliberately wrong implementation before it was kept. The measured failures
//! are recorded in the doc comment of the test that catches them.

use volterra_dec::mimetic::assemble_star;
use volterra_dec::stokes_3d::BoundedStokes3D;
use volterra_dec::tet_mesh::{box_mesh, TetComplex};

// ---------------------------------------------------------------------------
// The manufactured solution.
//
// phi(s) = s^2 (1 - s)^2 vanishes with its first derivative at both ends, so
// with A = (phi(x) phi(y) phi(z), 0, 0) the field v = curl A is
// divergence-free and vanishes identically on the boundary of the unit cube,
// tangential part included. The pressure is linear, which exercises the
// pressure block without adding a compatibility condition.
// ---------------------------------------------------------------------------

fn phi(s: f64) -> f64 {
    s * s * (1.0 - s) * (1.0 - s)
}
fn dphi(s: f64) -> f64 {
    2.0 * s - 6.0 * s * s + 4.0 * s * s * s
}
fn ddphi(s: f64) -> f64 {
    2.0 - 12.0 * s + 12.0 * s * s
}
fn dddphi(s: f64) -> f64 {
    -12.0 + 24.0 * s
}

fn velocity(x: [f64; 3]) -> [f64; 3] {
    let (a, b, c) = (x[0], x[1], x[2]);
    [
        0.0,
        phi(a) * phi(b) * dphi(c),
        -phi(a) * dphi(b) * phi(c),
    ]
}

fn laplacian_velocity(x: [f64; 3]) -> [f64; 3] {
    let (a, b, c) = (x[0], x[1], x[2]);
    let vy = ddphi(a) * phi(b) * dphi(c) + phi(a) * ddphi(b) * dphi(c) + phi(a) * phi(b) * dddphi(c);
    let vz = ddphi(a) * dphi(b) * phi(c) + phi(a) * dddphi(b) * phi(c) + phi(a) * dphi(b) * ddphi(c);
    [0.0, vy, -vz]
}

fn source(eta: f64) -> impl Fn([f64; 3]) -> [f64; 3] + Copy {
    move |x: [f64; 3]| {
        let l = laplacian_velocity(x);
        [-eta * l[0] + 1.0, -eta * l[1], -eta * l[2]]
    }
}

fn m2_norm(m2: &sprs::CsMat<f64>, x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (v, (r, c)) in m2.iter() {
        s += x[r] * v * x[c];
    }
    s.max(0.0).sqrt()
}

/// Solve the manufactured problem on an `n by n by n` box and return the
/// relative error of the velocity in the discrete L2 norm, together with the
/// divergence residual and the wall slip.
fn manufactured_error(n: usize, eta: f64) -> (f64, f64, f64) {
    let mesh = box_mesh(n, n, n, 1.0, 1.0, 1.0).unwrap();
    let m2 = assemble_star(&mesh, 2).unwrap();
    let exact = mesh.flux_dofs(velocity);
    let f = mesh.flux_dofs(source(eta));
    let wall = vec![0.0; mesh.n_faces()];

    let solver = BoundedStokes3D::new(mesh).unwrap();
    let flow = solver.solve(&f, &wall, eta).unwrap();

    let err: Vec<f64> = flow
        .flux
        .iter()
        .zip(&exact)
        .map(|(a, b)| a - b)
        .collect();
    let rel = m2_norm(&m2, &err) / m2_norm(&m2, &exact);
    let slip = flow.wall_slip(solver.mesh()).rms;
    (rel, flow.divergence_residual, slip)
}

/// Rung 3: the divergence residual sits at machine precision on every solve.
///
/// This is the property the masked spectral solver never had. It is
/// combinatorial rather than approximate, so the tolerance is absolute rather
/// than a rate. Measured across `n = 3` to `n = 10` the residual stays between
/// 1.3e-20 and 7.6e-20, so the bound below has four orders of room.
///
/// The residual has no power over the operator, only over the constraint: with
/// the viscous sign of the first draft it is unchanged to every printed digit,
/// because incompressibility is enforced combinatorially whatever the momentum
/// row says.
#[test]
fn the_divergence_residual_is_at_machine_precision() {
    let (_, div, _) = manufactured_error(4, 1.0);
    assert!(div < 1e-16, "cellwise net outflux reached {div}");
}

/// The force does positive work on the fluid.
///
/// At steady state `(f, v) = eta ||grad v||^2`, since the pressure pairs with a
/// divergence-free field of zero normal trace and drops out. That is a signed
/// statement, and it is the one thing in this ladder with power over the sign of
/// the first row.
///
/// The reason it is needed: with `+M1` in place of `-M1`, which is the viscous
/// sign of the first draft of the spec, the solution map takes the source to its
/// negative, so every magnitude in this file is unchanged to every printed
/// digit. The wall slip is a norm and the duct case has a vanishing momentum
/// source, so neither notices. Here the work changes sign, and the ratio below
/// goes from 0.98 to -0.98.
#[test]
fn the_force_does_positive_work() {
    let eta = 1.3;
    let n = 6;
    let mesh = box_mesh(n, n, n, 1.0, 1.0, 1.0).unwrap();
    let m2 = assemble_star(&mesh, 2).unwrap();
    let exact = mesh.flux_dofs(velocity);
    let f = mesh.flux_dofs(source(eta));
    let wall = vec![0.0; mesh.n_faces()];
    let solver = BoundedStokes3D::new(mesh).unwrap();
    let flow = solver.solve(&f, &wall, eta).unwrap();

    let work = pairing(&m2, &f, &flow.flux);
    let reference = pairing(&m2, &f, &exact);
    assert!(reference > 0.0, "the analytic dissipation must be positive");
    let ratio = work / reference;
    assert!(
        ratio > 0.5,
        "the force does work {work} against a dissipation of {reference}, ratio {ratio}"
    );
}

/// The `M2` pairing of two flux vectors, which is the discrete `(a, b)`.
fn pairing(m2: &sprs::CsMat<f64>, a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (v, (r, c)) in m2.iter() {
        s += a[r] * v * b[c];
    }
    s
}

/// Rung 4: the manufactured solution converges.
///
/// Measured relative errors at `n = 3, 4, 6, 8, 10` are 6.11e-1, 3.64e-1,
/// 1.78e-1, 1.08e-1 and 7.47e-2, whose pairwise rates run from 1.80 down to
/// 1.67, so the floor below sits well clear of the measurement and well clear of
/// the first order the lowest-order pair guarantees.
///
/// The falsification was run rather than assumed. With `+M1` in place of `-M1`,
/// the viscous sign of the first draft of the spec, the same errors are 2.49,
/// 2.27, 2.12, 2.07 and 2.04: the solution map takes the source to its negative,
/// so the error settles on two rather than falling, and the rate collapses to
/// 0.05. Both assertions below catch it.
#[test]
fn the_manufactured_solution_converges() {
    let eta = 1.3;
    let mut errs = Vec::new();
    for n in [4usize, 6, 8] {
        let (rel, div, _) = manufactured_error(n, eta);
        assert!(div < 1e-11, "n = {n}: divergence residual {div}");
        assert!(rel < 1.0, "n = {n}: relative error {rel} is not a solution at all");
        errs.push(rel);
    }
    assert!(errs[1] < errs[0], "error grew from n = 4 to n = 6: {errs:?}");
    assert!(errs[2] < errs[1], "error grew from n = 6 to n = 8: {errs:?}");
    let rate = (errs[0] / errs[2]).ln() / (8.0_f64 / 4.0).ln();
    assert!(
        rate > 0.8,
        "convergence rate {rate} over errors {errs:?} is below the first order the \
         lowest-order pair should give"
    );
}

/// Rung 6: the tangential velocity on the wall falls with mesh size.
///
/// No-slip is imposed weakly, so the reconstructed wall tangential velocity is a
/// computed quantity rather than an assumption. Under the derivation it goes to
/// zero; under free slip, which is what the first draft of the spec expected
/// this formulation to give, it would settle on a constant.
///
/// The measured root mean square runs 2.62e-4, 2.07e-4, 1.43e-4, 1.09e-4 and
/// 8.70e-5 at `n = 3, 4, 6, 8, 10`, a rate of 0.92 in the mesh size, and its
/// ratio to the L2 velocity scale falls 0.95, 0.72, 0.48, 0.36, 0.29. The rate
/// is the assertion, because it is what separates a slip going to zero from one
/// settling on a constant; the ratio at any single refinement does not.
///
/// This measurement is a magnitude, so it has no power over the sign of the
/// first row: the wrong sign reproduces every number here exactly.
#[test]
fn the_wall_slip_falls_with_mesh_size() {
    let eta = 1.0;
    let ns = [4usize, 6, 10];
    let slips: Vec<f64> = ns.iter().map(|&n| manufactured_error(n, eta).2).collect();
    assert!(slips[2] < slips[1] && slips[1] < slips[0], "wall slip did not fall: {slips:?}");
    let rate = (slips[0] / slips[2]).ln() / (10.0_f64 / 4.0).ln();
    assert!(
        rate > 0.6,
        "wall slip decays at rate {rate} over {slips:?}, which reads as a slip \
         settling on a constant rather than one going to zero"
    );
}

// ---------------------------------------------------------------------------
// Rung 5: Poiseuille flow in a rectangular duct.
// ---------------------------------------------------------------------------

/// The series solution for pressure-driven flow along `z` in the duct
/// `0 < x < a`, `0 < y < b`, with `-eta lap w = g`.
///
/// Separating in `x` gives `W_n'' - k_n^2 W_n = -g_n / eta` with
/// `k_n = n pi / a` and `g_n = 4 g / (n pi)` for odd `n`, whose solution
/// vanishing at both walls is the bracket below.
fn duct_profile(x: f64, y: f64, a: f64, b: f64, g: f64, eta: f64) -> f64 {
    let mut acc = 0.0;
    let mut n = 1usize;
    while n <= 81 {
        let k = n as f64 * std::f64::consts::PI / a;
        let amp = 4.0 * g * a * a
            / (eta * (n as f64).powi(3) * std::f64::consts::PI.powi(3));
        let shape = 1.0 - (k * (y - 0.5 * b)).cosh() / (0.5 * k * b).cosh();
        acc += amp * shape * (k * x).sin();
        n += 2;
    }
    acc
}

/// Rung 5: the duct case is an exact solution of the bounded problem.
///
/// With the analytic profile prescribed as the normal flux at the inlet and the
/// outlet, no-slip on the four side walls and no body force, the Poiseuille
/// field with its linear pressure drop satisfies the equations and every
/// boundary condition this formulation imposes, so any discrepancy is
/// discretisation error alone.
///
/// The case separates no-slip from free slip on the shape of its profile. Free
/// slip on the side walls answers a uniform plug of the same flow rate, whose
/// measured distance from the series profile in this same norm is 0.46 at
/// `n = 4` and 0.50 at `n = 8`, an order above the tolerance asserted here.
///
/// The case has a vanishing momentum source, so it is blind to the sign of the
/// first row: the wrong sign leaves the solution map acting on a zero and
/// reproduces every number here exactly. Its power is over the boundary
/// condition, which is what it is for.
#[test]
fn the_duct_reproduces_the_series_solution() {
    let (a, b, len) = (1.0, 0.6, 0.5);
    let (g, eta) = (1.0, 1.0);
    let profile = move |x: [f64; 3]| [0.0, 0.0, duct_profile(x[0], x[1], a, b, g, eta)];

    let mut errs = Vec::new();
    let mut grads = Vec::new();
    for n in [4usize, 8] {
        let mesh = box_mesh(n, n, n, a, b, len).unwrap();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let exact = mesh.flux_dofs(profile);
        let zero = vec![0.0; mesh.n_faces()];

        // The port faces are the two the extrusion put at z = 0 and z = len.
        let mut wall = vec![0.0; mesh.n_faces()];
        for f in 0..mesh.n_faces() {
            if !mesh.is_boundary_face(f) {
                continue;
            }
            let c = mesh.face_centroid(f);
            if c[2] < 1e-12 || c[2] > len - 1e-12 {
                wall[f] = exact[f];
            }
        }

        let solver = BoundedStokes3D::new(mesh).unwrap();
        let flow = solver.solve(&zero, &wall, eta).unwrap();
        assert!(
            flow.divergence_residual < 1e-11,
            "n = {n}: divergence residual {}",
            flow.divergence_residual
        );
        let err: Vec<f64> = flow.flux.iter().zip(&exact).map(|(p, q)| p - q).collect();
        errs.push(m2_norm(&m2, &err) / m2_norm(&m2, &exact));

        // The layer-mean pressure against the layer height. Regressing cell by
        // cell mixes in the transverse variation, which the layer mean removes.
        let mesh = solver.mesh();
        let dz = len / n as f64;
        let mut num = vec![0.0; n];
        let mut den = vec![0.0; n];
        for t in 0..mesh.n_tets() {
            let l = ((mesh.tet_centroid(t)[2] / dz).floor() as usize).min(n - 1);
            num[l] += flow.pressure[t];
            den[l] += mesh.tet_volume(t);
        }
        let (mut sz, mut szz, mut sp, mut spz) = (0.0, 0.0, 0.0, 0.0);
        for l in 0..n {
            let z = (l as f64 + 0.5) * dz;
            let pv = num[l] / den[l];
            sz += z;
            szz += z * z;
            sp += pv;
            spz += pv * z;
        }
        let nn = n as f64;
        grads.push(-(spz - sz * sp / nn) / (szz - sz * sz / nn) / g);
    }
    assert!(errs[1] < errs[0], "duct error grew under refinement: {errs:?}");
    assert!(
        grads[1] > grads[0] && grads[1] > 0.8,
        "the pressure gradient over the driving gradient is {grads:?}, which should \
         approach one from below"
    );
    assert!(
        errs[1] < 0.1,
        "duct relative error {} is far enough from the series solution to read as \
         a different boundary condition: {errs:?}",
        errs[1]
    );
}

// ---------------------------------------------------------------------------
// Reconstruction.
// ---------------------------------------------------------------------------

/// A constant field survives the projection onto face fluxes and the
/// Raviart-Thomas reconstruction exactly.
///
/// Both maps are lossy in the way any projection is, so the constant field is
/// the one case where the round trip must be exact, and it is what pins the
/// sign convention relating a canonically oriented flux to an outward one.
#[test]
fn a_constant_field_survives_the_round_trip() {
    let mesh = box_mesh(3, 2, 2, 1.0, 0.7, 0.4).unwrap();
    let v0 = [0.37, -0.81, 0.44];
    let flux = mesh.flux_dofs(|_| v0);
    let flow = volterra_dec::stokes_3d::Flow3D {
        vorticity: vec![0.0; mesh.n_edges()],
        flux,
        pressure: vec![0.0; mesh.n_tets()],
        divergence_residual: 0.0,
    };
    for t in 0..mesh.n_tets() {
        let x = mesh.tet_centroid(t);
        let v = flow.velocity_in_cell(&mesh, t, x);
        for i in 0..3 {
            assert!(
                (v[i] - v0[i]).abs() < 1e-12,
                "cell {t}, component {i}: {} against {}",
                v[i],
                v0[i]
            );
        }
    }
    for (v, _) in flow.velocity_at_vertices(&mesh).iter().zip(0..mesh.n_vertices()) {
        for i in 0..3 {
            assert!((v[i] - v0[i]).abs() < 1e-12);
        }
    }
}

/// A boundary condition whose fluxes do not sum to zero has no incompressible
/// solution, and the solver says so rather than returning a least-squares
/// answer.
#[test]
fn an_incompatible_boundary_flux_is_rejected() {
    let mesh: TetComplex = box_mesh(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
    let mut wall = vec![0.0; mesh.n_faces()];
    let f = (0..mesh.n_faces()).find(|&f| mesh.is_boundary_face(f)).unwrap();
    wall[f] = 1.0;
    let zero = vec![0.0; mesh.n_faces()];
    let solver = BoundedStokes3D::new(mesh).unwrap();
    let err = solver.solve(&zero, &wall, 1.0).unwrap_err();
    assert!(matches!(
        err,
        volterra_dec::stokes_3d::Stokes3DError::IncompatibleBoundaryFlux { .. }
    ));
}

/// A chamber with a pillar leaves no harmonic mode, so the pressure pinning is
/// the whole of the solver's null space.
///
/// The homogeneous reduced system forces `w = 0` through `w^T M1 w = 0`, and
/// what is left is `d2 u = 0` together with `d1^T M2 u = 0` tested over every
/// edge, the boundary ones included. The second condition states both `div` of
/// the dual and the vanishing of the tangential trace, so the space is the
/// harmonic fields with full Dirichlet data, which is trivial on any domain
/// whatever its Betti numbers are. The first draft of the spec listed a pillar
/// as an open gap on the strength of the first Betti number alone.
///
/// Measured on a slab with a pillar through it, whose Euler characteristic is
/// zero: the nullity is zero at every refinement, and the smallest retained
/// singular value is 2.5e-2 to 8.0e-2 of the largest, so the rank is not a
/// question of tolerance.
#[test]
fn a_chamber_with_a_pillar_leaves_no_harmonic_mode() {
    use nalgebra::DMatrix;
    use volterra_dec::tet_mesh::pillar_mesh;

    let mesh = pillar_mesh(8, 2, 1, 0.3, 1.0, 0.5).unwrap();
    let chi = mesh.n_vertices() as i64 - mesh.n_edges() as i64 + mesh.n_faces() as i64
        - mesh.n_tets() as i64;
    assert_eq!(chi, 0, "the pillar chamber should be a solid torus");

    let d1 = mesh.d1();
    let d2 = mesh.d2();
    let m2 = assemble_star(&mesh, 2).unwrap();
    let free: Vec<usize> = (0..mesh.n_faces())
        .filter(|&f| !mesh.is_boundary_face(f))
        .collect();
    let mut slot = vec![usize::MAX; mesh.n_faces()];
    for (i, &f) in free.iter().enumerate() {
        slot[f] = i;
    }
    let mut a = DMatrix::<f64>::zeros(mesh.n_tets() + mesh.n_edges(), free.len());
    for (v, (t, f)) in d2.iter() {
        if slot[f] != usize::MAX {
            a[(t, slot[f])] += v;
        }
    }
    for (v, (g, f)) in m2.iter() {
        if slot[f] == usize::MAX {
            continue;
        }
        if let Some(row) = d1.outer_view(g) {
            for (e, &s) in row.iter() {
                a[(mesh.n_tets() + e, slot[f])] += s * v;
            }
        }
    }
    let mut sv: Vec<f64> = a.singular_values().iter().cloned().collect();
    sv.sort_by(|p, q| q.partial_cmp(p).unwrap());
    let top = sv[0];
    let rank = sv.iter().filter(|&&x| x > 1e-10 * top * a.nrows() as f64).count();
    assert_eq!(free.len() - rank, 0, "harmonic dimension is not zero");
    assert!(
        sv[rank - 1] / top > 1e-3,
        "the smallest retained singular value is {} of the largest, so the rank is a \
         question of tolerance rather than a measurement",
        sv[rank - 1] / top
    );

    // The solver itself factorises and solves on the same chamber.
    let nf = mesh.n_faces();
    let f = mesh.flux_dofs(|x: [f64; 3]| [0.3 * x[2], -0.4, 0.2 * x[0]]);
    let solver = BoundedStokes3D::new(mesh).unwrap();
    let flow = solver.solve(&f, &vec![0.0; nf], 1.0).unwrap();
    assert!(flow.divergence_residual < 1e-16, "divergence {}", flow.divergence_residual);
}
