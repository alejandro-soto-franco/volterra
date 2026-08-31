//! Pressure and vorticity recovered from a stream-function Stokes solve.
//!
//! Steady Stokes is solved for `psi` alone, so the pressure is eliminated and
//! the velocity is a discrete curl. Two quantities have to be recovered
//! afterwards, and each has a sign and a scaling that nothing else pins:
//!
//!   * `p` from `Delta p = div f` with `dp/dn = f.n`. A force that IS a
//!     gradient has a known potential, so the solve has to return it.
//!   * `omega = Delta psi`. Differencing the recovered `u` instead chains two
//!     vertex-gradient operators and converges at `O(h^0.4)`, so the two must
//!     agree without being the same computation.

use volterra_dec::confined::{Epitrochoid, MeshOpts, confined_mesh};
use volterra_dec::poisson::PoissonSolver;
use volterra_dec::stokes::{extract_coords, pressure_rhs_from_force, vorticity_from_psi};
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;

fn nephroid() -> volterra_dec::confined::ConfinedMesh2 {
    confined_mesh(
        Epitrochoid { q: 2.0, d: 0.72, r: 53.071676 },
        MeshOpts { h_bulk: 2.0, h_min: 2.0, ..Default::default() },
    )
}

#[test]
fn the_pressure_solve_returns_the_potential_of_a_gradient_force() {
    let cm = nephroid();
    let mesh = &cm.mesh;
    let ops = Operators::from_mesh(mesh, &Euclidean::<2>);
    let nv = mesh.n_vertices();
    let coords = extract_coords(mesh);
    let poisson = PoissonSolver::new(&ops).expect("closed Poisson");
    let area = poisson.mass_diagonal().to_vec();
    let total: f64 = area.iter().sum();

    let k = 2.0 * std::f64::consts::PI / 53.071676;
    let phi: Vec<f64> =
        (0..nv).map(|i| (k * coords[i][0]).cos() * (k * coords[i][1]).cos()).collect();
    let f: Vec<[f64; 3]> = (0..nv)
        .map(|i| {
            let (x, y) = (coords[i][0], coords[i][1]);
            [-k * (k * x).sin() * (k * y).cos(), -k * (k * x).cos() * (k * y).sin(), 0.0]
        })
        .collect();

    let sol = poisson.solve(&pressure_rhs_from_force(&f, mesh, &coords, &area));
    let mp: f64 = (0..nv).map(|i| area[i] * sol[i]).sum::<f64>() / total;
    let me: f64 = (0..nv).map(|i| area[i] * phi[i]).sum::<f64>() / total;
    let num: f64 = (0..nv).map(|i| area[i] * (sol[i] - mp - (phi[i] - me)).powi(2)).sum();
    let den: f64 = (0..nv).map(|i| area[i] * (phi[i] - me).powi(2)).sum();
    let err = (num / den).sqrt();
    // A sign error returns the potential negated, a relative error of 2, so
    // this separates the two by two orders of magnitude.
    assert!(err < 5e-2, "pressure recovery is off by {err:.3e}");
}

#[test]
fn the_vorticity_is_the_laplacian_of_the_stream_function() {
    // `vorticity_from_psi` applies `-L` where `L = -Delta` is the stored
    // operator. Checking it against a manufactured psi pins the sign, which no
    // smoke test on a solved field would.
    let cm = nephroid();
    let mesh = &cm.mesh;
    let ops = Operators::from_mesh(mesh, &Euclidean::<2>);
    let coords = extract_coords(mesh);
    let nv = mesh.n_vertices();
    let k = 2.0 * std::f64::consts::PI / 53.071676;
    let psi: Vec<f64> =
        (0..nv).map(|i| (k * coords[i][0]).sin() * (k * coords[i][1]).sin()).collect();
    let w = vorticity_from_psi(&psi, &ops);

    // Delta psi = -2 k^2 psi for this psi. Compare in the interior only: the
    // discrete Laplacian is one sided at the boundary.
    let on_wall: std::collections::HashSet<usize> =
        cm.boundary_vertices.iter().copied().collect();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut n = 0usize;
    for i in 0..nv {
        if on_wall.contains(&i) {
            continue;
        }
        let want = -2.0 * k * k * psi[i];
        num += (w[i] - want).powi(2);
        den += want * want;
        n += 1;
    }
    assert!(n > 100, "only {n} interior vertices to compare");
    let rel = (num / den).sqrt();
    assert!(rel < 0.25, "vorticity is off by {rel:.3e} relative, sign or scaling wrong");
    // and the sign specifically: the two must be anticorrelated with psi
    let dot: f64 = (0..nv).filter(|i| !on_wall.contains(i)).map(|i| w[i] * psi[i]).sum();
    assert!(dot < 0.0, "Delta psi should oppose psi for this mode, got dot = {dot:.3e}");
}
