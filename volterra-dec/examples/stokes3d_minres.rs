//! Iteration counts for MINRES against the Riesz-map preconditioner.
//!
//! The direct sparse LU is exact and its memory grows fastest, so the question a
//! preconditioner has to answer is whether its iteration count stays put as the
//! mesh refines. Both variants are measured against the direct answer on the
//! same problem, so the table reports accuracy as well as effort.

use std::time::Instant;

use volterra_dec::mimetic::assemble_star;
use volterra_dec::saddle::RieszMode;
use volterra_dec::stokes_3d::{BoundedStokes3D, Inversion};
use volterra_dec::tet_mesh::box_mesh;

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
fn source(eta: f64) -> impl Fn([f64; 3]) -> [f64; 3] + Copy {
    move |x: [f64; 3]| {
        let (a, b, c) = (x[0], x[1], x[2]);
        let ly = ddphi(a) * phi(b) * dphi(c) + phi(a) * ddphi(b) * dphi(c)
            + phi(a) * phi(b) * dddphi(c);
        let lz = ddphi(a) * dphi(b) * phi(c) + phi(a) * dddphi(b) * phi(c)
            + phi(a) * dphi(b) * ddphi(c);
        [1.0, -eta * ly, eta * lz]
    }
}
fn m2_norm(m2: &sprs::CsMat<f64>, x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (v, (r, c)) in m2.iter() {
        s += x[r] * v * x[c];
    }
    s.max(0.0).sqrt()
}

fn main() {
    let eta = 1.3;
    println!("manufactured problem, unit box, eta = {eta}");
    println!(
        "   n     dof   direct s | jacobi it   s     | H(curl) it   s     rel diff | L2 it   s     rel diff"
    );
    for n in [3usize, 4, 6, 8, 10] {
        let mesh = box_mesh(n, n, n, 1.0, 1.0, 1.0).unwrap();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let f = mesh.flux_dofs(source(eta));
        let wall = vec![0.0; mesh.n_faces()];

        let t0 = Instant::now();
        let direct = BoundedStokes3D::new(mesh.clone()).unwrap();
        let ref_flow = direct.solve(&f, &wall, eta).unwrap();
        let direct_s = t0.elapsed().as_secs_f64();
        let (a, b, c) = direct.dimensions();
        let refnorm = m2_norm(&m2, &ref_flow.flux).max(1e-300);

        let mut cells = Vec::new();
        for mode in [RieszMode::Jacobi, RieszMode::Exact, RieszMode::ExactMass] {
            let t0 = Instant::now();
            let it = BoundedStokes3D::with_inversion(
                mesh.clone(),
                Inversion::Minres { mode, tol: 1e-10, max_iter: 20000 },
            )
            .unwrap();
            let flow = it.solve(&f, &wall, eta).unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let r = flow.report.unwrap();
            let d: Vec<f64> = flow
                .flux
                .iter()
                .zip(&ref_flow.flux)
                .map(|(p, q)| p - q)
                .collect();
            cells.push((
                r.iterations,
                r.converged,
                secs,
                m2_norm(&m2, &d) / refnorm,
            ));
        }
        let (ji, jc, js, _jd) = cells[0];
        let (ei, ec, es, ed) = cells[1];
        let (mi, mc, ms, md) = cells[2];
        println!(
            "{n:4} {:7} {direct_s:10.3} | {ji:8}{} {js:7.3} | {ei:9}{} {es:7.3} {ed:9.2e} | {mi:5}{} {ms:7.3} {md:9.2e}",
            a + b + c,
            if jc { " " } else { "*" },
            if ec { " " } else { "*" },
            if mc { " " } else { "*" }
        );
    }
    println!();
    println!("A star marks a solve that reached the iteration cap. The relative");
    println!("difference is against the direct answer in the M2 norm.");
}
