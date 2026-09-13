//! Measured tables for the bounded three-dimensional Stokes solver.
//!
//! Prints the numbers the spec's validation ladder quotes: the manufactured
//! convergence rate, the wall slip against mesh size, and the thin-slab limit
//! against the Hele-Shaw law the depth-averaged two-dimensional solver assumes.

use volterra_dec::mimetic::assemble_star;
use volterra_dec::stokes_3d::BoundedStokes3D;
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
fn velocity(x: [f64; 3]) -> [f64; 3] {
    [
        0.0,
        phi(x[0]) * phi(x[1]) * dphi(x[2]),
        -phi(x[0]) * dphi(x[1]) * phi(x[2]),
    ]
}
fn source(eta: f64) -> impl Fn([f64; 3]) -> [f64; 3] + Copy {
    move |x: [f64; 3]| {
        let (a, b, c) = (x[0], x[1], x[2]);
        let ly =
            ddphi(a) * phi(b) * dphi(c) + phi(a) * ddphi(b) * dphi(c) + phi(a) * phi(b) * dddphi(c);
        let lz =
            ddphi(a) * dphi(b) * phi(c) + phi(a) * dddphi(b) * phi(c) + phi(a) * dphi(b) * ddphi(c);
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

fn duct_profile(x: f64, y: f64, a: f64, b: f64, g: f64, eta: f64) -> f64 {
    let mut acc = 0.0;
    let mut n = 1usize;
    while n <= 81 {
        let k = n as f64 * std::f64::consts::PI / a;
        let amp = 4.0 * g * a * a / (eta * (n as f64).powi(3) * std::f64::consts::PI.powi(3));
        acc += amp * (1.0 - (k * (y - 0.5 * b)).cosh() / (0.5 * k * b).cosh()) * (k * x).sin();
        n += 2;
    }
    acc
}

fn main() {
    let eta = 1.3;
    println!("manufactured solution, unit box, eta = {eta}");
    println!("   n     dof   rel err    rate      div        slip max    slip rms   slip/|u|");
    let mut prev: Option<(f64, f64)> = None;
    for n in [3usize, 4, 6, 8, 10] {
        let mesh = box_mesh(n, n, n, 1.0, 1.0, 1.0).unwrap();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let exact = mesh.flux_dofs(velocity);
        let f = mesh.flux_dofs(source(eta));
        let wall = vec![0.0; mesh.n_faces()];
        let solver = BoundedStokes3D::new(mesh).unwrap();
        let (a, b, c) = solver.dimensions();
        let flow = solver.solve(&f, &wall, eta).unwrap();
        let err: Vec<f64> = flow.flux.iter().zip(&exact).map(|(p, q)| p - q).collect();
        let un = m2_norm(&m2, &exact);
        let rel = m2_norm(&m2, &err) / un;
        let slip = flow.wall_slip(solver.mesh());
        // The L2 velocity scale, as a velocity rather than as a flux norm.
        let scale = un / solver.mesh().volume().sqrt();
        let rate = prev.map_or(f64::NAN, |(h0, e0)| {
            (e0 / rel).ln() / (h0 / (1.0 / n as f64)).ln()
        });
        println!(
            "{n:4} {:7} {rel:9.3e} {rate:7.3} {:9.2e} {:11.4e} {:11.4e} {:9.4}",
            a + b + c,
            flow.divergence_residual,
            slip.max,
            slip.rms,
            slip.rms / scale
        );
        prev = Some((1.0 / n as f64, rel));
    }

    println!();
    let (a, b, len) = (1.0, 0.6, 0.5);
    let (g, veta) = (1.0, 1.0);
    println!("duct {a} x {b} x {len}, prescribed inlet and outlet, eta = {veta}");
    println!("   n   rel err    div        slip rms     dp/dz / g");
    for n in [3usize, 4, 6, 8, 10] {
        let mesh = box_mesh(n, n, n, a, b, len).unwrap();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let exact =
            mesh.flux_dofs(|x: [f64; 3]| [0.0, 0.0, duct_profile(x[0], x[1], a, b, g, veta)]);
        let zero = vec![0.0; mesh.n_faces()];
        let mut wall = vec![0.0; mesh.n_faces()];
        for fi in 0..mesh.n_faces() {
            if mesh.is_boundary_face(fi) {
                let c = mesh.face_centroid(fi);
                if c[2] < 1e-12 || c[2] > len - 1e-12 {
                    wall[fi] = exact[fi];
                }
            }
        }
        let solver = BoundedStokes3D::new(mesh).unwrap();
        let flow = solver.solve(&zero, &wall, veta).unwrap();
        let err: Vec<f64> = flow.flux.iter().zip(&exact).map(|(p, q)| p - q).collect();
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
        let slope = (spz - sz * sp / nn) / (szz - sz * sz / nn);
        println!(
            "{n:4} {:9.3e} {:9.2e} {:11.4e} {:11.4}",
            m2_norm(&m2, &err) / m2_norm(&m2, &exact),
            flow.divergence_residual,
            flow.wall_slip(mesh).rms,
            -slope / g
        );
    }

    println!();
    println!("plug against Poiseuille, same flow rate, relative M2 distance");
    for n in [4usize, 8] {
        let mesh = box_mesh(n, n, 2, a, b, len).unwrap();
        let m2 = assemble_star(&mesh, 2).unwrap();
        let exact =
            mesh.flux_dofs(|x: [f64; 3]| [0.0, 0.0, duct_profile(x[0], x[1], a, b, g, veta)]);
        // The plug of equal flow rate: free slip on the side walls answers this
        // shape rather than the parabolic one.
        let mut q = 0.0;
        for (fi, e) in exact.iter().enumerate() {
            if mesh.is_boundary_face(fi) && mesh.face_centroid(fi)[2] < 1e-12 {
                q += e.abs();
            }
        }
        let mean = q / (a * b);
        let plug = mesh.flux_dofs(|_| [0.0, 0.0, mean]);
        let d: Vec<f64> = plug.iter().zip(&exact).map(|(p, e)| p - e).collect();
        println!("{n:4} {:9.4}", m2_norm(&m2, &d) / m2_norm(&m2, &exact));
    }

    println!();
    println!("thin slab: the flow rate is imposed at the ports and conserved");
    println!("exactly, so the solver's own answer is the pressure gradient, and");
    println!("dp/dz over the Hele-Shaw prediction 12 eta U / h^2 is the wall drag");
    println!("the depth-averaged two-dimensional solver assumes.");
    println!();
    println!("  h/a    ny   dy/dx   solved U    U / (g h^2/12eta)   1 - 0.63 h/a   dp/dz / g");
    let a = 1.0;
    let dlen = 0.4;
    let run = |h: f64, nx: usize, ny: usize, nz: usize| {
        let mesh = box_mesh(nx, ny, nz, a, h, dlen).unwrap();
        let exact =
            mesh.flux_dofs(|x: [f64; 3]| [0.0, 0.0, duct_profile(x[0], x[1], a, h, 1.0, 1.0)]);
        let zero = vec![0.0; mesh.n_faces()];
        let mut wall = vec![0.0; mesh.n_faces()];
        for fi in 0..mesh.n_faces() {
            if mesh.is_boundary_face(fi) {
                let c = mesh.face_centroid(fi);
                if c[2] < 1e-12 || c[2] > dlen - 1e-12 {
                    wall[fi] = exact[fi];
                }
            }
        }
        let solver = BoundedStokes3D::new(mesh).unwrap();
        let flow = solver.solve(&zero, &wall, 1.0).unwrap();
        let mesh = solver.mesh();

        let zmid = 0.5 * dlen;
        let mut q = 0.0;
        for fi in 0..mesh.n_faces() {
            let c = mesh.face_centroid(fi);
            let nrm = mesh.face_normal(fi);
            if (c[2] - zmid).abs() < 1e-9 && nrm[2].abs() > 0.9 {
                q += flow.flux[fi] * nrm[2].signum();
            }
        }
        let mean = q / (a * h);

        // The layer-mean pressure against the layer's height. Regressing cell by
        // cell mixes in the transverse pressure variation, which the layer mean
        // removes.
        let dz = dlen / nz as f64;
        let mut num = vec![0.0; nz];
        let mut den = vec![0.0; nz];
        for t in 0..mesh.n_tets() {
            let z = mesh.tet_centroid(t)[2];
            let l = ((z / dz).floor() as usize).min(nz - 1);
            let vol = mesh.tet_volume(t);
            num[l] += flow.pressure[t];
            den[l] += vol;
        }
        let (mut sz, mut szz, mut sp, mut spz) = (0.0, 0.0, 0.0, 0.0);
        for l in 0..nz {
            let z = (l as f64 + 0.5) * dz;
            let pv = num[l] / den[l];
            sz += z;
            szz += z * z;
            sp += pv;
            spz += pv * z;
        }
        let n = nz as f64;
        let slope = (spz - sz * sp / n) / (szz - sz * sz / n);
        let hs = h * h / 12.0;
        println!(
            "{:6.4} {ny:5} {:7.3} {mean:11.4e} {:17.4} {:14.4} {:11.4}",
            h / a,
            (h / ny as f64) / (a / nx as f64),
            mean / hs,
            1.0 - 0.63 * h / a,
            -slope
        );
    };
    for h in [0.5_f64, 0.25, 0.125, 0.0625] {
        run(h, 12, 8, 4);
    }
    println!();
    println!("across-gap refinement at h/a = 0.125, nx = 12, nz = 4");
    println!("  h/a    ny   dy/dx   solved U    U / (g h^2/12eta)   1 - 0.63 h/a   dp/dz / g");
    for ny in [4usize, 6, 8, 12, 16] {
        run(0.125, 12, ny, 4);
    }
    println!();
    println!("along-duct refinement at h/a = 0.125, nx = 12, ny = 8");
    println!("  h/a    ny   dy/dx   solved U    U / (g h^2/12eta)   1 - 0.63 h/a   dp/dz / g");
    for nz in [4usize, 8, 16] {
        run(0.125, 12, 8, nz);
    }
    println!();
    println!("in-plane refinement at h/a = 0.125, ny = 8, nz = 8");
    println!("  h/a    ny   dy/dx   solved U    U / (g h^2/12eta)   1 - 0.63 h/a   dp/dz / g");
    for nx in [8usize, 12, 20, 32] {
        run(0.125, nx, 8, 8);
    }
}
