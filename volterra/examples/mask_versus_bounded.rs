//! What a chamber wall costs when it is a mask rather than a boundary condition.
//!
//! `stokes_solve_3d` is spectral and periodic, so it has no wall. A chamber is
//! imposed afterwards by zeroing the velocity on the outer layer. This example
//! measures the two things that costs, on one Q-field, against the bounded
//! solver solving the same activity in the same box.
//!
//! The periodic velocity on the boundary layer is what the mask has to destroy,
//! and the divergence of the masked field is what destroying it costs. The
//! bounded solver states the wall as a boundary condition instead: the normal
//! component vanishes exactly on every wall face and the cellwise divergence
//! stays at machine precision.

use volterra_core::{ActiveNematicParams3D, QField3D, VelocityField3D};
use volterra_dec::active_stokes_3d::ConfinedActiveStokes3D;
use volterra_fd::stokes_solve_3d;

/// The largest speed on the outermost grid layer.
fn boundary_speed(u: &VelocityField3D) -> f64 {
    let (nx, ny, nz) = (u.nx, u.ny, u.nz);
    let mut worst = 0.0_f64;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
                    let v = u.u[(k * ny + j) * nx + i];
                    worst = worst.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                }
            }
        }
    }
    worst
}

fn interior_speed(u: &VelocityField3D) -> f64 {
    u.u.iter().fold(0.0_f64, |a, v| {
        a.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
    })
}

fn main() {
    let dx = 0.25;
    let mut p = ActiveNematicParams3D::default_test();
    p.zeta_eff = 1.0;
    p.eta = 1.0;

    println!("  grid   periodic |u| wall   masked max div   bounded wall normal   bounded div");
    for n in [9usize, 13, 17] {
        let q = QField3D::random_perturbation(n, n, n, dx, 0.2, 4);

        let (u_periodic, _) = stokes_solve_3d(&q, &p);
        let scale = interior_speed(&u_periodic).max(1e-300);
        let wall_periodic = boundary_speed(&u_periodic);

        // The mask: the chamber wall imposed after the fact.
        let mut masked = u_periodic.clone();
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    if i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1 {
                        masked.u[(k * n + j) * n + i] = [0.0; 3];
                    }
                }
            }
        }
        let div_masked = masked
            .divergence()
            .phi
            .iter()
            .fold(0.0_f64, |a, d| a.max(d.abs()));
        let div_periodic = u_periodic
            .divergence()
            .phi
            .iter()
            .fold(0.0_f64, |a, d| a.max(d.abs()));

        // The bounded solver, meshed one cell per grid cell.
        let solver = ConfinedActiveStokes3D::matching_grid((n, n, n), dx).unwrap();
        let flow = solver.solve_flow(&q, &p).unwrap();
        let mesh = solver.solver().mesh();
        let mut owner = vec![usize::MAX; mesh.n_faces()];
        for t in 0..mesh.n_tets() {
            for &f in &mesh.tet_faces[t] {
                if mesh.is_boundary_face(f) {
                    owner[f] = t;
                }
            }
        }
        let mut wall_normal = 0.0_f64;
        for f in 0..mesh.n_faces() {
            if !mesh.is_boundary_face(f) {
                continue;
            }
            let c = mesh.face_centroid(f);
            let v = flow.velocity_in_cell(mesh, owner[f], c);
            let nrm = mesh.face_normal(f);
            wall_normal = wall_normal.max((v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2]).abs());
        }
        let flux_scale = flow.flux.iter().fold(0.0_f64, |a, x| a.max(x.abs()));

        println!(
            "{n:6} {:16.4e} {:16.3e} {:21.3e} {:13.3e}",
            wall_periodic / scale,
            div_masked / scale * dx,
            wall_normal / interior_speed(&solver.sample_to_grid(&flow)).max(1e-300),
            flow.divergence_residual / flux_scale.max(1e-300)
        );
        let _ = div_periodic;
    }
    println!();
    println!("Columns two to five are relative: the wall speed and the wall normal");
    println!("velocity against each solver's own interior scale, and each divergence");
    println!("against the same. The periodic solver's divergence before masking is");
    println!("spectral, so the masked column is the mask's own contribution.");
}
