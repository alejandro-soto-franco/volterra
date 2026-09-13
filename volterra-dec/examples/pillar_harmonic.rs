//! Does a chamber with a pillar leave the bounded Stokes operator singular?
//!
//! The homogeneous reduced system forces `w = 0` through `w^T M1 w = 0`, and
//! what is left is `d2 u = 0` and `d1^T M2 u = 0` on the interior faces. That
//! space is the discrete harmonic 2-forms with vanishing normal trace, and the
//! open question is its dimension on a chamber whose first Betti number is one.
//!
//! Prints the nullity directly, from the singular values of the stacked
//! constraint, against the same measurement on a simply connected box.

use nalgebra::DMatrix;
use volterra_dec::mimetic::assemble_star;
use volterra_dec::stokes_3d::BoundedStokes3D;
use volterra_dec::tet_mesh::{TetComplex, box_mesh, pillar_mesh};

/// The dimension of `{u on interior faces : d2 u = 0, d1^T M2 u = 0}`, and the
/// singular values around the cut.
fn harmonic_dimension(mesh: &TetComplex) -> (usize, usize, f64, f64) {
    let d1 = mesh.d1();
    let d2 = mesh.d2();
    let m2 = assemble_star(mesh, 2).unwrap();

    let free: Vec<usize> = (0..mesh.n_faces())
        .filter(|&f| !mesh.is_boundary_face(f))
        .collect();
    let mut slot = vec![usize::MAX; mesh.n_faces()];
    for (i, &f) in free.iter().enumerate() {
        slot[f] = i;
    }

    let rows = mesh.n_tets() + mesh.n_edges();
    let mut a = DMatrix::<f64>::zeros(rows, free.len());
    for (v, (t, f)) in d2.iter() {
        if slot[f] != usize::MAX {
            a[(t, slot[f])] += v;
        }
    }
    // `(d1^T M2)[e, f] = sum_g d1[g, e] M2[g, f]`.
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

    let sv = a.singular_values();
    let mut s: Vec<f64> = sv.iter().cloned().collect();
    s.sort_by(|p, q| q.partial_cmp(p).unwrap());
    let top = s.first().cloned().unwrap_or(0.0);
    let tol = 1e-10 * top * (a.nrows().max(a.ncols()) as f64);
    let rank = s.iter().filter(|&&x| x > tol).count();
    let nullity = free.len() - rank;
    let smallest_kept = s.get(rank.saturating_sub(1)).cloned().unwrap_or(0.0);
    let largest_dropped = s.get(rank).cloned().unwrap_or(0.0);
    (
        free.len(),
        nullity,
        smallest_kept / top,
        largest_dropped / top,
    )
}

fn report(name: &str, mesh: TetComplex) {
    let chi = mesh.n_vertices() as i64 - mesh.n_edges() as i64 + mesh.n_faces() as i64
        - mesh.n_tets() as i64;
    let (n_free, nullity, kept, dropped) = harmonic_dimension(&mesh);
    println!(
        "{name:22} chi {chi:3}  interior faces {n_free:6}  harmonic dim {nullity:3}  \
         sigma_kept/sigma_0 {kept:.3e}  sigma_dropped/sigma_0 {dropped:.3e}"
    );

    // What the solver itself does with it.
    let nf = mesh.n_faces();
    let f = mesh.flux_dofs(|x: [f64; 3]| [0.3 * x[2], -0.4, 0.2 * x[0]]);
    match BoundedStokes3D::new(mesh) {
        Ok(solver) => match solver.solve(&f, &vec![0.0; nf], 1.0) {
            Ok(flow) => {
                let norm = flow.flux.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
                println!(
                    "{:22} factorised; max |flux| {norm:.4e}, divergence {:.2e}",
                    "", flow.divergence_residual
                );
            }
            Err(e) => println!("{:22} factorised, solve failed: {e}", ""),
        },
        Err(e) => println!("{:22} factorisation failed: {e}", ""),
    }
}

fn main() {
    println!("Euler characteristic 1 is a ball; 0 is a solid torus (b1 = 1).");
    println!();
    report("box 3x3x2", box_mesh(3, 3, 2, 1.0, 1.0, 0.5).unwrap());
    report("box 4x4x3", box_mesh(4, 4, 3, 1.0, 1.0, 0.5).unwrap());
    report("pillar 8x2x1", pillar_mesh(8, 2, 1, 0.3, 1.0, 0.5).unwrap());
    report(
        "pillar 12x3x2",
        pillar_mesh(12, 3, 2, 0.3, 1.0, 0.5).unwrap(),
    );
    report(
        "pillar 16x4x2",
        pillar_mesh(16, 4, 2, 0.3, 1.0, 0.5).unwrap(),
    );
}
