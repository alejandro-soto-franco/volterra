//! Coupling the active nematic stress into the bounded three-dimensional
//! solver, and the two projections it needs.

use volterra_core::{ActiveNematicParams3D, QField3D};
use volterra_dec::active_stokes_3d::{active_force_on_grid, ConfinedActiveStokes3D, GridField};
use volterra_dec::stokes_3d::Flow3D;
use volterra_dec::tet_mesh::StructuredBox;

/// The force operator is exact on a Q-field linear in one coordinate, on the
/// wall as well as inside.
///
/// With `q11 = alpha x` and every other component zero, `q33 = -alpha x` and the
/// divergence of the active stress is the constant `(-zeta alpha, 0, 0)`
/// everywhere. A central difference reproduces it inside, a one-sided difference
/// reproduces it on the wall, and the periodic wrap the spectral lane uses
/// differences across the chamber instead: at `i = 0` it reads `q11` from
/// `i = nx - 1`, which is the far wall, and answers a force larger by `nx - 1`
/// with the wrong sign. That error sits one cell deep all round the boundary,
/// which is exactly where a confined result is decided.
#[test]
fn the_active_force_is_exact_on_a_linear_q_field_including_the_wall() {
    let (nx, ny, nz, dx) = (6usize, 5, 4, 0.25);
    let alpha = 0.7;
    let zeta = 1.9;
    let mut q = QField3D::zeros(nx, ny, nz, dx);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                q.q[(k * ny + j) * nx + i][0] = alpha * i as f64 * dx;
            }
        }
    }
    let f = active_force_on_grid(&q, zeta);
    let expect = [-zeta * alpha, 0.0, 0.0];
    for (n, v) in f.iter().enumerate() {
        for a in 0..3 {
            assert!(
                (v[a] - expect[a]).abs() < 1e-12,
                "grid point {n}, component {a}: {} against {}",
                v[a],
                expect[a]
            );
        }
    }
}

/// Every grid point of the chamber, wall included, locates to a cell that
/// contains it, and so does a cloud of interior points.
#[test]
fn every_point_of_the_chamber_locates_to_a_cell_containing_it() {
    let boxed = StructuredBox::new(4, 3, 2, 1.0, 0.7, 0.4);
    let mesh = boxed.build().unwrap();
    let mut checked = 0usize;
    for k in 0..=8 {
        for j in 0..=8 {
            for i in 0..=8 {
                let x = [
                    boxed.lx * i as f64 / 8.0,
                    boxed.ly * j as f64 / 8.0,
                    boxed.lz * k as f64 / 8.0,
                ];
                let t = boxed
                    .locate(&mesh, x)
                    .unwrap_or_else(|| panic!("no cell found for {x:?}"));
                let b = mesh.barycentric(t, x);
                assert!(
                    b.iter().all(|&v| v >= -1e-9),
                    "cell {t} does not contain {x:?}: barycentric {b:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 729);
}

/// A constant field survives the grid-to-face projection exactly.
#[test]
fn a_constant_grid_field_projects_to_the_exact_flux() {
    let solver = ConfinedActiveStokes3D::new((5, 4, 3), 0.25, (3, 2, 2)).unwrap();
    let v0 = [0.41, -0.77, 0.36];
    let field = GridField::new(vec![v0; 5 * 4 * 3], 5, 4, 3, 0.25);
    let got = solver.project_to_faces(&field);
    let want = solver.solver().mesh().flux_dofs(|_| v0);
    for (f, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-13, "face {f}: {a} against {b}");
    }
}

/// A constant flow reaches every grid point, the wall layer included.
///
/// This is the face-to-grid projection's round trip, and it exercises the cell
/// location and the Raviart-Thomas reconstruction together.
#[test]
fn a_constant_flow_reaches_every_grid_point() {
    let solver = ConfinedActiveStokes3D::new((5, 4, 3), 0.25, (3, 2, 2)).unwrap();
    let mesh = solver.solver().mesh();
    let v0 = [0.29, 0.53, -0.61];
    let flow = Flow3D {
        vorticity: vec![0.0; mesh.n_edges()],
        flux: mesh.flux_dofs(|_| v0),
        pressure: vec![0.0; mesh.n_tets()],
        divergence_residual: 0.0,
    };
    let sampled = solver.sample_to_grid(&flow);
    for (n, v) in sampled.u.iter().enumerate() {
        for a in 0..3 {
            assert!(
                (v[a] - v0[a]).abs() < 1e-11,
                "grid point {n}, component {a}: {} against {}",
                v[a],
                v0[a]
            );
        }
    }
}

/// A spatially uniform Q-field exerts no active force, so it drives no flow.
#[test]
fn a_uniform_q_field_drives_no_flow() {
    let p = ActiveNematicParams3D::default_test();
    let q = QField3D::uniform(5, 5, 4, 0.25, [0.3, -0.1, 0.2, 0.05, -0.15]);
    let solver = ConfinedActiveStokes3D::new((5, 5, 4), 0.25, (3, 3, 2)).unwrap();
    let u = solver.solve(&q, &p).unwrap();
    for (n, v) in u.u.iter().enumerate() {
        for a in 0..3 {
            assert!(v[a].abs() < 1e-14, "grid point {n}, component {a}: {}", v[a]);
        }
    }
}

/// The normal velocity vanishes exactly on the wall, and the tangential
/// velocity falls under refinement.
///
/// The lowest-order Raviart-Thomas normal component is constant on a face and
/// equals the flux over the area, so a wall face's zero flux makes the normal
/// component identically zero there. That is the property the mask never had:
/// masking the velocity after a periodic solve leaves both components wrong by
/// an amount nobody measured.
#[test]
fn the_wall_is_a_boundary_condition_rather_than_a_mask() {
    let p = ActiveNematicParams3D::default_test();
    let q = QField3D::random_perturbation(7, 7, 5, 0.25, 0.2, 11);

    let mut slips = Vec::new();
    let mut scale = 0.0_f64;
    for cells in [(3usize, 3usize, 2usize), (6, 6, 4)] {
        let solver = ConfinedActiveStokes3D::new((7, 7, 5), 0.25, cells).unwrap();
        let flow = solver.solve_flow(&q, &p).unwrap();
        let mesh = solver.solver().mesh();
        let flux_scale = flow.flux.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        assert!(
            flow.divergence_residual < 1e-10 * flux_scale,
            "divergence residual {} against a flux scale of {flux_scale}",
            flow.divergence_residual
        );

        // The normal component at every wall face centroid, which the
        // reconstruction makes exactly zero.
        let mut owner = vec![usize::MAX; mesh.n_faces()];
        for t in 0..mesh.n_tets() {
            for &f in &mesh.tet_faces[t] {
                if mesh.is_boundary_face(f) {
                    owner[f] = t;
                }
            }
        }
        let mut worst_normal = 0.0_f64;
        for f in 0..mesh.n_faces() {
            if !mesh.is_boundary_face(f) {
                continue;
            }
            let c = mesh.face_centroid(f);
            let v = flow.velocity_in_cell(mesh, owner[f], c);
            let n = mesh.face_normal(f);
            worst_normal = worst_normal.max((v[0] * n[0] + v[1] * n[1] + v[2] * n[2]).abs());
        }

        let interior = solver
            .sample_to_grid(&flow)
            .u
            .iter()
            .fold(0.0_f64, |a, v| {
                a.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            });
        scale = scale.max(interior);
        assert!(
            worst_normal < 1e-12 * interior.max(1e-30),
            "wall normal velocity {worst_normal} against an interior scale of {interior}"
        );
        slips.push(flow.wall_slip(mesh).rms);
    }
    assert!(slips[1] < slips[0], "wall slip did not fall under refinement: {slips:?}");
    assert!(
        slips[1] < 0.25 * scale,
        "wall slip {} against an interior scale of {scale}",
        slips[1]
    );
}

/// The velocity is linear in the activity and inverse in the viscosity.
///
/// The matrix is free of both, so this tests that each reaches the place the
/// formulation puts it rather than testing the physics: the activity scales the
/// source and the viscosity divides it.
#[test]
fn the_velocity_scales_with_the_activity_and_the_viscosity() {
    let q = QField3D::random_perturbation(5, 5, 4, 0.25, 0.2, 7);
    let solver = ConfinedActiveStokes3D::new((5, 5, 4), 0.25, (3, 3, 2)).unwrap();

    let mut p = ActiveNematicParams3D::default_test();
    p.zeta_eff = 1.0;
    p.eta = 1.0;
    let base = solver.solve_flow(&q, &p).unwrap();

    p.zeta_eff = 3.0;
    let hot = solver.solve_flow(&q, &p).unwrap();
    p.zeta_eff = 1.0;
    p.eta = 4.0;
    let thick = solver.solve_flow(&q, &p).unwrap();

    let scale = base.flux.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    assert!(scale > 0.0, "the reference solve produced no flow at all");
    for i in 0..base.flux.len() {
        assert!(
            (hot.flux[i] - 3.0 * base.flux[i]).abs() < 1e-9 * scale,
            "activity: face {i}"
        );
        assert!(
            (thick.flux[i] - base.flux[i] / 4.0).abs() < 1e-9 * scale,
            "viscosity: face {i}"
        );
    }
}
