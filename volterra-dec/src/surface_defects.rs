//! Defect detection for a nematic on a triangulated surface.
//!
//! On a flat lattice a defect is found by walking a plaquette and adding up the
//! turning of the director. On a curved surface that walk is meaningless on its
//! own: the director at each vertex is written in that vertex's own frame, and
//! neighbouring frames differ by a rotation. Comparing two vertices without
//! undoing that rotation reports the frames' disagreement as though it were the
//! field's.
//!
//! The rotation is the discrete connection, already assembled for the
//! connection Laplacian, so the walk here is the same plaquette sum with each
//! step transported first. Around a triangle the transported argument of the
//! spin-2 field returns to itself up to `2 pi n`, and the director winds half
//! as fast as the spin-2 field does, so the charge is `n/2`.
//!
//! # The check that matters
//!
//! Poincaré-Hopf fixes the total: on a closed surface the charges sum to the
//! Euler characteristic, `+2` on a sphere, which is four `+1/2` defects' worth.
//! It holds for ANY field on the mesh, ordered or not, so it tests the detector
//! rather than the physics and it is what [`total_charge`] is for.

use crate::QField;

/// Half-integer charge in units of a half: `+1` is a `+1/2` defect.
pub type HalfCharge = i32;

/// A defect on the surface: a unit position and a charge in halves.
pub type SurfaceDefect = ([f64; 3], HalfCharge);

/// Wrap to `(-pi, pi]`.
fn wrap(x: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut y = x % tau;
    if y > std::f64::consts::PI {
        y -= tau;
    }
    if y <= -std::f64::consts::PI {
        y += tau;
    }
    y
}

/// Defects of a spin-2 field on a triangulated surface, one per triangle that
/// has any charge.
///
/// `phases` is the spin-2 transport angle of each mesh edge, for the stored
/// direction `boundaries[e][0] -> boundaries[e][1]`; traversing the other way
/// takes the negative.
///
/// The position of a defect is the triangle's centroid, normalised back onto
/// the unit sphere. A defect has no position finer than the face it is found
/// on, and pretending otherwise would invent a precision the winding does not
/// have.
pub fn detect_defects_surface(
    vertices: &[[f64; 3]],
    simplices: &[[usize; 3]],
    boundaries: &[[usize; 2]],
    simplex_boundary_ids: &[[usize; 3]],
    phases: &[f64],
    q: &QField,
) -> Vec<SurfaceDefect> {
    let arg = |v: usize| q.q2[v].atan2(q.q1[v]);
    let mut out = Vec::new();

    for (f, tri) in simplices.iter().enumerate() {
        let ids = &simplex_boundary_ids[f];
        let mut total = 0.0_f64;
        let mut ok = true;
        for step in 0..3 {
            let u = tri[step];
            let v = tri[(step + 1) % 3];
            // The edge of this face joining u and v, and which way it is stored.
            let mut phase = None;
            for &e in ids.iter() {
                let [a, b] = boundaries[e];
                if a == u && b == v {
                    phase = Some(phases[e]);
                    break;
                }
                if a == v && b == u {
                    phase = Some(-phases[e]);
                    break;
                }
            }
            let Some(p) = phase else {
                ok = false;
                break;
            };
            // Transport u's field into v's frame, then take the turning.
            total += wrap(arg(v) - (arg(u) + p));
        }
        if !ok {
            continue;
        }
        let n = (total / std::f64::consts::TAU).round() as i32;
        if n == 0 {
            continue;
        }
        let c = [
            (vertices[tri[0]][0] + vertices[tri[1]][0] + vertices[tri[2]][0]) / 3.0,
            (vertices[tri[0]][1] + vertices[tri[1]][1] + vertices[tri[2]][1]) / 3.0,
            (vertices[tri[0]][2] + vertices[tri[1]][2] + vertices[tri[2]][2]) / 3.0,
        ];
        let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let pos = if r > 0.0 { [c[0] / r, c[1] / r, c[2] / r] } else { c };
        out.push((pos, n));
    }
    out
}

/// Total charge in halves. On a closed surface this is the Euler
/// characteristic, `+4` halves on a sphere, whatever the field.
pub fn total_charge(defects: &[SurfaceDefect]) -> HalfCharge {
    defects.iter().map(|d| d.1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection_laplacian::ConnectionLaplacian;
    use crate::mesh_gen::icosphere;
    use crate::QField;
    use crate::DecDomain;
    use cartan_manifolds::sphere::Sphere;

    fn setup(refinement: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<[usize; 2]>,
                                    Vec<[usize; 3]>, Vec<f64>) {
        let domain = DecDomain::new(icosphere(refinement), Sphere::<3>)
            .expect("domain assembly");
        let mesh = &domain.mesh;
        let coords: Vec<[f64; 3]> =
            mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
        let h = &domain.ops.hodge;
        let star0: Vec<f64> = (0..h.star0().len()).map(|i| h.star0()[i]).collect();
        let star1: Vec<f64> = (0..h.star1().len()).map(|i| h.star1()[i]).collect();
        let conn = ConnectionLaplacian::new(mesh, &coords, &star0, &star1);
        (
            coords,
            mesh.simplices.clone(),
            mesh.boundaries.clone(),
            mesh.simplex_boundary_ids.clone(),
            conn.edge_phases(),
        )
    }

    #[test]
    fn the_total_charge_is_the_euler_characteristic() {
        // Poincaré-Hopf, on whatever field happens to be there. A random field
        // is the strongest form of the test: it has no structure the detector
        // could be tuned to, and the answer is still forced.
        let (verts, tris, edges, ids, phases) = setup(3);
        for seed in [1u64, 7, 99, 12345] {
            let q = QField::random_perturbation(verts.len(), 0.5, seed);
            let d = detect_defects_surface(&verts, &tris, &edges, &ids, &phases, &q);
            assert_eq!(
                total_charge(&d),
                4,
                "seed {seed} gave {} defects summing to {}",
                d.len(),
                total_charge(&d)
            );
        }
    }

    #[test]
    fn a_field_at_rest_still_owes_the_sphere_four_halves() {
        // The constraint is topological, so refining the mesh cannot change it.
        for refinement in [2usize, 3, 4] {
            let (verts, tris, edges, ids, phases) = setup(refinement);
            let q = QField::random_perturbation(verts.len(), 0.5, 3);
            let d = detect_defects_surface(&verts, &tris, &edges, &ids, &phases, &q);
            assert_eq!(total_charge(&d), 4, "refinement {refinement}");
        }
    }

    #[test]
    fn every_defect_sits_on_the_surface() {
        let (verts, tris, edges, ids, phases) = setup(3);
        let q = QField::random_perturbation(verts.len(), 0.5, 5);
        for (p, c) in detect_defects_surface(&verts, &tris, &edges, &ids, &phases, &q) {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "off the unit sphere at {r}");
            assert!(c != 0, "a zero charge should not be reported");
        }
    }
}
