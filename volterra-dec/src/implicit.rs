//! Meshes of implicit surfaces.
//!
//! A surface given as `f = 0` is marched into triangles, then smoothed and
//! flipped until the cotangent weights DEC needs are non-negative. Marching
//! alone is not enough: a cut can land arbitrarily close to a grid corner, and
//! the sliver that leaves has an angle under a tenth of a degree.
//!
//! This is the route to a surface of negative Euler characteristic. An
//! icosphere and a torus are built by hand because a sphere and a torus are
//! product grids; a genus-2 surface is not, and a constant-curvature
//! hyperbolic surface cannot be embedded in `R^3` at all, by Hilbert's
//! theorem. What is reachable is a surface whose curvature is negative over
//! most of it, which is enough to force `chi = -2` and with it a net defect
//! charge of `-2`.

use cartan_manifolds::euclidean::Euclidean;
use std::collections::HashMap;

/// A tube around the Bernoulli lemniscate.
///
/// The lemniscate `L = 0` is a figure eight, and a regular neighbourhood of a
/// figure eight is a genus-2 handlebody, so the boundary of the tube is a
/// genus-2 surface. Nothing here depends on that being true: the Euler
/// characteristic is measured, not assumed.
pub fn genus2(p: [f64; 3], r: f64) -> f64 {
    let (x, y, z) = (p[0], p[1], p[2]);
    let q = x * x + y * y;
    let l = q * q - (x * x - y * y);
    l * l + z * z - r * r
}

/// Marching tetrahedra on a regular grid.
///
/// Each cube splits into six tetrahedra sharing the main diagonal, and a
/// tetrahedron has three sign patterns up to symmetry, so there is no
/// ambiguous case to resolve and the output is watertight by construction.
/// Marching cubes would need its 256-entry table and still leave the ambiguous
/// faces to a convention.
pub fn marching_tets(
    f: &dyn Fn([f64; 3]) -> f64,
    lo: [f64; 3],
    hi: [f64; 3],
    n: usize,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let at = |i: usize, j: usize, k: usize| -> [f64; 3] {
        [
            lo[0] + (hi[0] - lo[0]) * i as f64 / n as f64,
            lo[1] + (hi[1] - lo[1]) * j as f64 / n as f64,
            lo[2] + (hi[2] - lo[2]) * k as f64 / n as f64,
        ]
    };
    let idx = |i: usize, j: usize, k: usize| (i * (n + 1) + j) * (n + 1) + k;

    let mut val = vec![0.0f64; (n + 1) * (n + 1) * (n + 1)];
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                val[idx(i, j, k)] = f(at(i, j, k));
            }
        }
    }
    // A grid value at exactly zero puts the cut on a corner, so two cuts on
    // different edges land on the same point and the triangle between them has
    // no area. Twenty such triangles appeared at one resolution and broke the
    // angle-defect identity while leaving V - E + F correct, which is the kind
    // of fault only Gauss-Bonnet catches. Nudging the sample off zero moves the
    // surface by less than the rounding already present and removes the case.
    let scale = val.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let eps = 1e-9 * scale;
    for v in val.iter_mut() {
        if v.abs() < eps {
            *v = eps;
        }
    }

    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    // One vertex per crossed grid EDGE, keyed by its endpoints, so the two
    // tetrahedra either side of a face place the same vertex and the surface
    // closes.
    let mut on_edge: HashMap<(usize, usize), usize> = HashMap::new();

    let mut cut = |a: usize, b: usize, verts: &mut Vec<[f64; 3]>, at_pt: &dyn Fn(usize) -> [f64; 3]| {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&v) = on_edge.get(&key) {
            return v;
        }
        let (fa, fb) = (val[key.0], val[key.1]);
        let t = if (fb - fa).abs() < 1e-300 { 0.5 } else { (fa / (fa - fb)).clamp(1e-6, 1.0 - 1e-6) };
        let (pa, pb) = (at_pt(key.0), at_pt(key.1));
        let p = [
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        ];
        verts.push(p);
        let v = verts.len() - 1;
        on_edge.insert(key, v);
        v
    };

    let at_pt = |flat: usize| -> [f64; 3] {
        let k = flat % (n + 1);
        let j = (flat / (n + 1)) % (n + 1);
        let i = flat / ((n + 1) * (n + 1));
        at(i, j, k)
    };

    // The six tetrahedra of a cube, as corner offsets 0..7 with
    // corner = 4*di + 2*dj + dk.
    const TETS: [[usize; 4]; 6] = [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
        [0, 2, 6, 7],
        [0, 2, 3, 7],
    ];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let corner = |c: usize| {
                    let (di, dj, dk) = (c >> 2, (c >> 1) & 1, c & 1);
                    idx(i + di, j + dj, k + dk)
                };
                for t in TETS {
                    let c: [usize; 4] = [corner(t[0]), corner(t[1]), corner(t[2]), corner(t[3])];
                    // Inside is f < 0. Order the corners so the negative ones
                    // come first; the sign patterns then reduce to a count.
                    let mut ord = c;
                    ord.sort_by(|&a, &b| val[a].partial_cmp(&val[b]).unwrap());
                    let n_neg = ord.iter().filter(|&&v| val[v] < 0.0).count();
                    match n_neg {
                        1 => {
                            let (a, b, cc, d) = (ord[0], ord[1], ord[2], ord[3]);
                            let (p, q, r) = (
                                cut(a, b, &mut verts, &at_pt),
                                cut(a, cc, &mut verts, &at_pt),
                                cut(a, d, &mut verts, &at_pt),
                            );
                            tris.push([p, q, r]);
                        }
                        2 => {
                            let (a, b, cc, d) = (ord[0], ord[1], ord[2], ord[3]);
                            let (p, q) = (
                                cut(a, cc, &mut verts, &at_pt),
                                cut(a, d, &mut verts, &at_pt),
                            );
                            let (r, s) = (
                                cut(b, cc, &mut verts, &at_pt),
                                cut(b, d, &mut verts, &at_pt),
                            );
                            tris.push([p, q, s]);
                            tris.push([p, s, r]);
                        }
                        3 => {
                            let (a, b, cc, d) = (ord[0], ord[1], ord[2], ord[3]);
                            let _ = (a, b, cc);
                            let (p, q, r) = (
                                cut(d, ord[0], &mut verts, &at_pt),
                                cut(d, ord[1], &mut verts, &at_pt),
                                cut(d, ord[2], &mut verts, &at_pt),
                            );
                            tris.push([p, q, r]);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    (verts, tris)
}

/// Move a point onto `f = 0` by Newton steps along the gradient.
///
/// Marching tetrahedra place a vertex by linear interpolation along a grid
/// edge, which is only first order, and the smoothing below moves vertices off
/// the surface outright. Both are corrected here.
pub fn project(f: &dyn Fn([f64; 3]) -> f64, mut p: [f64; 3], h: f64) -> [f64; 3] {
    for _ in 0..4 {
        let v = f(p);
        let g = [
            (f([p[0] + h, p[1], p[2]]) - f([p[0] - h, p[1], p[2]])) / (2.0 * h),
            (f([p[0], p[1] + h, p[2]]) - f([p[0], p[1] - h, p[2]])) / (2.0 * h),
            (f([p[0], p[1], p[2] + h]) - f([p[0], p[1], p[2] - h])) / (2.0 * h),
        ];
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < 1e-300 {
            break;
        }
        let t = v / g2;
        p = [p[0] - t * g[0], p[1] - t * g[1], p[2] - t * g[2]];
    }
    p
}

/// Laplacian smoothing with every vertex put back on the surface.
///
/// Marching output has slivers because a cut can land arbitrarily close to a
/// grid corner. Moving each vertex toward its neighbours' centroid spreads them
/// out, and reprojection stops that from shrinking the surface. Topology is
/// untouched, so `chi` cannot change.
pub fn smooth(
    f: &dyn Fn([f64; 3]) -> f64,
    verts: &mut [[f64; 3]],
    tris: &[[usize; 3]],
    passes: usize,
    rate: f64,
    h: f64,
) {
    let n = verts.len();
    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); n];
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            nbr[a].push(b);
            nbr[b].push(a);
        }
    }
    for v in nbr.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    for _ in 0..passes {
        let old = verts.to_vec();
        for i in 0..n {
            if nbr[i].is_empty() {
                continue;
            }
            let mut c = [0.0; 3];
            for &j in &nbr[i] {
                for k in 0..3 {
                    c[k] += old[j][k];
                }
            }
            let m = nbr[i].len() as f64;
            let target = [
                old[i][0] + rate * (c[0] / m - old[i][0]),
                old[i][1] + rate * (c[1] / m - old[i][1]),
                old[i][2] + rate * (c[2] / m - old[i][2]),
            ];
            verts[i] = project(f, target, h);
        }
    }
}

/// A genus-2 surface, marched from the lemniscate tube and remeshed.
///
/// `r` sets the tube radius, `grid` the marching resolution, and `rounds` how
/// many smoothing-and-flipping passes to take. Returns vertices and triangles
/// with every edge on two faces and no triangle without area.
///
/// The remeshing changes no topology, so the Euler characteristic of the
/// marched surface survives it and can be checked on the result.
pub fn genus2_mesh(r: f64, grid: usize, rounds: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let f = move |p: [f64; 3]| genus2(p, r);
    let (mut verts, mut tris) = marching_tets(&f, [-1.5, -1.0, -0.6], [1.5, 1.0, 0.6], grid);
    let eu = Euclidean::<3>;
    for _ in 0..rounds {
        smooth(&f, &mut verts, &tris, 12, 0.6, 1e-6);
        let sv: Vec<nalgebra::SVector<f64, 3>> =
            verts.iter().map(|p| nalgebra::SVector::from(*p)).collect();
        let m = cartan_dec::mesh::Mesh::from_simplices(&eu, sv, tris.clone());
        let m = cartan_dec::mesh_quality::make_delaunay(m, &eu);
        tris = m.simplices.iter().map(|t| [t[0], t[1], t[2]]).collect();
    }
    orient(&mut tris);
    (verts, tris)
}

/// Orient every triangle consistently by walking the face adjacency, and
/// report whether the surface is orientable and closed.
pub fn orient(tris: &mut [[usize; 3]]) -> (bool, usize) {
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (fi, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            by_edge.entry(if a < b { (a, b) } else { (b, a) }).or_default().push(fi);
        }
    }
    let boundary = by_edge.values().filter(|v| v.len() != 2).count();
    let mut seen = vec![false; tris.len()];
    let mut ok = true;
    for start in 0..tris.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(fi) = stack.pop() {
            let t = tris[fi];
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                for &fj in by_edge.get(&key).into_iter().flatten() {
                    if fj == fi || seen[fj] {
                        continue;
                    }
                    let u = tris[fj];
                    // A shared edge must appear in opposite directions.
                    let same = [(u[0], u[1]), (u[1], u[2]), (u[2], u[0])]
                        .iter()
                        .any(|&(c, d)| c == a && d == b);
                    if same {
                        tris[fj].swap(1, 2);
                    }
                    seen[fj] = true;
                    stack.push(fj);
                }
            }
        }
    }
    if boundary > 0 {
        ok = false;
    }
    (ok, boundary)
}

