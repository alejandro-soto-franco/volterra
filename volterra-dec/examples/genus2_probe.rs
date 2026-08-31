//! Does the curved Stokes operator behave where the Euler characteristic is
//! negative?
//!
//! The sphere forces four `+1/2` defects and has three Killing fields, so the
//! shifted operator `Delta + 2K` annihilates three directions there. A genus-2
//! surface has `chi = -2`, forcing a net charge of `-2`, and has no continuous
//! isometry at all, so the same operator should annihilate nothing. Neither
//! statement can be checked on a sphere, which is the reason for this probe.
//!
//! Three numbers a geometry cannot fake:
//!
//!   1. `V - E + F`, which is `chi` and so fixes the genus.
//!   2. The angle defect summed over vertices, which discrete Gauss-Bonnet
//!      makes exactly `2 pi chi`.
//!   3. The number of eigenvalues of `Delta + 2K` at zero, taken from the
//!      SPECTRUM rather than from the solver's three-candidate heuristic.
//!
//!     cargo run --release -p volterra-dec --example genus2_probe

use cartan_manifolds::sphere::Sphere;
use nalgebra::DMatrix;
use std::collections::HashMap;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::poisson::PoissonSolver;
use volterra_dec::stokes::{extract_coords, gaussian_curvature};
use volterra_dec::DecDomain;

/// A tube around the Bernoulli lemniscate.
///
/// The lemniscate `L = 0` is a figure eight, and a regular neighbourhood of a
/// figure eight is a genus-2 handlebody, so the boundary of the tube is a
/// genus-2 surface. Nothing here depends on that being true: the Euler
/// characteristic is measured, not assumed.
fn genus2(p: [f64; 3], r: f64) -> f64 {
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
fn marching_tets(
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
fn project(f: &dyn Fn([f64; 3]) -> f64, mut p: [f64; 3], h: f64) -> [f64; 3] {
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
fn smooth(
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

/// `V - E + F`, counting each undirected edge once.
fn euler(n_verts: usize, tris: &[[usize; 3]]) -> i64 {
    let mut edges = std::collections::HashSet::new();
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    n_verts as i64 - edges.len() as i64 + tris.len() as i64
}

/// Orient every triangle consistently by walking the face adjacency, and
/// report whether the surface is orientable and closed.
fn orient(tris: &mut [[usize; 3]]) -> (bool, usize) {
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

/// Every eigenvalue of the operator the solve inverts, in the mass inner
/// product, smallest first.
fn spectrum(solver: &PoissonSolver, n: usize) -> Vec<f64> {
    let m = solver.mass();
    let inv_sqrt: Vec<f64> = m.iter().map(|&v| 1.0 / v.max(1e-300).sqrt()).collect();
    // B = M^{-1/2} S M^{-1/2}, symmetric with the same spectrum as M^{-1} S.
    let mut b = DMatrix::<f64>::zeros(n, n);
    let mut e = vec![0.0f64; n];
    for j in 0..n {
        e[j] = inv_sqrt[j];
        let col = solver.apply_operator(&e);
        e[j] = 0.0;
        for i in 0..n {
            b[(i, j)] = col[i] * inv_sqrt[i];
        }
    }
    // Symmetrise away the last bit of assembly asymmetry before the solve.
    let bt = b.transpose();
    let sym = (&b + &bt) * 0.5;
    let mut ev: Vec<f64> = nalgebra::SymmetricEigen::new(sym).eigenvalues.iter().copied().collect();
    ev.sort_by(|a, c| a.abs().partial_cmp(&c.abs()).unwrap());
    ev
}

/// The six smallest eigenvalues by magnitude, and the mesh scale they are
/// judged against.
fn low_spectrum(solver: &PoissonSolver, n: usize) -> Vec<f64> {
    spectrum(solver, n).into_iter().take(6).collect()
}

/// Mean edge length, the `h` a discrete kernel converges in.
fn mesh_h(coords: &[[f64; 3]], tris: &[[usize; 3]]) -> f64 {
    let mut edges = std::collections::HashSet::new();
    for t in tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert(if a < b { (a, b) } else { (b, a) });
        }
    }
    let s: f64 = edges
        .iter()
        .map(|&(a, b)| {
            let (p, q) = (coords[a], coords[b]);
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        })
        .sum();
    s / edges.len() as f64
}

fn probe<M: cartan_core::Manifold>(
    name: &str,
    coords: &[[f64; 3]],
    tris: &[[usize; 3]],
    ops: &cartan_dec::Operators<M, 3, 2>,
    mesh_ref: &cartan_dec::Mesh<M, 3, 2>,
    manifold_ref: &M,
    chi_expect: i64,
) {
    let n = coords.len();
    let chi = euler(n, tris);
    let star0: Vec<f64> = (0..ops.hodge.star0().len()).map(|i| ops.hodge.star0()[i]).collect();
    let k = gaussian_curvature(n, tris, coords, &star0);

    // Discrete Gauss-Bonnet: the angle defects sum to 2 pi chi exactly, so this
    // reads back the topology from the geometry.
    let total: f64 = k.iter().zip(&star0).map(|(a, b)| a * b).sum();
    let want = 2.0 * std::f64::consts::PI * chi as f64;

    // The RAW defect sum is a combinatorial identity, `2 pi V - pi F`, and owes
    // nothing to the dual areas. Comparing it against the integral above
    // separates a mesh whose connectivity is wrong from one whose triangles are
    // merely bad: `gaussian_curvature` returns zero wherever a dual area is not
    // positive, so a sliver silently drops its defect from the integral.
    let mut angle_sum = vec![0.0f64; n];
    for t in tris {
        for a in 0..3 {
            let (i, j, l) = (t[a], t[(a + 1) % 3], t[(a + 2) % 3]);
            let (p, q, r) = (coords[i], coords[j], coords[l]);
            let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
            let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv).max(1e-300);
            angle_sum[i] += d.clamp(-1.0, 1.0).acos();
        }
    }
    let raw: f64 = angle_sum.iter().map(|a| std::f64::consts::TAU - a).sum();
    let bad = star0.iter().filter(|&&v| v <= 0.0).count();

    let shift: Vec<f64> = k.iter().map(|v| 2.0 * v).collect();
    let a = PoissonSolver::new_shifted(ops, &shift, coords).unwrap();
    let low = low_spectrum(&a, n);
    let h = mesh_h(coords, tris);

    println!("\n=== {name} ===");
    println!("  V = {n}, F = {}, chi = {chi} (expected {chi_expect}), h = {h:.4}", tris.len());
    println!("  Gauss-Bonnet: int K dA = {total:.6}, 2 pi chi = {want:.6}, error {:.2e}",
             (total - want).abs());
    println!("  raw angle defect sum = {raw:.6} (combinatorial, must equal 2 pi chi), \
              error {:.2e}", (raw - want).abs());
    println!("  vertices with non-positive dual area: {bad} of {n}");
    // 2 pi V - pi F is the identity the raw sum must satisfy. When it does not,
    // the triangles are not all honest triangles.
    let ident = std::f64::consts::TAU * n as f64 - std::f64::consts::PI * tris.len() as f64;
    let degen = tris.iter().filter(|t| t[0] == t[1] || t[1] == t[2] || t[0] == t[2]).count();
    let tiny = tris
        .iter()
        .filter(|t| {
            let (p, q, r) = (coords[t[0]], coords[t[1]], coords[t[2]]);
            let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
            let c = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() < 1e-14
        })
        .count();
    let mut used = vec![false; n];
    for t in tris {
        for &v in t {
            used[v] = true;
        }
    }
    let orphan = used.iter().filter(|&&u| !u).count();
    println!("  2 pi V - pi F = {ident:.6}; repeated-vertex tris {degen}, \
              zero-area tris {tiny}, unused vertices {orphan}");
    print!("  Delta + 2K, six smallest |lambda|:");
    for v in &low {
        print!(" {:.3e}", v.abs());
    }
    println!();
    println!("  h^2 = {:.3e}; solver's three-candidate count: {}", h * h, a.kernel_dimension());
    // Whether the triangles are good enough for the DEC to be meaningful:
    // non-negative cotangent weights, and dual cells that are not inverted.
    let q = cartan_dec::mesh_quality::quality_report(mesh_ref, manifold_ref);
    println!("  quality: min angle {:.2} deg, max {:.2} deg, non-Delaunay edges {}",
             q.min_angle.to_degrees(), q.max_angle.to_degrees(), q.non_delaunay_edges);

    // Reproduce the heuristic's own numbers. It divides the response of a
    // linear coordinate by the response of a PSEUDO-RANDOM probe. A random
    // vector is all high frequency, so its response grows like h^-2 while a
    // smooth field's does not, and the ratio falls under any fixed threshold
    // once the mesh is fine enough, kernel or no kernel.
    let m = a.mass();
    let mdot = |x: &[f64], y: &[f64]| -> f64 {
        x.iter().zip(y).zip(m).map(|((p, q), w)| p * q * w).sum::<f64>()
    };
    let mnorm = |x: &[f64]| mdot(x, x).sqrt();
    let probe: Vec<f64> = (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) % 1000) as f64 / 500.0 - 1.0)
        .collect();
    let pscale = mnorm(&a.apply_operator(&probe)) / mnorm(&probe);
    print!("  heuristic: random-probe scale {pscale:.3e}, per-axis ratio");
    for axis in 0..3 {
        let v: Vec<f64> = coords.iter().map(|c| c[axis]).collect();
        print!(" {:.2e}", mnorm(&a.apply_operator(&v)) / (pscale * mnorm(&v)));
    }
    println!("  (kept when < 1e-3)");

    // The proposed replacement: the SAME field's Rayleigh quotient under the
    // shifted operator against its quotient under the plain one. Both are
    // smooth-field quantities at the same frequency, so the ratio vanishes like
    // h^2 for a Killing direction and stays O(1) for anything else.
    print!("  proposed:  |rho_shift| / rho_plain per axis");
    for axis in 0..3 {
        let v: Vec<f64> = coords.iter().map(|c| c[axis]).collect();
        let mm = mdot(&v, &v);
        let rs = mdot(&v, &a.apply_operator(&v)) / mm;
        let rp = mdot(&v, &a.apply_unshifted(&v)) / mm;
        print!(" {:.2e}", rs.abs() / rp.abs().max(1e-300));
    }
    println!();
    // The residual form, which a sign cancellation cannot fake: a norm in the
    // numerator instead of a quadratic form.
    print!("  residual:  ||A v|| / (rho_plain ||v||)      ");
    for axis in 0..3 {
        let v: Vec<f64> = coords.iter().map(|c| c[axis]).collect();
        let mm = mdot(&v, &v);
        let rp = mdot(&v, &a.apply_unshifted(&v)) / mm;
        print!(" {:.2e}", mnorm(&a.apply_operator(&v)) / (rp.abs().max(1e-300) * mnorm(&v)));
    }
    println!();
}

fn main() {
    println!("The shifted Stokes operator across Euler characteristics.");
    println!("Delta + 2K annihilates the Killing fields. A sphere has three");
    println!("rotations, so three eigenvalues fall to zero as the mesh refines.");
    println!("A genus-2 surface has NO continuous isometry, so nothing should.");

    for level in [1usize, 2, 3] {
        let mesh = icosphere(level);
        let dom = DecDomain::new(mesh, Sphere::<3>).unwrap();
        let coords = extract_coords(&dom.mesh);
        let tris: Vec<[usize; 3]> = dom.mesh.simplices.iter().map(|s| [s[0], s[1], s[2]]).collect();
        probe(&format!("sphere, icosphere level {level}"), &coords, &tris, &dom.ops, &dom.mesh, &Sphere::<3>, 2);
    }

    // The genus-2 tube, with the resolution chosen to land near the sphere
    // meshes above so the comparison is at a like scale.
    for n in [26usize, 30] {
        let r = 0.24;
        let f = move |p: [f64; 3]| genus2(p, r);
        let (mut verts, mut tris) = marching_tets(&f, [-1.5, -1.0, -0.6], [1.5, 1.0, 0.6], n);
        // Smoothing spreads the slivers and flipping restores the Delaunay
        // condition the cotangent weights need. Each helps the other: a flip
        // opens room for the next smoothing pass, and smoothing makes the next
        // flip worth taking. Neither changes the topology.
        let eu = cartan_manifolds::euclidean::Euclidean::<3>;
        for _ in 0..6 {
            smooth(&f, &mut verts, &tris, 12, 0.6, 1e-6);
            let sv: Vec<nalgebra::SVector<f64, 3>> =
                verts.iter().map(|p| nalgebra::SVector::from(*p)).collect();
            let m = cartan_dec::mesh::Mesh::from_simplices(&eu, sv, tris.clone());
            let m = cartan_dec::mesh_quality::make_delaunay(m, &eu);
            tris = m.simplices.iter().map(|t| [t[0], t[1], t[2]]).collect();
        }
        let (closed, bnd) = orient(&mut tris);
        if !closed {
            println!("\n=== genus-2 tube, grid {n} ===");
            println!("  NOT CLOSED: {bnd} edge(s) without two faces; skipping");
            continue;
        }
        let sv: Vec<nalgebra::SVector<f64, 3>> =
            verts.iter().map(|p| nalgebra::SVector::from([p[0], p[1], p[2]])).collect();
        let mesh = cartan_dec::mesh::Mesh::from_simplices(
            &cartan_manifolds::euclidean::Euclidean::<3>,
            sv,
            tris.clone(),
        );
        match DecDomain::new(mesh, cartan_manifolds::euclidean::Euclidean::<3>) {
            Ok(dom) => {
                let coords = extract_coords(&dom.mesh);
                probe(&format!("genus-2 tube, grid {n}"), &coords, &tris, &dom.ops, &dom.mesh, &cartan_manifolds::euclidean::Euclidean::<3>, -2);
            }
            Err(e) => println!("\n=== genus-2 tube, grid {n} ===\n  DEC assembly failed: {e:?}"),
        }
    }
}
