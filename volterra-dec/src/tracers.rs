//! Tracers advected by a stored surface flow.
//!
//! Shared by the measurements that need particle paths rather than fields: the
//! stretching rate and the ensemble entropy both advect points through the same
//! velocity snapshots, and a second copy of this would be a second chance for
//! the two to disagree about what the flow is.

use std::io::Read;
use std::path::Path;

pub fn norm3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-300 { [v[0] / n, v[1] / n, v[2] / n] } else { v }
}
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn geodesic(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = cross3(a, b);
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(dot3(a, b))
}

pub fn read_npy(path: &Path, rows: usize, cols: usize) -> std::io::Result<Vec<f64>> {
    use std::io::{Error, ErrorKind};
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 10];
    f.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(Error::new(ErrorKind::InvalidData, "not a .npy file"));
    }
    let hlen = u16::from_le_bytes([magic[8], magic[9]]) as usize;
    let mut header = vec![0u8; hlen];
    f.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header).to_string();
    let want = format!("'shape': ({rows}, {cols})");
    if !header.contains(&want) {
        return Err(Error::new(ErrorKind::InvalidData, format!("header is {header}")));
    }
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok((0..rows * cols)
        .map(|i| f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect())
}

/// A bucket grid over the sphere, for locating the vertex nearest a point.
pub struct Buckets {
    n_z: usize,
    n_p: usize,
    cells: Vec<Vec<usize>>,
}

impl Buckets {
    pub fn new(verts: &[[f64; 3]]) -> Self {
        let n_z = 48;
        let n_p = 96;
        let mut cells = vec![Vec::new(); n_z * n_p];
        for (i, v) in verts.iter().enumerate() {
            let (iz, ip) = Self::cell(*v, n_z, n_p);
            cells[iz * n_p + ip].push(i);
        }
        Self { n_z, n_p, cells }
    }

    fn cell(v: [f64; 3], n_z: usize, n_p: usize) -> (usize, usize) {
        let z = v[2].clamp(-1.0, 1.0);
        let iz = (((z + 1.0) * 0.5 * n_z as f64) as usize).min(n_z - 1);
        let phi = v[1].atan2(v[0]) + std::f64::consts::PI;
        let ip = ((phi / std::f64::consts::TAU * n_p as f64) as usize).min(n_p - 1);
        (iz, ip)
    }

    /// The nearest vertex, searched over the point's own cell and its ring of
    /// neighbours, widening until something is found.
    pub fn nearest(&self, p: [f64; 3], verts: &[[f64; 3]]) -> usize {
        let (iz, ip) = Self::cell(p, self.n_z, self.n_p);
        for ring in 1..6 {
            let mut best = (usize::MAX, f64::INFINITY);
            for dz in -(ring as i64)..=(ring as i64) {
                let z = iz as i64 + dz;
                if z < 0 || z >= self.n_z as i64 {
                    continue;
                }
                for dp in -(ring as i64)..=(ring as i64) {
                    let pp = (ip as i64 + dp).rem_euclid(self.n_p as i64) as usize;
                    for &i in &self.cells[z as usize * self.n_p + pp] {
                        let d = geodesic(p, verts[i]);
                        if d < best.1 {
                            best = (i, d);
                        }
                    }
                }
            }
            if best.0 != usize::MAX {
                return best.0;
            }
        }
        0
    }
}

/// The mesh a tracer moves over, borrowed as one argument.
///
/// Every function here reads the same four pieces together, and passing them
/// separately put `advect` one over clippy's argument ceiling. Grouping them
/// also makes it impossible to hand one function a vertex list and another the
/// faces of a different mesh.
#[derive(Clone, Copy)]
pub struct MeshRef<'a> {
    pub verts:      &'a [[f64; 3]],
    pub tris:       &'a [[usize; 3]],
    pub vert_faces: &'a [Vec<usize>],
    pub buckets:    &'a Buckets,
}

/// Velocity at a point, from the incident faces of the nearest vertex.
///
/// Barycentric within the containing triangle where the point falls in one,
/// and the nearest vertex's own value otherwise. Nearest-vertex alone is first
/// order in the mesh spacing, which a measurement of exponential separation
/// cannot afford.
pub fn velocity_at(p: [f64; 3], mesh: &MeshRef<'_>, u: &[f64]) -> [f64; 3] {
    let MeshRef { verts, tris, vert_faces, buckets } = *mesh;
    let v0 = buckets.nearest(p, verts);
    for &f in &vert_faces[v0] {
        let [a, b, c] = tris[f];
        let (pa, pb, pc) = (verts[a], verts[b], verts[c]);
        // Barycentric by the areas the point cuts the triangle into.
        let n = cross3(
            [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]],
            [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]],
        );
        let area = dot3(n, n).sqrt();
        if area < 1e-300 {
            continue;
        }
        let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
        let wa = dot3(cross3(sub(pb, p), sub(pc, p)), n) / (area * area);
        let wb = dot3(cross3(sub(pc, p), sub(pa, p)), n) / (area * area);
        let wc = dot3(cross3(sub(pa, p), sub(pb, p)), n) / (area * area);
        if wa >= -1e-9 && wb >= -1e-9 && wc >= -1e-9 {
            return [
                wa * u[a * 3] + wb * u[b * 3] + wc * u[c * 3],
                wa * u[a * 3 + 1] + wb * u[b * 3 + 1] + wc * u[c * 3 + 1],
                wa * u[a * 3 + 2] + wb * u[b * 3 + 2] + wc * u[c * 3 + 2],
            ];
        }
    }
    [u[v0 * 3], u[v0 * 3 + 1], u[v0 * 3 + 2]]
}

/// One midpoint step on the sphere, projected back after each move.
pub fn advect(p: [f64; 3], dt: f64, mesh: &MeshRef<'_>, u0: &[f64], u1: &[f64]) -> [f64; 3] {
    let vel = |x: [f64; 3], s: f64| -> [f64; 3] {
        let a = velocity_at(x, mesh, u0);
        let b = velocity_at(x, mesh, u1);
        // Linear in time between the two snapshots the step spans.
        let v = [
            (1.0 - s) * a[0] + s * b[0],
            (1.0 - s) * a[1] + s * b[1],
            (1.0 - s) * a[2] + s * b[2],
        ];
        // Only the tangential part moves a point on the sphere.
        let r = dot3(v, x);
        [v[0] - r * x[0], v[1] - r * x[1], v[2] - r * x[2]]
    };
    let k1 = vel(p, 0.0);
    let mid = norm3([p[0] + 0.5 * dt * k1[0], p[1] + 0.5 * dt * k1[1], p[2] + 0.5 * dt * k1[2]]);
    let k2 = vel(mid, 0.5);
    norm3([p[0] + dt * k2[0], p[1] + dt * k2[1], p[2] + dt * k2[2]])
}

