//! End to end: seed a disclination loop, evolve it, measure it, export it.
//!
//! Seeds a `+1/2` loop as the initial Q field, advances the dry active nematic
//! solver, then reads the disclination lines off the density tensor and writes
//! what a renderer needs: the density `s` whose isosurface is the tube, the
//! `cos(beta)` that colours it, and the curves themselves with their geometry.
//!
//! Colouring by `cos(beta)` is what Head, Negro and co-authors do
//! (`head-2024-3d-act-nem`, Fig. 1): `+1` is a comet, the `+1/2` profile, `-1` a
//! triradius, the `-1/2` profile, and `0` a twist between them.
//!
//! Usage: `loop_demo <out_dir> [n] [radius] [steps]`

use std::path::Path;

use volterra_braid::disclination::{
    cos_beta_field, disclination_lines_at_fraction, disclination_magnitude,
};
use volterra_core::{ActiveNematicParams3D, QField3D};
use volterra_fd::runner_3d::{run_dry_active_nematic_3d, run_wet_active_nematic_3d};

/// `Q = q (nn - I/3)`.
fn uniaxial(n: [f64; 3], q: f64) -> [f64; 5] {
    let t = 1.0 / 3.0;
    [
        q * (n[0] * n[0] - t),
        q * (n[0] * n[1]),
        q * (n[0] * n[2]),
        q * (n[1] * n[1] - t),
        q * (n[1] * n[2]),
    ]
}

/// A `+1/2` disclination loop of radius `r` in the mid-plane.
///
/// Around the core the director winds by `pi` in the meridional plane, which is
/// the loop's defining texture: at meridional angle `psi` from the outward
/// radial direction the director is `cos(psi/2) e_rho + sin(psi/2) e_z`.
fn seed_loop(n: usize, r: f64, q_eq: f64) -> Vec<[f64; 5]> {
    let c = (n as f64 - 1.0) / 2.0;
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, l as f64 - c);
                let rho = (x * x + y * y).sqrt();
                let half = 0.5 * z.atan2(rho - r);
                let e = if rho > 1e-9 {
                    [x / rho, y / rho]
                } else {
                    [1.0, 0.0]
                };
                let dir = [half.cos() * e[0], half.cos() * e[1], half.sin()];
                q[((i * n) + j) * n + l] = uniaxial(dir, q_eq);
            }
        }
    }
    q
}

/// A disclination loop whose winding character varies around it.
///
/// The rotation axis is tilted away from the tangent by an angle that sweeps
/// once around the loop, `Omega(phi) = cos(phi) T + sin(phi) z`, so
/// `cos(beta) = Omega . T = cos(phi)` runs from a comet at `phi = 0` through a
/// twist at `pi/2` to a triradius at `pi` and back. That is the object Head and
/// Negro report in a double emulsion, whose loops change "back and forth from
/// +1/2 to -1/2" along their length, and it is what makes the character scale
/// visible in one picture.
///
/// The director at a nearby point is the reference director rotated about
/// `Omega` by half the meridional angle, which is the half-integer winding.
fn seed_twisted_loop(n: usize, r: f64, q_eq: f64) -> Vec<[f64; 5]> {
    let c = (n as f64 - 1.0) / 2.0;
    let mut q = vec![[0.0; 5]; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                let (x, y, z) = (i as f64 - c, j as f64 - c, l as f64 - c);
                let rho = (x * x + y * y).sqrt().max(1e-9);
                let phi = y.atan2(x);
                let e_rho = [x / rho, y / rho, 0.0];
                let tangent = [-y / rho, x / rho, 0.0];

                // The rotation axis, tilted out of the tangent by phi.
                let (cp, sp) = (phi.cos(), phi.sin());
                let omega = normalise([cp * tangent[0] + sp * 0.0, cp * tangent[1] + sp * 0.0, sp]);

                // A reference director perpendicular to the axis.
                let dot = e_rho[0] * omega[0] + e_rho[1] * omega[1] + e_rho[2] * omega[2];
                let n0 = normalise([
                    e_rho[0] - dot * omega[0],
                    e_rho[1] - dot * omega[1],
                    e_rho[2] - dot * omega[2],
                ]);

                // Half the meridional angle about the core, which is the
                // half-integer winding.
                let psi = z.atan2(rho - r);
                let dir = rodrigues(n0, omega, 0.5 * psi);
                q[((i * n) + j) * n + l] = uniaxial(dir, q_eq);
            }
        }
    }
    q
}

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / m, v[1] / m, v[2] / m]
}

/// Rotate `v` about the unit axis `k` by `angle`.
fn rodrigues(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, cth) = angle.sin_cos();
    let kv = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    let kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    normalise([
        v[0] * cth + kv[0] * s + k[0] * kd * (1.0 - cth),
        v[1] * cth + kv[1] * s + k[1] * kd * (1.0 - cth),
        v[2] * cth + kv[2] * s + k[2] * kd * (1.0 - cth),
    ])
}

fn write_npy(path: &Path, data: &[f64], n: usize) -> std::io::Result<()> {
    use std::io::Write;
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({n}, {n}, {n}), }}");
    let mut pad = header.len() + 11;
    while pad % 64 != 0 {
        pad += 1;
    }
    let header = format!("{header}{}\n", " ".repeat(pad - header.len() - 11));
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// A `(n, n, n, 3)` array, for a vector field.
fn write_vector_npy(path: &Path, data: &[[f64; 3]], n: usize) -> std::io::Result<()> {
    use std::io::Write;
    let header =
        format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({n}, {n}, {n}, 3), }}");
    let mut pad = header.len() + 11;
    while pad % 64 != 0 {
        pad += 1;
    }
    let header = format!("{header}{}\n", " ".repeat(pad - header.len() - 11));
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        for c in v {
            f.write_all(&c.to_le_bytes())?;
        }
    }
    Ok(())
}

/// The curves as JSON, with the geometry a renderer reads.
fn curves_json(
    curves: &[volterra_braid::disclination::DisclinationCurve],
    threshold: f64,
    n: usize,
    dx: f64,
) -> String {
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"threshold\": {threshold},\n"));
    json.push_str(&format!("  \"n\": {n},\n  \"dx\": {dx},\n"));
    json.push_str("  \"curves\": [\n");
    for (c, curve) in curves.iter().enumerate() {
        let pts: Vec<String> = curve
            .sites
            .iter()
            .map(|s| format!("[{},{},{}]", s.pos[0], s.pos[1], s.pos[2]))
            .collect();
        let cb: Vec<String> = curve
            .sites
            .iter()
            .map(|s| format!("{}", s.disclination.cos_beta))
            .collect();
        json.push_str(&format!(
            "    {{\"points\": [{}], \"cos_beta\": [{}], \"length\": {}, \"is_loop\": {}, \
             \"mean_cos_beta\": {}, \"mean_curvature\": {}, \"surface_mean_curvature\": {}}}{}\n",
            pts.join(","),
            cb.join(","),
            curve.length,
            curve.is_loop,
            curve.mean_cos_beta,
            curve.mean_curvature,
            curve.surface_mean_curvature,
            if c + 1 == curves.len() { "" } else { "," }
        ));
    }
    json.push_str("  ]\n}\n");
    json
}

/// Minimal reader for the `.npy` the runner writes: C-order float64.
fn read_npy_f64(path: &Path) -> std::io::Result<Vec<f64>> {
    let raw = std::fs::read(path)?;
    let header_len = if raw[6] == 1 {
        u16::from_le_bytes([raw[8], raw[9]]) as usize + 10
    } else {
        u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize + 12
    };
    Ok(raw[header_len..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("output/loop_demo"),
    );
    let n: usize = args.get(2).map_or(64, |s| s.parse().unwrap());
    let radius: f64 = args.get(3).map_or(16.0, |s| s.parse().unwrap());
    let steps: usize = args.get(4).map_or(400, |s| s.parse().unwrap());
    // "comet" seeds a loop of uniform character, "twisted" one whose character
    // sweeps the whole scale.
    let kind = args.get(5).map(String::as_str).unwrap_or("twisted");
    // "wet" solves Stokes each step and advects the texture in the flow its own
    // activity drives; "dry" solves no flow at all.
    let hydro = args.get(6).map(String::as_str).unwrap_or("wet");
    // Snapshot interval. Equal to `steps` gives one frame at the end; smaller
    // gives a series a film can be cut from.
    let snap: usize = args.get(7).map_or(steps, |s| s.parse().unwrap());
    // The elastic constant, the activity and the step. The defaults leave the
    // core and the active length under one voxel, which is what a resolved run
    // has to move away from.
    let k_r: f64 = args.get(8).map_or(1.0, |s| s.parse().unwrap());
    let zeta: f64 = args.get(9).map_or(2.0, |s| s.parse().unwrap());
    let dt: f64 = args.get(10).map_or(0.01, |s| s.parse().unwrap());
    std::fs::create_dir_all(out)?;

    let mut p = ActiveNematicParams3D::default_test();
    p.nx = n;
    p.ny = n;
    p.nz = n;
    p.k_r = k_r;
    p.zeta_eff = zeta;
    p.dt = dt;
    let q_eq = p.equilibrium_q();

    // The two lengths a disclination lives on, against the mesh. The core is
    // sqrt(K / 2|a|) and the active instability's wavelength is sqrt(K / zeta);
    // where either falls under a voxel, what breaks the loop is a lattice mode
    // rather than the physics. The explicit elastic step needs
    // dt < dx^2 / (6 Gamma K) in three dimensions.
    let xi = (p.k_r / (2.0 * p.a_landau.abs())).sqrt();
    let l_d = (p.k_r / p.zeta_eff).sqrt();
    let dt_max = p.dx * p.dx / (6.0 * p.gamma_r * p.k_r);
    println!(
        "core xi = {:.2} voxels, active length = {:.2} voxels, dt = {} against a limit of {:.4}",
        xi / p.dx,
        l_d / p.dx,
        p.dt,
        dt_max
    );
    if xi < 2.0 * p.dx || l_d < 2.0 * p.dx {
        println!("  WARNING: under two voxels, so the breakup is a mesh artefact");
    }
    if p.dt > 0.5 * dt_max {
        println!("  WARNING: dt is over half the explicit limit");
    }

    let mut field = QField3D::zeros(n, n, n, p.dx);
    field.q = match kind {
        "comet" => seed_loop(n, radius, q_eq),
        "twisted" => seed_twisted_loop(n, radius, q_eq),
        other => return Err(format!("unknown seed '{other}' (expected comet|twisted)").into()),
    };

    println!("seeded a {kind} loop of radius {radius} in a {n}^3 box, q_eq = {q_eq:.6}, {hydro}");
    let (evolved, velocity, stats) = match hydro {
        "wet" => {
            let (q, v, st) = run_wet_active_nematic_3d(&field, &p, steps, snap, out, true);
            (q, Some(v), st)
        }
        "dry" => {
            let (q, st) = run_dry_active_nematic_3d(&field, &p, steps, snap, out, true);
            (q, None, st)
        }
        other => return Err(format!("unknown mode '{other}' (expected wet|dry)").into()),
    };
    if let Some(s) = stats.last() {
        println!(
            "after {steps} steps: S = {:.4}, threshold = {:.3e}, {} lines of which {} closed, \
             fastest flow {:.4e}, mean flow {:.4e}",
            s.mean_s,
            s.disclination_threshold,
            s.n_disclination_lines,
            s.n_disclination_loops,
            s.max_speed,
            s.mean_speed
        );
    }

    // The flow, for a renderer that wants to show what moved the loop.
    if let Some(v) = &velocity {
        write_vector_npy(&out.join("velocity.npy"), &v.u, n)?;
    }

    // The two fields a renderer needs, from the same density tensor the
    // detector used, so the surface drawn is the surface measured.
    let s = disclination_magnitude(&evolved.q, n, n, n, p.dx);
    let beta = cos_beta_field(&evolved.q, n, n, n, p.dx);
    write_npy(&out.join("density.npy"), &s, n)?;
    write_npy(&out.join("cos_beta.npy"), &beta, n)?;

    let (curves, threshold) = disclination_lines_at_fraction(
        &evolved.q,
        n,
        n,
        n,
        p.dx,
        p.disclination_threshold_fraction,
        p.disclination_floor(),
    );

    std::fs::write(
        out.join("curves.json"),
        curves_json(&curves, threshold, n, p.dx),
    )?;

    for (c, curve) in curves.iter().enumerate() {
        println!(
            "  curve {c}: {} sites, length {:.2}, {}, mean cos(beta) {:+.3}, \
             line curvature {:.4}, surface curvature {:+.4}",
            curve.sites.len(),
            curve.length,
            if curve.is_loop { "closed" } else { "open" },
            curve.mean_cos_beta,
            curve.mean_curvature,
            curve.surface_mean_curvature
        );
    }
    println!(
        "wrote density.npy, cos_beta.npy and curves.json to {}",
        out.display()
    );

    // The same three products for every frame the runner wrote, so a film reads
    // the measured quantities per frame rather than interpolating between two.
    if snap < steps {
        let mut frames: Vec<_> = std::fs::read_dir(out)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|f| {
                f.extension().is_some_and(|x| x == "npy")
                    && f.file_name()
                        .is_some_and(|s| s.to_string_lossy().starts_with("q_"))
            })
            .collect();
        frames.sort();
        println!("deriving fields for {} frames", frames.len());
        for (k, frame) in frames.iter().enumerate() {
            let flat = read_npy_f64(frame)?;
            let qf: Vec<[f64; 5]> = flat
                .chunks_exact(5)
                .map(|c| [c[0], c[1], c[2], c[3], c[4]])
                .collect();
            let s = disclination_magnitude(&qf, n, n, n, p.dx);
            let b = cos_beta_field(&qf, n, n, n, p.dx);
            write_npy(&out.join(format!("density_{k:04}.npy")), &s, n)?;
            // The nematic itself, for a volume rendering: `S` is high in the
            // ordered bulk and melts at a core, so it shows the texture
            // everywhere rather than only on the isosurface.
            let mut frame_q = QField3D::zeros(n, n, n, p.dx);
            frame_q.q = qf.clone();
            write_npy(
                &out.join(format!("order_{k:04}.npy")),
                &frame_q.scalar_order_s(),
                n,
            )?;
            // The director, for rod glyphs. A nematic rod is apolar, so only the
            // axis matters and the sign the eigensolver happens to return does
            // not.
            write_vector_npy(
                &out.join(format!("director_{k:04}.npy")),
                &frame_q.director(),
                n,
            )?;
            write_npy(&out.join(format!("cos_beta_{k:04}.npy")), &b, n)?;
            let (cs, th) = disclination_lines_at_fraction(
                &qf,
                n,
                n,
                n,
                p.dx,
                p.disclination_threshold_fraction,
                p.disclination_floor(),
            );
            std::fs::write(
                out.join(format!("curves_{k:04}.json")),
                curves_json(&cs, th, n, p.dx),
            )?;
        }
        println!("wrote {} frames of derived fields", frames.len());
    }
    Ok(())
}
