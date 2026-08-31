//! Per-vertex flow and power fields for a snapshot of a sphere run.
//!
//! Recomputes, from a stored `Q`, the fields the reference's figure panels are
//! coloured by: the velocity, its magnitude, the vorticity, the active force
//! the nematic exerts, and the local power density that force delivers to the
//! fluid.
//!
//! The power density is `P = u . f`, with `f` the active force. It is positive
//! where the nematic does work on the fluid, which the reference reports as a
//! region of positive power around every `+1/2` defect. That is a checkable
//! statement rather than a convention, and `sphere_power_is_positive_at_plus_half`
//! in the tests asserts it.
//!
//!     sphere_fields <run-dir> <step> [<step> ...]
//!
//! Writes `fields_<step>.npy` beside the snapshot: one `(n_vertices, 9)` array
//! holding `ux uy uz |u| omega fx fy fz P`.

use std::io::{Read, Write};
use std::path::Path;

use cartan_manifolds::sphere::Sphere;
use volterra_core::ActiveNematicParams;
use volterra_dec::mesh_gen::icosphere;
use volterra_dec::stokes::{SurfaceStokes, vertex_force_from_stress};
use volterra_dec::DecDomain;
use volterra_dec::QField;

fn read_q(path: &Path, nv: usize) -> std::io::Result<QField> {
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
    let want = format!("'descr': '<f8', 'fortran_order': False, 'shape': ({nv}, 2)");
    if !header.contains(&want) {
        return Err(Error::new(ErrorKind::InvalidData, format!("header is {header}")));
    }
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let mut q = QField::zeros(nv);
    for i in 0..nv {
        let o = i * 16;
        q.q1[i] = f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        q.q2[i] = f64::from_le_bytes(buf[o + 8..o + 16].try_into().unwrap());
    }
    Ok(q)
}

fn write_npy(path: &Path, rows: usize, cols: usize, data: &[f64]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    );
    let mut h = header.into_bytes();
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(b' ');
    }
    h.push(b'\n');
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(h.len() as u16).to_le_bytes())?;
    f.write_all(&h)?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("sphere_fields <run-dir> <step> [<step> ...]");
        std::process::exit(2);
    }
    let run = Path::new(&args[0]);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("meta.json")).unwrap()).unwrap();
    let refinement = meta["refinement"].as_u64().unwrap() as usize;
    let pe = meta["pe"].as_f64().unwrap();
    let zeta = meta["zeta_eff"].as_f64().unwrap();
    let k_r = meta["k_r"].as_f64().unwrap();

    let mesh = icosphere(refinement);
    let nv = mesh.n_vertices();
    let domain = DecDomain::new(mesh, Sphere::<3>).expect("DecDomain");
    let coords: Vec<[f64; 3]> =
        domain.mesh.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let stokes = SurfaceStokes::new(&domain.ops, &domain.mesh).expect("Stokes");

    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = zeta;
    params.k_r = k_r;
    params.eta = meta["eta"].as_f64().unwrap_or(1.0);
    println!("run {} at Pe = {pe}, zeta = {zeta}", run.display());

    for step_s in &args[1..] {
        let step: usize = step_s.parse().expect("step must be a number");
        let q = read_q(&run.join(format!("q_{step:06}.npy")), nv).expect("read Q");
        let (vel, psi, _) =
            stokes.solve_warm(&q, &params, &domain.ops, &domain.mesh, None, 1e-10);

        // The active stress is `-zeta Q`, so its symmetric parts are the two
        // components of `Q` scaled, and it has no antisymmetric part.
        let sym1: Vec<f64> = q.q1.iter().map(|v| -zeta * v).collect();
        let sym2: Vec<f64> = q.q2.iter().map(|v| -zeta * v).collect();
        let anti = vec![0.0; nv];
        let force = vertex_force_from_stress(
            &sym1, &sym2, &anti, &domain.mesh, &coords,
            stokes.normals(), stokes.e1_frames(),
        );

        // Vorticity from the stream function: `omega = -Delta psi`, which the
        // Laplace-Beltrami operator gives directly.
        let mut omega = vec![0.0_f64; nv];
        for (i, o) in omega.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (&v, (r, c)) in domain.ops.laplace_beltrami.iter() {
                if r == i {
                    acc += v * psi[c];
                }
            }
            *o = acc;
        }

        let mut out = Vec::with_capacity(nv * 9);
        for i in 0..nv {
            let u = vel.v[i];
            let f = force[i];
            let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let power = u[0] * f[0] + u[1] * f[1] + u[2] * f[2];
            out.extend_from_slice(&[
                u[0], u[1], u[2], speed, omega[i], f[0], f[1], f[2], power,
            ]);
        }
        let path = run.join(format!("fields_{step:06}.npy"));
        write_npy(&path, nv, 9, &out).expect("write fields");
        let l2: f64 = (0..nv).map(|i| out[i * 9 + 3].powi(2)).sum::<f64>() / nv as f64;
        let ens: f64 = (0..nv).map(|i| out[i * 9 + 4].powi(2)).sum::<f64>() / nv as f64;
        println!("  step {step}: mean |u|^2 {l2:.3e}, mean omega^2 {ens:.3e} -> {}",
                 path.file_name().unwrap().to_string_lossy());
    }
}
