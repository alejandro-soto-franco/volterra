//! Differential harness: every kernel against its CPU counterpart, on the same
//! input, on the boundary the braid runs actually use.
//!
//! Correctness comes before speed here as it does in `volterra-cuda`: nothing
//! is timed until the matching check has passed.
//!
//! Phases (first CLI argument selects one; default `all`):
//!
//! - `validate`: each operator against `volterra_cgpo::ops`, on a random field
//!   over the 100x100 steady-winding circle, reporting the worst elementwise
//!   difference over the whole grid.
//! - `all`: every phase above, in order.
//!
//! ```text
//! cargo oxide build --arch sm_120a && ./target/release/volterra-cgpo-cuda validate
//! ```

use rand::{rngs::StdRng, RngExt, SeedableRng};

use volterra_cgpo::boundary;
use volterra_cgpo::ops;
use volterra_cgpo_cuda::{Device, DeviceBoundary};

/// The grid the golden and silver runs use.
const LX: usize = 100;

fn random_scalar(n: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| rng.random::<f64>() - 0.5).collect()
}

/// How far apart two fields are: the worst absolute difference, the worst
/// difference measured in units in the last place of the larger operand, and
/// how many elements differ at all.
struct Gap {
    abs: f64,
    ulps: f64,
    differing: usize,
}

fn compare(a: &[f64], b: &[f64]) -> Gap {
    let mut abs = 0.0_f64;
    let mut ulps = 0.0_f64;
    let mut differing = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > 0.0 {
            differing += 1;
        }
        abs = abs.max(d);
        // One ulp of a double near `m` is `m * 2^-52`, so the ratio below is
        // the difference counted in ulps. Values below 1 are measured against
        // 1, which understates nothing.
        let m = x.abs().max(y.abs()).max(1.0);
        ulps = ulps.max(d / (m * f64::EPSILON));
    }
    Gap { abs, ulps, differing }
}

/// `tol_ulps` is how many units in the last place the kernel may differ by.
///
/// Zero means bit for bit, which is what an operator with no fused
/// multiply-add opportunity should manage. A small non-zero budget is for
/// kernels whose compiled PTX contracts a multiply and an add into one
/// instruction: that rounds once where the CPU rounds twice, so the device
/// result is a correctly rounded evaluation of the same expression and differs
/// in the last bit. It is not a licence for an algebra error, which shows up as
/// a large `ulps` rather than a small one.
fn report(name: &str, g: &Gap, n: usize, tol_ulps: f64) -> bool {
    let ok = g.ulps <= tol_ulps;
    println!(
        "[{name}] max abs {:.3e}, max {:.2} ulp, {}/{} elements differ ({})",
        g.abs,
        g.ulps,
        g.differing,
        n,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// `volterra_cgpo::ops::upwind_advective_term` with each accumulation written
/// as a fused multiply-add.
///
/// Exists to identify the device's arithmetic rather than to be used: if the
/// device matches this and not the plain CPU form, the difference between them
/// is FMA contraction and nothing else.
fn upwind_reference_fma(
    u: &[f64],
    arr: &[f64],
    out: &mut [f64],
    bnd: &volterra_cgpo::boundary::Boundary,
    coeff: f64,
) {
    let (lx, ly) = (bnd.lx, bnd.ly);
    let half = coeff * 0.5;
    let v = |a: usize, b: usize, c: usize| (a * ly + b) * 2 + c;
    for x in 0..lx {
        let xup = (x + 1) % lx;
        let xdn = (x + lx - 1) % lx;
        let xupup = (x + 2) % lx;
        let xdndn = (x + lx - 2) % lx;
        for y in 0..ly {
            let idx = x * ly + y;
            if !bnd.inside[idx] {
                continue;
            }
            let yup = (y + 1) % ly;
            let ydn = (y + ly - 1) % ly;
            let yupup = (y + 2) % ly;
            let ydndn = (y + ly - 2) % ly;
            let ux = u[v(x, y, 0)];
            let uy = u[v(x, y, 1)];
            let tmp_x = half * ux;
            let tmp_y = half * uy;
            for c in 0..2 {
                let e = if ux > 0.0 {
                    3.0 * arr[v(x, y, c)] - 4.0 * arr[v(xdn, y, c)] + arr[v(xdndn, y, c)]
                } else {
                    -3.0 * arr[v(x, y, c)] + 4.0 * arr[v(xup, y, c)] - arr[v(xupup, y, c)]
                };
                out[v(x, y, c)] = tmp_x.mul_add(e, out[v(x, y, c)]);
            }
            for c in 0..2 {
                let e = if uy > 0.0 {
                    3.0 * arr[v(x, y, c)] - 4.0 * arr[v(x, ydn, c)] + arr[v(x, ydndn, c)]
                } else {
                    -3.0 * arr[v(x, y, c)] + 4.0 * arr[v(x, yup, c)] - arr[v(x, yupup, c)]
                };
                out[v(x, y, c)] = tmp_y.mul_add(e, out[v(x, y, c)]);
            }
        }
    }
}

fn phase_validate() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new(0)?;
    let bnd = boundary::circular_boundary(LX, LX);
    let n = LX * LX;
    let d_bnd = DeviceBoundary::upload(dev.stream(), LX, LX, &bnd.inside)?;
    println!(
        "grid {LX}x{LX}, {} interior cells",
        bnd.inside.iter().filter(|&&b| b).count()
    );

    let mut rng = StdRng::seed_from_u64(7);
    let scalar = random_scalar(n, &mut rng);
    let vector = random_scalar(n * 2, &mut rng);
    let velocity = random_scalar(n * 2, &mut rng);
    // A non-zero seed, so a kernel that wrongly writes outside the mask, or
    // wrongly overwrites where it should accumulate, shows up as a difference.
    let seed_scalar = random_scalar(n, &mut rng);
    let seed_vector = random_scalar(n * 2, &mut rng);

    let mut all_ok = true;

    // The three operators that only read and write, with no accumulation, are
    // held to bit-for-bit agreement: they evaluate the same expressions in the
    // same order on the same doubles. The accumulating one gets a two-ulp
    // budget, because its compiled PTX contracts multiplies and adds; the
    // measurement below shows what that costs, and an FMA-written CPU
    // reference accounts for most of it.
    for coeff in [1.0_f64, 16384.0] {
        let mut cpu = seed_scalar.clone();
        ops::laplacian(&scalar, &mut cpu, &bnd, coeff);
        let gpu = dev.laplacian_scalar(&scalar, &d_bnd, coeff, &seed_scalar)?;
        let g = compare(&cpu, &gpu);
        all_ok &= report(&format!("laplacian_scalar coeff={coeff}"), &g, n, 0.0);
    }

    {
        let mut cpu = seed_vector.clone();
        ops::laplacian_vector(&vector, &mut cpu, &bnd, 1.0);
        let gpu = dev.laplacian_vector(&vector, &d_bnd, 1.0, &seed_vector)?;
        let g = compare(&cpu, &gpu);
        all_ok &= report("laplacian_vector", &g, n * 2, 0.0);
    }

    {
        let mut cpu = seed_scalar.clone();
        ops::div_vector(&vector, &mut cpu, &bnd);
        let gpu = dev.div_vector(&vector, &d_bnd, &seed_scalar)?;
        let g = compare(&cpu, &gpu);
        all_ok &= report("div_vector", &g, n, 0.0);
    }

    {
        // coeff = -1 is what `get_q_update` passes: it subtracts (u.grad)Q.
        let mut cpu = seed_vector.clone();
        ops::upwind_advective_term(&velocity, &vector, &mut cpu, &bnd, -1.0);
        let gpu = dev.upwind_advective(&velocity, &vector, &d_bnd, -1.0, &seed_vector)?;
        let g = compare(&cpu, &gpu);
        all_ok &= report("upwind_advective coeff=-1", &g, n * 2, 2.0);
    }

    {
        // Is the one-ulp gap above FMA contraction? The device is free to fuse
        // `slot += tmp * expr` into a single fused multiply-add, which rounds
        // once where the CPU's separate multiply and add round twice. If that
        // is what it is doing, a CPU reference written with `mul_add` matches
        // the device exactly, and nothing else would make it do so.
        let mut cpu_fma = seed_vector.clone();
        upwind_reference_fma(&velocity, &vector, &mut cpu_fma, &bnd, -1.0);
        let gpu = dev.upwind_advective(&velocity, &vector, &d_bnd, -1.0, &seed_vector)?;
        let g = compare(&cpu_fma, &gpu);
        all_ok &= report("upwind_advective against an FMA reference", &g, n * 2, 2.0);
    }

    {
        // A velocity of exactly zero takes the same branch on both axes on both
        // sides, which is the case a sign convention gets wrong.
        let zero = vec![0.0_f64; n * 2];
        let mut cpu = seed_vector.clone();
        ops::upwind_advective_term(&zero, &vector, &mut cpu, &bnd, -1.0);
        let gpu = dev.upwind_advective(&zero, &vector, &d_bnd, -1.0, &seed_vector)?;
        let g = compare(&cpu, &gpu);
        all_ok &= report("upwind_advective u=0", &g, n * 2, 0.0);
    }

    if all_ok {
        println!("ALL VALIDATION PASSED");
        Ok(())
    } else {
        Err("a kernel disagreed with its CPU counterpart".into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("all") {
        "validate" | "all" => phase_validate()?,
        other => {
            eprintln!("unknown phase {other}; expected validate or all");
            std::process::exit(2);
        }
    }
    Ok(())
}
