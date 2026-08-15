//! Differential harness: every kernel against its CPU counterpart, on the same
//! input, on the boundary the braid runs actually use.
//!
//! Correctness comes before speed here as it does in `volterra-cuda`: nothing
//! is timed until the matching check has passed.
//!
//! Phases (first CLI argument selects one; default `all`):
//!
//! - `validate`: each operator against `volterra_fd2d::ops`, on a random field
//!   over the 100x100 steady-winding circle, reporting the worst elementwise
//!   difference over the whole grid.
//! - `all`: every phase above, in order.
//!
//! ```text
//! cargo oxide build --arch sm_120a && ./target/release/volterra-fd2d-cuda validate
//! ```

use rand::{rngs::StdRng, RngExt, SeedableRng};

use volterra_fd2d::boundary;
use volterra_fd2d::ops;
use volterra_fd2d_cuda::{Device, DeviceBoundary, DeviceState, StepParams};

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

/// As [`compare`], but each element's difference is measured against a scale
/// the caller supplies rather than against the result.
///
/// For an expression that cancels, the result is much smaller than the terms
/// that formed it, and rounding in those terms carries through undiminished. A
/// difference counted in ulps of the result then reports the cancellation
/// rather than the agreement. `Pi_A = 2 (Q0 H1 - H0 Q1)` is the case here: the
/// bulk part of `H` is parallel to `Q` and cancels out of the cross product
/// exactly, so what survives is the elastic part alone, tens of times smaller
/// than either product. Passing `|Q0 H1| + |H0 Q1|` measures the two codes
/// against the arithmetic they actually performed.
fn compare_scaled(a: &[f64], b: &[f64], scale: &[f64]) -> Gap {
    let mut abs = 0.0_f64;
    let mut ulps = 0.0_f64;
    let mut differing = 0usize;
    for ((x, y), s) in a.iter().zip(b.iter()).zip(scale.iter()) {
        let d = (x - y).abs();
        if d > 0.0 {
            differing += 1;
        }
        abs = abs.max(d);
        let m = s.abs().max(1.0);
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

/// `volterra_fd2d::ops::upwind_advective_term` with each accumulation written
/// as a fused multiply-add.
///
/// Exists to identify the device's arithmetic rather than to be used: if the
/// device matches this and not the plain CPU form, the difference between them
/// is FMA contraction and nothing else.
fn upwind_reference_fma(
    u: &[f64],
    arr: &[f64],
    out: &mut [f64],
    bnd: &volterra_fd2d::boundary::Boundary,
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

    // The nematic kernels, on parameters the golden run actually uses.
    let params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50).with_net_charge(1.5);

    {
        // Isolate the fused Laplacian inside `h_s_from_q` from everything else
        // it does: with the bulk coefficients and the flow-alignment parameter
        // all zero, `H` is exactly `K lap Q` and must equal what the standalone
        // Laplacian kernel produced, which was bit for bit.
        let mut h_cpu = seed_vector.clone();
        ops::laplacian_vector(&vector, &mut h_cpu, &bnd, params.k_elastic);
        let mut s_scratch = vec![0.0; n * 2];
        let (h_gpu, _) = dev.h_s_from_q(
            &velocity,
            &vector,
            &d_bnd,
            0.0,
            0.0,
            params.k_elastic,
            0.0,
            &seed_vector,
            &s_scratch,
        )?;
        s_scratch.clear();
        let g = compare(&h_cpu, &h_gpu);
        all_ok &= report("h_s_from_q with the bulk term off, against laplacian_vector", &g, n * 2, 0.0);
    }

    // A smooth Q, which is what a run carries. The random field used above is
    // the worst case for a 9-point stencil: neighbouring values are
    // uncorrelated, so the -20/4/1 weights cancel almost entirely and the
    // result lands orders of magnitude below the terms that formed it, which
    // amplifies any last-bit difference in those terms. At `K = 16384` that
    // shows as a hundred ulp on a kernel whose own Laplacian is bit for bit,
    // which the isolation check above establishes. A smooth field cancels no
    // worse than the physics does.
    // The two components carry different frequencies on purpose. A single
    // phase in both makes `Q` a plane wave, and the Laplacian of a plane wave
    // is a multiple of itself, so `H` comes out parallel to `Q` and
    // `Pi_A = 2 (Q0 H1 - H0 Q1)` cancels to nothing. Comparing two codes on a
    // quantity that is identically zero measures their rounding and not their
    // agreement.
    let smooth: Vec<f64> = (0..n * 2)
        .map(|i| {
            let cell = i / 2;
            let (x, y) = ((cell / LX) as f64, (cell % LX) as f64);
            if i % 2 == 0 {
                0.4 * (0.10 * x + 0.07 * y).cos()
            } else {
                0.3 * (0.05 * x - 0.09 * y).sin()
            }
        })
        .collect();
    // A smooth velocity, for the same reason Q is smooth: the viscous term is
    // `nu lap u` at `nu = sqrt(10 K)` near 405, and a Laplacian of uncorrelated
    // noise cancels to far below the terms that formed it.
    let smooth_u: Vec<f64> = (0..n * 2)
        .map(|i| {
            let cell = i / 2;
            let (x, y) = ((cell / LX) as f64, (cell % LX) as f64);
            if i % 2 == 0 {
                0.02 * (0.08 * x - 0.05 * y).sin()
            } else {
                0.02 * (0.06 * x + 0.11 * y).cos()
            }
        })
        .collect();
    {
        let mut h_cpu = vec![0.0; n * 2];
        let mut s_cpu = vec![0.0; n * 2];
        volterra_fd2d::nematic::h_s_from_q(
            &velocity, &smooth, &mut h_cpu, &mut s_cpu,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda, &bnd,
        );
        let (h_gpu, _) = dev.h_s_from_q(
            &velocity, &smooth, &d_bnd,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda,
            &vec![0.0; n * 2], &vec![0.0; n * 2],
        )?;
        let g = compare(&h_cpu, &h_gpu);
        all_ok &= report("h_s_from_q: H on a smooth Q", &g, n * 2, 4.0);
    }
    {
        // The stresses, on the same smooth Q and the H it produces, so the
        // cancellations are the ones a run meets.
        let mut h_cpu = vec![0.0; n * 2];
        let mut s_cpu = vec![0.0; n * 2];
        volterra_fd2d::nematic::h_s_from_q(
            &velocity, &smooth, &mut h_cpu, &mut s_cpu,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda, &bnd,
        );
        let (h_gpu, s_gpu) = dev.h_s_from_q(
            &velocity, &smooth, &d_bnd,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda,
            &vec![0.0; n * 2], &vec![0.0; n * 2],
        )?;
        all_ok &= report("h_s_from_q: S", &compare(&s_cpu, &s_gpu), n * 2, 4.0);
        let _ = h_gpu;

        let mut pi_s_cpu = vec![0.0; n * 2];
        let mut pi_a_cpu = vec![0.0; n];
        volterra_fd2d::nematic::calculate_pi(
            &mut pi_s_cpu, &mut pi_a_cpu, &h_cpu, &smooth,
            params.lambda, params.zeta, params.k_elastic, &bnd,
        );
        let (pi_s_gpu, pi_a_gpu) = dev.calculate_pi(
            &h_cpu, &smooth, &d_bnd, params.lambda, params.zeta, params.k_elastic,
        )?;
        all_ok &= report("calculate_pi: Pi_S", &compare(&pi_s_cpu, &pi_s_gpu), n * 2, 4.0);
        // Pi_A cancels by construction, so it is measured against the size of
        // the two products it differences.
        let scale: Vec<f64> = (0..n)
            .map(|c| {
                let (q0, q1) = (smooth[c * 2], smooth[c * 2 + 1]);
                let (h0, h1) = (h_cpu[c * 2], h_cpu[c * 2 + 1]);
                2.0 * ((q0 * h1).abs() + (h0 * q1).abs())
            })
            .collect();
        all_ok &= report(
            "calculate_pi: Pi_A, against the scale of its two products",
            &compare_scaled(&pi_a_cpu, &pi_a_gpu, &scale),
            n,
            4.0,
        );
    }

    // The Stokes tranche, on the smooth Q and the fields it produces.
    {
        let mut h = vec![0.0; n * 2];
        let mut s = vec![0.0; n * 2];
        volterra_fd2d::nematic::h_s_from_q(
            &velocity, &smooth, &mut h, &mut s,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda, &bnd,
        );
        let mut pi_s = vec![0.0; n * 2];
        let mut pi_a = vec![0.0; n];
        volterra_fd2d::nematic::calculate_pi(
            &mut pi_s, &mut pi_a, &h, &smooth,
            params.lambda, params.zeta, params.k_elastic, &bnd,
        );

        {
            let rhs_seed = random_scalar(n, &mut rng);
            let mut cpu = rhs_seed.clone();
            volterra_fd2d::stokes::calculate_pressure_terms(
                &velocity, params.rho, &pi_s, &mut cpu, &bnd,
            );
            let gpu = dev.pressure_terms(&velocity, &pi_s, &d_bnd, params.rho, &rhs_seed)?;
            all_ok &= report("pressure_terms", &compare(&cpu, &gpu), n, 4.0);
        }

        {
            let p_aux = random_scalar(n, &mut rng);
            let rhs = random_scalar(n, &mut rng);
            let p_seed = random_scalar(n, &mut rng);
            let mut cpu = p_seed.clone();
            volterra_fd2d::stokes::relax_pressure_inner_loop(&mut cpu, &p_aux, &rhs, &bnd);
            let gpu = dev.jacobi_sweep(&p_aux, &rhs, &d_bnd, &p_seed)?;
            all_ok &= report("jacobi_sweep", &compare(&cpu, &gpu), n, 0.0);
        }

        {
            let p = random_scalar(n, &mut rng);
            let dudt_seed = random_scalar(n * 2, &mut rng);
            let mut cpu = dudt_seed.clone();
            volterra_fd2d::stokes::get_u_update(
                &mut cpu, &smooth_u, &p, params.rho, &pi_s, &pi_a, params.eta, &bnd,
            );
            let gpu = dev.u_update(
                &smooth_u, &p, &pi_s, &pi_a, &d_bnd, params.rho, params.eta, &dudt_seed,
            )?;
            all_ok &= report("u_update", &compare(&cpu, &gpu), n * 2, 4.0);
        }
    }

    // The boundary conditions, which need the layers and normals, not just the
    // interior mask.
    {
        let d_full = DeviceBoundary::upload_full(dev.stream(), &bnd)?;

        {
            let mut cpu = velocity.clone();
            volterra_fd2d::bc::apply_u_boundary_conditions(&mut cpu, &bnd);
            let gpu = dev.apply_u_bc(&d_full, &velocity)?;
            all_ok &= report("apply_u_bc", &compare(&cpu, &gpu), n * 2, 0.0);
        }

        for charge in [1.0_f64, 1.5, 2.0] {
            let mut cpu = smooth.clone();
            volterra_fd2d::bc::apply_q_boundary_conditions(&mut cpu, &bnd, params.s0, charge);
            let gpu = dev.apply_q_bc(&d_full, params.s0, charge, &smooth)?;
            // The anchoring angle runs through acos and then cos and sin. The
            // device's transcendentals are not the host libm's, and the two
            // round differently on a minority of inputs, which is why a few
            // hundred of the boundary cells differ while the rest are exact.
            all_ok &= report(
                &format!("apply_q_bc net_charge={charge}"),
                &compare(&cpu, &gpu),
                n * 2,
                16.0,
            );
        }

        {
            let mut h = vec![0.0; n * 2];
            let mut s = vec![0.0; n * 2];
            volterra_fd2d::nematic::h_s_from_q(
                &velocity, &smooth, &mut h, &mut s,
                params.a_landau, params.c_landau, params.k_elastic, params.lambda, &bnd,
            );
            let h_seed = h.clone();
            let mut cpu = h.clone();
            volterra_fd2d::bc::apply_h_boundary_conditions(
                &mut cpu, params.gamma, &smooth, &velocity, &s, &bnd,
            );
            let gpu = dev.apply_h_bc(&smooth, &velocity, &s, &d_full, params.gamma, &h_seed)?;
            all_ok &= report("apply_h_bc", &compare(&cpu, &gpu), n * 2, 4.0);

            let mut pi_s = vec![0.0; n * 2];
            let mut pi_a = vec![0.0; n];
            volterra_fd2d::nematic::calculate_pi(
                &mut pi_s, &mut pi_a, &h, &smooth,
                params.lambda, params.zeta, params.k_elastic, &bnd,
            );
            let p_aux = random_scalar(n, &mut rng);
            let p_seed = random_scalar(n, &mut rng);
            let mut cpu_p = p_seed.clone();
            volterra_fd2d::bc::apply_p_boundary_conditions(
                &mut cpu_p, &p_aux, &smooth_u, params.rho, params.eta, &pi_s, &pi_a, &bnd,
            );
            let gpu_p = dev.apply_p_bc(
                &p_aux, &smooth_u, &pi_s, &pi_a, &d_full, params.rho, params.eta, &p_seed,
            )?;
            all_ok &= report("apply_p_bc", &compare(&cpu_p, &gpu_p), n, 32.0);

            // Isolate the stencil from the stresses it differences. With no
            // stress and no flow the whole forcing term vanishes and the
            // condition reduces to the Neumann stencil on p_aux alone, which
            // carries no cancellation. If that is exact, the residual above is
            // in the stress differences, whose operands are of order 1e4 and
            // whose result is not.
            let zero_v = vec![0.0_f64; n * 2];
            let zero_s = vec![0.0_f64; n];
            let mut cpu_q = p_seed.clone();
            volterra_fd2d::bc::apply_p_boundary_conditions(
                &mut cpu_q, &p_aux, &zero_v, params.rho, params.eta, &zero_v, &zero_s, &bnd,
            );
            let gpu_q = dev.apply_p_bc(
                &p_aux, &zero_v, &zero_v, &zero_s, &d_full, params.rho, params.eta, &p_seed,
            )?;
            // A division cannot come out bit for bit when the compiler is free
            // to contract the multiplies feeding it, so half an ulp is the
            // floor here, against the full condition's 18.75.
            all_ok &= report(
                "apply_p_bc with no stress and no flow",
                &compare(&cpu_q, &gpu_q),
                n,
                1.0,
            );
        }
    }

    // The Q update and the integrator, which close the stage list.
    {
        let mut h = vec![0.0; n * 2];
        let mut s = vec![0.0; n * 2];
        volterra_fd2d::nematic::h_s_from_q(
            &smooth_u, &smooth, &mut h, &mut s,
            params.a_landau, params.c_landau, params.k_elastic, params.lambda, &bnd,
        );
        let mut cpu = vec![0.0; n * 2];
        volterra_fd2d::step::get_q_update(&mut cpu, &smooth, &h, &s, &smooth_u, params.gamma, &bnd);
        let gpu = dev.q_update(&smooth, &h, &s, &smooth_u, &d_bnd, params.gamma)?;
        all_ok &= report("q_update", &compare(&cpu, &gpu), n * 2, 4.0);

        let dt = params.dt;
        let mut cpu_i = smooth.clone();
        for i in 0..n * 2 {
            cpu_i[i] += dt * cpu[i];
        }
        let gpu_i = dev.integrate(&smooth, &cpu, dt)?;
        all_ok &= report("integrate", &compare(&cpu_i, &gpu_i), n * 2, 1.0);
    }

    if all_ok {
        println!("ALL VALIDATION PASSED");
        Ok(())
    } else {
        Err("a kernel disagreed with its CPU counterpart".into())
    }
}

/// Whole steps on the device against whole steps on the CPU, from the golden
/// run's own initial condition.
///
/// The per-kernel checks say each stage computes what the CPU computes. This
/// says the stages are wired together in the same order, and it is the check
/// that would catch a field read before it was written or a boundary condition
/// applied at the wrong point.
fn phase_step(steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new(0)?;
    let bnd = boundary::circular_boundary(LX, LX);
    let n = LX * LX;
    let d_bnd = DeviceBoundary::upload_full(dev.stream(), &bnd)?;

    let params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50).with_net_charge(1.5);
    let step_params = StepParams::from_cpu(&params);
    let target_rel_change = 1e-3;

    // The golden run's own initial condition, on both sides.
    let mut cpu = volterra_fd2d::step::State::new(LX, LX);
    let mut rng = StdRng::seed_from_u64(0);
    random_theta_ic(&mut cpu.q, params.s0, LX, LX, &bnd.inside, &mut rng);

    let mut gpu = DeviceState::zeroed(dev.stream(), LX, LX)?;
    gpu.upload_from(dev.stream(), &cpu.q, &cpu.u, &cpu.p)?;

    println!("{steps} steps at {LX}x{LX}, golden parameters, from the same initial condition");
    let mut sweeps_differ = 0usize;
    for step in 0..steps {
        let cpu_iters =
            volterra_fd2d::step::update_step_inner(&mut cpu, &params, &bnd, target_rel_change);
        let gpu_iters = dev.step(&mut gpu, &d_bnd, &step_params, target_rel_change)?;
        if cpu_iters != gpu_iters {
            sweeps_differ += 1;
            if sweeps_differ <= 3 {
                println!("  step {step}: {cpu_iters} sweeps on the CPU, {gpu_iters} on the device");
            }
        }
    }

    let (q, u, p) = gpu.download(dev.stream())?;
    println!(
        "sweep counts differed on {sweeps_differ} of {steps} steps"
    );
    // A whole field is compared against its own largest value, not element by
    // element: an element near zero carries no information about whether two
    // trajectories have parted, and the fields here span many orders.
    let relative = |a: &[f64], b: &[f64]| -> (f64, f64) {
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
        let worst = a
            .iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        (worst, worst / scale)
    };
    let mut ok = true;
    for (name, a, b) in [
        ("Q", &cpu.q[..], &q[..]),
        ("u", &cpu.u[..], &u[..]),
        ("p", &cpu.p[..], &p[..]),
    ] {
        let (worst, rel) = relative(a, b);
        let pass = rel <= 1e-9;
        ok &= pass;
        println!(
            "[step: {name}] max abs {worst:.3e}, {rel:.3e} of the field's own range ({})",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    if ok {
        println!("STEP COMPARISON COMPLETE");
        Ok(())
    } else {
        Err("a field drifted further than a stepped comparison should".into())
    }
}

/// The golden run's initial condition, matching `fd2d`'s `random_theta_ic`.
fn random_theta_ic(q: &mut [f64], s0: f64, lx: usize, ly: usize, inside: &[bool], rng: &mut StdRng) {
    use std::f64::consts::PI;
    for x in 0..lx {
        for y in 0..ly {
            let idx = x * ly + y;
            let (qxx, qxy) = if inside[idx] {
                let theta: f64 = PI * rng.random::<f64>();
                let (c, s) = (theta.cos(), theta.sin());
                (s0 * (c * c - 0.5), s0 * (c * s))
            } else {
                (0.0, 0.0)
            };
            q[idx * 2] = qxx;
            q[idx * 2 + 1] = qxy;
        }
    }
}

/// Wall clock for the same run on each side.
///
/// The CPU side is `volterra_fd2d` as the braid runs use it, which at 100x100
/// is the serial path: `par_gate::PAR_THRESHOLD` is 250,000 cells and the grid
/// is 10,000, so rayon never engages.
fn phase_time(steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new(0)?;
    let bnd = boundary::circular_boundary(LX, LX);
    let d_bnd = DeviceBoundary::upload_full(dev.stream(), &bnd)?;
    let params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50).with_net_charge(1.5);
    let step_params = StepParams::from_cpu(&params);
    let target = 1e-3;

    let fresh = || {
        let mut st = volterra_fd2d::step::State::new(LX, LX);
        let mut rng = StdRng::seed_from_u64(0);
        random_theta_ic(&mut st.q, params.s0, LX, LX, &bnd.inside, &mut rng);
        st
    };

    // Warm up both, so neither pays a first-touch cost inside the timing.
    {
        let mut w = fresh();
        for _ in 0..64 {
            volterra_fd2d::step::update_step_inner(&mut w, &params, &bnd, target);
        }
        let mut g = DeviceState::zeroed(dev.stream(), LX, LX)?;
        g.upload_from(dev.stream(), &w.q, &w.u, &w.p)?;
        for _ in 0..64 {
            dev.step(&mut g, &d_bnd, &step_params, target)?;
        }
    }

    let mut cpu = fresh();
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        volterra_fd2d::step::update_step_inner(&mut cpu, &params, &bnd, target);
    }
    let cpu_s = t0.elapsed().as_secs_f64();

    let mut gpu_times = Vec::new();
    for fixed in [None, Some(1usize)] {
        let mut sp = step_params;
        sp.fixed_sweeps = fixed;
        let start = fresh();
        let mut gpu = DeviceState::zeroed(dev.stream(), LX, LX)?;
        gpu.upload_from(dev.stream(), &start.q, &start.u, &start.p)?;
        // Warm the queue for this mode before timing it.
        for _ in 0..64 {
            dev.step(&mut gpu, &d_bnd, &sp, target)?;
        }
        let _ = gpu.download(dev.stream())?;
        let t1 = std::time::Instant::now();
        for _ in 0..steps {
            dev.step(&mut gpu, &d_bnd, &sp, target)?;
        }
        // Reading a field back forces the queue to drain, so the timing covers
        // the work rather than the launches that queued it.
        let _ = gpu.download(dev.stream())?;
        gpu_times.push(t1.elapsed().as_secs_f64());
    }
    let gpu_s = gpu_times[0];
    let gpu_fixed_s = gpu_times[1];

    println!("{steps} steps at {LX}x{LX}, golden parameters");
    println!(
        "  CPU  {cpu_s:.3} s total, {:.1} us/step",
        cpu_s / steps as f64 * 1e6
    );
    println!(
        "  GPU, adaptive sweeps   {gpu_s:.3} s total, {:.1} us/step, {:.2}x",
        gpu_s / steps as f64 * 1e6,
        cpu_s / gpu_s
    );
    println!(
        "  GPU, one fixed sweep   {gpu_fixed_s:.3} s total, {:.1} us/step, {:.2}x",
        gpu_fixed_s / steps as f64 * 1e6,
        cpu_s / gpu_fixed_s
    );
    println!(
        "  the convergence readback costs {:.1} us/step, the only sync in a step",
        (gpu_s - gpu_fixed_s) / steps as f64 * 1e6
    );
    Ok(())
}

/// Many configurations at once, one stream each.
///
/// This is the shape the work actually has. One 100x100 grid is 10,000 cells
/// and leaves most of the device idle, so a single run is bound by launch
/// latency rather than by arithmetic. A sweep over `q` and seeds has as many
/// independent runs as the sweep is wide, and they share nothing but the
/// boundary, so they can be in flight together.
fn phase_batch(runs: usize, steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new(0)?;
    let bnd = boundary::circular_boundary(LX, LX);
    let d_bnd = DeviceBoundary::upload_full(dev.stream(), &bnd)?;
    let target = 1e-3;

    // A sweep over the winding, which is what the paper's own regime map is
    // drawn against, cycled so the batch is as wide as asked for.
    let charges = [1.0_f64, 1.5, 2.0, 2.5];

    let mut states = Vec::with_capacity(runs);
    let mut params = Vec::with_capacity(runs);
    for r in 0..runs {
        let charge = charges[r % charges.len()];
        let cpu_params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50)
            .with_net_charge(charge);
        let mut sp = StepParams::from_cpu(&cpu_params);
        sp.fixed_sweeps = Some(1);
        let mut init = volterra_fd2d::step::State::new(LX, LX);
        let mut rng = StdRng::seed_from_u64(r as u64);
        random_theta_ic(&mut init.q, cpu_params.s0, LX, LX, &bnd.inside, &mut rng);

        let stream = dev.new_stream()?;
        let mut st = DeviceState::zeroed(&stream, LX, LX)?;
        st.upload_from(&stream, &init.q, &init.u, &init.p)?;
        states.push(st);
        params.push(sp);
    }

    // Warm every stream before timing any of them.
    for (st, sp) in states.iter_mut().zip(&params) {
        for _ in 0..32 {
            dev.step(st, &d_bnd, sp, target)?;
        }
    }
    for st in &states {
        let _ = st.download(&st.stream.clone())?;
    }

    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        for (st, sp) in states.iter_mut().zip(&params) {
            dev.step(st, &d_bnd, sp, target)?;
        }
    }
    for st in &states {
        let _ = st.download(&st.stream.clone())?;
    }
    let wall = t0.elapsed().as_secs_f64();

    let run_steps = (runs * steps) as f64;
    println!(
        "{runs} runs x {steps} steps at {LX}x{LX}"
    );
    println!(
        "  GPU, one stream each   {wall:.3} s, {:.1} us per run-step, {:.0} run-step/s",
        wall / run_steps * 1e6,
        run_steps / wall
    );

    // The comparison a sweep deserves. `volterra_fd2d` is single threaded at
    // this grid size, so a sweep on the CPU runs its configurations in parallel
    // processes across the cores rather than one after another. Comparing a
    // batched device against one core would flatter the device by the core
    // count.
    let t1 = std::time::Instant::now();
    std::thread::scope(|scope| {
        for r in 0..runs {
            let bnd = &bnd;
            scope.spawn(move || {
                let charge = charges[r % charges.len()];
                let cpu_params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50)
                    .with_net_charge(charge);
                let mut st = volterra_fd2d::step::State::new(LX, LX);
                let mut rng = StdRng::seed_from_u64(r as u64);
                random_theta_ic(&mut st.q, cpu_params.s0, LX, LX, &bnd.inside, &mut rng);
                for _ in 0..steps {
                    volterra_fd2d::step::update_step_inner(&mut st, &cpu_params, bnd, target);
                }
            });
        }
    });
    let cpu_wall = t1.elapsed().as_secs_f64();
    println!(
        "  CPU, one thread each   {cpu_wall:.3} s, {:.1} us per run-step, {:.0} run-step/s",
        cpu_wall / run_steps * 1e6,
        run_steps / cpu_wall
    );
    println!("  {:.2}x", cpu_wall / wall);
    Ok(())
}

/// The golden trajectory end to end on the device, writing the same frames
/// `fd2d` writes, so the published braid can be extracted from them.
///
/// The topology is what the paper claims and what the CPU run reproduces. A
/// device that agrees to `1e-15` for five thousand steps says nothing on its
/// own about a run a hundred and fifty times longer through a chaotic regime;
/// only extracting the braid does.
fn phase_golden(
    steps: usize,
    save_every: usize,
    charge: f64,
    out: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let dev = Device::new(0)?;
    let bnd = boundary::circular_boundary(LX, LX);
    let d_bnd = DeviceBoundary::upload_full(dev.stream(), &bnd)?;
    let params = volterra_fd2d::Params::new(LX, 3.99, 0.975, 1.0, 1e-4, 50).with_net_charge(charge);
    let step_params = StepParams::from_cpu(&params);
    let target = 1e-3;

    let dir = format!("{out}/Q");
    std::fs::create_dir_all(&dir)?;

    let mut init = volterra_fd2d::step::State::new(LX, LX);
    let mut rng = StdRng::seed_from_u64(0);
    random_theta_ic(&mut init.q, params.s0, LX, LX, &bnd.inside, &mut rng);

    let stream = dev.stream().clone();
    let mut st = DeviceState::zeroed(&stream, LX, LX)?;
    st.upload_from(&stream, &init.q, &init.u, &init.p)?;

    println!("golden trajectory on the device: {steps} steps, q={charge}, saving every {save_every}");
    let t0 = std::time::Instant::now();
    let mut saved = 0usize;
    for step in 0..=steps {
        if step % save_every == 0 {
            let (q, _, _) = st.download(&stream)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(format!(
                "{dir}/Q_{step:08}.txt"
            ))?);
            for c in 0..LX * LX {
                writeln!(f, "{:.17e} {:.17e}", q[c * 2], q[c * 2 + 1])?;
            }
            saved += 1;
        }
        if step < steps {
            dev.step(&mut st, &d_bnd, &step_params, target)?;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "  {steps} steps in {wall:.1} s, {:.1} us/step, {saved} frames under {dir}",
        wall / steps as f64 * 1e6
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("all") {
        "step" => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
            phase_step(steps)?;
        }
        "time" => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
            phase_time(steps)?;
        }
        "batch" => {
            let runs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
            let steps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);
            phase_batch(runs, steps)?;
        }
        "golden" => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(750_000);
            let every = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(750);
            let charge: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.5);
            let out = args.get(5).cloned().unwrap_or_else(|| "/tmp/fd2d-gpu-golden".into());
            phase_golden(steps, every, charge, &out)?;
        }
        "validate" => phase_validate()?,
        "all" => {
            phase_validate()?;
            phase_step(1)?;
        }
        other => {
            eprintln!("unknown phase {other}; expected validate or all");
            std::process::exit(2);
        }
    }
    Ok(())
}
