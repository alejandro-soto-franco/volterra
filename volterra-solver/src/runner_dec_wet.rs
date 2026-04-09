//! Wet active nematic runner on 2D DEC meshes.
//!
//! Solves the coupled Beris-Edwards + Stokes system on a triangulated
//! 2-manifold using operator splitting:
//!
//! 1. **Stokes**: solve for velocity from active stress sigma = -zeta Q.
//! 2. **Beris-Edwards (RK4)**: advance Q with molecular field + advection.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use volterra_core::ActiveNematicParams;
use volterra_dec::stokes_dec::{StokesSolverDec, advect_q};
use volterra_dec::{molecular_field_dec, QFieldDec};

use crate::runner_dec::SnapStatsDec;

/// Run the wet active nematic on a 2D DEC mesh.
///
/// # Returns
///
/// `Ok((q_final, stats))` on success, or an error if the Poisson
/// factorisation for the Stokes solver fails.
#[allow(clippy::too_many_arguments)]
pub fn run_wet_active_nematic_dec<M: Manifold>(
    q_init: &QFieldDec,
    params: &ActiveNematicParams,
    ops: &Operators<M, 3, 2>,
    mesh: &Mesh<M, 3, 2>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    n_steps: usize,
    snap_every: usize,
) -> Result<(QFieldDec, Vec<SnapStatsDec>), String> {
    let stokes = StokesSolverDec::new(ops, mesh)?;

    // Extract vertex coordinates for advection.
    let coords: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| {
        let s = format!("{:?}", v);
        let nums: Vec<f64> = s
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == ',' || *c == ' ' || *c == 'e' || *c == '+')
            .collect::<String>()
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        match nums.len() {
            2 => [nums[0], nums[1], 0.0],
            n if n >= 3 => [nums[0], nums[1], nums[2]],
            _ => [0.0, 0.0, 0.0],
        }
    }).collect();

    let mut q = q_init.clone();
    let mut stats = Vec::new();

    for step in 0..=n_steps {
        if step % snap_every == 0 {
            stats.push(SnapStatsDec {
                time: step as f64 * params.dt,
                mean_s: q.mean_order_param(),
                n_vertices: q.n_vertices,
            });
        }
        if step < n_steps {
            // 1. Stokes solve: get 3D tangent velocity.
            let vel = stokes.solve(&q, params, ops, mesh);

            // 2. RK4 step: molecular field + directional advection.
            let rhs = |qq: &QFieldDec| -> QFieldDec {
                let h = molecular_field_dec(qq, params, ops, curvature_correction);
                let mut dq = h.scale(params.gamma_r);

                let adv = advect_q(
                    qq, &vel,
                    &mesh.boundaries,
                    &mesh.vertex_boundaries,
                    &coords,
                );
                for i in 0..qq.n_vertices {
                    dq.q1[i] -= adv.q1[i];
                    dq.q2[i] -= adv.q2[i];
                }
                dq
            };

            let k1 = rhs(&q);
            let q2 = q.add(&k1.scale(0.5 * params.dt));
            let k2 = rhs(&q2);
            let q3 = q.add(&k2.scale(0.5 * params.dt));
            let k3 = rhs(&q3);
            let q4 = q.add(&k3.scale(params.dt));
            let k4 = rhs(&q4);
            let update = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
            q = q.add(&update.scale(params.dt / 6.0));
        }
    }

    Ok((q, stats))
}
