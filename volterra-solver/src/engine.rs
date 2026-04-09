//! Unified active nematohydrodynamics engine on Riemannian 2-manifolds.
//!
//! Solves the nondimensionalised Beris-Edwards + Stokes system using
//! operator splitting:
//!
//! ```text
//! 1. Stokes:       (1/Er) Delta(Delta+K) psi = Pe curl(div(V(z)))
//! 2. Advection:    z_adv = SemiLagrangian(z, u, dt)
//! 3. Diffusion:    z_new = z_adv + dt * (Delta_B z_adv + La z_adv - Lc |z_adv|^2 z_adv)
//! ```
//!
//! All parameters are dimensionless (Pe, Er, La, Lc). The timestep is
//! auto-computed from the diffusive CFL bound.

use cartan_core::Manifold;
use cartan_dec::{Mesh, Operators};
use volterra_core::NematicParams;
use volterra_dec::connection_laplacian::{ConnectionLaplacian, molecular_field_conn};
use volterra_dec::curved_stokes::{CurvedStokesSolver, nematic_vorticity_source};
use volterra_dec::semi_lagrangian::SemiLagrangian;
use volterra_dec::stokes_dec::VelocityFieldDec;
use volterra_dec::QFieldDec;

/// Per-snapshot statistics from the nematic engine.
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Simulation time (nondimensionalised).
    pub time: f64,
    /// Mean scalar order parameter.
    pub mean_s: f64,
    /// RMS velocity magnitude.
    pub velocity_rms: f64,
    /// Number of vertices.
    pub n_vertices: usize,
}

/// Active nematohydrodynamics engine on a Riemannian 2-manifold.
pub struct NematicEngine {
    /// Dimensionless parameters.
    params: NematicParams,
    /// Connection Laplacian (Bochner on L^2, spin-2 parallel transport).
    conn_lap: ConnectionLaplacian,
    /// Curved-surface Stokes solver (modified biharmonic).
    stokes: CurvedStokesSolver,
    /// Semi-Lagrangian advection operator.
    semi_lag: SemiLagrangian,
    /// Vertex coordinates in R^3.
    coords: Vec<[f64; 3]>,
    /// Dual cell areas (from star_0).
    dual_areas: Vec<f64>,
    /// Mesh boundary connectivity (for vorticity source computation).
    simplices: Vec<[usize; 3]>,
    boundaries: Vec<[usize; 2]>,
    vertex_boundaries: Vec<Vec<usize>>,
    /// Timestep (auto-computed from diffusive CFL).
    dt: f64,
    /// Number of vertices.
    n_vertices: usize,
}

impl NematicEngine {
    /// Construct the engine from a mesh, manifold, and dimensionless parameters.
    ///
    /// Precomputes all DEC operators, connection Laplacian, Stokes solver,
    /// and semi-Lagrangian data structures. The timestep is auto-computed
    /// from the diffusive CFL bound based on the mean edge length.
    pub fn new<M: Manifold>(
        mesh: Mesh<M, 3, 2>,
        manifold: M,
        params: NematicParams,
        gaussian_curvature: Vec<f64>,
    ) -> Result<Self, String> {
        params.validate()?;

        let ops = Operators::from_mesh_generic(&mesh, &manifold)
            .map_err(|e| format!("DEC operator assembly: {e:?}"))?;

        let nv = mesh.n_vertices();

        // Extract vertex coordinates.
        let coords: Vec<[f64; 3]> = volterra_dec::stokes_dec::extract_coords(&mesh);

        // Hodge stars for the connection Laplacian.
        let star0: Vec<f64> = (0..ops.hodge.star0().len())
            .map(|i| ops.hodge.star0()[i]).collect();
        let star1: Vec<f64> = (0..ops.hodge.star1().len())
            .map(|i| ops.hodge.star1()[i]).collect();

        // Connection Laplacian.
        let conn_lap = ConnectionLaplacian::new(&mesh, &coords, &star0, &star1);

        // Curved Stokes solver.
        let stokes = CurvedStokesSolver::new(&ops, &mesh, &gaussian_curvature)?;

        // Semi-Lagrangian advection.
        let semi_lag = SemiLagrangian::new(coords.clone(), mesh.simplices.clone());

        // Auto-compute timestep from mean edge length.
        let ne = mesh.n_boundaries();
        let mean_edge_len = if ne > 0 {
            let total: f64 = (0..ne).map(|e| {
                let [v0, v1] = mesh.boundaries[e];
                let d = [
                    coords[v1][0] - coords[v0][0],
                    coords[v1][1] - coords[v0][1],
                    coords[v1][2] - coords[v0][2],
                ];
                (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt()
            }).sum();
            total / ne as f64
        } else {
            0.01
        };
        let dt = params.dt_diffusive(mean_edge_len);

        Ok(Self {
            params,
            conn_lap,
            stokes,
            semi_lag,
            coords,
            dual_areas: star0,
            simplices: mesh.simplices.clone(),
            boundaries: mesh.boundaries.clone(),
            vertex_boundaries: mesh.vertex_boundaries.clone(),
            dt,
            n_vertices: nv,
        })
    }

    /// The auto-computed timestep.
    pub fn dt(&self) -> f64 { self.dt }

    /// Override the timestep (use with caution).
    pub fn set_dt(&mut self, dt: f64) { self.dt = dt; }

    /// Number of vertices in the mesh.
    pub fn n_vertices(&self) -> usize { self.n_vertices }

    /// The dimensionless parameters.
    pub fn params(&self) -> &NematicParams { &self.params }

    /// Advance the nematic field by one timestep using operator splitting.
    ///
    /// 1. Stokes solve for velocity from active stress.
    /// 2. Semi-Lagrangian advection (backward trace + barycentric interp).
    /// 3. Diffusion + bulk LdG (explicit, connection Laplacian).
    pub fn step(&self, q: &mut QFieldDec) -> VelocityFieldDec {
        let nv = self.n_vertices;
        let dt = self.dt;

        // 1. Stokes: compute vorticity source and solve for stream function.
        let source = nematic_vorticity_source(
            q, self.params.pe,
            // We need to pass the mesh data. Use stored simplices + coords.
            &self.simplices, &self.coords, &self.dual_areas,
        );
        let (_psi, _vel) = self.stokes.solve(&source, self.params.er);

        // For now, use the vorticity-based velocity from the old solver
        // until CurvedStokesSolver's velocity extraction is wired up.
        // Compute velocity from the stream function gradient.
        let vel = self.compute_velocity_from_source(q);

        // 2. Semi-Lagrangian advection.
        let q_adv = self.semi_lag.advect(q, &vel, dt);

        // 3. Diffusion + bulk LdG.
        let h = molecular_field_conn(
            &q_adv,
            1.0,                // K_frank = 1 in nondimensionalised units
            -self.params.la,    // a_eff = -La (nondimensionalised)
            self.params.lc,
            &self.conn_lap,
        );

        // Euler step for diffusion + bulk.
        for i in 0..nv {
            q.q1[i] = q_adv.q1[i] + dt * h.q1[i];
            q.q2[i] = q_adv.q2[i] + dt * h.q2[i];
        }

        vel
    }

    /// Temporary: compute velocity using the old Stokes solver approach.
    fn compute_velocity_from_source(&self, q: &QFieldDec) -> VelocityFieldDec {
        // Use the per-face gradient vorticity source.
        let nv = self.n_vertices;
        let pe = self.params.pe;
        let er = self.params.er;

        let mut omega = vec![0.0_f64; nv];
        let mut areas = vec![0.0_f64; nv];

        for &[i0, i1, i2] in &self.simplices {
            let p0 = self.coords[i0];
            let p1 = self.coords[i1];
            let p2 = self.coords[i2];
            let e01 = sub3(p1, p0);
            let e02 = sub3(p2, p0);
            let e12 = sub3(p2, p1);
            let e20 = sub3(p0, p2);

            let fn_vec = cross3(e01, e02);
            let area2 = norm3(fn_vec);
            if area2 < 1e-30 { continue; }
            let fn_hat = scale3(fn_vec, 1.0 / area2);
            let inv_2a = 1.0 / area2;

            let rot_e12 = cross3(fn_hat, e12);
            let rot_e20 = cross3(fn_hat, e20);
            let rot_e01 = cross3(fn_hat, e01);

            let gq1 = scale3(add3(add3(
                scale3(rot_e12, q.q1[i0]),
                scale3(rot_e20, q.q1[i1])),
                scale3(rot_e01, q.q1[i2])), inv_2a);
            let gq2 = scale3(add3(add3(
                scale3(rot_e12, q.q2[i0]),
                scale3(rot_e20, q.q2[i1])),
                scale3(rot_e01, q.q2[i2])), inv_2a);

            let fx = -pe * (gq1[0] + gq2[1]);
            let fy = -pe * (gq2[0] - gq1[1]);

            let circ_01 = fx * e01[0] + fy * e01[1];
            let circ_12 = fx * e12[0] + fy * e12[1];
            let circ_20 = fx * e20[0] + fy * e20[1];

            let face_area = 0.5 * area2;
            let third = face_area / 3.0;
            areas[i0] += third;
            areas[i1] += third;
            areas[i2] += third;

            omega[i0] += 0.5 * (circ_01 - circ_20);
            omega[i1] += 0.5 * (circ_12 - circ_01);
            omega[i2] += 0.5 * (circ_20 - circ_12);
        }

        for i in 0..nv {
            if areas[i] > 1e-30 {
                omega[i] /= er * areas[i];
            }
        }

        // Solve for psi via the standard Poisson (temporarily using the
        // omega directly as the velocity proxy until the full pipeline is wired).
        // This is a simplified velocity that captures the flow direction.
        let _ne = self.boundaries.len();
        let mut vel = vec![[0.0_f64; 3]; nv];

        // Use omega as a rough velocity magnitude indicator.
        // The actual velocity extraction uses the stream function, but for
        // the semi-Lagrangian to work we need SOME velocity.
        // Use the per-face force directly as a tangent velocity estimate.
        for &[i0, i1, i2] in &self.simplices {
            let p0 = self.coords[i0];
            let p1 = self.coords[i1];
            let p2 = self.coords[i2];
            let e01 = sub3(p1, p0);
            let e02 = sub3(p2, p0);

            let fn_vec = cross3(e01, e02);
            let area2 = norm3(fn_vec);
            if area2 < 1e-30 { continue; }
            let fn_hat = scale3(fn_vec, 1.0 / area2);
            let inv_2a = 1.0 / area2;

            let e12 = sub3(p2, p1);
            let e20 = sub3(p0, p2);
            let rot_e12 = cross3(fn_hat, e12);
            let rot_e20 = cross3(fn_hat, e20);
            let rot_e01 = cross3(fn_hat, e01);

            let gq1 = scale3(add3(add3(
                scale3(rot_e12, q.q1[i0]),
                scale3(rot_e20, q.q1[i1])),
                scale3(rot_e01, q.q1[i2])), inv_2a);
            let gq2 = scale3(add3(add3(
                scale3(rot_e12, q.q2[i0]),
                scale3(rot_e20, q.q2[i1])),
                scale3(rot_e01, q.q2[i2])), inv_2a);

            // Active force (tangent to surface).
            let fx = -pe / er * (gq1[0] + gq2[1]);
            let fy = -pe / er * (gq2[0] - gq1[1]);
            let fz = 0.0;

            // Project force onto tangent plane and distribute to vertices.
            let f_tang = [fx - fn_hat[0] * (fx * fn_hat[0] + fy * fn_hat[1]),
                          fy - fn_hat[1] * (fx * fn_hat[0] + fy * fn_hat[1]),
                          fz - fn_hat[2] * (fx * fn_hat[0] + fy * fn_hat[1])];

            for &vi in &[i0, i1, i2] {
                vel[vi] = add3(vel[vi], scale3(f_tang, 1.0 / 3.0));
            }
        }

        // Normalise by vertex valence.
        for (v, boundaries) in vel.iter_mut().zip(&self.vertex_boundaries) {
            let valence = boundaries.len().max(1) as f64;
            *v = scale3(*v, 1.0 / valence);
        }

        VelocityFieldDec { v: vel, n_vertices: nv }
    }

    /// Run the engine for n_steps, calling the callback at each snapshot.
    pub fn run(
        &self,
        q: &mut QFieldDec,
        n_steps: usize,
        snap_every: usize,
        mut callback: impl FnMut(usize, &QFieldDec, &VelocityFieldDec, &EngineStats),
    ) {
        for step in 0..=n_steps {
            if step % snap_every == 0 {
                let vel = if step > 0 {
                    self.compute_velocity_from_source(q)
                } else {
                    VelocityFieldDec::zeros(self.n_vertices)
                };
                let v_rms = (vel.v.iter()
                    .map(|[x, y, z]| x*x + y*y + z*z)
                    .sum::<f64>() / self.n_vertices as f64).sqrt();
                let stats = EngineStats {
                    time: step as f64 * self.dt,
                    mean_s: q.mean_order_param(),
                    velocity_rms: v_rms,
                    n_vertices: self.n_vertices,
                };
                callback(step, q, &vel, &stats);
            }
            if step < n_steps {
                self.step(q);
            }
        }
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s, a[1]*s, a[2]*s] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
