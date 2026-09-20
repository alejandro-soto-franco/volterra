//! Stokes with through-flow: what the lift adds, and what it leaves alone.
//!
//! Stokes is linear, so the stream function splits as `psi = flux * psi_unit +
//! psi_0`, with `psi_unit` solving the port data alone against a zero source
//! and `psi_0` the homogeneous solve the active stress already drove. The lift
//! is therefore solved once per geometry and a whole sweep over flux reuses it.
//!
//! Two traps this codebase has already paid for once. `clamped_condition()` is
//! rank deficient by construction, so it is reported and never asserted on.
//! Velocity is force-zeroed at the no-slip set, so asserting that it vanishes
//! there tests the zeroing rather than the wall treatment; the wall is measured
//! through `dpsi/dn` instead, against the simply supported solve, which is what
//! the clamp changes.

use std::f64::consts::TAU;

use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use nalgebra::DVector;
use volterra_core::{ActiveNematicParams, Screening};
use volterra_dec::QField;
use volterra_dec::confined::{ConfinedMesh2, MeshOpts, confined_mesh};
use volterra_dec::curve::PlaneCurve;
use volterra_dec::lobe_chain::LobeChain;
use volterra_dec::ports::{PortRole, PortSpan, Ports};
use volterra_dec::stokes::SurfaceStokes;

/// Fraction of the circuit spent on each long wall, matching [`LobeChain`].
const F_WALL: f64 = 0.45;
/// Fraction spent on each end wall.
const F_END: f64 = 0.05;

fn port_spans() -> [PortSpan; 2] {
    [
        PortSpan {
            role: PortRole::Outlet,
            from: TAU * F_WALL,
            to: TAU * (F_WALL + F_END),
        },
        PortSpan {
            role: PortRole::Inlet,
            from: TAU * (2.0 * F_WALL + F_END),
            to: TAU,
        },
    ]
}

/// A straight channel, traversed in the same leg order as [`LobeChain`], so one
/// pair of port spans serves both.
#[derive(Debug, Clone, Copy)]
struct Channel {
    len: f64,
    half: f64,
}

impl Channel {
    /// Leg index and the fraction along it.
    fn leg(&self, u: f64) -> (usize, f64) {
        let t = u.rem_euclid(TAU) / TAU;
        if t < F_WALL {
            (0, t / F_WALL)
        } else if t < F_WALL + F_END {
            (1, (t - F_WALL) / F_END)
        } else if t < 2.0 * F_WALL + F_END {
            (2, (t - F_WALL - F_END) / F_WALL)
        } else {
            (3, (t - 2.0 * F_WALL - F_END) / F_END)
        }
    }
}

impl PlaneCurve for Channel {
    fn point(&self, u: f64) -> [f64; 2] {
        let (leg, f) = self.leg(u);
        match leg {
            0 => [self.len * f, -self.half],
            1 => [self.len, -self.half + 2.0 * self.half * f],
            2 => [self.len * (1.0 - f), self.half],
            _ => [0.0, self.half - 2.0 * self.half * f],
        }
    }

    fn d1(&self, u: f64) -> [f64; 2] {
        let long = self.len / (TAU * F_WALL);
        let end = 2.0 * self.half / (TAU * F_END);
        match self.leg(u).0 {
            0 => [long, 0.0],
            1 => [0.0, end],
            2 => [-long, 0.0],
            _ => [0.0, -end],
        }
    }

    fn d2(&self, _u: f64) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn features(&self) -> Vec<f64> {
        vec![
            0.0,
            TAU * F_WALL,
            TAU * (F_WALL + F_END),
            TAU * (2.0 * F_WALL + F_END),
        ]
    }

    fn interior(&self) -> [f64; 2] {
        [self.len / 2.0, 0.0]
    }
}

struct Rig {
    cm: ConfinedMesh2,
    ops: Operators<Euclidean<2>, 3, 2>,
    ports: Ports,
}

fn rig<C: PlaneCurve + Copy + 'static>(curve: C, h_bulk: f64) -> Rig {
    let cm = confined_mesh(
        curve,
        MeshOpts {
            h_bulk,
            h_min: 0.008,
            cusp_edge: 0.01,
            ..Default::default()
        },
    );
    let ops = Operators::from_mesh(&cm.mesh, &Euclidean::<2>);
    let ports = Ports::new(&cm.boundary_vertices, &cm.boundary_params, &port_spans())
        .expect("the two end walls are disjoint spans of the circuit");
    Rig { cm, ops, ports }
}

fn chain_rig() -> Rig {
    rig(LobeChain::new(2, 0.25), 0.10)
}

fn clamped(r: &Rig) -> SurfaceStokes {
    SurfaceStokes::new_confined_clamped_screened(
        &r.ops,
        &r.cm.mesh,
        &r.cm.boundary_vertices,
        Screening::None,
    )
    .expect("the chain mesh assembles")
}

fn ported(r: &Rig, flux: f64) -> SurfaceStokes {
    clamped(r)
        .with_ports(&r.cm.mesh, &r.ports, flux)
        .expect("the ports match the boundary the solver was built on")
}

/// A source with structure, so the homogeneous part is not trivially zero.
fn source(r: &Rig) -> DVector<f64> {
    let n = r.cm.mesh.n_vertices();
    DVector::from_iterator(
        n,
        (0..n).map(|i| {
            let p = r.cm.mesh.vertices[i];
            (2.0 * p[0]).sin() * (3.0 * p[1]).cos()
        }),
    )
}

#[test]
fn zero_flux_leaves_the_stream_function_bit_for_bit() {
    let r = chain_rig();
    let s = source(&r);
    let (_v0, psi0) = clamped(&r).stream_and_velocity(&s, &r.cm.mesh);
    let (_v1, psi1) = ported(&r, 0.0).stream_and_velocity(&s, &r.cm.mesh);
    assert_eq!(psi0, psi1, "opening a port at zero flux moved the solve");
}

#[test]
fn zero_flux_leaves_the_velocity_alone_away_from_the_ports() {
    let r = chain_rig();
    let s = source(&r);
    let (v0, _) = clamped(&r).stream_and_velocity(&s, &r.cm.mesh);
    let (v1, _) = ported(&r, 0.0).stream_and_velocity(&s, &r.cm.mesh);
    let port: Vec<usize> = (0..r.ports.n_ports())
        .flat_map(|k| r.ports.port_vertices(k).to_vec())
        .collect();
    for i in 0..v0.n_vertices {
        if port.contains(&i) {
            continue;
        }
        assert_eq!(v0.v[i], v1.v[i], "vertex {i} moved with no flux through it");
    }
}

#[test]
fn the_lift_is_exactly_linear_in_the_flux() {
    let r = chain_rig();
    let zero = DVector::zeros(r.cm.mesh.n_vertices());
    let (_, one) = ported(&r, 1.0).stream_and_velocity(&zero, &r.cm.mesh);
    let (_, two) = ported(&r, 2.0).stream_and_velocity(&zero, &r.cm.mesh);
    for (a, b) in one.iter().zip(&two) {
        assert_eq!(2.0 * a, *b, "the lift is not exactly linear in the flux");
    }
    assert!(
        one.iter().any(|p| p.abs() > 1e-6),
        "the lift is identically zero, so the linearity is vacuous"
    );
}

#[test]
fn setting_the_flux_costs_nothing_and_changes_nothing_else() {
    let r = chain_rig();
    let s = source(&r);
    let (_, built) = ported(&r, 1.5).stream_and_velocity(&s, &r.cm.mesh);
    let mut solver = ported(&r, 0.0);
    solver.set_flux(1.5).expect("the solver has ports");
    let (_, reset) = solver.stream_and_velocity(&s, &r.cm.mesh);
    assert_eq!(built, reset, "the flux is not a scale on a fixed lift");
}

#[test]
fn the_inlet_draws_fluid_in_and_the_outlet_lets_it_out() {
    let r = chain_rig();
    let zero = DVector::zeros(r.cm.mesh.n_vertices());
    let (vel, _) = ported(&r, 1.0).stream_and_velocity(&zero, &r.cm.mesh);
    for k in 0..r.ports.n_ports() {
        let want = match r.ports.role(k) {
            PortRole::Inlet => 1.0,
            PortRole::Outlet => -1.0,
        };
        let mut seen = 0usize;
        for &v in r.ports.port_vertices(k) {
            let i =
                r.cm.boundary_vertices
                    .iter()
                    .position(|&b| b == v)
                    .expect("a port vertex is a boundary vertex");
            let n = r.cm.boundary_normals[i];
            let d = vel.v[v][0] * n[0] + vel.v[v][1] * n[1];
            // The two corner vertices of a port sit where the profile vanishes.
            if vel.v[v][0].hypot(vel.v[v][1]) < 1e-3 {
                continue;
            }
            assert!(
                d * want > 0.0,
                "port {k} at vertex {v} drives {d:.4e} through its own inward normal"
            );
            seen += 1;
        }
        assert!(seen >= 6, "port {k} moved fluid at only {seen} vertices");
    }
}

#[test]
fn the_activity_does_not_change_what_the_ports_pass() {
    let r = chain_rig();
    let flux = 1.0;
    let quiet = DVector::zeros(r.cm.mesh.n_vertices());
    let (_, psi_q) = ported(&r, flux).stream_and_velocity(&quiet, &r.cm.mesh);
    let (_, psi_a) = ported(&r, flux).stream_and_velocity(&source(&r), &r.cm.mesh);
    // The through-flux is the change in psi from one wall to the other, and
    // telescoping makes that exact whatever the interior is doing.
    let wall_value = |psi: &[f64], upper: bool| -> f64 {
        let mut vals: Vec<f64> = r
            .ports
            .wall_vertices()
            .iter()
            .filter(|&&v| (r.cm.mesh.vertices[v][1] > 0.0) == upper)
            .map(|&v| psi[v])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("psi is finite"));
        vals[vals.len() / 2]
    };
    for (name, psi) in [("quiet", &psi_q), ("active", &psi_a)] {
        let drop = wall_value(psi, false) - wall_value(psi, true);
        assert!(
            (drop - flux).abs() < 1e-12,
            "the {name} run passes {drop} rather than {flux}"
        );
    }
}

#[test]
fn the_wall_is_held_and_the_ports_are_left_free() {
    let r = chain_rig();
    let zero = DVector::zeros(r.cm.mesh.n_vertices());
    let (vel, _) = ported(&r, 1.0).stream_and_velocity(&zero, &r.cm.mesh);
    for &v in r.ports.wall_vertices() {
        assert_eq!(vel.v[v], [0.0, 0.0, 0.0], "wall vertex {v} was left moving");
    }
    let moving = (0..r.ports.n_ports())
        .flat_map(|k| r.ports.port_vertices(k).to_vec())
        .filter(|&v| vel.v[v][0].hypot(vel.v[v][1]) > 1e-3)
        .count();
    assert!(
        moving >= 12,
        "only {moving} port vertices are moving, so the zeroing took the inflow"
    );
}

#[test]
fn the_clamp_flattens_the_wall_gradient_and_the_free_wall_does_not() {
    let r = chain_rig();
    let zero = DVector::zeros(r.cm.mesh.n_vertices());
    let free = SurfaceStokes::new_confined_screened(
        &r.ops,
        &r.cm.mesh,
        &r.cm.boundary_vertices,
        Screening::None,
    )
    .expect("the chain mesh assembles")
    .with_ports(&r.cm.mesh, &r.ports, 1.0)
    .expect("the ports match the boundary");
    let (_, psi_free) = free.stream_and_velocity(&zero, &r.cm.mesh);
    let (_, psi_held) = ported(&r, 1.0).stream_and_velocity(&zero, &r.cm.mesh);

    // dpsi/dn at a wall vertex, read from its interior neighbours. psi is
    // constant along the wall, so the whole of the difference is normal.
    let on_boundary: Vec<bool> = {
        let mut m = vec![false; r.cm.mesh.n_vertices()];
        for &b in &r.cm.boundary_vertices {
            m[b] = true;
        }
        m
    };
    // The statistic is the median over the wall, not the maximum. The cusp
    // tips are meshed an order of magnitude finer than the bulk and psi turns
    // through a right angle across them, so a maximum reports the tips of a
    // chain whatever the wall treatment is.
    let slope = |psi: &[f64]| -> (f64, f64) {
        let mut all: Vec<f64> = r
            .ports
            .wall_vertices()
            .iter()
            .map(|&b| {
                let mut worst = 0.0_f64;
                for t in &r.cm.mesh.simplices {
                    if !t.contains(&b) {
                        continue;
                    }
                    for &j in t {
                        if on_boundary[j] {
                            continue;
                        }
                        let d = ((r.cm.mesh.vertices[j][0] - r.cm.mesh.vertices[b][0]).powi(2)
                            + (r.cm.mesh.vertices[j][1] - r.cm.mesh.vertices[b][1]).powi(2))
                        .sqrt();
                        worst = worst.max((psi[j] - psi[b]).abs() / d);
                    }
                }
                worst
            })
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).expect("the slopes are finite"));
        (all[all.len() / 2], all[all.len() - 1])
    };
    let (held, held_max) = slope(&psi_held);
    let (freed, freed_max) = slope(&psi_free);
    println!(
        "dpsi/dn at the wall: clamped median {held:.4e} max {held_max:.4e}, \
         simply supported median {freed:.4e} max {freed_max:.4e}"
    );
    // Rank deficient by construction, so this is reported and never asserted on.
    println!(
        "clamped condition: {:?}",
        ported(&r, 1.0).clamped_condition()
    );
    assert!(
        held < freed / 3.0,
        "the clamp left {held:.4e} against the free wall's {freed:.4e}"
    );
}

#[test]
fn a_straight_channel_at_rest_is_poiseuille() {
    let half = 0.5_f64;
    let len = 6.0_f64;
    let flux = 1.0_f64;
    // u_x = U (1 - y^2/h^2) with U = 3Q/4h, and psi = -integral u_x dy, fixed
    // by the lower wall sitting at zero and the upper at -Q.
    let u_peak = 3.0 * flux / (4.0 * half);
    let psi_exact =
        |y: f64| -> f64 { -u_peak * (y - y * y * y / (3.0 * half * half)) - flux / 2.0 };

    let measure = |h: f64| -> (f64, f64, usize) {
        let r = rig(Channel { len, half }, h);
        let zero = DVector::zeros(r.cm.mesh.n_vertices());
        let (vel, psi) = ported(&r, flux).stream_and_velocity(&zero, &r.cm.mesh);
        let (mut np, mut dp, mut nu, mut du) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut sampled = 0usize;
        for (i, &pi) in psi.iter().enumerate() {
            let (x, y) = (r.cm.mesh.vertices[i][0], r.cm.mesh.vertices[i][1]);
            // Away from both ends, where the profile is the whole answer.
            if !(2.0..=4.0).contains(&x) {
                continue;
            }
            let e = psi_exact(y);
            np += (pi - e) * (pi - e);
            dp += e * e;
            let ux = u_peak * (1.0 - y * y / (half * half));
            nu += (vel.v[i][0] - ux) * (vel.v[i][0] - ux);
            du += ux * ux;
            sampled += 1;
        }
        ((np / dp).sqrt(), (nu / du).sqrt(), sampled)
    };

    let (psi_c, u_c, n_c) = measure(0.06);
    let (psi_f, u_f, n_f) = measure(0.04);
    println!("coarse over {n_c} vertices: psi {psi_c:.3e}, u {u_c:.3e}");
    println!("  fine over {n_f} vertices: psi {psi_f:.3e}, u {u_f:.3e}");
    assert!(n_f > 500, "only {n_f} vertices in the sampled window");
    // Consistency, which is the statement worth making: both errors fall with
    // the mesh at the first order the vertex gradient recovery gives.
    assert!(
        psi_c / psi_f > 1.3,
        "the stream function error went {psi_c:.3e} to {psi_f:.3e} under a \
         refinement of 1.5, so it is not converging"
    );
    assert!(
        u_c / u_f > 1.3,
        "the velocity error went {u_c:.3e} to {u_f:.3e} under a refinement of 1.5"
    );
    assert!(
        psi_f < 0.025,
        "the stream function is {psi_f:.3e} from Poiseuille"
    );
    assert!(u_f < 0.07, "the velocity is {u_f:.3e} from the parabola");
}

#[test]
fn a_resting_nematic_does_not_stop_the_ports() {
    let r = chain_rig();
    let nv = r.cm.mesh.n_vertices();
    let (_, want) = ported(&r, 1.0).stream_and_velocity(&DVector::zeros(nv), &r.cm.mesh);
    // Zero activity is the diffusive baseline the mixer is measured against,
    // and the entry points short-circuit it. A closed chamber is then at rest;
    // a ported one is not.
    let mut params = ActiveNematicParams::default_test();
    params.zeta_eff = 0.0;
    let q = QField::zeros(nv);
    let (vel, psi, its) = ported(&r, 1.0).solve_warm(&q, &params, &r.ops, &r.cm.mesh, None, 1e-10);
    assert_eq!(its, 0, "the short circuit was not taken");
    assert_eq!(psi, want, "the resting run lost the port data");
    assert!(
        vel.v.iter().any(|v| v[0].hypot(v[1]) > 1e-3),
        "the ports drive nothing at zero activity"
    );

    let force = vec![[0.0_f64; 2]; nv];
    let (_, psi_f, _) = ported(&r, 1.0).solve_force_warm(&force, 0.0, &r.cm.mesh, None, 1e-10);
    assert_eq!(psi_f, want, "the forced entry point lost the port data");
}
