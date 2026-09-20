//! How the wall of a chain splits into no-slip wall and prescribed port.
//!
//! The boundary of a through-flow chamber is one loop whose wall segments are
//! each a streamline, so `psi` is constant on each of them, and whose ports
//! ramp `psi` from one wall's value to the other's. The total change round the
//! loop is zero, which is the statement that what enters leaves.

use std::f64::consts::TAU;

use volterra_dec::confined::{MeshOpts, confined_mesh};
use volterra_dec::lobe_chain::LobeChain;
use volterra_dec::ports::{PortRole, PortSpan, Ports};

struct Rig {
    ports: Ports,
    boundary: Vec<usize>,
    params: Vec<f64>,
    y: Vec<f64>,
}

fn spans(chain: &LobeChain) -> [PortSpan; 2] {
    let (a, b) = chain.outlet_span();
    let (c, d) = chain.inlet_span();
    [
        PortSpan {
            role: PortRole::Outlet,
            from: a,
            to: b,
        },
        PortSpan {
            role: PortRole::Inlet,
            from: c,
            to: d,
        },
    ]
}

fn rig(n: usize, neck: f64) -> Rig {
    let chain = LobeChain::new(n, neck);
    let sp = spans(&chain);
    let cm = confined_mesh(
        chain,
        MeshOpts {
            h_bulk: 0.10,
            h_min: 0.008,
            cusp_edge: 0.01,
            ..Default::default()
        },
    );
    let y: Vec<f64> = cm
        .boundary_vertices
        .iter()
        .map(|&v| cm.mesh.vertices[v][1])
        .collect();
    let ports = Ports::new(&cm.boundary_vertices, &cm.boundary_params, &sp)
        .expect("the two end walls are disjoint spans of the circuit");
    Rig {
        ports,
        boundary: cm.boundary_vertices,
        params: cm.boundary_params,
        y,
    }
}

/// Positions in the boundary ordering of the vertices of port `k`.
fn positions(r: &Rig, k: usize) -> Vec<usize> {
    r.ports
        .port_vertices(k)
        .iter()
        .map(|v| {
            r.boundary
                .iter()
                .position(|b| b == v)
                .expect("a port vertex is a boundary vertex")
        })
        .collect()
}

#[test]
fn the_wall_and_the_two_ports_partition_the_boundary() {
    let r = rig(2, 0.25);
    let mut seen: Vec<usize> = r.ports.wall_vertices().to_vec();
    seen.extend_from_slice(r.ports.port_vertices(0));
    seen.extend_from_slice(r.ports.port_vertices(1));
    seen.sort_unstable();
    let mut all = r.boundary.clone();
    all.sort_unstable();
    assert_eq!(seen, all, "every boundary vertex lands in exactly one part");
    // Both end walls are resolved well enough to carry a profile.
    for k in 0..2 {
        assert!(
            r.ports.port_vertices(k).len() >= 8,
            "port {k} has only {} vertices",
            r.ports.port_vertices(k).len()
        );
    }
}

#[test]
fn a_port_is_one_contiguous_run_along_the_wall() {
    let r = rig(2, 0.25);
    for k in 0..2 {
        let mut p = positions(&r, k);
        p.sort_unstable();
        assert_eq!(
            p[p.len() - 1] - p[0],
            p.len() - 1,
            "port {k} is not one run of the circuit"
        );
    }
}

#[test]
fn each_wall_sits_at_one_value_and_they_differ_by_the_flux() {
    let r = rig(2, 0.25);
    let psi = r.ports.psi_unit();
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for (i, &v) in r.boundary.iter().enumerate() {
        if r.ports.wall_vertices().contains(&v) {
            if r.y[i] < 0.0 { &mut lower } else { &mut upper }.push(psi[i]);
        }
    }
    assert!(!lower.is_empty() && !upper.is_empty());
    for (name, side) in [("lower", &lower), ("upper", &upper)] {
        let first = side[0];
        for &p in side.iter() {
            assert_eq!(p, first, "the {name} wall is not a single streamline");
        }
    }
    assert_eq!(
        (upper[0] - lower[0]).abs(),
        1.0,
        "the two walls differ by exactly the unit flux"
    );
}

#[test]
fn the_ramp_across_a_port_spans_the_unit_flux_monotonically() {
    let r = rig(2, 0.25);
    let psi = r.ports.psi_unit();
    let n = r.boundary.len();
    for k in 0..2 {
        let mut p = positions(&r, k);
        p.sort_unstable();
        let run: Vec<f64> = p.iter().map(|&i| psi[i]).collect();
        let rising = run[run.len() - 1] > run[0];
        for w in run.windows(2) {
            assert!(
                if rising { w[1] > w[0] } else { w[1] < w[0] },
                "port {k} reverses inside its own span"
            );
        }
        // The wall vertices either side bracket the run, and the values they
        // sit at differ by exactly one unit of flux. The inlet's run ends at
        // the parameter origin, so the circuit is walked modulo its length.
        let before = psi[(p[0] + n - 1) % n];
        let after = psi[(p[p.len() - 1] + 1) % n];
        assert_eq!(
            (after - before).abs(),
            1.0,
            "port {k} does not span the unit flux"
        );
    }
}

#[test]
fn the_prescribed_profile_vanishes_at_the_port_corners() {
    let r = rig(2, 0.25);
    let psi = r.ports.psi_unit();
    let n = r.boundary.len();
    for k in 0..2 {
        let mut p = positions(&r, k);
        p.sort_unstable();
        // The corner itself is a wall vertex, so the run is extended to the
        // bracketing wall vertex at each end before the rate is read.
        let mut walk = vec![(p[0] + n - 1) % n];
        walk.extend_from_slice(&p);
        walk.push((p[p.len() - 1] + 1) % n);
        let rate: Vec<f64> = walk
            .windows(2)
            .map(|w| {
                let du = (r.params[w[1]] - r.params[w[0]]).rem_euclid(TAU);
                (psi[w[1]] - psi[w[0]]).abs() / du
            })
            .collect();
        let peak = rate.iter().cloned().fold(0.0_f64, f64::max);
        for (name, edge) in [("start", rate[0]), ("end", rate[rate.len() - 1])] {
            assert!(
                edge < peak / 3.0,
                "port {k} drives flow into its own corner at the {name}: \
                 {edge:.4} against a peak of {peak:.4}"
            );
        }
    }
}

#[test]
fn overlapping_spans_are_refused() {
    let chain = LobeChain::new(2, 0.25);
    let (a, b) = chain.outlet_span();
    let sp = [
        PortSpan {
            role: PortRole::Outlet,
            from: a,
            to: b,
        },
        PortSpan {
            role: PortRole::Inlet,
            from: (a + b) / 2.0,
            to: b + 0.1,
        },
    ];
    let err = Ports::new(&[0, 1], &[0.0, 1.0], &sp).expect_err("the spans overlap");
    assert!(err.contains("overlap"), "{err}");
}

#[test]
fn a_port_set_that_does_not_balance_is_refused() {
    let chain = LobeChain::new(2, 0.25);
    let (a, b) = chain.outlet_span();
    let (c, d) = chain.inlet_span();
    let sp = [
        PortSpan {
            role: PortRole::Inlet,
            from: a,
            to: b,
        },
        PortSpan {
            role: PortRole::Inlet,
            from: c,
            to: d,
        },
    ];
    let err = Ports::new(&[0, 1], &[0.0, 1.0], &sp).expect_err("two inlets and no outlet");
    assert!(err.contains("balance"), "{err}");
}
