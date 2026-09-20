//! Inlet and outlet cut into the wall of a confined domain.
//!
//! # What a port is
//!
//! In the stream-function formulation a boundary curve is a streamline exactly
//! when `psi` is constant along it, so a closed wall with no through-flow takes
//! one value everywhere. Cutting a port into the wall does one thing to that
//! picture: `psi` ramps across the port's own span, and the two wall segments
//! either side of it sit at values differing by the flux the port passes. The
//! normal velocity at the wall is the tangential derivative of `psi`, so the
//! ramp IS the inlet profile, and the profile integrates to the flux by
//! construction rather than by a quadrature that leaves a residue.
//!
//! Round the whole circuit `psi` must return to where it started, which is the
//! statement that what enters leaves. [`Ports::new`] refuses a set of ports
//! that does not balance, because the alternative is a multivalued `psi` and a
//! solve that reports success.
//!
//! # The profile
//!
//! Across a port spanning `[u0, u1]` of the curve parameter, with
//! `x = (u - u0) / (u1 - u0)`,
//!
//! ```text
//! psi(x) = psi(u0) + sigma (3 x^2 - 2 x^3),      v_n(x) = 6 sigma x (1 - x) / w.
//! ```
//!
//! The profile vanishes at both corners, where the port meets a no-slip wall,
//! so the prescribed data is compatible with the wall it is cut into. A plug
//! profile would drive a finite velocity into the corner and the solve would
//! spend its resolution on that singularity instead of on the mixer.
//!
//! `x` runs in the curve parameter rather than in arclength. The two agree on a
//! straight port, which is the case here, since [`LobeChain`]'s end walls are
//! parametrised affinely in `y`. On a curved port they differ, and the profile
//! is then a shape chosen in the parametrisation; the flux it passes is exactly
//! `sigma` either way, which is the property the solve depends on.
//!
//! [`LobeChain`]: crate::lobe_chain::LobeChain

/// Which way a port drives the flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortRole {
    /// Fluid enters the domain here.
    Inlet,
    /// Fluid leaves the domain here.
    Outlet,
}

impl PortRole {
    /// Change in `psi` across the port per unit flux, in the direction of
    /// increasing curve parameter.
    ///
    /// The velocity recovery is `v = n x grad psi`, so with an anticlockwise
    /// wall and an outward normal `N = (T_y, -T_x)` the outward flux density is
    /// `v . N = -dpsi/ds`. An outlet therefore lowers `psi` along the circuit
    /// and an inlet raises it. `tests/ported_stokes.rs` measures the direction
    /// against the mesher's own inward normals rather than trusting this note.
    fn sigma(self) -> f64 {
        match self {
            PortRole::Inlet => 1.0,
            PortRole::Outlet => -1.0,
        }
    }
}

/// A port, as an interval of the wall curve's parameter.
#[derive(Debug, Clone, Copy)]
pub struct PortSpan {
    pub role: PortRole,
    /// Parameter at which the port begins, inclusive.
    pub from: f64,
    /// Parameter at which it ends, exclusive.
    pub to: f64,
}

/// Boundary data for a chamber with through-flow.
///
/// Built once per geometry from the mesher's ordered boundary. Everything it
/// reports is in units of one unit of volumetric flux, so a run scales it by
/// the flux it wants and a sweep over flux costs one solve, not one per value.
#[derive(Debug)]
pub struct Ports {
    boundary: Vec<usize>,
    psi_unit: Vec<f64>,
    wall: Vec<usize>,
    ports: Vec<Vec<usize>>,
    spans: Vec<PortSpan>,
}

impl Ports {
    /// Partition an ordered boundary against a set of port spans.
    ///
    /// `boundary` and `params` are the mesher's boundary vertices and their
    /// curve parameters, in order along the curve; `spans` need not be sorted.
    pub fn new(boundary: &[usize], params: &[f64], spans: &[PortSpan]) -> Result<Self, String> {
        if boundary.len() != params.len() {
            return Err(format!(
                "{} boundary vertices against {} parameters",
                boundary.len(),
                params.len()
            ));
        }
        let mut spans = spans.to_vec();
        // Validated before the sort, which compares the same parameters.
        for s in &spans {
            if !s.from.is_finite() || !s.to.is_finite() || s.to <= s.from {
                return Err(format!(
                    "a port spans [{}, {}), which is empty or not a pair of parameters",
                    s.from, s.to
                ));
            }
        }
        spans.sort_by(|a, b| {
            a.from
                .partial_cmp(&b.from)
                .expect("port spans were checked finite")
        });
        for w in spans.windows(2) {
            if w[1].from < w[0].to {
                return Err(format!(
                    "ports [{}, {}) and [{}, {}) overlap",
                    w[0].from, w[0].to, w[1].from, w[1].to
                ));
            }
        }
        let net: f64 = spans.iter().map(|s| s.role.sigma()).sum();
        if net != 0.0 {
            return Err(format!(
                "the ports do not balance: their fluxes sum to {net} rather than zero, \
                 which leaves psi multivalued round the circuit"
            ));
        }

        let n = boundary.len();
        let mut psi_unit = vec![0.0_f64; n];
        let mut wall = Vec::new();
        let mut ports = vec![Vec::new(); spans.len()];
        // One pass up the circuit. `run` is the wall value reached after every
        // port already passed, so a wall vertex reads it directly and a port
        // vertex ramps off it.
        let mut run = 0.0_f64;
        let mut k = 0usize;
        for i in 0..n {
            let u = params[i];
            while k < spans.len() && u >= spans[k].to {
                run += spans[k].role.sigma();
                k += 1;
            }
            match spans.get(k) {
                Some(s) if u >= s.from => {
                    let x = (u - s.from) / (s.to - s.from);
                    psi_unit[i] = run + s.role.sigma() * ramp(x);
                    ports[k].push(boundary[i]);
                }
                _ => {
                    psi_unit[i] = run;
                    wall.push(boundary[i]);
                }
            }
        }
        Ok(Self {
            boundary: boundary.to_vec(),
            psi_unit,
            wall,
            ports,
            spans,
        })
    }

    /// Boundary vertices, in the order the parameters were given.
    pub fn boundary(&self) -> &[usize] {
        &self.boundary
    }

    /// `psi` at each boundary vertex per unit flux, aligned to [`Self::boundary`].
    pub fn psi_unit(&self) -> &[f64] {
        &self.psi_unit
    }

    /// Boundary vertices that stay no-slip.
    pub fn wall_vertices(&self) -> &[usize] {
        &self.wall
    }

    /// Boundary vertices of port `k`, ports ordered by parameter.
    pub fn port_vertices(&self, k: usize) -> &[usize] {
        &self.ports[k]
    }

    /// Role of port `k`.
    pub fn role(&self, k: usize) -> PortRole {
        self.spans[k].role
    }

    pub fn n_ports(&self) -> usize {
        self.spans.len()
    }
}

/// Smoothstep, the integral of the parabola that vanishes at both corners.
fn ramp(x: f64) -> f64 {
    x * x * (3.0 - 2.0 * x)
}
