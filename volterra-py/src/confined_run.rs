// volterra-py/src/confined_run.rs
//
// PyO3 bindings for the confined active nematic run.
//
// Exposed to Python (import volterra):
//   volterra.ConfinedRun -- Beris-Edwards on a conforming mesh, stepped from Python

use numpy::ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use volterra_dec::confined_ldg::LdgProblem;
use volterra_dec::nematic_params::NematicParams;
use volterra_dec::qfield::QField;
use volterra_dec::stokes::{SurfaceStokes, VelocityField};

use crate::mesh::PyConfinedMesh;

/// One confined active nematic run, stepped from Python.
///
/// The scheme is the one the Rust driver uses. At each step the Stokes problem
/// is solved for the velocity from the stress, and the Q field is advanced by
/// the Beris-Edwards equation,
///
/// ```text
/// dQ/dt + u.grad Q = H / gamma + S
/// ```
///
/// with the Frank term implicit, so the step is bounded by accuracy rather than
/// by the smallest element. Boundary vertices are pinned to the anchored value
/// after every step.
///
/// The wall condition on the velocity is the `wall` argument. `"noslip"` is the
/// clamped plate, `psi = 0` with `dpsi/dn = 0`. `"freeslip"` is the simply
/// supported plate, `psi = 0` with `Laplacian psi = 0`, which on a disc under
/// uniform load runs faster than no slip by exactly `2 sqrt 2`. Both take the
/// same anchoring, so the two are an A/B on one mesh.
// The Stokes solver keeps its warm start in a `RefCell`, so it is not `Sync`.
// A run is single-threaded state anyway: it belongs to the thread that built
// it, and `unsendable` says so rather than leaving it to chance.
#[pyclass(name = "ConfinedRun", module = "volterra", unsendable)]
pub struct PyConfinedRun {
    problem: LdgProblem,
    stokes: SurfaceStokes,
    rotor: volterra_core::ActiveNematicParams,
    q: QField,
    psi: Option<Vec<f64>>,
    velocity: Option<VelocityField>,
    dt: f64,
    cg_tol: f64,
    stokes_tol: f64,
    picard: usize,
    full_stress: bool,
    elastic_mask: Vec<usize>,
    noslip: Vec<usize>,
    local_h: Vec<f64>,
    merge: f64,
    wall: String,
    steps_taken: usize,
    last_dq: f64,
    last_cfl: f64,
    last_cg: usize,
    last_stokes_iters: usize,
}

impl PyConfinedRun {
    /// One step of the scheme, with no Python state touched.
    fn advance_one(&mut self) -> Result<(), String> {
        let nv = self.q.n_vertices;
        let mut q_next = self.q.clone();
        let mut vel_out: Option<VelocityField> = None;
        let mut psi_out: Vec<f64> = Vec::new();
        let mut iters_out = 0usize;
        let mut dq = 0.0_f64;
        let mut cg = 0usize;

        for pass in 0..self.picard {
            let src = if pass == 0 { &self.q } else { &q_next };
            let (vel_p, psi_p, iters_p) = if self.full_stress {
                let (s1, s2, sa) = self
                    .problem
                    .beris_edwards_stress_masked(src, &self.elastic_mask);
                self.stokes.solve_stress_warm(
                    &s1,
                    &s2,
                    &sa,
                    self.problem.params.eta,
                    &self.problem.mesh.mesh,
                    self.psi.as_deref(),
                    self.stokes_tol,
                )
            } else {
                self.stokes.solve_warm(
                    src,
                    &self.rotor,
                    &self.problem.ops,
                    &self.problem.mesh.mesh,
                    self.psi.as_deref(),
                    self.stokes_tol,
                )
            };
            let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel_p.v[i][0], vel_p.v[i][1]]).collect();
            let mut trial = self.q.clone();
            // `grad u` from the stream function rather than from a second
            // gradient of the recovered velocity: the first is O(h^1.1) and
            // exactly divergence free at every vertex, the second O(h^0.4).
            let du = self.problem.velocity_gradients_from_psi(&psi_p);
            let (dq_p, cg_p) = self.problem.step_active_with_du(
                &mut trial,
                &v2,
                &du,
                self.dt,
                self.cg_tol,
                None,
            );
            dq = dq_p;
            cg = cg_p;
            q_next = trial;
            vel_out = Some(vel_p);
            psi_out = psi_p;
            iters_out = iters_p;
        }

        let vel = vel_out.expect("at least one pass");
        let v2: Vec<[f64; 2]> = (0..nv).map(|i| [vel.vx(i), vel.vy(i)]).collect();
        self.last_cfl = self.problem.courant(&v2, self.dt, &self.local_h).0;
        self.psi = Some(psi_out);
        self.velocity = Some(vel);
        self.q = q_next;
        self.steps_taken += 1;
        self.last_dq = dq;
        self.last_cg = cg;
        self.last_stokes_iters = iters_out;

        // A saturated field is a fixed point of the semi-implicit solve, so `dq`
        // reads zero once it has run away and finiteness alone catches nothing.
        // The order parameter cannot exceed the equilibrium the Landau potential
        // sets by any meaningful factor, so bound it against that.
        let s0 = self.problem.params.s0();
        let (worst, s_max) = (0..nv)
            .map(|i| {
                (
                    i,
                    (2.0 * (self.q.q1[i] * self.q.q1[i] + self.q.q2[i] * self.q.q2[i])).sqrt(),
                )
            })
            .fold((0usize, 0.0_f64), |a, x| if x.1 > a.1 { x } else { a });
        if !dq.is_finite() || !s_max.is_finite() || s_max > 100.0 * s0 {
            let m = &self.problem.mesh.mesh;
            let p = m.vertex(worst);
            let on_wall = self.problem.mesh.boundary_vertices.contains(&worst);
            let hot = (0..nv)
                .filter(|&i| {
                    (2.0 * (self.q.q1[i] * self.q.q1[i] + self.q.q2[i] * self.q.q2[i])).sqrt()
                        > 2.0 * s0
                })
                .count();
            return Err(format!(
                "the run went unstable at step {} (t = {:.4}): S reached {s_max:.3e} against an \
                 equilibrium s0 of {s0:.4}, worst dQ {dq:.3e}. The worst vertex is {worst} at \
                 ({:.3}, {:.3}), local h {:.4}, {} the wall; {hot} of {nv} vertices are above \
                 2 s0. Reduce dt, or raise h_min when meshing.",
                self.steps_taken,
                self.steps_taken as f64 * self.dt,
                p[0],
                p[1],
                self.local_h[worst],
                if on_wall { "on" } else { "off" },
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyConfinedRun {
    /// Assemble the operators, the anchoring and the wall for one mesh.
    ///
    /// Parameters
    /// ----------
    /// mesh : ConfinedMesh
    ///     The domain, from `confined_mesh`.
    /// active_length : float
    ///     `als = sqrt(K / zeta)`, the active length scale in lattice units.
    /// coherence_length : float
    ///     `ncl = sqrt(K / C)`, the nematic coherence length, which is the defect
    ///     core size. It has to sit above about twice the bulk element for a
    ///     defect count to mean anything.
    /// resolution : int
    ///     Lattice side the two lengths are quoted against. Provenance only; the
    ///     dimensional constants do not depend on it.
    /// q_anchor : float
    ///     Anchoring winding. 1 is planar anchoring along the wall tangent; `q`
    ///     forces a total winding of `2 pi q` in the boundary director.
    /// wall : str
    ///     `"noslip"` for the clamped plate, `"freeslip"` for the simply
    ///     supported one.
    /// wall_h : float
    ///     Interior vertices on elements below this size take the wall condition
    ///     along with the wall itself. The velocity is recovered as a
    ///     discrete curl, which divides edge fluxes by dual areas, so a strongly
    ///     graded element amplifies any error in `psi` by the same factor. The
    ///     anchoring is untouched, so the imposed winding is unaffected.
    /// dt : float, optional
    ///     Timestep. Defaults to the parameter set's own.
    /// seed : int
    ///     Seed for the random initial field.
    /// picard : int
    ///     Passes of the coupling per step. 1 is the sequential scheme, which is
    ///     the default; iterating it does not stabilise the step.
    /// full_stress : bool
    ///     The full Beris-Edwards stress, or the active term alone. The active
    ///     term alone omits the elastic backflow that opposes the active flow.
    /// elastic_mask : bool
    /// elastic_h : float
    ///     Suppress the elastic stress on elements below `elastic_h`, keeping the
    ///     active term. Those elements are below the core size, so a free energy
    ///     differentiated twice across them is not a force the physics exerts,
    ///     and it is what makes the explicit stress unstable there.
    /// cg_tol, stokes_tol : float
    ///     Conjugate-gradient tolerances for the Q solve and the Stokes solve.
    #[new]
    #[allow(clippy::too_many_arguments)]
    // The negations are deliberate: `!(x > 0.0)` rejects NaN as well as a
    // non-positive value, which a positive test would let through. The indexed
    // loops read `local_h` against a set of wall vertices, which is clearer by
    // index than by iterator.
    #[allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]
    #[pyo3(signature = (
        mesh,
        active_length,
        coherence_length,
        resolution,
        q_anchor = 1.0,
        wall = "noslip",
        wall_h = 0.05,
        dt = None,
        seed = 0,
        picard = 1,
        full_stress = true,
        elastic_mask = true,
        elastic_h = 0.5,
        cg_tol = 1e-8,
        stokes_tol = 1e-8
    ))]
    fn new(
        mesh: &PyConfinedMesh,
        active_length: f64,
        coherence_length: f64,
        resolution: usize,
        q_anchor: f64,
        wall: &str,
        wall_h: f64,
        dt: Option<f64>,
        seed: u64,
        picard: usize,
        full_stress: bool,
        elastic_mask: bool,
        elastic_h: f64,
        cg_tol: f64,
        stokes_tol: f64,
    ) -> PyResult<Self> {
        if !(active_length > 0.0) || !active_length.is_finite() {
            return Err(PyValueError::new_err("active_length must be positive"));
        }
        if !(coherence_length > 0.0) || !coherence_length.is_finite() {
            return Err(PyValueError::new_err("coherence_length must be positive"));
        }
        if resolution == 0 {
            return Err(PyValueError::new_err("resolution must be positive"));
        }
        if picard == 0 {
            return Err(PyValueError::new_err("picard must be at least 1"));
        }
        let wall = wall.to_ascii_lowercase();
        if wall != "noslip" && wall != "freeslip" {
            return Err(PyValueError::new_err(format!(
                "wall must be \"noslip\" or \"freeslip\", got {wall:?}"
            )));
        }

        let params = NematicParams::from_length_scales(active_length, coherence_length, resolution);
        let dt = dt.unwrap_or(params.dt);
        if !(dt > 0.0) || !dt.is_finite() {
            return Err(PyValueError::new_err("dt must be positive and finite"));
        }

        let inner = mesh.clone_inner();
        let h_bulk_guess = inner.quality.max_edge;
        let problem = LdgProblem::new(inner, params, q_anchor)
            .map_err(|e| PyRuntimeError::new_err(format!("assembling the operators: {e}")))?;

        let nv = problem.mesh.mesh.n_vertices();
        let local_h = problem.local_h();
        let on_wall: std::collections::HashSet<usize> =
            problem.mesh.boundary_vertices.iter().copied().collect();

        let mut noslip = problem.mesh.boundary_vertices.clone();
        for i in 0..nv {
            if local_h[i] > 0.0 && local_h[i] < wall_h && !on_wall.contains(&i) {
                noslip.push(i);
            }
        }

        let stokes = if wall == "freeslip" {
            SurfaceStokes::new_confined(&problem.ops, &problem.mesh.mesh, &noslip)
        } else {
            SurfaceStokes::new_confined_clamped(&problem.ops, &problem.mesh.mesh, &noslip)
        }
        .map_err(|e| PyRuntimeError::new_err(format!("factorising the Stokes wall: {e}")))?;

        let mask = if elastic_mask {
            let mut m = problem.mesh.boundary_vertices.clone();
            for i in 0..nv {
                if local_h[i] > 0.0 && local_h[i] < elastic_h && !on_wall.contains(&i) {
                    m.push(i);
                }
            }
            m
        } else {
            Vec::new()
        };

        // The Stokes solver reads only `zeta_eff` and `eta` off this struct; the
        // molecular field comes from `LdgProblem`, which has its own constants,
        // so the rotor convention's `a_eff` never enters.
        let mut rotor = volterra_core::ActiveNematicParams::default_test();
        rotor.zeta_eff = problem.params.zeta;
        rotor.eta = problem.params.eta;

        let q = problem.random_state(seed);

        Ok(Self {
            rotor,
            q,
            psi: None,
            velocity: None,
            dt,
            cg_tol,
            stokes_tol,
            picard,
            full_stress,
            elastic_mask: mask,
            noslip,
            local_h,
            merge: 1.5 * h_bulk_guess,
            wall,
            steps_taken: 0,
            last_dq: 0.0,
            last_cfl: 0.0,
            last_cg: 0,
            last_stokes_iters: 0,
            problem,
            stokes,
        })
    }

    /// Advance the run by `n` steps.
    ///
    /// Raises `RuntimeError` if the field runs away, naming the step, the worst
    /// vertex and whether it sits on the wall. The state at the failure is kept,
    /// so it can be inspected.
    #[pyo3(signature = (n = 1))]
    fn step(&mut self, py: Python<'_>, n: usize) -> PyResult<()> {
        py.detach(|| {
            for _ in 0..n {
                self.advance_one().map_err(PyRuntimeError::new_err)?;
            }
            Ok(())
        })
    }

    /// Relax the field with the flow switched off.
    ///
    /// Runs at most `n` passive steps and stops early once the largest change in
    /// Q falls below `tol`. Returns `(steps taken, last change)`. Starting an
    /// active run from a settled field rather than from noise is a different
    /// initial condition, so use it deliberately.
    #[pyo3(signature = (n, tol = 1e-10))]
    fn relax(&mut self, py: Python<'_>, n: usize, tol: f64) -> (usize, f64) {
        py.detach(|| {
            let (steps, last) = self.problem.relax(&mut self.q, self.dt, n, tol);
            self.velocity = None;
            (steps, last)
        })
    }

    /// Draw a fresh random initial field and reset the clock.
    #[pyo3(signature = (seed = 0))]
    fn reset(&mut self, seed: u64) {
        self.q = self.problem.random_state(seed);
        self.psi = None;
        self.velocity = None;
        self.steps_taken = 0;
        self.last_dq = 0.0;
        self.last_cfl = 0.0;
    }

    /// Q at every vertex as `(Qxx, Qxy)`, shape `(n_vertices, 2)`.
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.q.n_vertices;
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = self.q.q1[i];
            out[[i, 1]] = self.q.q2[i];
        }
        out.into_pyarray(py)
    }

    /// Replace the field, then re-impose the anchoring on the wall.
    ///
    /// Takes `(n_vertices, 2)`. Use it to resume from a saved state.
    fn set_q(&mut self, q: PyArrayLike2<'_, f64, AllowTypeChange>) -> PyResult<()> {
        let a = q.as_array();
        let n = self.q.n_vertices;
        if a.nrows() != n || a.ncols() != 2 {
            return Err(PyValueError::new_err(format!(
                "expected shape ({n}, 2), got ({}, {})",
                a.nrows(),
                a.ncols()
            )));
        }
        for i in 0..n {
            self.q.q1[i] = a[[i, 0]];
            self.q.q2[i] = a[[i, 1]];
        }
        self.problem.impose_anchoring(&mut self.q);
        self.velocity = None;
        Ok(())
    }

    /// Velocity at every vertex, shape `(n_vertices, 2)`.
    ///
    /// The field from the last step. Before the first step it is solved on
    /// demand from the current Q.
    #[getter]
    fn velocity<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        if self.velocity.is_none() {
            let (vel, psi, _) = if self.full_stress {
                let (s1, s2, sa) = self
                    .problem
                    .beris_edwards_stress_masked(&self.q, &self.elastic_mask);
                self.stokes.solve_stress_warm(
                    &s1,
                    &s2,
                    &sa,
                    self.problem.params.eta,
                    &self.problem.mesh.mesh,
                    self.psi.as_deref(),
                    self.stokes_tol,
                )
            } else {
                self.stokes.solve_warm(
                    &self.q,
                    &self.rotor,
                    &self.problem.ops,
                    &self.problem.mesh.mesh,
                    self.psi.as_deref(),
                    self.stokes_tol,
                )
            };
            self.psi = Some(psi);
            self.velocity = Some(vel);
        }
        let vel = self.velocity.as_ref().expect("solved above");
        let n = self.q.n_vertices;
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = vel.vx(i);
            out[[i, 1]] = vel.vy(i);
        }
        out.into_pyarray(py)
    }

    /// Scalar order parameter `S = sqrt(Tr Q^2)` at every vertex.
    #[getter]
    fn order_parameter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.problem.order_parameter(&self.q)).into_pyarray(py)
    }

    /// Director angle in `[-pi/2, pi/2)` at every vertex.
    #[getter]
    fn director_angle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = (0..self.q.n_vertices)
            .map(|i| 0.5 * self.q.q2[i].atan2(self.q.q1[i]))
            .collect();
        Array1::from(v).into_pyarray(py)
    }

    /// Detected disclinations as `(x, y, charge)`, charge in half units.
    ///
    /// `merge` is the distance within which two cores of the same sign are read
    /// as one; it defaults to one and a half of the largest element.
    #[pyo3(signature = (merge = None))]
    fn defects(&self, merge: Option<f64>) -> Vec<(f64, f64, i32)> {
        let m = merge.unwrap_or(self.merge);
        self.problem.defect_summary(&self.q, m).3
    }

    /// Per-step diagnostics as a dict.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (pos, neg, charge, _) = self.problem.defect_summary(&self.q, self.merge);
        let mut s = self.problem.order_parameter(&self.q);
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if s.is_empty() { 0.0 } else { s[s.len() / 2] };
        let speed_max = match &self.velocity {
            Some(v) => (0..self.q.n_vertices).map(|i| v.speed(i)).fold(0.0, f64::max),
            None => 0.0,
        };
        let d = PyDict::new(py);
        d.set_item("step", self.steps_taken)?;
        d.set_item("time", self.steps_taken as f64 * self.dt)?;
        d.set_item("n_plus", pos)?;
        d.set_item("n_minus", neg)?;
        d.set_item("charge", charge)?;
        d.set_item("s_median", median)?;
        d.set_item("speed_max", speed_max)?;
        d.set_item("worst_dq", self.last_dq)?;
        d.set_item("courant", self.last_cfl)?;
        d.set_item("cg_iterations", self.last_cg)?;
        d.set_item("stokes_iterations", self.last_stokes_iters)?;
        Ok(d)
    }

    /// Landau-de Gennes free energy of the current field.
    fn free_energy(&self) -> f64 {
        self.problem.free_energy(&self.q)
    }

    /// The dimensional constants, as a dict.
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.problem.params;
        let d = PyDict::new(py);
        d.set_item("k_frank", p.k_frank)?;
        d.set_item("a_landau", p.a_landau)?;
        d.set_item("c_landau", p.c_landau)?;
        d.set_item("gamma", p.gamma)?;
        d.set_item("lambda", p.lambda)?;
        d.set_item("zeta", p.zeta)?;
        d.set_item("eta", p.eta)?;
        d.set_item("rho", p.rho)?;
        d.set_item("dt", self.dt)?;
        d.set_item("s0", p.s0())?;
        d.set_item("coherence_length", p.coherence_length())?;
        d.set_item("q_anchor", self.problem.q_anchor)?;
        Ok(d)
    }

    /// Explicit diffusive limit `gamma h^2 / (4 K)` at the smallest element.
    ///
    /// The Frank term is taken implicitly, so this bounds nothing on its own.
    /// The bulk and advective terms are explicit and do have to respect it.
    #[getter]
    fn diffusive_dt_limit(&self) -> f64 {
        let h = self
            .local_h
            .iter()
            .copied()
            .filter(|h| *h > 0.0)
            .fold(f64::INFINITY, f64::min);
        self.problem.params.q_diffusive_dt_limit(h)
    }

    /// Local element size at every vertex.
    #[getter]
    fn local_element_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.local_h.clone()).into_pyarray(py)
    }

    /// The mesh this run is on.
    #[getter]
    fn mesh(&self) -> PyConfinedMesh {
        PyConfinedMesh::from_inner(self.problem.mesh.clone())
    }

    #[getter]
    fn time(&self) -> f64 {
        self.steps_taken as f64 * self.dt
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.steps_taken
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.dt
    }

    /// `"noslip"` or `"freeslip"`.
    #[getter]
    fn wall(&self) -> String {
        self.wall.clone()
    }

    /// Vertices taking the wall condition, the wall itself included.
    #[getter]
    fn wall_vertices(&self) -> usize {
        self.noslip.len()
    }

    /// Vertices where the elastic stress is suppressed.
    #[getter]
    fn elastic_mask_vertices(&self) -> usize {
        self.elastic_mask.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfinedRun(vertices={}, wall={}, dt={:.2e}, step={}, t={:.4})",
            self.q.n_vertices,
            self.wall,
            self.dt,
            self.steps_taken,
            self.steps_taken as f64 * self.dt,
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfinedRun>()?;
    Ok(())
}
