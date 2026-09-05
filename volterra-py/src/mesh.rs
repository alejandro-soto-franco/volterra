// volterra-py/src/mesh.rs
//
// PyO3 bindings for the confined boundary geometry and its conforming mesh.
//
// Exposed to Python (import volterra):
//   volterra.PlaneCurve    -- a closed wall: analytic, tabulated, or from a callable
//   volterra.ConfinedMesh  -- the graded mesh of its interior, with the wall tagged
//   volterra.confined_mesh -- build one from a curve

use std::f64::consts::TAU;
use std::sync::Arc;

use numpy::ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use volterra_dec::confined::{ConfinedMesh2, Epitrochoid, MeshOpts, confined_mesh};
use volterra_dec::confined_ldg::anchored_q;
use volterra_dec::curve::{PlaneCurve, PolyCurve};

// ─────────────────────────────────────────────────────────────────────────────
// PyPlaneCurve
// ─────────────────────────────────────────────────────────────────────────────

/// A closed plane curve, the wall a confined run is meshed against.
///
/// Three constructors, in increasing generality:
///
/// * `PlaneCurve.epitrochoid(q, d, r)` is analytic, and its cusp parameters are
///   exact. `q = 1 + k/2` gives `k` cusps, `d` in `(0, 1]` regularises them,
///   and `r` is the outer scale.
/// * `PlaneCurve.from_points(points, features=None)` interpolates a closed
///   table of points with a periodic cubic spline. The parameter is the sample
///   index and the period is the sample count.
/// * `PlaneCurve.from_callable(f, samples=1024, period=2*pi, features=None)`
///   evaluates `f(u) -> (x, y)` on a uniform grid of the caller's own parameter
///   and splines the result.
///
/// `features` are the parameters of the corners and cusps that set the local
/// element size. Leave them out for a smooth wall, or pass `"auto"` on either
/// splined constructor to read them off the curvature.
/// What a `features` argument asked for: a list of parameters, or the string
/// `"auto"`, which reads them off the curvature.
enum Features {
    Given(Vec<f64>),
    Auto,
}

/// Read `features`: a sequence of floats, `"auto"`, or nothing.
fn read_features(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Features> {
    let Some(o) = obj else {
        return Ok(Features::Given(Vec::new()));
    };
    if o.is_none() {
        return Ok(Features::Given(Vec::new()));
    }
    if let Ok(word) = o.extract::<String>() {
        if word == "auto" {
            return Ok(Features::Auto);
        }
        return Err(PyValueError::new_err(format!(
            "features takes a sequence of parameters or the string \"auto\", not \"{word}\""
        )));
    }
    let v: Vec<f64> = o.extract().map_err(|_| {
        PyValueError::new_err("features must be a sequence of floats or the string \"auto\"")
    })?;
    Ok(Features::Given(v))
}

#[pyclass(name = "PlaneCurve", module = "volterra", from_py_object)]
#[derive(Clone)]
pub struct PyPlaneCurve {
    inner: Arc<dyn PlaneCurve>,
    label: String,
}

#[pymethods]
impl PyPlaneCurve {
    /// The regularised epitrochoid, `k = 2(q - 1)` cusps at outer scale `r`.
    ///
    /// ```text
    /// x(u) = a [ (k+1) cos u + d cos((k+1) u) ],   a = r / (k + 2)
    /// y(u) = a [ (k+1) sin u + d sin((k+1) u) ]
    /// ```
    ///
    /// `d = 1` is the epicycloid with true cusps; `d = 0` is a circle. `q = 1`
    /// is the circle, `1.5` the cardioid, `2` the nephroid, `2.5` the
    /// trefoiloid, `3` the quatrefoiloid.
    #[staticmethod]
    #[pyo3(signature = (q, d = 0.99, r = 1.0))]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn epitrochoid(q: f64, d: f64, r: f64) -> PyResult<Self> {
        if !q.is_finite() || q < 1.0 {
            return Err(PyValueError::new_err("q must be at least 1"));
        }
        let k = 2.0 * (q - 1.0);
        if (k - k.round()).abs() > 1e-9 {
            return Err(PyValueError::new_err(
                "q must be a half-integer, so that 2(q - 1) is the cusp count",
            ));
        }
        if !(d > 0.0 && d <= 1.0) {
            return Err(PyValueError::new_err("d must lie in (0, 1]"));
        }
        if !(r > 0.0) {
            return Err(PyValueError::new_err("r must be positive"));
        }
        Ok(Self {
            inner: Arc::new(Epitrochoid { q, d, r }),
            label: format!("epitrochoid(q={q}, d={d}, r={r})"),
        })
    }

    /// A wall given as a closed table of points, in order along it.
    ///
    /// `points` is `(n, 2)` and must not repeat the first point at the end: the
    /// curve closes on its own. The parametrisation is the sample index, so
    /// `curve.point(k)` returns row `k` exactly, and sampling more densely where
    /// the wall turns is how a sharp feature is described.
    ///
    /// `features` are sample indices, as floats, or `"auto"`, which reports the
    /// samples whose radius of curvature falls below a quarter of the curve's
    /// circle-equivalent radius, one per run, at the tightest of each. That
    /// scale is the wall's own, so the answer is independent of the units the
    /// points came in and of the mesh they are later meshed at. A circle sits
    /// at exactly that radius everywhere and reports none.
    ///
    /// A true corner stays a corner under the spline, and the boundary sampling
    /// steps across it in one go: the director then turns by more than a
    /// quarter turn in a single step and [`ConfinedMesh::imposed_charge`]
    /// reports the wrap, with a worst step past 90 degrees and a non-zero count
    /// beside it. A square sampled at 40 points a side reads 0.5 rather than 1,
    /// and refining to 160 a side reads the same, since the corner is scale
    /// free. Round the corner over a radius the sampling resolves.
    #[staticmethod]
    #[pyo3(signature = (points, features = None))]
    fn from_points(
        points: PyArrayLike2<'_, f64, AllowTypeChange>,
        features: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let a = points.as_array();
        if a.ncols() != 2 {
            return Err(PyValueError::new_err("points must have shape (n, 2)"));
        }
        let pts: Vec<[f64; 2]> = a.rows().into_iter().map(|r| [r[0], r[1]]).collect();
        let n = pts.len();
        let bad = "a wall needs at least three points enclosing a non-zero area; \
                   check for a repeated first point or a self-intersection";
        let curve = match read_features(features)? {
            Features::Auto => PolyCurve::new_auto(&pts),
            Features::Given(feats) => {
                for f in &feats {
                    if !f.is_finite() || *f < 0.0 || *f >= n as f64 {
                        return Err(PyValueError::new_err(format!(
                            "feature {f} is outside the parameter range [0, {n})"
                        )));
                    }
                }
                PolyCurve::new(&pts, &feats)
            }
        }
        .ok_or_else(|| PyValueError::new_err(bad))?;
        let found = curve.features().len();
        Ok(Self {
            inner: Arc::new(curve),
            label: format!("from_points(n={n}, features={found})"),
        })
    }

    /// A wall given as a parametrisation `f(u) -> (x, y)` over `[0, period)`.
    ///
    /// `f` is evaluated at `samples` points, uniformly in `u`, and the result is
    /// splined. `features` are given in the caller's own parameter and converted
    /// to sample indices, so a cusp at `u = pi` stays at `u = pi`, or `"auto"`
    /// to read them off the curvature.
    ///
    /// Resolution is the sample count: a wall that turns inside one sample
    /// interval is described by raising `samples`, or by tabulating it directly
    /// with `from_points`.
    #[staticmethod]
    #[pyo3(signature = (f, samples = 1024, period = TAU, features = None))]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn from_callable(
        f: &Bound<'_, PyAny>,
        samples: usize,
        period: f64,
        features: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if samples < 3 {
            return Err(PyValueError::new_err("samples must be at least 3"));
        }
        if !(period > 0.0) || !period.is_finite() {
            return Err(PyValueError::new_err("period must be positive and finite"));
        }
        let mut pts = Vec::with_capacity(samples);
        for i in 0..samples {
            let u = period * i as f64 / samples as f64;
            let out = f.call1((u,))?;
            let (x, y): (f64, f64) = out.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "f({u}) must return two floats (x, y)"
                ))
            })?;
            if !x.is_finite() || !y.is_finite() {
                return Err(PyValueError::new_err(format!("f({u}) returned a non-finite point")));
            }
            pts.push([x, y]);
        }
        let scale = samples as f64 / period;
        let bad = "the sampled wall encloses no area; check the parametrisation and the period";
        let curve = match read_features(features)? {
            Features::Auto => PolyCurve::new_auto(&pts),
            Features::Given(given) => {
                let feats: Vec<f64> = given
                    .iter()
                    .map(|u| {
                        let w = u - period * (u / period).floor();
                        w * scale
                    })
                    .collect();
                PolyCurve::new(&pts, &feats)
            }
        }
        .ok_or_else(|| PyValueError::new_err(bad))?;
        Ok(Self {
            inner: Arc::new(curve),
            label: format!("from_callable(samples={samples}, period={period})"),
        })
    }

    /// Parameter length of one circuit.
    #[getter]
    fn period(&self) -> f64 {
        self.inner.period()
    }

    /// Parameters of the corners and cusps.
    #[getter]
    fn features(&self) -> Vec<f64> {
        self.inner.features()
    }

    /// Position at parameter `u`.
    fn point(&self, u: f64) -> (f64, f64) {
        let p = self.inner.point(u);
        (p[0], p[1])
    }

    /// Unit tangent at parameter `u`.
    fn tangent(&self, u: f64) -> (f64, f64) {
        let t = self.inner.tangent(u);
        (t[0], t[1])
    }

    /// Inward unit normal at parameter `u`.
    fn inward_normal(&self, u: f64) -> (f64, f64) {
        let n = self.inner.inward_normal(u);
        (n[0], n[1])
    }

    /// Radius of curvature at parameter `u`, `inf` where the wall is straight.
    fn curvature_radius(&self, u: f64) -> f64 {
        self.inner.curvature_radius(u)
    }

    /// `|r'(u)|`, the speed of the parametrisation.
    fn speed(&self, u: f64) -> f64 {
        self.inner.speed(u)
    }

    /// `n` points spread uniformly in the parameter, as an `(n, 2)` array.
    ///
    /// This is the wall to plot, and it is independent of any mesh.
    fn sample<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if n < 1 {
            return Err(PyValueError::new_err("n must be positive"));
        }
        let period = self.inner.period();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let p = self.inner.point(period * i as f64 / n as f64);
            out[[i, 0]] = p[0];
            out[[i, 1]] = p[1];
        }
        Ok(out.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("PlaneCurve.{}", self.label)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyConfinedMesh
// ─────────────────────────────────────────────────────────────────────────────

/// A boundary-conforming graded mesh of a curve's interior.
///
/// The wall's own samples are the first `n_boundary` vertices, in order along
/// the curve, so the anchoring indexes them directly.
#[pyclass(name = "ConfinedMesh", module = "volterra")]
pub struct PyConfinedMesh {
    inner: ConfinedMesh2,
}

impl PyConfinedMesh {
    /// Wrap an already-built mesh, so a run can hand its own domain back.
    pub(crate) fn from_inner(inner: ConfinedMesh2) -> Self {
        Self { inner }
    }

    /// A copy of the mesh, for a consumer that takes it by value.
    pub(crate) fn clone_inner(&self) -> ConfinedMesh2 {
        self.inner.clone()
    }
}

#[pymethods]
impl PyConfinedMesh {
    /// Vertex positions, `(n_vertices, 2)`.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.mesh.n_vertices();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = self.inner.mesh.vertices[i].x;
            out[[i, 1]] = self.inner.mesh.vertices[i].y;
        }
        out.into_pyarray(py)
    }

    /// Triangle vertex indices, `(n_triangles, 3)`.
    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let t = &self.inner.mesh.simplices;
        let mut out = Array2::<i64>::zeros((t.len(), 3));
        for (i, tri) in t.iter().enumerate() {
            for c in 0..3 {
                out[[i, c]] = tri[c] as i64;
            }
        }
        out.into_pyarray(py)
    }

    /// Indices of the vertices on the wall, in order along it.
    #[getter]
    fn boundary_vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.boundary_vertices.iter().map(|&i| i as i64).collect();
        Array1::from(v).into_pyarray(py)
    }

    /// Curve parameter at each boundary vertex.
    #[getter]
    fn boundary_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.inner.boundary_params.clone()).into_pyarray(py)
    }

    /// Inward unit normal at each boundary vertex, `(n_boundary, 2)`.
    #[getter]
    fn boundary_normals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.boundary_normals.len();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = self.inner.boundary_normals[i][0];
            out[[i, 1]] = self.inner.boundary_normals[i][1];
        }
        out.into_pyarray(py)
    }

    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.quality.vertices
    }

    #[getter]
    fn n_triangles(&self) -> usize {
        self.inner.quality.triangles
    }

    #[getter]
    fn n_boundary(&self) -> usize {
        self.inner.quality.boundary_vertices
    }

    /// Smallest angle over all triangles, in degrees.
    ///
    /// The element shape measure. The mesher aims to keep it above about 25
    /// degrees, and a strong grading ratio pulls it well below that: a nephroid
    /// at `h_bulk = 1.0` and `h_min = 0.25` measures 6.4 degrees against 27.6
    /// for the same curve at `h_bulk = 3.0` and `h_min = 0.8`.
    ///
    /// The quantity that reaches the operator is [`Self::worst_cot_weight`],
    /// since a cotangent weight turns negative at an angle past a right angle
    /// rather than at a small one.
    #[getter]
    fn min_angle_deg(&self) -> f64 {
        self.inner.quality.min_angle_deg
    }

    #[getter]
    fn max_angle_deg(&self) -> f64 {
        self.inner.quality.max_angle_deg
    }

    /// Triangles with an angle past a right angle.
    #[getter]
    fn obtuse(&self) -> usize {
        self.inner.quality.obtuse
    }

    /// The most negative cotangent weight in the Laplacian; zero when every
    /// weight is non-negative.
    #[getter]
    fn worst_cot_weight(&self) -> f64 {
        self.inner.quality.worst_cot_weight
    }

    #[getter]
    fn min_edge(&self) -> f64 {
        self.inner.quality.min_edge
    }

    #[getter]
    fn max_edge(&self) -> f64 {
        self.inner.quality.max_edge
    }

    #[getter]
    fn min_area(&self) -> f64 {
        self.inner.quality.min_area
    }

    /// Total defect charge the anchoring imposes, read off this mesh's own
    /// boundary.
    ///
    /// Returns `(charge, worst_step_deg, steps_over_90_deg)`. The charge is the
    /// winding of the anchored director as a line field, in units of a full
    /// defect, so tangential anchoring on a smooth wall returns 1. The two
    /// diagnostics say whether the sampling resolved the wall: the doubled-angle
    /// increment between neighbouring boundary vertices has to stay under a half
    /// turn for the wrap to book the right branch, and a lattice mask at a cusp
    /// is exactly where that fails.
    ///
    /// Read this number rather than the geometry parameters. A mesh too coarse
    /// for its own wall reports the charge it will actually impose.
    #[pyo3(signature = (q_anchor = 1.0))]
    fn imposed_charge(&self, q_anchor: f64) -> (f64, f64, usize) {
        self.inner.imposed_charge(q_anchor)
    }

    /// Angle of the outward normal at each boundary vertex, in radians.
    ///
    /// This is the `theta` the anchoring is a function of.
    #[getter]
    fn boundary_normal_angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self
            .inner
            .boundary_normals
            .iter()
            .map(|n| (-n[1]).atan2(-n[0]))
            .collect();
        Array1::from(v).into_pyarray(py)
    }

    /// The Dirichlet `(Qxx, Qxy)` the anchoring pins at each boundary vertex,
    /// `(n_boundary, 2)`.
    ///
    /// At a wall site with outward normal at angle `theta`,
    ///
    /// ```text
    /// nn = (cos(q theta), sin(q theta))
    /// Qxx = s0 (nn_y^2 - 1/2)      Qxy = -s0 nn_x nn_y
    /// ```
    ///
    /// which is `Q = s0 (m m - I/2)` for `m` the vector at angle `q theta`
    /// turned by a quarter turn. At `q_anchor = 1` that `m` is the wall tangent,
    /// which is planar anchoring.
    #[pyo3(signature = (q_anchor = 1.0, s0 = 1.0))]
    fn anchoring_q<'py>(
        &self,
        py: Python<'py>,
        q_anchor: f64,
        s0: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.boundary_normals.len();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let nrm = self.inner.boundary_normals[i];
            let theta = (-nrm[1]).atan2(-nrm[0]);
            let (q1, q2) = anchored_q(theta, q_anchor, s0);
            out[[i, 0]] = q1;
            out[[i, 1]] = q2;
        }
        out.into_pyarray(py)
    }

    /// The anchored director at each boundary vertex, `(n_boundary, 2)`.
    ///
    /// The unit vector `m` of `anchoring_q`, which is the wall tangent at
    /// `q_anchor = 1`.
    #[pyo3(signature = (q_anchor = 1.0))]
    fn anchoring_director<'py>(
        &self,
        py: Python<'py>,
        q_anchor: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.boundary_normals.len();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let nrm = self.inner.boundary_normals[i];
            let theta = (-nrm[1]).atan2(-nrm[0]);
            let a = q_anchor * theta;
            out[[i, 0]] = a.sin();
            out[[i, 1]] = -a.cos();
        }
        out.into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfinedMesh(vertices={}, triangles={}, boundary={}, min_angle={:.2} deg)",
            self.inner.quality.vertices,
            self.inner.quality.triangles,
            self.inner.quality.boundary_vertices,
            self.inner.quality.min_angle_deg,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// confined_mesh
// ─────────────────────────────────────────────────────────────────────────────

/// Build a boundary-conforming graded mesh of a curve's interior.
///
/// Three stages. The wall is sampled first, so no arc exceeds `boundary_frac` of
/// the local radius of curvature, and those samples become the mesh's first
/// vertices in order. Layers then march inward along the normal at the local
/// size, so the elements stay near-isotropic as the grading tightens. The
/// remainder is filled by dart throwing at `h_bulk`, and every point is then
/// triangulated, smoothed and re-triangulated.
///
/// Parameters
/// ----------
/// curve : PlaneCurve
///     The wall.
/// h_bulk : float
///     Target element size away from any feature.
/// h_min : float
///     Smallest element size, used at a cusp. Set it from the local radius of
///     curvature: four elements across the tip is what a charge measurement
///     needs. The explicit diffusive limit is `gamma h^2 / (4 K)`, and it has to
///     stay above the timestep, so this parameter sets the run length.
/// grade : float
///     Geometric growth of element size per layer, marching inward.
/// boundary_frac : float
///     Boundary arc as a fraction of the local radius of curvature.
/// smooth_passes : int
///     Passes of area-weighted smoothing over the interior vertices. The wall
///     never moves.
/// seed : int
///     Seed for the dart throwing, so a mesh is reproducible.
/// cusp_edge : float
///     Length of the two boundary edges meeting at a cusp vertex, for the sharp
///     treatment at `d = 1`. Zero samples the feature naively.
///
/// Cost
/// ----
/// The Delaunay stage tests every candidate point against every triangle, so
/// the time is quadratic in the vertex count: measured on a nephroid at
/// `d = 0.85` and `r = 60`, 3.5k vertices take 0.4 s, 7.8k take 2.1 s and 12.1k
/// take 4.9 s, and `smooth_passes` re-triangulates on top of that. An element
/// count past about 50k runs into minutes. Halving `h_bulk` quadruples the
/// vertex count and so costs a factor of sixteen.
///
/// Returns
/// -------
/// ConfinedMesh
///
/// Examples
/// --------
/// >>> import volterra as v
/// >>> c = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=98.0)
/// >>> m = v.confined_mesh(c, h_bulk=1.5, h_min=0.3)
/// >>> m.min_angle_deg > 25.0
/// True
/// >>> charge, worst_deg, over = m.imposed_charge(1.0)
#[pyfunction]
#[pyo3(name = "confined_mesh")]
#[pyo3(signature = (
    curve,
    h_bulk = 1.0,
    h_min = 0.05,
    grade = 1.3,
    boundary_frac = 0.25,
    smooth_passes = 8,
    seed = 0,
    cusp_edge = 0.0
))]
#[allow(clippy::too_many_arguments)]
// The negations below are deliberate: `!(x > 0.0)` rejects NaN as well as a
// non-positive value, which a positive test would let through.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn confined_mesh_py(
    py: Python<'_>,
    curve: &PyPlaneCurve,
    h_bulk: f64,
    h_min: f64,
    grade: f64,
    boundary_frac: f64,
    smooth_passes: usize,
    seed: u64,
    cusp_edge: f64,
) -> PyResult<PyConfinedMesh> {
    if !(h_bulk > 0.0) || !h_bulk.is_finite() {
        return Err(PyValueError::new_err("h_bulk must be positive and finite"));
    }
    if !(h_min > 0.0) || !h_min.is_finite() {
        return Err(PyValueError::new_err("h_min must be positive and finite"));
    }
    if h_min > h_bulk {
        return Err(PyValueError::new_err(format!(
            "h_min {h_min} exceeds h_bulk {h_bulk}; the smallest element cannot be the largest"
        )));
    }
    if !(grade > 1.0) {
        return Err(PyValueError::new_err(
            "grade must exceed 1: it is the geometric growth of element size per layer",
        ));
    }
    if !(boundary_frac > 0.0 && boundary_frac <= 1.0) {
        return Err(PyValueError::new_err("boundary_frac must lie in (0, 1]"));
    }
    if !cusp_edge.is_finite() || cusp_edge < 0.0 {
        return Err(PyValueError::new_err("cusp_edge must be non-negative"));
    }
    // The mesher grades the element size geometrically towards the smallest, so
    // the layer count grows with the logarithm of the ratio. A ratio past a
    // million means the feature has driven the element size to zero.
    if h_bulk / h_min > 1e6 {
        return Err(PyValueError::new_err(format!(
            "grading ratio h_bulk / h_min = {:.3e} is past 1e6; \
             raise h_min, or excise the feature with cusp_edge",
            h_bulk / h_min
        )));
    }

    let opts = MeshOpts {
        h_bulk,
        h_min,
        grade,
        boundary_frac,
        smooth_passes,
        seed,
        cusp_edge,
    };
    // The curve was sampled into Rust data at construction, so the mesher never
    // calls back into Python and the interpreter is free while it runs. A mesh
    // of any size takes seconds to minutes, which is long enough for that to
    // matter to anything else in the process.
    let curve = curve.inner.clone();
    let inner = py.detach(move || confined_mesh(curve, opts));
    if inner.quality.triangles == 0 {
        return Err(PyValueError::new_err(
            "the mesh came out empty; check that the curve encloses an area at this h_bulk",
        ));
    }
    Ok(PyConfinedMesh { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlaneCurve>()?;
    m.add_class::<PyConfinedMesh>()?;
    m.add_function(wrap_pyfunction!(confined_mesh_py, m)?)?;
    Ok(())
}
