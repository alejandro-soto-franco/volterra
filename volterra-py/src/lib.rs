// volterra-py/src/lib.rs
//
// PyO3 0.25 / numpy 0.25 bindings for the volterra active nematics library.
//
// Exposed to Python (import volterra):
//   volterra.MarsParams          -- physical / numerical parameters
//   volterra.QField2D            -- Q-tensor field with numpy interop
//   volterra.SnapStats           -- per-snapshot statistics
//   volterra.DefectInfo          -- detected disclination
//   volterra.run_mars_component1 -- Component 1 simulation runner
//   volterra.k0_convolution      -- K₀ transfer map (Component 2)
//   volterra.scan_defects        -- holonomy-based defect detection

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use volterra_core::MarsParams;
use volterra_fields::QField2D;
use volterra_solver::{DefectInfo, SnapStats, k0_convolution, run_mars_component1, scan_defects};

// ─────────────────────────────────────────────────────────────────────────────
// PyMarsParams
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the MARS + lipid simulation.
#[pyclass(name = "MarsParams")]
#[derive(Clone)]
pub struct PyMarsParams {
    inner: MarsParams,
}

#[pymethods]
impl PyMarsParams {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nx: usize,
        ny: usize,
        dx: f64,
        dt: f64,
        k_r: f64,
        gamma_r: f64,
        zeta_eff: f64,
        eta: f64,
        a_landau: f64,
        c_landau: f64,
        lambda: f64,
        k_l: f64,
        gamma_l: f64,
        xi_l: f64,
    ) -> PyResult<Self> {
        let p = MarsParams {
            nx, ny, dx, dt, k_r, gamma_r, zeta_eff, eta,
            a_landau, c_landau, lambda, k_l, gamma_l, xi_l,
        };
        p.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: p })
    }

    #[staticmethod]
    fn default_test() -> Self {
        Self { inner: MarsParams::default_test() }
    }

    #[getter] fn nx(&self) -> usize    { self.inner.nx }
    #[getter] fn ny(&self) -> usize    { self.inner.ny }
    #[getter] fn dx(&self) -> f64      { self.inner.dx }
    #[getter] fn dt(&self) -> f64      { self.inner.dt }
    #[getter] fn k_r(&self) -> f64     { self.inner.k_r }
    #[getter] fn gamma_r(&self) -> f64 { self.inner.gamma_r }
    #[getter] fn zeta_eff(&self) -> f64{ self.inner.zeta_eff }
    #[getter] fn eta(&self) -> f64     { self.inner.eta }
    #[getter] fn a_landau(&self) -> f64{ self.inner.a_landau }
    #[getter] fn c_landau(&self) -> f64{ self.inner.c_landau }
    #[getter] fn lambda(&self) -> f64  { self.inner.lambda }
    #[getter] fn k_l(&self) -> f64     { self.inner.k_l }
    #[getter] fn gamma_l(&self) -> f64 { self.inner.gamma_l }
    #[getter] fn xi_l(&self) -> f64    { self.inner.xi_l }

    #[setter] fn set_zeta_eff(&mut self, v: f64) { self.inner.zeta_eff = v; }
    #[setter] fn set_dt(&mut self, v: f64)        { self.inner.dt = v; }
    #[setter] fn set_nx(&mut self, v: usize)      { self.inner.nx = v; }
    #[setter] fn set_ny(&mut self, v: usize)      { self.inner.ny = v; }

    fn defect_length(&self) -> f64 { self.inner.defect_length() }
    fn pi_number(&self) -> f64     { self.inner.pi_number() }
    fn a_eff(&self) -> f64         { self.inner.a_eff() }

    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "MarsParams(nx={}, ny={}, zeta_eff={:.4}, a_eff={:.4}, Pi={:.4})",
            self.inner.nx, self.inner.ny,
            self.inner.zeta_eff, self.inner.a_eff(), self.inner.pi_number(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyQField2D
// ─────────────────────────────────────────────────────────────────────────────

/// 2D Q-tensor field with numpy interop.
///
/// Internal layout: flat Vec<[f64;2]> in row-major order (i*ny + j).
/// numpy conversions use a (nx*ny, 2) f64 array; reshape in Python to (nx, ny, 2).
#[pyclass(name = "QField2D")]
#[derive(Clone)]
pub struct PyQField2D {
    inner: QField2D,
}

#[pymethods]
impl PyQField2D {
    #[staticmethod]
    fn zeros(nx: usize, ny: usize, dx: f64) -> Self {
        Self { inner: QField2D::zeros(nx, ny, dx) }
    }

    #[staticmethod]
    fn uniform(nx: usize, ny: usize, dx: f64, q1: f64, q2: f64) -> Self {
        Self { inner: QField2D::uniform(nx, ny, dx, [q1, q2]) }
    }

    #[staticmethod]
    fn random_perturbation(nx: usize, ny: usize, dx: f64, amplitude: f64, seed: u64) -> Self {
        Self { inner: QField2D::random_perturbation(nx, ny, dx, amplitude, seed) }
    }

    /// Import from a numpy array of shape (nx*ny, 2) or (nx*ny*2,).
    /// Data is copied into the Rust-owned Vec.
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray2<f64>, nx: usize, ny: usize, dx: f64) -> PyResult<Self> {
        let view = arr.as_array();
        let arr_shape = view.shape();
        let expected_n = nx * ny;
        if arr_shape[0] != expected_n || arr_shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "expected shape ({}, 2), got ({}, {})",
                expected_n, arr_shape[0], arr_shape[1],
            )));
        }
        let q: Vec<[f64; 2]> = (0..expected_n)
            .map(|k| [view[[k, 0]], view[[k, 1]]])
            .collect();
        Ok(Self { inner: QField2D { q, nx, ny, dx } })
    }

    /// Export as numpy array of shape (nx*ny, 2). Reshape in Python:
    ///   q_arr = q_field.to_numpy().reshape(nx, ny, 2)
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.q.len();
        let mut arr = Array2::<f64>::zeros((n, 2));
        for (k, [q1, q2]) in self.inner.q.iter().enumerate() {
            arr[[k, 0]] = *q1;
            arr[[k, 1]] = *q2;
        }
        arr.into_pyarray(py)
    }

    /// Scalar order parameter S = 2*sqrt(q1^2 + q2^2), shape (nx*ny,).
    /// Reshape in Python: S = q_field.order_param().reshape(nx, ny)
    fn order_param<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let s: Array1<f64> = Array1::from_iter(
            self.inner.q.iter().map(|[q1, q2]| 2.0 * (q1 * q1 + q2 * q2).sqrt()),
        );
        s.into_pyarray(py)
    }

    /// Director angle theta = atan2(q2, q1)/2 in [-pi/2, pi/2], shape (nx*ny,).
    fn director_angle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let th: Array1<f64> = Array1::from_iter(
            self.inner.q.iter().map(|[q1, q2]| q2.atan2(*q1) / 2.0),
        );
        th.into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    fn mean_order_param(&self) -> f64 { self.inner.mean_order_param() }
    fn max_norm(&self) -> f64         { self.inner.max_norm() }

    fn __repr__(&self) -> String {
        format!(
            "QField2D(nx={}, ny={}, dx={:.3}, <S>={:.4})",
            self.inner.nx, self.inner.ny, self.inner.dx,
            self.inner.mean_order_param(),
        )
    }

    fn __len__(&self) -> usize { self.inner.q.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// PySnapStats
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics from a simulation run.
#[pyclass(name = "SnapStats")]
#[derive(Clone)]
pub struct PySnapStats {
    inner: SnapStats,
}

#[pymethods]
impl PySnapStats {
    #[getter] fn time(&self)           -> f64   { self.inner.time }
    #[getter] fn mean_s(&self)         -> f64   { self.inner.mean_s }
    #[getter] fn n_defects(&self)      -> usize { self.inner.n_defects }
    #[getter] fn n_plus(&self)         -> usize { self.inner.n_plus }
    #[getter] fn n_minus(&self)        -> usize { self.inner.n_minus }
    #[getter] fn defect_density(&self) -> f64   { self.inner.defect_density }

    fn __repr__(&self) -> String {
        format!(
            "SnapStats(t={:.3}, <S>={:.4}, n_def={}, rho_d={:.4e})",
            self.inner.time, self.inner.mean_s,
            self.inner.n_defects, self.inner.defect_density,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyDefectInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Detected topological disclination with plaquette location and charge.
#[pyclass(name = "DefectInfo")]
#[derive(Clone)]
pub struct PyDefectInfo {
    inner: DefectInfo,
}

#[pymethods]
impl PyDefectInfo {
    #[getter] fn plaquette(&self) -> (usize, usize) { self.inner.plaquette }
    #[getter] fn angle(&self) -> f64    { self.inner.angle }
    #[getter] fn charge_sign(&self) -> i32 { self.inner.charge_sign.into() }

    fn __repr__(&self) -> String {
        format!(
            "DefectInfo(plaquette=({},{}), angle={:.3}, charge={})",
            self.inner.plaquette.0, self.inner.plaquette.1,
            self.inner.angle, self.inner.charge_sign,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Run Component 1: single-phase MARS dry active nematic.
///
/// Returns (QField2D, list[SnapStats]).
#[pyfunction]
#[pyo3(name = "run_mars_component1")]
fn run_mars_component1_py(
    q_init: &PyQField2D,
    params: &PyMarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (PyQField2D, Vec<PySnapStats>) {
    let (q_final, stats) = run_mars_component1(
        &q_init.inner,
        &params.inner,
        n_steps,
        snap_every,
    );
    let py_stats = stats.into_iter().map(|s| PySnapStats { inner: s }).collect();
    (PyQField2D { inner: q_final }, py_stats)
}

/// Apply the K₀ transfer map Q_lip = M_SM(Q_rot).
#[pyfunction]
#[pyo3(name = "k0_convolution")]
fn k0_convolution_py(q_rot: &PyQField2D, params: &PyMarsParams) -> PyQField2D {
    PyQField2D { inner: k0_convolution(&q_rot.inner, &params.inner) }
}

/// Holonomy-based disclination detection.
///
/// threshold: rotation angle cutoff in radians (default: pi/2).
#[pyfunction]
#[pyo3(name = "scan_defects", signature = (q, threshold = std::f64::consts::FRAC_PI_2))]
fn scan_defects_py(q: &PyQField2D, threshold: f64) -> Vec<PyDefectInfo> {
    scan_defects(&q.inner, threshold)
        .into_iter()
        .map(|d| PyDefectInfo { inner: d })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────

/// volterra -- active nematics simulation library (PyO3 bindings).
///
/// Build
/// -----
///   pip install maturin
///   cd volterra/   # workspace root
///   maturin develop --release
///
/// Quick start
/// -----------
///   import volterra, numpy as np
///   p  = volterra.MarsParams.default_test()
///   q0 = volterra.QField2D.random_perturbation(p.nx, p.ny, p.dx, 0.001, 42)
///   q_fin, stats = volterra.run_mars_component1(q0, p, 5000, 100)
///   S = np.array(q_fin.order_param()).reshape(p.nx, p.ny)
#[pymodule]
fn volterra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMarsParams>()?;
    m.add_class::<PyQField2D>()?;
    m.add_class::<PySnapStats>()?;
    m.add_class::<PyDefectInfo>()?;
    m.add_function(wrap_pyfunction!(run_mars_component1_py, m)?)?;
    m.add_function(wrap_pyfunction!(k0_convolution_py, m)?)?;
    m.add_function(wrap_pyfunction!(scan_defects_py, m)?)?;
    Ok(())
}
