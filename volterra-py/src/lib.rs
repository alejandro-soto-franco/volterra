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

mod bindings_3d;

use volterra_core::MarsParams;
use volterra_fields::{QField2D, ScalarField2D, VelocityField2D};
use volterra_solver::{
    BechStats, DefectInfo, SnapStats,
    ch_step_etd,
    k0_convolution,
    run_mars_bech,
    run_mars_component1,
    run_mars_component1_hydro,
    scan_defects,
    stokes_solve,
};

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
    #[pyo3(signature = (nx, ny, dx, dt, k_r, gamma_r, zeta_eff, eta, a_landau, c_landau, lambda_, k_l, gamma_l, xi_l, noise_amp=0.0, chi_ms=0.5, kappa_ch=1.0, a_ch=1.0, b_ch=1.0, m_l=0.1))]
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
        lambda_: f64,
        k_l: f64,
        gamma_l: f64,
        xi_l: f64,
        noise_amp: f64,
        chi_ms: f64,
        kappa_ch: f64,
        a_ch: f64,
        b_ch: f64,
        m_l: f64,
    ) -> PyResult<Self> {
        let p = MarsParams {
            nx, ny, dx, dt, k_r, gamma_r, zeta_eff, eta,
            a_landau, c_landau, lambda: lambda_, k_l, gamma_l, xi_l, noise_amp,
            chi_ms, kappa_ch, a_ch, b_ch, m_l,
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
    #[getter] fn lambda_(&self) -> f64    { self.inner.lambda }
    #[getter] fn k_l(&self) -> f64        { self.inner.k_l }
    #[getter] fn gamma_l(&self) -> f64   { self.inner.gamma_l }
    #[getter] fn xi_l(&self) -> f64      { self.inner.xi_l }
    #[getter] fn noise_amp(&self) -> f64 { self.inner.noise_amp }

    #[setter] fn set_noise_amp(&mut self, v: f64) { self.inner.noise_amp = v; }

    #[setter] fn set_zeta_eff(&mut self, v: f64) { self.inner.zeta_eff = v; }
    #[setter] fn set_dt(&mut self, v: f64)        { self.inner.dt = v; }
    #[setter] fn set_nx(&mut self, v: usize)      { self.inner.nx = v; }
    #[setter] fn set_ny(&mut self, v: usize)      { self.inner.ny = v; }

    #[getter] fn chi_ms(&self) -> f64    { self.inner.chi_ms }
    #[getter] fn kappa_ch(&self) -> f64  { self.inner.kappa_ch }
    #[getter] fn a_ch(&self) -> f64      { self.inner.a_ch }
    #[getter] fn b_ch(&self) -> f64      { self.inner.b_ch }
    #[getter] fn m_l(&self) -> f64       { self.inner.m_l }

    #[setter] fn set_chi_ms(&mut self, v: f64)   { self.inner.chi_ms = v; }
    #[setter] fn set_kappa_ch(&mut self, v: f64) { self.inner.kappa_ch = v; }
    #[setter] fn set_a_ch(&mut self, v: f64)     { self.inner.a_ch = v; }
    #[setter] fn set_b_ch(&mut self, v: f64)     { self.inner.b_ch = v; }
    #[setter] fn set_m_l(&mut self, v: f64)      { self.inner.m_l = v; }

    fn defect_length(&self) -> f64       { self.inner.defect_length() }
    fn pi_number(&self) -> f64           { self.inner.pi_number() }
    fn a_eff(&self) -> f64               { self.inner.a_eff() }
    fn ch_coherence_length(&self) -> f64 { self.inner.ch_coherence_length() }
    fn phi_eq(&self) -> f64              { self.inner.phi_eq() }

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
// PyVelocityField2D
// ─────────────────────────────────────────────────────────────────────────────

/// 2D velocity field with numpy interop.
#[pyclass(name = "VelocityField2D")]
#[derive(Clone)]
pub struct PyVelocityField2D {
    inner: VelocityField2D,
}

#[pymethods]
impl PyVelocityField2D {
    /// Export as numpy array of shape (nx*ny, 2). Reshape to (nx, ny, 2) in Python.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.v.len();
        let mut arr = Array2::<f64>::zeros((n, 2));
        for (k, [vx, vy]) in self.inner.v.iter().enumerate() {
            arr[[k, 0]] = *vx;
            arr[[k, 1]] = *vy;
        }
        arr.into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    fn __repr__(&self) -> String {
        format!("VelocityField2D(nx={}, ny={}, dx={:.3})", self.inner.nx, self.inner.ny, self.inner.dx)
    }
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

/// Run Component 1 with full hydrodynamic flow coupling (spectral Stokes).
///
/// At each time step the Stokes velocity field is re-computed from the active
/// stress `σ^a = ζ_eff Q` via the stream-function biharmonic equation solved
/// spectrally. This enables the active flow instability that drives turbulence
/// with the scaling `ρ_d ~ ζ_eff / K_r`.
///
/// Returns `(QField2D, list[SnapStats])`.
#[pyfunction]
#[pyo3(name = "run_mars_component1_hydro")]
fn run_mars_component1_hydro_py(
    q_init: &PyQField2D,
    params: &PyMarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (PyQField2D, Vec<PySnapStats>) {
    let (q_final, stats) = run_mars_component1_hydro(
        &q_init.inner,
        &params.inner,
        n_steps,
        snap_every,
    );
    let py_stats = stats.into_iter().map(|s| PySnapStats { inner: s }).collect();
    (PyQField2D { inner: q_final }, py_stats)
}

/// Solve the 2D incompressible Stokes equation for the active velocity field.
///
/// Returns the velocity field driven by `σ^a = ζ_eff Q` via spectral inversion
/// of the stream-function biharmonic equation.
#[pyfunction]
#[pyo3(name = "stokes_solve")]
fn stokes_solve_py(q: &PyQField2D, params: &PyMarsParams) -> PyVelocityField2D {
    PyVelocityField2D { inner: stokes_solve(&q.inner, &params.inner) }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyScalarField2D
// ─────────────────────────────────────────────────────────────────────────────

/// 2D scalar field φ(x,y) with numpy interop.
///
/// Used for the lipid volume fraction in the BECH simulation.
/// numpy conversions use a 1D array of length nx*ny; reshape in Python to (nx, ny).
#[pyclass(name = "ScalarField2D")]
#[derive(Clone)]
pub struct PyScalarField2D {
    inner: ScalarField2D,
}

#[pymethods]
impl PyScalarField2D {
    #[staticmethod]
    fn zeros(nx: usize, ny: usize, dx: f64) -> Self {
        Self { inner: ScalarField2D::zeros(nx, ny, dx) }
    }

    #[staticmethod]
    fn uniform(nx: usize, ny: usize, dx: f64, val: f64) -> Self {
        Self { inner: ScalarField2D::uniform(nx, ny, dx, val) }
    }

    /// Import from a 1D numpy array of length nx*ny.
    #[staticmethod]
    fn from_numpy(arr: numpy::PyReadonlyArray1<f64>, nx: usize, ny: usize, dx: f64) -> PyResult<Self> {
        let view = arr.as_array();
        if view.len() != nx * ny {
            return Err(PyValueError::new_err(format!(
                "expected length {}, got {}", nx * ny, view.len()
            )));
        }
        let phi: Vec<f64> = view.iter().copied().collect();
        Ok(Self { inner: ScalarField2D { phi, nx, ny, dx } })
    }

    /// Export as 1D numpy array of length nx*ny. Reshape in Python to (nx, ny).
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let arr = numpy::ndarray::Array1::from_vec(self.inner.phi.clone());
        arr.into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    fn mean_value(&self) -> f64      { self.inner.mean_value() }
    fn variance(&self) -> f64        { self.inner.variance() }
    fn max_value(&self) -> f64       { self.inner.max_value() }
    fn min_value(&self) -> f64       { self.inner.min_value() }
    fn mean_gradient_sq(&self) -> f64 { self.inner.mean_gradient_sq() }

    fn __repr__(&self) -> String {
        format!(
            "ScalarField2D(nx={}, ny={}, dx={:.3}, <φ>={:.4}, Var[φ]={:.4e})",
            self.inner.nx, self.inner.ny, self.inner.dx,
            self.inner.mean_value(), self.inner.variance(),
        )
    }

    fn __len__(&self) -> usize { self.inner.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyBechStats
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics from a BECH (full two-field) simulation run.
#[pyclass(name = "BechStats")]
#[derive(Clone)]
pub struct PyBechStats {
    inner: BechStats,
}

#[pymethods]
impl PyBechStats {
    #[getter] fn time(&self)              -> f64   { self.inner.time }
    #[getter] fn mean_s(&self)            -> f64   { self.inner.mean_s }
    #[getter] fn n_defects(&self)         -> usize { self.inner.n_defects }
    #[getter] fn n_plus(&self)            -> usize { self.inner.n_plus }
    #[getter] fn n_minus(&self)           -> usize { self.inner.n_minus }
    #[getter] fn defect_density(&self)    -> f64   { self.inner.defect_density }
    #[getter] fn mean_phi(&self)          -> f64   { self.inner.mean_phi }
    #[getter] fn phi_variance(&self)      -> f64   { self.inner.phi_variance }
    #[getter] fn mean_grad_phi_sq(&self)  -> f64   { self.inner.mean_grad_phi_sq }

    fn __repr__(&self) -> String {
        format!(
            "BechStats(t={:.3}, <S>={:.4}, n_def={}, <φ>={:.4}, Var[φ]={:.4e})",
            self.inner.time, self.inner.mean_s,
            self.inner.n_defects, self.inner.mean_phi, self.inner.phi_variance,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BECH Python bindings
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full Beris-Edwards-Cahn-Hilliard (BECH) simulation.
///
/// Couples the active rotor Q-field (BE + Stokes) to the lyotropic lipid
/// volume fraction φ_l (CH-ETD1 + Maier-Saupe).  See `run_mars_bech` in
/// volterra-solver for the full algorithm description.
///
/// Returns `(QField2D, ScalarField2D, list[BechStats])`.
#[pyfunction]
#[pyo3(name = "run_mars_bech")]
fn run_mars_bech_py(
    q_init: &PyQField2D,
    phi_init: &PyScalarField2D,
    params: &PyMarsParams,
    n_steps: usize,
    snap_every: usize,
) -> (PyQField2D, PyScalarField2D, Vec<PyBechStats>) {
    let (q_fin, phi_fin, stats) = run_mars_bech(
        &q_init.inner,
        &phi_init.inner,
        &params.inner,
        n_steps,
        snap_every,
    );
    let py_stats = stats.into_iter().map(|s| PyBechStats { inner: s }).collect();
    (PyQField2D { inner: q_fin }, PyScalarField2D { inner: phi_fin }, py_stats)
}

/// Advance the CH field by one ETD1 step.
///
/// Useful for custom time-loop control from Python (e.g., alternating
/// parameter sweeps or checkpointing at non-uniform intervals).
#[pyfunction]
#[pyo3(name = "ch_step_etd")]
fn ch_step_etd_py(
    phi: &PyScalarField2D,
    q_lip: &PyQField2D,
    v: &PyVelocityField2D,
    params: &PyMarsParams,
) -> PyScalarField2D {
    PyScalarField2D {
        inner: ch_step_etd(&phi.inner, &q_lip.inner, &v.inner, &params.inner),
    }
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
    m.add_class::<PyScalarField2D>()?;
    m.add_class::<PyVelocityField2D>()?;
    m.add_class::<PySnapStats>()?;
    m.add_class::<PyBechStats>()?;
    m.add_class::<PyDefectInfo>()?;
    m.add_function(wrap_pyfunction!(run_mars_component1_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_mars_component1_hydro_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_mars_bech_py, m)?)?;
    m.add_function(wrap_pyfunction!(stokes_solve_py, m)?)?;
    m.add_function(wrap_pyfunction!(k0_convolution_py, m)?)?;
    m.add_function(wrap_pyfunction!(scan_defects_py, m)?)?;
    m.add_function(wrap_pyfunction!(ch_step_etd_py, m)?)?;
    bindings_3d::register(m)?;
    Ok(())
}
