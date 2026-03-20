// volterra-py/src/bindings_3d.rs
//
// PyO3 bindings for 3D simulation types.
//
// Exposed to Python (import volterra):
//   volterra.MarsParams3D   -- physical / numerical parameters for the 3D simulation
//   volterra.QField3D       -- 3D Q-tensor field with numpy interop
//   volterra.VelocityField3D -- 3D velocity field with numpy interop
//   volterra.ScalarField3D  -- 3D scalar (concentration / pressure) field with numpy interop
//   volterra.SnapStats3D    -- per-snapshot statistics for the dry active nematic run
//   volterra.BechStats3D    -- per-snapshot statistics for the full BECH run

use numpy::ndarray::{Array1, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volterra_core::MarsParams3D;
use volterra_fields::{QField3D, ScalarField3D, VelocityField3D};
use volterra_solver::{BechStats3D, SnapStats3D};

// ─────────────────────────────────────────────────────────────────────────────
// PyMarsParams3D
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the 3D MARS + lipid simulation.
#[pyclass(name = "MarsParams3D")]
#[derive(Clone)]
pub struct PyMarsParams3D {
    inner: MarsParams3D,
}

#[pymethods]
impl PyMarsParams3D {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        nx, ny, nz, dx, dt,
        k_r, gamma_r, zeta_eff, eta,
        a_landau, c_landau, b_landau, lambda_,
        k_l, gamma_l, xi_l, chi_ms,
        kappa_ch, a_ch, b_ch, m_l,
        chi_a=0.0, b0=1.0, omega_b=1.0, noise_amp=0.0
    ))]
    fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        k_r: f64,
        gamma_r: f64,
        zeta_eff: f64,
        eta: f64,
        a_landau: f64,
        c_landau: f64,
        b_landau: f64,
        lambda_: f64,
        k_l: f64,
        gamma_l: f64,
        xi_l: f64,
        chi_ms: f64,
        kappa_ch: f64,
        a_ch: f64,
        b_ch: f64,
        m_l: f64,
        chi_a: f64,
        b0: f64,
        omega_b: f64,
        noise_amp: f64,
    ) -> PyResult<Self> {
        let p = MarsParams3D {
            nx, ny, nz, dx, dt,
            k_r, gamma_r, zeta_eff, eta,
            a_landau, c_landau, b_landau,
            lambda: lambda_,
            noise_amp, chi_a, b0, omega_b,
            k_l, gamma_l, xi_l, chi_ms,
            kappa_ch, a_ch, b_ch, m_l,
        };
        p.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: p })
    }

    /// Construct the default test parameter set (16x16x16 grid, active turbulent phase).
    #[staticmethod]
    fn default_test() -> Self {
        Self { inner: MarsParams3D::default_test() }
    }

    // ── Getters ────────────────────────────────────────────────────────────

    #[getter] fn nx(&self) -> usize       { self.inner.nx }
    #[getter] fn ny(&self) -> usize       { self.inner.ny }
    #[getter] fn nz(&self) -> usize       { self.inner.nz }
    #[getter] fn dx(&self) -> f64         { self.inner.dx }
    #[getter] fn dt(&self) -> f64         { self.inner.dt }
    #[getter] fn k_r(&self) -> f64        { self.inner.k_r }
    #[getter] fn gamma_r(&self) -> f64    { self.inner.gamma_r }
    #[getter] fn zeta_eff(&self) -> f64   { self.inner.zeta_eff }
    #[getter] fn eta(&self) -> f64        { self.inner.eta }
    #[getter] fn a_landau(&self) -> f64   { self.inner.a_landau }
    #[getter] fn c_landau(&self) -> f64   { self.inner.c_landau }
    #[getter] fn b_landau(&self) -> f64   { self.inner.b_landau }
    #[getter] fn lambda_(&self) -> f64    { self.inner.lambda }
    #[getter] fn noise_amp(&self) -> f64  { self.inner.noise_amp }
    #[getter] fn chi_a(&self) -> f64      { self.inner.chi_a }
    #[getter] fn b0(&self) -> f64         { self.inner.b0 }
    #[getter] fn omega_b(&self) -> f64    { self.inner.omega_b }
    #[getter] fn k_l(&self) -> f64        { self.inner.k_l }
    #[getter] fn gamma_l(&self) -> f64    { self.inner.gamma_l }
    #[getter] fn xi_l(&self) -> f64       { self.inner.xi_l }
    #[getter] fn chi_ms(&self) -> f64     { self.inner.chi_ms }
    #[getter] fn kappa_ch(&self) -> f64   { self.inner.kappa_ch }
    #[getter] fn a_ch(&self) -> f64       { self.inner.a_ch }
    #[getter] fn b_ch(&self) -> f64       { self.inner.b_ch }
    #[getter] fn m_l(&self) -> f64        { self.inner.m_l }

    // ── Setters ────────────────────────────────────────────────────────────

    #[setter] fn set_noise_amp(&mut self, v: f64)  { self.inner.noise_amp = v; }
    #[setter] fn set_zeta_eff(&mut self, v: f64)   { self.inner.zeta_eff = v; }
    #[setter] fn set_dt(&mut self, v: f64)          { self.inner.dt = v; }
    #[setter] fn set_nx(&mut self, v: usize)        { self.inner.nx = v; }
    #[setter] fn set_ny(&mut self, v: usize)        { self.inner.ny = v; }
    #[setter] fn set_nz(&mut self, v: usize)        { self.inner.nz = v; }

    // ── Derived quantities ─────────────────────────────────────────────────

    /// Defect length scale ld = sqrt(K_r / zeta_eff).
    fn defect_length(&self) -> f64       { self.inner.defect_length() }

    /// Dimensionless existence condition Pi = K_r / (Gamma_l * eta * K_l).
    fn pi_number(&self) -> f64           { self.inner.pi_number() }

    /// Effective Landau parameter a_eff = a_landau - zeta_eff / 2.
    fn a_eff(&self) -> f64               { self.inner.a_eff() }

    /// Cahn-Hilliard coherence length xi_CH = sqrt(kappa_ch / a_ch).
    fn ch_coherence_length(&self) -> f64 { self.inner.ch_coherence_length() }

    /// Equilibrium lipid fraction phi_eq = sqrt(a_ch / b_ch).
    fn phi_eq(&self) -> f64              { self.inner.phi_eq() }

    /// Validate that parameters are physically reasonable.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "MarsParams3D(nx={}, ny={}, nz={}, zeta_eff={:.4}, a_eff={:.4}, Pi={:.4})",
            self.inner.nx, self.inner.ny, self.inner.nz,
            self.inner.zeta_eff, self.inner.a_eff(), self.inner.pi_number(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyQField3D
// ─────────────────────────────────────────────────────────────────────────────

/// 3D Q-tensor field with numpy interop.
///
/// Internal layout: flat Vec<[f64;5]> with linear index k = (i*ny + j)*nz + l.
/// numpy conversions use a (nx*ny*nz, 5) f64 array; the `.q` property returns
/// a (nx, ny, nz, 5) array. Scalar observables have shape (nx*ny*nz,); reshape
/// in Python to (nx, ny, nz).
#[pyclass(name = "QField3D")]
#[derive(Clone)]
pub struct PyQField3D {
    pub(crate) inner: QField3D,
}

#[pymethods]
impl PyQField3D {
    /// All-zero 3D Q-tensor field.
    #[staticmethod]
    fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { inner: QField3D::zeros(nx, ny, nz, dx) }
    }

    /// Uniform 3D Q-tensor field. Provide the five independent components
    /// q11, q12, q13, q22, q23 explicitly; q33 = -(q11+q22) is recovered on demand.
    #[staticmethod]
    fn uniform(nx: usize, ny: usize, nz: usize, dx: f64,
               q11: f64, q12: f64, q13: f64, q22: f64, q23: f64) -> Self {
        Self { inner: QField3D::uniform(nx, ny, nz, dx, [q11, q12, q13, q22, q23]) }
    }

    /// Small-amplitude random perturbation around zero.
    ///
    /// Each component is drawn uniformly from [-amplitude, amplitude].
    #[staticmethod]
    fn random_perturbation(nx: usize, ny: usize, nz: usize, dx: f64,
                           amplitude: f64, seed: u64) -> Self {
        Self { inner: QField3D::random_perturbation(nx, ny, nz, dx, amplitude, seed) }
    }

    /// Import from a numpy array of shape (nx*ny*nz, 5).
    /// Data is copied into the Rust-owned Vec.
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray2<f64>, nx: usize, ny: usize, nz: usize,
                  dx: f64) -> PyResult<Self> {
        let view = arr.as_array();
        let arr_shape = view.shape();
        let expected_n = nx * ny * nz;
        if arr_shape[0] != expected_n || arr_shape[1] != 5 {
            return Err(PyValueError::new_err(format!(
                "expected shape ({}, 5), got ({}, {})",
                expected_n, arr_shape[0], arr_shape[1],
            )));
        }
        let q: Vec<[f64; 5]> = (0..expected_n)
            .map(|k| [view[[k, 0]], view[[k, 1]], view[[k, 2]], view[[k, 3]], view[[k, 4]]])
            .collect();
        Ok(Self { inner: QField3D { q, nx, ny, nz, dx } })
    }

    /// Q-tensor components as a numpy array of shape (nx, ny, nz, 5).
    ///
    /// Access individual components in Python: arr[..., 0] = q11, arr[..., 1] = q12, etc.
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        let n = self.inner.len();
        let mut arr = Array4::<f64>::zeros((self.inner.nx, self.inner.ny, self.inner.nz, 5));
        for k in 0..n {
            let (i, j, l) = self.inner.ijk(k);
            for c in 0..5 {
                arr[[i, j, l, c]] = self.inner.q[k][c];
            }
        }
        arr.into_pyarray(py)
    }

    /// Scalar order parameter S at each vertex, shape (nx*ny*nz,).
    /// Reshape in Python to (nx, ny, nz).
    fn scalar_order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let s = self.inner.scalar_order_s();
        Array1::from_vec(s).into_pyarray(py)
    }

    /// Biaxiality parameter P = lambda_mid - lambda_min at each vertex, shape (nx*ny*nz,).
    /// Reshape in Python to (nx, ny, nz).
    fn biaxiality<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = self.inner.biaxiality_p();
        Array1::from_vec(p).into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn nz(&self) -> usize { self.inner.nz }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    /// Mean scalar order parameter over the whole field.
    fn mean_s(&self) -> f64    { self.inner.mean_s() }

    /// Maximum Frobenius norm over all vertices.
    fn max_norm(&self) -> f64  { self.inner.max_norm() }

    fn __repr__(&self) -> String {
        format!(
            "QField3D(nx={}, ny={}, nz={}, dx={:.3}, <S>={:.4})",
            self.inner.nx, self.inner.ny, self.inner.nz,
            self.inner.dx, self.inner.mean_s(),
        )
    }

    fn __len__(&self) -> usize { self.inner.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyVelocityField3D
// ─────────────────────────────────────────────────────────────────────────────

/// 3D velocity field with numpy interop.
///
/// Internal layout: flat Vec<[f64;3]> with linear index k = (i*ny + j)*nz + l.
/// The `.u` property returns a (nx, ny, nz, 3) array.
#[pyclass(name = "VelocityField3D")]
#[derive(Clone)]
pub struct PyVelocityField3D {
    pub(crate) inner: VelocityField3D,
}

#[pymethods]
impl PyVelocityField3D {
    /// All-zero 3D velocity field.
    #[staticmethod]
    fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { inner: VelocityField3D::zeros(nx, ny, nz, dx) }
    }

    /// Uniform 3D velocity field. Provide the three velocity components [ux, uy, uz].
    #[staticmethod]
    fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, u: [f64; 3]) -> Self {
        Self { inner: VelocityField3D::uniform(nx, ny, nz, dx, u) }
    }

    /// Velocity components as a numpy array of shape (nx, ny, nz, 3).
    ///
    /// Access individual components in Python: arr[..., 0] = ux, arr[..., 1] = uy, arr[..., 2] = uz.
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        let nx = self.inner.nx;
        let ny = self.inner.ny;
        let nz = self.inner.nz;
        let mut arr = Array4::<f64>::zeros((nx, ny, nz, 3));
        for k in 0..self.inner.u.len() {
            let l = k % nz;
            let ij = k / nz;
            let j = ij % ny;
            let i = ij / ny;
            for c in 0..3 {
                arr[[i, j, l, c]] = self.inner.u[k][c];
            }
        }
        arr.into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn nz(&self) -> usize { self.inner.nz }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    fn __repr__(&self) -> String {
        format!(
            "VelocityField3D(nx={}, ny={}, nz={}, dx={:.3})",
            self.inner.nx, self.inner.ny, self.inner.nz, self.inner.dx,
        )
    }

    fn __len__(&self) -> usize { self.inner.u.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyScalarField3D
// ─────────────────────────────────────────────────────────────────────────────

/// 3D scalar field with numpy interop.
///
/// Internal layout: flat Vec<f64> with linear index k = (i*ny + j)*nz + l.
/// The `.phi` property returns a flat 1D array of length nx*ny*nz.
/// Reshape in Python to (nx, ny, nz) as needed.
#[pyclass(name = "ScalarField3D")]
#[derive(Clone)]
pub struct PyScalarField3D {
    pub(crate) inner: ScalarField3D,
}

#[pymethods]
impl PyScalarField3D {
    /// All-zero 3D scalar field.
    #[staticmethod]
    fn zeros(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { inner: ScalarField3D::zeros(nx, ny, nz, dx) }
    }

    /// Uniform 3D scalar field with every vertex set to `val`.
    #[staticmethod]
    fn uniform(nx: usize, ny: usize, nz: usize, dx: f64, val: f64) -> Self {
        Self { inner: ScalarField3D::uniform(nx, ny, nz, dx, val) }
    }

    /// Import from a flat 1D numpy array of length nx*ny*nz.
    /// Data is copied into the Rust-owned Vec.
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray1<f64>, nx: usize, ny: usize, nz: usize,
                  dx: f64) -> PyResult<Self> {
        let view = arr.as_array();
        let expected_n = nx * ny * nz;
        if view.len() != expected_n {
            return Err(PyValueError::new_err(format!(
                "expected array of length {}, got {}",
                expected_n, view.len(),
            )));
        }
        let phi: Vec<f64> = view.iter().copied().collect();
        Ok(Self { inner: ScalarField3D { phi, nx, ny, nz, dx } })
    }

    /// Scalar values as a flat 1D numpy array of length nx*ny*nz.
    /// Reshape in Python to (nx, ny, nz) as needed.
    #[getter]
    fn phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.phi.clone()).into_pyarray(py)
    }

    #[getter] fn nx(&self) -> usize { self.inner.nx }
    #[getter] fn ny(&self) -> usize { self.inner.ny }
    #[getter] fn nz(&self) -> usize { self.inner.nz }
    #[getter] fn dx(&self) -> f64   { self.inner.dx }

    /// Mean value over all vertices.
    fn mean(&self) -> f64 { self.inner.mean() }

    /// Maximum value over all vertices.
    fn max(&self) -> f64 { self.inner.max() }

    fn __repr__(&self) -> String {
        format!(
            "ScalarField3D(nx={}, ny={}, nz={}, dx={:.3}, mean={:.4})",
            self.inner.nx, self.inner.ny, self.inner.nz,
            self.inner.dx, self.inner.mean(),
        )
    }

    fn __len__(&self) -> usize { self.inner.phi.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// PySnapStats3D
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics for the dry active nematic run (run_mars_3d).
#[pyclass(name = "SnapStats3D")]
#[derive(Clone)]
pub struct PySnapStats3D {
    inner: SnapStats3D,
}

#[pymethods]
impl PySnapStats3D {
    /// Simulation time at this snapshot.
    #[getter] fn time(&self) -> f64 { self.inner.time }
    /// Spatial mean of the scalar order parameter S.
    #[getter] fn mean_s(&self) -> f64 { self.inner.mean_s }
    /// Spatial mean of the biaxiality parameter P.
    #[getter] fn biaxiality_p(&self) -> f64 { self.inner.biaxiality_p }
    /// Number of connected disclination lines detected.
    #[getter] fn n_disclination_lines(&self) -> usize { self.inner.n_disclination_lines }
    /// Total disclination line length in vertex units.
    #[getter] fn total_line_length(&self) -> f64 { self.inner.total_line_length }
    /// Mean Frenet curvature along all disclination lines.
    #[getter] fn mean_line_curvature(&self) -> f64 { self.inner.mean_line_curvature }
    /// Number of topological events since the previous snapshot.
    #[getter] fn n_events(&self) -> usize { self.inner.n_events }

    fn __repr__(&self) -> String {
        format!(
            "SnapStats3D(time={:.4}, mean_s={:.4}, n_disclination_lines={}, total_line_length={:.4})",
            self.inner.time, self.inner.mean_s,
            self.inner.n_disclination_lines, self.inner.total_line_length,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyBechStats3D
// ─────────────────────────────────────────────────────────────────────────────

/// Per-snapshot statistics for the full BECH run (run_mars_3d_full).
#[pyclass(name = "BechStats3D")]
#[derive(Clone)]
pub struct PyBechStats3D {
    inner: BechStats3D,
}

#[pymethods]
impl PyBechStats3D {
    /// Simulation time at this snapshot.
    #[getter] fn time(&self) -> f64 { self.inner.time }
    /// Spatial mean of the scalar order parameter S.
    #[getter] fn mean_s(&self) -> f64 { self.inner.mean_s }
    /// Spatial mean of the biaxiality parameter P.
    #[getter] fn biaxiality_p(&self) -> f64 { self.inner.biaxiality_p }
    /// Spatial mean of the lipid concentration phi.
    #[getter] fn mean_phi(&self) -> f64 { self.inner.mean_phi }
    /// Number of connected disclination lines detected.
    #[getter] fn n_disclination_lines(&self) -> usize { self.inner.n_disclination_lines }
    /// Total disclination line length in vertex units.
    #[getter] fn total_line_length(&self) -> f64 { self.inner.total_line_length }
    /// Mean Frenet curvature along all disclination lines.
    #[getter] fn mean_line_curvature(&self) -> f64 { self.inner.mean_line_curvature }
    /// Number of topological events since the previous snapshot.
    #[getter] fn n_events(&self) -> usize { self.inner.n_events }

    fn __repr__(&self) -> String {
        format!(
            "BechStats3D(time={:.4}, mean_s={:.4}, mean_phi={:.4}, n_disclination_lines={}, total_line_length={:.4})",
            self.inner.time, self.inner.mean_s, self.inner.mean_phi,
            self.inner.n_disclination_lines, self.inner.total_line_length,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

/// Register 3D binding classes into the volterra Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMarsParams3D>()?;
    m.add_class::<PyQField3D>()?;
    m.add_class::<PyVelocityField3D>()?;
    m.add_class::<PyScalarField3D>()?;
    m.add_class::<PySnapStats3D>()?;
    m.add_class::<PyBechStats3D>()?;
    Ok(())
}
