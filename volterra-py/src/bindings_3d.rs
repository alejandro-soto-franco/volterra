// volterra-py/src/bindings_3d.rs
//
// PyO3 bindings for 3D simulation types.
//
// Exposed to Python (import volterra):
//   volterra.ActiveNematicParams3D -- physical / numerical parameters for the 3D simulation
//   volterra.QField3D       -- 3D Q-tensor field with numpy interop
//   volterra.VelocityField3D -- 3D velocity field with numpy interop
//   volterra.ScalarField3D  -- 3D scalar (concentration / pressure) field with numpy interop
//   volterra.SnapStats3D    -- per-snapshot statistics for the dry active nematic run
//   volterra.BechStats3D    -- per-snapshot statistics for the full BECH run

use numpy::ndarray::{Array1, Array2, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volterra_core::ActiveNematicParams3D;
use volterra_fields::{QField3D, ScalarField3D, VelocityField3D};
use volterra_solver::{BechStats3D, SnapStats3D};
use cartan_geo::{DisclinationLine, DisclinationEvent, EventKind, DisclinationCharge, Sign};

// ─────────────────────────────────────────────────────────────────────────────
// PyActiveNematicParams3D
// ─────────────────────────────────────────────────────────────────────────────

/// All physical and numerical parameters for the 3D active nematic simulation.
#[pyclass(name = "ActiveNematicParams3D")]
#[derive(Clone)]
pub struct PyActiveNematicParams3D {
    inner: ActiveNematicParams3D,
}

#[pymethods]
impl PyActiveNematicParams3D {
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
        let p = ActiveNematicParams3D {
            nx, ny, nz, dx, dt,
            k_r, gamma_r, zeta_eff, eta,
            a_landau, c_landau, b_landau,
            lambda: lambda_,
            noise_amp, chi_a, b0, omega_b,
            k_l, gamma_l, xi_l, chi_ms,
            kappa_ch, a_ch, b_ch, m_l,
            c0_sp: 0.0,
            kappa_w: 0.0,
            kappa_bar_g: 0.0,
            epsilon_ch: 1.0,
        };
        p.validate().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: p })
    }

    /// Construct the default test parameter set (16x16x16 grid, active turbulent phase).
    #[staticmethod]
    fn default_test() -> Self {
        Self { inner: ActiveNematicParams3D::default_test() }
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
            "ActiveNematicParams3D(nx={}, ny={}, nz={}, zeta_eff={:.4}, a_eff={:.4}, Pi={:.4})",
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
        // VelocityField3D layout: k = (i*ny + j)*nz + l, so inverse is:
        // l = k % nz, ij = k/nz, j = ij % ny, i = ij / ny.
        // (VelocityField3D has no ijk() helper; this matches VelocityField3D::idx.)
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

/// Per-snapshot statistics for the dry active nematic run (run_dry_active_nematic_3d).
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

/// Per-snapshot statistics for the full BECH run (run_bech_3d).
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
// PyDisclinationLine
// ─────────────────────────────────────────────────────────────────────────────

/// A reconstructed disclination line: ordered vertices with Frenet-Serret geometry.
///
/// Wraps `cartan_geo::DisclinationLine`. Instances are produced by the runner
/// (Task 17) and are read-only from Python.
#[pyclass(name = "DisclinationLine")]
#[derive(Clone)]
pub struct PyDisclinationLine {
    pub(crate) inner: DisclinationLine,
}

#[pymethods]
impl PyDisclinationLine {
    /// Ordered vertex positions, shape (n, 3).
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.vertices.len();
        let mut arr = Array2::<f64>::zeros((n, 3));
        for (i, v) in self.inner.vertices.iter().enumerate() {
            arr[[i, 0]] = v[0];
            arr[[i, 1]] = v[1];
            arr[[i, 2]] = v[2];
        }
        arr.into_pyarray(py)
    }

    /// Unit tangent vectors at each vertex, shape (n, 3).
    #[getter]
    fn tangents<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.tangents.len();
        let mut arr = Array2::<f64>::zeros((n, 3));
        for (i, t) in self.inner.tangents.iter().enumerate() {
            arr[[i, 0]] = t[0];
            arr[[i, 1]] = t[1];
            arr[[i, 2]] = t[2];
        }
        arr.into_pyarray(py)
    }

    /// Frenet curvature |dT/ds| at each vertex, shape (n,).
    #[getter]
    fn curvatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.curvatures.clone()).into_pyarray(py)
    }

    /// Frenet-Serret torsion at each vertex, shape (n,).
    #[getter]
    fn torsions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.torsions.clone()).into_pyarray(py)
    }

    /// Topological charge as a string: "half_plus", "half_minus", or "anti".
    #[getter]
    fn charge(&self) -> String {
        match &self.inner.charge {
            DisclinationCharge::Half(Sign::Positive) => "half_plus".to_string(),
            DisclinationCharge::Half(Sign::Negative) => "half_minus".to_string(),
            DisclinationCharge::Anti => "anti".to_string(),
        }
    }

    /// True if the line forms a closed loop.
    #[getter]
    fn is_loop(&self) -> bool {
        self.inner.is_loop
    }

    /// Total arc length: sum of Euclidean distances between consecutive vertices.
    fn length(&self) -> f64 {
        let verts = &self.inner.vertices;
        if verts.len() < 2 {
            return 0.0;
        }
        verts.windows(2).map(|w| {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            let dz = w[1][2] - w[0][2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        }).sum()
    }

    /// Mean Frenet curvature over all vertices.
    fn mean_curvature(&self) -> f64 {
        let c = &self.inner.curvatures;
        if c.is_empty() {
            return 0.0;
        }
        c.iter().sum::<f64>() / c.len() as f64
    }

    fn __len__(&self) -> usize {
        self.inner.vertices.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DisclinationLine(n_vertices={}, charge={}, is_loop={}, length={:.4})",
            self.inner.vertices.len(),
            self.charge(),
            self.inner.is_loop,
            self.length(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyDisclinationEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A detected topological event between two consecutive simulation frames.
///
/// Wraps `cartan_geo::DisclinationEvent`. Instances are produced by the runner
/// and are read-only from Python.
#[pyclass(name = "DisclinationEvent")]
#[derive(Clone)]
pub struct PyDisclinationEvent {
    inner: DisclinationEvent,
}

#[pymethods]
impl PyDisclinationEvent {
    /// Frame index at which the event is recorded.
    #[getter]
    fn frame(&self) -> usize {
        self.inner.frame
    }

    /// Event type as a string: "creation", "annihilation", or "reconnection".
    #[getter]
    fn kind(&self) -> String {
        match self.inner.kind {
            EventKind::Creation     => "creation".to_string(),
            EventKind::Annihilation => "annihilation".to_string(),
            EventKind::Reconnection => "reconnection".to_string(),
        }
    }

    /// Approximate position of the event, shape (3,).
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.inner.position.to_vec()).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "DisclinationEvent(frame={}, kind={}, position=[{:.3}, {:.3}, {:.3}])",
            self.inner.frame,
            self.kind(),
            self.inner.position[0],
            self.inner.position[1],
            self.inner.position[2],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runner bindings
// ─────────────────────────────────────────────────────────────────────────────

/// Run the dry 3D active nematic model for `n_steps` Euler steps.
///
/// Snapshots are written to `out_dir` every `snap_every` steps. Returns
/// `(q_final, stats)` where `stats` contains one `SnapStats3D` per snapshot.
#[pyfunction]
#[pyo3(name = "run_dry_active_nematic_3d")]
fn run_dry_active_nematic_3d_py(
    q_init: &PyQField3D,
    params: &PyActiveNematicParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &str,
    track_defects: bool,
) -> PyResult<(PyQField3D, Vec<PySnapStats3D>)> {
    let path = std::path::Path::new(out_dir);
    let (q_final, stats) = volterra_solver::run_dry_active_nematic_3d(
        &q_init.inner,
        &params.inner,
        n_steps,
        snap_every,
        path,
        track_defects,
    );
    let py_q = PyQField3D { inner: q_final };
    let py_stats: Vec<PySnapStats3D> = stats.into_iter().map(|s| PySnapStats3D { inner: s }).collect();
    Ok((py_q, py_stats))
}

/// Run the full BECH 3D model (Beris-Edwards + Stokes + Cahn-Hilliard) for `n_steps` steps.
///
/// Snapshots are written to `out_dir` every `snap_every` steps. Returns
/// `(q_final, phi_final, stats)` where `stats` contains one `BechStats3D` per snapshot.
#[pyfunction]
#[pyo3(name = "run_bech_3d")]
fn run_bech_3d_py(
    q_init: &PyQField3D,
    phi_init: &PyScalarField3D,
    params: &PyActiveNematicParams3D,
    n_steps: usize,
    snap_every: usize,
    out_dir: &str,
    track_defects: bool,
) -> PyResult<(PyQField3D, PyScalarField3D, Vec<PyBechStats3D>)> {
    let path = std::path::Path::new(out_dir);
    let (q_final, phi_final, stats) = volterra_solver::run_bech_3d(
        &q_init.inner,
        &phi_init.inner,
        &params.inner,
        n_steps,
        snap_every,
        path,
        track_defects,
    );
    let py_q   = PyQField3D     { inner: q_final   };
    let py_phi = PyScalarField3D { inner: phi_final };
    let py_stats: Vec<PyBechStats3D> = stats.into_iter().map(|s| PyBechStats3D { inner: s }).collect();
    Ok((py_q, py_phi, py_stats))
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

/// Register 3D binding classes and runner functions into the volterra Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyActiveNematicParams3D>()?;
    m.add_class::<PyQField3D>()?;
    m.add_class::<PyVelocityField3D>()?;
    m.add_class::<PyScalarField3D>()?;
    m.add_class::<PySnapStats3D>()?;
    m.add_class::<PyBechStats3D>()?;
    m.add_class::<PyDisclinationLine>()?;
    m.add_class::<PyDisclinationEvent>()?;
    m.add_function(wrap_pyfunction!(run_dry_active_nematic_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_bech_3d_py, m)?)?;
    Ok(())
}
