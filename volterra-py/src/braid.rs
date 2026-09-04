//! Python bindings for `volterra-braid`: defect detection, braid-word
//! extraction, and topological entropy.
//!
//! Two layers are exposed:
//!
//! - A high-level [`PyBraidWord`] class (`volterra.BraidWord`) with `entropy()`,
//!   `permutation()`, `from_frames(...)`, equality, and `repr`.
//! - Plain-list free functions for differential testing against the reference
//!   Python implementation (no numpy, no classes, so comparison is trivial):
//!   `braid_detect_defects`, `braid_word_from_frames`, `braid_topological_entropy`.
//!
//! `frames` is a list of frames, each a list of `(x, y, charge)` triples; `codes`
//! is the list of signed 1-based generator codes (`+i` is `sigma_i`, `-i` is
//! `sigma_i^-1`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use volterra_braid::{
    BraidWord, Defect, detect_defects, detect_defects_winding, topological_entropy,
};

fn frames_to_defects(frames: Vec<Vec<(f64, f64, i64)>>) -> Vec<Vec<Defect>> {
    frames
        .into_iter()
        .map(|f| {
            f.into_iter()
                .map(|(x, y, c)| Defect {
                    pos: [x, y],
                    charge: c as i8,
                })
                .collect()
        })
        .collect()
}

fn make_braidword(n_strands: usize, codes: &[i32]) -> PyResult<BraidWord> {
    if n_strands < 1 {
        return Err(PyValueError::new_err("n_strands must be >= 1"));
    }
    for &c in codes {
        let i = c.unsigned_abs() as usize;
        if c == 0 || i >= n_strands {
            return Err(PyValueError::new_err(format!(
                "generator code {c} out of range for {n_strands} strands (need 1 <= |code| < n_strands)"
            )));
        }
    }
    Ok(BraidWord::from_codes(n_strands, codes))
}

/// A braid word in the Artin generators (`volterra.BraidWord`).
#[pyclass(name = "BraidWord", from_py_object)]
#[derive(Clone)]
pub struct PyBraidWord {
    inner: BraidWord,
}

#[pymethods]
impl PyBraidWord {
    /// `BraidWord(n_strands, codes)` from signed 1-based generator codes.
    #[new]
    fn new(n_strands: usize, codes: Vec<i32>) -> PyResult<Self> {
        Ok(PyBraidWord {
            inner: make_braidword(n_strands, &codes)?,
        })
    }

    /// Track a defect-position time series into worldlines and extract the braid.
    ///
    /// `frames` is a list of frames, each a list of `(x, y, charge)` triples.
    #[staticmethod]
    fn from_frames(frames: Vec<Vec<(f64, f64, i64)>>) -> Self {
        PyBraidWord {
            inner: BraidWord::from_frames(&frames_to_defects(frames)),
        }
    }

    /// Number of strands `n` (the braid lives in `B_n`).
    #[getter]
    fn n_strands(&self) -> usize {
        self.inner.n_strands
    }

    /// The signed 1-based generator codes.
    #[getter]
    fn codes(&self) -> Vec<i32> {
        self.inner.codes()
    }

    /// Topological entropy: `log` of the dilatation (Burau at `t = -1`).
    fn entropy(&self) -> f64 {
        self.inner.topological_entropy()
    }

    /// The permutation induced on the strands: `perm[i]` is the final position of
    /// the strand that started at position `i`.
    fn permutation(&self) -> Vec<usize> {
        self.inner.permutation()
    }

    /// Exponent sum (abelianisation): `+1` per `sigma_i`, `-1` per `sigma_i^-1`.
    fn exponent_sum(&self) -> i32 {
        self.inner.exponent_sum()
    }

    /// The shortest generating period if the word is an exact repetition, else
    /// the whole word (as signed codes).
    fn fundamental_period(&self) -> Vec<i32> {
        self.inner
            .fundamental_period()
            .iter()
            .map(|g| g.code())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.gens.len()
    }

    fn __eq__(&self, other: &PyBraidWord) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "BraidWord(n_strands={}, {})",
            self.inner.n_strands, self.inner
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Detect defects by the saddle-splay density, thresholded.
///
/// `threshold` bounds the SADDLE-SPLAY quantity
/// `d_x Qxy d_y Qxx - d_x Qxx d_y Qxy`, taken by central differences, which is
/// what the reference implementation marks its defects with. It is not an
/// angle, and its scale follows the field's own gradients: the reference draws
/// at `0.05 * S0`, about 0.07 for `S0 = sqrt 2`. Passing an angle such as
/// `pi / 2` is 22 times too large and returns nothing on a smooth field while
/// still firing on a noisy initial condition, which makes the mistake look like
/// a physical result.
///
/// Prefer [`braid_detect_defects_winding`], which reads the director's holonomy
/// and needs no threshold.
///
/// Returns one `(x, y, charge)` triple per detected defect.
#[pyfunction]
#[pyo3(signature = (qxx, qxy, nx, ny, threshold, mask))]
fn braid_detect_defects(
    qxx: Vec<f64>,
    qxy: Vec<f64>,
    nx: usize,
    ny: usize,
    threshold: f64,
    mask: Vec<bool>,
) -> Vec<(f64, f64, i64)> {
    detect_defects(&qxx, &qxy, nx, ny, threshold, &mask)
        .into_iter()
        .map(|d| (d.pos[0], d.pos[1], d.charge as i64))
        .collect()
}

/// Detect defects by the director's holonomy on a row-major `nx * ny` grid.
///
/// Sums the wrapped director increments round each plaquette, so a `+1/2` core
/// returns a half turn and a `-1/2` core minus that. There is no threshold to
/// choose: the sum is a topological quantity and takes one of a few values.
///
/// `mask` marks the cells inside the domain; a plaquette is read only where all
/// four of its corners are in.
///
/// Returns one `(x, y, charge)` triple per detected defect, charge in half
/// units.
#[pyfunction]
#[pyo3(signature = (qxx, qxy, nx, ny, mask))]
fn braid_detect_defects_winding(
    qxx: Vec<f64>,
    qxy: Vec<f64>,
    nx: usize,
    ny: usize,
    mask: Vec<bool>,
) -> Vec<(f64, f64, i64)> {
    detect_defects_winding(&qxx, &qxy, nx, ny, &mask)
        .into_iter()
        .map(|d| (d.pos[0], d.pos[1], d.charge as i64))
        .collect()
}

/// Track a defect-position time series and extract its braid word.
///
/// `frames` is a list of frames, each a list of `(x, y, charge)` triples.
/// Returns `(n_strands, codes)`.
#[pyfunction]
#[pyo3(signature = (frames))]
fn braid_word_from_frames(frames: Vec<Vec<(f64, f64, i64)>>) -> (usize, Vec<i32>) {
    let word = BraidWord::from_frames(&frames_to_defects(frames));
    (word.n_strands, word.codes())
}

/// Topological entropy of the braid given by `(n_strands, codes)`.
#[pyfunction]
#[pyo3(signature = (n_strands, codes))]
fn braid_topological_entropy(n_strands: usize, codes: Vec<i32>) -> PyResult<f64> {
    Ok(topological_entropy(&make_braidword(n_strands, &codes)?))
}

/// Register the braid class and functions on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBraidWord>()?;
    m.add_function(wrap_pyfunction!(braid_detect_defects, m)?)?;
    m.add_function(wrap_pyfunction!(braid_detect_defects_winding, m)?)?;
    m.add_function(wrap_pyfunction!(braid_word_from_frames, m)?)?;
    m.add_function(wrap_pyfunction!(braid_topological_entropy, m)?)?;
    Ok(())
}
