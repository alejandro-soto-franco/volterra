//! Python bindings for `volterra-braid`: defect detection, braid-word
//! extraction, and topological entropy.
//!
//! The FFI surface is deliberately plain numeric lists (no numpy, no custom
//! classes) so the comparison against the reference Python implementation is
//! trivial and free of version coupling.
//!
//! Exposed to Python (import volterra):
//!   volterra.braid_detect_defects(qxx, qxy, nx, ny, threshold, mask) -> [(x, y, charge), ...]
//!   volterra.braid_word_from_frames(frames)  -> (n_strands, codes)
//!   volterra.braid_topological_entropy(n_strands, codes) -> float
//!
//! where `frames` is a list of frames, each a list of `(x, y, charge)` triples,
//! and `codes` is the list of signed 1-based generator codes (`+i` / `-i`).

use pyo3::prelude::*;

use volterra_braid::{
    detect_defects, extract_braidword, topological_entropy, track, BraidWord, Defect,
};

/// Detect defects in a row-major `nx * ny` Q-tensor grid.
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

/// Track a defect-position time series into worldlines and extract its braid word.
///
/// `frames` is a list of frames, each a list of `(x, y, charge)` triples.
/// Returns `(n_strands, codes)`.
#[pyfunction]
#[pyo3(signature = (frames))]
fn braid_word_from_frames(frames: Vec<Vec<(f64, f64, i64)>>) -> (usize, Vec<i32>) {
    let frames: Vec<Vec<Defect>> = frames
        .into_iter()
        .map(|f| {
            f.into_iter()
                .map(|(x, y, c)| Defect {
                    pos: [x, y],
                    charge: c as i8,
                })
                .collect()
        })
        .collect();
    let word = extract_braidword(&track(&frames));
    (word.n_strands, word.codes())
}

/// Topological entropy of the braid given by `(n_strands, codes)`.
#[pyfunction]
#[pyo3(signature = (n_strands, codes))]
fn braid_topological_entropy(n_strands: usize, codes: Vec<i32>) -> f64 {
    topological_entropy(&BraidWord::from_codes(n_strands, &codes))
}

/// Register the braid functions on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(braid_detect_defects, m)?)?;
    m.add_function(wrap_pyfunction!(braid_word_from_frames, m)?)?;
    m.add_function(wrap_pyfunction!(braid_topological_entropy, m)?)?;
    Ok(())
}
