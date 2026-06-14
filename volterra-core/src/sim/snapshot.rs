//! Canonical `.npy` writer and snapshot sinks.

use std::io::Write;
use std::path::Path;
use super::stats::StepStats;

/// Write a flat C-contiguous `f64` buffer as NumPy `.npy` v1.0 with on-disk
/// shape `(nx, ny, nz, n_comp)`. `data.len()` must equal `nx*ny*nz*n_comp`.
///
/// The header is padded to a multiple of 64 bytes (magic 8 + len-field 2 +
/// header body), matching the byte-exact rule from the legacy `runner_3d`
/// writer: `(8 + 2 + header_len) % 64 == 0`, i.e. `header_data_len % 64 == 54`.
pub fn write_npy(
    path: &Path,
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    n_comp: usize,
) -> std::io::Result<()> {
    let header_dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({nx}, {ny}, {nz}, {n_comp}), }}"
    );
    let magic: &[u8] = b"\x93NUMPY\x01\x00";
    let dict_plus_newline = header_dict.len() + 1;
    let header_len = {
        let needed = dict_plus_newline;
        let rem = (needed as isize).rem_euclid(64) as usize;
        let pad_needed = if rem == 54 { 0 } else { (54 + 64 - rem) % 64 };
        needed + pad_needed
    };
    let padding = header_len - dict_plus_newline;
    let mut padded_header = header_dict;
    for _ in 0..padding { padded_header.push(' '); }
    padded_header.push('\n');
    debug_assert_eq!((8 + 2 + header_len) % 64, 0);

    let mut f = std::fs::File::create(path)?;
    f.write_all(magic)?;
    f.write_all(&(header_len as u16).to_le_bytes())?;
    f.write_all(padded_header.as_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Observer sink that accumulates per-snapshot `StepStats`.
#[derive(Default)]
pub struct StatsSink {
    /// All snapshots collected during the run, in order.
    pub snapshots: Vec<StepStats>,
}

use super::runner::Observer;

impl<F> Observer<F> for StatsSink {
    fn observe(&mut self, _step: usize, t: f64, _field: &F, stats: &StepStats) {
        // Carry the snapshot time even when the step's stats predate it (step 0).
        let mut s = stats.clone();
        if s.time.is_none() {
            s.time = Some(t);
        }
        self.snapshots.push(s);
    }
}

#[cfg(test)]
mod tests {
    use super::write_npy;

    #[test]
    fn npy_header_is_64_byte_aligned_and_roundtrips_with_numpy_layout() {
        let tmp = std::env::temp_dir().join("vsim_npy_test.npy");
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]; // (2,1,1,2)
        write_npy(&tmp, &data, 2, 1, 1, 2).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(&bytes[0..6], b"\x93NUMPY");
        // magic(8) + len(2) + header must be a multiple of 64.
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        // 4 f64 payload at the end.
        assert_eq!(bytes.len(), 10 + header_len + 4 * 8);
    }
}
