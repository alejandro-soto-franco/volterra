//! Snapshot I/O for DEC Q-tensor fields.
//!
//! Writes Q-tensor data in NumPy `.npy` format (v1.0) for consumption
//! by the Python visualisation pipeline.

use std::io::Write;
use std::path::Path;

use crate::QFieldDec;

/// Write a QFieldDec snapshot as a `.npy` file.
///
/// The output array has shape (n_vertices, 2) with columns [q1, q2],
/// stored as float64 in C-contiguous (row-major) order.
pub fn write_snapshot(q: &QFieldDec, path: &Path) -> std::io::Result<()> {
    let nv = q.n_vertices;
    let shape_str = format!("({}, 2)", nv);

    // NumPy v1.0 header: magic + version + HEADER_LEN + dict + padding + newline.
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    // Header block must be aligned to 64 bytes (magic=6 + version=2 + header_len=2 = 10).
    let header_content_len = dict.len() + 1; // +1 for trailing newline
    let total_prefix = 10 + header_content_len;
    let padding = (64 - (total_prefix % 64)) % 64;
    let header_len = (header_content_len + padding) as u16;

    let mut file = std::fs::File::create(path)?;

    // Magic: \x93NUMPY
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    // Version 1.0
    file.write_all(&[1, 0])?;
    // HEADER_LEN (little-endian u16)
    file.write_all(&header_len.to_le_bytes())?;
    // Dict string
    file.write_all(dict.as_bytes())?;
    // Padding spaces
    for _ in 0..padding {
        file.write_all(b" ")?;
    }
    // Trailing newline
    file.write_all(b"\n")?;

    // Data: interleaved [q1_0, q2_0, q1_1, q2_1, ...]
    for i in 0..nv {
        file.write_all(&q.q1[i].to_le_bytes())?;
        file.write_all(&q.q2[i].to_le_bytes())?;
    }

    Ok(())
}

/// Write a velocity field snapshot as a `.npy` file.
///
/// The output array has shape (n_vertices, 3) with columns [vx, vy, vz],
/// stored as float64 in C-contiguous (row-major) order.
pub fn write_velocity_snapshot(vel: &[[f64; 3]], path: &Path) -> std::io::Result<()> {
    let nv = vel.len();
    let shape_str = format!("({}, 3)", nv);

    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    let header_content_len = dict.len() + 1;
    let total_prefix = 10 + header_content_len;
    let padding = (64 - (total_prefix % 64)) % 64;
    let header_len = (header_content_len + padding) as u16;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    file.write_all(&[1, 0])?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(dict.as_bytes())?;
    for _ in 0..padding {
        file.write_all(b" ")?;
    }
    file.write_all(b"\n")?;

    for v in vel {
        file.write_all(&v[0].to_le_bytes())?;
        file.write_all(&v[1].to_le_bytes())?;
        file.write_all(&v[2].to_le_bytes())?;
    }

    Ok(())
}

/// Write simulation metadata as a JSON file.
pub fn write_meta(path: &Path, meta: &serde_json::Value) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(meta)
        .map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn snapshot_roundtrip() {
        let q = QFieldDec::uniform(10, 0.3, 0.4);
        let dir = std::env::temp_dir().join("volterra_snap_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_q.npy");

        write_snapshot(&q, &path).unwrap();

        // Verify the file starts with the numpy magic
        let mut file = std::fs::File::open(&path).unwrap();
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"\x93NUMPY");

        // Verify file size: header (aligned to 64) + 10 * 2 * 8 = 160 bytes data
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(file_size >= 64 + 160, "file too small: {file_size}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
