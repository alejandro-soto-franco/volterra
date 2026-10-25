//! Flat-index maps of the row-major periodic grids.
//!
//! Every field type indexes through these functions. `verify/` extracts this
//! file to Lean with Charon and Aeneas and proves the maps in range, inverse to
//! each other and periodic, for every grid size.

/// Least non-negative residue of `i` modulo `n`, for `1 <= n <= i64::MAX`.
///
/// Equal to `i.rem_euclid(n as i64) as usize`, written with `%` so that Aeneas
/// can translate it.
#[inline]
pub fn wrap_signed(i: i64, n: usize) -> usize {
    let n = n as i64;
    let r = i % n;
    if r < 0 { (r + n) as usize } else { r as usize }
}

/// Flat index of `(i, j)` on an `nx` by `ny` grid, wrapping each coordinate.
#[inline]
pub fn periodic_idx2(i: usize, j: usize, nx: usize, ny: usize) -> usize {
    (i % nx) * ny + (j % ny)
}

/// Flat index of a signed `(i, j)` on an `nx` by `ny` grid, wrapping each
/// coordinate.
#[inline]
pub fn periodic_idx2_signed(i: i64, j: i64, nx: usize, ny: usize) -> usize {
    wrap_signed(i, nx) * ny + wrap_signed(j, ny)
}

/// Flat index of `(i, j, l)` on an `nx` by `ny` by `nz` grid, wrapping each
/// coordinate.
#[inline]
pub fn periodic_idx3(i: usize, j: usize, l: usize, nx: usize, ny: usize, nz: usize) -> usize {
    ((i % nx) * ny + (j % ny)) * nz + (l % nz)
}

/// Grid coordinates `(i, j, l)` of the flat index `k` on a grid with `ny` by
/// `nz` cross-sections. Inverse of [`periodic_idx3`] on `k < nx * ny * nz`.
#[inline]
pub fn coords3(k: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let l = k % nz;
    let ij = k / nz;
    (ij / ny, ij % ny, l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_signed_matches_rem_euclid() {
        let values = [
            i64::MIN,
            i64::MIN + 1,
            -1000,
            -7,
            -1,
            0,
            1,
            6,
            7,
            1000,
            i64::MAX,
        ];
        for &i in &values {
            for n in [1usize, 2, 3, 7, 64, 1 << 20, i64::MAX as usize] {
                assert_eq!(
                    wrap_signed(i, n),
                    i.rem_euclid(n as i64) as usize,
                    "{i} mod {n}"
                );
            }
        }
    }

    #[test]
    fn coords3_inverts_periodic_idx3() {
        let (nx, ny, nz) = (3, 4, 5);
        for k in 0..nx * ny * nz {
            let (i, j, l) = coords3(k, ny, nz);
            assert_eq!(periodic_idx3(i, j, l, nx, ny, nz), k);
        }
    }
}
