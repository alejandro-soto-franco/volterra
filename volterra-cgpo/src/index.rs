//! Flat-array indexing for the row-major `x * ly + y` CGPO grid layout.

/// Flat scalar index for grid cell `(x, y)` with `ly` columns.
#[inline(always)]
pub fn si(x: usize, y: usize, ly: usize) -> usize {
    x * ly + y
}

/// Flat index of component `c in {0,1}` of a 2-vector field at `(x, y)`.
#[inline(always)]
pub fn vi(x: usize, y: usize, ly: usize, c: usize) -> usize {
    (x * ly + y) * 2 + c
}
