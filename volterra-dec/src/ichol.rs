//! Incomplete Cholesky preconditioner for the DEC Poisson stiffness.
//!
//! The confined Stokes solve is a biharmonic, so two Poisson solves per step, and
//! on the lattice-matched nephroid those two accounted for 415 seconds of a 416
//! second run. Jacobi is the weakest preconditioner there is, and on a cotan
//! stiffness it leaves the iteration count scaling with the mesh diameter.
//!
//! `IChol` is the zero-fill factorisation `A ~ L L^T` restricted to the sparsity
//! of `tril(A)`. It costs one factorisation for the whole run, since the mesh
//! never moves and the Dirichlet set never changes, and one pair of triangular
//! solves per iteration thereafter.
//!
//! Zero fill-in is not unconditionally stable: a cotan stiffness with obtuse
//! triangles carries positive off-diagonal entries, and the factorisation can
//! reach a non-positive pivot even though `A` is positive definite. The remedy
//! is the standard one, factor `A + alpha diag(A)` and raise `alpha` until it
//! completes. A shifted factorisation is still a symmetric positive definite
//! preconditioner, so the conjugate gradient it drives is still the conjugate
//! gradient; only the clustering of the spectrum is weaker.

/// Zero-fill incomplete Cholesky factor, lower triangle in compressed rows.
///
/// Each row holds its columns in ascending order with the diagonal last, which
/// is what lets both triangular solves run off row access alone.
pub struct IChol {
    n: usize,
    /// Column index of each stored entry, row by row.
    cols: Vec<usize>,
    /// Value of each stored entry, in the same order.
    vals: Vec<f64>,
    /// Start of each row in `cols` and `vals`, length `n + 1`.
    row_start: Vec<usize>,
    /// The shift that was needed, as a multiple of the diagonal. Zero when the
    /// unshifted factorisation succeeded.
    pub shift: f64,
}

impl IChol {
    /// Factor the lower triangle of a symmetric matrix given as
    /// `(row, col, value)` triples, raising the diagonal shift until the
    /// factorisation completes.
    ///
    /// Entries with `col > row` are ignored, so either triangle or the full
    /// matrix may be passed. Returns `None` if no shift up to `2^20` succeeds,
    /// which would mean the matrix is not positive definite at all.
    pub fn factor(n: usize, triples: impl Fn(&mut dyn FnMut(usize, usize, f64))) -> Option<Self> {
        // Gather the lower triangle once, summing duplicates.
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        triples(&mut |r, c, v| {
            if c <= r {
                rows[r].push((c, v));
            }
        });
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut row_start = Vec::with_capacity(n + 1);
        let mut diag = vec![0.0f64; n];
        row_start.push(0);
        for (i, row) in rows.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(c, _)| c);
            let mut j = 0usize;
            while j < row.len() {
                let c = row[j].0;
                let mut v = 0.0;
                while j < row.len() && row[j].0 == c {
                    v += row[j].1;
                    j += 1;
                }
                cols.push(c);
                vals.push(v);
                if c == i {
                    diag[i] = v;
                }
            }
            // A row with no diagonal entry cannot be factored; supply a zero one
            // so the shift has something to act on and the failure is reported
            // through the pivot test rather than through a missing index.
            if cols[*row_start.last().unwrap()..].last() != Some(&i) {
                cols.push(i);
                vals.push(0.0);
            }
            row_start.push(cols.len());
        }

        let mut shift = 0.0f64;
        loop {
            if let Some(l) = Self::try_factor(n, &cols, &vals, &row_start, &diag, shift) {
                return Some(Self { n, cols: l.0, vals: l.1, row_start, shift });
            }
            shift = if shift == 0.0 { 1e-3 } else { shift * 2.0 };
            if shift > (1 << 20) as f64 {
                return None;
            }
        }
    }

    /// One attempt at the zero-fill factorisation with the diagonal scaled by
    /// `1 + shift`. `None` on a non-positive pivot.
    fn try_factor(
        n: usize,
        cols: &[usize],
        vals: &[f64],
        row_start: &[usize],
        diag: &[f64],
        shift: f64,
    ) -> Option<(Vec<usize>, Vec<f64>)> {
        let mut l: Vec<f64> = vals.to_vec();
        for i in 0..n {
            let (bi, ei) = (row_start[i], row_start[i + 1]);
            if shift != 0.0 {
                // The diagonal is the last entry of the row.
                l[ei - 1] = diag[i] * (1.0 + shift);
            }
            for pj in bi..ei {
                let j = cols[pj];
                // Sparse dot of row i and row j over their shared columns below j.
                let (bj, ej) = (row_start[j], row_start[j + 1]);
                let mut s = l[pj];
                let (mut a, mut b) = (bi, bj);
                while a < pj && b < ej && cols[b] < j {
                    match cols[a].cmp(&cols[b]) {
                        std::cmp::Ordering::Less => a += 1,
                        std::cmp::Ordering::Greater => b += 1,
                        std::cmp::Ordering::Equal => {
                            s -= l[a] * l[b];
                            a += 1;
                            b += 1;
                        }
                    }
                }
                if j == i {
                    // NaN must take this branch, so the negated comparison
                    // stays; `partial_cmp` would need the same special case.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(s > 0.0) || !s.is_finite() {
                        return None;
                    }
                    l[pj] = s.sqrt();
                } else {
                    let d = l[row_start[j + 1] - 1];
                    if d == 0.0 || !d.is_finite() {
                        return None;
                    }
                    l[pj] = s / d;
                }
            }
        }
        Some((cols.to_vec(), l))
    }

    /// Apply `M^{-1} = (L L^T)^{-1}` to `r`.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let mut z = r.to_vec();
        // Forward: L y = r.
        for i in 0..self.n {
            let (b, e) = (self.row_start[i], self.row_start[i + 1]);
            let mut acc = z[i];
            for p in b..e - 1 {
                acc -= self.vals[p] * z[self.cols[p]];
            }
            z[i] = acc / self.vals[e - 1];
        }
        // Backward: L^T z = y, scattered so row access suffices.
        for i in (0..self.n).rev() {
            let (b, e) = (self.row_start[i], self.row_start[i + 1]);
            z[i] /= self.vals[e - 1];
            let zi = z[i];
            for p in b..e - 1 {
                z[self.cols[p]] -= self.vals[p] * zi;
            }
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D Laplacian with Dirichlet ends: tridiagonal, SPD, and its zero-fill
    /// factorisation is exact, so `apply` has to invert it to round-off.
    #[test]
    fn a_tridiagonal_factorisation_is_exact() {
        let n = 64;
        let a = |f: &mut dyn FnMut(usize, usize, f64)| {
            for i in 0..n {
                f(i, i, 2.0);
                if i > 0 {
                    f(i, i - 1, -1.0);
                    f(i - 1, i, -1.0);
                }
            }
        };
        let ic = IChol::factor(n, a).expect("tridiagonal is positive definite");
        assert_eq!(ic.shift, 0.0, "no shift should be needed");
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut b = vec![0.0; n];
        a(&mut |r, c, v| b[r] += v * x[c]);
        let got = ic.apply(&b);
        for (g, e) in got.iter().zip(&x) {
            assert!((g - e).abs() < 1e-10, "got {g}, expected {e}");
        }
    }

    /// The preconditioner has to be symmetric positive definite, or the
    /// conjugate gradient it drives is not a conjugate gradient. Test it on a
    /// matrix with positive off-diagonals, which is what an obtuse triangle puts
    /// into a cotan stiffness and what makes zero fill-in break down.
    #[test]
    fn the_preconditioner_stays_symmetric_and_positive() {
        let n = 40;
        let a = |f: &mut dyn FnMut(usize, usize, f64)| {
            for i in 0..n {
                f(i, i, 4.0);
                if i > 0 {
                    let w = if i % 7 == 0 { 0.9 } else { -1.0 };
                    f(i, i - 1, w);
                    f(i - 1, i, w);
                }
                if i > 5 {
                    f(i, i - 6, -0.4);
                    f(i - 6, i, -0.4);
                }
            }
        };
        let ic = IChol::factor(n, a).expect("factorisation");
        // Symmetry: e_i^T M^{-1} e_j must equal e_j^T M^{-1} e_i.
        let mut m = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let c = ic.apply(&e);
            for i in 0..n {
                m[i][j] = c[i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (m[i][j] - m[j][i]).abs() < 1e-9 * (1.0 + m[i][j].abs()),
                    "M^-1 not symmetric at ({i},{j}): {} vs {}",
                    m[i][j],
                    m[j][i]
                );
            }
        }
        // Positive definite, tested on the diagonal and on random directions.
        for i in 0..n {
            assert!(m[i][i] > 0.0, "M^-1 diagonal {i} is {}", m[i][i]);
        }
        for s in 0..8 {
            let v: Vec<f64> = (0..n).map(|i| ((i * 31 + s * 17) as f64 * 0.61).sin()).collect();
            let mv = ic.apply(&v);
            let q: f64 = v.iter().zip(&mv).map(|(a, b)| a * b).sum();
            assert!(q > 0.0, "quadratic form {q} not positive on direction {s}");
        }
    }

    /// An indefinite matrix has no Cholesky factor at any shift of the sign the
    /// shift can reach, so the escalation has to give up rather than return a
    /// factor that is not one.
    #[test]
    fn a_negative_diagonal_is_refused() {
        let n = 4;
        let a = |f: &mut dyn FnMut(usize, usize, f64)| {
            for i in 0..n {
                f(i, i, if i == 2 { -1.0 } else { 1.0 });
            }
        };
        assert!(IChol::factor(n, a).is_none());
    }
}
