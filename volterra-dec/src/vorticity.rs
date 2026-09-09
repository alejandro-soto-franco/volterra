//! Vorticity direction, regularised so a null evaluates to zero.
//!
//! In two dimensions the vorticity read off a stream function is a scalar and
//! its direction is a sign; in three it is a vector and its direction is a unit
//! vector. Both are undefined where the vorticity vanishes, and a vorticity null
//! is a curve, so a field of any size contains
//! them and a visualisation walks straight into `0/0`.
//!
//! # The regularisation
//!
//! ```text
//! d_eps(w) = w / sqrt(|w|^2 + eps^2)
//! ```
//!
//! It is defined everywhere, needs no branch, and returns exactly zero at a
//! null. Away from one it is the direction to relative accuracy `eps^2 / 2|w|^2`,
//! so at `|w| = 10 eps` it is already within half a per cent of the unit vector.
//!
//! Its magnitude is the null indicator, which is what serves a picture:
//! `|d|` runs from 0 at a null to 1 in a strong
//! region, so one field colours by direction and fades by confidence, and no
//! separate mask is needed.
//!
//! # Why not `w / (|w| + eps)`
//!
//! That form also avoids the division, and it is what most codes reach for. It
//! is continuous and NOT differentiable at the null, because `|w|` has a corner
//! there, so its gradient jumps across every null curve and a streamline or an
//! isosurface drawn through one shows a crease that is an artefact of the
//! regulariser. The square-root form is smooth to every order. `direction_2d`
//! and `direction_3d` use the smooth one, and
//! `the_square_root_form_is_smooth_where_the_absolute_form_kinks` states the
//! difference as a test rather than as a claim in a comment.
//!
//! # Choosing `eps`
//!
//! Vorticity has units of inverse time, so a bare constant is wrong the moment
//! the problem is rescaled. [`Epsilon::relative`] takes a fraction of the
//! field's own root mean square, which is scale free and is the sensible
//! default; [`Epsilon::Absolute`] is there for a caller comparing two fields on
//! one colour scale, where a shared `eps` is what keeps them comparable.

/// How the regularisation length is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Epsilon {
    /// A fraction of the field's own root mean square. Scale free.
    Relative(f64),
    /// A fixed value, in the field's own units.
    Absolute(f64),
}

impl Epsilon {
    /// A fraction of the root mean square, which is the default choice.
    pub fn relative(fraction: f64) -> Self {
        Self::Relative(fraction)
    }

    /// Resolve against a field's root mean square.
    ///
    /// A field that is identically zero has no scale of its own, so a relative
    /// epsilon resolves to the smallest positive normal number rather than to
    /// zero: the direction is then zero everywhere, which is the truthful answer
    /// for a field with no vorticity anywhere.
    pub fn resolve(&self, rms: f64) -> f64 {
        match *self {
            Self::Absolute(e) => e.abs().max(f64::MIN_POSITIVE),
            Self::Relative(f) => (f.abs() * rms).max(f64::MIN_POSITIVE),
        }
    }
}

/// Root mean square of a scalar vorticity field.
pub fn rms_2d(omega: &[f64]) -> f64 {
    if omega.is_empty() {
        return 0.0;
    }
    (omega.iter().map(|w| w * w).sum::<f64>() / omega.len() as f64).sqrt()
}

/// Root mean square magnitude of a vector vorticity field.
pub fn rms_3d(omega: &[[f64; 3]]) -> f64 {
    if omega.is_empty() {
        return 0.0;
    }
    let acc: f64 = omega.iter().map(|w| w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sum();
    (acc / omega.len() as f64).sqrt()
}

/// Regularised direction of a scalar vorticity field, in `(-1, 1)`.
///
/// The sign is the sense of rotation and the magnitude is how far the point sits
/// from a null. Exactly `0.0` where the vorticity is exactly zero, and finite
/// for every input including infinities.
pub fn direction_2d(omega: &[f64], eps: Epsilon) -> Vec<f64> {
    let e = eps.resolve(rms_2d(omega));
    omega
        .iter()
        .map(|&w| {
            // An exact zero is a null and answers zero. Falling through would
            // answer `0.0_f64.signum()`, which is 1.0 in Rust, and the
            // regularisation underflows to zero for a field with no vorticity anywhere, so
            // this branch is reached rather than hypothetical.
            if w == 0.0 {
                return 0.0;
            }
            if !w.is_finite() {
                return w.signum();
            }
            // Scale before squaring. `w * w` overflows to infinity above about
            // 1.34e154, which would give `w / inf = 0.0`; that is finite, so no
            // fallback fires and a huge vorticity would read as a null, the one
            // value reserved for its opposite. Dividing through by the larger of
            // the two magnitudes is exact and cannot overflow.
            let m = w.abs().max(e);
            let (wn, en) = (w / m, e / m);
            wn / (wn * wn + en * en).sqrt()
        })
        .collect()
}

/// Regularised direction of a vector vorticity field, of magnitude below one.
///
/// The zero vector where the vorticity is exactly zero, and finite for every
/// finite input and for an infinity, whose sign is kept componentwise. A NaN
/// component propagates, for the reason [`direction_2d`] records.
pub fn direction_3d(omega: &[[f64; 3]], eps: Epsilon) -> Vec<[f64; 3]> {
    let e = eps.resolve(rms_3d(omega));
    omega
        .iter()
        .map(|w| {
            let mag = (w[0].abs()).max(w[1].abs()).max(w[2].abs());
            if mag == 0.0 {
                return [0.0; 3];
            }
            if !mag.is_finite() {
                // An infinite component keeps its sign, matching `direction_2d`.
                // Answering the zero vector would report the strongest possible
                // vorticity as a null.
                let mut out = [0.0; 3];
                for k in 0..3 {
                    out[k] = if w[k].is_infinite() { w[k].signum() } else { 0.0 };
                }
                return out;
            }
            // Scale by the largest component before squaring, for the overflow
            // reason `direction_2d` records.
            let m = mag.max(e);
            let (a, b, c) = (w[0] / m, w[1] / m, w[2] / m);
            let en = e / m;
            let den = (a * a + b * b + c * c + en * en).sqrt();
            [a / den, b / den, c / den]
        })
        .collect()
}

/// Indices where the regularised direction has magnitude below `threshold`.
///
/// These are the points a picture should fade out or leave unmarked, since the
/// direction there is set by the regularisation rather than by the flow. A
/// threshold of `0.1` marks everything within about a tenth of `eps` of a null.
pub fn nulls_2d(direction: &[f64], threshold: f64) -> Vec<usize> {
    direction
        .iter()
        .enumerate()
        .filter(|(_, d)| d.abs() < threshold)
        .map(|(i, _)| i)
        .collect()
}

/// As [`nulls_2d`], for a vector field.
pub fn nulls_3d(direction: &[[f64; 3]], threshold: f64) -> Vec<usize> {
    direction
        .iter()
        .enumerate()
        .filter(|(_, d)| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < threshold)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null evaluates, and nothing in the field is NaN or infinite.
    ///
    /// This is the division by zero the regularisation exists for: the middle
    /// entry is an exact zero vorticity, which `w / |w|` would turn into NaN.
    #[test]
    fn a_null_evaluates_to_zero_and_no_entry_is_nan() {
        let omega = vec![3.0, 0.0, -3.0, f64::INFINITY, f64::NEG_INFINITY];
        let d = direction_2d(&omega, Epsilon::Absolute(0.1));
        assert_eq!(d[1], 0.0, "an exact null must evaluate to exactly zero");
        for (i, v) in d.iter().enumerate() {
            assert!(v.is_finite(), "entry {i} is {v}");
            assert!(v.abs() <= 1.0, "entry {i} exceeds unit magnitude: {v}");
        }
        assert_eq!(d[3], 1.0, "an infinite vorticity keeps its sign");
        assert_eq!(d[4], -1.0);

        let w3 = vec![[0.0; 3], [0.0, 0.0, 5.0]];
        let d3 = direction_3d(&w3, Epsilon::Absolute(0.1));
        assert_eq!(d3[0], [0.0; 3], "an exact null must evaluate to the zero vector");
        for v in d3.iter().flatten() {
            assert!(v.is_finite());
        }
    }

    /// Far from a null the direction is the unit direction, to the accuracy the
    /// module documents: `eps^2 / 2 |w|^2`.
    ///
    /// Drives `direction_2d` itself. An earlier version measured the local
    /// `sqrt_form` helper, which left the test passing for any shipped
    /// implementation at all.
    #[test]
    fn the_direction_recovers_the_unit_direction_away_from_a_null() {
        let e = 0.1_f64;
        let omega = vec![1.0_f64, -1.0, 10.0, -50.0];
        let got = direction_2d(&omega, Epsilon::Absolute(e));
        for (i, &w) in omega.iter().enumerate() {
            let want = w.signum();
            let bound = e * e / (2.0 * w * w);
            assert!(
                (got[i] - want).abs() <= bound * 1.001,
                "at w={w} the direction is {}, off the unit direction by more than {bound:.3e}",
                got[i]
            );
        }
    }

    /// A vorticity too large to square still reads as a direction.
    ///
    /// `w * w` overflows to infinity above about `1.34e154`, which gives
    /// `w / inf = 0.0`. That is finite, so no fallback fires, and the strongest
    /// possible vorticity would report the one value reserved for a null.
    #[test]
    fn a_vorticity_too_large_to_square_is_not_read_as_a_null() {
        let big = 1e200_f64;
        let d = direction_2d(&[big, -big], Epsilon::Absolute(1.0));
        assert!((d[0] - 1.0).abs() < 1e-12, "a huge positive vorticity gave {}", d[0]);
        assert!((d[1] + 1.0).abs() < 1e-12, "a huge negative vorticity gave {}", d[1]);

        let d3 = direction_3d(&[[big, 0.0, 0.0]], Epsilon::Absolute(1.0));
        let m = (d3[0][0] * d3[0][0] + d3[0][1] * d3[0][1] + d3[0][2] * d3[0][2]).sqrt();
        assert!((m - 1.0).abs() < 1e-12, "a huge vector vorticity gave magnitude {m}");
    }

    /// The square-root form has a continuous second derivative across a null and
    /// the absolute form does not.
    ///
    /// Both are odd, so a central second difference AT the origin vanishes for
    /// either and states nothing. The difference lives on the two sides: for
    /// `w / (|w| + eps)` the second derivative approaches `-2/eps^2` from above
    /// and `+2/eps^2` from below, a jump of `4/eps^2` that puts a crease along
    /// every null curve of a streamline plot or an isosurface.
    ///
    /// The measured side is `direction_2d` ITSELF, sampled through the shipped
    /// function, so replacing it with the absolute form makes this test fail.
    /// An earlier version compared two local helpers and bound nothing.
    #[test]
    fn the_square_root_form_is_smooth_where_the_absolute_form_kinks() {
        let e = 1e-2_f64;
        let h = e * 1e-3;

        // Second difference of the SHIPPED function at `x`, evaluated through
        // one call so the epsilon resolution is the shipped one too.
        let shipped_second = |x: f64| {
            let d = direction_2d(&[x + h, x, x - h], Epsilon::Absolute(e));
            (d[0] - 2.0 * d[1] + d[2]) / (h * h)
        };
        let absolute_second = |x: f64| {
            let f = |w: f64| w / (w.abs() + e);
            (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
        };

        let sq_jump = (shipped_second(2.0 * h) - shipped_second(-2.0 * h)).abs();
        let ab_jump = (absolute_second(2.0 * h) - absolute_second(-2.0 * h)).abs();

        // The absolute form's jump is the analytic `4 / eps^2`, which is the
        // discriminating quantity; without it this test states nothing.
        let scale = 1.0 / (e * e);
        assert!(
            ab_jump > 3.0 * scale,
            "the absolute form must jump by about 4/eps^2 = {:.3e}, measured {ab_jump:.3e}",
            4.0 * scale
        );
        assert!(
            sq_jump < ab_jump / 100.0,
            "the shipped direction should barely move where the absolute form jumps: \
             {sq_jump:.4e} against {ab_jump:.4e}"
        );
    }

    /// Scaling the field and the regularisation together leaves the direction
    /// alone, which is what makes a relative epsilon the sensible default.
    ///
    /// The only test of `Epsilon::Relative` on a field that has a scale, and the
    /// only coverage `rms_3d` has. It was deleted in the review-fix wave and
    /// restored here.
    #[test]
    fn the_direction_is_invariant_under_a_shared_rescaling() {
        let omega: Vec<f64> = vec![-2.0, -0.5, 0.0, 0.25, 3.0];
        let a = direction_2d(&omega, Epsilon::Relative(1e-3));
        let scaled: Vec<f64> = omega.iter().map(|w| w * 1e6).collect();
        let b = direction_2d(&scaled, Epsilon::Relative(1e-3));
        for i in 0..omega.len() {
            assert!(
                (a[i] - b[i]).abs() < 1e-12,
                "entry {i} moved under rescaling: {} against {}",
                a[i],
                b[i]
            );
        }

        let w3 = vec![[1.0, -2.0, 0.5], [0.0; 3], [-3.0, 0.25, 4.0]];
        let a3 = direction_3d(&w3, Epsilon::Relative(1e-3));
        let s3: Vec<[f64; 3]> = w3.iter().map(|w| [w[0] * 1e6, w[1] * 1e6, w[2] * 1e6]).collect();
        let b3 = direction_3d(&s3, Epsilon::Relative(1e-3));
        for i in 0..w3.len() {
            for k in 0..3 {
                assert!(
                    (a3[i][k] - b3[i][k]).abs() < 1e-12,
                    "vector entry {i} component {k} moved under rescaling"
                );
            }
        }
        assert!(rms_3d(&w3) > 0.0, "rms_3d must see a scale in a nonzero field");
    }

    /// The magnitude doubles as the null mask, which is what saves a separate
    /// field in a picture.
    #[test]
    fn the_magnitude_marks_the_nulls() {
        let e = 0.1_f64;
        let omega = vec![0.0, e * 0.01, e * 10.0, -e * 10.0];
        let d = direction_2d(&omega, Epsilon::Absolute(e));
        let marked = nulls_2d(&d, 0.1);
        assert_eq!(marked, vec![0, 1], "only the two points near the null should be marked");

        let d3 = direction_3d(&[[0.0; 3], [0.0, 0.0, e * 10.0]], Epsilon::Absolute(e));
        assert_eq!(nulls_3d(&d3, 0.1), vec![0]);
    }

    /// A field with no vorticity anywhere has no scale of its own, and a
    /// relative epsilon must not divide by the zero that gives.
    #[test]
    fn an_everywhere_zero_field_gives_a_zero_direction() {
        let d = direction_2d(&[0.0; 4], Epsilon::Relative(1e-3));
        assert_eq!(d, vec![0.0; 4]);
        for v in d {
            assert!(v.is_finite());
        }
    }
}
