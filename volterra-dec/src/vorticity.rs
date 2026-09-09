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
    let e2 = e * e;
    omega
        .iter()
        .map(|&w| {
            // An exact zero is a null and answers zero. Taking the fallback
            // below would answer `0.0_f64.signum()`, which is 1.0 in Rust, and
            // `e2` underflows to zero for a field with no vorticity anywhere,
            // so this branch is reached rather than hypothetical.
            if w == 0.0 {
                return 0.0;
            }
            let d = w / (w * w + e2).sqrt();
            if d.is_finite() { d } else { w.signum() }
        })
        .collect()
}

/// Regularised direction of a vector vorticity field, of magnitude below one.
///
/// The zero vector where the vorticity is exactly zero, and finite for every
/// input.
pub fn direction_3d(omega: &[[f64; 3]], eps: Epsilon) -> Vec<[f64; 3]> {
    let e = eps.resolve(rms_3d(omega));
    let e2 = e * e;
    omega
        .iter()
        .map(|w| {
            let mag2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            let den = (mag2 + e2).sqrt();
            let mut out = [0.0; 3];
            for k in 0..3 {
                let d = w[k] / den;
                out[k] = if d.is_finite() { d } else { 0.0 };
            }
            out
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

    /// The form this module rejects, kept so the tests can state the difference
    /// against it rather than assert smoothness of the chosen one alone.
    fn absolute_form(w: f64, e: f64) -> f64 {
        w / (w.abs() + e)
    }

    fn sqrt_form(w: f64, e: f64) -> f64 {
        w / (w * w + e * e).sqrt()
    }

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
    #[test]
    fn the_direction_recovers_the_unit_direction_away_from_a_null() {
        let e = 0.1_f64;
        for &w in &[1.0_f64, -1.0, 10.0, -50.0] {
            let d = sqrt_form(w, e);
            let want = w.signum();
            let bound = e * e / (2.0 * w * w);
            assert!(
                (d - want).abs() <= bound * 1.001,
                "at w={w} the direction is {d}, off the unit direction by more than {bound:.3e}"
            );
        }
    }

    /// Scaling the field and the regularisation together leaves the direction
    /// alone, which is what makes a relative epsilon the sensible default.
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
    }

    /// The square-root form has a continuous second derivative across a null and
    /// the absolute form does not.
    ///
    /// Both are odd, so a central second difference AT the origin vanishes for
    /// either and states nothing. The difference lives on the two sides: for
    /// `w / (|w| + eps)` the second derivative approaches `-2/eps^2` from above
    /// and `+2/eps^2` from below, a jump of `4/eps^2` that puts a crease along
    /// every null curve of a streamline plot or an isosurface. For the
    /// square-root form the second derivative is `-3 eps^2 w (w^2+eps^2)^{-5/2}`,
    /// which passes through zero continuously.
    #[test]
    fn the_square_root_form_is_smooth_where_the_absolute_form_kinks() {
        let e = 1e-2_f64;
        let h = e * 1e-3;
        let second = |f: &dyn Fn(f64, f64) -> f64, x: f64| {
            (f(x + h, e) - 2.0 * f(x, e) + f(x - h, e)) / (h * h)
        };

        let sq_hi = second(&sqrt_form, 2.0 * h);
        let sq_lo = second(&sqrt_form, -2.0 * h);
        let ab_hi = second(&absolute_form, 2.0 * h);
        let ab_lo = second(&absolute_form, -2.0 * h);

        let sq_jump = (sq_hi - sq_lo).abs();
        let ab_jump = (ab_hi - ab_lo).abs();

        // The absolute form's jump is the analytic `4 / eps^2`, which is the
        // discriminating quantity; without it this test states nothing about
        // the chosen form.
        let scale = 1.0 / (e * e);
        assert!(
            ab_jump > 3.0 * scale,
            "the absolute form must jump by about 4/eps^2 = {:.3e}, measured {ab_jump:.3e}",
            4.0 * scale
        );

        // The square-root form's second derivative is `-3 eps^2 w (w^2+eps^2)^{-5/2}`,
        // which is small near the null and vanishes at it, so the two forms are
        // compared against each other. An absolute bound would encode the
        // sample offset instead of the property.
        assert!(
            sq_jump < ab_jump / 100.0,
            "the square-root form should barely move where the absolute form jumps: \
             {sq_jump:.4e} against {ab_jump:.4e}, a ratio of {:.3e}",
            sq_jump / ab_jump
        );
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
