//! Generic explicit integrators over a minimal field vector-space trait.

/// A field that supports the affine combination integrators need.
pub trait FieldVec: Clone {
    /// Return `self + other * factor` as a new field.
    fn add_scaled(&self, other: &Self, factor: f64) -> Self;
}

/// Classic RK4 step: `rhs(state) -> d state / dt`.
pub fn rk4<F, R>(state: &F, dt: f64, rhs: R) -> F
where
    F: FieldVec,
    R: Fn(&F) -> F,
{
    let k1 = rhs(state);
    let k2 = rhs(&state.add_scaled(&k1, 0.5 * dt));
    let k3 = rhs(&state.add_scaled(&k2, 0.5 * dt));
    let k4 = rhs(&state.add_scaled(&k3, dt));
    // state + (dt/6)(k1 + 2k2 + 2k3 + k4)
    let sum = k1
        .add_scaled(&k2, 2.0)
        .add_scaled(&k3, 2.0)
        .add_scaled(&k4, 1.0);
    state.add_scaled(&sum, dt / 6.0)
}

/// Explicit Euler step.
pub fn euler<F, R>(state: &F, dt: f64, rhs: R) -> F
where
    F: FieldVec,
    R: Fn(&F) -> F,
{
    let k = rhs(state);
    state.add_scaled(&k, dt)
}

#[cfg(test)]
mod tests {
    use super::{rk4, euler, FieldVec};

    // Minimal field: a single f64 wrapped so it implements FieldVec.
    #[derive(Clone, PartialEq, Debug)]
    struct Scalar(f64);
    impl FieldVec for Scalar {
        fn add_scaled(&self, other: &Self, factor: f64) -> Self {
            Scalar(self.0 + other.0 * factor)
        }
    }

    #[test]
    fn rk4_integrates_dy_dt_equals_y_to_machine_order() {
        // dy/dt = y, y(0)=1, one step dt=0.1 -> RK4 estimate of e^0.1.
        let y0 = Scalar(1.0);
        let y1 = rk4(&y0, 0.1, |y| Scalar(y.0));
        let exact = 0.1f64.exp();
        assert!((y1.0 - exact).abs() < 1e-6);
    }

    #[test]
    fn euler_step_is_first_order() {
        let y0 = Scalar(1.0);
        let y1 = euler(&y0, 0.1, |y| Scalar(y.0));
        assert!((y1.0 - 1.1).abs() < 1e-15);
    }
}
