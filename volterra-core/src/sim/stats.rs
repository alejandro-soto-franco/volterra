//! The internal per-step statistics currency of the runner loop.
//!
//! All fields are `Option` so each `PhysicsStep` populates only what it
//! measures. Public runner structs convert from this via `From<StepStats>`.

/// Diagnostics produced by one `PhysicsStep::step` and consumed by observers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepStats {
    /// Simulation time at the end of the step.
    pub time: Option<f64>,
    /// Mean scalar order parameter.
    pub order_param: Option<f64>,
    /// Total free / elastic energy, if computed.
    pub energy: Option<f64>,
    /// Detected defect / disclination count.
    pub defect_count: Option<usize>,
    /// Maximum velocity divergence (incompressibility residual).
    pub max_divergence: Option<f64>,
}

impl StepStats {
    /// Set the time field (builder style).
    pub fn with_time(mut self, t: f64) -> Self { self.time = Some(t); self }
    /// Set the order parameter (builder style).
    pub fn with_order_param(mut self, s: f64) -> Self { self.order_param = Some(s); self }
    /// Set the energy (builder style).
    pub fn with_energy(mut self, e: f64) -> Self { self.energy = Some(e); self }
    /// Set the defect count (builder style).
    pub fn with_defect_count(mut self, n: usize) -> Self { self.defect_count = Some(n); self }
    /// Set the max divergence (builder style).
    pub fn with_max_divergence(mut self, d: f64) -> Self { self.max_divergence = Some(d); self }
}

#[cfg(test)]
mod tests {
    use super::StepStats;

    #[test]
    fn default_is_all_none_except_constructible() {
        let s = StepStats::default();
        assert!(s.order_param.is_none());
        assert!(s.energy.is_none());
        assert!(s.defect_count.is_none());
        assert!(s.max_divergence.is_none());
        assert!(s.time.is_none());
    }

    #[test]
    fn builder_sets_fields() {
        let s = StepStats::default().with_time(1.5).with_order_param(0.8);
        assert_eq!(s.time, Some(1.5));
        assert_eq!(s.order_param, Some(0.8));
    }
}
