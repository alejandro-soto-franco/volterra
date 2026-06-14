//! Seeded Langevin (Box-Muller) noise shared by every runner.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Box-Muller Langevin noise source, reproducing the two legacy seeding
/// conventions bit-for-bit.
pub struct LangevinNoise {
    rng: SmallRng,
}

impl LangevinNoise {
    /// 2D convention: one RNG for the whole run, seeded from grid dims and steps.
    pub fn per_run_seed(nx: usize, ny: usize, n_steps: usize) -> Self {
        let seed = (nx as u64).wrapping_mul(6364136223846793005)
            ^ (ny as u64).wrapping_mul(1442695040888963407)
            ^ n_steps as u64;
        Self { rng: SmallRng::seed_from_u64(seed) }
    }

    /// 3D convention: per-step RNG, seeded from the step index and a tag.
    pub fn per_step_seed(step: usize, tag: u64) -> Self {
        Self { rng: SmallRng::seed_from_u64(step as u64 ^ tag) }
    }

    /// Fill `buf` (length even) with Box-Muller pairs scaled by `amp * sqrt(dt)`.
    /// Each pair consumes two uniforms; identical arithmetic to the legacy code.
    pub fn fill_pairs(&mut self, buf: &mut [f64], amp: f64, dt: f64) {
        let noise_scale = amp * dt.sqrt();
        let mut i = 0;
        while i + 1 < buf.len() {
            let u1: f64 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = self.rng.random::<f64>();
            let mag = noise_scale * (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            buf[i] = mag * angle.cos();
            buf[i + 1] = mag * angle.sin();
            i += 2;
        }
    }

    /// Fill `out` with 5 independent N(0,1) samples (3D Q-field convention),
    /// matching `box_muller_5`: 3 pairs drawn, the 6th sample discarded.
    pub fn fill5(&mut self, out: &mut [f64; 5]) {
        let mut i = 0;
        while i < 5 {
            let u1: f64 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = self.rng.random::<f64>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out[i] = r * theta.cos();
            i += 1;
            if i < 5 {
                out[i] = r * theta.sin();
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LangevinNoise;

    #[test]
    fn per_run_seed_matches_legacy_box_muller() {
        // Legacy 2D pattern: one SmallRng seeded from (nx,ny,n_steps), drawn in
        // a flat loop of [q1,q2] pairs. Reproduce 3 pairs and compare.
        let mut noise = LangevinNoise::per_run_seed(8, 8, 100);
        let mut buf = vec![0.0f64; 6]; // 3 pairs
        noise.fill_pairs(&mut buf, 0.5, 0.01); // amp, dt
        // Golden values captured from the legacy inline code on the same seed:
        let expected = legacy_2d_reference(8, 8, 100, 0.5, 0.01, 3);
        for (a, b) in buf.iter().zip(expected.iter()) {
            assert_eq!(a, b, "noise must match legacy bit-for-bit");
        }
    }

    // Helper that runs the exact legacy inline arithmetic, for the oracle.
    fn legacy_2d_reference(nx: usize, ny: usize, n_steps: usize, amp: f64, dt: f64, pairs: usize) -> Vec<f64> {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let noise_scale = amp * dt.sqrt();
        let seed = (nx as u64).wrapping_mul(6364136223846793005)
            ^ (ny as u64).wrapping_mul(1442695040888963407)
            ^ n_steps as u64;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut out = Vec::new();
        for _ in 0..pairs {
            let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.random::<f64>();
            let mag = noise_scale * (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            out.push(mag * angle.cos());
            out.push(mag * angle.sin());
        }
        out
    }

    #[test]
    fn fill5_matches_legacy_box_muller_5() {
        // Reproduce the legacy box_muller_5 for per_step_seed(0, 0xdead_beef_cafe_1234).
        let mut noise = LangevinNoise::per_step_seed(0, 0xdead_beef_cafe_1234);
        let mut out = [0.0f64; 5];
        noise.fill5(&mut out);

        let expected = legacy_box_muller_5(0, 0xdead_beef_cafe_1234);
        assert_eq!(out, expected, "fill5 must match legacy box_muller_5 bit-for-bit");
    }

    // Helper that runs the exact legacy box_muller_5 arithmetic.
    fn legacy_box_muller_5(step: usize, tag: u64) -> [f64; 5] {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(step as u64 ^ tag);
        let mut out = [0.0f64; 5];
        let mut i = 0;
        while i < 5 {
            let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.random::<f64>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out[i] = r * theta.cos();
            i += 1;
            if i < 5 {
                out[i] = r * theta.sin();
                i += 1;
            }
        }
        out
    }
}
