//! 确定性 PRNG（splitmix64）。零依赖、同 seed 完全可复现。

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// [0, 1) 均匀分布。
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// [lo, hi) 均匀分布。
    pub fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// 标准正态分布（Box-Muller）。
    pub fn next_std_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn range_bounds() {
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let v = rng.next_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&v));
        }
    }

    #[test]
    fn std_normal_reasonable() {
        let mut rng = Rng::new(9);
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let v = rng.next_std_normal();
            assert!(v.abs() < 6.0);
            sum += v;
        }
        assert!((sum / n as f32).abs() < 0.1);
    }
}
