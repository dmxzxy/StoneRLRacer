//! 奖励函数：进度 + 速度 + 走线 + 时间成本，量级控制在 O(1)。
//!
//! 奖励按分量计算（[`RewardParts`]），既能合成标量奖励，也能在训练看板上
//! 拆开来看——这是理解"策略到底在追求什么"的关键。

/// 撞墙/完圈的终局奖励在 `episode.rs` 里加减。
#[derive(Clone, Copy, Debug)]
pub struct RewardConfig {
    /// 每米中心线弧长进度的奖励（带符号，倒车自然受罚）。
    pub w_progress: f32,
    /// 速度奖励系数（作用于 speed/max_speed）。
    pub w_speed: f32,
    /// 偏离中心线惩罚系数（作用于 (lateral/half_w)²）。
    pub w_center: f32,
    /// 每步固定时间成本。
    pub w_time: f32,
    /// 撞墙惩罚（取正值，使用时减去）。
    pub r_crash: f32,
    /// 完圈奖励。
    pub r_lap: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            w_progress: 0.5,
            w_speed: 0.2,
            w_center: 0.3,
            w_time: 0.05,
            r_crash: 5.0,
            r_lap: 10.0,
        }
    }
}

/// 一步奖励的各分量。
///
/// 约定：`progress` / `speed` / `lap` 通常为正，`center` / `time` / `crash`
/// 通常为负（惩罚项直接存负值，方便看板上叠加）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardParts {
    /// 沿中心线的带符号弧长进度奖励。
    pub progress: f32,
    /// 速度奖励。
    pub speed: f32,
    /// 偏离中心线罚分（≤0）。
    pub center: f32,
    /// 每步时间成本（≤0）。
    pub time: f32,
    /// 撞墙罚分（≤0，仅终结步）。
    pub crash: f32,
    /// 完圈奖励（≥0，仅终结步）。
    pub lap: f32,
}

impl RewardParts {
    pub const ZERO: Self = Self {
        progress: 0.0,
        speed: 0.0,
        center: 0.0,
        time: 0.0,
        crash: 0.0,
        lap: 0.0,
    };

    /// 合成标量奖励。
    pub fn total(&self) -> f32 {
        self.progress + self.speed + self.center + self.time + self.crash + self.lap
    }

    /// 累加（按局统计用）。
    pub fn add(&mut self, other: &Self) {
        self.progress += other.progress;
        self.speed += other.speed;
        self.center += other.center;
        self.time += other.time;
        self.crash += other.crash;
        self.lap += other.lap;
    }

    /// 各项的展示名（顺序与 [`Self::channels`] 一致）。
    pub const LABELS: [&'static str; 6] = ["进度", "速度", "走线", "时间", "撞墙", "完圈"];

    /// 各分量数值，顺序与 [`Self::LABELS`] 一致。
    pub fn channels(&self) -> [f32; 6] {
        [
            self.progress,
            self.speed,
            self.center,
            self.time,
            self.crash,
            self.lap,
        ]
    }
}

/// 每步（非终局）奖励的分解。
pub fn step_reward_parts(
    cfg: &RewardConfig,
    ds: f32,
    speed_frac: f32,
    lateral_frac: f32,
) -> RewardParts {
    RewardParts {
        progress: cfg.w_progress * ds,
        speed: cfg.w_speed * speed_frac,
        center: -cfg.w_center * lateral_frac * lateral_frac,
        time: -cfg.w_time,
        crash: 0.0,
        lap: 0.0,
    }
}

/// 每步（非终局）奖励。
pub fn step_reward(cfg: &RewardConfig, ds: f32, speed_frac: f32, lateral_frac: f32) -> f32 {
    step_reward_parts(cfg, ds, speed_frac, lateral_frac).total()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_fast_is_positive() {
        let cfg = RewardConfig::default();
        // 20 m/s、dt=0.05 → 每步前进 1 米
        let r = step_reward(&cfg, 1.0, 0.67, 0.0);
        assert!(r > 0.3, "r={r}");
    }

    #[test]
    fn standing_still_is_negative() {
        let cfg = RewardConfig::default();
        let r = step_reward(&cfg, 0.0, 0.0, 0.0);
        assert!(r < -0.04, "r={r}");
    }

    #[test]
    fn reversing_is_strongly_negative() {
        let cfg = RewardConfig::default();
        let r = step_reward(&cfg, -1.0, -0.2, 0.5);
        assert!(r < -0.5, "r={r}");
    }

    #[test]
    fn off_center_penalized() {
        let cfg = RewardConfig::default();
        let center = step_reward(&cfg, 1.0, 0.5, 0.0);
        let edge = step_reward(&cfg, 1.0, 0.5, 1.0);
        assert!(edge < center);
    }

    #[test]
    fn parts_sum_to_scalar_reward() {
        let cfg = RewardConfig::default();
        let parts = step_reward_parts(&cfg, 0.7, 0.4, -0.6);
        assert!((parts.total() - step_reward(&cfg, 0.7, 0.4, -0.6)).abs() < 1e-6);
        assert!(parts.center < 0.0);
        assert!(parts.time < 0.0);
        assert_eq!(parts.channels().len(), RewardParts::LABELS.len());
    }

    #[test]
    fn parts_add_accumulates() {
        let mut a = RewardParts { progress: 1.0, ..RewardParts::ZERO };
        a.add(&RewardParts { crash: -5.0, ..RewardParts::ZERO });
        assert_eq!(a.total(), -4.0);
    }
}
