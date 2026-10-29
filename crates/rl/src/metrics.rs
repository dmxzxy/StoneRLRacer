//! 训练指标（纯数据，app 负责展示）。

use sim::episode::EpisodeResult;
use sim::reward::RewardParts;

/// 每 K 步合并一条的训练指标。
#[derive(Clone, Copy, Debug, Default)]
pub struct StepMetrics {
    pub step: u64,
    pub loss: f32,
    /// DQN：Q(s,a) 均值；PPO 无此概念（0）。
    pub q_mean: f32,
    /// DQN：探索率；PPO 恒为 0（策略自带随机性）。
    pub epsilon: f32,
    /// DQN：replay buffer 长度；PPO：本 rollout 已收集步数。
    pub buffer_len: usize,
    /// 额外通道：PPO 存策略熵，DQN 为 0。
    pub extra: f32,
    /// PPO：价值网络对状态的估值均值；DQN 为 0。
    pub value: f32,
}

/// 一局的摘要。
#[derive(Clone, Copy, Debug)]
pub struct EpisodeSummary {
    pub episode: u64,
    pub steps: u32,
    pub reward: f32,
    pub avg_speed: f32,
    pub result: EpisodeResult,
    /// 完圈时为圈时（秒），否则无意义。
    pub lap_time: f32,
    /// 本局最高车速（m/s）。
    pub max_speed: f32,
    /// 本局奖励分解累计。
    pub parts: RewardParts,
}

impl Default for EpisodeSummary {
    fn default() -> Self {
        Self {
            episode: 0,
            steps: 0,
            reward: 0.0,
            avg_speed: 0.0,
            result: EpisodeResult::Running,
            lap_time: 0.0,
            max_speed: 0.0,
            parts: RewardParts::ZERO,
        }
    }
}

impl EpisodeSummary {
    pub fn has_lap(&self) -> bool {
        matches!(self.result, EpisodeResult::LapDone)
    }
}
