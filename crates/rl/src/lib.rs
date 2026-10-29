//! rl-racer RL crate：burn 网络与 DQN/PPO 算法。
//!
//! 与训练线程的接口是 [`Agent`] trait：`act` 选动作，`feedback` 回填环境反馈。

pub mod buffer;
pub mod dqn;
pub mod metrics;
pub mod policy;
pub mod ppo;

use sim::action::{ContinuousAction, DiscreteAction};
use sim::observation::Observation;

/// DQN 的一条经验。
#[derive(Clone, Copy)]
pub struct Transition {
    pub obs: Observation,
    pub action: DiscreteAction,
    pub reward: f32,
    pub next_obs: Observation,
    pub done: bool,
}

/// 环境反馈给 agent 的一步信息。
pub struct Feedback<'a> {
    pub next_obs: &'a Observation,
    pub reward: f32,
    /// 本局是否结束（撞墙/完圈/超时）。
    pub done: bool,
    /// 是否真终局（撞墙/完圈）。超时属于截断，PPO 需要区分以做 bootstrap。
    pub terminal: bool,
}

/// 训练 agent 的统一接口。
pub trait Agent {
    /// 选择动作（PPO 会同时把 (obs, action, log_prob, value) 记入 rollout）。
    fn act(&mut self, obs: &Observation) -> ContinuousAction;
    /// 回填环境反馈。DQN 在此入 replay buffer 并周期训练；PPO 在此攒 rollout。
    fn feedback(&mut self, fb: Feedback<'_>);
    /// 局结束钩子（统计用）。
    fn on_episode_end(&mut self) {}
    /// 环境（赛道 / 车型）切换后调用。默认什么都不做；
    /// DQN 会丢弃 replay buffer——旧物理下的过渡会污染新环境的估值。
    fn on_env_changed(&mut self) {}
    /// 最近一次训练指标。
    fn metrics(&self) -> metrics::StepMetrics;
    /// 当前策略快照（供 UI 线程推理副本）。
    fn snapshot(&self) -> policy::PolicySnapshot;
    /// 从已保存的模型恢复（继续训练）。默认不支持。
    fn restore(&mut self, _record: burn::store::ModuleRecord, _steps: u64) {}
}
