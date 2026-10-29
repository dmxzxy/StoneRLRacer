//! 推理策略：训练线程把 ModuleRecord 快照发给 UI，UI 用它在本地环境上跑 greedy 动作。

use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::Device;

use crate::dqn::QNetwork;
use crate::ppo::ActorCritic;
use sim::action::{discrete_to_continuous, ContinuousAction, DiscreteAction, N_ACTIONS};
use sim::observation::{Observation, OBS_DIM};

/// 算法种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgoKind {
    Dqn,
    Ppo,
}

impl AlgoKind {
    pub fn label(self) -> &'static str {
        match self {
            AlgoKind::Dqn => "DQN",
            AlgoKind::Ppo => "PPO",
        }
    }
}

/// 策略快照（ModuleRecord 是 Clone + Send 的非泛型类型，可直接跨线程）。
pub struct PolicySnapshot {
    pub algo: AlgoKind,
    /// 网络隐藏层宽度（加载时结构需一致）。
    pub hidden: usize,
    pub record: ModuleRecord,
}

/// UI 线程持有的推理副本。
pub enum Policy {
    Dqn(QNetwork),
    Ppo(ActorCritic),
}

impl Policy {
    pub fn load(snapshot: PolicySnapshot) -> Self {
        let device = Device::flex();
        match snapshot.algo {
            AlgoKind::Dqn => {
                let net = QNetwork::init(OBS_DIM, N_ACTIONS, snapshot.hidden, &device);
                Policy::Dqn(net.load_record(snapshot.record))
            }
            AlgoKind::Ppo => {
                let cfg = crate::ppo::PpoConfig {
                    hidden: snapshot.hidden,
                    ..Default::default()
                };
                let net = ActorCritic::init(OBS_DIM, &cfg, &device);
                Policy::Ppo(net.load_record(snapshot.record))
            }
        }
    }

    /// greedy 动作（DQN：argmax Q；PPO：均值 μ）。
    pub fn act(&self, obs: &Observation) -> ContinuousAction {
        let device = Device::flex();
        let t = crate::dqn::obs_tensor(&device, std::slice::from_ref(obs));
        match self {
            Policy::Dqn(net) => {
                let q = crate::dqn::t_to_vec(net.forward(t));
                let mut best = 0usize;
                let mut best_v = f32::MIN;
                for (i, v) in q.iter().enumerate() {
                    if *v > best_v {
                        best_v = *v;
                        best = i;
                    }
                }
                discrete_to_continuous(DiscreteAction(best as u8))
            }
            Policy::Ppo(net) => {
                let (mu, _v) = net.forward(t);
                let mu = crate::dqn::t_to_vec(mu);
                ContinuousAction {
                    steer: mu[0].clamp(-1.0, 1.0),
                    throttle: mu[1].clamp(-1.0, 1.0),
                }
            }
        }
    }

    /// 当前状态下策略"在想什么"：DQN 给出 9 个动作的 Q 值，
    /// PPO 给出动作均值/标准差与状态价值。
    pub fn action_values(&self, obs: &Observation) -> ActionValues {
        let device = Device::flex();
        let t = crate::dqn::obs_tensor(&device, std::slice::from_ref(obs));
        match self {
            Policy::Dqn(net) => ActionValues {
                q: Some(crate::dqn::t_to_vec(net.forward(t))),
                ..ActionValues::default()
            },
            Policy::Ppo(net) => {
                let (mu, v) = net.forward(t);
                let mu = crate::dqn::t_to_vec(mu);
                let std = net.std_values();
                ActionValues {
                    q: None,
                    mu: Some([mu[0], mu[1]]),
                    std: Some([
                        std.first().copied().unwrap_or(0.0),
                        std.get(1).copied().unwrap_or(0.0),
                    ]),
                    value: Some(crate::dqn::t_to_scalar(v)),
                }
            }
        }
    }
}

/// 策略解读快照（每帧更新一次，画布角标展示）。
#[derive(Clone, Debug, Default)]
pub struct ActionValues {
    /// DQN：9 个离散动作的 Q 值，索引 = 转向档*3 + 油门档。
    pub q: Option<Vec<f32>>,
    /// PPO：动作均值 [steer, throttle]。
    pub mu: Option<[f32; 2]>,
    /// PPO：动作标准差 [steer, throttle]。
    pub std: Option<[f32; 2]>,
    /// PPO：状态价值估计。
    pub value: Option<f32>,
}

impl ActionValues {
    /// DQN 的 argmax 动作索引。
    pub fn greedy_index(&self) -> Option<usize> {
        let q = self.q.as_ref()?;
        let mut best = 0usize;
        let mut best_v = f32::MIN;
        for (i, v) in q.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best = i;
            }
        }
        Some(best)
    }
}
