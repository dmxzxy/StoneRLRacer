//! Double DQN：MLP Q 网络 + 环形 replay buffer + epsilon-greedy + 周期 target 同步。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::buffer::ReplayBuffer;
use crate::metrics::StepMetrics;
use crate::{policy::PolicySnapshot, Agent, Feedback, Transition};
use sim::action::{discrete_to_continuous, random_discrete, ContinuousAction, DiscreteAction, N_ACTIONS};
use sim::observation::{Observation, OBS_DIM};
use sim::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct DqnConfig {
    pub hidden: usize,
    pub buffer_size: usize,
    pub batch: usize,
    /// 开始训练前先收集的过渡数。
    pub warmup: usize,
    /// 每隔多少环境步训练一次。
    pub train_every: usize,
    /// target 网络硬同步间隔（环境步）。
    pub target_sync: usize,
    pub gamma: f32,
    pub lr: f32,
    pub eps_start: f32,
    pub eps_end: f32,
    pub eps_decay_steps: u64,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            buffer_size: 100_000,
            batch: 128,
            warmup: 2_000,
            train_every: 4,
            target_sync: 1_000,
            gamma: 0.99,
            lr: 1e-3,
            eps_start: 1.0,
            eps_end: 0.05,
            eps_decay_steps: 50_000,
        }
    }
}

#[derive(Module, Debug)]
pub struct QNetwork {
    fc1: Linear,
    fc2: Linear,
    out: Linear,
    activation: Relu,
}

impl QNetwork {
    pub fn init(obs_dim: usize, n_actions: usize, hidden: usize, device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, hidden).init(device),
            out: LinearConfig::new(hidden, n_actions).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));
        self.out.forward(x)
    }
}

pub struct DqnAgent {
    cfg: DqnConfig,
    /// autodiff 设备：online/target 都建在其上。target 的梯度在
    /// `GradientsParams::from_grads(.., &online)` 时被过滤，不参与更新。
    device: Device,
    online: QNetwork,
    target: QNetwork,
    opt: ModuleOptimizer,
    buffer: ReplayBuffer,
    rng: Rng,
    step: u64,
    epsilon: f32,
    last_loss: f32,
    last_q_mean: f32,
    /// act() 暂存的 (obs, action)，等 feedback() 拼成完整 Transition。
    pending: Option<(Observation, DiscreteAction)>,
}

impl DqnAgent {
    pub fn new(cfg: DqnConfig, seed: u64) -> Self {
        let device = Device::flex().autodiff();
        let online = QNetwork::init(OBS_DIM, N_ACTIONS, cfg.hidden, &device);
        let target = online.clone();
        let opt = AdamConfig::new().init();
        Self {
            cfg,
            device,
            online,
            target,
            opt,
            buffer: ReplayBuffer::new(cfg.buffer_size),
            rng: Rng::new(seed),
            step: 0,
            epsilon: cfg.eps_start,
            last_loss: 0.0,
            last_q_mean: 0.0,
            pending: None,
        }
    }

    /// 从已保存的模型恢复，继续训练。
    ///
    /// 优化器动量（Adam moments）与 replay buffer 不保留：buffer 重新收集
    /// （warmup 期间用恢复后的低 ε 策略填充，质量有保证）；ε 按已训练步数重算。
    pub fn restore(&mut self, record: burn::store::ModuleRecord, steps: u64) {
        self.online = self.online.clone().load_record(record);
        self.target = self.online.clone();
        self.step = steps;
        let t = (steps as f32 / self.cfg.eps_decay_steps as f32).min(1.0);
        self.epsilon = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * t;
        self.buffer = ReplayBuffer::new(self.cfg.buffer_size);
    }

    fn q_values(&self, obs: &Observation) -> Vec<f32> {
        let t = obs_tensor(&self.device, std::slice::from_ref(obs));
        let q = self.online.forward(t);
        t_to_vec(q)
    }

    fn train_batch(&mut self) {
        let batch = self.buffer.sample(self.cfg.batch, &mut self.rng);
        let b = batch.len();
        if b == 0 {
            return;
        }

        let mut states = Vec::with_capacity(b * OBS_DIM);
        let mut next_states = Vec::with_capacity(b * OBS_DIM);
        let mut actions = Vec::with_capacity(b);
        let mut rewards = Vec::with_capacity(b);
        let mut not_dones = Vec::with_capacity(b);
        for t in &batch {
            states.extend_from_slice(t.obs.as_slice());
            next_states.extend_from_slice(t.next_obs.as_slice());
            actions.push(t.action.index() as i64);
            rewards.push(t.reward);
            not_dones.push(if t.done { 0.0f32 } else { 1.0 });
        }

        let states = Tensor::<2>::from_data(
            TensorData::new(states, [b, OBS_DIM]).convert::<f32>(),
            &self.device,
        );
        let next_states = Tensor::<2>::from_data(
            TensorData::new(next_states, [b, OBS_DIM]).convert::<f32>(),
            &self.device,
        );
        let actions_idx =
            Tensor::<2, Int>::from_data(TensorData::new(actions, [b, 1]), &self.device);
        let rewards_t = Tensor::<1>::from_data(
            TensorData::new(rewards, [b]).convert::<f32>(),
            &self.device,
        );
        let not_done = Tensor::<1>::from_data(
            TensorData::new(not_dones, [b]).convert::<f32>(),
            &self.device,
        );

        // Q(s, a)
        let q = self.online.forward(states);
        let q_sa = q.gather(1, actions_idx).squeeze_dim::<1>(1);

        // Double DQN：a* = argmax_a Q_online(s')，y = r + γ·Q_target(s')[a*]
        let next_q_online = self.online.forward(next_states.clone());
        let best = next_q_online.argmax(1); // [b, 1] Int
        let next_q_target = self.target.forward(next_states);
        let best_next = next_q_target.gather(1, best).squeeze_dim::<1>(1);
        let y = rewards_t + best_next * self.cfg.gamma * not_done;

        // Huber loss：|x|≤1 时 0.5x²，否则 |x|-0.5
        let diff = q_sa.clone() - y;
        let a = diff.abs();
        let a_clip = a.clone().clamp(0.0, 1.0);
        let huber = a_clip.clone() * a_clip * 0.5 + (a - 1.0).clamp(0.0, f32::MAX);
        let loss = huber.mean();

        // 指标（detach 路径上直接读值）
        let q_mean = t_to_scalar(q_sa.clone().mean());
        let loss_v = t_to_scalar(loss.clone());

        let grads = GradientsParams::from_grads(loss.backward(), &self.online);
        self.online = self.opt.step(self.cfg.lr as f64, self.online.clone(), grads);

        self.last_loss = loss_v;
        self.last_q_mean = q_mean;
    }
}

impl Agent for DqnAgent {
    fn act(&mut self, obs: &Observation) -> ContinuousAction {
        self.step += 1;
        // epsilon 线性衰减
        let t = (self.step as f32 / self.cfg.eps_decay_steps as f32).min(1.0);
        self.epsilon = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * t;

        let action = if self.rng.next_f32() < self.epsilon {
            random_discrete(&mut self.rng)
        } else {
            let q = self.q_values(obs);
            let mut best = 0usize;
            let mut best_v = f32::MIN;
            for (i, v) in q.iter().enumerate() {
                if *v > best_v {
                    best_v = *v;
                    best = i;
                }
            }
            DiscreteAction(best as u8)
        };
        self.pending = Some((*obs, action));
        discrete_to_continuous(action)
    }

    fn feedback(&mut self, fb: Feedback<'_>) {
        if let Some((obs, action)) = self.pending.take() {
            self.buffer.push(Transition {
                obs,
                action,
                reward: fb.reward,
                next_obs: *fb.next_obs,
                done: fb.done,
            });
        }
        if self.step % self.cfg.train_every as u64 == 0
            && self.buffer.len() >= self.cfg.warmup
        {
            self.train_batch();
        }
        if self.step % self.cfg.target_sync as u64 == 0 && self.buffer.len() >= self.cfg.warmup {
            // Module 的 clone 是深拷贝：直接整体替换 target
            self.target = self.online.clone();
        }
    }

    fn metrics(&self) -> StepMetrics {
        StepMetrics {
            step: self.step,
            loss: self.last_loss,
            q_mean: self.last_q_mean,
            epsilon: self.epsilon,
            buffer_len: self.buffer.len(),
            extra: 0.0,
            value: 0.0,
        }
    }

    fn snapshot(&self) -> PolicySnapshot {
        PolicySnapshot {
            algo: crate::policy::AlgoKind::Dqn,
            hidden: self.cfg.hidden,
            record: self.online.clone().into_record(),
        }
    }

    fn restore(&mut self, record: burn::store::ModuleRecord, steps: u64) {
        DqnAgent::restore(self, record, steps);
    }

    fn on_env_changed(&mut self) {
        // 换赛道/换车后旧经验不再代表新物理，清空重收（策略权重保留，继续学）
        self.buffer = ReplayBuffer::new(self.cfg.buffer_size);
        self.pending = None;
    }
}

/// [b, OBS_DIM] 张量。
pub(crate) fn obs_tensor(device: &Device, obs: &[Observation]) -> Tensor<2> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.data.iter().copied()).collect();
    Tensor::<2>::from_data(
        TensorData::new(flat, [obs.len(), OBS_DIM]).convert::<f32>(),
        device,
    )
}

pub(crate) fn t_to_vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec_as::<f32>().unwrap()
}

pub(crate) fn t_to_scalar(t: Tensor<1>) -> f32 {
    t.into_data().try_to_vec_as::<f32>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dqn_smoke_train() {
        // 少量步数的冒烟：能 act、能 feedback、能训练、loss 有限
        let cfg = DqnConfig {
            warmup: 16,
            batch: 8,
            train_every: 2,
            target_sync: 8,
            ..Default::default()
        };
        let mut agent = DqnAgent::new(cfg, 1);
        let mut obs = Observation { data: [0.0; OBS_DIM] };
        for _ in 0..64 {
            let _a = agent.act(&obs);
            let mut next = obs;
            next.data[0] += 0.01;
            agent.feedback(Feedback {
                next_obs: &next,
                reward: 1.0,
                done: false,
                terminal: false,
            });
            obs = next;
        }
        let m = agent.metrics();
        assert!(m.loss.is_finite(), "loss={}", m.loss);
        assert!(m.buffer_len > 0);
        let _snap = agent.snapshot();
        // epsilon 已按 step 衰减
        assert!(m.epsilon < 1.0);
    }
}
