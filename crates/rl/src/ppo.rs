//! PPO：actor-critic 双头网络 + GAE + clip surrogate loss。
//!
//! 动作不套 tanh squash（避免 log-det-Jacobian 修正）：N(μ,σ) 采样后 clamp 到 [-1,1]，
//! log_prob 用截断前的高斯公式（常见简化）。

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor, TensorData};

use crate::metrics::StepMetrics;
use crate::{policy::PolicySnapshot, Agent, Feedback};
use sim::action::ContinuousAction;
use sim::observation::{Observation, OBS_DIM};
use sim::rng::Rng;

const ACT_DIM: usize = 2; // steer, throttle
const LOG_2PI: f32 = 1.83787706641;

#[derive(Clone, Copy, Debug)]
pub struct PpoConfig {
    pub hidden: usize,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip: f32,
    pub entropy_coef: f32,
    pub value_coef: f32,
    /// rollout 长度（步）。
    pub rollout: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub lr: f32,
    /// log_std 初始值。
    pub log_std_init: f32,
    /// 采样 std 的下限，防止过早塌缩。
    pub min_std: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            rollout: 2048,
            epochs: 6,
            minibatch: 256,
            lr: 3e-4,
            log_std_init: -0.4, // σ ≈ 0.67
            min_std: 0.05,
        }
    }
}

#[derive(Module, Debug)]
pub struct ActorCritic {
    fc1: Linear,
    fc2: Linear,
    actor: Linear,
    value: Linear,
    activation: Relu,
    /// 每个动作维的 log σ（可学习）。
    pub log_std: Param<Tensor<1>>,
}

impl ActorCritic {
    pub fn init(obs_dim: usize, cfg: &PpoConfig, device: &Device) -> Self {
        let log_std = Tensor::<1>::from_data(
            TensorData::new(vec![cfg.log_std_init; ACT_DIM], [ACT_DIM]).convert::<f32>(),
            device,
        );
        Self {
            fc1: LinearConfig::new(obs_dim, cfg.hidden).init(device),
            fc2: LinearConfig::new(cfg.hidden, cfg.hidden).init(device),
            actor: LinearConfig::new(cfg.hidden, ACT_DIM).init(device),
            value: LinearConfig::new(cfg.hidden, 1).init(device),
            activation: Relu::new(),
            log_std: Param::from_tensor(log_std),
        }
    }

    /// 返回 (动作均值 [B,2], 状态价值 [B])。
    pub fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<1>) {
        let h = self.activation.forward(self.fc1.forward(x));
        let h = self.activation.forward(self.fc2.forward(h));
        let mu = self.actor.forward(h.clone());
        let v = self.value.forward(h).squeeze_dim::<1>(1);
        (mu, v)
    }

    /// 当前 σ（每维，f32）。
    pub fn std_values(&self) -> Vec<f32> {
        let ls: Vec<f32> = self
            .log_std
            .val()
            .clone()
            .into_data()
            .try_to_vec_as::<f32>()
            .unwrap();
        ls.iter().map(|v| v.exp().max(1e-6)).collect()
    }
}

pub struct PpoAgent {
    cfg: PpoConfig,
    device: Device,
    net: ActorCritic,
    opt: ModuleOptimizer,
    rng: Rng,
    step: u64,
    // rollout 收集
    obs_buf: Vec<Observation>,
    act_buf: Vec<[f32; 2]>,
    logp_buf: Vec<f32>,
    val_buf: Vec<f32>,
    rew_buf: Vec<f32>,
    done_buf: Vec<bool>,
    terminal_buf: Vec<bool>,
    /// 超时截断时的下一状态价值（bootstrap），其余为 0。
    bootstrap_buf: Vec<f32>,
    /// act() 暂存、等待 feedback() 回填的下标。
    pending_idx: usize,
    last_loss: f32,
    last_entropy: f32,
}

impl PpoAgent {
    pub fn new(cfg: PpoConfig, seed: u64) -> Self {
        let device = Device::flex().autodiff();
        let net = ActorCritic::init(OBS_DIM, &cfg, &device);
        let opt = AdamConfig::new().init();
        Self {
            cfg,
            device,
            net,
            opt,
            rng: Rng::new(seed),
            step: 0,
            obs_buf: Vec::with_capacity(cfg.rollout),
            act_buf: Vec::with_capacity(cfg.rollout),
            logp_buf: Vec::with_capacity(cfg.rollout),
            val_buf: Vec::with_capacity(cfg.rollout),
            rew_buf: Vec::with_capacity(cfg.rollout),
            done_buf: Vec::with_capacity(cfg.rollout),
            terminal_buf: Vec::with_capacity(cfg.rollout),
            bootstrap_buf: Vec::with_capacity(cfg.rollout),
            pending_idx: 0,
            last_loss: 0.0,
            last_entropy: 0.0,
        }
    }

    /// 从已保存的模型恢复，继续训练（rollout 从零收集，log_std 随 record 恢复）。
    pub fn restore(&mut self, record: burn::store::ModuleRecord, steps: u64) {
        self.net = self.net.clone().load_record(record);
        self.step = steps;
        self.obs_buf.clear();
        self.act_buf.clear();
        self.logp_buf.clear();
        self.val_buf.clear();
        self.rew_buf.clear();
        self.done_buf.clear();
        self.terminal_buf.clear();
        self.bootstrap_buf.clear();
    }

    fn log_prob_f32(mu: &[f32], std: &[f32], a: &[f32; 2]) -> f32 {
        let mut lp = 0.0;
        for d in 0..ACT_DIM {
            let z = (a[d] - mu[d]) / std[d];
            lp += -0.5 * z * z - std[d].ln() - 0.5 * LOG_2PI;
        }
        lp
    }

    fn value_of(&self, obs: &Observation) -> f32 {
        let t = crate::dqn::obs_tensor(&self.device, std::slice::from_ref(obs));
        let (_mu, v) = self.net.forward(t);
        crate::dqn::t_to_scalar(v)
    }

    /// rollout 攒满后的一次更新。`boundary_v`：rollout 末步未结束时其 next_obs 的价值。
    fn update(&mut self, boundary_v: f32) {
        let n = self.obs_buf.len();
        if n < self.cfg.minibatch {
            return;
        }

        // ---- GAE ----
        let mut adv = vec![0.0f32; n];
        let mut last_gae = 0.0f32;
        for t in (0..n).rev() {
            let v_next = if self.done_buf[t] {
                self.bootstrap_buf[t]
            } else if t + 1 < n {
                self.val_buf[t + 1]
            } else {
                boundary_v
            };
            let delta = self.rew_buf[t] + self.cfg.gamma * v_next - self.val_buf[t];
            // 真终局（撞墙/完圈）切断 GAE 传递；超时截断不切断
            let nonterminal = if self.done_buf[t] && self.terminal_buf[t] {
                0.0
            } else {
                1.0
            };
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_gae;
            adv[t] = last_gae;
        }
        let returns: Vec<f32> = (0..n).map(|t| adv[t] + self.val_buf[t]).collect();

        // advantage 归一化
        let mean = adv.iter().sum::<f32>() / n as f32;
        let var = adv.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n as f32;
        let std = var.sqrt().max(1e-6);
        let adv_norm: Vec<f32> = adv.iter().map(|a| (a - mean) / std).collect();

        // 打乱 minibatch 顺序
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        for _ in 0..self.cfg.epochs {
            for chunk in order.chunks(self.cfg.minibatch) {
                let mb = chunk.len();
                let mut states = Vec::with_capacity(mb * OBS_DIM);
                let mut actions = Vec::with_capacity(mb * ACT_DIM);
                let mut old_logp = Vec::with_capacity(mb);
                let mut advs = Vec::with_capacity(mb);
                let mut rets = Vec::with_capacity(mb);
                for &i in chunk {
                    states.extend_from_slice(self.obs_buf[i].as_slice());
                    actions.extend_from_slice(&self.act_buf[i]);
                    old_logp.push(self.logp_buf[i]);
                    advs.push(adv_norm[i]);
                    rets.push(returns[i]);
                }

                let states_t = Tensor::<2>::from_data(
                    TensorData::new(states, [mb, OBS_DIM]).convert::<f32>(),
                    &self.device,
                );
                let actions_t = Tensor::<2>::from_data(
                    TensorData::new(actions, [mb, ACT_DIM]).convert::<f32>(),
                    &self.device,
                );
                let old_logp_t = Tensor::<1>::from_data(
                    TensorData::new(old_logp, [mb]).convert::<f32>(),
                    &self.device,
                );
                let adv_t = Tensor::<1>::from_data(
                    TensorData::new(advs, [mb]).convert::<f32>(),
                    &self.device,
                );
                let ret_t = Tensor::<1>::from_data(
                    TensorData::new(rets, [mb]).convert::<f32>(),
                    &self.device,
                );

                let (mu, value) = self.net.forward(states_t);
                // log_std [2] → [1,2]，与 [mb,2] 广播
                let log_std_t = self.net.log_std.val();
                let log_std_2d = log_std_t.clone().unsqueeze::<2>();

                // log π(a|s) = Σ_d [ -0.5·((a-μ)/σ)² - log σ - 0.5·log2π ]
                let z = (actions_t - mu.clone()) / log_std_2d.clone().exp();
                let logp = ((z.clone() * z) * -0.5 - log_std_2d - 0.5 * LOG_2PI)
                    .sum_dim(1)
                    .squeeze_dim::<1>(1);

                // ratio 与 clip surrogate
                let ratio = (logp - old_logp_t).exp();
                let unclipped = ratio.clone() * adv_t.clone();
                let clipped =
                    ratio.clamp(1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_t;
                // 元素级 min(a,b) = (a + b - |a - b|) / 2（burn 0.22 无元素级 min 方法）
                let mins = (unclipped.clone() + clipped.clone()
                    - (unclipped - clipped).abs())
                    * 0.5;
                let policy_loss = -mins.mean();

                // value loss
                let vdiff = value.clone() - ret_t.clone();
                let value_loss = (vdiff.clone() * vdiff).mean() * 0.5;

                // 熵：Σ_d (log σ + 0.5 + 0.5·log2π)，[1] 张量，梯度流向 log_std
                let entropy = (log_std_t + (0.5 + 0.5 * LOG_2PI)).sum_dim(0);

                let loss = policy_loss + value_loss * self.cfg.value_coef
                    - entropy.clone() * self.cfg.entropy_coef;

                self.last_loss = crate::dqn::t_to_scalar(loss.clone());
                self.last_entropy = crate::dqn::t_to_scalar(entropy);

                let grads = GradientsParams::from_grads(loss.backward(), &self.net);
                self.net = self.opt.step(self.cfg.lr as f64, self.net.clone(), grads);
            }
        }

        // 清空 rollout
        self.obs_buf.clear();
        self.act_buf.clear();
        self.logp_buf.clear();
        self.val_buf.clear();
        self.rew_buf.clear();
        self.done_buf.clear();
        self.terminal_buf.clear();
        self.bootstrap_buf.clear();
    }
}

impl Agent for PpoAgent {
    fn act(&mut self, obs: &Observation) -> ContinuousAction {
        self.step += 1;
        let t = crate::dqn::obs_tensor(&self.device, std::slice::from_ref(obs));
        let (mu, v) = self.net.forward(t);
        let mu_v = crate::dqn::t_to_vec(mu);
        let v_f = crate::dqn::t_to_scalar(v);
        let stds = self.net.std_values();

        // N(μ,σ) 采样（环境动作 clamp，log_prob 用截断前采样值）
        let mut a = [0.0f32; ACT_DIM];
        for d in 0..ACT_DIM {
            a[d] = mu_v[d] + stds[d].max(self.cfg.min_std) * self.rng.next_std_normal();
        }
        let logp = Self::log_prob_f32(&mu_v, &stds, &a);

        self.obs_buf.push(*obs);
        self.act_buf.push(a);
        self.logp_buf.push(logp);
        self.val_buf.push(v_f);
        self.rew_buf.push(0.0); // 由 feedback 回填
        self.done_buf.push(false);
        self.terminal_buf.push(false);
        self.bootstrap_buf.push(0.0);
        self.pending_idx = self.obs_buf.len() - 1;

        ContinuousAction {
            steer: a[0].clamp(-1.0, 1.0),
            throttle: a[1].clamp(-1.0, 1.0),
        }
    }

    fn feedback(&mut self, fb: Feedback<'_>) {
        let i = self.pending_idx;
        self.rew_buf[i] = fb.reward;
        self.done_buf[i] = fb.done;
        self.terminal_buf[i] = fb.terminal;
        if fb.done && !fb.terminal {
            // 超时截断：bootstrap 用下一状态价值
            self.bootstrap_buf[i] = self.value_of(fb.next_obs);
        }
        if self.obs_buf.len() >= self.cfg.rollout {
            // rollout 边界未结束时，用末步 next_obs 的价值 bootstrap
            let last = self.obs_buf.len() - 1;
            let boundary_v = if self.done_buf[last] {
                0.0
            } else {
                self.value_of(fb.next_obs)
            };
            self.update(boundary_v);
        }
    }

    fn metrics(&self) -> StepMetrics {
        // 最近 rollout 的价值估计均值：策略对「我现在有多好」的判断
        let value = if self.val_buf.is_empty() {
            0.0
        } else {
            self.val_buf.iter().sum::<f32>() / self.val_buf.len() as f32
        };
        StepMetrics {
            step: self.step,
            loss: self.last_loss,
            q_mean: 0.0,
            epsilon: 0.0,
            buffer_len: self.obs_buf.len(),
            extra: self.last_entropy,
            value,
        }
    }

    fn snapshot(&self) -> PolicySnapshot {
        PolicySnapshot {
            algo: crate::policy::AlgoKind::Ppo,
            hidden: self.cfg.hidden,
            record: self.net.clone().into_record(),
        }
    }

    fn restore(&mut self, record: burn::store::ModuleRecord, steps: u64) {
        PpoAgent::restore(self, record, steps);
    }

    fn on_env_changed(&mut self) {
        // 丢弃旧环境里攒了一半的 rollout，下一条轨迹从新环境重新开始
        self.obs_buf.clear();
        self.act_buf.clear();
        self.logp_buf.clear();
        self.val_buf.clear();
        self.rew_buf.clear();
        self.done_buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_smoke_update() {
        // 小 rollout 冒烟：act/feedback 攒满即 update，loss 有限
        let cfg = PpoConfig {
            rollout: 16,
            minibatch: 8,
            epochs: 2,
            ..Default::default()
        };
        let mut agent = PpoAgent::new(cfg, 3);
        let mut obs = Observation { data: [0.2; OBS_DIM] };
        for _ in 0..32 {
            let _a = agent.act(&obs);
            let mut next = obs;
            next.data[0] += 0.01;
            agent.feedback(Feedback {
                next_obs: &next,
                reward: 0.5,
                done: false,
                terminal: false,
            });
            obs = next;
        }
        let m = agent.metrics();
        assert!(m.loss.is_finite(), "loss={}", m.loss);
        assert!(m.extra.is_finite());
        let _snap = agent.snapshot();
    }
}
