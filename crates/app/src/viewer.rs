//! 展示厅的观赛环境：把**已训练的模型**放进来自己跑，与训练线程完全无关。
//!
//! 这里的状态不会被训练影响，也不会影响训练；训练中改赛道/车型碰不到它。

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use rl::policy::{ActionValues, Policy};

use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::track::Track;

use crate::messages::{Trace, TracePoint};

/// 最近一帧的驾驶遥测（画布下方实时曲线）。
#[derive(Clone, Copy, Debug)]
pub struct LiveSample {
    /// 车速 km/h
    pub speed: f32,
    /// 横向偏移比例（车宽单位，±1 = 贴边）
    pub lateral: f32,
    pub steer: f32,
    pub throttle: f32,
}

/// 实时曲线保留的点数（DT=0.05 → 300 点约 15 秒驾驶）。
const LIVE_CAP: usize = 300;

pub struct Viewer {
    pub env: SimState,
    pub running: bool,
    /// 每帧步进数（倍速）。
    pub speed_mult: u32,
    pub policy: Option<Policy>,
    /// 当前策略的来源说明（存档名）。
    pub policy_source: Option<String>,
    /// 当前载入的模型文件名（判断"选中的模型是否已在跑"）。
    pub policy_file: Option<String>,
    /// 车身颜色（RGB），由车型选择设定。
    pub car_color: [u8; 3],
    /// 最近的行车轨迹（世界坐标，画拖影用）。
    pub trail: Vec<sim::math::Vec2>,
    pub laps: u32,
    pub best_lap: Option<f32>,
    /// 最近一次完圈/撞墙的结果，只在短时间内显示（连续跑圈不能一直挂"完圈"）。
    result_flash: Option<(EpisodeResult, Instant)>,
    pub ep_reward: f32,
    pub ep_steps: u32,
    /// 当前状态下策略对各动作的估值（画布角标展示）。
    pub last_values: ActionValues,
    /// 最近一小段驾驶遥测。
    pub live: VecDeque<LiveSample>,
    /// 本局最高车速（m/s）。
    pub ep_max_speed: f32,
    /// 本次载入模型后跑出的所有完整圈轨迹（越新越亮）。
    pub traces: VecDeque<Trace>,
    /// 正在跑的这一圈的轨迹点。
    lap_trace: Vec<TracePoint>,
}

impl Viewer {
    pub fn new(seed: u64, cfg: &SimConfig) -> Self {
        let mut cfg = cfg.clone().sanitized();
        cfg.start_noise = false;
        let track = Track::generate(seed, &cfg.track);
        Self {
            env: SimState::new(track, seed, cfg),
            // 载入模型前不跑：running 表示"正在播放"
            running: false,
            speed_mult: 1,
            policy: None,
            policy_source: None,
            policy_file: None,
            car_color: [230, 80, 80],
            trail: Vec::new(),
            laps: 0,
            best_lap: None,
            result_flash: None,
            ep_reward: 0.0,
            ep_steps: 0,
            last_values: ActionValues::default(),
            live: VecDeque::new(),
            ep_max_speed: 0.0,
            traces: VecDeque::new(),
            lap_trace: Vec::new(),
        }
    }

    /// 换赛道（同 seed 确定性生成）。保留已加载的策略——换赛道后可以看
    /// 同一策略在新赛道上的表现（泛化实验）。
    pub fn regenerate(&mut self, seed: u64, cfg: &SimConfig) {
        let policy = self.policy.take();
        let policy_source = self.policy_source.take();
        let policy_file = self.policy_file.take();
        let running = self.running;
        let speed_mult = self.speed_mult;
        let car_color = self.car_color;
        *self = Self::new(seed, cfg);
        self.policy = policy;
        self.policy_source = policy_source;
        self.policy_file = policy_file;
        self.running = running;
        self.speed_mult = speed_mult;
        self.car_color = car_color;
    }

    /// 换车不换赛道：只替换车辆物理参数并复位。
    pub fn set_car(&mut self, car: sim::car::CarConfig) {
        self.env.cfg.car = car.sanitized();
        self.reset_stats();
    }

    pub fn reset_stats(&mut self) {
        self.laps = 0;
        self.best_lap = None;
        self.result_flash = None;
        self.env.reset();
        self.ep_reward = 0.0;
        self.ep_steps = 0;
        self.ep_max_speed = 0.0;
        self.traces.clear();
        self.live.clear();
        self.trail.clear();
        self.lap_trace.clear();
    }

    /// 画布右上/左上的结果徽标：只显示 2.5 秒，然后自己消失。
    pub fn shown_result(&self) -> Option<EpisodeResult> {
        match &self.result_flash {
            Some((r, at)) if at.elapsed() < Duration::from_millis(2500) => Some(*r),
            _ => None,
        }
    }

    /// 换模型时清掉旧轨迹（否则不同模型的圈会混在一起）。
    pub fn clear_history(&mut self) {
        self.traces.clear();
        self.trail.clear();
        self.lap_trace.clear();
        self.reset_stats();
    }

    /// 每帧调用：用当前策略（无策略则静止）步进观赛环境。
    pub fn step_frame(&mut self) {
        // 没载入模型就不跑：车停在起点，不产生"站着被罚时间"的负奖励
        if !self.running || self.policy.is_none() {
            return;
        }
        for _ in 0..self.speed_mult {
            let Some(policy) = &self.policy else {
                return;
            };
            let action = policy.act(&self.env.observation());
            let out = self.env.step(action);
            self.ep_reward += out.reward;
            self.ep_steps += 1;
            self.ep_max_speed = self.ep_max_speed.max(self.env.car.speed);
            // 行车轨迹拖影（上限 ~3 秒）
            self.trail.push(self.env.car.pos);
            let cap = (60 * self.speed_mult.max(1)) as usize;
            if self.trail.len() > cap.min(600) {
                self.trail.remove(0);
            }
            // 本圈轨迹（画完一圈后存档用于叠加对比）
            if self.ep_steps % 8 == 0 {
                let p = self.env.car.pos;
                if p.x.is_finite() && p.y.is_finite() {
                    self.lap_trace.push(TracePoint {
                        x: p.x,
                        y: p.y,
                        speed: self.env.car.speed,
                    });
                }
            }
            // 实时遥测
            let loc = self.env.locate();
            self.live.push_back(LiveSample {
                speed: self.env.car.speed * 3.6,
                lateral: (loc.lateral / loc.half_width).clamp(-2.0, 2.0),
                steer: action.steer,
                throttle: action.throttle,
            });
            while self.live.len() > LIVE_CAP {
                self.live.pop_front();
            }
            // 完圈：连续跑圈模式下仿真自己回绕，这里只记账（不休车、不回起点）
            if matches!(out.result, EpisodeResult::LapDone) {
                self.laps += 1;
                if out.lap_time > 0.0 {
                    self.best_lap =
                        Some(self.best_lap.map_or(out.lap_time, |b| b.min(out.lap_time)));
                }
                if self.lap_trace.len() >= 2 {
                    self.traces.push_back(Trace {
                        points: std::mem::take(&mut self.lap_trace),
                        lap: true,
                    });
                    while self.traces.len() > 8 {
                        self.traces.pop_front();
                    }
                }
                self.lap_trace.clear();
                self.ep_reward = 0.0;
                self.ep_steps = 0;
                self.ep_max_speed = 0.0;
                self.result_flash = Some((EpisodeResult::LapDone, Instant::now()));
            }

            // 撞墙 / 超时：这个确实得回起点
            if out.done {
                self.result_flash = Some((out.result, Instant::now()));
                self.lap_trace.clear();
                self.env.reset();
                self.ep_reward = 0.0;
                self.ep_steps = 0;
                self.ep_max_speed = 0.0;
            }
        }
        // 每帧解释一次当前策略（前向量级极小，不拖帧）
        if let Some(p) = &self.policy {
            self.last_values = p.action_values(&self.env.observation());
        } else {
            self.last_values = ActionValues::default();
        }
    }
}
