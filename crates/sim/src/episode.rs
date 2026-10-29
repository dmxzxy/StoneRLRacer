//! 环境状态机：reset / step / 终止判定 / 进度与奖励结算。

use crate::action::ContinuousAction;
use crate::car::{self, CarState};
use crate::config::SimConfig;
use crate::math::Vec2;
use crate::observation::Observation;
use crate::reward::{step_reward_parts, RewardParts};
use crate::rng::Rng;
use crate::track::{Track, TrackLocate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeResult {
    Running,
    Crashed,
    LapDone,
    Timeout,
}

impl EpisodeResult {
    pub fn label(self) -> &'static str {
        match self {
            EpisodeResult::Running => "进行中",
            EpisodeResult::Crashed => "撞墙",
            EpisodeResult::LapDone => "完圈",
            EpisodeResult::Timeout => "超时",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepOutcome {
    pub obs: Observation,
    /// 标量奖励（= `parts.total()`）。
    pub reward: f32,
    /// 奖励分解（训练看板逐项展示）。
    pub parts: RewardParts,
    pub done: bool,
    pub result: EpisodeResult,
    /// 完圈时的本圈用时（秒）；其它情况为 0。
    ///
    /// 连续跑圈模式下 `elapsed` 会就地回绕，所以圈时必须在这里带出来。
    pub lap_time: f32,
}

pub struct SimState {
    pub track: Track,
    pub car: CarState,
    pub cfg: SimConfig,
    rng: Rng,
    pub step_count: u32,
    /// 带符号累计弧长进度（米）。≥ track.length 即完圈。
    pub total_progress: f32,
    pub elapsed: f32,
    last_action: ContinuousAction,
    hint: Option<usize>,
    pub result: EpisodeResult,
}

impl SimState {
    pub fn new(track: Track, seed: u64, cfg: SimConfig) -> Self {
        let mut s = Self {
            track,
            car: CarState {
                pos: Vec2::ZERO,
                heading: 0.0,
                speed: 0.0,
            },
            cfg,
            rng: Rng::new(seed),
            step_count: 0,
            total_progress: 0.0,
            elapsed: 0.0,
            last_action: ContinuousAction::ZERO,
            hint: None,
            result: EpisodeResult::Running,
        };
        s.reset();
        s
    }

    /// 复位到起点（samples[0]，朝向切线）。
    pub fn reset(&mut self) {
        let s0 = self.track.samples[0];
        let lateral = if self.cfg.start_noise {
            // 起步随机横向偏移 ±3m，避免策略只会从精确中心线起步
            self.rng.next_range(-3.0, 3.0)
        } else {
            0.0
        };
        self.car.pos = s0.pos + s0.normal * lateral;
        self.car.heading = s0.tangent.angle();
        self.car.speed = 0.0;
        self.step_count = 0;
        self.total_progress = 0.0;
        self.elapsed = 0.0;
        self.last_action = ContinuousAction::ZERO;
        self.hint = None;
        self.result = EpisodeResult::Running;
    }

    pub fn observation(&self) -> Observation {
        let loc = self.track.locate(self.car.pos, self.hint);
        Observation::build(&self.track, &self.car, &self.cfg.car, &loc, self.last_action)
    }

    /// 车在赛道上的定位（渲染/调试用）。
    pub fn locate(&self) -> TrackLocate {
        self.track.locate(self.car.pos, self.hint)
    }

    pub fn step(&mut self, action: ContinuousAction) -> StepOutcome {
        let loc_before = self.track.locate(self.car.pos, self.hint);
        car::integrate(&mut self.car, action.steer, action.throttle, &self.cfg.car, crate::DT);
        let loc_after = self.track.locate(self.car.pos, Some(loc_before.sample_idx));
        self.hint = Some(loc_after.sample_idx);

        // 带符号弧长差（处理 s 回绕）
        let len = self.track.length;
        let mut ds = loc_after.s - loc_before.s;
        if ds > 0.5 * len {
            ds -= len;
        } else if ds < -0.5 * len {
            ds += len;
        }
        self.total_progress += ds;
        self.step_count += 1;
        self.elapsed += crate::DT;

        // 每步奖励
        let hw = loc_after.half_width;
        let speed_frac = (self.car.speed / self.cfg.car.max_speed).clamp(-1.0, 1.0);
        let lateral_frac = (loc_after.lateral / hw).clamp(-2.0, 2.0);
        let mut parts = step_reward_parts(&self.cfg.reward, ds, speed_frac, lateral_frac);

        // 终止判定
        let crashed = loc_after.lateral.abs() > hw - self.cfg.car.half_width;
        if crashed {
            parts.crash = -self.cfg.reward.r_crash;
            self.result = EpisodeResult::Crashed;
        } else if self.total_progress >= len {
            parts.lap = self.cfg.reward.r_lap;
            self.result = EpisodeResult::LapDone;
        } else if self.step_count >= self.cfg.max_steps {
            self.result = EpisodeResult::Timeout;
        }
        let mut done = !matches!(self.result, EpisodeResult::Running);
        let result = self.result;
        let lap_time = if result == EpisodeResult::LapDone {
            self.elapsed
        } else {
            0.0
        };
        // 连续跑圈：完圈不停车，只把圈计时与里程回绕，车继续开
        if result == EpisodeResult::LapDone && self.cfg.continuous_laps {
            self.rollover_lap();
            done = false;
        }

        self.last_action = action;
        let obs = Observation::build(&self.track, &self.car, &self.cfg.car, &loc_after, action);
        StepOutcome {
            obs,
            reward: parts.total(),
            parts,
            done,
            result,
            lap_time,
        }
    }

    /// 完圈回绕：车不动，圈里程/计时/步数清零，状态回到 Running。
    /// 只保留超出终点线的部分作为新一圈的进度（不会丢里程）。
    pub fn rollover_lap(&mut self) {
        let len = self.track.length;
        if self.total_progress >= len {
            self.total_progress -= len;
        }
        self.step_count = 0;
        self.elapsed = 0.0;
        self.result = EpisodeResult::Running;
    }

    /// 完成度 [0, 1]（总进度/一圈）。
    pub fn progress_frac(&self) -> f32 {
        (self.total_progress / self.track.length).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;
    use crate::math::Vec2;
    use crate::track::{Track, TrackGenConfig};

    fn circle_env() -> SimState {
        let cfg = SimConfig {
            track: TrackGenConfig {
                radius_noise: 0.0,
                angular_jitter: 0.0,
                ..Default::default()
            },
            max_steps: 5000,
            start_noise: false,
            ..Default::default()
        };
        let track = Track::generate(0, &cfg.track);
        SimState::new(track, 0, cfg)
    }

    #[test]
    fn full_throttle_reward_is_positive_at_start() {
        let mut env = circle_env();
        let mut total = 0.0;
        for _ in 0..30 {
            let out = env.step(ContinuousAction { steer: 0.0, throttle: 1.0 });
            total += out.reward;
            assert!(!out.done);
        }
        assert!(total > 0.0, "total={total}");
        assert!(env.car.speed > 5.0, "speed={}", env.car.speed);
    }

    #[test]
    fn off_track_crashes() {
        let mut env = circle_env();
        // 把车挪出赛道边界再 step
        env.car.pos = env.car.pos + env.track.samples[0].normal * (env.track.width * 0.5 + 5.0);
        let out = env.step(ContinuousAction::ZERO);
        assert!(out.done);
        assert_eq!(out.result, EpisodeResult::Crashed);
        assert!(out.reward < -4.0, "reward={}", out.reward);
    }

    #[test]
    fn timeout_terminates() {
        let cfg = SimConfig {
            max_steps: 10,
            ..Default::default()
        };
        let track = Track::generate(1, &cfg.track);
        let mut env = SimState::new(track, 1, cfg);
        for _ in 0..10 {
            let out = env.step(ContinuousAction::ZERO);
            if out.done {
                assert_eq!(out.result, EpisodeResult::Timeout);
                return;
            }
        }
        panic!("应在 max_steps 步后超时");
    }

    #[test]
    fn perfect_circle_driver_completes_lap() {
        // 正圆赛道 + 精确圆轨迹驾驶 → 一圈后 LapDone
        let mut env = circle_env();
        let r = env.track.samples[0].pos.len(); // 实际中心线半径 ≈ 130
        let v = 8.0f32;
        // 转弯半径 R = v / ω → ω = v / R
        let omega = v / r;
        let steps_needed = (std::f32::consts::TAU * r / v / crate::DT) as u32 + 100;
        let mut done_result = None;
        for _ in 0..steps_needed {
            // 圆周驾驶：车头沿切线前进。直接设定速度并按 ω 转向。
            env.car.speed = v;
            let steer_needed = (omega * env.cfg.car.wheelbase / v).atan()
                / env.cfg.car.max_steer;
            let out = env.step(ContinuousAction { steer: steer_needed.clamp(-1.0, 1.0), throttle: 0.35 });
            env.car.speed = v; // 防止油门补偿漂移，速度钉在 v
            if out.done {
                done_result = Some(out.result);
                break;
            }
        }
        assert_eq!(done_result, Some(EpisodeResult::LapDone), "一圈内应完圈");
    }

    #[test]
    fn continuous_laps_do_not_stop_the_car() {
        let mut env = circle_env();
        env.cfg.continuous_laps = true;
        env.car.speed = 10.0;
        // 先把进度推到差一点点就过线
        env.total_progress = env.track.length - 0.2;
        let out = env.step(ContinuousAction { steer: 0.0, throttle: 1.0 });
        assert_eq!(out.result, EpisodeResult::LapDone, "应记一次完圈");
        assert!(!out.done, "连续模式下不算结束");
        assert!(env.total_progress < env.track.length * 0.5, "里程应回绕到新一圈");
        assert_eq!(env.result, EpisodeResult::Running);
        // 继续开不会卡在 LapDone 状态
        let out2 = env.step(ContinuousAction { steer: 0.0, throttle: 1.0 });
        assert!(!out2.done);
        assert!(env.car.speed > 0.0);
    }

    #[test]
    fn lap_still_terminates_without_continuous_mode() {
        let mut env = circle_env();
        env.total_progress = env.track.length - 0.2;
        env.car.speed = 10.0;
        let out = env.step(ContinuousAction { steer: 0.0, throttle: 1.0 });
        assert_eq!(out.result, EpisodeResult::LapDone);
        assert!(out.done, "训练模式下完圈即结束一局");
    }

    #[test]
    fn reset_is_deterministic_without_noise() {
        let mut env = circle_env();
        let p1 = env.car.pos;
        env.step(ContinuousAction { steer: 0.5, throttle: 1.0 });
        env.reset();
        let p2 = env.car.pos;
        assert!((p1 - p2).len() < 1e-5);
        assert_eq!(env.result, EpisodeResult::Running);
        assert_eq!(env.step_count, 0);
    }

    #[test]
    fn progress_accumulates_signed() {
        let mut env = circle_env();
        // 前进几步
        for _ in 0..50 {
            env.car.speed = 10.0;
            env.step(ContinuousAction { steer: 0.0, throttle: 0.4 });
            env.car.speed = 10.0;
        }
        let fwd = env.total_progress;
        assert!(fwd > 20.0, "fwd={fwd}");
        // 倒车应减少进度
        for _ in 0..50 {
            env.car.speed = -6.0;
            env.step(ContinuousAction { steer: 0.0, throttle: -0.5 });
            env.car.speed = -6.0;
        }
        assert!(env.total_progress < fwd, "total={}", env.total_progress);
    }

    #[test]
    fn vec2_available() {
        // 占位：确保 math 类型在本模块可见
        let _ = Vec2::ZERO;
    }
}
