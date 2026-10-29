//! 游戏状态：赛道/车型/模型选择 + 观赛环境步进（无渲染依赖）。

use bevy::prelude::*;
use rl::policy::Policy;
use crate::train::TrainState;
use sim::action::ContinuousAction;
use sim::car::CarConfig;
use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::track::{RealTrack, Track, TrackGenConfig};

pub enum TrackChoice {
    Random(&'static str),
    Real(RealTrack),
}

pub const TRACKS: [(&str, TrackChoice); 8] = [
    ("随机·简单", TrackChoice::Random("简单")),
    ("随机·中等", TrackChoice::Random("中等")),
    ("随机·困难", TrackChoice::Random("困难")),
    ("蒙扎", TrackChoice::Real(RealTrack::Monza)),
    ("摩纳哥", TrackChoice::Real(RealTrack::Monaco)),
    ("斯帕", TrackChoice::Real(RealTrack::Spa)),
    ("银石", TrackChoice::Real(RealTrack::Silverstone)),
    ("上海", TrackChoice::Real(RealTrack::Shanghai)),
];

pub const CARS: [(&str, [u8; 3], fn() -> CarConfig); 5] = [
    ("均衡", [235, 120, 50], CarConfig::hatch),
    ("跑车", [240, 200, 70], CarConfig::sport),
    ("肌肉", [80, 105, 225], CarConfig::muscle),
    ("卡丁", [85, 200, 120], CarConfig::kart),
    ("方程式", [225, 50, 50], CarConfig::formula),
];

pub const SPEEDS: [u32; 5] = [1, 2, 4, 8, 16];

/// 相机模式：跟随（贴车，可捏合缩放，拖拽即滑入自由）/ 自由（画布）。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum CamMode {
    #[default]
    Follow,
    Free,
}

impl CamMode {
    pub fn label(self) -> &'static str {
        match self {
            CamMode::Follow => "跟随",
            CamMode::Free => "自由",
        }
    }

    pub fn next(self) -> Self {
        match self {
            CamMode::Follow => CamMode::Free,
            CamMode::Free => CamMode::Follow,
        }
    }
}

/// 屏幕：游玩 / 训练（dashboard）。
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Screen {
    #[default]
    Play,
    Train,
}

pub fn track_cfg_pub(choice: &TrackChoice) -> TrackGenConfig {
    track_cfg(choice)
}

fn track_cfg(choice: &TrackChoice) -> TrackGenConfig {
    match choice {
        TrackChoice::Random(d) => match *d {
            "简单" => TrackGenConfig::easy(),
            "困难" => TrackGenConfig::hard(),
            _ => TrackGenConfig::medium(),
        },
        TrackChoice::Real(t) => t.to_track_gen_config(),
    }
}

/// 观赛环境 + 统计。
pub struct SimViewer {
    pub env: SimState,
    pub policy: Option<Policy>,
    pub policy_label: String,
    pub laps: u32,
    pub best_lap: Option<f32>,
    pub last_result: Option<EpisodeResult>,
    pub ep_reward: f32,
    pub ep_steps: u32,
    step_count: u64,
}

impl SimViewer {
    fn new(cfg: SimConfig, seed: u64) -> Self {
        let track = Track::generate(seed, &cfg.track);
        Self {
            env: SimState::new(track, seed, cfg),
            policy: None,
            policy_label: "无策略".into(),
            laps: 0,
            best_lap: None,
            last_result: None,
            ep_reward: 0.0,
            ep_steps: 0,
            step_count: 0,
        }
    }
}

/// 全局游戏状态（Bevy Resource）。
#[derive(Resource)]
pub struct Game {
    pub track_idx: usize,
    pub car_idx: usize,
    pub speed_idx: usize,
    pub models: Vec<crate::model::ModelMeta>,
    pub model_idx: usize,
    pub cam_mode: CamMode,
    pub screen: Screen,
    pub show_rays: bool,
    pub running: bool,
    /// 当前赛道 seed（训练复现同一条赛道）。
    pub seed: u64,
    /// 内置训练状态。
    pub training: TrainState,
    pub viewer: SimViewer,
    /// 供 fx 系统读取的粒子发射请求。
    pub trail_requests: Vec<(bevy::math::Vec2, f32)>,
    /// 帧跳过：偶数步推理，奇数步复用动作（省一半推理开销）。
    last_action: ContinuousAction,
}

impl Default for Game {
    fn default() -> Self {
        let models = crate::model::scan_models();
        let cfg = SimConfig {
            track: track_cfg(&TRACKS[1].1),
            car: CARS[0].2(),
            start_noise: false,
            ..Default::default()
        };
        let mut viewer = SimViewer::new(cfg, 42);
        // 默认尝试载入步数最多的模型
        let mut model_idx = 0;
        if !models.is_empty() {
            model_idx = 0; // models 已按步数降序
            if let Some(p) = crate::model::load_model(&models[model_idx]) {
                viewer.policy = Some(p);
                viewer.policy_label = format!("模型 {}（{} 步）", models[model_idx].algo_label(), crate::model::fmt_steps(models[model_idx].steps));
            }
        }
        Self {
            track_idx: 1,
            car_idx: 0,
            speed_idx: 0,
            models,
            model_idx,
            cam_mode: CamMode::Follow,
            screen: Screen::Play,
            show_rays: true,
            running: true,
            seed: 42,
            training: TrainState::default(),
            viewer,
            trail_requests: Vec::new(),
            last_action: ContinuousAction::ZERO,
        }
    }
}

impl Game {
    /// 切赛道（随机类用新 seed）。
    pub fn regen_track(&mut self) {
        let seed = match &TRACKS[self.track_idx].1 {
            TrackChoice::Random(_) => rand_seed(),
            TrackChoice::Real(_) => 42,
        };
        let cfg = SimConfig {
            track: track_cfg(&TRACKS[self.track_idx].1),
            car: CARS[self.car_idx].2(),
            start_noise: false,
            ..Default::default()
        };
        let keep_policy = self.viewer.policy.take();
        let keep_label = std::mem::take(&mut self.viewer.policy_label);
        let mut viewer = SimViewer::new(cfg, seed);
        viewer.policy = keep_policy;
        viewer.policy_label = keep_label;
        self.viewer = viewer;
        self.seed = seed;
    }

    /// 换车不换赛道。
    pub fn apply_car(&mut self) {
        self.viewer.env.cfg.car = CARS[self.car_idx].2();
        self.viewer.env.reset();
        self.viewer.ep_reward = 0.0;
        self.viewer.ep_steps = 0;
    }

    pub fn track_label(&self) -> &str {
        TRACKS[self.track_idx].0
    }

    pub fn car_label(&self) -> &str {
        CARS[self.car_idx].0
    }

    pub fn model_label(&self) -> String {
        if self.models.is_empty() {
            "无可用模型".into()
        } else {
            let m = &self.models[self.model_idx];
            format!("{}·{}", m.algo_label(), crate::model::fmt_steps(m.steps))
        }
    }

    pub fn speed(&self) -> u32 {
        SPEEDS[self.speed_idx]
    }

    /// 载入当前选中的模型。
    pub fn load_selected_model(&mut self) {
        if self.models.is_empty() {
            return;
        }
        if let Some(p) = crate::model::load_model(&self.models[self.model_idx]) {
            self.viewer.policy = Some(p);
            self.viewer.policy_label = format!(
                "模型 {}（{} 步）",
                self.models[self.model_idx].algo_label(),
                crate::model::fmt_steps(self.models[self.model_idx].steps)
            );
        }
    }

    /// 单步推进（含统计与轨迹粒子请求）。
    /// 每 2 步推理一次（帧跳过），奇数步复用上一动作——推理开销减半。
    pub fn step(&mut self, _commands: &mut Commands) {
        let action = if self.viewer.step_count % 2 == 0 {
            match &self.viewer.policy {
                Some(p) => p.act(&self.viewer.env.observation()),
                None => ContinuousAction::ZERO,
            }
        } else {
            self.last_action
        };
        self.last_action = action;
        let out = self.viewer.env.step(action);
        self.viewer.ep_reward += out.reward;
        self.viewer.ep_steps += 1;
        self.viewer.step_count += 1;

        // 轨迹粒子：每 3 步一发
        if self.viewer.step_count % 5 == 0 {
            let pos = self.viewer.env.car.pos;
            self.trail_requests.push((bevy::math::Vec2::new(pos.x, pos.y), self.viewer.env.car.speed));
        }

        if out.done {
            self.viewer.last_result = Some(out.result);
            if matches!(out.result, EpisodeResult::LapDone) {
                self.viewer.laps += 1;
                let t = self.viewer.env.elapsed;
                self.viewer.best_lap = Some(match self.viewer.best_lap {
                    Some(b) if b <= t => b,
                    _ => t,
                });
            }
            self.viewer.env.reset();
            self.viewer.ep_reward = 0.0;
            self.viewer.ep_steps = 0;
        }
    }
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
        ^ (std::process::id() as u64)
}
