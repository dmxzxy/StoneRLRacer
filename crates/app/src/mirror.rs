//! 训练镜像：把训练线程里的真实环境"投影"到画布上。
//!
//! 训练环境只有一个，住在训练线程里；这里只保存**最近一帧**（车的位置、
//! 朝向、车速、当时的 9 条观测射线、本局进度），以及从帧序列串起来的拖影。
//! 所以画布上的车不是"另一个环境里的演示车"，就是训练中的那辆车。

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use sim::config::SimConfig;
use sim::math::Vec2;
use sim::track::Track;

use crate::messages::{Trace, TrainFrame};

/// 超过这个时长没有新帧，就认为训练没在跑（暂停/停止）。
const STALE_AFTER: Duration = Duration::from_millis(1500);
/// 拖影保留的点数。
const TRAIL_CAP: usize = 260;
/// 局末轨迹叠加保留的局数。
const TRACE_CAP: usize = 10;

pub struct Mirror {
    /// 与训练线程同 seed / 同配置生成的赛道（确定性，两边一致）。
    pub track: Track,
    pub seed: u64,
    pub cfg: SimConfig,
    pub frame: Option<TrainFrame>,
    last_frame_at: Option<Instant>,
    /// 拖影（由帧序列串起来，不是训练线程推的）。
    pub trail: Vec<Vec2>,
    /// 训练中跑完的局末轨迹。
    pub traces: VecDeque<Trace>,
}

impl Mirror {
    pub fn new(seed: u64, cfg: &SimConfig) -> Self {
        let mut cfg = cfg.clone().sanitized();
        cfg.start_noise = false; // 只用于渲染，与训练线程的赛道几何无关
        let track = Track::generate(seed, &cfg.track);
        Self {
            track,
            seed,
            cfg,
            frame: None,
            last_frame_at: None,
            trail: Vec::new(),
            traces: VecDeque::new(),
        }
    }

    /// 环境切换后重建赛道（丢弃旧帧与拖影）。
    pub fn set_env(&mut self, seed: u64, cfg: &SimConfig) {
        *self = Self::new(seed, cfg);
    }

    pub fn clear_history(&mut self) {
        self.trail.clear();
        self.traces.clear();
        self.frame = None;
        self.last_frame_at = None;
    }

    /// 收一帧镜像；坐标非有限的帧直接丢弃，避免画布画到 NaN。
    pub fn push_frame(&mut self, f: TrainFrame) {
        if !f.all_finite() {
            return;
        }
        self.trail.push(Vec2::new(f.x, f.y));
        if self.trail.len() > TRAIL_CAP {
            self.trail.remove(0);
        }
        self.frame = Some(f);
        self.last_frame_at = Some(Instant::now());
    }

    pub fn push_trace(&mut self, trace: crate::messages::EpisodeTrace) {
        if trace.points.len() < 2 {
            return;
        }
        self.traces.push_back(Trace {
            points: trace.points,
            lap: trace.lap,
        });
        while self.traces.len() > TRACE_CAP {
            self.traces.pop_front();
        }
    }

    /// 镜像是否新鲜（训练确实在跑）。
    pub fn is_live(&self) -> bool {
        self.frame.is_some()
            && self
                .last_frame_at
                .map(|t| t.elapsed() < STALE_AFTER)
                .unwrap_or(false)
    }

    pub fn car_half_width(&self) -> f32 {
        self.cfg.car.half_width
    }
}
