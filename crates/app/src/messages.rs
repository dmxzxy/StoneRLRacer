//! 训练线程 ↔ UI 线程的消息协议。

use rl::metrics::{EpisodeSummary, StepMetrics};
use rl::policy::AlgoKind;
use sim::config::SimConfig;
use sim::episode::EpisodeResult;
use sim::observation::N_RAYS;

/// 轨迹上的一个点（世界坐标 + 当时车速，画布按速度着色）。
#[derive(Clone, Copy, Debug)]
pub struct TracePoint {
    pub x: f32,
    pub y: f32,
    pub speed: f32,
}

/// 一局跑完后的行驶轨迹。
pub struct Trace {
    pub points: Vec<TracePoint>,
    /// 本局是否完圈（完圈轨迹更亮）。
    pub lap: bool,
}

/// 一局跑完后的轨迹消息（带局号）。
pub struct EpisodeTrace {
    pub episode: u64,
    pub points: Vec<TracePoint>,
    pub lap: bool,
}

/// 训练环境的一帧镜像（画布直接画训练中的那辆车）。
///
/// 训练线程独占真实环境，这里只是"摄像机"：车的位置/朝向/速度、
/// 当时的观测射线、本局进度，全部来自训练环境本身。
#[derive(Clone, Copy, Debug)]
pub struct TrainFrame {
    pub step: u64,
    pub episode: u64,
    pub ep_step: u32,
    pub ep_reward: f32,
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub speed: f32,
    /// 本局进度 [0,1]。
    pub progress: f32,
    /// 当前局状态。
    pub result: EpisodeResult,
    /// 9 条雷达射线的归一化距离（d / MAX_RAY_DIST）。
    pub rays: [f32; N_RAYS],
}

impl TrainFrame {
    /// 镜像帧是否可用（坐标/朝向/射线全部有限）。
    pub fn all_finite(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.heading.is_finite()
            && self.speed.is_finite()
            && self.rays.iter().all(|r| r.is_finite())
    }
}

/// trainer → UI。
pub enum ToUi {
    StepMetrics(StepMetrics),
    EpisodeDone(EpisodeSummary),
    /// 局末轨迹（与 EpisodeDone 成对出现）。
    EpisodeTrace(EpisodeTrace),
    /// 训练环境实时镜像（约 20–30 Hz）。
    TrainFrame(TrainFrame),
    /// 环境切换已生效（赛道/车型/seed 已重建，当前局作废）。
    EnvApplied {
        seed: u64,
        sim_cfg: SimConfig,
    },
    /// 训练停止时保存了 checkpoint。
    CheckpointSaved(String),
    /// 训练线程退出（Some(err) 表示出错原因）。
    Stopped(Option<String>),
}

/// UI → trainer。
pub enum ToTrainer {
    Pause(bool),
    Stop,
    /// 训练中换环境：训练线程在下一个安全点重建环境，保留策略权重。
    SetEnv {
        seed: u64,
        sim_cfg: SimConfig,
    },
}

/// 续训信息。
pub struct ResumeInfo {
    pub record: burn::store::ModuleRecord,
    /// 已训练步数（计数与 ε 调度接着走）。
    pub steps: u64,
    /// 续训来源的模型文件名——保存新模型后删除它，保持一条血统一个文件。
    pub replace_file: String,
}

/// 一次训练请求（UI spawn 线程时直接传入）。
pub struct TrainRequest {
    pub algo: AlgoKind,
    pub track_seed: u64,
    pub sim_cfg: SimConfig,
    /// 继续训练。
    pub resume: Option<ResumeInfo>,
}
