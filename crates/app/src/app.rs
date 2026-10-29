//! eframe 主应用。
//!
//! 结构上把两件事彻底分开：
//!
//! - **训练工作台**（[`Mode::Train`]）：训练线程 + 训练环境镜像 + 训练看板。
//!   这里的赛道/车型/seed 改动会发给训练线程，由它重建环境（当前局作废、策略保留）。
//! - **展示厅**（[`Mode::Demo`]）：只放已训练的模型自己跑，自带一套环境选择。
//!   这里没有任何训练入口，也碰不到训练线程。
//!
//! 两个模式各自持有自己的环境选择与画布状态，互不影响。

use std::sync::mpsc::{sync_channel, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use eframe::egui;
use rl::policy::AlgoKind;

use crate::messages::{ResumeInfo, ToTrainer, ToUi, TrainRequest};
use crate::mirror::Mirror;
use crate::trainer;
use crate::view::{controls, dashboard, live, track_view};
use crate::viewer::Viewer;
use crate::{MODE_DEMO, MODE_TRAIN};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrainState {
    Idle,
    Running,
    Paused,
}

/// 顶层模式：训练 / 展示。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Train,
    Demo,
}

/// 训练工作台的中央视图。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrainTab {
    /// 训练环境实时镜像 + 遥测
    Mirror,
    /// 训练看板
    Board,
}

/// 一套环境选择（训练和展示各有一份，互不影响）。
pub struct EnvSel {
    pub track_sel: controls::TrackSel,
    pub difficulty: controls::Difficulty,
    pub seed: u64,
    pub seed_text: String,
    pub car_kind: controls::CarKind,
    pub car_cfg: sim::car::CarConfig,
}

impl EnvSel {
    pub fn new(kind: controls::CarKind, seed: u64) -> Self {
        Self {
            track_sel: controls::TrackSel::Random,
            difficulty: controls::Difficulty::Easy,
            seed,
            seed_text: seed.to_string(),
            car_kind: kind,
            car_cfg: kind.to_cfg(),
        }
    }

    pub fn sim_config(&self) -> sim::config::SimConfig {
        controls::sim_config_for(self.track_sel, self.difficulty, &self.car_cfg).sanitized()
    }

    /// 展示厅专用：完圈不停车，接着跑下一圈。
    pub fn demo_config(&self) -> sim::config::SimConfig {
        let mut cfg = self.sim_config();
        cfg.continuous_laps = true;
        cfg
    }

    /// 简短描述："随机/简单 seed 42 · 均衡"
    pub fn label(&self) -> String {
        let track = match self.track_sel {
            controls::TrackSel::Random => format!("随机/{}", self.difficulty.label()),
            controls::TrackSel::Real(t) => t.label().to_owned(),
        };
        format!("{track} seed {} · {}", self.seed, self.car_kind.label())
    }
}

pub struct RlRacerApp {
    pub mode: Mode,
    /// 首帧给窗口标题栏上深色（Windows 标题栏跟随系统，默认可能是浅色）。
    theme_sent: bool,
    pub show_rays: bool,

    // ---------------- 训练工作台 ----------------
    pub train_env: EnvSel,
    pub train_tab: TrainTab,
    pub train_algo: AlgoKind,
    trainer_handle: Option<JoinHandle<()>>,
    tx_trainer: Option<Sender<ToTrainer>>,
    rx_ui: Option<Receiver<ToUi>>,
    train_paused: bool,
    pub train_steps: u64,
    pub sps: f32,
    last_metrics_step: u64,
    last_metrics_time: Option<Instant>,
    /// 本次训练使用的奖励口径（看板按它解释奖励分解）。
    pub train_reward: sim::reward::RewardConfig,
    pub metrics: dashboard::Metrics,
    pub mirror: Mirror,
    /// 训练镜像画布的镜头（自由/跟随）。
    pub cam_train: crate::view::cam::CamState,
    /// 已下发环境切换、等训练线程确认。
    pub env_pending: bool,
    pub last_checkpoint: Option<String>,

    // ---------------- 展示厅 ----------------
    pub demo_env: EnvSel,
    pub viewer: Viewer,
    /// 展示厅画布的镜头（自由/跟随）。
    pub cam_demo: crate::view::cam::CamState,
    /// 展示厅当前选中的模型文件名（选中不等于已载入，要按播放）。
    pub demo_model: Option<String>,
    /// 待确认删除的模型名（两步删除防误删，只在训练工作台里）。
    pub pending_delete: Option<String>,
}

impl RlRacerApp {
    pub fn new() -> Self {
        let seed = 42u64;
        let train_env = EnvSel::new(controls::CarKind::Hatch, seed);
        let demo_env = EnvSel::new(controls::CarKind::Hatch, seed);
        let mirror = Mirror::new(seed, &train_env.sim_config());
        let viewer = Viewer::new(seed, &demo_env.demo_config());
        Self {
            mode: Mode::Train,
            theme_sent: false,
            show_rays: true,
            train_env,
            train_tab: TrainTab::Mirror,
            train_algo: AlgoKind::Dqn,
            trainer_handle: None,
            tx_trainer: None,
            rx_ui: None,
            train_paused: false,
            train_steps: 0,
            sps: 0.0,
            last_metrics_step: 0,
            last_metrics_time: None,
            train_reward: sim::reward::RewardConfig::default(),
            metrics: dashboard::Metrics::default(),
            mirror,
            cam_train: crate::view::cam::CamState::default(),
            env_pending: false,
            last_checkpoint: None,
            demo_env,
            viewer,
            cam_demo: crate::view::cam::CamState::default(),
            demo_model: None,
            pending_delete: None,
        }
    }

    /// 演示/自动化用：`RL_RACER_DEMO=1` 启动即开训，
    /// `RL_RACER_ALGO=ppo` 选算法，`RL_RACER_TAB=board` 停在训练看板，
    /// `RL_RACER_MODE=demo` 直接进展示厅。正常启动不受影响。
    pub fn apply_demo_env(&mut self) {
        if std::env::var("RL_RACER_ALGO").as_deref() == Ok("ppo") {
            self.train_algo = AlgoKind::Ppo;
        }
        if std::env::var("RL_RACER_MODE").as_deref() == Ok("demo") {
            self.mode = Mode::Demo;
        }
        // 演示用镜头模式：RL_RACER_CAM=follow 直接进跟随镜头
        if std::env::var("RL_RACER_CAM").as_deref() == Ok("follow") {
            self.cam_demo
                .set_mode(crate::view::cam::CamMode::Follow);
            self.cam_train
                .set_mode(crate::view::cam::CamMode::Follow);
        }
        // 直接打开某个模型播放：RL_RACER_MODEL=dqn-h128-seed7-s200000.rec
        if let Ok(name) = std::env::var("RL_RACER_MODEL") {
            if !name.trim().is_empty() {
                self.mode = Mode::Demo;
                self.demo_model = Some(name.trim().to_owned());
                self.demo_play_selected();
            }
        }
        if std::env::var("RL_RACER_DEMO").is_ok() {
            let seed = self.train_env.seed;
            self.start_training(seed, None);
        }
        if std::env::var("RL_RACER_TAB").as_deref() == Ok("board") {
            self.train_tab = TrainTab::Board;
        }
    }

    pub fn train_state(&self) -> TrainState {
        if self.trainer_handle.is_some() {
            if self.train_paused {
                TrainState::Paused
            } else {
                TrainState::Running
            }
        } else {
            TrainState::Idle
        }
    }

    // ------------------------------------------------------------------
    // 训练
    // ------------------------------------------------------------------

    pub fn start_training(&mut self, seed: u64, resume: Option<ResumeInfo>) {
        if self.trainer_handle.is_some() {
            return;
        }
        let cfg = self.train_env.sim_config();
        self.metrics.reset();
        self.metrics.set_algo(self.train_algo);
        self.train_steps = resume.as_ref().map(|r| r.steps).unwrap_or(0);
        self.sps = 0.0;
        self.last_metrics_step = self.train_steps;
        self.last_metrics_time = None;
        self.train_paused = false;
        self.train_reward = cfg.reward;
        self.mirror = Mirror::new(seed, &cfg);
        self.mirror.cfg.car = cfg.car;
        self.cam_train.reset();
        self.env_pending = false;
        self.mode = Mode::Train;
        self.train_tab = TrainTab::Mirror;

        let request = TrainRequest {
            algo: self.train_algo,
            track_seed: seed,
            sim_cfg: cfg,
            resume,
        };
        let (tx_trainer, rx_trainer) = std::sync::mpsc::channel::<ToTrainer>();
        let (tx_ui, rx_ui) = sync_channel::<ToUi>(1024);
        self.trainer_handle = Some(trainer::spawn(request, rx_trainer, tx_ui));
        self.tx_trainer = Some(tx_trainer);
        self.rx_ui = Some(rx_ui);
    }

    pub fn pause_training(&mut self) {
        if let Some(tx) = &self.tx_trainer {
            let _ = tx.send(ToTrainer::Pause(!self.train_paused));
            self.train_paused = !self.train_paused;
        }
    }

    pub fn stop_training(&mut self) {
        if let Some(tx) = &self.tx_trainer {
            let _ = tx.send(ToTrainer::Stop);
        }
    }

    /// 训练中换环境：交给训练线程在安全点重建（保留策略权重，清空经验池）。
    /// 没在训练时只更新镜像预览。
    pub fn apply_train_env(&mut self) {
        let cfg = self.train_env.sim_config();
        let seed = self.train_env.seed;
        if let Some(tx) = &self.tx_trainer {
            let _ = tx.send(ToTrainer::SetEnv { seed, sim_cfg: cfg });
            self.env_pending = true;
        } else {
            self.mirror = Mirror::new(seed, &cfg);
            self.mirror.cfg.car = cfg.car;
            self.cam_train.reset();
        }
    }

    /// 展示厅播放键的唯一开关：
    /// - 正在跑 → 暂停
    /// - 选了模型 → 载入（已载入同一个就只继续）并开跑
    pub fn demo_toggle_play(&mut self) {
        if self.viewer.running && self.viewer.policy.is_some() {
            self.viewer.running = false;
            return;
        }
        self.demo_play_selected();
    }

    /// 载入当前选中的模型并开跑（已载入同一个就只继续）。双击模型行也走这里。
    pub fn demo_play_selected(&mut self) {
        let Some(name) = self.demo_model.clone() else {
            return;
        };
        if self.viewer.policy_file.as_deref() == Some(name.as_str()) && self.viewer.policy.is_some()
        {
            self.viewer.running = true;
            return;
        }
        let Some(record) = burn::store::ModuleRecord::load(format!("checkpoints/{name}")).ok()
        else {
            // 文件没了（被删/被续训替换）：别留一个指向空气的选中态
            self.demo_model = None;
            return;
        };
        let Some(meta) = controls::parse_meta(&name) else {
            return;
        };
        let snap = rl::policy::PolicySnapshot {
            algo: meta.algo,
            hidden: meta.hidden,
            record,
        };
        self.viewer.policy = Some(rl::policy::Policy::load(snap));
        self.viewer.policy_source = Some(format!(
            "模型 {}（{} 步）",
            meta.algo.label(),
            controls::fmt_steps(meta.steps)
        ));
        self.viewer.policy_file = Some(name);
        self.viewer.clear_history();
        self.viewer.running = true;
    }

    /// 展示厅换环境（赛道 / seed / 难度）：只动展示环境。
    pub fn apply_demo_track(&mut self) {
        let cfg = self.demo_env.demo_config();
        let seed = self.demo_env.seed;
        self.viewer.regenerate(seed, &cfg);
        self.viewer.car_color = self.demo_env.car_kind.color();
        self.cam_demo.reset();
    }

    pub fn apply_demo_car(&mut self) {
        self.viewer.set_car(self.demo_env.car_cfg);
        self.viewer.car_color = self.demo_env.car_kind.color();
    }

    /// 训练中换车（不换赛道）：也走 SetEnv，训练线程重建环境。
    pub fn apply_train_car(&mut self) {
        self.apply_train_env();
    }

    // ------------------------------------------------------------------
    // 消息泵
    // ------------------------------------------------------------------

    fn drain_trainer_messages(&mut self) {
        let mut disconnected = false;
        loop {
            let msg = match &self.rx_ui {
                None => break,
                Some(rx) => match rx.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                },
            };
            match msg {
                ToUi::StepMetrics(m) => {
                    // SPS 由相邻两次指标的步差/时差计算
                    if let Some(t) = self.last_metrics_time {
                        let ds = m.step.saturating_sub(self.last_metrics_step);
                        let dt = t.elapsed().as_secs_f32();
                        if ds > 0 && dt > 0.0 {
                            self.sps = ds as f32 / dt;
                        }
                    }
                    self.last_metrics_step = m.step;
                    self.last_metrics_time = Some(Instant::now());
                    self.train_steps = m.step;
                    self.metrics.push_step(&m, self.sps);
                }
                ToUi::EpisodeDone(summary) => {
                    self.metrics.push_episode(&summary);
                }
                ToUi::EpisodeTrace(trace) => {
                    self.mirror.push_trace(trace);
                }
                ToUi::TrainFrame(frame) => {
                    self.mirror.push_frame(frame);
                }
                ToUi::EnvApplied { seed, sim_cfg } => {
                    self.mirror.set_env(seed, &sim_cfg);
                    self.mirror.cfg.car = sim_cfg.car;
                    self.cam_train.reset();
                    self.env_pending = false;
                    self.metrics
                        .push_marker(format!("换环境 → {}", self.train_env.label()));
                }
                ToUi::CheckpointSaved(path) => {
                    self.last_checkpoint = Some(path);
                }
                ToUi::Stopped(reason) => {
                    if let Some(err) = reason {
                        eprintln!("训练线程异常：{err}");
                    }
                    self.finish_trainer();
                }
            }
        }
        if disconnected {
            self.finish_trainer();
        }
    }

    fn finish_trainer(&mut self) {
        if let Some(h) = self.trainer_handle.take() {
            let _ = h.join();
        }
        self.tx_trainer = None;
        self.rx_ui = None;
        self.train_paused = false;
        self.sps = 0.0;
        self.env_pending = false;
    }
}

// ---------------------------------------------------------------------------
// 每帧逻辑与绘制
// ---------------------------------------------------------------------------

impl eframe::App for RlRacerApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 窗口已存在之后再发一次，保证标题栏跟着深色（create 阶段发的那次可能被丢弃）
        if !self.theme_sent {
            ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(egui::SystemTheme::Dark));
            self.theme_sent = true;
        }
        self.drain_trainer_messages();
        if self.mode == Mode::Demo {
            self.viewer.step_frame();
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let screen_w = ui
            .input(|i| i.viewport().inner_rect.map(|r| r.width()))
            .unwrap_or(1400.0);
        let panel_w = if screen_w < 900.0 { 300.0 } else { 412.0 };
        egui::Panel::right("controls")
            // 面板自带 (8,2) 内边距会把内容挤到 372/260，去掉后才是规范要求的 388/276。
            .frame(egui::Frame::side_top_panel(ui.style()).inner_margin(0))
            .size_range(300.0..=520.0)
            .default_size(panel_w)
            .resizable(true)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    controls::draw(ui, self);
                });
            });

        egui::CentralPanel::default().show(ui, |ui| match self.mode {
            Mode::Train => self.train_center(ui),
            Mode::Demo => self.demo_center(ui),
        });
    }
}

impl RlRacerApp {
    // ------------------------------------------------------------------
    // 训练工作台中央区
    // ------------------------------------------------------------------

    fn train_center(&mut self, ui: &mut egui::Ui) {
        self.center_header(ui);

        match self.train_tab {
            TrainTab::Mirror => {
                let avail = ui.available_size();
                let status_h = 26.0;
                let size = egui::vec2(avail.x, (avail.y - status_h).max(160.0));
                let scene = Self::mirror_scene(
                    &self.mirror,
                    &self.train_env,
                    self.show_rays,
                    self.train_state(),
                );
                let cam = &mut self.cam_train;
                ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
                    track_view::draw(ui, size, &scene, cam, crate::MODE_TRAIN)
                });
                self.mirror_status(ui);
            }
            TrainTab::Board => {
                egui::ScrollArea::vertical()
                    .id_salt("board_scroll")
                    .show(ui, |ui| {
                        dashboard::draw(ui, &self.metrics, &self.train_reward);
                    });
            }
        }
    }

    /// 中央区头部（训练 / 展示共用）：页签或标题 + 状态 + 统计 + 右侧开关。
    ///
    /// 行高 36、竖直居中；下方一条 1px `#2A2D37` 细线 + 8px 留白（取代 `ui.separator()`）。
    fn center_header(&mut self, ui: &mut egui::Ui) {
        let mode = self.mode;
        let accent = match mode {
            Mode::Train => MODE_TRAIN,
            Mode::Demo => MODE_DEMO,
        };
        let w = ui.available_width();
        let (row, _) = ui.allocate_exact_size(egui::vec2(w, 36.0), egui::Sense::hover());
        let cy = row.center().y;

        // ---------------- 右侧：射线开关（展示厅另有倍速） ----------------
        let right_w = match mode {
            Mode::Demo => 264.0,
            Mode::Train => 88.0,
        };
        let right_rect = egui::Rect::from_min_size(
            egui::pos2(row.right() - right_w, row.top()),
            egui::vec2(right_w, row.height()),
        );
        let mut speed = self.viewer.speed_mult;
        {
            let show_rays = &mut self.show_rays;
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(right_rect)
                    .layout(egui::Layout::right_to_left(egui::Align::Center)),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    if mode == Mode::Demo {
                        ui.scope(|ui| {
                            ui.spacing_mut().slider_width = 96.0;
                            ui.add(egui::Slider::new(&mut speed, 1..=16));
                        });
                        ui.label(
                            egui::RichText::new("倍速")
                                .size(11.0)
                                .color(controls::C_DIM),
                        );
                    }
                    ui.checkbox(show_rays, "射线");
                },
            );
        }
        self.viewer.speed_mult = speed;

        // ---------------- 左侧：训练页签 ----------------
        let mut tab_clicked = None;
        if mode == Mode::Train {
            let tab_rect = egui::Rect::from_min_size(
                egui::pos2(row.left(), cy - 14.0),
                egui::vec2(172.0, 28.0),
            );
            let sel = match self.train_tab {
                TrainTab::Mirror => 0,
                TrainTab::Board => 1,
            };
            ui.scope_builder(egui::UiBuilder::new().max_rect(tab_rect), |ui| {
                let items = [
                    controls::SegItem::new("训练镜像"),
                    controls::SegItem::new("训练看板"),
                ];
                tab_clicked = controls::segmented(ui, "train_tab", &items, Some(sel), accent);
            });
        }
        if let Some(i) = tab_clicked {
            self.train_tab = if i == 0 {
                TrainTab::Mirror
            } else {
                TrainTab::Board
            };
        }

        // ---------------- 状态 + 统计 ----------------
        {
            let painter = ui.painter();
            let limit = right_rect.left() - 12.0;
            let mut x = row.left();

            match mode {
                Mode::Train => {
                    x += 172.0 + 12.0;
                    let (state, color, filled) = match self.train_state() {
                        TrainState::Running => ("训练中", controls::C_PLAY, true),
                        TrainState::Paused => ("已暂停", controls::C_WARN, false),
                        TrainState::Idle => ("未训练", controls::C_DIM, false),
                    };
                    x = paint_dot_label(painter, x, cy, state, color, filled, (limit - x).max(0.0))
                        + 12.0;

                    let stats = [
                        (self.train_steps.to_string(), " 步 · "),
                        (format!("{:.0}", self.sps), " SPS · "),
                        (self.metrics.totals.episodes.to_string(), " 局 · 完圈率 "),
                        (format!("{:.0}", self.metrics.lap_rate_50() * 100.0), "%"),
                    ];
                    let max_w = limit - x;
                    if max_w > 80.0 {
                        let job = stat_job(&stats, max_w);
                        let g = painter.layout_job(job);
                        painter.galley(
                            egui::pos2(x, cy - g.size().y * 0.5),
                            g.clone(),
                            controls::C_TEXT,
                        );
                        x += g.size().x + 12.0;
                    }
                    if self.env_pending && limit - x > 40.0 {
                        paint_clipped(
                            painter,
                            x,
                            cy,
                            "· 环境切换中…",
                            egui::FontId::proportional(11.0),
                            controls::C_WARN,
                            limit - x,
                        );
                    }
                }
                Mode::Demo => {
                    // 3×12 模式色竖条 + 13px 主色标题（伪粗体：同 galley 在 x+0.5 再画一次）
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(row.left(), cy - 6.0),
                            egui::vec2(3.0, 12.0),
                        ),
                        1.5,
                        accent,
                    );
                    let g = painter.layout_no_wrap(
                        "展示厅".to_owned(),
                        egui::FontId::proportional(13.0),
                        controls::C_TEXT,
                    );
                    let gw = g.size().x;
                    let pos = egui::pos2(row.left() + 10.0, cy - g.size().y * 0.5);
                    painter.galley(pos, g.clone(), controls::C_TEXT);
                    painter.galley(pos + egui::vec2(0.5, 0.0), g, controls::C_TEXT);
                    x += 10.0 + gw + 12.0;

                    let loaded = self.viewer.policy.is_some();
                    let (state, color, filled) = match (loaded, self.viewer.running) {
                        (true, true) => ("播放中", controls::C_PLAY, true),
                        (true, false) => ("已暂停", MODE_DEMO, false),
                        (false, _) => ("未载入", controls::C_DIM, false),
                    };
                    x = paint_dot_label(painter, x, cy, state, color, filled, (limit - x).max(0.0))
                        + 12.0;

                    let (info, info_color) = match &self.viewer.policy_source {
                        Some(src) => (src.clone(), controls::C_DIM),
                        None => (
                            "未播放：右侧选模型后点绿色播放键".to_owned(),
                            controls::C_WARN,
                        ),
                    };
                    if limit - x > 40.0 {
                        paint_clipped(
                            painter,
                            x,
                            cy,
                            &info,
                            egui::FontId::proportional(11.0),
                            info_color,
                            limit - x,
                        );
                    }
                }
            }
        }

        // ---------------- 细线分隔（1px `#2A2D37`）+ 8px 留白 ----------------
        let full_w = ui.available_width();
        let (rule, _) = ui.allocate_exact_size(egui::vec2(full_w, 1.0), egui::Sense::hover());
        ui.painter().rect_filled(rule, 0.0, controls::C_RULE);
        ui.add_space(8.0);
    }

    /// 训练镜像的 Scene：数据全部来自训练线程推来的帧。
    fn mirror_scene<'a>(
        m: &'a Mirror,
        train_env: &'a EnvSel,
        show_rays: bool,
        train_state: TrainState,
    ) -> track_view::Scene<'a> {
        let f = m.frame;
        let live = m.is_live();
        let (badge, color) = if live {
            ("训练环境 · 实时镜像", egui::Color32::from_rgb(226, 175, 74))
        } else if train_state == TrainState::Paused {
            ("训练已暂停", egui::Color32::from_rgb(220, 185, 95))
        } else if f.is_some() {
            (
                "训练已停止 · 最后一帧",
                egui::Color32::from_rgb(150, 155, 165),
            )
        } else {
            ("等待训练开始", egui::Color32::from_rgb(150, 155, 165))
        };
        let status_line = match f {
            Some(f) => vec![
                ("局", f.episode.to_string()),
                ("第", format!("{} 步", f.ep_step)),
                ("进度", format!("{:.0}%", f.progress * 100.0)),
                ("本局奖励", format!("{:.1}", f.ep_reward)),
            ],
            None => vec![("", "训练线程还没有推出第一帧".to_owned())],
        };
        track_view::Scene {
            track: &m.track,
            car_pos: f.map(|f| sim::math::Vec2::new(f.x, f.y)),
            car_heading: f.map(|f| f.heading).unwrap_or(0.0),
            car_speed: f.map(|f| f.speed).unwrap_or(0.0),
            car_cfg: &m.cfg.car,
            car_color: train_env.car_kind.color(),
            rays: if show_rays {
                f.map(|f| f.rays)
            } else {
                None
            },
            trail: &m.trail,
            trail_dots: true,
            traces: &m.traces,
            badge: (badge, color),
            status_line,
            result: f
                .map(|f| f.result)
                .filter(|r| *r != sim::episode::EpisodeResult::Running),
            speed_kmh: f.map(|f| f.speed * 3.6).unwrap_or(0.0),
            values: None,
        }
    }

    fn mirror_status(&self, ui: &mut egui::Ui) {
        let m = &self.metrics;
        ui.horizontal(|ui| {
            let text = format!(
                "累计 {} 局 · {} 次完圈 · 最佳 {} · 奖励 {:.0}",
                m.totals.episodes,
                m.totals.laps,
                m.best_lap
                    .map(|t| format!("{t:.1}s"))
                    .unwrap_or_else(|| "—".into()),
                m.avg_reward_50(),
            );
            // 数字走 monospace 11，文字部分 11px 弱色。
            ui.label(dim_job(&text, egui::Color32::from_rgb(150, 155, 165)));
            if let Some(path) = &self.last_checkpoint {
                ui.label(dim_job(
                    &format!("已存档 {path}"),
                    egui::Color32::from_rgb(120, 160, 130),
                ));
            }
        });
    }

    // ------------------------------------------------------------------
    // 展示厅中央区
    // ------------------------------------------------------------------

    fn demo_center(&mut self, ui: &mut egui::Ui) {
        self.center_header(ui);

        let avail = ui.available_size();
        let live_h = if self.viewer.live.len() >= 8 {
            132.0
        } else {
            24.0
        };
        let size = egui::vec2(avail.x, (avail.y - live_h).max(160.0));
        let scene = Self::demo_scene(&self.viewer, self.show_rays);
        let cam = &mut self.cam_demo;
        ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
            track_view::draw(ui, size, &scene, cam, crate::MODE_DEMO)
        });
        live::draw(ui, &self.viewer);
    }

    fn demo_scene<'a>(v: &'a Viewer, show_rays: bool) -> track_view::Scene<'a> {
        let (badge, color) = match (&v.policy, v.running) {
            (Some(_), true) => ("展示厅 · 播放中", egui::Color32::from_rgb(120, 170, 220)),
            (Some(_), false) => ("展示厅 · 已暂停", egui::Color32::from_rgb(90, 125, 160)),
            (None, _) => (
                "展示厅 · 未载入模型",
                egui::Color32::from_rgb(150, 155, 165),
            ),
        };
        let rays = if show_rays && v.policy.is_some() {
            let obs = v.env.observation();
            Some(std::array::from_fn(|i| obs.data[i]))
        } else {
            None
        };
        track_view::Scene {
            track: &v.env.track,
            car_pos: Some(v.env.car.pos),
            car_heading: v.env.car.heading,
            car_speed: v.env.car.speed,
            car_cfg: &v.env.cfg.car,
            car_color: v.car_color,
            rays,
            trail: &v.trail,
            trail_dots: false,
            traces: &v.traces,
            badge: (badge, color),
            status_line: vec![
                ("进度", format!("{:.0}%", v.env.progress_frac() * 100.0)),
                ("圈数", v.laps.to_string()),
                (
                    "最佳",
                    v.best_lap
                        .map(|t| format!("{t:.1}s"))
                        .unwrap_or_else(|| "—".into()),
                ),
                ("本圈奖励", format!("{:.0}", v.ep_reward)),
            ],
            result: v.shown_result(),
            speed_kmh: v.env.car.speed * 3.6,
            values: if v.policy.is_some() {
                Some(&v.last_values)
            } else {
                None
            },
        }
    }
}

/// 单行文字（超宽省略），竖直居中于 `cy`，返回文字右侧 x。
fn paint_clipped(
    painter: &egui::Painter,
    x: f32,
    cy: f32,
    text: &str,
    font: egui::FontId,
    color: egui::Color32,
    max_w: f32,
) -> f32 {
    let mut job = egui::text::LayoutJob::simple(text.to_owned(), font, color, max_w.max(0.0));
    job.wrap = egui::text::TextWrapping {
        max_width: max_w.max(0.0),
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('…'),
    };
    let g = painter.layout_job(job);
    let w = g.size().x;
    painter.galley(egui::pos2(x, cy - g.size().y * 0.5), g, color);
    x + w
}

/// 6px 状态圆点 + 12px 文字，返回文字右侧 x。`filled = false` 时画空心圆点。
fn paint_dot_label(
    painter: &egui::Painter,
    x: f32,
    cy: f32,
    text: &str,
    color: egui::Color32,
    filled: bool,
    max_w: f32,
) -> f32 {
    let c = egui::pos2(x + 3.0, cy);
    if filled {
        painter.circle_filled(c, 3.0, color);
    } else {
        painter.circle_stroke(c, 2.6, egui::Stroke::new(1.0, color));
    }
    paint_clipped(
        painter,
        x + 12.0,
        cy,
        text,
        egui::FontId::proportional(12.0),
        color,
        max_w,
    )
}

/// 数值行：数值 12px monospace 主色，单位/说明 11px 弱色，单行不换行。
fn stat_job(parts: &[(String, &str)], max_w: f32) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    for (value, unit) in parts {
        job.append(
            value,
            0.0,
            egui::TextFormat {
                font_id: egui::FontId::monospace(12.0),
                color: controls::C_TEXT,
                ..Default::default()
            },
        );
        if !unit.is_empty() {
            job.append(
                unit,
                0.0,
                egui::TextFormat {
                    font_id: egui::FontId::proportional(11.0),
                    color: controls::C_DIM,
                    ..Default::default()
                },
            );
        }
    }
    job.wrap = egui::text::TextWrapping {
        max_width: max_w.max(0.0),
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('…'),
    };
    job
}

/// 说明行：数字用 monospace 11，其余 proportional 11（单色，自动折行）。
pub(crate) fn dim_job(text: &str, color: egui::Color32) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let mut buf = String::new();
    let mut buf_is_num = false;
    let flush = |job: &mut egui::text::LayoutJob, buf: &mut String, is_num: bool| {
        if buf.is_empty() {
            return;
        }
        let font_id = if is_num {
            egui::FontId::monospace(11.0)
        } else {
            egui::FontId::proportional(11.0)
        };
        job.append(
            buf.as_str(),
            0.0,
            egui::TextFormat {
                font_id,
                color,
                ..Default::default()
            },
        );
        buf.clear();
    };
    for ch in text.chars() {
        let is_num = ch.is_ascii_digit() || ch == '.';
        if buf_is_num != is_num && !buf.is_empty() {
            flush(&mut job, &mut buf, buf_is_num);
        }
        buf_is_num = is_num;
        buf.push(ch);
    }
    flush(&mut job, &mut buf, buf_is_num);
    job
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::controls::{CarKind, TrackSel};
    use sim::track::RealTrack;

    /// 训练中换赛道 + 换车型：UI 侧全链路（下发 → 训练线程重建 → 镜像换赛道 → 看板标记）。
    /// 用 release 跑（`cargo test --release -p rl-racer-app`）。
    #[test]
    fn switch_env_while_training() {
        let mut app = RlRacerApp::new();
        app.train_env.seed = 42;
        app.start_training(42, None);

        let wait = |app: &mut RlRacerApp, cond: &mut dyn FnMut(&RlRacerApp) -> bool| {
            for _ in 0..800 {
                app.drain_trainer_messages();
                if cond(app) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            false
        };

        assert!(
            wait(&mut app, &mut |a| a.mirror.frame.is_some()),
            "训练镜像应收到第一帧"
        );
        let old_len = app.mirror.track.length;

        // 换成真实赛道 + 跑车：训练中下发
        app.train_env.track_sel = TrackSel::Real(RealTrack::Monza);
        app.train_env.car_kind = CarKind::Sport;
        app.train_env.car_cfg = CarKind::Sport.to_cfg();
        app.apply_train_env();
        assert!(app.env_pending, "应进入等待环境重建状态");

        assert!(
            wait(&mut app, &mut |a| !a.env_pending),
            "训练线程应回报环境已切换"
        );
        // 镜像赛道换成新赛道（长度不同）、车型换成跑车
        assert!(
            (app.mirror.track.length - old_len).abs() > 1.0,
            "镜像赛道应换成蒙扎：{} → {}",
            old_len,
            app.mirror.track.length
        );
        assert!(
            (app.mirror.cfg.car.max_speed - CarKind::Sport.to_cfg().max_speed).abs() < 1e-3,
            "镜像车型应换成跑车"
        );
        // 看板留下换环境标记
        assert!(
            app.metrics
                .milestones
                .iter()
                .any(|m| m.text.contains("换环境")),
            "看板应记录换环境里程碑"
        );
        // 训练还在继续（用镜像帧的步数判断：指标每 2000 步才发一次）
        let before = app.mirror.frame.map(|f| f.step).unwrap_or(0);
        std::thread::sleep(Duration::from_millis(200));
        app.drain_trainer_messages();
        let after = app.mirror.frame.map(|f| f.step).unwrap_or(0);
        assert!(after > before, "换环境后训练应继续推进：{before} → {after}");

        // 暂停中也要能换环境 / 停止（命令不能被"步数是 64 的整数倍"卡住）
        app.pause_training();
        std::thread::sleep(Duration::from_millis(60));
        app.train_env.seed = 99;
        app.apply_train_env();
        assert!(
            wait(&mut app, &mut |a| !a.env_pending),
            "暂停中也应能完成环境切换"
        );
        let paused_steps = app.mirror.frame.map(|f| f.step).unwrap_or(0);
        std::thread::sleep(Duration::from_millis(150));
        app.drain_trainer_messages();
        assert_eq!(
            app.mirror.frame.map(|f| f.step).unwrap_or(0),
            paused_steps,
            "暂停后步数不应推进"
        );

        app.stop_training();
        assert!(
            wait(&mut app, &mut |a| a.train_state() == TrainState::Idle),
            "暂停中也应能停止"
        );
    }
}

#[cfg(test)]
mod demo_tests {
    use super::*;
    use crate::view::controls;

    /// 展示厅播放键：没选模型 → 不动；选了 → 载入并开跑；再按 → 暂停；再按 → 继续。
    #[test]
    fn demo_play_button_states() {
        let mut app = RlRacerApp::new();
        // 至少得有一个存档才能测；没有就跳过（CI 上可能没有 checkpoints/）
        let models = controls::scan_models();
        let Some(first) = models.first() else {
            eprintln!("checkpoints/ 里没有模型，跳过");
            return;
        };

        // 1) 没选模型：按播放键什么都不该发生
        app.demo_model = None;
        app.demo_toggle_play();
        assert!(app.viewer.policy.is_none(), "没选模型不应载入策略");
        assert!(!app.viewer.running, "没选模型不应开始跑");

        // 2) 选中 → 播放：载入并开跑
        app.demo_model = Some(first.name.clone());
        app.demo_toggle_play();
        assert!(app.viewer.policy.is_some(), "选中后应载入策略");
        assert!(app.viewer.running, "选中后应开始跑");
        assert_eq!(
            app.viewer.policy_file.as_deref(),
            Some(first.name.as_str()),
            "应记住载入的是哪个文件"
        );

        // 3) 再按 → 暂停（策略仍在，不重新载入）
        app.demo_toggle_play();
        assert!(!app.viewer.running, "第二次按应是暂停");
        assert!(app.viewer.policy.is_some(), "暂停不应卸载策略");

        // 4) 再按 → 继续
        app.demo_toggle_play();
        assert!(app.viewer.running, "第三次按应是继续");
    }
}
