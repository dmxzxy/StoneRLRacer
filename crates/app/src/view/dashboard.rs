//! 训练看板：关键指标卡片 + 曲线组。
//!
//! 设计原则：先给结论（卡片大数字），再给过程（曲线），最后给"为什么"（奖励分解）。
//! 所有曲线在 2000 点时自动隔点降采样，长时间训练也不会卡。

use std::collections::VecDeque;

use egui::{Color32, RichText, Stroke};
use egui_plot::{HLine, Line, LineStyle, Plot, PlotPoints, VLine};

use rl::metrics::{EpisodeSummary, StepMetrics};
use rl::policy::AlgoKind;
use sim::episode::EpisodeResult;
use sim::reward::RewardConfig;

// ---- 配色（与赛道画布同一套低饱和色）----
const C_TEXT: Color32 = Color32::from_rgb(220, 223, 230);
const C_DIM: Color32 = Color32::from_rgb(140, 145, 155);
const C_ACCENT: Color32 = Color32::from_rgb(226, 175, 74);
const C_PROGRESS: Color32 = Color32::from_rgb(96, 190, 140);
const C_SPEED: Color32 = Color32::from_rgb(226, 175, 74);
const C_CENTER: Color32 = Color32::from_rgb(92, 140, 200);
const C_TIME: Color32 = Color32::from_rgb(146, 122, 198);
const C_CRASH: Color32 = Color32::from_rgb(206, 92, 84);
const C_LAP: Color32 = Color32::from_rgb(120, 200, 120);
const C_CARD_BG: Color32 = Color32::from_rgb(30, 32, 39);

/// 指标序列（带 2000 点上限的自动降采样）。
#[derive(Default, Clone)]
pub struct Series {
    pub points: Vec<[f64; 2]>,
}

impl Series {
    pub fn push(&mut self, x: f64, y: f64) {
        // 跳过 NaN/Inf，避免画板炸
        if !y.is_finite() || !x.is_finite() {
            return;
        }
        self.points.push([x, y]);
        if self.points.len() > 2000 {
            let decimated: Vec<[f64; 2]> = self.points.iter().step_by(2).copied().collect();
            self.points = decimated;
        }
    }

    pub fn last_y(&self) -> Option<f64> {
        self.points.last().map(|p| p[1])
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// 曲线上标注的一个里程碑（首次完圈、圈时刷新…）。
pub struct Milestone {
    pub ep: u64,
    pub text: String,
}

/// 训练累计统计。
#[derive(Default, Clone, Copy)]
pub struct Totals {
    pub episodes: u64,
    pub laps: u64,
    pub crashes: u64,
    pub timeouts: u64,
}

/// 全部训练曲线与统计。
#[derive(Default)]
pub struct Metrics {
    // ---- 局级 ----
    /// [局数, 局总奖励]
    pub ep_reward: Series,
    /// [局数, 局步数]
    pub ep_len: Series,
    /// [局数, 平均车速 km/h]
    pub ep_speed: Series,
    /// [局数, 圈时]（仅完圈局）
    pub lap_time: Series,
    /// [局数, 近 50 局完圈率 0..1]
    pub success: Series,
    /// 奖励分解（每局累计值，可正可负）
    pub part_progress: Series,
    pub part_speed: Series,
    pub part_center: Series,
    pub part_time: Series,
    pub part_crash: Series,
    pub part_lap: Series,

    // ---- 步级 ----
    /// [步数, loss]
    pub loss: Series,
    /// [步数, Q 均值]
    pub q_mean: Series,
    /// [步数, 策略熵]
    pub entropy: Series,
    /// [步数, 状态价值]
    pub value: Series,
    /// [步数, epsilon]
    pub epsilon: Series,
    /// [步数, SPS]
    pub sps: Series,
    /// [步数, buffer / rollout 长度]
    pub buffer: Series,

    // ---- 汇总 ----
    pub totals: Totals,
    pub best_lap: Option<f32>,
    pub best_lap_ep: u64,
    /// 首次完圈（局号, 当时的横坐标值）。
    pub first_lap: Option<(u64, f64)>,
    pub milestones: Vec<Milestone>,
    /// 最近 80 局结果（结果条）。
    pub recent_results: VecDeque<EpisodeResult>,
    pub last_step: u64,
    pub last_buffer: usize,
    /// 当前算法的展示用信息。
    pub algo: Option<AlgoKind>,

    /// 最近 50 局奖励（滑动平均，卡片用）
    recent_rewards: VecDeque<f32>,
    /// 最近 50 局是否完圈
    recent_laps: VecDeque<bool>,
}

impl Metrics {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn set_algo(&mut self, algo: AlgoKind) {
        self.algo = Some(algo);
    }

    /// 一条步级指标。
    pub fn push_step(&mut self, m: &StepMetrics, sps: f32) {
        let x = m.step as f64;
        self.loss.push(x, m.loss as f64);
        self.q_mean.push(x, m.q_mean as f64);
        self.entropy.push(x, m.extra as f64);
        self.value.push(x, m.value as f64);
        self.epsilon.push(x, m.epsilon as f64);
        self.sps.push(x, sps as f64);
        self.buffer.push(x, m.buffer_len as f64);
        self.last_step = m.step;
        self.last_buffer = m.buffer_len;
    }

    /// 一局结束。
    pub fn push_episode(&mut self, s: &EpisodeSummary) {
        let ep = s.episode as f64;
        self.ep_reward.push(ep, s.reward as f64);
        self.ep_len.push(ep, s.steps as f64);
        self.ep_speed.push(ep, (s.avg_speed * 3.6) as f64);

        // 奖励分解：每局各部分累计
        self.part_progress.push(ep, s.parts.progress as f64);
        self.part_speed.push(ep, s.parts.speed as f64);
        self.part_center.push(ep, s.parts.center as f64);
        self.part_time.push(ep, s.parts.time as f64);
        self.part_crash.push(ep, s.parts.crash as f64);
        self.part_lap.push(ep, s.parts.lap as f64);

        self.totals.episodes += 1;
        match s.result {
            EpisodeResult::Crashed => self.totals.crashes += 1,
            EpisodeResult::Timeout => self.totals.timeouts += 1,
            EpisodeResult::LapDone => self.totals.laps += 1,
            EpisodeResult::Running => {}
        }
        self.recent_results.push_back(s.result);
        if self.recent_results.len() > 80 {
            self.recent_results.pop_front();
        }

        // 完圈统计
        self.recent_laps.push_back(s.has_lap());
        if self.recent_laps.len() > 50 {
            self.recent_laps.pop_front();
        }
        let rate = self.recent_laps.iter().filter(|x| **x).count() as f64
            / self.recent_laps.len().max(1) as f64;
        self.success.push(ep, rate);

        if s.has_lap() {
            let t = s.lap_time;
            self.lap_time.push(ep, t as f64);
            if self.first_lap.is_none() {
                self.first_lap = Some((s.episode, ep));
                self.milestones.push(Milestone {
                    ep: s.episode,
                    text: "首次完圈".into(),
                });
            }
            let improved = match self.best_lap {
                Some(b) => t < b - 0.3,
                None => true,
            };
            if improved {
                self.best_lap = Some(t);
                self.best_lap_ep = s.episode;
                self.milestones.push(Milestone {
                    ep: s.episode,
                    text: format!("圈时 {t:.1}s"),
                });
            }
        }

        self.recent_rewards.push_back(s.reward);
        if self.recent_rewards.len() > 50 {
            self.recent_rewards.pop_front();
        }
    }

    /// 外部事件标记（换赛道 / 换车型），画在奖励曲线上。
    pub fn push_marker(&mut self, text: String) {
        let ep = self.totals.episodes.max(1);
        self.milestones.push(Milestone { ep, text });
        if self.milestones.len() > 24 {
            self.milestones.remove(0);
        }
    }

    pub fn avg_reward_50(&self) -> f32 {
        if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32
        }
    }

    pub fn lap_rate_50(&self) -> f32 {
        self.success.last_y().unwrap_or(0.0) as f32
    }

    pub fn is_empty(&self) -> bool {
        self.totals.episodes == 0 && self.loss.is_empty()
    }
}

// ---------------------------------------------------------------------------
// 绘图小工具
// ---------------------------------------------------------------------------

fn fmt_num(v: f64) -> String {
    let a = v.abs();
    if a >= 100_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if a >= 10_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if a >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else if a >= 1.0 {
        format!("{v:.0}")
    } else if a == 0.0 {
        "0".into()
    } else if a >= 0.01 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

/// 统一样式的图：网格、细坐标轴、可拖拽缩放。
fn base_plot(id: &str, height: f32) -> Plot<'static> {
    Plot::new(id)
        .height(height)
        .show_grid(true)
        .allow_zoom(true)
        .allow_drag(true)
        .allow_scroll(true)
        .y_axis_min_width(38.0)
        .x_axis_formatter(|m, _| fmt_num(m.value))
        .y_axis_formatter(|m, _| fmt_num(m.value))
}

fn heading(ui: &mut egui::Ui, text: &str, hint: &str) {
    heading_chips(ui, text, hint, &[]);
}

/// 分区标题：`小色块 + 名称` 充当图例，比压在曲线上的浮层好读。
fn heading_chips(ui: &mut egui::Ui, text: &str, hint: &str, chips: &[(&str, Color32)]) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(text).strong().color(C_TEXT));
        if !hint.is_empty() {
            ui.label(RichText::new(hint).small().color(C_DIM));
        }
        for (name, color) in chips {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, *color);
            ui.label(RichText::new(*name).small().color(C_DIM));
        }
    });
    ui.add_space(6.0);
}

fn line(name: &str, points: &[[f64; 2]], color: Color32, width: f32) -> Line<'static> {
    Line::new(name.to_owned(), PlotPoints::from(points.to_vec()))
        .color(color)
        .width(width)
}

/// 滑动平均（窗口 w）。
fn moving_avg(points: &[[f64; 2]], w: usize) -> Vec<[f64; 2]> {
    if points.is_empty() {
        return Vec::new();
    }
    let w = w.max(1);
    let mut out = Vec::with_capacity(points.len());
    let mut sum = 0.0;
    for (i, p) in points.iter().enumerate() {
        sum += p[1];
        if i >= w {
            sum -= points[i - w][1];
        }
        let n = (i + 1).min(w);
        out.push([p[0], sum / n as f64]);
    }
    out
}

/// 把若干序列累加成"堆叠"坐标（用于奖励分解）。
fn stack(xs: &[f64], series: &[&Series]) -> Vec<Vec<[f64; 2]>> {
    let mut out = vec![Vec::with_capacity(xs.len()); series.len()];
    let mut acc = vec![0.0f64; xs.len()];
    for (si, s) in series.iter().enumerate() {
        for (i, x) in xs.iter().enumerate() {
            let y = s.points.get(i).map(|p| p[1]).unwrap_or(0.0);
            acc[i] += y;
            out[si].push([*x, acc[i]]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 卡片
// ---------------------------------------------------------------------------

fn card(ui: &mut egui::Ui, label: &str, value: String, hint: String, accent: Color32) {
    egui::Frame::new()
        .fill(C_CARD_BG)
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(label).size(11.0).color(C_DIM));
            ui.label(RichText::new(value).size(19.0).strong().color(accent));
            ui.label(RichText::new(hint).size(10.0).color(C_DIM));
        });
}

fn kpi_cards(ui: &mut egui::Ui, m: &Metrics) {
    let sps = m.sps.last_y().unwrap_or(0.0);

    let mut cards: Vec<(&str, String, String, Color32)> = vec![
        ("已训练", fmt_num(m.last_step as f64), "步".into(), C_TEXT),
        ("速度", format!("{sps:.0}"), "步/秒".into(), C_TEXT),
        (
            "局数",
            m.totals.episodes.to_string(),
            format!("撞 {} · 超 {}", m.totals.crashes, m.totals.timeouts),
            C_TEXT,
        ),
        (
            "完圈率",
            format!("{:.0}%", m.lap_rate_50() * 100.0),
            "近 50 局".into(),
            C_LAP,
        ),
        (
            "最佳圈时",
            m.best_lap
                .map(|t| format!("{t:.1}s"))
                .unwrap_or_else(|| "—".into()),
            m.best_lap
                .map(|_| format!("第 {} 局", m.best_lap_ep))
                .unwrap_or_else(|| "还没完圈".into()),
            C_ACCENT,
        ),
        (
            "平均奖励",
            format!("{:.0}", m.avg_reward_50()),
            "近 50 局".into(),
            C_ACCENT,
        ),
    ];
    match m.algo {
        Some(AlgoKind::Dqn) => {
            let e = m.epsilon.last_y().unwrap_or(0.0);
            cards.push((
                "探索率 ε",
                format!("{e:.2}"),
                "随机动作比例".into(),
                C_CENTER,
            ));
        }
        Some(AlgoKind::Ppo) => {
            let h = m.entropy.last_y().unwrap_or(0.0);
            cards.push(("策略熵", format!("{h:.2}"), "动作不确定性".into(), C_CENTER));
        }
        None => cards.push(("算法", "—".into(), String::new(), C_DIM)),
    }
    cards.push((
        match m.algo {
            Some(AlgoKind::Ppo) => "rollout",
            _ => "经验池",
        },
        fmt_num(m.last_buffer as f64),
        "条样本".into(),
        C_CENTER,
    ));

    // 固定列数由宽度决定：列宽均分，卡片永远不会溢出
    let avail = ui.available_width();
    let cols = if avail >= 620.0 {
        4
    } else if avail >= 330.0 {
        2
    } else {
        1
    };
    for chunk in cards.chunks(cols) {
        ui.columns(cols, |cs| {
            for (i, (label, value, hint, accent)) in chunk.iter().enumerate() {
                card(&mut cs[i], label, value.clone(), hint.clone(), *accent);
            }
        });
        ui.add_space(4.0);
    }
}

// ---------------------------------------------------------------------------
// 各分区
// ---------------------------------------------------------------------------

fn reward_section(ui: &mut egui::Ui, m: &Metrics) {
    heading_chips(
        ui,
        "局总奖励",
        "竖线：里程碑",
        &[
            ("每局", Color32::from_rgb(96, 104, 122)),
            ("滑动平均(50)", C_ACCENT),
            ("首次完圈/新纪录", Color32::from_rgb(90, 130, 110)),
        ],
    );
    let avg = moving_avg(&m.ep_reward.points, 50);
    let mut plot = base_plot("plot_reward", 180.0);
    plot = plot.include_y(0.0);
    plot.show(ui, |pu| {
        pu.line(line(
            "",
            &m.ep_reward.points,
            Color32::from_rgb(96, 104, 122),
            1.0,
        ));
        pu.line(line("", &avg, C_ACCENT, 2.0));
        pu.hline(
            HLine::new("", 0.0)
                .color(Color32::from_rgb(70, 74, 84))
                .width(1.0),
        );
        for ms in m.milestones.iter().take(6) {
            let _ = &ms.text;
            pu.vline(
                VLine::new("", ms.ep as f64)
                    .color(Color32::from_rgb(90, 130, 110))
                    .width(1.0)
                    .style(LineStyle::Dashed { length: 3.0 }),
            );
        }
    });
}

fn success_section(ui: &mut egui::Ui, m: &Metrics) {
    heading_chips(
        ui,
        "完圈率",
        "近 50 局窗口：策略\"会不会开\"的判据",
        &[("完圈率 (%)", C_LAP)],
    );
    base_plot("plot_success", 130.0)
        .include_y(-0.02)
        .include_y(1.02)
        .show(ui, |pu| {
            let pts: Vec<[f64; 2]> = m
                .success
                .points
                .iter()
                .map(|p| [p[0], p[1] * 100.0])
                .collect();
            pu.line(line("", &pts, C_LAP, 2.0));
            pu.hline(
                HLine::new("", 50.0)
                    .color(Color32::from_rgb(70, 74, 84))
                    .width(1.0)
                    .style(LineStyle::Dashed { length: 3.0 }),
            );
        });
}

fn breakdown_section(ui: &mut egui::Ui, m: &Metrics, cfg: &RewardConfig) {
    heading_chips(
        ui,
        "奖励分解",
        &format!(
            "进度 {:.2}/m · 速度 {:.2} · 走线 {:.2} · 时间 {:.2}/步 · 撞墙 −{:.0} · 完圈 +{:.0}",
            cfg.w_progress, cfg.w_speed, cfg.w_center, cfg.w_time, cfg.r_crash, cfg.r_lap
        ),
        &[
            ("进度", C_PROGRESS),
            ("速度", C_SPEED),
            ("完圈", C_LAP),
            ("走线", C_CENTER),
            ("时间", C_TIME),
            ("撞墙", C_CRASH),
        ],
    );
    let xs: Vec<f64> = m.part_progress.points.iter().map(|p| p[0]).collect();
    if xs.is_empty() {
        ui.label(RichText::new("还没有跑完一局").small().color(C_DIM));
        return;
    }
    // 正项向上堆，惩罚项向下堆
    let pos = stack(&xs, &[&m.part_progress, &m.part_speed, &m.part_lap]);
    let neg = stack(&xs, &[&m.part_center, &m.part_time, &m.part_crash]);
    let zero: Vec<[f64; 2]> = xs.iter().map(|x| [*x, 0.0]).collect();

    base_plot("plot_breakdown", 190.0).show(ui, |pu| {
        add_area(pu, &xs, "进度", C_PROGRESS, &zero, &pos[0]);
        add_area(pu, &xs, "速度", C_SPEED, &pos[0], &pos[1]);
        add_area(pu, &xs, "完圈", C_LAP, &pos[1], &pos[2]);
        add_area(pu, &xs, "走线", C_CENTER, &zero, &neg[0]);
        add_area(pu, &xs, "时间", C_TIME, &neg[0], &neg[1]);
        add_area(pu, &xs, "撞墙", C_CRASH, &neg[1], &neg[2]);
        pu.hline(
            HLine::new("零线", 0.0)
                .color(Color32::from_rgb(78, 82, 92))
                .width(1.0),
        );
    });
}

/// 堆叠面积：填充 `lower..upper` 两条曲线之间。
fn add_area(
    pu: &mut egui_plot::PlotUi<'_>,
    xs: &[f64],
    name: &str,
    color: Color32,
    lower: &[[f64; 2]],
    upper: &[[f64; 2]],
) {
    let lo: Vec<f64> = lower.iter().map(|p| p[1]).collect();
    let hi: Vec<f64> = upper.iter().map(|p| p[1]).collect();
    pu.add(
        egui_plot::FilledArea::new(name.to_owned(), xs, &lo, &hi)
            .fill_color(color.gamma_multiply(0.55))
            .stroke(Stroke::new(1.0, color)),
    );
}

fn result_strip(ui: &mut egui::Ui, m: &Metrics) {
    heading(ui, "最近 80 局", "绿=完圈 · 红=撞墙 · 黄=超时");
    let n = m.recent_results.len();
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 16.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    if n == 0 {
        return;
    }
    let cell_w = (rect.width() / 80.0).max(2.0);
    for (i, r) in m.recent_results.iter().enumerate() {
        let x = rect.left() + i as f32 * cell_w;
        let (col, _) = match r {
            EpisodeResult::LapDone => (C_LAP, "完圈"),
            EpisodeResult::Crashed => (C_CRASH, "撞墙"),
            EpisodeResult::Timeout => (Color32::from_rgb(196, 168, 92), "超时"),
            EpisodeResult::Running => (C_DIM, "进行中"),
        };
        let cell = egui::Rect::from_min_size(
            egui::pos2(x, rect.top()),
            egui::vec2((cell_w - 1.0).max(1.0), rect.height()),
        );
        painter.rect_filled(cell, 1.0, col.gamma_multiply(0.85));
    }
    ui.add_space(4.0);
    ui.label(
        RichText::new(format!(
            "完圈 {} · 撞墙 {} · 超时 {}",
            m.totals.laps, m.totals.crashes, m.totals.timeouts
        ))
        .small()
        .color(C_DIM),
    );
}

fn pairs_section(ui: &mut egui::Ui, m: &Metrics) {
    two_col(
        ui,
        |ui| {
            heading_chips(
                ui,
                "局步数",
                "越短说明越早撞或越早完圈",
                &[("步数", C_TEXT)],
            );
            base_plot("plot_len", 120.0).show(ui, |pu| {
                pu.line(line("", &m.ep_len.points, C_TEXT, 1.2));
            });
        },
        |ui| {
            heading_chips(
                ui,
                "圈时",
                "只有完圈局才有点",
                &[("圈时 (s)", C_ACCENT), ("最佳", C_LAP)],
            );
            base_plot("plot_lap", 120.0).show(ui, |pu| {
                if m.lap_time.is_empty() {
                    pu.line(line("", &[[0.0, 0.0]], Color32::TRANSPARENT, 0.0));
                } else {
                    pu.line(line("", &m.lap_time.points, C_ACCENT, 1.6));
                    if let Some(b) = m.best_lap {
                        pu.hline(
                            HLine::new("", b as f64)
                                .color(C_LAP)
                                .width(1.0)
                                .style(LineStyle::Dashed { length: 4.0 }),
                        );
                    }
                }
            });
        },
    );
}

fn loss_section(ui: &mut egui::Ui, m: &Metrics) {
    two_col(
        ui,
        |ui| {
            heading_chips(
                ui,
                "loss",
                "训练误差：抖动正常，看趋势",
                &[("loss", Color32::from_rgb(200, 140, 200))],
            );
            base_plot("plot_loss", 120.0).show(ui, |pu| {
                pu.line(line(
                    "",
                    &m.loss.points,
                    Color32::from_rgb(200, 140, 200),
                    1.2,
                ));
            });
        },
        |ui| match m.algo {
            Some(AlgoKind::Ppo) => {
                heading_chips(
                    ui,
                    "策略熵 / 状态价值",
                    "熵降 = 策略变笃定",
                    &[("熵", C_CENTER), ("价值", C_PROGRESS)],
                );
                base_plot("plot_entropy", 120.0).show(ui, |pu| {
                    pu.line(line("", &m.entropy.points, C_CENTER, 1.2));
                    pu.line(line("", &m.value.points, C_PROGRESS, 1.2));
                });
            }
            _ => {
                heading_chips(ui, "Q 均值", "策略对\"当前动作\"的估值", &[("Q", C_CENTER)]);
                base_plot("plot_q", 120.0).show(ui, |pu| {
                    pu.line(line("", &m.q_mean.points, C_CENTER, 1.2));
                });
            }
        },
    );
}

fn rate_section(ui: &mut egui::Ui, m: &Metrics) {
    two_col(
        ui,
        |ui| {
            heading_chips(
                ui,
                "探索率 ε",
                "DQN：随训练线性衰减到 5%",
                &[("ε", C_SPEED)],
            );
            base_plot("plot_eps", 110.0).show(ui, |pu| {
                if m.algo == Some(AlgoKind::Ppo) {
                    pu.text(egui_plot::Text::new(
                        "",
                        egui_plot::PlotPoint::new(0.0, 0.0),
                        "PPO 没有 ε",
                    ));
                } else {
                    pu.line(line("", &m.epsilon.points, C_SPEED, 1.2));
                }
            });
        },
        |ui| {
            heading_chips(ui, "训练速度", "步/秒", &[("SPS", C_TEXT)]);
            base_plot("plot_sps", 110.0).show(ui, |pu| {
                pu.line(line("", &m.sps.points, C_TEXT, 1.2));
            });
        },
    );
}

fn speed_buffer_section(ui: &mut egui::Ui, m: &Metrics) {
    two_col(
        ui,
        |ui| {
            heading_chips(
                ui,
                "平均车速",
                "km/h：完圈局通常更高",
                &[("km/h", C_PROGRESS)],
            );
            base_plot("plot_speed", 110.0).show(ui, |pu| {
                pu.line(line("", &m.ep_speed.points, C_PROGRESS, 1.2));
            });
        },
        |ui| {
            heading_chips(
                ui,
                "经验池 / rollout",
                "DQN 存样本，PPO 是上一轮 rollout",
                &[("样本", C_CENTER)],
            );
            base_plot("plot_buf", 110.0).show(ui, |pu| {
                pu.line(line("", &m.buffer.points, C_CENTER, 1.2));
            });
        },
    );
}

/// 宽屏两列，窄屏上下堆叠。
fn two_col(ui: &mut egui::Ui, left: impl FnOnce(&mut egui::Ui), right: impl FnOnce(&mut egui::Ui)) {
    if ui.available_width() >= 720.0 {
        ui.columns(2, |cols| {
            left(&mut cols[0]);
            right(&mut cols[1]);
        });
    } else {
        left(ui);
        ui.add_space(4.0);
        right(ui);
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

pub fn draw(ui: &mut egui::Ui, m: &Metrics, cfg: &RewardConfig) {
    if m.is_empty() {
        ui.vertical_centered(|ui| {
            ui.add_space(60.0);
            ui.label(RichText::new("还没有训练数据").size(16.0).color(C_DIM));
            ui.add_space(6.0);
            ui.label(
                RichText::new("右侧点「开始训练」，这里会实时长出奖励、完圈率、损失曲线")
                    .small()
                    .color(C_DIM),
            );
        });
        return;
    }

    ui.label(
        RichText::new("指标全部来自训练环境（无头跑、随机起步）；「训练镜像」页画的就是这辆车")
            .small()
            .color(C_DIM),
    );
    ui.add_space(6.0);
    kpi_cards(ui, m);
    ui.add_space(8.0);

    reward_section(ui, m);
    ui.add_space(6.0);

    two_col(ui, |ui| success_section(ui, m), |ui| result_strip(ui, m));
    ui.add_space(6.0);

    breakdown_section(ui, m, cfg);
    ui.add_space(6.0);

    pairs_section(ui, m);
    ui.add_space(6.0);

    loss_section(ui, m);
    ui.add_space(6.0);

    rate_section(ui, m);
    ui.add_space(6.0);

    speed_buffer_section(ui, m);
    ui.add_space(10.0);

    ui.label(
        RichText::new("提示：所有图都可以拖拽平移、滚轮缩放；双击图内恢复自适应。")
            .small()
            .color(C_DIM),
    );
    ui.add_space(6.0);
    ui.label(
        RichText::new(format!(
            "累计 {} 局 · {} 次完圈 · 奖励口径来自 sim::reward（右侧可换车/换赛道后重新训练）",
            m.totals.episodes, m.totals.laps
        ))
        .small()
        .color(C_DIM),
    );
}
