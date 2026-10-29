//! 控制面板。
//!
//! 两个模式的操作区完全分开：
//! - **训练工作台**：算法、开始/暂停/停止、环境（赛道/车型/seed，训练中也能改，改了由训练线程重建环境）、
//!   存档模型（续训 + 删除）、奖励说明。这里没有"播放/观看"。
//! - **展示厅**：选模型 → 大播放键 → 赛道/车型。这里没有任何训练入口，也没有删除。
//!
//! 视觉按 `docs/ui-spec.md` 实现：卡片分区 + 一个模式色（训练琥珀 / 展示冷蓝），
//! 图标全部用 `Painter` 几何（不用 emoji 字形），数值用 `FontId::monospace`。

use rl::policy::AlgoKind;
use sim::car::CarConfig;
use sim::config::SimConfig;
use sim::track::{RealTrack, TrackGenConfig};

use crate::app::{EnvSel, Mode, RlRacerApp, TrainState, TrainTab};
use crate::{MODE_DEMO, MODE_TRAIN};

// ---------------------------------------------------------------------------
// 色板（docs/ui-spec.md B 节）
// ---------------------------------------------------------------------------

/// 卡片底。
const C_CARD: egui::Color32 = egui::Color32::from_rgb(30, 32, 39); // #1E2027
/// 控件底。
const C_CTRL: egui::Color32 = egui::Color32::from_rgb(38, 41, 50); // #262932
/// 悬停底。
const C_HOVER: egui::Color32 = egui::Color32::from_rgb(47, 51, 62); // #2F333E
/// 普通描边。
const C_STROKE: egui::Color32 = egui::Color32::from_rgb(52, 56, 69); // #343845
/// 强描边。
const C_STROKE_STRONG: egui::Color32 = egui::Color32::from_rgb(69, 74, 89); // #454A59
/// 区块标题右侧细线。
pub(crate) const C_RULE: egui::Color32 = egui::Color32::from_rgb(42, 45, 55); // #2A2D37
/// 折叠内容底。
const C_SUNKEN: egui::Color32 = egui::Color32::from_rgb(20, 22, 27); // #14161B
/// 主文字。
pub(crate) const C_TEXT: egui::Color32 = egui::Color32::from_rgb(220, 223, 230); // #DCDFE6
/// 次要文字。
const C_TEXT2: egui::Color32 = egui::Color32::from_rgb(180, 185, 196); // #B4B9C4
/// 弱文字。
pub(crate) const C_DIM: egui::Color32 = egui::Color32::from_rgb(140, 145, 155); // #8C919B
/// 警告。
pub(crate) const C_WARN: egui::Color32 = egui::Color32::from_rgb(220, 185, 95); // #DCB95F
/// 危险。
const C_DANGER: egui::Color32 = egui::Color32::from_rgb(206, 92, 84); // #CE5C54
/// 危险文字。
const C_DANGER_TEXT: egui::Color32 = egui::Color32::from_rgb(229, 127, 120); // #E57F78
/// 危险实心底上的文字。
const C_ON_DANGER: egui::Color32 = egui::Color32::from_rgb(250, 250, 252);
/// 琥珀悬停。
const C_AMBER_HOVER: egui::Color32 = egui::Color32::from_rgb(226, 160, 74); // #E2A04A
/// 琥珀底上的文字。
const C_ON_AMBER: egui::Color32 = egui::Color32::from_rgb(26, 18, 5); // #1A1205
/// 琥珀选中文字。
const C_AMBER_TEXT: egui::Color32 = egui::Color32::from_rgb(232, 181, 101); // #E8B565
/// 绿色：正在跑 / 可播放。
pub(crate) const C_PLAY: egui::Color32 = egui::Color32::from_rgb(72, 178, 112); // #48B270
/// 绿色悬停。
const C_PLAY_HOVER: egui::Color32 = egui::Color32::from_rgb(88, 198, 130); // #58C682
/// 绿色按下。
const C_PLAY_DOWN: egui::Color32 = egui::Color32::from_rgb(62, 154, 96); // #3E9A60
/// 绿底上的文字。
const C_ON_PLAY: egui::Color32 = egui::Color32::from_rgb(8, 24, 14); // #08180E
/// 暂停态大播放键的底。
const C_PAUSE_BG: egui::Color32 = egui::Color32::from_rgb(43, 47, 57); // #2B2F39
/// 暂停态大播放键悬停底。
const C_PAUSE_BG_HOVER: egui::Color32 = egui::Color32::from_rgb(50, 55, 69); // #323745
/// 暂停态大播放键按下底。
const C_PAUSE_BG_DOWN: egui::Color32 = egui::Color32::from_rgb(38, 42, 51); // #262A33
/// 暂停态大播放键的文字。
const C_PAUSE_TEXT: egui::Color32 = egui::Color32::from_rgb(232, 235, 240); // #E8EBF0
/// 暂停双竖条。
const C_PAUSE_BAR: egui::Color32 = egui::Color32::from_rgb(111, 213, 150); // #6FD596
/// 大播放键禁用底。
const C_PLAY_OFF_BG: egui::Color32 = egui::Color32::from_rgb(35, 39, 48); // #232730
/// 大播放键禁用前景。
const C_PLAY_OFF_FG: egui::Color32 = egui::Color32::from_rgb(95, 100, 112); // #5F6470
/// 禁用控件底。
const C_OFF_FILL: egui::Color32 = egui::Color32::from_rgb(34, 37, 44); // #22252C
/// 禁用控件文字。
const C_OFF_TEXT: egui::Color32 = egui::Color32::from_rgb(107, 112, 123); // #6B707B

// ---------------------------------------------------------------------------
// 赛道 / 难度 / 车型
// ---------------------------------------------------------------------------

/// 赛道来源：随机生成或真实赛道近似轮廓。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackSel {
    Random,
    Real(RealTrack),
}

impl TrackSel {
    pub fn label(self) -> &'static str {
        match self {
            TrackSel::Random => "随机",
            TrackSel::Real(t) => t.label(),
        }
    }

    pub const ALL: [TrackSel; 6] = [
        TrackSel::Random,
        TrackSel::Real(RealTrack::Monza),
        TrackSel::Real(RealTrack::Monaco),
        TrackSel::Real(RealTrack::Spa),
        TrackSel::Real(RealTrack::Silverstone),
        TrackSel::Real(RealTrack::Shanghai),
    ];
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    pub fn label(self) -> &'static str {
        match self {
            Difficulty::Easy => "简单",
            Difficulty::Medium => "中等",
            Difficulty::Hard => "困难",
        }
    }

    pub fn to_cfg(self) -> TrackGenConfig {
        match self {
            Difficulty::Easy => TrackGenConfig::easy(),
            Difficulty::Medium => TrackGenConfig::medium(),
            Difficulty::Hard => TrackGenConfig::hard(),
        }
    }

    pub const ALL: [Difficulty; 3] = [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard];
}

/// 车型预设（调校滑杆改动后仍保持选中，但参数已自定义）。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CarKind {
    Hatch,
    Sport,
    Muscle,
    Kart,
    Formula,
}

impl CarKind {
    pub fn label(self) -> &'static str {
        match self {
            CarKind::Hatch => "均衡",
            CarKind::Sport => "跑车",
            CarKind::Muscle => "肌肉",
            CarKind::Kart => "卡丁",
            CarKind::Formula => "方程式",
        }
    }

    /// 车身颜色。
    pub fn color(self) -> [u8; 3] {
        match self {
            CarKind::Hatch => [235, 120, 50],
            CarKind::Sport => [240, 200, 70],
            CarKind::Muscle => [80, 105, 225],
            CarKind::Kart => [85, 200, 120],
            CarKind::Formula => [225, 50, 50],
        }
    }

    pub fn to_cfg(self) -> CarConfig {
        match self {
            CarKind::Hatch => CarConfig::hatch(),
            CarKind::Sport => CarConfig::sport(),
            CarKind::Muscle => CarConfig::muscle(),
            CarKind::Kart => CarConfig::kart(),
            CarKind::Formula => CarConfig::formula(),
        }
    }

    pub const ALL: [CarKind; 5] = [
        CarKind::Hatch,
        CarKind::Sport,
        CarKind::Muscle,
        CarKind::Kart,
        CarKind::Formula,
    ];
}

// ---------------------------------------------------------------------------
// 基础绘制件：文字 / 卡片 / 分段选择器 / 按钮
// ---------------------------------------------------------------------------

/// 按宽度自适应缩小字号（绝不溢出容器）。
fn fitting_galley(
    painter: &egui::Painter,
    text: &str,
    size: f32,
    color: egui::Color32,
    max_w: f32,
) -> std::sync::Arc<egui::Galley> {
    let mut s = size;
    loop {
        let g = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(s), color);
        if g.size().x <= max_w || s <= 9.0 {
            return g;
        }
        s -= 1.0;
    }
}

/// 卡片底板：`#1E2027`、圆角 8、内边距 12、描边 `#343845`。
fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(C_CARD)
        .stroke(egui::Stroke::new(1.0, C_STROKE))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::same(12))
}

/// 区块标题：3×12 模式色竖条 + 12px strong 次色标题 + 右侧 1px 细线。
fn section_title(ui: &mut egui::Ui, title: &str, accent: egui::Color32, right: Option<&str>) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 14.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(
        egui::Rect::from_min_size(rect.left_top(), egui::vec2(3.0, 12.0)),
        1.5,
        accent,
    );
    let g = painter.layout_no_wrap(title.to_owned(), egui::FontId::proportional(12.0), C_TEXT2);
    let text_x = rect.left() + 3.0 + 7.0;
    let text_w = g.size().x;
    painter.galley(
        egui::pos2(text_x, rect.center().y - g.size().y * 0.5),
        g,
        C_TEXT2,
    );

    let line_x0 = text_x + text_w + 8.0;
    let mut line_x1 = rect.right();
    if let Some(right) = right {
        let rg = painter.layout_no_wrap(right.to_owned(), egui::FontId::proportional(11.0), C_DIM);
        line_x1 = rect.right() - rg.size().x - 8.0;
        let rx = rect.right() - rg.size().x;
        painter.galley(
            egui::pos2(rx, rect.center().y - rg.size().y * 0.5),
            rg,
            C_DIM,
        );
    }
    if line_x1 > line_x0 + 4.0 {
        let y = rect.center().y;
        painter.line_segment(
            [egui::pos2(line_x0, y), egui::pos2(line_x1, y)],
            egui::Stroke::new(1.0, C_RULE),
        );
    }
}

/// 卡片 + 区块标题 + 内容。
pub(crate) fn section_card<R>(
    ui: &mut egui::Ui,
    title: &str,
    accent: egui::Color32,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    section_card_right(ui, title, accent, None, None, add)
}

/// 卡片，标题右端可以带一段 11px 弱色说明（例如"N 个"）。
pub(crate) fn section_card_right<R>(
    ui: &mut egui::Ui,
    title: &str,
    accent: egui::Color32,
    right: Option<&str>,
    left_bar: Option<egui::Color32>,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let resp = card_frame().show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.spacing_mut().item_spacing.y = 0.0;
        section_title(ui, title, accent, right);
        ui.add_space(8.0);
        add(ui)
    });
    if let Some(col) = left_bar {
        let r = resp.response.rect;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(r.left() + 1.0, r.top() + 8.0),
                egui::vec2(3.0, (r.height() - 16.0).max(4.0)),
            ),
            1.5,
            col,
        );
    }
    resp.inner
}

/// 分段选择器的一格。
pub(crate) struct SegItem {
    pub label: &'static str,
    /// 车型格底部的 3px 车色条。
    pub bar: Option<egui::Color32>,
}

impl SegItem {
    pub(crate) const fn new(label: &'static str) -> Self {
        Self { label, bar: None }
    }

    pub(crate) const fn with_bar(label: &'static str, rgb: [u8; 3]) -> Self {
        Self {
            label,
            bar: Some(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])),
        }
    }
}

/// 选中态文字：琥珀 `#E8B565` / 展示蓝 `#78AADC`。
fn selected_text(accent: egui::Color32) -> egui::Color32 {
    if accent == MODE_DEMO {
        MODE_DEMO
    } else {
        C_AMBER_TEXT
    }
}

/// 分段选择器（赛道 / 难度 / 车型 / 模式）。
///
/// - 等分宽度：`cell_w = floor((W − (cols−1)·4) / cols)`，余数补给每行最后一格；
/// - `W/n ≥ 52` 单行，否则 `cols = ceil(n/2)`；
/// - 格宽 < 56 时字号 12，否则 13。
///
/// 返回被点击的索引。
pub(crate) fn segmented(
    ui: &mut egui::Ui,
    id_salt: impl egui::AsIdSalt + Copy,
    items: &[SegItem],
    selected: Option<usize>,
    accent: egui::Color32,
) -> Option<usize> {
    let n = items.len();
    if n == 0 {
        return None;
    }
    let w = ui.available_width();
    let gap = 4.0;
    let h = 28.0;
    let cols = if w / n as f32 >= 52.0 { n } else { (n + 1) / 2 };
    let rows = (n + cols - 1) / cols;
    let cell_w = (((w - (cols as f32 - 1.0) * gap) / cols as f32).floor()).max(24.0);
    let total_h = rows as f32 * h + (rows as f32 - 1.0) * gap;
    let (outer, _) = ui.allocate_exact_size(egui::vec2(w, total_h), egui::Sense::hover());

    let mut clicked = None;
    for (i, item) in items.iter().enumerate() {
        let r = i / cols;
        let c = i % cols;
        let in_row = cols.min(n - r * cols);
        let x = outer.left() + c as f32 * (cell_w + gap);
        let cw = if c + 1 == in_row {
            (outer.right() - x).max(24.0)
        } else {
            cell_w
        };
        let y = outer.top() + r as f32 * (h + gap);
        let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cw, h));
        let resp = ui.interact(rect, ui.scope_id().with((id_salt, i)), egui::Sense::click());

        let is_sel = selected == Some(i);
        let (bg, stroke, fg) = if is_sel {
            (
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 46),
                egui::Stroke::new(1.5, accent),
                selected_text(accent),
            )
        } else if resp.hovered() {
            (C_HOVER, egui::Stroke::new(1.0, C_STROKE_STRONG), C_TEXT2)
        } else {
            (C_CTRL, egui::Stroke::new(1.0, C_STROKE), C_TEXT2)
        };

        let painter = ui.painter();
        painter.rect_filled(rect, 6.0, bg);
        painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);

        let font_size = if cw < 56.0 { 12.0 } else { 13.0 };
        let g = fitting_galley(painter, item.label, font_size, fg, cw - 8.0);
        let cy = if item.bar.is_some() {
            rect.center().y - 3.0
        } else {
            rect.center().y
        };
        painter.galley(
            egui::pos2(rect.center().x - g.size().x * 0.5, cy - g.size().y * 0.5),
            g,
            fg,
        );

        if let Some(bar) = item.bar {
            let bw = (cw - 20.0).max(8.0);
            let col = if is_sel {
                bar
            } else {
                egui::Color32::from_rgba_unmultiplied(bar.r(), bar.g(), bar.b(), 160)
            };
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.center().x - bw * 0.5, rect.bottom() - 9.0),
                    egui::vec2(bw, 3.0),
                ),
                1.5,
                col,
            );
        }

        if resp.clicked() {
            clicked = Some(i);
        }
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }
    clicked
}

/// 按钮图标（全部 Painter 几何）。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Icon {
    None,
    /// 开始 / 播放。
    Triangle,
    /// 停止。
    Square,
    /// 暂停。
    Pause,
}

impl Icon {
    fn width(self) -> f32 {
        match self {
            Icon::None => 0.0,
            Icon::Triangle => 10.0,
            Icon::Square => 10.0,
            Icon::Pause => 10.0,
        }
    }

    fn height(self) -> f32 {
        match self {
            Icon::None => 0.0,
            Icon::Triangle => 12.0,
            Icon::Square => 10.0,
            Icon::Pause => 12.0,
        }
    }
}

fn paint_icon(painter: &egui::Painter, rect: egui::Rect, icon: Icon, color: egui::Color32) {
    let c = rect.center();
    match icon {
        Icon::None => {}
        Icon::Triangle => {
            painter.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(rect.left(), c.y - 6.0),
                    egui::pos2(rect.left(), c.y + 6.0),
                    egui::pos2(rect.left() + 10.0, c.y),
                ],
                color,
                egui::Stroke::NONE,
            ));
        }
        Icon::Square => {
            painter.rect_filled(
                egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0)),
                2.0,
                color,
            );
        }
        Icon::Pause => {
            for dx in [-5.0, 2.0] {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(c.x + dx, c.y - 6.0),
                        egui::vec2(3.0, 12.0),
                    ),
                    1.0,
                    color,
                );
            }
        }
    }
}

/// 按钮类型。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActionKind {
    /// 一屏唯一的主操作：琥珀实心。
    Primary,
    /// 危险操作：危险色 alpha 30 底 + 描边。
    Danger,
    /// 删除二次确认：实心危险色 + 白字。
    DangerArmed,
    /// 次要操作：控件灰。
    Secondary,
}

impl ActionKind {
    pub(crate) fn height(self) -> f32 {
        match self {
            ActionKind::Primary => 36.0,
            ActionKind::Danger | ActionKind::DangerArmed => 32.0,
            ActionKind::Secondary => 28.0,
        }
    }

    fn font_size(self) -> f32 {
        match self {
            ActionKind::Primary => 14.0,
            _ => 13.0,
        }
    }
}

fn paint_button(
    ui: &egui::Ui,
    rect: egui::Rect,
    resp: &egui::Response,
    label: &str,
    kind: ActionKind,
    icon: Icon,
    enabled: bool,
) {
    let painter = ui.painter();
    let hovered = resp.hovered() && enabled;
    let down = resp.is_pointer_button_down_on() && enabled;

    let (fill, stroke, fg) = if !enabled {
        (C_OFF_FILL, egui::Stroke::new(1.0, C_RULE), C_OFF_TEXT)
    } else {
        match kind {
            ActionKind::Primary => {
                let fill = if down {
                    MODE_TRAIN.gamma_multiply(0.86)
                } else if hovered {
                    C_AMBER_HOVER
                } else {
                    MODE_TRAIN
                };
                (fill, egui::Stroke::NONE, C_ON_AMBER)
            }
            ActionKind::Danger => {
                let a = if hovered { 44 } else { 30 };
                (
                    egui::Color32::from_rgba_unmultiplied(
                        C_DANGER.r(),
                        C_DANGER.g(),
                        C_DANGER.b(),
                        a,
                    ),
                    egui::Stroke::new(1.0, C_DANGER),
                    C_DANGER_TEXT,
                )
            }
            ActionKind::DangerArmed => (C_DANGER, egui::Stroke::NONE, C_ON_DANGER),
            ActionKind::Secondary => {
                if hovered {
                    (C_HOVER, egui::Stroke::new(1.0, C_STROKE_STRONG), C_TEXT)
                } else {
                    (C_CTRL, egui::Stroke::new(1.0, C_STROKE), C_TEXT2)
                }
            }
        }
    };

    painter.rect_filled(rect, 6.0, fill);
    if stroke != egui::Stroke::NONE {
        painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);
    }

    let icon_w = icon.width();
    let gap = if icon == Icon::None { 0.0 } else { 8.0 };
    let max_text_w = (rect.width() - 8.0 - icon_w - gap).max(8.0);
    let g = fitting_galley(painter, label, kind.font_size(), fg, max_text_w);
    let total = icon_w + gap + g.size().x;
    let x0 = rect.center().x - total * 0.5;
    if icon != Icon::None {
        let icon_rect = egui::Rect::from_min_size(
            egui::pos2(x0, rect.center().y - icon.height() * 0.5),
            egui::vec2(icon_w, icon.height()),
        );
        paint_icon(painter, icon_rect, icon, fg);
    }
    painter.galley(
        egui::pos2(x0 + icon_w + gap, rect.center().y - g.size().y * 0.5),
        g,
        fg,
    );
}

/// 全宽自绘按钮（高度由 `kind` 决定）。
pub(crate) fn action_button(
    ui: &mut egui::Ui,
    label: &str,
    kind: ActionKind,
    enabled: bool,
) -> egui::Response {
    let w = ui.available_width();
    // 主操作固定带左侧 12px 三角（规范 D 节"按钮"）。
    let icon = if kind == ActionKind::Primary {
        Icon::Triangle
    } else {
        Icon::None
    };
    action_button_sized(ui, egui::vec2(w, kind.height()), label, kind, icon, enabled)
}

/// 指定尺寸的自绘按钮。
pub(crate) fn action_button_sized(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    label: &str,
    kind: ActionKind,
    icon: Icon,
    enabled: bool,
) -> egui::Response {
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    paint_button(ui, rect, &resp, label, kind, icon, enabled);
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// 在给定矩形里放一个自绘按钮（模型行内使用，先注册行再注册按钮）。
fn action_button_at(
    ui: &egui::Ui,
    rect: egui::Rect,
    id_salt: impl egui::AsIdSalt + Copy,
    label: &str,
    kind: ActionKind,
    icon: Icon,
    enabled: bool,
) -> egui::Response {
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let resp = ui.interact(rect, ui.scope_id().with(id_salt), sense);
    paint_button(ui, rect, &resp, label, kind, icon, enabled);
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// 弱色字段标签（11px，距下方控件 4px）。
fn field_label(ui: &mut egui::Ui, text: &str) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 14.0), egui::Sense::hover());
    let painter = ui.painter();
    let g = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(11.0), C_DIM);
    painter.galley(
        egui::pos2(rect.left(), rect.center().y - g.size().y * 0.5),
        g,
        C_DIM,
    );
    ui.add_space(4.0);
}

/// CollapsingHeader：标题 12px strong 次色，内容放进 `#14161B` 圆角 6 的 Frame。
fn collapsible<R>(
    ui: &mut egui::Ui,
    title: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    let mut out = None;
    egui::CollapsingHeader::new(
        egui::RichText::new(title)
            .size(12.0)
            .strong()
            .color(C_TEXT2),
    )
    .default_open(false)
    .show(ui, |ui| {
        egui::Frame::new()
            .fill(C_SUNKEN)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                out = Some(add(ui));
            });
    });
    out
}

// ---------------------------------------------------------------------------
// 面板总入口
// ---------------------------------------------------------------------------

pub fn draw(ui: &mut egui::Ui, app: &mut RlRacerApp) {
    let accent = match app.mode {
        Mode::Train => MODE_TRAIN,
        Mode::Demo => MODE_DEMO,
    };
    crate::set_mode_accent(ui.ctx(), accent);

    // 面板顶部 2px 模式色条
    let w = ui.available_width();
    let (bar, _) = ui.allocate_exact_size(egui::vec2(w, 2.0), egui::Sense::hover());
    ui.painter().rect_filled(bar, 1.0, accent);
    ui.add_space(10.0);

    egui::Frame::new()
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 0.0;

            header_row(ui, app);
            ui.add_space(8.0);
            mode_switch(ui, app, accent);
            ui.add_space(6.0);
            ui.add(
                egui::Label::new(
                    egui::RichText::new(match app.mode {
                        Mode::Train => {
                            "训练 = 训练线程 + 环境镜像 + 看板；这里改环境会重建训练环境"
                        }
                        Mode::Demo => "展示 = 载入模型自己跑；这里不会影响、也不会启动训练",
                    })
                    .size(11.0)
                    .color(C_DIM),
                )
                .wrap(),
            );
            ui.add_space(8.0);

            match app.mode {
                Mode::Train => train_panel(ui, app, accent),
                Mode::Demo => demo_panel(ui, app, accent),
            }
        });
}

/// 标题行："RL Racer" + 右侧状态 pill。
fn header_row(ui: &mut egui::Ui, app: &RlRacerApp) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 24.0), egui::Sense::hover());
    let painter = ui.painter();
    let g = painter.layout_no_wrap(
        "RL Racer".to_owned(),
        egui::FontId::proportional(16.0),
        C_TEXT,
    );
    // 不能加载粗体字体：同一 galley 在 x+0.5 再画一次做伪粗体。
    let pos = egui::pos2(rect.left(), rect.center().y - g.size().y * 0.5);
    painter.galley(pos, g.clone(), C_TEXT);
    painter.galley(pos + egui::vec2(0.5, 0.0), g, C_TEXT);

    // 状态 pill：展示厅看策略/播放，工作台看训练线程。
    let (text, color, icon) = match app.mode {
        Mode::Demo => {
            let loaded = app.viewer.policy.is_some();
            if loaded && app.viewer.running {
                ("播放中", C_PLAY, PillIcon::DotFilled)
            } else if loaded {
                ("已暂停", MODE_DEMO, PillIcon::PauseBars)
            } else {
                ("未载入", C_DIM, PillIcon::DotHollow)
            }
        }
        Mode::Train => match app.train_state() {
            TrainState::Idle => ("空闲", C_DIM, PillIcon::DotHollow),
            TrainState::Running => ("训练中", C_PLAY, PillIcon::DotFilled),
            TrainState::Paused => ("已暂停", C_WARN, PillIcon::PauseBars),
        },
    };
    status_pill(ui, rect, text, color, icon);
}

/// 状态 pill 左侧的小图标（全部 Painter 几何）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum PillIcon {
    /// 空心圆点：空闲 / 未载入。
    DotHollow,
    /// 实心圆点：正在跑。
    DotFilled,
    /// 双竖条：已暂停。
    PauseBars,
}

/// 状态 pill：图标 + 11px 文字，底色控件灰、1px 描边、圆角取高度一半。
fn status_pill(ui: &egui::Ui, area: egui::Rect, text: &str, color: egui::Color32, icon: PillIcon) {
    let painter = ui.painter();
    let g = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(11.0), color);
    let pad = egui::vec2(9.0, 5.0);
    let icon_w = 8.0;
    let w = pad.x * 2.0 + icon_w + 4.0 + g.size().x;
    let h = (g.size().y + pad.y * 2.0).max(22.0);
    let rect = egui::Rect::from_center_size(
        egui::pos2(area.right() - w * 0.5, area.center().y),
        egui::vec2(w, h),
    );
    painter.rect_filled(rect, h * 0.5, C_CTRL);
    painter.rect_stroke(
        rect,
        h * 0.5,
        egui::Stroke::new(1.0, C_STROKE),
        egui::StrokeKind::Inside,
    );
    let icon_c = egui::pos2(rect.left() + pad.x + 3.0, rect.center().y);
    match icon {
        PillIcon::DotHollow => {
            painter.circle_stroke(icon_c, 2.6, egui::Stroke::new(1.0, color));
        }
        PillIcon::DotFilled => {
            painter.circle_filled(icon_c, 3.0, color);
        }
        PillIcon::PauseBars => {
            for dx in [-2.2, 0.2] {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(icon_c.x + dx, icon_c.y - 3.5),
                        egui::vec2(2.0, 7.0),
                    ),
                    0.5,
                    color,
                );
            }
        }
    }
    painter.galley(
        egui::pos2(
            rect.left() + pad.x + icon_w + 4.0,
            rect.center().y - g.size().y * 0.5,
        ),
        g,
        color,
    );
}

/// 顶层模式切换（分段选择器，避免"以为在看演示其实在训练"）。
fn mode_switch(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    let items = [SegItem::new("训练工作台"), SegItem::new("展示厅")];
    let sel = match app.mode {
        Mode::Train => 0,
        Mode::Demo => 1,
    };
    if let Some(i) = segmented(ui, "mode_switch", &items, Some(sel), accent) {
        app.mode = if i == 0 { Mode::Train } else { Mode::Demo };
    }
}

// ---------------------------------------------------------------------------
// 训练工作台
// ---------------------------------------------------------------------------

fn train_panel(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    train_card(ui, app, accent);
    ui.add_space(8.0);
    env_section(ui, app, true, accent);
    ui.add_space(8.0);
    model_admin(ui, app, accent);
    ui.add_space(8.0);
    reward_note(ui, &app.train_reward);
}

fn train_card(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    let running = app.train_state() != TrainState::Idle;
    section_card_right(
        ui,
        "训练",
        accent,
        None,
        if running { Some(accent) } else { None },
        |ui| match app.train_state() {
            TrainState::Idle => {
                let items = [
                    SegItem::new(AlgoKind::Dqn.label()),
                    SegItem::new(AlgoKind::Ppo.label()),
                ];
                let sel = match app.train_algo {
                    AlgoKind::Dqn => 0,
                    AlgoKind::Ppo => 1,
                };
                if let Some(i) = segmented(ui, "train_algo", &items, Some(sel), accent) {
                    app.train_algo = if i == 0 { AlgoKind::Dqn } else { AlgoKind::Ppo };
                }
                ui.add_space(8.0);
                if action_button(ui, "开始训练", ActionKind::Primary, true).clicked() {
                    let seed = app.train_env.seed;
                    app.start_training(seed, None);
                }
                if let Some(path) = &app.last_checkpoint {
                    ui.add_space(4.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("上次存档：{path}"))
                                .size(11.0)
                                .color(C_DIM),
                        )
                        .wrap(),
                    );
                }
            }
            state => {
                let paused = matches!(state, TrainState::Paused);
                stat_grid(ui, app, paused);
                ui.add_space(8.0);

                let w = ui.available_width();
                let bw = ((w - 12.0) / 3.0).max(30.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    let pause_label = if paused { "继续" } else { "暂停" };
                    let pause_kind = if paused {
                        ActionKind::Primary
                    } else {
                        ActionKind::Secondary
                    };
                    let pause_icon = if paused { Icon::Triangle } else { Icon::Pause };
                    if action_button_sized(
                        ui,
                        egui::vec2(bw, pause_kind.height()),
                        pause_label,
                        pause_kind,
                        pause_icon,
                        true,
                    )
                    .clicked()
                    {
                        app.pause_training();
                    }
                    if action_button_sized(
                        ui,
                        egui::vec2(bw, ActionKind::Danger.height()),
                        "停止并保存",
                        ActionKind::Danger,
                        Icon::Square,
                        true,
                    )
                    .clicked()
                    {
                        app.stop_training();
                    }
                    if action_button_sized(
                        ui,
                        egui::vec2(bw, ActionKind::Secondary.height()),
                        "看板",
                        ActionKind::Secondary,
                        Icon::None,
                        true,
                    )
                    .clicked()
                    {
                        app.train_tab = TrainTab::Board;
                    }
                });
                ui.add_space(6.0);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(
                            "换赛道/换车型会重建训练环境：当前局作废，策略权重保留，经验池清空",
                        )
                        .size(11.0)
                        .color(C_DIM),
                    )
                    .wrap(),
                );
            }
        },
    );
}

/// 训练中的 2×2 统计网格（步数 / SPS / 局数 / 完圈率）。
fn stat_grid(ui: &mut egui::Ui, app: &RlRacerApp, paused: bool) {
    let w = ui.available_width();
    let cell_w = ((w - 8.0) / 2.0).max(40.0);
    let cell_h = 46.0;
    let gap = 8.0;
    let items = [
        ("步数", format!("{}", app.train_steps), ""),
        ("SPS", format!("{:.0}", app.sps), ""),
        ("局数", format!("{}", app.metrics.totals.episodes), ""),
        (
            "完圈率",
            format!("{:.0}", app.metrics.lap_rate_50() * 100.0),
            "%",
        ),
    ];
    let (outer, _) =
        ui.allocate_exact_size(egui::vec2(w, cell_h * 2.0 + gap), egui::Sense::hover());
    for (i, (label, value, unit)) in items.iter().enumerate() {
        let rect = egui::Rect::from_min_size(
            egui::pos2(
                outer.left() + (i % 2) as f32 * (cell_w + gap),
                outer.top() + (i / 2) as f32 * (cell_h + gap),
            ),
            egui::vec2(cell_w, cell_h),
        );
        let painter = ui.painter();
        let lg =
            painter.layout_no_wrap((*label).to_owned(), egui::FontId::proportional(11.0), C_DIM);
        painter.galley(egui::pos2(rect.left(), rect.top() + 2.0), lg, C_DIM);
        let val_color = if paused {
            egui::Color32::from_rgba_unmultiplied(C_TEXT.r(), C_TEXT.g(), C_TEXT.b(), 150)
        } else {
            C_TEXT
        };
        let vg = fitting_galley(painter, value, 18.0, val_color, (cell_w - 20.0).max(16.0));
        let (vw, vh) = (vg.size().x, vg.size().y);
        let vy = rect.bottom() - vh - 4.0;
        painter.galley(egui::pos2(rect.left(), vy), vg, val_color);
        if !unit.is_empty() {
            let ug =
                painter.layout_no_wrap((*unit).to_owned(), egui::FontId::proportional(11.0), C_DIM);
            let uy = rect.bottom() - ug.size().y - 6.0;
            painter.galley(egui::pos2(rect.left() + vw + 3.0, uy), ug, C_DIM);
        }
    }
}

// ---------------------------------------------------------------------------
// 展示厅
// ---------------------------------------------------------------------------

fn demo_panel(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    model_picker(ui, app, accent);
    ui.add_space(8.0);
    transport(ui, app);
    ui.add_space(8.0);
    env_section(ui, app, false, accent);
}

/// 展示厅第一段：选模型（单选，选中不等于开跑）。
fn model_picker(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    let models = models_cached();
    let count = format!("{} 个", models.len());
    section_card_right(ui, "选择模型", accent, Some(&count), None, |ui| {
        if models.is_empty() {
            ui.add(
                egui::Label::new(
                    egui::RichText::new("checkpoints/ 里还没有模型——先去训练工作台训一个")
                        .size(11.0)
                        .color(C_DIM),
                )
                .wrap(),
            );
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("demo_models")
            .max_height(188.0)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                for (i, c) in models.iter().enumerate() {
                    if i > 0 {
                        ui.add_space(4.0);
                    }
                    let selected = app.demo_model.as_deref() == Some(c.meta.name.as_str());
                    let is_current = app.viewer.policy_file.as_deref()
                        == Some(c.meta.name.as_str())
                        && app.viewer.policy.is_some();
                    let current = if is_current {
                        Some(app.viewer.running)
                    } else {
                        None
                    };
                    let hit = model_row(ui, c, selected, current, accent, None, false);
                    if hit.row.clicked() {
                        app.demo_model = Some(c.meta.name.clone());
                        app.pending_delete = None;
                    }
                    if hit.row.double_clicked() {
                        app.demo_model = Some(c.meta.name.clone());
                        app.demo_play_selected();
                    }
                }
            });
    });
}

/// 自绘大播放键：高 64、圆角 12，绿色播放态 / 描边暂停态。
/// 不用 emoji 字形（会显示成彩色小方块，跟配色打架）。
fn transport_button(ui: &mut egui::Ui, playing: bool, enabled: bool) -> egui::Response {
    let w = ui.available_width();
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 64.0), sense);
    let hovered = resp.hovered() && enabled;
    let down = resp.is_pointer_button_down_on() && enabled;

    let painter = ui.painter();
    let mut stroke: Option<egui::Stroke> = None;
    let (fill, fg, icon_col) = if !enabled {
        (C_PLAY_OFF_BG, C_PLAY_OFF_FG, C_PLAY_OFF_FG)
    } else if playing {
        let fill = if down {
            C_PAUSE_BG_DOWN
        } else if hovered {
            C_PAUSE_BG_HOVER
        } else {
            C_PAUSE_BG
        };
        stroke = Some(if hovered || down {
            egui::Stroke::new(2.0, C_PLAY_HOVER)
        } else {
            egui::Stroke::new(1.5, C_PLAY)
        });
        (fill, C_PAUSE_TEXT, C_PAUSE_BAR)
    } else if down {
        (C_PLAY_DOWN, C_ON_PLAY, C_ON_PLAY)
    } else if hovered {
        (C_PLAY_HOVER, C_ON_PLAY, C_ON_PLAY)
    } else {
        (C_PLAY, C_ON_PLAY, C_ON_PLAY)
    };

    painter.rect_filled(rect, 12.0, fill);
    if let Some(stroke) = stroke {
        painter.rect_stroke(rect, 12.0, stroke, egui::StrokeKind::Inside);
    } else if enabled {
        // 顶部内缩 12px 的高光线
        let y = rect.top() + 1.0;
        painter.line_segment(
            [
                egui::pos2(rect.left() + 12.0, y),
                egui::pos2(rect.right() - 12.0, y),
            ],
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
            ),
        );
    }

    let cx = rect.center().x;
    let cy = rect.center().y;
    let label = if playing { "暂停" } else { "播放" };
    let g = painter.layout_no_wrap(label.to_owned(), egui::FontId::proportional(22.0), fg);
    let icon_w = 22.0;
    let gap = 12.0;
    let x0 = cx - (icon_w + gap + g.size().x) * 0.5;
    if playing {
        for dx in [-7.5_f32, 7.5] {
            painter.rect_filled(
                egui::Rect::from_center_size(
                    egui::pos2(x0 + icon_w * 0.5 + dx, cy),
                    egui::vec2(7.0, 22.0),
                ),
                2.0,
                icon_col,
            );
        }
    } else {
        painter.add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(x0, cy - 11.0),
                egui::pos2(x0, cy + 11.0),
                egui::pos2(x0 + 22.0, cy),
            ],
            icon_col,
            egui::Stroke::NONE,
        ));
    }
    painter.galley(egui::pos2(x0 + icon_w + gap, cy - g.size().y * 0.5), g, fg);
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// 展示厅第二段：大播放键（绿箭头 / 暂停）+ 一行状态（右侧「回到起点」）。
///
/// 不用 `with_layout(right_to_left)`：那会占满剩余高度，「回到起点」会飘到面板底部，
/// 且把展示环境卡挤出可视区。整行高度手算，按钮用 `action_button_at` 钉在右侧。
fn transport(ui: &mut egui::Ui, app: &mut RlRacerApp) {
    let has_selection = app.demo_model.is_some();
    let loaded = app.viewer.policy.is_some();
    let playing = loaded && app.viewer.running;

    if transport_button(ui, playing, has_selection || loaded).clicked() {
        app.demo_toggle_play();
    }

    ui.add_space(8.0);
    let status = match (&app.viewer.policy_source, app.viewer.running) {
        (Some(src), true) => format!("{src} · 正在跑"),
        (Some(src), false) => format!("{src} · 已暂停"),
        (None, _) => {
            if has_selection {
                "已选择 · 点上面的播放键开始".to_owned()
            } else {
                "先选一个模型，再点播放".to_owned()
            }
        }
    };

    let w = ui.available_width();
    let row_h = if loaded { 28.0 } else { 16.0 };
    let (row, _) = ui.allocate_exact_size(egui::vec2(w, row_h), egui::Sense::hover());

    // 左侧状态文字：单行、超宽就省略，宽度上限 W-96（右侧给按钮留位）。
    let text_w = (w - if loaded { 96.0 } else { 0.0 }).max(24.0);
    let mut job =
        egui::text::LayoutJob::simple(status, egui::FontId::proportional(11.0), C_DIM, text_w);
    job.wrap = egui::text::TextWrapping {
        max_width: text_w,
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('…'),
    };
    let galley = ui.painter().layout_job(job);
    ui.painter().galley(
        egui::pos2(row.left(), row.center().y - galley.size().y * 0.5),
        galley,
        C_DIM,
    );

    if loaded {
        let btn = egui::Rect::from_min_size(
            egui::pos2(row.right() - 88.0, row.top()),
            egui::vec2(88.0, 28.0),
        );
        if action_button_at(
            ui,
            btn,
            "demo_reset_origin",
            "回到起点",
            ActionKind::Secondary,
            Icon::None,
            true,
        )
        .clicked()
        {
            app.viewer.reset_stats();
        }
    }
}

// ---------------------------------------------------------------------------
// 环境选择（训练 / 展示共用 UI，作用对象不同）
// ---------------------------------------------------------------------------

fn env_section(ui: &mut egui::Ui, app: &mut RlRacerApp, training: bool, accent: egui::Color32) {
    let title = if training {
        "训练环境"
    } else {
        "展示环境"
    };
    let pending = training && app.env_pending;
    let (changed_track, changed_car) = if training {
        section_card(ui, title, accent, |ui| {
            env_body(ui, &mut app.train_env, pending, accent)
        })
    } else {
        section_card(ui, title, accent, |ui| {
            env_body(ui, &mut app.demo_env, pending, accent)
        })
    };

    if changed_track || changed_car {
        if training {
            app.apply_train_env();
        } else if changed_track {
            app.apply_demo_track();
        } else {
            app.apply_demo_car();
        }
    }
}

/// 环境卡内容。返回（换赛道, 换车型）。
fn env_body(
    ui: &mut egui::Ui,
    env: &mut EnvSel,
    pending: bool,
    accent: egui::Color32,
) -> (bool, bool) {
    let mut changed_track = false;
    let mut changed_car = false;

    // ---- 赛道 ----
    let items: Vec<SegItem> = TrackSel::ALL
        .iter()
        .map(|t| SegItem::new(t.label()))
        .collect();
    let sel = TrackSel::ALL.iter().position(|t| *t == env.track_sel);
    if let Some(i) = segmented(ui, "track", &items, sel, accent) {
        if env.track_sel != TrackSel::ALL[i] {
            env.track_sel = TrackSel::ALL[i];
            changed_track = true;
        }
    }

    if env.track_sel == TrackSel::Random {
        ui.add_space(8.0);
        seed_row(ui, env, &mut changed_track);

        ui.add_space(12.0);
        field_label(ui, "难度");
        let ditems: Vec<SegItem> = Difficulty::ALL
            .iter()
            .map(|d| SegItem::new(d.label()))
            .collect();
        let dsel = Difficulty::ALL.iter().position(|d| *d == env.difficulty);
        if let Some(i) = segmented(ui, "difficulty", &ditems, dsel, accent) {
            if env.difficulty != Difficulty::ALL[i] {
                env.difficulty = Difficulty::ALL[i];
                changed_track = true;
            }
        }
    }

    ui.add_space(12.0);
    field_label(ui, "车型");
    let citems: Vec<SegItem> = CarKind::ALL
        .iter()
        .map(|k| SegItem::with_bar(k.label(), k.color()))
        .collect();
    let csel = CarKind::ALL.iter().position(|k| *k == env.car_kind);
    if let Some(i) = segmented(ui, "car", &citems, csel, accent) {
        if env.car_kind != CarKind::ALL[i] {
            env.car_kind = CarKind::ALL[i];
            env.car_cfg = env.car_kind.to_cfg();
            changed_car = true;
        }
    }

    ui.add_space(8.0);
    if tuning_section(ui, env) {
        changed_car = true;
    }

    ui.add_space(8.0);
    ui.add(
        egui::Label::new(
            egui::RichText::new(format!("当前：{}", env.label()))
                .size(11.0)
                .color(C_DIM),
        )
        .wrap(),
    );
    if pending {
        ui.add_space(2.0);
        ui.add(
            egui::Label::new(
                egui::RichText::new("已下发：训练线程正在重建环境…")
                    .size(11.0)
                    .color(C_WARN),
            )
            .wrap(),
        );
    }
    (changed_track, changed_car)
}

/// seed 输入框（monospace 13）+ 随机按钮（64×28）。
fn seed_row(ui: &mut egui::Ui, env: &mut EnvSel, changed_track: &mut bool) {
    let w = ui.available_width();
    let edit_w = (w - 72.0).max(60.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        let resp = ui.add(
            egui::TextEdit::singleline(&mut env.seed_text)
                .font(egui::FontId::monospace(13.0))
                .margin(egui::Margin {
                    left: 8,
                    right: 8,
                    top: 6,
                    bottom: 6,
                })
                .desired_width(edit_w)
                .min_size(egui::vec2(edit_w, 28.0))
                .clip_text(true),
        );
        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let Ok(seed) = env.seed_text.trim().parse() {
                env.seed = seed;
                *changed_track = true;
            }
        }
        if action_button_sized(
            ui,
            egui::vec2(64.0, 28.0),
            "随机",
            ActionKind::Secondary,
            Icon::None,
            true,
        )
        .clicked()
        {
            env.seed = rand_seed();
            env.seed_text = env.seed.to_string();
            *changed_track = true;
        }
    });
}

/// 调校滑杆：直接改车辆物理参数。返回是否改动。
fn tuning_section(ui: &mut egui::Ui, env: &mut EnvSel) -> bool {
    let mut changed = false;
    collapsible(ui, "调校", |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let c = &mut env.car_cfg;
        let mut kw = c.power / 1000.0;
        if slider_row(ui, "马力", "kW", &mut kw, 30.0..=500.0, 0) {
            c.power = kw * 1000.0;
            changed = true;
        }
        let mut kn = c.torque_force / 1000.0;
        if slider_row(ui, "扭矩", "kN", &mut kn, 2.0..=15.0, 1) {
            c.torque_force = kn * 1000.0;
            changed = true;
        }
        let mut kg = c.mass;
        if slider_row(ui, "车重", "kg", &mut kg, 150.0..=2000.0, 0) {
            c.mass = kg;
            changed = true;
        }
        let mut kmh = c.max_speed * 3.6;
        if slider_row(ui, "极速", "km/h", &mut kmh, 54.0..=180.0, 0) {
            c.max_speed = kmh / 3.6;
            changed = true;
        }
        let mut grip = c.max_lat_accel;
        if slider_row(ui, "抓地", "m/s²", &mut grip, 8.0..=22.0, 1) {
            c.max_lat_accel = grip;
            changed = true;
        }
        let mut kn = c.brake_force / 1000.0;
        if slider_row(ui, "刹车", "kN", &mut kn, 4.0..=25.0, 1) {
            c.brake_force = kn * 1000.0;
            changed = true;
        }
        let mut len = c.length;
        if slider_row(ui, "车长", "m", &mut len, 1.8..=5.5, 2) {
            c.length = len;
            c.wheelbase = (len * 0.63).clamp(1.2, 3.4);
            changed = true;
        }
        let mut steer = c.max_steer;
        if slider_row(ui, "转向角", "", &mut steer, 0.30..=0.60, 2) {
            c.max_steer = steer;
            changed = true;
        }
    });
    changed
}

/// 一行滑杆：上一行标签 + monospace 数值（含弱色单位），下一行滑杆本体。
fn slider_row(
    ui: &mut egui::Ui,
    label: &str,
    unit: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    decimals: usize,
) -> bool {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 16.0), egui::Sense::hover());
    {
        let painter = ui.painter();
        let lg =
            painter.layout_no_wrap(label.to_owned(), egui::FontId::proportional(12.0), C_TEXT2);
        painter.galley(
            egui::pos2(rect.left(), rect.center().y - lg.size().y * 0.5),
            lg,
            C_TEXT2,
        );
        let vg = painter.layout_no_wrap(
            format!("{value:.decimals$}"),
            egui::FontId::monospace(12.0),
            C_TEXT,
        );
        let ug = painter.layout_no_wrap(unit.to_owned(), egui::FontId::proportional(11.0), C_DIM);
        let (vw, vh) = (vg.size().x, vg.size().y);
        let unit_w = if unit.is_empty() {
            0.0
        } else {
            ug.size().x + 3.0
        };
        let vx = rect.right() - unit_w - vw;
        let vy = rect.center().y - vh * 0.5;
        painter.galley(egui::pos2(vx, vy), vg, C_TEXT);
        if !unit.is_empty() {
            let uy = vy + (vh - ug.size().y);
            painter.galley(egui::pos2(rect.right() - ug.size().x, uy), ug, C_DIM);
        }
    }
    ui.add_space(4.0);
    let track_w = (ui.available_width() - 64.0).max(48.0);
    let changed = ui
        .scope(|ui| {
            ui.spacing_mut().slider_width = track_w;
            ui.add(
                egui::Slider::new(value, range)
                    .show_value(true)
                    .trailing_fill(true),
            )
        })
        .inner
        .changed();
    ui.add_space(10.0);
    changed
}

fn reward_note(ui: &mut egui::Ui, r: &sim::reward::RewardConfig) {
    collapsible(ui, "奖励口径", |ui| {
        ui.add(
            egui::Label::new(
                egui::RichText::new(format!(
                    "进度 {:.2}/m · 速度 {:.2} · 走线 {:.2} · 时间 {:.2}/步 · 撞墙 −{:.0} · 完圈 +{:.0}",
                    r.w_progress, r.w_speed, r.w_center, r.w_time, r.r_crash, r.r_lap
                ))
                .size(11.0)
                .color(C_DIM),
            )
            .wrap(),
        );
    });
}

// ---------------------------------------------------------------------------
// 模型列表（缓存 1 秒，避免每帧读目录和 .stats）
// ---------------------------------------------------------------------------

/// 模型文件元信息（从文件名解析，不读文件内容）。
#[derive(Clone)]
pub(crate) struct ModelMeta {
    pub name: String,
    pub algo: AlgoKind,
    pub hidden: usize,
    pub steps: u64,
}

/// 文件名格式：{algo}-h{hidden}-seed{seed}-s{steps}.rec
pub(crate) fn parse_meta(name: &str) -> Option<ModelMeta> {
    let stem = name.strip_suffix(".rec")?;
    let mut parts = stem.split('-');
    let algo = match parts.next()? {
        "dqn" => AlgoKind::Dqn,
        "ppo" => AlgoKind::Ppo,
        _ => return None,
    };
    let hidden: usize = parts.next()?.strip_prefix('h')?.parse().ok()?;
    let _seed: u64 = parts.next()?.strip_prefix("seed")?.parse().ok()?;
    let steps: u64 = parts.next()?.strip_prefix('s')?.parse().ok()?;
    Some(ModelMeta {
        name: name.to_owned(),
        algo,
        hidden,
        steps,
    })
}

/// 存档旁的训练统计（trainer 写出的 `.stats` 文本）。
#[derive(Clone, Copy)]
struct ModelStats {
    episodes: u64,
    laps: u64,
    best_lap: f32,
}

/// 读旁文件；没有就返回 None（旧存档没有）。
fn read_stats(name: &str) -> Option<ModelStats> {
    let stem = name.strip_suffix(".rec").unwrap_or(name);
    let body = std::fs::read_to_string(format!("checkpoints/{stem}.stats")).ok()?;
    let mut s = ModelStats {
        episodes: 0,
        laps: 0,
        best_lap: 0.0,
    };
    for line in body.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "episodes" => s.episodes = v.trim().parse().unwrap_or(0),
            "laps" => s.laps = v.trim().parse().unwrap_or(0),
            "best_lap" => s.best_lap = v.trim().parse().unwrap_or(0.0),
            _ => {}
        }
    }
    Some(s)
}

/// 按钮点击时才真正读取模型文件。
fn load_model(meta: &ModelMeta) -> Option<burn::store::ModuleRecord> {
    burn::store::ModuleRecord::load(format!("checkpoints/{}", meta.name)).ok()
}

pub(crate) fn fmt_steps(s: u64) -> String {
    if s >= 1000 {
        format!("{:.0}k", s as f64 / 1000.0)
    } else {
        s.to_string()
    }
}

/// 一条模型记录（元信息 + 旁文件统计）。
#[derive(Clone)]
struct CachedModel {
    meta: ModelMeta,
    stats: Option<ModelStats>,
}

thread_local! {
    /// (采样时刻, 模型列表)：1 秒内复用，别每帧扫目录。
    static MODEL_CACHE: std::cell::RefCell<Option<(std::time::Instant, Vec<CachedModel>)>> =
        const { std::cell::RefCell::new(None) };
}

fn models_cached() -> Vec<CachedModel> {
    let now = std::time::Instant::now();
    let fresh = MODEL_CACHE.with(|c| {
        c.borrow().as_ref().and_then(|(at, list)| {
            (now.duration_since(*at) < std::time::Duration::from_secs(1)).then(|| list.clone())
        })
    });
    if let Some(list) = fresh {
        return list;
    }
    let list = scan_models_uncached();
    MODEL_CACHE.with(|c| *c.borrow_mut() = Some((now, list.clone())));
    list
}

fn invalidate_model_cache() {
    MODEL_CACHE.with(|c| *c.borrow_mut() = None);
}

fn scan_models_uncached() -> Vec<CachedModel> {
    let mut models: Vec<ModelMeta> = std::fs::read_dir("checkpoints")
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|x| x == "rec").unwrap_or(false))
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_owned()))
                .filter_map(|name| parse_meta(&name))
                .collect()
        })
        .unwrap_or_default();
    // 训练步数多的（通常最新）排前面
    models.sort_by_key(|m| std::cmp::Reverse(m.steps));
    models
        .into_iter()
        .map(|meta| {
            let stats = read_stats(&meta.name);
            CachedModel { meta, stats }
        })
        .collect()
}

/// 只要元信息（app.rs 的测试用这个入口）。
#[allow(dead_code)]
pub(crate) fn scan_models() -> Vec<ModelMeta> {
    models_cached().into_iter().map(|c| c.meta).collect()
}

/// 模型行的点击结果（行本身 + 行内按钮）。
struct ModelRowHit {
    row: egui::Response,
    resume: bool,
    delete: bool,
}

/// 模型行（高 44）：左侧 3px 指示条 + 两行文字 + 右侧状态/按钮。
///
/// - `current`：`Some(播放中)` 表示展示厅当前载入的模型；
/// - `admin`：`Some(可续训)` 表示训练工作台（右侧换成"续训 / 删除"）；
/// - `armed`：删除二次确认态（实心危险色）。
#[allow(clippy::too_many_arguments)]
fn model_row(
    ui: &mut egui::Ui,
    m: &CachedModel,
    selected: bool,
    current: Option<bool>,
    accent: egui::Color32,
    admin: Option<bool>,
    armed: bool,
) -> ModelRowHit {
    let w = ui.available_width();
    let h = 44.0;
    let sense = if admin.is_some() {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, row) = ui.allocate_exact_size(egui::vec2(w, h), sense);
    let painter = ui.painter();

    if selected {
        painter.rect_filled(
            rect,
            6.0,
            egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 28),
        );
        painter.rect_stroke(
            rect,
            6.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 120),
            ),
            egui::StrokeKind::Inside,
        );
    } else if row.hovered() {
        painter.rect_filled(rect, 6.0, C_HOVER);
    } else if admin.is_some() {
        // 管理行静止时补一层细描边，否则行内两个按钮像浮在空中。
        painter.rect_stroke(
            rect,
            6.0,
            egui::Stroke::new(1.0, C_RULE),
            egui::StrokeKind::Inside,
        );
    }

    // 左侧 3px 指示条：播放中绿色 / 选中模式色
    let bar_col = if current == Some(true) {
        Some(C_PLAY)
    } else if selected {
        Some(accent)
    } else {
        None
    };
    if let Some(col) = bar_col {
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.left() + 1.0, rect.top() + 6.0),
                egui::vec2(3.0, rect.height() - 12.0),
            ),
            1.5,
            col,
        );
    }

    // 右侧占位（状态文字或两个按钮）
    let btn_w = 48.0;
    let btn_h = 28.0;
    let right_w = if admin.is_some() {
        6.0 + btn_w + 4.0 + btn_w
    } else if let Some(playing) = current {
        let t = if playing { "播放中" } else { "已暂停" };
        let g = painter.layout_no_wrap(t.to_owned(), egui::FontId::proportional(11.0), C_DIM);
        8.0 + 10.0 + g.size().x
    } else {
        8.0
    };
    let text_w = (rect.width() - 14.0 - right_w - 4.0).max(20.0);

    // 第一行：算法 + 步数
    let line1 = format!("{} · {} 步", m.meta.algo.label(), fmt_steps(m.meta.steps));
    let g1 = fitting_galley(painter, &line1, 13.0, C_TEXT, text_w);
    painter.galley(egui::pos2(rect.left() + 14.0, rect.top() + 8.0), g1, C_TEXT);

    // 第二行：完圈率 / 最佳圈时；没有统计就说明一下模型结构
    if let Some(s) = m.stats.filter(|s| s.episodes > 0) {
        let mut line2 = format!("完圈 {:.0}%", s.laps as f32 / s.episodes as f32 * 100.0);
        if s.best_lap > 0.0 {
            line2.push_str(&format!(" · 最佳 {:.1}s", s.best_lap));
        }
        let g2 = fitting_galley(painter, &line2, 11.0, C_DIM, text_w);
        painter.galley(egui::pos2(rect.left() + 14.0, rect.top() + 25.0), g2, C_DIM);
    } else {
        let line2 = format!("隐藏层 {} · 无统计", m.meta.hidden);
        let g2 = fitting_galley(painter, &line2, 11.0, C_DIM, text_w);
        painter.galley(egui::pos2(rect.left() + 14.0, rect.top() + 25.0), g2, C_DIM);
    }

    let mut resume = false;
    let mut delete = false;

    if let Some(can_resume) = admin {
        let del_rect = egui::Rect::from_center_size(
            egui::pos2(rect.right() - 6.0 - btn_w * 0.5, rect.center().y),
            egui::vec2(btn_w, btn_h),
        );
        let resume_rect = egui::Rect::from_center_size(
            egui::pos2(del_rect.left() - 4.0 - btn_w * 0.5, rect.center().y),
            egui::vec2(btn_w, btn_h),
        );
        resume = action_button_at(
            ui,
            resume_rect,
            ("resume", m.meta.name.as_str()),
            "续训",
            ActionKind::Secondary,
            Icon::None,
            can_resume,
        )
        .clicked();
        let (label, kind) = if armed {
            ("确认删除?", ActionKind::DangerArmed)
        } else {
            ("删除", ActionKind::Secondary)
        };
        delete = action_button_at(
            ui,
            del_rect,
            ("delete", m.meta.name.as_str()),
            label,
            kind,
            Icon::None,
            true,
        )
        .clicked();
    } else if let Some(playing) = current {
        let (text, col) = if playing {
            ("播放中", C_PLAY)
        } else {
            ("已暂停", C_DIM)
        };
        let g = fitting_galley(painter, text, 11.0, col, 60.0);
        let tx = rect.right() - 8.0 - g.size().x;
        painter.galley(egui::pos2(tx, rect.center().y - g.size().y * 0.5), g, col);
        painter.circle_filled(egui::pos2(tx - 10.0, rect.center().y), 3.0, col);
    }

    ModelRowHit {
        row,
        resume,
        delete,
    }
}

/// 训练工作台：存档模型（续训 / 删除）。删除放在这里，
/// 展示厅只负责"选模型 + 播放"。
fn model_admin(ui: &mut egui::Ui, app: &mut RlRacerApp, accent: egui::Color32) {
    let models = models_cached();
    let count = format!("{} 个", models.len());
    let can_resume = app.train_state() == TrainState::Idle;
    section_card_right(ui, "存档模型", accent, Some(&count), None, |ui| {
        if models.is_empty() {
            app.pending_delete = None;
            ui.add(
                egui::Label::new(
                    egui::RichText::new("还没有存档：训练结束会自动存到 checkpoints/")
                        .size(11.0)
                        .color(C_DIM),
                )
                .wrap(),
            );
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("resume_list")
            .max_height(140.0)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                for (i, c) in models.iter().enumerate() {
                    if i > 0 {
                        ui.add_space(4.0);
                    }
                    let armed = app.pending_delete.as_deref() == Some(c.meta.name.as_str());
                    let hit = model_row(ui, c, false, None, accent, Some(can_resume), armed);
                    if hit.resume {
                        if let Some(record) = load_model(&c.meta) {
                            app.train_algo = c.meta.algo;
                            let seed = app.train_env.seed;
                            let resume = crate::messages::ResumeInfo {
                                record,
                                steps: c.meta.steps,
                                replace_file: c.meta.name.clone(),
                            };
                            app.start_training(seed, Some(resume));
                        }
                    }
                    if hit.delete {
                        if armed {
                            // 真删：模型文件 + 统计旁文件
                            let _ = std::fs::remove_file(format!("checkpoints/{}", c.meta.name));
                            let stem = c.meta.name.strip_suffix(".rec").unwrap_or(&c.meta.name);
                            let _ = std::fs::remove_file(format!("checkpoints/{stem}.stats"));
                            app.pending_delete = None;
                            invalidate_model_cache();
                            // 展示厅里如果正引用这个文件，清掉引用（内存里的策略继续跑）
                            if app.demo_model.as_deref() == Some(c.meta.name.as_str()) {
                                app.demo_model = None;
                            }
                            if app.viewer.policy_file.as_deref() == Some(c.meta.name.as_str()) {
                                app.viewer.policy_file = None;
                            }
                        } else {
                            app.pending_delete = Some(c.meta.name.clone());
                        }
                    }
                }
            });
    });
}

pub fn sim_config_for(sel: TrackSel, difficulty: Difficulty, car: &CarConfig) -> SimConfig {
    let mut cfg = SimConfig::default();
    cfg.track = match sel {
        TrackSel::Random => difficulty.to_cfg(),
        TrackSel::Real(t) => t.to_track_gen_config(),
    };
    cfg.car = *car;
    cfg
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
        ^ (std::process::id() as u64)
}
