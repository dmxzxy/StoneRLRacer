//! 赛道画布：接收一份 [`Scene`]（来自训练镜像或展示厅），世界坐标 → 屏幕坐标绘制。

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2 as EVec2};

use sim::car::CarConfig;
use sim::episode::EpisodeResult;
use sim::math::Vec2;
use sim::observation::{ray_angle, MAX_RAY_DIST, N_RAYS};
use sim::track::Track;

use crate::messages::Trace;

// ---- 配色（低饱和、克制）----
const GRASS: Color32 = Color32::from_rgb(38, 44, 40);
const ASPHALT: Color32 = Color32::from_rgb(52, 55, 62);
const CURB_RED: Color32 = Color32::from_rgb(170, 60, 55);
const CURB_WHITE: Color32 = Color32::from_rgb(200, 202, 206);
const CENTERLINE: Color32 = Color32::from_rgb(150, 138, 92);
const HUD_TEXT: Color32 = Color32::from_rgb(220, 223, 230);
const HUD_DIM: Color32 = Color32::from_rgb(140, 145, 155);
const TRAIL: Color32 = Color32::from_rgb(235, 160, 60);
const SHADOW: Color32 = Color32::from_black_alpha(55);
const WHEEL: Color32 = Color32::from_rgb(24, 24, 28);

/// 画布需要的全部数据。训练镜像和展示厅各自组装一份，绘制逻辑共用。
pub struct Scene<'a> {
    pub track: &'a Track,
    /// None = 还没有车可画（训练镜像尚未收到第一帧）。
    pub car_pos: Option<Vec2>,
    pub car_heading: f32,
    pub car_speed: f32,
    pub car_cfg: &'a CarConfig,
    pub car_color: [u8; 3],
    /// 归一化射线距离（d / MAX_RAY_DIST），None = 不画。
    pub rays: Option<[f32; N_RAYS]>,
    pub trail: &'a [Vec2],
    /// 拖影画成散点（训练镜像的帧间隔远大于物理步长，连线会画成"乱麻"）。
    pub trail_dots: bool,
    /// 局末轨迹叠加（旧 → 新）。
    pub traces: &'a std::collections::VecDeque<Trace>,
    /// 左上角徽标：文字 + 颜色。
    pub badge: (&'a str, Color32),
    /// 状态行（进度 / 圈数 / 最佳…）：（标签, 数值）对。
    /// 标签画 12px 弱色，数值画 13px monospace 主色。
    pub status_line: Vec<(&'static str, String)>,
    /// 当前局结果（None = 进行中）。
    pub result: Option<EpisodeResult>,
    /// 车速显示（km/h）。
    pub speed_kmh: f32,
    /// 策略解读面板（只对"有策略"的场景显示）。
    pub values: Option<&'a rl::policy::ActionValues>,
}

/// 一帧用的屏幕变换（由 [`crate::view::cam::CamState`] 生成）。
struct Camera {
    cam: crate::view::cam::CamState,
    rect: Rect,
}

impl Camera {
    fn to_screen(&self, p: Vec2) -> Pos2 {
        self.cam.to_screen(&self.rect, p)
    }

    fn stroke_w(&self, meters: f32) -> f32 {
        (meters * self.cam.zoom()).max(1.0)
    }
}

/// 绘制画布（含镜头：自由拖动/滚轮缩放、跟随车、右下角镜头控件）。
pub fn draw(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    scene: &Scene<'_>,
    cam: &mut crate::view::cam::CamState,
    accent: Color32,
) {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    cam.ensure_fit(scene.track, &rect);
    // 追随阈值：半个视口。超过就直贴（训练镜像一帧能跳一百多米）
    let snap_world = 0.5 * rect.width().min(rect.height()) / cam.zoom();
    cam.follow(scene.car_pos, snap_world);
    cam.handle_input(ui, &response, &rect, scene.track);
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 8.0, GRASS);
    painter.rect_stroke(
        rect,
        8.0,
        Stroke::new(1.0, Color32::from_rgb(52, 56, 69)),
        egui::StrokeKind::Inside,
    );

    let track = scene.track;
    let camera = Camera { cam: *cam, rect };
    let m = track.samples.len();

    // ---- 路面 ----
    let mut road: Vec<Pos2> = track
        .samples
        .iter()
        .map(|s| camera.to_screen(s.pos))
        .collect();
    road.push(road[0]);
    painter.line(road, Stroke::new(camera.stroke_w(track.width), ASPHALT));

    // ---- 中心虚线 ----
    for i in (0..m).step_by(2) {
        let a = camera.to_screen(track.samples[i].pos);
        let b = camera.to_screen(track.samples[(i + 1) % m].pos);
        painter.line_segment([a, b], Stroke::new(1.0, CENTERLINE));
    }

    // ---- 路缘石（细，红白相间）----
    let (left, right) = track.boundary_polylines();
    draw_curbs(&painter, &camera, left);
    draw_curbs(&painter, &camera, right);

    // ---- 终点线：黑白格 ----
    draw_start_line(&painter, &camera, &track.samples[0]);

    // ---- 局末轨迹叠加（越新越亮，最新一条按车速着色）----
    draw_traces(&painter, &camera, scene.traces);

    // ---- 雷达射线（近红远绿，细线）----
    if let Some(rays) = scene.rays {
        for (i, norm) in rays.iter().enumerate() {
            let ang = scene.car_heading + ray_angle(i);
            let dir = Vec2::from_angle(ang);
            let d = (norm.clamp(0.0, 1.0) * MAX_RAY_DIST).max(0.0);
            let t = norm.clamp(0.0, 1.0);
            let col = Color32::from_rgb(
                (200.0 + (70.0 - 200.0) * t) as u8,
                (80.0 + (160.0 - 80.0) * t) as u8,
                80,
            );
            let Some(pos) = scene.car_pos else { continue };
            let a = camera.to_screen(pos);
            let b = camera.to_screen(pos + dir * d);
            painter.line_segment([a, b], Stroke::new(1.0, col));
        }
    }

    // ---- 行车拖影 ----
    let n = scene.trail.len();
    if scene.trail_dots {
        // 训练镜像：帧与帧之间隔了几十步，画点不画线
        for (i, p) in scene.trail.iter().enumerate() {
            let age = (i as f32) / n.max(1) as f32;
            let alpha = (25.0 + 105.0 * age) as u8;
            painter.circle_filled(
                camera.to_screen(*p),
                1.6,
                Color32::from_rgba_unmultiplied(TRAIL.r(), TRAIL.g(), TRAIL.b(), alpha),
            );
        }
    } else {
        for (i, w) in scene.trail.windows(2).enumerate() {
            let age = (i as f32) / n.max(1) as f32;
            let alpha = (age * 80.0) as u8;
            let col = Color32::from_rgba_premultiplied(
                (TRAIL.r() as f32 * age as f32) as u8,
                (TRAIL.g() as f32 * age as f32) as u8,
                (TRAIL.b() as f32 * age as f32) as u8,
                alpha,
            );
            let a = camera.to_screen(w[0]);
            let b = camera.to_screen(w[1]);
            painter.line_segment([a, b], Stroke::new(2.0 * age + 0.5, col));
        }
    }

    // ---- 赛车 ----
    if let Some(pos) = scene.car_pos.filter(|p| p.x.is_finite() && p.y.is_finite()) {
        draw_car(
            &painter,
            &camera,
            pos,
            scene.car_heading,
            scene.car_cfg,
            scene.car_color,
        );
    }

    // ---- HUD ----
    draw_hud(&painter, &rect, scene);
    crate::view::cam::draw_controls(ui, &rect, cam, accent);
    if let Some(values) = scene.values {
        draw_policy_panel(&painter, &rect, values);
    }
}

/// 红白路缘石：每 3 个采样段交替一色。
fn draw_curbs(painter: &egui::Painter, camera: &Camera, pts: &[Vec2]) {
    for (i, w) in pts.windows(2).enumerate() {
        let col = if (i / 3) % 2 == 0 {
            CURB_RED
        } else {
            CURB_WHITE
        };
        painter.line_segment(
            [camera.to_screen(w[0]), camera.to_screen(w[1])],
            Stroke::new(camera.stroke_w(0.9), col),
        );
    }
}

/// 黑白格终点线：2 行 × 8 列。
fn draw_start_line(painter: &egui::Painter, camera: &Camera, s0: &sim::track::TrackSample) {
    let cols = 8;
    let cell_l = 1.2f32;
    let cell_w = (s0.half_width * 2.0) / cols as f32;
    for r in 0..2 {
        for c in 0..cols {
            let base = s0.pos - s0.tangent * (r as f32 * cell_l) - s0.normal * s0.half_width
                + s0.normal * (c as f32 * cell_w);
            let quad = [
                base,
                base + s0.tangent * cell_l,
                base + s0.tangent * cell_l + s0.normal * cell_w,
                base + s0.normal * cell_w,
            ];
            let screen: Vec<Pos2> = quad.iter().map(|p| camera.to_screen(*p)).collect();
            let col = if (r + c) % 2 == 0 {
                Color32::from_rgb(20, 20, 24)
            } else {
                Color32::from_rgb(225, 225, 230)
            };
            painter.add(egui::Shape::convex_polygon(screen, col, Stroke::NONE));
        }
    }
}

/// 局末轨迹：旧淡新亮，完圈绿 / 撞墙红；最新一条按车速着色。
fn draw_traces(
    painter: &egui::Painter,
    camera: &Camera,
    traces: &std::collections::VecDeque<Trace>,
) {
    let n = traces.len();
    for (i, t) in traces.iter().enumerate() {
        let age = (i + 1) as f32 / n as f32; // 1 = 最新
        let (r, g, b) = if t.lap {
            (120, 200, 130)
        } else {
            (205, 100, 92)
        };
        let alpha = (12.0 + 62.0 * age * age) as u8;
        let pts: Vec<Pos2> = t
            .points
            .iter()
            .map(|p| camera.to_screen(Vec2::new(p.x, p.y)))
            .collect();
        if pts.len() < 2 {
            continue;
        }
        if i + 1 == n {
            // 最新一条：按车速分段着色（慢=蓝，快=橙）
            for w in 0..pts.len() - 1 {
                let s = (t.points[w].speed / 40.0).clamp(0.0, 1.0);
                let col = Color32::from_rgb(
                    (70.0 + 160.0 * s) as u8,
                    (110.0 + 50.0 * s) as u8,
                    (170.0 - 110.0 * s) as u8,
                );
                painter.line_segment(
                    [pts[w], pts[w + 1]],
                    Stroke::new(1.4, col.gamma_multiply(0.6)),
                );
            }
        } else {
            painter.add(egui::Shape::line(
                pts,
                Stroke::new(
                    1.0 + 0.6 * age,
                    Color32::from_rgba_unmultiplied(r, g, b, alpha),
                ),
            ));
        }
    }
}

/// 赛车：阴影 + 四轮 + 箭形车身 + 座舱 + 尾翼。
fn draw_car(
    painter: &egui::Painter,
    camera: &Camera,
    pos: Vec2,
    heading: f32,
    cfg: &CarConfig,
    color: [u8; 3],
) {
    let fwd = Vec2::from_angle(heading);
    let side = fwd.perp();
    let l = cfg.length;
    let w = cfg.half_width;
    let body = Color32::from_rgb(color[0], color[1], color[2]);
    let body_dark = Color32::from_rgb(
        (color[0] as f32 * 0.55) as u8,
        (color[1] as f32 * 0.55) as u8,
        (color[2] as f32 * 0.55) as u8,
    );

    // 阴影
    let shadow_off = Vec2::new(0.9, -0.9);
    let corners = car_corners(pos, fwd, side, l, w);
    let shadow: Vec<Pos2> = corners
        .iter()
        .map(|p| camera.to_screen(*p + shadow_off))
        .collect();
    painter.add(egui::Shape::convex_polygon(shadow, SHADOW, Stroke::NONE));

    // 四轮
    let wheel_pos = [
        fwd * l * 0.30 + side * (w + 0.12),
        fwd * l * 0.30 - side * (w + 0.12),
        -fwd * l * 0.32 + side * (w + 0.12),
        -fwd * l * 0.32 - side * (w + 0.12),
    ];
    for wp in wheel_pos {
        let c = pos + wp;
        let q = [
            c + fwd * 0.45 + side * 0.22,
            c + fwd * 0.45 - side * 0.22,
            c - fwd * 0.45 - side * 0.22,
            c - fwd * 0.45 + side * 0.22,
        ];
        let screen: Vec<Pos2> = q.iter().map(|p| camera.to_screen(*p)).collect();
        painter.add(egui::Shape::convex_polygon(screen, WHEEL, Stroke::NONE));
    }

    // 车身：箭形
    let body_poly = [
        pos + fwd * l * 0.55,
        pos + fwd * l * 0.18 + side * w * 0.62,
        pos + fwd * l * 0.05 + side * w,
        pos - fwd * l * 0.30 + side * w,
        pos - fwd * l * 0.45 + side * w * 0.85,
        pos - fwd * l * 0.45 - side * w * 0.85,
        pos - fwd * l * 0.30 - side * w,
        pos + fwd * l * 0.05 - side * w,
        pos + fwd * l * 0.18 - side * w * 0.62,
    ];
    let screen: Vec<Pos2> = body_poly.iter().map(|p| camera.to_screen(*p)).collect();
    painter.add(egui::Shape::convex_polygon(
        screen,
        body,
        Stroke::new(1.0, body_dark),
    ));

    // 座舱
    let cockpit = [
        pos + fwd * l * 0.12 + side * w * 0.4,
        pos + fwd * l * 0.12 - side * w * 0.4,
        pos - fwd * l * 0.12 - side * w * 0.4,
        pos - fwd * l * 0.12 + side * w * 0.4,
    ];
    let screen: Vec<Pos2> = cockpit.iter().map(|p| camera.to_screen(*p)).collect();
    painter.add(egui::Shape::convex_polygon(screen, body_dark, Stroke::NONE));

    // 尾翼
    let wing_a = camera.to_screen(pos - fwd * l * 0.48 + side * w * 0.95);
    let wing_b = camera.to_screen(pos - fwd * l * 0.48 - side * w * 0.95);
    painter.line_segment(
        [wing_a, wing_b],
        Stroke::new(camera.stroke_w(0.35), body_dark),
    );
}

fn car_corners(pos: Vec2, fwd: Vec2, side: Vec2, l: f32, w: f32) -> [Vec2; 4] {
    [
        pos + fwd * l * 0.55 + side * (w + 0.3),
        pos + fwd * l * 0.55 - side * (w + 0.3),
        pos - fwd * l * 0.5 - side * (w + 0.3),
        pos - fwd * l * 0.5 + side * (w + 0.3),
    ]
}

/// 圆角小徽标：最左 6px 状态圆点 + 文字。返回整块 rect，方便下游链式锚点。
fn draw_pill(painter: &egui::Painter, pos: Pos2, text: &str, color: Color32) -> Rect {
    let font = egui::FontId::proportional(12.0);
    let galley = painter.layout_no_wrap(text.to_owned(), font, color);
    let pad = EVec2::new(9.0, 5.0);
    let dot = 6.0;
    let gap = 6.0;
    let size = EVec2::new(galley.size().x + dot + gap, galley.size().y) + pad * 2.0;
    let rect = Rect::from_min_size(pos, size);
    let radius = rect.height() * 0.5;
    painter.rect_filled(rect, radius, Color32::from_black_alpha(150));
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0, color.gamma_multiply(0.5)),
        egui::StrokeKind::Inside,
    );
    painter.circle_filled(
        egui::pos2(rect.left() + pad.x + dot * 0.5, rect.center().y),
        dot * 0.5,
        color,
    );
    let th = galley.size().y;
    painter.galley(
        egui::pos2(rect.left() + pad.x + dot + gap, rect.center().y - th * 0.5),
        galley,
        color,
    );
    rect
}

/// HUD：左上徽标 + 状态行 + 结果徽标（锚点链式推算），右上大速度数字。
fn draw_hud(painter: &egui::Painter, rect: &Rect, scene: &Scene<'_>) {
    let pill = draw_pill(
        painter,
        rect.left_top() + EVec2::new(12.0, 12.0),
        scene.badge.0,
        scene.badge.1,
    );

    // ---- 状态行：徽标左侧 +2、下方 8；标签 12px 弱色，数值 13px monospace 主色 ----
    let mut x = pill.left() + 2.0;
    let base_y = pill.bottom() + 8.0;
    let line_h = 18.0;
    for (i, (label, value)) in scene.status_line.iter().enumerate() {
        if i > 0 {
            let sep =
                painter.layout_no_wrap("·".to_owned(), egui::FontId::proportional(12.0), HUD_DIM);
            let h = sep.size().y;
            painter.galley(
                egui::pos2(x, base_y + (line_h - h) * 0.5),
                sep.clone(),
                HUD_DIM,
            );
            x += sep.size().x + 6.0;
        }
        if !label.is_empty() {
            let lg = painter.layout_no_wrap(
                (*label).to_owned(),
                egui::FontId::proportional(12.0),
                HUD_DIM,
            );
            let h = lg.size().y;
            painter.galley(
                egui::pos2(x, base_y + (line_h - h) * 0.5),
                lg.clone(),
                HUD_DIM,
            );
            x += lg.size().x + 4.0;
        }
        let vg = painter.layout_no_wrap(value.clone(), egui::FontId::monospace(13.0), HUD_TEXT);
        let h = vg.size().y;
        painter.galley(
            egui::pos2(x, base_y + (line_h - h) * 0.5),
            vg.clone(),
            HUD_TEXT,
        );
        x += vg.size().x;
    }
    let status_bottom = base_y + line_h;

    // ---- 结果徽标：状态行下方 6、与徽标左对齐 ----
    if let Some(result) = scene.result {
        if let Some((col, txt)) = result_style(result) {
            let g = painter.layout_no_wrap(txt.to_owned(), egui::FontId::proportional(12.0), col);
            let pad = EVec2::new(9.0, 5.0);
            let gw = g.size().x;
            let gh = g.size().y;
            let r = Rect::from_min_size(
                egui::pos2(pill.left(), status_bottom + 6.0),
                EVec2::new(gw, gh) + pad * 2.0,
            );
            let radius = r.height() * 0.5;
            painter.rect_filled(
                r,
                radius,
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 40),
            );
            painter.rect_stroke(
                r,
                radius,
                Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 140),
                ),
                egui::StrokeKind::Inside,
            );
            painter.galley(r.min + pad, g, col);
        }
    }

    // ---- 右上大速度：monospace 34 + 弱色 km/h，外包半透明底板 ----
    // 底板顶边与徽标对齐（top+12）：num_pos.y = top + 20，外扩 8px。
    let num = painter.layout_no_wrap(
        format!("{:.0}", scene.speed_kmh),
        egui::FontId::monospace(34.0),
        HUD_TEXT,
    );
    let unit = painter.layout_no_wrap("km/h".to_owned(), egui::FontId::proportional(11.0), HUD_DIM);
    let right = rect.right() - 22.0;
    let (nw, nh) = (num.size().x, num.size().y);
    let (uw, uh) = (unit.size().x, unit.size().y);
    let num_pos = egui::pos2(right - nw, rect.top() + 20.0);
    let unit_pos = egui::pos2(right - uw, num_pos.y + nh + 2.0);
    let content = Rect::from_min_max(num_pos, egui::pos2(right, unit_pos.y + uh));
    painter.rect_filled(
        content.expand2(EVec2::new(10.0, 8.0)),
        10.0,
        Color32::from_black_alpha(110),
    );
    painter.galley(num_pos, num, HUD_TEXT);
    painter.galley(unit_pos, unit, HUD_DIM);
}

/// 结果徽标的颜色与文字（Running 时没有徽标）。
fn result_style(result: EpisodeResult) -> Option<(Color32, &'static str)> {
    match result {
        EpisodeResult::LapDone => Some((Color32::from_rgb(110, 200, 120), "完圈")),
        EpisodeResult::Crashed => Some((Color32::from_rgb(220, 100, 92), "撞墙")),
        EpisodeResult::Timeout => Some((Color32::from_rgb(220, 185, 95), "超时")),
        EpisodeResult::Running => None,
    }
}

/// 左下角：策略此刻对每个动作的判断。
/// DQN 画 3×3 Q 值表（行=油门 +1/0/−1，列=转向 −/0/+），PPO 画 μ/σ 与状态价值。
fn draw_policy_panel(painter: &egui::Painter, rect: &Rect, v: &rl::policy::ActionValues) {
    let w = 178.0;
    let h = if v.q.is_some() { 156.0 } else { 96.0 };
    if rect.height() < h + 24.0 {
        return;
    }
    let panel = Rect::from_min_size(
        egui::pos2(rect.left() + 12.0, rect.bottom() - 12.0 - h),
        egui::vec2(w, h),
    );
    painter.rect_filled(panel, 8.0, Color32::from_black_alpha(150));
    painter.rect_stroke(
        panel,
        8.0,
        Stroke::new(1.0, Color32::from_white_alpha(22)),
        egui::StrokeKind::Inside,
    );
    let title = if v.q.is_some() {
        "策略判断：各动作 Q 值"
    } else {
        "策略输出"
    };
    painter.text(
        panel.left_top() + EVec2::new(10.0, 8.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(11.0),
        HUD_DIM,
    );

    if let Some(q) = &v.q {
        let best = v.greedy_index().unwrap_or(0);
        let cw = 50.0;
        let ch = 26.0;
        let gap = 4.0;
        let x0 = panel.left() + 10.0;
        let y0 = panel.top() + 26.0;
        for row in 0..3 {
            for col in 0..3 {
                // 行 0 = 油门 +1（索引 +1），行 2 = 油门 −1
                let throttle_idx = 2 - row;
                let idx = col * 3 + throttle_idx;
                let cell = Rect::from_min_size(
                    egui::pos2(x0 + col as f32 * (cw + gap), y0 + row as f32 * (ch + gap)),
                    egui::vec2(cw, ch),
                );
                let is_best = idx == best;
                let bg = if is_best {
                    Color32::from_rgb(120, 170, 220) // 展示用冷蓝，不用训练琥珀
                } else {
                    Color32::from_black_alpha(120)
                };
                painter.rect_filled(cell, 4.0, bg);
                let fg = if is_best {
                    Color32::from_rgb(11, 22, 32) // #0B1620
                } else {
                    HUD_TEXT
                };
                painter.text(
                    cell.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.1}", q.get(idx).copied().unwrap_or(f32::NAN)),
                    egui::FontId::monospace(11.0),
                    fg,
                );
            }
        }
        painter.text(
            panel.left_bottom() + EVec2::new(10.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "行=油门 +1/0/−1  列=转向 −/0/+",
            egui::FontId::proportional(11.0),
            HUD_DIM,
        );
        return;
    }

    // PPO：均值与标准差 + 状态价值
    let mut y = panel.top() + 28.0;
    if let (Some(mu), Some(std)) = (v.mu.as_ref(), v.std.as_ref()) {
        for (i, label) in ["转向", "油门"].iter().enumerate() {
            painter.text(
                egui::pos2(panel.left() + 10.0, y),
                egui::Align2::LEFT_TOP,
                format!("{label} μ {:+.2}  σ {:.2}", mu[i], std[i]),
                egui::FontId::proportional(11.0),
                HUD_TEXT,
            );
            // 均值条：中线 0，向右为正
            let bar = Rect::from_min_size(
                egui::pos2(panel.left() + 10.0, y + 15.0),
                egui::vec2(w - 20.0, 5.0),
            );
            painter.rect_filled(bar, 2.0, Color32::from_black_alpha(120));
            let mid = bar.center().x;
            let val = mu[i].clamp(-1.0, 1.0);
            let end = mid + val * (bar.width() * 0.5);
            painter.rect_filled(
                Rect::from_min_max(
                    egui::pos2(mid.min(end), bar.top()),
                    egui::pos2(mid.max(end), bar.bottom()),
                ),
                2.0,
                Color32::from_rgb(120, 170, 220),
            );
            y += 34.0;
        }
    }
    if let Some(val) = v.value {
        painter.text(
            egui::pos2(panel.left() + 10.0, y),
            egui::Align2::LEFT_TOP,
            format!("状态价值 V {val:.1}"),
            egui::FontId::proportional(11.0),
            HUD_DIM,
        );
    }
}
