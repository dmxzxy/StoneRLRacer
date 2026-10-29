//! 画布镜头：自由（拖动平移 + 滚轮缩放）与跟随（锁车 + 可调缩放）。
//!
//! 两种模式各有自己的缩放档位：切回去时不会把对方的手感改掉。
//! 单位统一用 `px/m`（屏幕像素 / 世界米），避免和场景缩放混淆。

use egui::{Color32, Pos2, Rect, Stroke, Vec2 as EVec2};
use sim::math::Vec2;
use sim::track::Track;

/// 镜头模式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamMode {
    /// 自由：整条赛道随便看，拖动平移、滚轮缩放。
    Free,
    /// 跟随：车始终在画面中心，缩放可调。
    Follow,
}

impl CamMode {
    pub fn index(self) -> usize {
        match self {
            CamMode::Free => 0,
            CamMode::Follow => 1,
        }
    }
    pub fn from_index(i: usize) -> Self {
        if i == 1 { CamMode::Follow } else { CamMode::Free }
    }
}

/// 缩放下限 / 上限（px/m）。上限 30 时 16m 宽赛道占 480px，够贴近看姿态。
pub const ZOOM_MIN: f32 = 0.30;
pub const ZOOM_MAX: f32 = 30.0;
/// 跟随模式初始缩放（px/m）。
const FOLLOW_ZOOM_DEFAULT: f32 = 7.0;

#[derive(Clone, Copy, Debug)]
pub struct CamState {
    mode: CamMode,
    /// 画面中心（世界坐标）。
    center: Vec2,
    /// 自由模式缩放。
    free_zoom: f32,
    /// 跟随模式缩放。
    follow_zoom: f32,
    /// 是否已按首帧尺寸对准过。
    fitted: bool,
}

impl Default for CamState {
    fn default() -> Self {
        Self {
            mode: CamMode::Free,
            center: Vec2::ZERO,
            free_zoom: 1.0,
            follow_zoom: FOLLOW_ZOOM_DEFAULT,
            fitted: false,
        }
    }
}

impl CamState {
    pub fn mode(&self) -> CamMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: CamMode) {
        // 两种模式各自保留缩放；中心点沿用当前值，跟随会从当前视角平滑贴上去
        self.mode = mode;
    }

    /// 当前生效的缩放（px/m）。
    pub fn zoom(&self) -> f32 {
        match self.mode {
            CamMode::Free => self.free_zoom,
            CamMode::Follow => self.follow_zoom,
        }
    }

    /// 设置当前模式的缩放（会被 clamp）。
    pub fn set_zoom(&mut self, z: f32) {
        let z = if z.is_finite() {
            z.clamp(ZOOM_MIN, ZOOM_MAX)
        } else {
            self.zoom()
        };
        match self.mode {
            CamMode::Free => self.free_zoom = z,
            CamMode::Follow => self.follow_zoom = z,
        }
    }

    /// 换赛道 / 换环境后调用：下一帧重新对准整条赛道。
    pub fn reset(&mut self) {
        self.fitted = false;
    }

    /// 自由模式"整条赛道铺满"的缩放。
    fn fit_zoom(track: &Track, rect: &Rect) -> f32 {
        let hw = track.width * 0.5;
        let w = (track.bbox_max.x - track.bbox_min.x + 2.0 * hw).max(1.0);
        let h = (track.bbox_max.y - track.bbox_min.y + 2.0 * hw).max(1.0);
        ((rect.width() / w).min(rect.height() / h) * 0.92).clamp(ZOOM_MIN, ZOOM_MAX)
    }

    /// 首帧（或换赛道后）对准整条赛道：自由模式铺满，跟随模式仍有默认缩放。
    pub fn ensure_fit(&mut self, track: &Track, rect: &Rect) {
        if self.fitted {
            return;
        }
        self.center = (track.bbox_min + track.bbox_max) * 0.5;
        self.free_zoom = Self::fit_zoom(track, rect);
        if self.follow_zoom <= ZOOM_MIN + 1e-3 {
            self.follow_zoom = FOLLOW_ZOOM_DEFAULT;
        }
        self.fitted = true;
    }

    /// 世界 → 屏幕。
    pub fn to_screen(&self, rect: &Rect, p: Vec2) -> Pos2 {
        let c = rect.center();
        let z = self.zoom();
        Pos2::new(
            c.x + (p.x - self.center.x) * z,
            c.y - (p.y - self.center.y) * z,
        )
    }

    /// 屏幕 → 世界。
    pub fn to_world(&self, rect: &Rect, p: Pos2) -> Vec2 {
        let c = rect.center();
        let z = self.zoom();
        Vec2::new(
            self.center.x + (p.x - c.x) / z,
            self.center.y - (p.y - c.y) / z,
        )
    }

    /// 跟随模式：镜头贴住车。
    ///
    /// - 正常速度（展示厅）：每帧收敛 22%，平滑不抖；
    /// - 训练镜像一秒跑几千步，车每帧能跳一百多米，插值永远追不上，
    ///   所以落后超过 `snap_world`（半个视口）就直接贴上去，变成"追车镜头"。
    pub fn follow(&mut self, car: Option<Vec2>, snap_world: f32) {
        if self.mode != CamMode::Follow {
            return;
        }
        let Some(car) = car else { return };
        if !car.x.is_finite() || !car.y.is_finite() {
            return;
        }
        let dx = car.x - self.center.x;
        let dy = car.y - self.center.y;
        if (dx * dx + dy * dy).sqrt() > snap_world.max(1.0) {
            self.center = car;
        } else {
            self.center.x += dx * 0.22;
            self.center.y += dy * 0.22;
        }
    }

    /// 处理画布上的鼠标输入：拖动平移（自由）、滚轮/捏合缩放、双击回正。
    pub fn handle_input(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        rect: &Rect,
        track: &Track,
    ) {
        if !resp.hovered() {
            return;
        }

        // ---- 缩放：滚轮 / 触控板捏合 ----
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let mut factor = 1.0f32;
        if scroll.abs() > 0.0 {
            factor *= (scroll * 0.0015).exp();
        }
        if (pinch - 1.0).abs() > 1e-3 {
            factor *= pinch;
        }
        if (factor - 1.0).abs() > 1e-4 {
            let old = self.zoom();
            let new = (old * factor).clamp(ZOOM_MIN, ZOOM_MAX);
            // 自由模式：以光标为锚点缩放，视线不跑
            if self.mode == CamMode::Free {
                if let Some(ptr) = ui.input(|i| i.pointer.hover_pos()) {
                    let before = self.to_world(rect, ptr);
                    self.set_zoom(new);
                    let after = self.to_world(rect, ptr);
                    self.center.x += before.x - after.x;
                    self.center.y += before.y - after.y;
                } else {
                    self.set_zoom(new);
                }
            } else {
                self.set_zoom(new);
            }
        }

        // ---- 平移：只有自由模式拖得动 ----
        if self.mode == CamMode::Free && resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            let z = self.zoom();
            self.center.x -= d.x / z;
            self.center.y += d.y / z;
        }

        // ---- 双击回正 ----
        if resp.double_clicked() {
            self.center = (track.bbox_min + track.bbox_max) * 0.5;
            if self.mode == CamMode::Free {
                self.free_zoom = Self::fit_zoom(track, rect);
            }
        }
    }

    /// 把镜头拉回赛道（不改模式）。
    pub fn recentre(&mut self, track: &Track, rect: &Rect) {
        self.center = (track.bbox_min + track.bbox_max) * 0.5;
        self.free_zoom = Self::fit_zoom(track, rect);
    }
}

/// 画布镜头控件（右下角浮层）：模式分段 + 缩放滑杆 + 一行操作提示。
pub fn draw_controls(
    ui: &mut egui::Ui,
    rect: &Rect,
    cam: &mut CamState,
    accent: egui::Color32,
) {
    const PANEL_W: f32 = 216.0;
    const PANEL_H: f32 = 92.0;
    if rect.width() < PANEL_W + 120.0 || rect.height() < PANEL_H + 60.0 {
        return; // 画布太小就不放浮层，别挡赛道
    }
    let panel = Rect::from_min_size(
        egui::pos2(rect.right() - 12.0 - PANEL_W, rect.bottom() - 12.0 - PANEL_H),
        EVec2::new(PANEL_W, PANEL_H),
    );
    let painter = ui.painter();
    painter.rect_filled(panel, 8.0, Color32::from_black_alpha(165));
    painter.rect_stroke(
        panel,
        8.0,
        Stroke::new(1.0, Color32::from_white_alpha(26)),
        egui::StrokeKind::Inside,
    );

    // ---- 模式分段 ----
    let seg_rect = Rect::from_min_size(
        egui::pos2(panel.left() + 8.0, panel.top() + 8.0),
        EVec2::new(PANEL_W - 16.0, 28.0),
    );
    let mut clicked = None;
    ui.scope_builder(egui::UiBuilder::new().max_rect(seg_rect), |ui| {
        let items = [
            crate::view::controls::SegItem::new("自由"),
            crate::view::controls::SegItem::new("跟随"),
        ];
        clicked = crate::view::controls::segmented(
            ui,
            "cam_mode",
            &items,
            Some(cam.mode().index()),
            accent,
        );
    });
    if let Some(i) = clicked {
        cam.set_mode(CamMode::from_index(i));
    }

    // ---- 缩放滑杆 ----
    let row = Rect::from_min_size(
        egui::pos2(panel.left() + 10.0, seg_rect.bottom() + 8.0),
        EVec2::new(PANEL_W - 20.0, 24.0),
    );
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(row)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
        |ui| {
            ui.label(
                egui::RichText::new("缩放")
                    .size(11.0)
                    .color(crate::view::controls::C_DIM),
            );
            ui.spacing_mut().slider_width = 120.0;
            let mut z = cam.zoom();
            if ui
                .add(egui::Slider::new(&mut z, ZOOM_MIN..=ZOOM_MAX).show_value(false))
                .changed()
            {
                cam.set_zoom(z);
            }
        },
    );

    // ---- 提示 ----
    let hint = match cam.mode() {
        CamMode::Free => "拖动平移 · 滚轮缩放 · 双击回正",
        CamMode::Follow => "车居中跟随 · 滚轮/滑杆调缩放",
    };
    ui.painter().text(
        egui::pos2(panel.left() + 10.0, panel.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        hint,
        egui::FontId::proportional(10.0),
        crate::view::controls::C_DIM,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::config::SimConfig;

    fn rect() -> Rect {
        Rect::from_min_size(egui::pos2(0.0, 0.0), EVec2::new(800.0, 600.0))
    }

    fn track() -> Track {
        Track::generate(42, &SimConfig::default().track)
    }

    #[test]
    fn fit_puts_track_inside_view() {
        let t = track();
        let r = rect();
        let mut cam = CamState::default();
        cam.ensure_fit(&t, &r);
        let a = cam.to_screen(&r, t.bbox_min);
        let b = cam.to_screen(&r, t.bbox_max);
        assert!(a.x > r.left() - 1.0 && b.x < r.right() + 1.0, "赛道应完整可见");
        assert!(a.y < r.bottom() + 1.0 && b.y > r.top() - 1.0);
    }

    #[test]
    fn zoom_keeps_cursor_anchor() {
        let mut cam = CamState::default();
        let r = rect();
        cam.ensure_fit(&track(), &r);
        let ptr = egui::pos2(500.0, 220.0);
        let before = cam.to_world(&r, ptr);
        // 模拟以光标为锚点的缩放
        let factor = 1.8;
        let after_zoom = (cam.zoom() * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        cam.set_zoom(after_zoom);
        let at = cam.to_world(&r, ptr);
        cam.center.x += before.x - at.x;
        cam.center.y += before.y - at.y;
        let now = cam.to_world(&r, ptr);
        assert!((now.x - before.x).abs() < 1e-2, "光标下的世界点应保持不动");
        assert!((now.y - before.y).abs() < 1e-2);
    }

    #[test]
    fn follow_centres_on_car() {
        let mut cam = CamState::default();
        cam.ensure_fit(&track(), &rect());
        cam.set_mode(CamMode::Follow);
        cam.set_zoom(8.0);
        let car = Vec2::new(120.0, -40.0);
        for _ in 0..80 {
            cam.follow(Some(car), 1e9);
        }
        assert!((cam.center.x - car.x).abs() < 0.5);
        assert!((cam.center.y - car.y).abs() < 0.5);
        // 车在画面中心
        let p = cam.to_screen(&rect(), car);
        assert!((p.x - 400.0).abs() < 0.5 && (p.y - 300.0).abs() < 0.5);
    }

    #[test]
    fn modes_keep_their_own_zoom() {
        let mut cam = CamState::default();
        cam.set_mode(CamMode::Free);
        cam.set_zoom(2.0);
        cam.set_mode(CamMode::Follow);
        cam.set_zoom(12.0);
        assert_eq!(cam.zoom(), 12.0);
        cam.set_mode(CamMode::Free);
        assert_eq!(cam.zoom(), 2.0);
        // clamp 生效
        cam.set_zoom(9999.0);
        assert_eq!(cam.zoom(), ZOOM_MAX);
    }
}
