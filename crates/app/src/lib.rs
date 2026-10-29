//! RL Racer 应用库：桌面 bin 与安卓 cdylib 共用。

pub mod app;
pub mod messages;
pub mod mirror;
pub mod trainer;
pub mod view;
pub mod viewer;

/// 构建 NativeOptions（桌面默认配置）。
fn native_options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 880.0])
            .with_title("RL Racer — 强化学习赛车实验台"),
        ..Default::default()
    }
}

/// 创建应用的闭包（桌面/安卓共用）。
fn app_creator() -> Box<
    dyn FnOnce(
        &eframe::CreationContext,
    ) -> Result<Box<dyn eframe::App>, Box<dyn std::error::Error + Send + Sync>>,
> {
    Box::new(|cc| {
        configure_style(&cc.egui_ctx);
        let mut app = app::RlRacerApp::new();
        app.apply_demo_env();
        Ok(Box::new(app))
    })
}

/// 桌面入口。
pub fn run() -> eframe::Result<()> {
    eframe::run_native("RL Racer", native_options(), app_creator())
}

/// 安卓入口：由 android-activity (NativeActivity) 胶水层调用。
#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(
    android_app: egui_winit::winit::platform::android::activity::AndroidApp,
) -> eframe::Result<()> {
    use egui_winit::winit::platform::android::EventLoopBuilderExtAndroid as _;

    let mut options = native_options();
    // 安卓没有窗口装饰，铺满屏幕
    options.viewport = egui::ViewportBuilder::default();
    options.android_app = Some(android_app);
    eframe::run_native("RL Racer", options, app_creator())
}

/// 训练模式强调色：琥珀 `#D69134`。
pub(crate) const MODE_TRAIN: egui::Color32 = egui::Color32::from_rgb(214, 145, 52);
/// 展示模式强调色：冷蓝 `#78AADC`。
pub(crate) const MODE_DEMO: egui::Color32 = egui::Color32::from_rgb(120, 170, 220);

/// 暗色主题：炭黑面板 + 模式色强调（训练琥珀 / 展示冷蓝）。
///
/// 色板见 `docs/ui-spec.md` B 节；模式色是全面板唯一的强调色，其余一律中性灰。
pub(crate) fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    let v = &mut style.visuals;

    // ---- 背景 ----
    v.panel_fill = egui::Color32::from_rgb(23, 25, 31); // #17191F
    v.window_fill = egui::Color32::from_rgb(30, 32, 39); // #1E2027（卡片）
    v.extreme_bg_color = egui::Color32::from_rgb(20, 22, 27); // #14161B
    v.faint_bg_color = egui::Color32::from_rgb(38, 41, 50); // #262932（控件）

    // ---- 描边 ----
    let stroke_normal = egui::Stroke::new(1.0, egui::Color32::from_rgb(52, 56, 69)); // #343845
    let stroke_strong = egui::Stroke::new(1.0, egui::Color32::from_rgb(69, 74, 89)); // #454A59
    v.window_stroke = stroke_normal;

    // ---- 文字 ----
    let text_main = egui::Color32::from_rgb(220, 223, 230); // #DCDFE6
    let text_second = egui::Color32::from_rgb(180, 185, 196); // #B4B9C4

    // ---- 选中态：跟随模式色（默认训练琥珀；运行期由 set_mode_accent 更新）----
    v.selection.bg_fill = MODE_TRAIN;
    v.selection.stroke = egui::Stroke::new(1.0, MODE_TRAIN);
    v.hyperlink_color = MODE_TRAIN;
    v.slider_trailing_fill = true;

    // ---- 控件（控件 #262932 / 悬停 #2F333E / 圆角 6 / 描边 1px）----
    let control = egui::Color32::from_rgb(38, 41, 50); // #262932
    let hover = egui::Color32::from_rgb(47, 51, 62); // #2F333E

    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(30, 32, 39);
    v.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(30, 32, 39);
    v.widgets.noninteractive.bg_stroke = stroke_normal;
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, text_second);
    v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(6);

    v.widgets.inactive.bg_fill = control;
    v.widgets.inactive.weak_bg_fill = control;
    v.widgets.inactive.bg_stroke = stroke_normal;
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text_main);
    v.widgets.inactive.corner_radius = egui::CornerRadius::same(6);

    v.widgets.hovered.bg_fill = hover;
    v.widgets.hovered.weak_bg_fill = hover;
    v.widgets.hovered.bg_stroke = stroke_strong;
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, text_main);
    v.widgets.hovered.corner_radius = egui::CornerRadius::same(6);

    v.widgets.active.bg_fill = hover;
    v.widgets.active.weak_bg_fill = hover;
    v.widgets.active.bg_stroke = stroke_strong;
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, text_main);
    v.widgets.active.corner_radius = egui::CornerRadius::same(6);

    v.widgets.open.bg_fill = MODE_TRAIN.gamma_multiply(0.28);
    v.widgets.open.weak_bg_fill = MODE_TRAIN.gamma_multiply(0.28);
    v.widgets.open.bg_stroke = egui::Stroke::new(1.0, MODE_TRAIN);
    v.widgets.open.fg_stroke = egui::Stroke::new(1.0, text_main);
    v.widgets.open.corner_radius = egui::CornerRadius::same(6);

    // ---- 间距与字号阶梯 ----
    style.spacing.item_spacing = egui::Vec2::new(8.0, 6.0);
    style.spacing.button_padding = egui::Vec2::new(10.0, 5.0);
    style.spacing.slider_width = 180.0;
    style.spacing.slider_rail_height = 4.0;
    style.spacing.indent = 12.0;
    // 所有可点控件至少 28px 高/宽
    style.spacing.interact_size = egui::Vec2::new(40.0, 28.0);
    style.spacing.window_margin = egui::Margin::same(12);
    style.spacing.menu_margin = egui::Margin::same(8);

    style
        .text_styles
        .insert(egui::TextStyle::Heading, egui::FontId::proportional(16.0));
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(11.0));
    style
        .text_styles
        .insert(egui::TextStyle::Monospace, egui::FontId::monospace(13.0));

    // 滚动条：细一点，静止时半透明（P2）
    style.spacing.scroll.bar_width = 6.0;
    style.spacing.scroll.dormant_handle_opacity = 0.5;

    ctx.set_style_of(egui::Theme::Dark, style);

    // 锁定深色主题：不锁的话会跟随系统，控件走 Light style
    // （白底输入框、浅蓝页签、白底图表，整窗与深色面板打架）。
    ctx.set_theme(egui::Theme::Dark);
    // 窗口标题栏也跟随深色（否则系统是浅色时，标题栏会白一块）
    ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(egui::SystemTheme::Dark));
}

/// 让全局选中态（滑杆填充、输入框聚焦描边、行选中）跟随当前模式色。
///
/// 只在模式色变化时改写 style，避免每帧克隆整份 [`egui::Style`]。
pub(crate) fn set_mode_accent(ctx: &egui::Context, accent: egui::Color32) {
    // 进程内只有一个应用实例，用原子量记住当前模式色就够了。
    static LAST_ACCENT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let key = u32::from_be_bytes([0, accent.r(), accent.g(), accent.b()]);
    if LAST_ACCENT.load(std::sync::atomic::Ordering::Relaxed) == key {
        return;
    }
    LAST_ACCENT.store(key, std::sync::atomic::Ordering::Relaxed);

    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.visuals.selection.bg_fill = accent;
    style.visuals.selection.stroke = egui::Stroke::new(1.0, accent);
    style.visuals.hyperlink_color = accent;
    style.visuals.widgets.open.bg_fill = accent.gamma_multiply(0.28);
    style.visuals.widgets.open.weak_bg_fill = accent.gamma_multiply(0.28);
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, accent);
    ctx.set_style_of(egui::Theme::Dark, style);
}
