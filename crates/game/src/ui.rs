//! UI：双屏幕——游玩模式（顶栏 + HUD）与训练模式（统计面板 + 奖励曲线）。
//!
//! 中文显示：所有文本用打包进 assets 的中文字体子集。

use bevy::prelude::*;
use bevy::ui_widgets::{Activate, Button};

use crate::state::{CamMode, Game, Screen, SPEEDS, TRACKS};

/// 按钮动作。
#[derive(Component, Clone, Copy)]
pub enum Btn {
    TrackPrev,
    TrackNext,
    CarPrev,
    CarNext,
    ModelPrev,
    ModelNext,
    CamToggle,
    SpeedDown,
    SpeedUp,
    PlayPause,
    RaysToggle,
    /// 去训练模式屏。
    GoTrain,
    /// 返回游玩模式屏。
    GoPlay,
    /// 训练屏内：开始/停止训练。
    TrainStartStop,
}

/// 动态文本标签。
#[derive(Component)]
pub struct TrackName;
#[derive(Component)]
pub struct CarName;
#[derive(Component)]
pub struct ModelName;
#[derive(Component)]
pub struct SpeedName;
#[derive(Component)]
pub struct CamLabel;
#[derive(Component)]
pub struct PlayLabel;
#[derive(Component)]
pub struct RaysLabel;
#[derive(Component)]
pub struct HudLine1;
#[derive(Component)]
pub struct HudLine2;
#[derive(Component)]
pub struct HudFps;
/// FPS 可视化条（宽度∝帧率，绿色≥50，黄30-50，红<30）。
#[derive(Component)]
pub struct HudFpsBar;
#[derive(Component)]
pub struct TrainLabel;
/// 训练屏统计值。
#[derive(Component)]
pub struct DSteps;
#[derive(Component)]
pub struct DEps;
#[derive(Component)]
pub struct DReward;
#[derive(Component)]
pub struct DSps;
#[derive(Component)]
pub struct DTrainBtn;
/// 训练屏图表容器与柱条组。
#[derive(Component)]
pub struct ChartArea;
#[derive(Component)]
pub struct ChartBars;

/// 游玩屏根（顶栏 + HUD）。
#[derive(Component)]
pub struct PlayScreen;
/// 训练屏根。
#[derive(Component)]
pub struct TrainScreen;

const BG: Color = Color::srgb(0.09, 0.10, 0.12);
const PANEL: Color = Color::srgb(0.12, 0.13, 0.16);
const BTN_BG: Color = Color::srgb(0.14, 0.15, 0.18);
const ACCENT: Color = Color::srgb(0.84, 0.57, 0.20);
const GOOD: Color = Color::srgb(0.35, 0.75, 0.45);
const BAD: Color = Color::srgb(0.80, 0.32, 0.30);
const TEXT: Color = Color::srgb(0.87, 0.88, 0.91);
const DIM: Color = Color::srgb(0.55, 0.57, 0.62);

fn text_font(f: &Handle<Font>, size: f32) -> TextFont {
    TextFont {
        font: FontSource::Handle(f.clone()),
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn button(commands: &mut ChildSpawnerCommands, f: &Handle<Font>, label: &str, action: Btn) {
    commands
        .spawn((
            Button,
            Node {
                height: Val::Px(34.0),
                padding: UiRect::horizontal(Val::Px(10.0)),
                margin: UiRect::horizontal(Val::Px(2.0)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(BTN_BG),
            BorderColor::from(Color::srgb(0.25, 0.26, 0.30)),
            action,
        ))
        .with_child((Text::new(label), text_font(f, 15.0), TextColor(TEXT)));
}

fn label_text(commands: &mut ChildSpawnerCommands, f: &Handle<Font>, tag: impl Component) {
    commands.spawn((
        Text::new("…"),
        text_font(f, 15.0),
        TextColor(ACCENT),
        Node {
            width: Val::Px(86.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        tag,
    ));
}

fn group_label(commands: &mut ChildSpawnerCommands, f: &Handle<Font>, text: &str) {
    commands.spawn((
        Text::new(text),
        text_font(f, 13.0),
        TextColor(DIM),
        Node {
            margin: UiRect::horizontal(Val::Px(4.0)),
            ..default()
        },
    ));
}

pub fn build_ui(mut commands: Commands, server: Res<AssetServer>) {
    let f = server.load("fonts/cjk.ttf");

    // ================= 游玩屏 =================
    let play = commands
        .spawn((
            PlayScreen,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
        ))
        .id();

    // ---- 顶栏 ----
    commands.entity(play).with_children(|bar_root| {
        bar_root
            .spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(52.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(2.0),
                    border: UiRect::bottom(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(BG),
                BorderColor::from(Color::srgb(0.2, 0.21, 0.24)),
            ))
            .with_children(|bar| {
                bar.spawn((
                    Text::new("RL Racer"),
                    text_font(&f, 18.0),
                    TextColor(TEXT),
                    Node {
                        margin: UiRect::horizontal(Val::Px(10.0)),
                        ..default()
                    },
                ));

                let sel = |bar: &mut ChildSpawnerCommands, name: &str, prev, next| {
                    group_label(bar, &f, name);
                    button(bar, &f, "←", prev);
                    button(bar, &f, "→", next);
                };

                sel(bar, "赛道", Btn::TrackPrev, Btn::TrackNext);
                label_text(bar, &f, TrackName);
                sel(bar, "车型", Btn::CarPrev, Btn::CarNext);
                label_text(bar, &f, CarName);
                sel(bar, "模型", Btn::ModelPrev, Btn::ModelNext);
                label_text(bar, &f, ModelName);

                group_label(bar, &f, "倍速");
                button(bar, &f, "-", Btn::SpeedDown);
                label_text(bar, &f, SpeedName);
                button(bar, &f, "+", Btn::SpeedUp);

                button(bar, &f, "镜头", Btn::CamToggle);
                label_text(bar, &f, CamLabel);

                button(bar, &f, "暂停", Btn::PlayPause);
                label_text(bar, &f, PlayLabel);

                button(bar, &f, "射线", Btn::RaysToggle);
                label_text(bar, &f, RaysLabel);

                button(bar, &f, "训练 ▸", Btn::GoTrain);
            });

        // ---- 左下 HUD ----
        bar_root
            .spawn(Node {
                position_type: PositionType::Absolute,
                left: Val::Px(14.0),
                bottom: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                ..default()
            })
            .with_children(|p| {
                p.spawn((Text::new("…"), text_font(&f, 14.0), TextColor(TEXT), HudLine1));
                p.spawn((Text::new("…"), text_font(&f, 15.0), TextColor(DIM), HudLine2));
                p.spawn((Text::new(""), text_font(&f, 12.0), TextColor(DIM), HudFps));
                p.spawn((
                    Node {
                        width: Val::Px(0.0),
                        height: Val::Px(6.0),
                        ..default()
                    },
                    BackgroundColor(GOOD),
                    HudFpsBar,
                ));
            });
        // 训练中的小提示（右下）
        bar_root.spawn(Node {
            position_type: PositionType::Absolute,
            right: Val::Px(14.0),
            bottom: Val::Px(12.0),
            ..default()
        })
        .with_child((Text::new(""), text_font(&f, 13.0), TextColor(ACCENT), TrainLabel));
    });

    // ================= 训练屏 =================
    commands
        .spawn((
            TrainScreen,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(16.0)),
                row_gap: Val::Px(12.0),
                ..default()
            },
            BackgroundColor(Color::srgb(0.07, 0.08, 0.10)),
            Visibility::Hidden,
        ))
        .with_children(|col| {
            // 标题行
            col.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(12.0),
                ..default()
            })
            .with_children(|row| {
                row.spawn((Text::new("训练模式"), text_font(&f, 24.0), TextColor(TEXT)));
                button(row, &f, "开始训练", Btn::TrainStartStop);
                row.spawn((Text::new("…"), text_font(&f, 15.0), TextColor(ACCENT), DTrainBtn));
                row.spawn(Node { flex_grow: 1.0, ..default() });
                button(row, &f, "◂ 游玩", Btn::GoPlay);
            });

            // 统计卡片行
            col.spawn(Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(12.0),
                ..default()
            })
            .with_children(|cards| {
                fn card(cards: &mut ChildSpawnerCommands, f: &Handle<Font>, name: &str, tag: impl Component) {
                    cards
                        .spawn((
                            Node {
                                width: Val::Px(200.0),
                                padding: UiRect::all(Val::Px(14.0)),
                                flex_direction: FlexDirection::Column,
                                row_gap: Val::Px(4.0),
                                ..default()
                            },
                            BackgroundColor(PANEL),
                            BorderColor::from(Color::srgb(0.22, 0.23, 0.27)),
                        ))
                        .with_children(|c| {
                            c.spawn((Text::new(name), text_font(f, 13.0), TextColor(DIM)));
                            c.spawn((Text::new("0"), text_font(f, 26.0), TextColor(TEXT), tag));
                        });
                }
                card(cards, &f, "步数", DSteps);
                card(cards, &f, "局数", DEps);
                card(cards, &f, "平均奖励", DReward);
                card(cards, &f, "SPS", DSps);
            });

            // 奖励曲线
            col.spawn((
                Text::new("奖励曲线（近 100 次统计，绿正红负）"),
                text_font(&f, 15.0),
                TextColor(DIM),
            ));
            col.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(260.0),
                    ..default()
                },
                BackgroundColor(PANEL),
                BorderColor::from(Color::srgb(0.22, 0.23, 0.27)),
                ChartArea,
            ))
            .with_child((Node::default(), ChartBars));
        });
}

/// 按钮激活（ui_widgets 事件模型的 observer）。
pub fn handle_activate(
    ev: On<Activate>,
    actions: Query<&Btn>,
    mut game: ResMut<Game>,
    mut track_ev: MessageWriter<crate::TrackChanged>,
    mut car_ev: MessageWriter<crate::CarChanged>,
) {
    let Ok(btn) = actions.get(ev.entity) else {
        return;
    };
    match *btn {
        Btn::TrackPrev => {
            game.track_idx = (game.track_idx + TRACKS.len() - 1) % TRACKS.len();
            game.regen_track();
            track_ev.write(crate::TrackChanged);
        }
        Btn::TrackNext => {
            game.track_idx = (game.track_idx + 1) % TRACKS.len();
            game.regen_track();
            track_ev.write(crate::TrackChanged);
        }
        Btn::CarPrev => {
            game.car_idx = (game.car_idx + 4) % 5;
            game.apply_car();
            car_ev.write(crate::CarChanged);
        }
        Btn::CarNext => {
            game.car_idx = (game.car_idx + 1) % 5;
            game.apply_car();
            car_ev.write(crate::CarChanged);
        }
        Btn::ModelPrev | Btn::ModelNext => {
            if game.models.is_empty() {
                // 空列表：重新扫描（支持 adb push 新模型后刷新）
                game.models = crate::model::scan_models();
            }
            if !game.models.is_empty() {
                let n = game.models.len();
                let dir = if matches!(btn, Btn::ModelNext) { n - 1 } else { 1 };
                game.model_idx = (game.model_idx + dir) % n;
                game.load_selected_model();
            }
        }
        Btn::CamToggle => {
            game.cam_mode = game.cam_mode.next();
        }
        Btn::SpeedUp => {
            game.speed_idx = (game.speed_idx + 1).min(SPEEDS.len() - 1);
        }
        Btn::SpeedDown => {
            game.speed_idx = game.speed_idx.saturating_sub(1);
        }
        Btn::PlayPause => {
            game.running = !game.running;
        }
        Btn::RaysToggle => {
            game.show_rays = !game.show_rays;
        }
        Btn::GoTrain => {
            game.screen = Screen::Train;
        }
        Btn::GoPlay => {
            game.screen = Screen::Play;
        }
        Btn::TrainStartStop => {
            if game.training.handle.is_some() {
                game.training.stop();
            } else {
                let cfg = sim::config::SimConfig {
                    track: crate::state::track_cfg_pub(&crate::state::TRACKS[game.track_idx].1),
                    car: crate::state::CARS[game.car_idx].2(),
                    start_noise: false,
                    ..Default::default()
                };
                game.training = crate::train::TrainState {
                    handle: Some(crate::train::start(cfg, game.seed)),
                    ..Default::default()
                };
            }
        }
    }
}

/// 游玩屏文本刷新。
pub fn update_hud(
    game: Res<Game>,
    mut w: TextUiWriter,
    track: Query<Entity, With<TrackName>>,
    car: Query<Entity, With<CarName>>,
    model: Query<Entity, With<ModelName>>,
    speed: Query<Entity, With<SpeedName>>,
    cam: Query<Entity, With<CamLabel>>,
    play: Query<Entity, With<PlayLabel>>,
    rays: Query<Entity, With<RaysLabel>>,
    train: Query<Entity, With<TrainLabel>>,
    h1: Query<Entity, With<HudLine1>>,
    h2: Query<Entity, With<HudLine2>>,
    fps_label: Query<Entity, (With<HudFps>, Without<HudLine1>, Without<HudLine2>)>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut hud_timer: Local<f32>,
) {
    // 文本重排布开销大：HUD 文本 10Hz 足够
    *hud_timer += 0.0;
    let update_text = {
        // 用帧计数近似 6 帧一次
        use std::cell::Cell;
        thread_local! { static FRAME: Cell<u32> = Cell::new(0); }
        FRAME.with(|f| {
            let n = f.get() + 1;
            f.set(n);
            n % 6 == 0
        })
    };
    if !update_text {
        return;
    }
    macro_rules! set {
        ($q:expr, $s:expr) => {
            if let Ok(e) = $q.single() {
                if let Some((_ent, _idx, mut text, ..)) = w.get(e, 0) {
                    *text = $s;
                }
            }
        };
    }

    set!(track, game.track_label().to_string());
    set!(car, game.car_label().to_string());
    set!(model, game.model_label());
    set!(speed, format!("{}x", game.speed()));
    set!(cam, game.cam_mode.label().to_string());
    set!(play, if game.running { "运行中".into() } else { "已暂停".into() });
    set!(rays, if game.show_rays { "开".into() } else { "关".into() });

    let v = &game.viewer;
    if game.training.handle.is_some() {
        set!(h1, "AI 驾驶 · 训练中的最新策略".to_string());
        set!(
            train,
            format!(
                "训练中 · {}步 · {:.0}SPS · 奖励{:.0}",
                game.training.steps, game.training.sps, game.training.avg_reward
            )
        );
    } else {
        set!(h1, format!("AI 驾驶 · {}", v.policy_label));
        set!(train, String::new());
    }
    let result = v
        .last_result
        .map(|r| r.label().to_string())
        .unwrap_or_default();
    set!(
        h2,
        format!(
            "速度 {:.0} km/h   进度 {:.0}%   圈数 {}   最佳 {}   {}",
            v.env.car.speed * 3.6,
            v.env.progress_frac() * 100.0,
            v.laps,
            v.best_lap.map(|t| format!("{t:.1}s")).unwrap_or_else(|| "—".into()),
            result,
        )
    );
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    set!(fps_label, format!("FPS {fps:.0}"));
}

/// FPS 条更新（宽度 ∝ 帧率）。
pub fn update_fps_bar(
    mut bars: Query<&mut Node, With<HudFpsBar>>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
    let Ok(mut node) = bars.single_mut() else {
        return;
    };
    let fps = diagnostics
        .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    node.width = Val::Px(((fps / 60.0 * 200.0) as f32).clamp(2.0, 200.0));
}

/// 训练屏：显隐 + 统计文本 + 奖励曲线柱条（仅数据变化时重建）。
pub fn update_dashboard(
    game: Res<Game>,
    mut w: TextUiWriter,
    mut play_screens: Query<&mut Visibility, With<PlayScreen>>,
    mut train_screens: Query<&mut Visibility, (With<TrainScreen>, Without<PlayScreen>)>,
    dsteps: Query<Entity, (With<DSteps>, Without<DEps>)>,
    deps: Query<Entity, (With<DEps>, Without<DSteps>)>,
    dreward: Query<Entity, (With<DReward>, Without<DSteps>, Without<DEps>)>,
    dsps: Query<Entity, (With<DSps>, Without<DSteps>, Without<DEps>, Without<DReward>)>,
    dbtn: Query<
        Entity,
        (
            With<DTrainBtn>,
            Without<DSteps>,
            Without<DEps>,
            Without<DReward>,
            Without<DSps>,
        ),
    >,
    areas: Query<&ComputedNode, With<ChartArea>>,
    bars: Query<Entity, With<ChartBars>>,
    mut commands: Commands,
    mut last_len: Local<usize>,
) {
    // 屏幕显隐
    if let Ok(mut vis) = play_screens.single_mut() {
        *vis = if game.screen == Screen::Play {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if let Ok(mut vis) = train_screens.single_mut() {
        *vis = if game.screen == Screen::Train {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }

    macro_rules! set {
        ($q:expr, $s:expr) => {
            if let Ok(e) = $q.single() {
                if let Some((_ent, _idx, mut text, ..)) = w.get(e, 0) {
                    *text = $s;
                }
            }
        };
    }

    let t = &game.training;
    set!(dsteps, format!("{}", t.steps));
    set!(deps, format!("{}", t.episodes));
    set!(dreward, format!("{:.1}", t.avg_reward));
    set!(dsps, format!("{:.0}", t.sps));
    if t.handle.is_some() {
        set!(dbtn, "训练中…（再点停止）".to_string());
    } else {
        set!(dbtn, "未训练".to_string());
    }

    // 图表：仅训练屏可见、数据变化时重建
    if game.screen != Screen::Train {
        return;
    }
    let hist = &game.training.history;
    if hist.is_empty() || hist.len() == *last_len {
        return;
    }
    *last_len = hist.len();
    let Ok(area) = areas.single() else {
        return;
    };
    let Ok(bars_parent) = bars.single() else {
        return;
    };
    commands.entity(bars_parent).despawn_children();
    let w_px = area.size().x - 20.0;
    let h_px = area.size().y - 20.0;
    let max_abs = hist.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let n = hist.len() as f32;
    let bar_w = (w_px / n).clamp(2.0, 18.0);
    for (i, v) in hist.iter().enumerate() {
        let frac = (v / max_abs).clamp(-1.0, 1.0) * (h_px * 0.45);
        let (top, height) = if frac >= 0.0 {
            (10.0 + h_px * 0.5 - frac, frac.max(1.0))
        } else {
            (10.0 + h_px * 0.5, (-frac).max(1.0))
        };
        let color = if *v >= 0.0 { GOOD } else { BAD };
        commands.entity(bars_parent).with_child((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0 + i as f32 * bar_w),
                top: Val::Px(top),
                width: Val::Px((bar_w - 1.5).max(1.0)),
                height: Val::Px(height),
                ..default()
            },
            BackgroundColor(color),
        ));
    }
}
