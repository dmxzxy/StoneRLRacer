//! RL Racer 游玩端：Bevy 渲染 + sim/rl 核心。
//!
//! 选赛道/车型/模型看 AI 驾驶 + 内置训练（边训练边看进步）。
//! 纹理与车精灵由 PaperArt（10.10.110.29:8790）生成，源图在 artgen/。

pub mod camera;
pub mod car;
pub mod fx;
pub mod model;
pub mod state;
pub mod track_mesh;
pub mod train;
pub mod ui;

use bevy::image::{ImageSampler, ImageSamplerDescriptor};
use bevy::render::view::Msaa;
use bevy::prelude::*;
use bevy::image::ImageAddressMode;
use state::Game;

/// 赛道实体树根标签。
#[derive(Component)]
pub struct TrackRoot;

/// 赛车实体树根标签。
#[derive(Component)]
pub struct CarRoot;

/// 雷达射线组标签。
#[derive(Component)]
pub struct RaysRoot;

/// 赛道变更消息（重建赛道网格）。
#[derive(Message)]
pub struct TrackChanged;

/// 车型变更消息（重建赛车精灵）。
#[derive(Message)]
pub struct CarChanged;

/// 美术资源句柄。
#[derive(Resource)]
pub struct ArtAssets {
    pub grass: Handle<Image>,
    pub asphalt: Handle<Image>,
    pub car: Handle<Image>,
}

/// 安卓入口（GameActivity 经 bevy_main 注册）。
#[cfg(target_os = "android")]
#[bevy::prelude::bevy_main]
pub fn main() {
    run();
}

pub fn run() {
    let mut app = App::new();

    #[cfg(target_os = "android")]
    {
        use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
        use bevy::render::RenderPlugin;
        use bevy::window::{MonitorSelection, WindowMode};
        use bevy::winit::{UpdateMode, WinitSettings};

        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "RL Racer".into(),
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                        // 降分辨率渲染（合成器放大）：填充负载 1/4，帧时间远离 vsync 边缘
                        resolution: bevy::window::WindowResolution::new(1400, 1000),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        backends: Some(Backends::VULKAN),
                        ..default()
                    })),
                    ..default()
                }),
        );
        // 双 Continuous：彻底排除"空闲时反应式节流"（焦点检测在部分安卓设备上不可靠）
        app.insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Continuous,
        });
    }

    #[cfg(not(target_os = "android"))]
    {
        app.add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "RL Racer — 游玩端".into(),
                    resolution: bevy::window::WindowResolution::new(1400, 880),
                    ..default()
                }),
                ..default()
            }),
        );
    }

    app.insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .init_resource::<Game>()
        .add_message::<TrackChanged>()
        .add_message::<CarChanged>()
        .add_systems(Startup, (setup_world, ui::build_ui))
        .add_observer(ui::handle_activate)
        .add_systems(
            Update,
            (
                ui::update_hud,
                ui::update_fps_bar,
                ui::update_dashboard,
                train::poll_training,
                step_game,
                fx::spawn_trail,
                camera::free_camera_input,
                car::sync_car,
                fx::ensure_rays,
                fx::sync_rays,
                fx::update_particles,
                camera::update_camera,
            )
                .chain(),
        )
        .add_systems(Update, (rebuild_track, rebuild_car))
        .run();
}

fn setup_world(mut commands: Commands, server: Res<AssetServer>, mut images: ResMut<Assets<Image>>) {
    // MSAA 4x 是 0.20 默认——2800x2000 平板上 4 倍着色负载，必须关
    commands.spawn((Camera2d, camera::CameraRig::default(), Msaa::Off));

    // 加载美术资源 + 平铺采样器
    let grass = server.load("grass.png");
    let asphalt = server.load("asphalt.png");
    let car = server.load("car_white.png");
    for h in [&grass, &asphalt] {
        if let Some(mut img) = images.get_mut(h) {
            img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                // 镜像重复：贴图边缘镜像对齐，平铺完全无接缝
                // （生成的纹理不是无缝的，普通 Repeat 会在每格边界出横线）
                address_mode_u: ImageAddressMode::MirrorRepeat,
                address_mode_v: ImageAddressMode::MirrorRepeat,
                address_mode_w: ImageAddressMode::MirrorRepeat,
                ..ImageSamplerDescriptor::linear()
            });
        }
    }
    commands.insert_resource(ArtAssets {
        grass,
        asphalt,
        car,
    });

    commands.spawn((RaysRoot, Visibility::default()));
    commands.write_message(TrackChanged);
    commands.write_message(CarChanged);
}

/// 每帧步进：speed_mult 次 env.step + 轨迹粒子发射。
fn step_game(mut commands: Commands, mut game: ResMut<Game>) {
    if !game.running {
        return;
    }
    for _ in 0..game.speed() {
        game.step(&mut commands);
    }
}

fn rebuild_track(
    mut commands: Commands,
    mut events: MessageReader<TrackChanged>,
    roots: Query<Entity, With<TrackRoot>>,
    art: Option<Res<ArtAssets>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    game: Res<Game>,
) {
    if events.read().next().is_none() {
        return;
    }
    let Some(art) = art else { return };
    if let Ok(root) = roots.single() {
        commands.entity(root).despawn();
    }
    track_mesh::spawn_track(
        &mut commands,
        &art,
        &mut meshes,
        &mut materials,
        &game.viewer.env.track,
    );
}

fn rebuild_car(
    mut commands: Commands,
    mut events: MessageReader<CarChanged>,
    roots: Query<Entity, With<CarRoot>>,
    art: Option<Res<ArtAssets>>,
    game: Res<Game>,
) {
    if events.read().next().is_none() {
        return;
    }
    let Some(art) = art else { return };
    if let Ok(root) = roots.single() {
        commands.entity(root).despawn();
    }
    car::spawn_car(
        &mut commands,
        &art,
        &game.viewer.env.cfg.car,
        state::CARS[game.car_idx].1,
    );
}
