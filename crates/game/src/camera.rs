//! 相机：跟随（贴车 + 捏合缩放，拖拽无缝滑入自由）/ 自由（画布：拖拽平移 + 滚轮/双指缩放）。

use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;

use crate::state::{CamMode, Game};

#[derive(Component)]
pub struct CameraRig {
    /// 跟随模式的平滑位置。
    pub smooth: Vec3,
    /// 跟随模式的基准视野高度（捏合可调）。
    pub follow_zoom: f32,
    /// 自由模式：中心位置（世界坐标）。
    pub free_pos: Vec2,
    /// 自由模式：可见世界高度（米）。
    pub free_zoom: f32,
    /// 自由模式：上一帧指针位置（拖拽用）。
    pub last_cursor: Option<Vec2>,
    /// 自由模式：上一帧双指距离。
    pub last_pinch: Option<f32>,
    /// 上一帧模式（进入自由时从当前视图无缝衔接）。
    pub prev_mode: CamMode,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            smooth: Vec3::ZERO,
            follow_zoom: 90.0,
            free_pos: Vec2::ZERO,
            free_zoom: 90.0,
            last_cursor: None,
            last_pinch: None,
            prev_mode: CamMode::Follow,
        }
    }
}

/// 手势/鼠标输入：跟随模式捏合缩放 + 拖拽滑入自由；自由模式拖拽 + 缩放。
pub fn free_camera_input(
    mut rigs: Query<&mut CameraRig, With<Camera2d>>,
    mut cams: Query<&mut Transform, (With<Camera2d>, Without<CameraRig>)>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    touches: Res<Touches>,
    mut game: ResMut<Game>,
) {
    let Ok(mut rig) = rigs.single_mut() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };

    // ---- 双指捏合缩放（两种模式都生效）----
    let fingers: Vec<&bevy::input::touch::Touch> = touches.iter().collect();
    if fingers.len() >= 2 {
        let a = fingers[0].position();
        let b = fingers[1].position();
        let dist = a.distance(b);
        if let Some(prev) = rig.last_pinch {
            if prev > 1.0 && dist > 1.0 {
                let ratio = prev / dist;
                match game.cam_mode {
                    CamMode::Free => {
                        rig.free_zoom = (rig.free_zoom * ratio).clamp(20.0, 1500.0);
                    }
                    CamMode::Follow => {
                        rig.follow_zoom = (rig.follow_zoom * ratio).clamp(30.0, 600.0);
                    }
                }
            }
        }
        rig.last_pinch = Some(dist);
        rig.last_cursor = None;
        return;
    }
    rig.last_pinch = None;

    // ---- 单指拖拽 ----
    if fingers.len() == 1 {
        let pos = fingers[0].position();
        let delta = fingers[0].delta();
        if game.cam_mode == CamMode::Follow {
            // 跟随模式：拖动即无缝滑入自由（从当前位置继续）
            if delta.length() > 2.0 {
                enter_free(&mut rig, &mut cams, window, &mut game);
            }
        }
        if game.cam_mode == CamMode::Free {
            let world_per_px = rig.free_zoom / window.height().max(1.0);
            rig.free_pos.x -= delta.x * world_per_px;
            rig.free_pos.y += delta.y * world_per_px;
        }
        let _ = pos;
        return;
    }

    // ---- 鼠标拖拽（避开顶栏）----
    if let Some(cursor) = window.cursor_position() {
        let in_bar = cursor.y < 60.0;
        if mouse.pressed(MouseButton::Left) && !in_bar {
            if game.cam_mode == CamMode::Follow {
                enter_free(&mut rig, &mut cams, window, &mut game);
            }
            if let Some(last) = rig.last_cursor {
                let delta = cursor - last;
                let world_per_px = rig.free_zoom / window.height().max(1.0);
                rig.free_pos.x -= delta.x * world_per_px;
                rig.free_pos.y += delta.y * world_per_px;
            }
            rig.last_cursor = Some(cursor);
        } else {
            rig.last_cursor = None;
        }

        // ---- 滚轮缩放（自由模式）----
        let dy = scroll.delta.y;
        if dy.abs() > 1e-4 && game.cam_mode == CamMode::Free {
            let factor = if dy > 0.0 { 0.9 } else { 1.1 };
            rig.free_zoom = (rig.free_zoom * factor).clamp(20.0, 1500.0);
        }
    }
}

/// 从当前视图无缝进入自由模式。
fn enter_free(
    rig: &mut CameraRig,
    cams: &mut Query<&mut Transform, (With<Camera2d>, Without<CameraRig>)>,
    window: &Window,
    game: &mut Game,
) {
    game.cam_mode = CamMode::Free;
    // 位置：从当前相机位置继续
    if let Ok(tf) = cams.single_mut() {
        rig.free_pos = Vec2::new(tf.translation.x, tf.translation.y);
    }
    // 视野：从跟随缩放继续
    rig.free_zoom = rig.follow_zoom.clamp(20.0, 1500.0);
    let _ = window;
}

pub fn update_camera(
    mut cams: Query<(&mut Transform, &mut Projection, &mut CameraRig), With<Camera2d>>,
    windows: Query<&Window>,
    game: Res<Game>,
) {
    let Ok((mut tf, mut proj, mut rig)) = cams.single_mut() else {
        return;
    };
    let Projection::Orthographic(ref mut ortho) = *proj else {
        return;
    };
    let track = &game.viewer.env.track;
    let car = &game.viewer.env.car;
    let Ok(window) = windows.single() else { return };

    match game.cam_mode {
        CamMode::Follow => {
            let target = Vec3::new(car.pos.x, car.pos.y, 0.0);
            if rig.smooth == Vec3::ZERO {
                rig.smooth = target;
            }
            rig.smooth = rig.smooth.lerp(target, 0.08);
            tf.translation = rig.smooth;
            // 速度越快看得越远
            ortho.scaling_mode = bevy::camera::ScalingMode::AutoMax {
                max_width: rig.follow_zoom * 2.0,
                max_height: rig.follow_zoom + car.speed.abs() * 0.8,
            };
        }
        CamMode::Free => {
            // 进入自由的第一帧：若从跟随切来，由 enter_free 初始化；
            // 按钮切换（未经过 enter_free）也在此兜底衔接
            if rig.prev_mode == CamMode::Follow && rig.free_pos == Vec2::ZERO {
                rig.free_pos = Vec2::new(tf.translation.x, tf.translation.y);
                rig.free_zoom = rig.follow_zoom.clamp(20.0, 1500.0);
            }
            rig.smooth = Vec3::ZERO;
            let aspect = window.width() / window.height().max(1.0);
            ortho.scaling_mode = bevy::camera::ScalingMode::Fixed {
                width: rig.free_zoom * aspect,
                height: rig.free_zoom,
            };
            tf.translation = Vec3::new(rig.free_pos.x, rig.free_pos.y, 0.0);
        }
    }
    let _ = track;
    rig.prev_mode = game.cam_mode;
}
