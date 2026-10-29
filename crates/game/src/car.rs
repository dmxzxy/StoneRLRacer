//! 赛车：AI 生成的俯视精灵图 + 按车型着色，每帧同步位姿。

use bevy::prelude::*;
use sim::car::CarConfig;

use crate::{state::Game, ArtAssets, CarRoot};

/// 生成赛车精灵实体（CarRoot，精灵默认朝右 = heading 0）。
pub fn spawn_car(commands: &mut Commands, art: &ArtAssets, cfg: &CarConfig, color: [u8; 3]) {
    // 精灵含车轮，视觉尺寸略大于车长
    let len = cfg.length * 1.35;
    let size = Vec2::new(len, len * 0.5);
    let tint = Color::srgb(
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
    );
    commands.spawn((
        CarRoot,
        Sprite {
            image: art.car.clone(),
            custom_size: Some(size),
            color: tint,
            ..default()
        },
        Transform::default(),
    ));
}

/// 每帧把 env 里的车位姿同步到实体。
pub fn sync_car(mut roots: Query<&mut Transform, With<CarRoot>>, game: Res<Game>) {
    let Ok(mut tf) = roots.single_mut() else {
        return;
    };
    let car = &game.viewer.env.car;
    tf.translation = Vec3::new(car.pos.x, car.pos.y, 0.2);
    tf.rotation = Quat::from_rotation_z(car.heading);
}
