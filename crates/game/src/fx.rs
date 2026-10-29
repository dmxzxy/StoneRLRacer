//! 特效：雷达射线（按距离渐变色的细长条）+ 轨迹烟粒子（渐隐）。

use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh, PrimitiveTopology};
use sim::observation::{ray_angle, MAX_RAY_DIST, N_RAYS};

use crate::{state::Game, RaysRoot};

/// 共享特效资源：网格 + 共享静态材质。
/// 性能关键：材质不逐帧变更（每次材质热更新都触发 GPU 重绑定）。
#[derive(Resource)]
pub struct FxAssets {
    strip: Handle<Mesh>,
    disc: Handle<Mesh>,
    ray_material: Handle<ColorMaterial>,
    particle_material: Handle<ColorMaterial>,
}

pub fn setup_fx(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // 射线条：从原点沿 +x，长 1m 宽 0.3m
    let mut strip = Mesh::new(PrimitiveTopology::TriangleList, bevy::asset::RenderAssetUsages::default());
    strip.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 0.15, 0.0], [1.0, 0.15, 0.0], [1.0, -0.15, 0.0], [0.0, -0.15, 0.0]],
    );
    strip.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));

    // 粒子圆盘（八边形近似）
    let mut disc = Mesh::new(PrimitiveTopology::TriangleList, bevy::asset::RenderAssetUsages::default());
    let mut positions = vec![[0.0, 0.0, 0.0]];
    for i in 0..8 {
        let a = std::f32::consts::TAU * i as f32 / 8.0;
        positions.push([a.cos() * 0.5, a.sin() * 0.5, 0.0]);
    }
    let indices: Vec<u32> = (1..8).flat_map(|i| [0u32, i, i + 1]).collect();
    disc.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    disc.insert_indices(Indices::U32(indices));

    commands.insert_resource(FxAssets {
        strip: meshes.add(strip),
        disc: meshes.add(disc),
        ray_material: materials.add(ColorMaterial::from(Color::srgba(0.45, 0.85, 0.45, 0.30))),
        particle_material: materials.add(ColorMaterial::from(Color::srgba(0.92, 0.63, 0.24, 0.30))),
    });
}

/// 单条射线实体。
#[derive(Component)]
pub struct Ray {
    pub idx: usize,
}

/// 轨迹烟粒子。
#[derive(Component)]
pub struct Particle {
    pub t: f32,
    /// 初始缩放。
    pub base: f32,
}

const PARTICLE_LIFE: f32 = 2.2;

/// 发射一枚轨迹烟粒子。
pub fn spawn_particle(
    commands: &mut Commands,
    fx: &FxAssets,
    pos: bevy::math::Vec2,
    speed: f32,
) {
    let scale = 0.6 + speed * 0.05;
    commands.spawn((
        Mesh2d(fx.disc.clone()),
        MeshMaterial2d(fx.particle_material.clone()),
        Transform::from_translation(Vec3::new(pos.x, pos.y, 0.15))
            .with_scale(Vec3::splat(scale)),
        Particle { t: 0.0, base: scale },
    ));
}

/// 消费步进产生的粒子请求。
pub fn spawn_trail(
    mut commands: Commands,
    fx: Option<Res<FxAssets>>,
    mut game: ResMut<Game>,
) {
    let Some(fx) = fx else { return };
    if game.trail_requests.is_empty() {
        return;
    }
    for (pos, speed) in game.trail_requests.drain(..) {
        spawn_particle(&mut commands, &fx, pos, speed);
    }
}

/// 惰性生成 9 条射线（挂在 RaysRoot 下）。
pub fn ensure_rays(
    mut commands: Commands,
    fx: Option<Res<FxAssets>>,
    roots: Query<Entity, With<RaysRoot>>,
    rays: Query<(), With<Ray>>,
) {
    let Some(fx) = fx else { return };
    if !rays.is_empty() {
        return;
    }
    let Ok(root) = roots.single() else { return };
    for i in 0..N_RAYS {
        commands
            .spawn((
                Mesh2d(fx.strip.clone()),
                MeshMaterial2d(fx.ray_material.clone()),
                Transform::default(),
                Ray { idx: i },
            ))
            .insert(ChildOf(root));
    }
}

/// 每帧同步射线：位置/朝向/长度 + 距离渐变色 + 显隐。
pub fn sync_rays(
    mut rays: Query<(&Ray, &mut Transform)>,
    mut roots: Query<&mut Visibility, With<RaysRoot>>,
    game: Res<Game>,
) {
    if let Ok(mut vis) = roots.single_mut() {
        *vis = if game.show_rays {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if !game.show_rays {
        return;
    }
    let car = &game.viewer.env.car;
    let track = &game.viewer.env.track;
    for (ray, mut tf) in rays.iter_mut() {
        let ang = car.heading + ray_angle(ray.idx);
        let dir = sim::math::Vec2::from_angle(ang);
        let d = track.raycast(car.pos, dir, MAX_RAY_DIST);
        tf.translation = Vec3::new(car.pos.x, car.pos.y, 0.4);
        tf.rotation = Quat::from_rotation_z(ang);
        tf.scale = Vec3::new(d, 1.0, 1.0);
    }
}

/// 粒子生命周期：膨胀 + 渐隐 + 消亡。
pub fn update_particles(
    mut commands: Commands,
    time: Res<Time>,
    mut particles: Query<(Entity, &mut Particle, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (entity, mut p, mut tf) in particles.iter_mut() {
        p.t += dt;
        if p.t >= PARTICLE_LIFE {
            commands.entity(entity).despawn();
            continue;
        }
        let k = p.t / PARTICLE_LIFE;
        // 膨胀代替渐隐（共享材质不可变）
        tf.scale = Vec3::splat(p.base + k * 2.2);
    }
}
