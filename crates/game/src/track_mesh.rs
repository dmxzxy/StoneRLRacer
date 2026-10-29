//! 赛道网格：沥青纹理路面 / 路缘石 / 终点格纹 / 草地背景。
//! 纹理由 PaperArt 生成（artgen/ 里的源图 → assets/）。

use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh, PrimitiveTopology};
use sim::math::Vec2;
use sim::track::Track;

use crate::{ArtAssets, TrackRoot};

const CURB_RED: Color = Color::srgb(0.75, 0.24, 0.22);
const CURB_WHITE: Color = Color::srgb(0.79, 0.80, 0.81);
const CHECK_DARK: Color = Color::srgb(0.08, 0.08, 0.10);
const CHECK_LIGHT: Color = Color::srgb(0.88, 0.88, 0.90);

/// 沥青纹理每格的世界尺寸（米）。
const ASPHALT_TILE: f32 = 9.0;

/// 若干四边形 → 网格（可选 UV：按世界坐标 / tile）。
fn quads_mesh(quads: &[[Vec2; 4]], z: f32, uv_tile: Option<f32>) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(quads.len() * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(quads.len() * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(quads.len() * 6);
    for (i, q) in quads.iter().enumerate() {
        let base = (i * 4) as u32;
        for p in q {
            positions.push([p.x, p.y, z]);
            if let Some(tile) = uv_tile {
                uvs.push([p.x / tile, p.y / tile]);
            }
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    if uv_tile.is_some() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// 生成整条赛道实体树（含草地背景 + 纹理路面）。
pub fn spawn_track(
    commands: &mut Commands,
    art: &ArtAssets,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    track: &Track,
) {
    let parent = commands.spawn(TrackRoot).id();
    let _ = art.grass.clone();
    let mut spawn = |commands: &mut Commands,
                     mesh: Mesh,
                     color: Color,
                     texture: Option<Handle<Image>>| {
        let mat = match texture {
            Some(tex) => ColorMaterial {
                color,
                texture: Some(tex),
                ..default()
            },
            None => ColorMaterial::from(color),
        };
        commands
            .spawn((Mesh2d(meshes.add(mesh)), MeshMaterial2d(materials.add(mat))))
            .insert(ChildOf(parent));
    };

    let m = track.samples.len();

    // ---- 路面：沥青纹理（UV 按世界坐标平铺）----
    let mut road: Vec<[Vec2; 4]> = Vec::with_capacity(m);
    for i in 0..m {
        let j = (i + 1) % m;
        let a = &track.samples[i];
        let b = &track.samples[j];
        road.push([
            a.pos + a.normal * a.half_width,
            b.pos + b.normal * b.half_width,
            b.pos - b.normal * b.half_width,
            a.pos - a.normal * a.half_width,
        ]);
    }
    spawn(
        commands,
        quads_mesh(&road, 0.0, Some(ASPHALT_TILE)),
        Color::WHITE,
        Some(art.asphalt.clone()),
    );

    // ---- 路缘石：沿两侧墙的 0.9m 窄条，每 3 段红白交替 ----
    const CURB_W: f32 = 0.9;
    for side in [1.0f32, -1.0f32] {
        let mut reds: Vec<[Vec2; 4]> = Vec::new();
        let mut whites: Vec<[Vec2; 4]> = Vec::new();
        for i in 0..m {
            let a = &track.samples[i];
            let b = &track.samples[(i + 1) % m];
            let oa = side * a.half_width;
            let ob = side * b.half_width;
            let q = [
                a.pos + a.normal * (oa - side * CURB_W),
                a.pos + a.normal * oa,
                b.pos + b.normal * ob,
                b.pos + b.normal * (ob - side * CURB_W),
            ];
            if (i / 3) % 2 == 0 {
                reds.push(q);
            } else {
                whites.push(q);
            }
        }
        spawn(commands, quads_mesh(&reds, 0.05, None), CURB_RED, None);
        spawn(commands, quads_mesh(&whites, 0.05, None), CURB_WHITE, None);
    }

    // ---- 终点线：2×8 黑白格 ----
    let s0 = &track.samples[0];
    let cols = 8;
    let cell_l = 1.2f32;
    let cell_w = (s0.half_width * 2.0) / cols as f32;
    let mut darks: Vec<[Vec2; 4]> = Vec::new();
    let mut lights: Vec<[Vec2; 4]> = Vec::new();
    for r in 0..2 {
        for c in 0..cols {
            let base = s0.pos - s0.tangent * (r as f32 * cell_l) - s0.normal * s0.half_width
                + s0.normal * (c as f32 * cell_w);
            let q = [
                base,
                base + s0.tangent * cell_l,
                base + s0.tangent * cell_l + s0.normal * cell_w,
                base + s0.normal * cell_w,
            ];
            if (r + c) % 2 == 0 {
                darks.push(q);
            } else {
                lights.push(q);
            }
        }
    }
    spawn(commands, quads_mesh(&darks, 0.1, None), CHECK_DARK, None);
    spawn(commands, quads_mesh(&lights, 0.1, None), CHECK_LIGHT, None);
}
