//! 赛道：闭合 Catmull-Rom 样条中心线 + 沿法线展开的内外墙。
//!
//! 中心线按弧长均匀重采样为 `samples` 个点，墙体线段同时供 raycast 与渲染复用。

use crate::math::{norm_angle, Vec2};
use crate::raycast::ray_segment;
use crate::rng::Rng;

/// 赛道宏观形状。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackShape {
    /// 极坐标谐波：r(θ) = R·(1 + Σₖ aₖ·sin(kθ + φₖ))，星形、圆滑可控。
    Harmonic,
    /// 闭合随机游走：可产生发卡弯、S 弯、平行路段等非星形结构。
    RandomWalk,
    /// 真实赛道近似轮廓（手绘控制点，非测绘数据）。
    Real(RealTrack),
}

/// 真实赛道（手绘近似轮廓，保留标志性行结构）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealTrack {
    Monza,
    Monaco,
    Spa,
    Silverstone,
    Shanghai,
}

impl RealTrack {
    pub const ALL: [RealTrack; 5] = [
        RealTrack::Monza,
        RealTrack::Monaco,
        RealTrack::Spa,
        RealTrack::Silverstone,
        RealTrack::Shanghai,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RealTrack::Monza => "蒙扎",
            RealTrack::Monaco => "摩纳哥",
            RealTrack::Spa => "斯帕",
            RealTrack::Silverstone => "银石",
            RealTrack::Shanghai => "上海",
        }
    }

    /// 赛道宽（米）。摩纳哥窄是特色。
    pub fn width(self) -> f32 {
        match self {
            RealTrack::Monaco => 9.0,
            RealTrack::Monza => 13.0,
            RealTrack::Spa => 13.0,
            RealTrack::Silverstone => 13.0,
            RealTrack::Shanghai => 12.0,
        }
    }

    /// 手绘控制点（顺时针，y 向上，约 1:1 米制后整体按半径归一）。
    pub fn ctrl_points(self) -> Vec<Vec2> {
        /// 蒙扎：主直道 → Rettifilo → Curva Grande → 双 Lesmo →
        /// 背直道 → Ascari → 帕拉波利卡。
        const MONZA: [(f32, f32); 16] = [
            (150.0, 80.0),
            (140.0, 430.0),
            (165.0, 468.0),
            (200.0, 452.0),
            (420.0, 565.0),
            (585.0, 520.0),
            (630.0, 550.0),
            (662.0, 528.0),
            (750.0, 505.0),
            (792.0, 450.0),
            (795.0, 375.0),
            (772.0, 195.0),
            (805.0, 150.0),
            (772.0, 100.0),
            (555.0, 45.0),
            (345.0, 40.0),
        ];
        /// 摩纳哥（风格化）：梳状平行车道（隧道/回程/直道/上坡/顶部/发卡后段）
        /// + 两端交替折返 + 东缘下降，几何上保证无交叉；
        /// 窄路 + 发卡 + 隧道段保留街赛性格。
        const MONACO: [(f32, f32); 20] = [
            (630.0, 60.0),
            (480.0, 52.0),
            (300.0, 58.0),
            (165.0, 62.0),
            (95.0, 118.0),
            (230.0, 128.0),
            (380.0, 122.0),
            (520.0, 128.0),
            (575.0, 190.0),
            (150.0, 195.0),
            (75.0, 258.0),
            (540.0, 262.0),
            (598.0, 330.0),
            (230.0, 338.0),
            (135.0, 398.0),
            (195.0, 452.0),
            (430.0, 445.0),
            (648.0, 395.0),
            (668.0, 205.0),
            (640.0, 92.0),
        ];
        /// 斯帕：La Source 发卡 → Eau Rouge/Raidillon 上坡 → Kemmel 直道 →
        /// 东列下行 → Rivage 发卡（南端）→ 南侧回程西行 → Pouhon →
        /// Blanchimont 长直 → Bus Stop。
        const SPA: [(f32, f32); 20] = [
            (140.0, 100.0),
            (90.0, 70.0),
            (150.0, 35.0),
            (280.0, 55.0),
            (320.0, 100.0),
            (455.0, 195.0),
            (495.0, 230.0),
            (500.0, 320.0),
            (520.0, 395.0),
            (585.0, 412.0),
            (570.0, 458.0),
            (480.0, 455.0),
            (390.0, 430.0),
            (420.0, 340.0),
            (340.0, 310.0),
            (255.0, 245.0),
            (150.0, 195.0),
            (50.0, 140.0),
            (85.0, 118.0),
            (115.0, 152.0),
        ];
        /// 银石：Abbey-Village 复杂段 → Brooklands/Luffield →
        /// 顶部直道 → Maggotts-Becketts 连续 S → Hangar 长直 → Club。
        const SILVERSTONE: [(f32, f32); 18] = [
            (600.0, 100.0),
            (430.0, 75.0),
            (280.0, 60.0),
            (170.0, 100.0),
            (150.0, 180.0),
            (230.0, 240.0),
            (200.0, 320.0),
            (280.0, 400.0),
            (240.0, 470.0),
            (320.0, 500.0),
            (450.0, 520.0),
            (560.0, 500.0),
            (530.0, 440.0),
            (585.0, 395.0),
            (660.0, 300.0),
            (620.0, 180.0),
            (680.0, 120.0),
            (660.0, 95.0),
        ];
        /// 上海："上"字形——大直道（底部，东→西单次通行）→ T1 螺旋收紧（西端）→
        /// esses 东行 → T8/9 折返西行 → T10 发卡 → 顶部东行 → T14 发卡 →
        /// 背直（东侧南下）→ T15/16 接回大直道。
        const SHANGHAI: [(f32, f32); 22] = [
            (150.0, 70.0),
            (120.0, 130.0),
            (150.0, 190.0),
            (225.0, 208.0),
            (290.0, 195.0),
            (350.0, 228.0),
            (410.0, 198.0),
            (458.0, 228.0),
            (520.0, 272.0),
            (470.0, 332.0),
            (300.0, 342.0),
            (205.0, 338.0),
            (152.0, 372.0),
            (198.0, 412.0),
            (300.0, 432.0),
            (420.0, 452.0),
            (520.0, 470.0),
            (562.0, 428.0),
            (612.0, 308.0),
            (652.0, 178.0),
            (638.0, 108.0),
            (588.0, 72.0),
        ];

        let pts: &[(f32, f32)] = match self {
            RealTrack::Monza => &MONZA,
            RealTrack::Monaco => &MONACO,
            RealTrack::Spa => &SPA,
            RealTrack::Silverstone => &SILVERSTONE,
            RealTrack::Shanghai => &SHANGHAI,
        };
        // 摩纳哥梳状布局缩到 0.7 倍：车道间距 60→42m（仍远大于 9m 路宽），
        // 圈长回到街赛量级
        let scale = if self == RealTrack::Monaco { 0.7 } else { 1.0 };
        pts.iter()
            .map(|&(x, y)| Vec2::new(x * scale, y * scale))
            .collect()
    }

    pub fn to_track_gen_config(self) -> TrackGenConfig {
        TrackGenConfig {
            shape: TrackShape::Real(self),
            width: self.width(),
            ..Default::default()
        }
    }
}

/// 赛道生成参数。
#[derive(Clone, Debug)]
pub struct TrackGenConfig {
    pub shape: TrackShape,
    pub n_control_points: usize,
    pub radius: f32,
    /// 谐波径向扰动总幅度（0..1），0 = 正圆（仅 Harmonic 用）。
    pub radius_noise: f32,
    /// 控制点角度扰动（弧度），会被钳制以保持极角有序（仅 Harmonic 用）。
    pub angular_jitter: f32,
    /// 径向谐波阶数（2..=harmonics，权重随机分配，仅 Harmonic 用）。
    pub harmonics: usize,
    /// 赛道基准宽（米）。
    pub width: f32,
    /// 沿弧长的宽度起伏幅度（0 = 等宽，0.25 → 收窄到 75%/加宽到 125%）。
    pub width_noise: f32,
    /// 中心线弧长均匀采样数。
    pub samples: usize,
    /// 最小转弯半径下限（米），保证可驾驶性。
    pub min_corner_radius: f32,
    /// 要求至少存在一个转弯半径 ≤ 此值的真实弯道（防止生成"圆"）。
    /// None = 不限制。noise/jitter 均为 0 时视为显式正圆，此约束忽略。
    pub max_corner_radius: Option<f32>,
}

impl Default for TrackGenConfig {
    fn default() -> Self {
        Self {
            shape: TrackShape::Harmonic,
            n_control_points: 14,
            radius: 130.0,
            radius_noise: 0.24,
            angular_jitter: 0.16,
            harmonics: 3,
            width: 13.0,
            width_noise: 0.0,
            samples: 256,
            min_corner_radius: 25.0,
            max_corner_radius: Some(90.0),
        }
    }
}

impl TrackGenConfig {
    /// 简单（接近正圆的温和波浪，适合首次训练）。
    pub fn easy() -> Self {
        Self {
            n_control_points: 12,
            radius: 120.0,
            radius_noise: 0.10,
            angular_jitter: 0.08,
            harmonics: 2,
            width: 16.0,
            min_corner_radius: 60.0,
            max_corner_radius: None,
            ..Default::default()
        }
    }

    /// 中等（默认）：谐波形，有真实弯角与直道。
    pub fn medium() -> Self {
        Self::default()
    }

    /// 困难：闭合随机游走（发卡弯/S 弯/平行路段）+ 窄路 + 宽窄起伏。
    pub fn hard() -> Self {
        Self {
            shape: TrackShape::RandomWalk,
            n_control_points: 18,
            radius: 135.0,
            width: 11.0,
            width_noise: 0.22,
            min_corner_radius: 13.0,
            max_corner_radius: Some(45.0),
            ..Default::default()
        }
    }
}

/// 中心线采样点。
#[derive(Clone, Copy, Debug)]
pub struct TrackSample {
    pub pos: Vec2,
    /// 行进方向单位切向。
    pub tangent: Vec2,
    /// 切向的左法向（= tangent.perp()）。
    pub normal: Vec2,
    /// 累计弧长 [0, length)。
    pub s: f32,
    /// 该点的半宽（可变宽度）。
    pub half_width: f32,
}

pub struct Track {
    pub samples: Vec<TrackSample>,
    /// 内外墙线段（相邻采样点的墙点连线），raycast 与渲染共用。长度 2*samples.len()。
    pub wall_segments: Vec<[Vec2; 2]>,
    /// 墙点环：前 samples.len() 个为左墙，后 samples.len() 个为右墙。
    pub wall_points: Vec<Vec2>,
    pub width: f32,
    pub length: f32,
    pub bbox_min: Vec2,
    pub bbox_max: Vec2,
}

/// `locate` 的结果：车在赛道上的位置。
#[derive(Clone, Copy, Debug)]
pub struct TrackLocate {
    pub sample_idx: usize,
    /// 带符号横向偏移（左正右负，米）。
    pub lateral: f32,
    /// 精确弧长位置 [0, length)（采样点弧长 + 沿切向投影）。
    pub s: f32,
    pub tangent: Vec2,
    /// 所在位置的赛道半宽（可变宽度）。
    pub half_width: f32,
}

impl Track {
    /// 生成赛道。
    ///
    /// 约束：① 转弯半径 ≥ min（可驾驶）且 ≤ max（防止生成"圆"）
    /// ② 非相邻路段间距 > 赛道宽（防止自交/墙体重叠）。
    /// 不达标换 sub-seed 重试（最多 200 次），按 (合规, 可驾驶, 有弯) 打分取最优兜底。
    pub fn generate(seed: u64, cfg: &TrackGenConfig) -> Track {
        // 真实赛道：固定轮廓，无重试可言；防自交失败只告警（设计问题需修坐标）
        if let TrackShape::Real(t) = cfg.shape {
            let track = generate_once(seed, cfg);
            if !track.passes_proximity() {
                eprintln!("警告：{}（近似）存在墙体重叠，需要修正控制点", t.label());
            }
            return track;
        }
        // noise 与 jitter 均为 0 且星形，视为显式正圆，跳过"必须有弯"约束
        let explicit_circle = cfg.shape == TrackShape::Harmonic
            && cfg.radius_noise <= 0.0
            && cfg.angular_jitter <= 0.0;
        let max_corner = cfg.max_corner_radius.filter(|_| !explicit_circle);

        let mut best: Option<(u32, Track)> = None;
        for attempt in 0..200u64 {
            let track = generate_once(
                seed.wrapping_add(attempt.wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1),
                cfg,
            );
            let r = track.min_corner_radius();
            let drivable = r >= cfg.min_corner_radius;
            let has_corner = max_corner.map_or(true, |max| r <= max);
            let proximity_ok = track.passes_proximity();
            if drivable && has_corner && proximity_ok {
                return track;
            }
            // 打分：防自交 > 可驾驶 > 有真弯
            let score = (proximity_ok as u32) * 4 + (drivable as u32) * 2 + has_corner as u32;
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, track));
            }
        }
        best.map(|(_, t)| t)
            .unwrap_or_else(|| generate_once(seed, cfg))
    }

    /// 最近中心线采样点及其横向/纵向位置。
    /// `hint` 为上一次的采样索引，给出时只搜索附近窗口（O(窗口)）。
    pub fn locate(&self, pos: Vec2, hint: Option<usize>) -> TrackLocate {
        let m = self.samples.len();
        let (lo, hi) = match hint {
            Some(h) => (h.saturating_sub(48), h + 48),
            None => (0, m),
        };
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for k in lo..hi {
            let i = k % m;
            let d = (self.samples[i].pos - pos).len_sqr();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        let sm = &self.samples[best];
        let delta = pos - sm.pos;
        let along = delta.dot(sm.tangent);
        let lateral = delta.dot(sm.normal);
        let mut s = sm.s + along;
        if s < 0.0 {
            s += self.length;
        } else if s >= self.length {
            s -= self.length;
        }
        TrackLocate {
            sample_idx: best,
            lateral,
            s,
            tangent: sm.tangent,
            half_width: sm.half_width,
        }
    }

    /// 射线到最近墙体的距离，截断于 `max_dist`。`dir` 必须是单位向量。
    pub fn raycast(&self, origin: Vec2, dir: Vec2, max_dist: f32) -> f32 {
        let mut best = max_dist;
        for seg in &self.wall_segments {
            if let Some(t) = ray_segment(origin, dir, seg[0], seg[1]) {
                if t < best {
                    best = t;
                }
            }
        }
        best
    }

    /// 渲染用：左右墙点环。
    pub fn boundary_polylines(&self) -> (&[Vec2], &[Vec2]) {
        let m = self.samples.len();
        self.wall_points.split_at(m)
    }

    /// 诊断用：不做约束挑选，直接生成一次。
    pub fn generate_raw_for_diag(seed: u64, cfg: &TrackGenConfig) -> Track {
        generate_once(seed, cfg)
    }

    /// 中心线最小转弯半径（米）。
    pub fn min_corner_radius(&self) -> f32 {
        let m = self.samples.len();
        let mut min_r = f32::MAX;
        for i in 0..m {
            let a = self.samples[(i + m - 1) % m].tangent.angle();
            let b = self.samples[(i + 1) % m].tangent.angle();
            let dtheta = norm_angle(b - a).abs();
            if dtheta > 1e-6 {
                let ds = 2.0 * self.length / m as f32;
                min_r = min_r.min(ds / dtheta);
            }
        }
        min_r
    }

    /// 防自交校验：非相邻采样点之间的距离必须大于赛道宽，
    /// 否则墙体会重叠/穿越（随机游走形状必需，星形恒通过）。
    pub fn passes_proximity(&self) -> bool {
        let m = self.samples.len();
        let ds = self.length / m as f32;
        // 邻接豁免窗口：弧长上相距 > 窗口的点才算"远处"
        let guard = (self.width * 1.8 / ds).ceil() as usize + 2;
        let min_d = self.width * 1.15;
        for i in 0..m {
            let pi = self.samples[i].pos;
            for j in (i + guard)..m {
                // 环形首尾相邻豁免
                if j - i + guard > m {
                    break;
                }
                let d = (self.samples[j].pos - pi).len_sqr();
                if d < min_d * min_d {
                    return false;
                }
            }
        }
        true
    }
}

fn generate_once(seed: u64, cfg: &TrackGenConfig) -> Track {
    let mut rng = Rng::new(seed);
    let ctrl = match cfg.shape {
        TrackShape::Harmonic => harmonic_ctrl(&mut rng, cfg),
        TrackShape::RandomWalk => random_walk_ctrl(&mut rng, cfg),
        TrackShape::Real(t) => t.ctrl_points(),
    };

    // 闭合 Catmull-Rom 密集采样（每段 32 份）
    let dense = catmull_rom_dense(&ctrl, 32);
    let dn = dense.len();

    // 累计弧长
    let mut cum = vec![0.0f32; dn + 1];
    for i in 0..dn {
        let a = dense[i];
        let b = dense[(i + 1) % dn];
        cum[i + 1] = cum[i] + (b - a).len();
    }
    let total = cum[dn];

    // 弧长均匀重采样
    let m = cfg.samples.max(16);
    let mut pos = Vec::with_capacity(m);
    let mut seg = 0usize;
    for k in 0..m {
        let target = total * (k as f32) / (m as f32);
        while seg + 1 < cum.len() && cum[seg + 1] < target {
            seg += 1;
        }
        let a = dense[seg % dn];
        let b = dense[(seg + 1) % dn];
        let seg_len = cum[seg + 1] - cum[seg];
        let t = if seg_len > 1e-9 {
            (target - cum[seg]) / seg_len
        } else {
            0.0
        };
        pos.push(a + (b - a) * t);
    }

    // 沿弧长的宽度起伏（一周 3 个收窄/加宽周期；width_noise=0 时恒为基准宽）
    let hw_base = cfg.width * 0.5;
    let w_phase = rng.next_f32() * std::f32::consts::TAU;
    let width_at = |frac: f32| -> f32 {
        let wave = (std::f32::consts::TAU * 3.0 * frac + w_phase).sin() * cfg.width_noise;
        (hw_base * (1.0 + wave)).clamp(hw_base * 0.55, hw_base * 1.25)
    };

    // 切向（中心差分）/ 法向 / 弧长 / 宽度
    let ds = total / m as f32;
    let mut samples = Vec::with_capacity(m);
    let mut bbox_min = Vec2::new(f32::MAX, f32::MAX);
    let mut bbox_max = Vec2::new(f32::MIN, f32::MIN);
    for i in 0..m {
        let tangent = (pos[(i + 1) % m] - pos[(i + m - 1) % m]).normalized();
        let p = pos[i];
        bbox_min = Vec2::new(bbox_min.x.min(p.x), bbox_min.y.min(p.y));
        bbox_max = Vec2::new(bbox_max.x.max(p.x), bbox_max.y.max(p.y));
        samples.push(TrackSample {
            pos: p,
            tangent,
            normal: tangent.perp(),
            s: i as f32 * ds,
            half_width: width_at(i as f32 / m as f32),
        });
    }

    // 墙点与墙段（raycast 与渲染共用），宽度逐点可变
    let mut wall_points = Vec::with_capacity(2 * m);
    let mut wall_segments = Vec::with_capacity(2 * m);
    for s in &samples {
        wall_points.push(s.pos + s.normal * s.half_width);
    }
    for s in &samples {
        wall_points.push(s.pos - s.normal * s.half_width);
    }
    for i in 0..m {
        let j = (i + 1) % m;
        wall_segments.push([wall_points[i], wall_points[j]]);
        wall_segments.push([wall_points[m + i], wall_points[m + j]]);
    }

    Track {
        samples,
        wall_segments,
        wall_points,
        width: cfg.width,
        length: total,
        bbox_min,
        bbox_max,
    }
}

/// 谐波控制点：r(θ) = R·(1 + Σₖ aₖ·sin(kθ + φₖ))，多频起伏 → 直道 + 弯角。
fn harmonic_ctrl(rng: &mut Rng, cfg: &TrackGenConfig) -> Vec<Vec2> {
    let n = cfg.n_control_points.max(4);
    // 谐波阶数不超过 n/3，避免欠采样
    let harm = cfg.harmonics.min(n / 3);
    // jitter 钳制：保持极角有序（星形不自交）
    let jitter = cfg
        .angular_jitter
        .min(std::f32::consts::PI / n as f32 * 0.6);

    // 随机分配各阶谐波的权重与相位
    let mut weights = Vec::with_capacity(harm);
    let mut phases = Vec::with_capacity(harm);
    let mut wsum = 0.0f32;
    for _ in 0..harm {
        let w = rng.next_range(0.3, 1.0);
        wsum += w;
        weights.push(w);
        phases.push(rng.next_range(0.0, std::f32::consts::TAU));
    }

    let mut ctrl = Vec::with_capacity(n);
    for i in 0..n {
        let theta = std::f32::consts::TAU * (i as f32) / (n as f32)
            + rng.next_range(-jitter, jitter);
        let mut r = cfg.radius;
        for k in 0..harm {
            let amp = cfg.radius_noise * weights[k] / wsum;
            r += cfg.radius * amp * ((k as f32 + 2.0) * theta + phases[k]).sin();
        }
        ctrl.push(Vec2::from_angle(theta) * r);
    }
    ctrl
}

/// 闭合随机游走控制点：非星形，可产生发卡弯、S 弯、平行路段。
/// 闭合误差线性均摊回各点，再居中并归一到目标尺度。
fn random_walk_ctrl(rng: &mut Rng, cfg: &TrackGenConfig) -> Vec<Vec2> {
    let n = cfg.n_control_points.max(10);
    let step = std::f32::consts::TAU * cfg.radius / n as f32;
    let mut pts = Vec::with_capacity(n);
    let mut pos = Vec2::ZERO;
    let mut heading = rng.next_f32() * std::f32::consts::TAU;
    for _ in 0..n {
        pts.push(pos);
        // 平滑随机转向；偶尔较急的转向制造弯角（过急会被可驾驶性校验拒绝）
        let turn = if rng.next_f32() < 0.12 {
            rng.next_range(-1.25, 1.25)
        } else {
            rng.next_range(-0.7, 0.7)
        };
        heading += turn;
        let len = step * rng.next_range(0.7, 1.3);
        pos = pos + Vec2::from_angle(heading) * len;
    }

    // 闭合：把 (终点→起点) 的误差线性均摊到各控制点
    let err = pts[0] - pos;
    for (i, p) in pts.iter_mut().enumerate() {
        *p = *p + err * (i as f32 / n as f32);
    }

    // 两轮拉普拉斯平滑：磨掉锐角顶点（否则 Catmull-Rom 会产生
    // 极小的局部曲率，弯径校验大量失败）
    for _ in 0..2 {
        let mut next = pts.clone();
        for i in 0..n {
            let prev = pts[(i + n - 1) % n];
            let cur = pts[i];
            let after = pts[(i + 1) % n];
            next[i] = prev * 0.25 + cur * 0.5 + after * 0.25;
        }
        pts = next;
    }

    // 居中并归一到目标平均半径
    let mut cmin = Vec2::new(f32::MAX, f32::MAX);
    let mut cmax = Vec2::new(f32::MIN, f32::MIN);
    for p in &pts {
        cmin = Vec2::new(cmin.x.min(p.x), cmin.y.min(p.y));
        cmax = Vec2::new(cmax.x.max(p.x), cmax.y.max(p.y));
    }
    let center = (cmin + cmax) * 0.5;
    let mean_r: f32 = pts.iter().map(|p| (*p - center).len()).sum::<f32>() / n as f32;
    let scale = if mean_r > 1e-3 { cfg.radius / mean_r } else { 1.0 };
    pts.iter()
        .map(|p| (*p - center) * scale)
        .collect()
}

/// 闭合 Catmull-Rom：每段 `subdiv` 份，返回稠密点列（长度 n*subdiv，首尾不重复）。
fn catmull_rom_dense(ctrl: &[Vec2], subdiv: usize) -> Vec<Vec2> {
    let n = ctrl.len();
    let mut out = Vec::with_capacity(n * subdiv);
    for i in 0..n {
        let p0 = ctrl[(i + n - 1) % n];
        let p1 = ctrl[i];
        let p2 = ctrl[(i + 1) % n];
        let p3 = ctrl[(i + 2) % n];
        for k in 0..subdiv {
            let t = k as f32 / subdiv as f32;
            out.push(catmull_rom(p0, p1, p2, p3, t));
        }
    }
    out
}

fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    (p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (p3 - p0 + p1 * 3.0 - p2 * 3.0) * t3)
        * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_and_monotonic() {
        for seed in [1u64, 7, 42, 2026] {
            let t = Track::generate(seed, &TrackGenConfig::default());
            let m = t.samples.len();
            assert_eq!(m, TrackGenConfig::default().samples);
            // 弧长严格单调
            for i in 1..m {
                assert!(t.samples[i].s > t.samples[i - 1].s);
            }
            // 闭合：首尾采样点相邻
            let d = (t.samples[0].pos - t.samples[m - 1].pos).len();
            assert!(d < 6.0 * t.length / m as f32, "gap={d}");
            assert_eq!(t.wall_segments.len(), 2 * m);
            assert!(t.length > 400.0, "length={}", t.length);
            // 可驾驶性：默认参数生成的赛道都应通过校验
            assert!(t.min_corner_radius() >= TrackGenConfig::default().min_corner_radius);
        }
    }

    #[test]
    fn default_tracks_have_real_corners() {
        // 中等难度：双向约束内——不是圆，也不至于开不了
        let cfg = TrackGenConfig::default();
        for seed in [1u64, 7, 42, 2026] {
            let t = Track::generate(seed, &cfg);
            let r = t.min_corner_radius();
            assert!(r >= cfg.min_corner_radius, "seed={seed} 弯太急 r={r}");
            assert!(
                r <= cfg.max_corner_radius.unwrap() + 1e-3,
                "seed={seed} 太圆 r={r}"
            );
        }
    }

    #[test]
    fn hard_tracks_are_tight() {
        let cfg = TrackGenConfig::hard();
        assert_eq!(cfg.shape, TrackShape::RandomWalk);
        for seed in 0..8u64 {
            let t = Track::generate(seed, &cfg);
            let r = t.min_corner_radius();
            assert!(r >= cfg.min_corner_radius, "seed={seed} 弯太急 r={r}");
            assert!(r <= cfg.max_corner_radius.unwrap(), "seed={seed} 太圆 r={r}");
            // 随机游走形状必须通过防自交校验
            assert!(t.passes_proximity(), "seed={seed} 赛道自交/墙体重叠");
        }
    }

    #[test]
    fn real_tracks_generate_cleanly() {
        for t in RealTrack::ALL {
            let cfg = t.to_track_gen_config();
            let track = Track::generate(0, &cfg);
            assert!(track.length > 600.0, "{} 长度 {}", t.label(), track.length);
            assert!(
                track.passes_proximity(),
                "{}（近似）墙体重叠/自交",
                t.label()
            );
            // 可驾驶性宽松下限：太小说明控制点画坏了
            let r = track.min_corner_radius();
            assert!(r > 6.0, "{} 最小弯径 {} 过小", t.label(), r);
        }
    }

    #[test]
    fn tracks_vary_across_seeds() {
        // 种子间弯径差异明显 → 不是同一张"圆"图换皮
        let cfg = TrackGenConfig::default();
        let corners: Vec<f32> = (0..16u64)
            .map(|seed| Track::generate(seed, &cfg).min_corner_radius())
            .collect();
        let min = corners.iter().cloned().fold(f32::MAX, f32::min);
        let max = corners.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (max - min) / min > 0.15,
            "弯径差异过小: {min}..{max} ({corners:?})"
        );
    }

    #[test]
    fn circle_track() {
        let cfg = TrackGenConfig {
            radius_noise: 0.0,
            angular_jitter: 0.0,
            ..Default::default()
        };
        let t = Track::generate(0, &cfg);
        for s in &t.samples {
            let r = s.pos.len();
            assert!((r - cfg.radius).abs() < 8.0, "r={r}");
        }
        assert!(t.min_corner_radius() > 80.0, "r={}", t.min_corner_radius());
        // 弧长 ≈ 周长
        assert!((t.length - std::f32::consts::TAU * cfg.radius).abs() < 25.0);
    }

    #[test]
    fn locate_start() {
        let t = Track::generate(3, &TrackGenConfig::default());
        let start = t.samples[0].pos;
        let loc = t.locate(start, None);
        assert!(loc.lateral.abs() < 0.8, "lateral={}", loc.lateral);
        assert!(loc.s < 5.0 || loc.s > t.length - 5.0, "s={}", loc.s);
    }

    #[test]
    fn locate_hint_agrees_with_full_scan() {
        let t = Track::generate(11, &TrackGenConfig::default());
        for i in (0..t.samples.len()).step_by(17) {
            let p = t.samples[i].pos + t.samples[i].normal * 3.0;
            let full = t.locate(p, None);
            let hinted = t.locate(p, Some(i));
            assert_eq!(full.sample_idx, hinted.sample_idx);
            assert!((full.lateral - 3.0).abs() < 1.5, "lateral={}", full.lateral);
        }
    }

    #[test]
    fn raycast_hits_wall() {
        let t = Track::generate(3, &TrackGenConfig::default());
        let s0 = t.samples[0];
        let d = t.raycast(s0.pos, s0.normal, 100.0);
        assert!((d - t.width * 0.5).abs() < 2.5, "d={d}");
        // 反方向（右墙）
        let d2 = t.raycast(s0.pos, -s0.normal, 100.0);
        assert!((d2 - t.width * 0.5).abs() < 2.5, "d={d2}");
    }
}
