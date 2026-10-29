//! 射线-线段求交。

use crate::math::Vec2;

/// 从 `o` 沿单位方向 `d` 发射射线，与线段 `a`-`b` 求交。
/// 返回沿 `d` 的距离（`d` 必须归一化）。平行或反向时返回 `None`。
pub fn ray_segment(o: Vec2, d: Vec2, a: Vec2, b: Vec2) -> Option<f32> {
    let e = b - a;
    let denom = d.x * e.y - d.y * e.x;
    if denom.abs() < 1e-9 {
        return None;
    }
    let ao = a - o;
    let t = (ao.x * e.y - ao.y * e.x) / denom;
    let u = (ao.x * d.y - ao.y * d.x) / denom;
    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aligned_hit() {
        // 沿 +x 打到竖直墙 x=5
        let d = Vec2::new(1.0, 0.0);
        let t = ray_segment(Vec2::ZERO, d, Vec2::new(5.0, -1.0), Vec2::new(5.0, 1.0));
        assert_eq!(t, Some(5.0));
    }

    #[test]
    fn miss_behind_and_beyond() {
        let d = Vec2::new(1.0, 0.0);
        // 墙在射线反方向
        assert_eq!(ray_segment(Vec2::ZERO, d, Vec2::new(-5.0, -1.0), Vec2::new(-5.0, 1.0)), None);
        // 射线从墙侧面掠过
        assert_eq!(ray_segment(Vec2::ZERO, d, Vec2::new(5.0, 1.0), Vec2::new(5.0, 2.0)), None);
    }

    #[test]
    fn parallel_no_hit() {
        let d = Vec2::new(1.0, 0.0);
        assert_eq!(ray_segment(Vec2::ZERO, d, Vec2::new(1.0, 3.0), Vec2::new(5.0, 3.0)), None);
    }

    #[test]
    fn circle_polygon_vs_analytic() {
        // 用 128 段折线近似半径 10 的圆，中心射线距离应接近 10
        let n = 128;
        let pts: Vec<Vec2> = (0..=n)
            .map(|i| Vec2::from_angle(std::f32::consts::TAU * i as f32 / n as f32) * 10.0)
            .collect();
        let o = Vec2::new(-30.0, 0.0);
        let d = Vec2::new(1.0, 0.0);
        let mut best = f32::MAX;
        for w in pts.windows(2) {
            if let Some(t) = ray_segment(o, d, w[0], w[1]) {
                best = best.min(t);
            }
        }
        // 圆心在原点、半径 10，射线起点 (-30,0) → 近端交点 x=-10，距离 20
        // 折线内切于圆，误差 < 半径*(1-cos(π/n)) ≈ 0.003
        assert!((best - 20.0).abs() < 0.05, "best={best}");
    }
}
