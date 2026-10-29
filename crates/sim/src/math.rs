//! 最小 2D 数学库。

use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// 世界坐标向量（米）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y
    }

    pub fn len_sqr(self) -> f32 {
        self.dot(self)
    }

    pub fn len(self) -> f32 {
        self.len_sqr().sqrt()
    }

    /// 逆时针旋转 90°。
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    pub fn normalized(self) -> Self {
        let l = self.len();
        if l < 1e-9 {
            Self::new(1.0, 0.0)
        } else {
            Self::new(self.x / l, self.y / l)
        }
    }

    pub fn from_angle(angle: f32) -> Self {
        Self::new(angle.cos(), angle.sin())
    }

    /// 与 x 正轴的夹角（弧度）。
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, o: Vec2) {
        *self = *self + o;
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

/// 线性插值。
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 归一化角度到 [-π, π]。
pub fn norm_angle(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut a = a.rem_euclid(tau);
    if a > std::f32::consts::PI {
        a -= tau;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_angle_wraps() {
        assert!((norm_angle(0.0)).abs() < 1e-6);
        assert!((norm_angle(std::f32::consts::PI) - std::f32::consts::PI).abs() < 1e-6);
        // 3π ≡ π (mod 2π)
        assert!((norm_angle(3.0 * std::f32::consts::PI).abs() - std::f32::consts::PI).abs() < 1e-4);
        assert!((norm_angle(-std::f32::consts::FRAC_PI_2) + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((norm_angle(2.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn perp_rotates_ccw() {
        let v = Vec2::new(1.0, 0.0);
        let p = v.perp();
        assert!((p.x - 0.0).abs() < 1e-6 && (p.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec_ops() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, -1.0);
        assert_eq!(a + b, Vec2::new(4.0, 1.0));
        assert_eq!(a - b, Vec2::new(-2.0, 3.0));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(a.dot(b), 1.0);
        assert!((Vec2::new(3.0, 4.0).len() - 5.0).abs() < 1e-6);
    }
}
