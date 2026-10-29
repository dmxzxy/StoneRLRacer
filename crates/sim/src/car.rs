//! 车辆物理：扭矩/功率双限制驱动力 + 抓地上限转向。
//!
//! 驱动力 F = min(扭矩力, 功率/速度)：低速受扭矩限制（弹射感），
//! 高速受功率限制（再踩也快不起来）。加速度 = F/车重。
//! 转向 ω = v/L·tan(δ)，受最大横向加速度（抓地）钳制，模拟不足转向。

use crate::math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct CarState {
    pub pos: Vec2,
    /// 车头朝向（弧度，世界系）。
    pub heading: f32,
    pub speed: f32,
}

/// 车辆物理参数（有量纲，可自由调校）。
#[derive(Clone, Copy, Debug)]
pub struct CarConfig {
    /// 车重（kg）。
    pub mass: f32,
    /// 轮上扭矩力（N）：低速加速度上限 = F/m。
    pub torque_force: f32,
    /// 功率（W）：高速时驱动力 = P/v。
    pub power: f32,
    /// 刹车力（N）。
    pub brake_force: f32,
    /// 极速钳制（m/s）。
    pub max_speed: f32,
    /// 二次风阻系数（加速度量纲：a = k·v·|v|）。
    pub drag_k: f32,
    /// 滚动阻力（m/s²）。
    pub rolling: f32,
    /// 前轮最大转角（弧度）。
    pub max_steer: f32,
    /// 轴距（m）：越长越稳、转弯半径越大。
    pub wheelbase: f32,
    /// 车长（m，渲染尺寸）。
    pub length: f32,
    /// 半车宽（m，撞墙判定）。
    pub half_width: f32,
    /// 抓地：最大横向加速度（m/s²）。
    pub max_lat_accel: f32,
    /// 倒车极速（m/s，负值）。
    pub min_speed: f32,
}

impl Default for CarConfig {
    fn default() -> Self {
        Self::hatch()
    }
}

impl CarConfig {
    /// 均衡掀背车。
    pub fn hatch() -> Self {
        Self {
            mass: 1200.0,
            torque_force: 6500.0,
            power: 130_000.0,
            brake_force: 13_000.0,
            max_speed: 32.0,
            drag_k: 0.0034,
            rolling: 0.6,
            max_steer: 0.45,
            wheelbase: 2.7,
            length: 4.3,
            half_width: 0.9,
            max_lat_accel: 12.0,
            min_speed: -6.0,
        }
    }

    /// 跑车：大马力，抓地跟得上。
    pub fn sport() -> Self {
        Self {
            mass: 1350.0,
            torque_force: 8500.0,
            power: 260_000.0,
            brake_force: 17_000.0,
            max_speed: 40.0,
            drag_k: 0.0031,
            rolling: 0.6,
            max_steer: 0.42,
            wheelbase: 2.9,
            length: 4.5,
            half_width: 0.95,
            max_lat_accel: 14.0,
            min_speed: -6.0,
        }
    }

    /// 肌肉车：扭矩爆炸、车身重、抓地差——起步猛、弯道推头。
    pub fn muscle() -> Self {
        Self {
            mass: 1650.0,
            torque_force: 11_000.0,
            power: 220_000.0,
            brake_force: 14_000.0,
            max_speed: 34.0,
            drag_k: 0.0032,
            rolling: 0.6,
            max_steer: 0.48,
            wheelbase: 3.1,
            length: 5.0,
            half_width: 1.05,
            max_lat_accel: 11.0,
            min_speed: -6.0,
        }
    }

    /// 卡丁车：极轻、短轴距、高抓地——慢但指哪打哪。
    pub fn kart() -> Self {
        Self {
            mass: 200.0,
            torque_force: 2600.0,
            power: 30_000.0,
            brake_force: 5200.0,
            max_speed: 24.0,
            drag_k: 0.010,
            rolling: 0.5,
            max_steer: 0.55,
            wheelbase: 1.6,
            length: 1.9,
            half_width: 0.8,
            max_lat_accel: 18.0,
            min_speed: -5.0,
        }
    }

    /// 方程式：轻 + 大马力 + 极限抓地——快到需要提前半段刹车。
    pub fn formula() -> Self {
        Self {
            mass: 750.0,
            torque_force: 9000.0,
            power: 400_000.0,
            brake_force: 22_000.0,
            max_speed: 46.0,
            drag_k: 0.006,
            rolling: 0.6,
            max_steer: 0.40,
            wheelbase: 3.2,
            length: 5.2,
            half_width: 0.95,
            max_lat_accel: 20.0,
            min_speed: -5.0,
        }
    }

    /// 夹掉非法参数（NaN/Inf/越界），保证训练与渲染不会被调校滑杆喂坏。
    ///
    /// 场景：滑杆直接改 `CarConfig`，极端值会让积分发散出 NaN，撞墙判定失效，
    /// 车"飞"到地图外；渲染层拿到 NaN 坐标也会出问题。统一在这里兜底。
    pub fn sanitized(self) -> Self {
        let fin = |v: f32, def: f32| if v.is_finite() { v } else { def };
        Self {
            mass: fin(self.mass, 1200.0).clamp(80.0, 3000.0),
            torque_force: fin(self.torque_force, 6500.0).clamp(200.0, 40_000.0),
            power: fin(self.power, 130_000.0).clamp(5_000.0, 900_000.0),
            brake_force: fin(self.brake_force, 13_000.0).clamp(500.0, 60_000.0),
            max_speed: fin(self.max_speed, 32.0).clamp(5.0, 120.0),
            drag_k: fin(self.drag_k, 0.0034).clamp(0.0, 0.2),
            rolling: fin(self.rolling, 0.6).clamp(0.0, 5.0),
            max_steer: fin(self.max_steer, 0.45).clamp(0.1, 0.9),
            wheelbase: fin(self.wheelbase, 2.7).clamp(0.8, 5.0),
            length: fin(self.length, 4.3).clamp(1.2, 7.0),
            half_width: fin(self.half_width, 0.9).clamp(0.3, 2.5),
            max_lat_accel: fin(self.max_lat_accel, 12.0).clamp(1.0, 40.0),
            min_speed: fin(self.min_speed, -6.0).clamp(-30.0, -0.5),
        }
    }

    /// 当前速度下的驱动力上限（N）。
    pub fn drive_force_cap(&self, speed: f32) -> f32 {
        (self.power / speed.abs().max(3.0)).min(self.torque_force)
    }
}

/// 单步积分。`steer`、`throttle` ∈ [-1, 1]（throttle < 0 为刹车/倒车）。
pub fn integrate(car: &mut CarState, steer: f32, throttle: f32, cfg: &CarConfig, dt: f32) {
    let steer = steer.clamp(-1.0, 1.0);
    let throttle = throttle.clamp(-1.0, 1.0);
    let steer_angle = steer * cfg.max_steer;

    // 驱动/刹车 → 加速度
    let drive_force = if throttle >= 0.0 {
        throttle * cfg.drive_force_cap(car.speed)
    } else {
        throttle * cfg.brake_force
    };
    let mut a = drive_force / cfg.mass;
    // 二次风阻（与速度方向相反）+ 滚动阻力
    a -= cfg.drag_k * car.speed.abs() * car.speed;
    a -= car.speed.signum() * cfg.rolling;

    let mut v = car.speed + a * dt;
    v = v.clamp(cfg.min_speed, cfg.max_speed);
    // 松油门低速时消除爬行
    if throttle == 0.0 && v.abs() < 0.3 {
        v = 0.0;
    }

    // 转向：ω = v/L·tan(δ)，并用横向加速度上限裁剪（高速不足转向）
    let mut omega = (v / cfg.wheelbase) * steer_angle.tan();
    let max_omega = cfg.max_lat_accel / v.abs().max(1.0);
    omega = omega.clamp(-max_omega, max_omega);
    car.heading += omega * dt;

    car.pos = car.pos + Vec2::from_angle(car.heading) * v * dt;
    car.speed = v;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_throttle_accelerates() {
        let cfg = CarConfig::hatch();
        let mut car = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
        for _ in 0..100 {
            integrate(&mut car, 0.0, 1.0, &cfg, 0.05);
        }
        assert!(car.speed > 15.0, "speed={}", car.speed);
        assert!(car.pos.x > 30.0);
    }

    #[test]
    fn top_speed_bounded() {
        for (cfg, expect) in [
            (CarConfig::hatch(), 32.0),
            (CarConfig::sport(), 40.0),
            (CarConfig::kart(), 24.0),
        ] {
            let mut car = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
            for _ in 0..6000 {
                integrate(&mut car, 0.0, 1.0, &cfg, 0.05);
            }
            assert!(car.speed <= cfg.max_speed + 1e-3);
            assert!(
                (car.speed - expect).abs() < 4.0,
                "{:?} speed={} expect~{expect}",
                cfg.length,
                car.speed
            );
        }
    }

    #[test]
    fn torque_vs_power_regimes() {
        // 低速：受扭矩限制——肌肉车（11000N/1650kg=6.7）比跑车（8500/1350=6.3）猛
        let mut muscle = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
        let mut sport = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
        for _ in 0..10 {
            integrate(&mut muscle, 0.0, 1.0, &CarConfig::muscle(), 0.05);
            integrate(&mut sport, 0.0, 1.0, &CarConfig::sport(), 0.05);
        }
        assert!(muscle.speed > sport.speed, "低速肌肉车应更快: {} vs {}", muscle.speed, sport.speed);
        // 高速：受功率限制——跑车（260kW）反超肌肉车（220kW）
        let mut muscle = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 25.0 };
        let mut sport = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 25.0 };
        for _ in 0..200 {
            integrate(&mut muscle, 0.0, 1.0, &CarConfig::muscle(), 0.05);
            integrate(&mut sport, 0.0, 1.0, &CarConfig::sport(), 0.05);
        }
        assert!(sport.speed > muscle.speed, "高速跑车应更快: {} vs {}", sport.speed, muscle.speed);
    }

    #[test]
    fn mass_matters() {
        // 同样动力，重的慢
        let light = CarConfig { mass: 800.0, ..CarConfig::hatch() };
        let heavy = CarConfig { mass: 1800.0, ..CarConfig::hatch() };
        let mut a = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
        let mut b = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 0.0 };
        for _ in 0..60 {
            integrate(&mut a, 0.0, 1.0, &light, 0.05);
            integrate(&mut b, 0.0, 1.0, &heavy, 0.05);
        }
        assert!(a.speed > b.speed + 3.0, "轻车应明显更快: {} vs {}", a.speed, b.speed);
    }

    #[test]
    fn steer_changes_heading_and_grip_limits() {
        let cfg = CarConfig::hatch();
        let mut car = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 8.0 };
        let mut prev_heading = car.heading;
        for _ in 0..200 {
            integrate(&mut car, 1.0, 0.4, &cfg, 0.05);
            assert!(car.heading >= prev_heading, "左转 heading 应增加");
            prev_heading = car.heading;
        }
        // 高速满舵：横向加速度不能超限 → 角速度被裁剪
        let mut fast = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 28.0 };
        let h0 = fast.heading;
        integrate(&mut fast, 1.0, 1.0, &cfg, 0.05);
        let omega = (fast.heading - h0) / 0.05;
        assert!(omega <= cfg.max_lat_accel / 28.0 + 1e-3, "omega={omega}");
    }

    #[test]
    fn brake_reverses_slowly() {
        let cfg = CarConfig::hatch();
        let mut car = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 10.0 };
        for _ in 0..100 {
            integrate(&mut car, 0.0, -1.0, &cfg, 0.05);
        }
        assert!(car.speed < -0.5 && car.speed >= cfg.min_speed, "speed={}", car.speed);
    }

    #[test]
    fn no_throttle_coasts_to_stop() {
        let cfg = CarConfig::hatch();
        let mut car = CarState { pos: Vec2::ZERO, heading: 0.0, speed: 10.0 };
        for _ in 0..2000 {
            integrate(&mut car, 0.0, 0.0, &cfg, 0.05);
        }
        assert!(car.speed.abs() < 0.35, "speed={}", car.speed);
    }
}
