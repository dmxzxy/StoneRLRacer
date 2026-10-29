//! 仿真总配置。

use crate::car::CarConfig;
use crate::reward::RewardConfig;
use crate::track::TrackGenConfig;

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub track: TrackGenConfig,
    pub reward: RewardConfig,
    pub car: CarConfig,
    /// 单局最大步数（超时终止）。
    pub max_steps: u32,
    /// 起步时是否加随机横向偏移（训练用，观赛关闭）。
    pub start_noise: bool,
    /// 完圈后不停车，接着跑下一圈（展示厅用）。
    ///
    /// false（训练默认）：跑完一圈就是一局结束，episode 终止；
    /// true：完圈只把圈计时/里程回绕，车继续开，撞墙/超时仍然复位。
    pub continuous_laps: bool,
}

impl SimConfig {
    /// 夹掉非法/极端参数。UI 下发环境、训练线程建环境、观赛环境都走这里，
    /// 保证三方看到的是同一份干净配置。
    pub fn sanitized(mut self) -> Self {
        self.car = self.car.sanitized();
        let t = &mut self.track;
        t.width = t.width.clamp(6.0, 40.0);
        t.width_noise = t.width_noise.clamp(0.0, 0.5);
        t.samples = t.samples.clamp(64, 1024);
        t.radius = t.radius.clamp(50.0, 400.0);
        t.min_corner_radius = t.min_corner_radius.clamp(8.0, 200.0);
        self.max_steps = self.max_steps.clamp(200, 200_000);
        self.reward.w_progress = self.reward.w_progress.clamp(0.0, 5.0);
        self.reward.w_speed = self.reward.w_speed.clamp(0.0, 5.0);
        self.reward.w_center = self.reward.w_center.clamp(0.0, 5.0);
        self.reward.w_time = self.reward.w_time.clamp(0.0, 1.0);
        self.reward.r_crash = self.reward.r_crash.clamp(0.0, 100.0);
        self.reward.r_lap = self.reward.r_lap.clamp(0.0, 100.0);
        self
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            track: TrackGenConfig::default(),
            reward: RewardConfig::default(),
            car: CarConfig::default(),
            max_steps: 3000,
            start_noise: false,
            continuous_laps: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_clamps_garbage() {
        let mut cfg = SimConfig::default();
        cfg.car.mass = f32::NAN;
        cfg.car.max_speed = 1e9;
        cfg.track.width = 0.0;
        cfg.max_steps = 0;
        let c = cfg.sanitized();
        assert!(c.car.mass.is_finite() && c.car.mass > 0.0);
        assert!(c.car.max_speed <= 120.0);
        assert!(c.track.width >= 6.0);
        assert!(c.max_steps >= 200);
    }
}
