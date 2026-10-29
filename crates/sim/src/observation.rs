//! 观测空间：`[f32; 14]` 栈上定长数组（零分配，无头狂跑的关键）。
//!
//! [0..9]  9 条射线距离 / MAX_RAY_DIST（扇形 -90°..+90°，绕车头均布）
//! [9]     speed / max_speed
//! [10]    heading_err：车头与所在采样点切向的带符号夹角 / π
//! [11]    lateral / (width/2)
//! [12]    上一步 steer
//! [13]    上一步 throttle

use crate::action::ContinuousAction;
use crate::car::{CarConfig, CarState};
use crate::math::{norm_angle, Vec2};
use crate::track::{Track, TrackLocate};

pub const N_RAYS: usize = 9;
pub const MAX_RAY_DIST: f32 = 60.0;
pub const OBS_DIM: usize = N_RAYS + 5;

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub data: [f32; OBS_DIM],
}

impl Observation {
    pub fn build(
        track: &Track,
        car: &CarState,
        car_cfg: &CarConfig,
        loc: &TrackLocate,
        last_action: ContinuousAction,
    ) -> Self {
        let mut data = [0.0f32; OBS_DIM];
        for i in 0..N_RAYS {
            let angle = car.heading + ray_angle(i);
            let d = track.raycast(car.pos, Vec2::from_angle(angle), MAX_RAY_DIST);
            data[i] = d / MAX_RAY_DIST;
        }
        data[N_RAYS] = (car.speed / car_cfg.max_speed).clamp(-1.0, 1.0);
        data[N_RAYS + 1] = norm_angle(car.heading - loc.tangent.angle()) / std::f32::consts::PI;
        data[N_RAYS + 2] = (loc.lateral / loc.half_width).clamp(-2.0, 2.0);
        data[N_RAYS + 3] = last_action.steer;
        data[N_RAYS + 4] = last_action.throttle;
        Self { data }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// 第 `i` 条射线相对车头的角度（-90°..+90° 均布）。
pub fn ray_angle(i: usize) -> f32 {
    let frac = i as f32 / (N_RAYS as f32 - 1.0);
    (frac - 0.5) * std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;
    use crate::episode::SimState;
    use crate::track::Track;

    #[test]
    fn ray_angles_span_fan() {
        assert!((ray_angle(0) + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(ray_angle(N_RAYS / 2).abs() < 1e-6);
        assert!((ray_angle(N_RAYS - 1) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn observation_at_start_is_centered() {
        let cfg = SimConfig::default();
        let track = Track::generate(3, &cfg.track);
        let env = SimState::new(track, 3, cfg);
        let obs = env.observation();
        // 中心线起步：航向误差与横向偏移近 0
        assert!(obs.data[N_RAYS + 1].abs() < 0.05, "heading_err={}", obs.data[N_RAYS + 1]);
        assert!(obs.data[N_RAYS + 2].abs() < 0.1, "lateral={}", obs.data[N_RAYS + 2]);
        // 正前方射线应比正侧方远
        assert!(obs.data[N_RAYS / 2] > obs.data[0]);
        // 所有值有限且大致有界
        for v in obs.data {
            assert!(v.is_finite());
            assert!(v.abs() < 2.5);
        }
    }
}
