//! 动作空间：离散 9 档（DQN）与连续（PPO）的统一表示。

use crate::rng::Rng;

/// 离散动作数：3 转向 × 3 油门。
pub const N_ACTIONS: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousAction {
    /// [-1, 1]，负为左转。
    pub steer: f32,
    /// [-1, 1]，负为刹车/倒车。
    pub throttle: f32,
}

impl ContinuousAction {
    pub const ZERO: Self = Self { steer: 0.0, throttle: 0.0 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteAction(pub u8);

impl DiscreteAction {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// 离散档位 → 连续动作。索引 = 转向档(0..3)*3 + 油门档(0..3)，
/// 档位值 {0,1,2} 映射 {-1, 0, +1}。
pub fn discrete_to_continuous(a: DiscreteAction) -> ContinuousAction {
    debug_assert!(a.0 < N_ACTIONS as u8);
    let steer = match a.0 / 3 {
        0 => -1.0,
        1 => 0.0,
        _ => 1.0,
    };
    let throttle = match a.0 % 3 {
        0 => -1.0,
        1 => 0.0,
        _ => 1.0,
    };
    ContinuousAction { steer, throttle }
}

/// 均匀随机离散动作（探索用）。
pub fn random_discrete(rng: &mut Rng) -> DiscreteAction {
    DiscreteAction((rng.next_u64() % N_ACTIONS as u64) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_roundtrip() {
        assert_eq!(discrete_to_continuous(DiscreteAction(0)), ContinuousAction { steer: -1.0, throttle: -1.0 });
        assert_eq!(discrete_to_continuous(DiscreteAction(4)), ContinuousAction::ZERO);
        assert_eq!(discrete_to_continuous(DiscreteAction(8)), ContinuousAction { steer: 1.0, throttle: 1.0 });
        assert_eq!(discrete_to_continuous(DiscreteAction(2)), ContinuousAction { steer: -1.0, throttle: 1.0 });
        assert_eq!(discrete_to_continuous(DiscreteAction(6)), ContinuousAction { steer: 1.0, throttle: -1.0 });
    }

    #[test]
    fn random_in_range() {
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            assert!(random_discrete(&mut rng).0 < N_ACTIONS as u8);
        }
    }
}
