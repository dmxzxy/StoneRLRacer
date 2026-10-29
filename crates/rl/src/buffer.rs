//! DQN 环形 replay buffer。

use crate::Transition;
use sim::rng::Rng;

pub struct ReplayBuffer {
    capacity: usize,
    data: Vec<Transition>,
    cursor: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Vec::with_capacity(capacity),
            cursor: 0,
        }
    }

    pub fn push(&mut self, t: Transition) {
        if self.data.len() < self.capacity {
            self.data.push(t);
        } else {
            self.data[self.cursor] = t;
            self.cursor = (self.cursor + 1) % self.capacity;
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 有放回均匀采样 `n` 条。
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Vec<Transition> {
        if self.is_empty() {
            return Vec::new();
        }
        (0..n)
            .map(|_| self.data[(rng.next_u64() % self.data.len() as u64) as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::action::DiscreteAction;
    use sim::observation::Observation;

    fn dummy_transition(i: u8) -> Transition {
        Transition {
            obs: Observation { data: [i as f32; sim::observation::OBS_DIM] },
            action: DiscreteAction(i % 9),
            reward: i as f32,
            next_obs: Observation { data: [i as f32 + 0.5; sim::observation::OBS_DIM] },
            done: false,
        }
    }

    #[test]
    fn ring_overwrite() {
        let mut buf = ReplayBuffer::new(8);
        for i in 0..20u8 {
            buf.push(dummy_transition(i));
        }
        assert_eq!(buf.len(), 8);
        // 20 条入 8 容量：data = [16,17,18,19,12,13,14,15]
        assert_eq!(buf.data[0].reward, 16.0);
        assert_eq!(buf.data[4].reward, 12.0);
        // 随机采样必须来自合法范围
        let mut rng = Rng::new(1);
        for t in buf.sample(100, &mut rng) {
            assert!((12.0..20.0).contains(&t.reward));
        }
    }
}
