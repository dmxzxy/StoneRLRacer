//! M2 无头性能测试：随机策略狂跑，验证 SPS 与奖励统计。
//!
//! 用法：`cargo run --release -p sim --example headless -- [steps] [seed]`

use sim::action::{discrete_to_continuous, random_discrete};
use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::rng::Rng;
use sim::track::Track;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total_steps: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut cfg = SimConfig::default();
    cfg.start_noise = true;
    let track = Track::generate(seed, &cfg.track);
    println!(
        "赛道 seed={seed}  长 {:.0}m  宽 {:.0}m  最小转弯半径 {:.0}m",
        track.length,
        track.width,
        track.min_corner_radius()
    );
    let mut env = SimState::new(track, seed, cfg);
    let mut rng = Rng::new(7);

    let t0 = std::time::Instant::now();
    let mut ep_reward = 0.0;
    let mut episodes = 0u64;
    let mut sum_ep_reward = 0.0;
    let mut sum_ep_len = 0u64;
    let mut laps = 0u64;
    let mut next_report = total_steps / 20;

    for step in 1..=total_steps {
        let a = discrete_to_continuous(random_discrete(&mut rng));
        let out = env.step(a);
        ep_reward += out.reward;
        if out.done {
            episodes += 1;
            sum_ep_reward += ep_reward;
            sum_ep_len += env.step_count as u64;
            if matches!(out.result, EpisodeResult::LapDone) {
                laps += 1;
            }
            ep_reward = 0.0;
            env.reset();
        }
        if step >= next_report {
            let dt = t0.elapsed().as_secs_f32();
            println!(
                "step {:>9}  sps {:>7.0}  eps {:>6}  avg_reward {:>8.2}  avg_len {:>6.0}  laps {laps}",
                step,
                step as f32 / dt,
                episodes,
                sum_ep_reward / episodes.max(1) as f32,
                sum_ep_len as f32 / episodes.max(1) as f32,
            );
            next_report += total_steps / 20;
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "完成：{total_steps} 步 / {dt:.1}s = {:.0} SPS，{episodes} 局，完圈 {laps} 次",
        total_steps as f32 / dt
    );
}
