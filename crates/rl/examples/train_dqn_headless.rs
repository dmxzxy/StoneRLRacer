//! M3 验证：固定椭圆（近似正圆）赛道上无头训练 DQN，打印指标。
//!
//! 用法：`cargo run --release -p rl --example train_dqn_headless -- [steps] [seed]`

use rl::dqn::{DqnAgent, DqnConfig};
use rl::metrics::EpisodeSummary;
use rl::Agent;
use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::track::{Track, TrackGenConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total_steps: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut cfg = SimConfig {
        track: TrackGenConfig::easy(),
        start_noise: true,
        ..Default::default()
    };
    let track = Track::generate(seed, &cfg.track);
    println!(
        "赛道 seed={seed}  长 {:.0}m  宽 {:.0}m",
        track.length,
        track.width
    );
    let mut env = SimState::new(track, seed, cfg.clone());

    let dqn_cfg = DqnConfig::default();
    println!("DQN 配置：{dqn_cfg:?}");
    let mut agent = DqnAgent::new(dqn_cfg, seed);

    let t0 = std::time::Instant::now();
    let mut episode = 0u64;
    let mut ep_reward = 0.0;
    let mut speed_sum = 0.0f32;
    let mut ep_parts = sim::reward::RewardParts::ZERO;
    let mut max_speed = 0.0f32;
    let mut next_report = 10_000u64;
    let mut report_rewards: Vec<f32> = Vec::new();
    let mut recent_lap_times: Vec<f32> = Vec::new();

    for step in 1..=total_steps {
        let obs = env.observation();
        let action = agent.act(&obs);
        let out = env.step(action);
        ep_reward += out.reward;
        speed_sum += env.car.speed;
        ep_parts.add(&out.parts);
        max_speed = max_speed.max(env.car.speed);
        agent.feedback(rl::Feedback {
            next_obs: &out.obs,
            reward: out.reward,
            done: out.done,
            terminal: out.done && !matches!(out.result, EpisodeResult::Timeout),
        });
        if out.done {
            episode += 1;
            let summary = EpisodeSummary {
                episode,
                steps: env.step_count,
                reward: ep_reward,
                avg_speed: speed_sum / env.step_count.max(1) as f32,
                result: out.result,
                lap_time: if matches!(out.result, EpisodeResult::LapDone) {
                    env.elapsed
                } else {
                    0.0
                },
                parts: ep_parts,
                max_speed: max_speed,
            };
            if summary.result == EpisodeResult::LapDone {
                recent_lap_times.push(summary.lap_time);
                if recent_lap_times.len() > 50 {
                    recent_lap_times.remove(0);
                }
            }
            report_rewards.push(summary.reward);
            agent.on_episode_end();
            ep_reward = 0.0;
            speed_sum = 0.0;
            ep_parts = sim::reward::RewardParts::ZERO;
            max_speed = 0.0;
            env.reset();
        }

        if step >= next_report {
            let dt = t0.elapsed().as_secs_f32();
            let m = agent.metrics();
            let n = report_rewards.len().min(100);
            let avg: f32 = report_rewards[report_rewards.len() - n..].iter().sum::<f32>()
                / n.max(1) as f32;
            let lap_info = recent_lap_times
                .last()
                .map(|t| format!("  最近圈时 {t:.1}s"))
                .unwrap_or_default();
            println!(
                "step {step:>8}  sps {:>7.0}  eps {episode:>6}  avg100_reward {avg:>8.2}  loss {:.4}  Q均值 {:.2}  ε {:.3}  buf {}{lap_info}",
                step as f32 / dt,
                m.loss,
                m.q_mean,
                m.epsilon,
                m.buffer_len,
            );
            next_report += 10_000;
        }
    }
    println!(
        "完成：{episode} 局，{} 次完圈，最近50局完圈 {} 次",
        recent_lap_times.len(),
        recent_lap_times.len().min(50),
    );
    // 保存模型（与桌面训练站/checkpoints 格式一致）
    let snap = agent.snapshot();
    let _ = std::fs::create_dir_all("checkpoints");
    let path = format!("checkpoints/dqn-h{}-seed{seed}-s{total_steps}.rec", snap.hidden);
    match snap.record.save(&path) {
        Ok(()) => println!("模型已保存：{path}"),
        Err(e) => eprintln!("保存失败：{e}"),
    }
}
