//! 训练线程：独占真实训练环境，狂跑 sim + agent；
//! 定期向 UI 推指标、策略快照、以及训练环境的实时镜像帧。
//!
//! 环境（赛道 / 车型 / seed）只在训练线程里被创建和替换——UI 想改环境，
//! 必须发 [`ToTrainer::SetEnv`]，由这里在安全点重建，保证"训练中的环境"
//! 永远只有一个真相来源。

use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rl::dqn::{DqnAgent, DqnConfig};
use rl::metrics::EpisodeSummary;
use rl::ppo::{PpoAgent, PpoConfig};
use rl::{Agent, Feedback};
use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::track::Track;

use crate::messages::{EpisodeTrace, ToTrainer, ToUi, TracePoint, TrainFrame, TrainRequest};

/// 每 2000 环境步发一次训练指标。
const METRICS_EVERY: u64 = 2000;
/// 训练镜像帧间隔（秒）。训练一两千 SPS，20–30 Hz 足够看清。
const FRAME_EVERY: Duration = Duration::from_millis(33);
/// 轨迹降采样：每 N 步记一个点。
const TRACE_STRIDE: u64 = 8;
/// 单局轨迹点上限（超出则隔点丢弃，保持全程覆盖）。
const TRACE_MAX: usize = 400;

pub fn spawn(
    request: TrainRequest,
    rx: Receiver<ToTrainer>,
    tx: SyncSender<ToUi>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("trainer".into())
        .spawn(move || run(request, rx, tx))
        .expect("spawn trainer")
}

fn run(request: TrainRequest, rx: Receiver<ToTrainer>, tx: SyncSender<ToUi>) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        train_loop(&request, &rx, &tx)
    }));
    match result {
        Ok(Some(err)) => {
            let _ = tx.send(ToUi::Stopped(Some(err)));
        }
        Ok(None) => {
            let _ = tx.send(ToUi::Stopped(None));
        }
        Err(_) => {
            let _ = tx.send(ToUi::Stopped(Some("训练线程 panic".into())));
        }
    }
}

/// 返回 Some(err) 表示异常退出。
fn train_loop(
    request: &TrainRequest,
    rx: &Receiver<ToTrainer>,
    tx: &SyncSender<ToUi>,
) -> Option<String> {
    let mut ctx = LoopCtx::default();
    let result = train_loop_inner(request, rx, tx, &mut ctx);
    // 退出前保存 checkpoint
    if let Some(agent) = ctx.agent.as_ref() {
        let snap = agent.snapshot();
        let path = checkpoint_path(&snap.algo, snap.hidden, ctx.track_seed, ctx.steps);
        if let Some(dir) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match snap.record.save(&path) {
            Ok(()) => {
                // 统计旁文件：模型列表里显示"完圈率 / 最佳圈时"，不用重新跑一遍
                write_stats_sidecar(
                    &path,
                    ctx.episodes,
                    ctx.laps,
                    ctx.best_lap,
                    ctx.avg_reward(),
                );
                let _ = tx.send(ToUi::CheckpointSaved(path));
                // 续训血统：保存了新文件后删除来源旧文件
                if let Some(resume) = &request.resume {
                    let _ = std::fs::remove_file(format!("checkpoints/{}", resume.replace_file));
                    let _ = std::fs::remove_file(stats_path(&format!(
                        "checkpoints/{}",
                        resume.replace_file
                    )));
                }
            }
            Err(e) => eprintln!("checkpoint 保存失败：{e}"),
        }
    }
    result
}

fn stats_path(rec_path: &str) -> String {
    rec_path.strip_suffix(".rec").unwrap_or(rec_path).to_owned() + ".stats"
}

/// 存档旁的纯文本统计（`key=value`，零依赖解析）。
fn write_stats_sidecar(
    path: &str,
    episodes: u64,
    laps: u64,
    best_lap: Option<f32>,
    avg_reward: f32,
) {
    let body = format!(
        "episodes={episodes}\nlaps={laps}\nbest_lap={}\navg_reward={avg_reward:.3}\n",
        best_lap
            .map(|t| format!("{t:.3}"))
            .unwrap_or_else(|| "0".into()),
    );
    let _ = std::fs::write(stats_path(path), body);
}

struct LoopCtx {
    agent: Option<Box<dyn Agent>>,
    steps: u64,
    /// 当前训练环境的赛道 seed（存档命名用）。
    track_seed: u64,
    /// 本次训练（含续训）内完成的局数。
    episodes: u64,
    /// 完圈局数。
    laps: u64,
    best_lap: Option<f32>,
    /// 最近 50 局奖励（旁文件里记平均值）。
    recent_rewards: Vec<f32>,
}

impl LoopCtx {
    fn avg_reward(&self) -> f32 {
        if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32
        }
    }
}

impl Default for LoopCtx {
    fn default() -> Self {
        Self {
            agent: None,
            steps: 0,
            track_seed: 0,
            episodes: 0,
            laps: 0,
            best_lap: None,
            recent_rewards: Vec::new(),
        }
    }
}

fn checkpoint_path(algo: &rl::policy::AlgoKind, hidden: usize, seed: u64, steps: u64) -> String {
    let name = match algo {
        rl::policy::AlgoKind::Dqn => "dqn",
        rl::policy::AlgoKind::Ppo => "ppo",
    };
    format!("checkpoints/{name}-h{hidden}-seed{seed}-s{steps}.rec")
}

/// 建训练环境（唯一入口，配置一律先消毒）。
fn build_env(seed: u64, cfg: &SimConfig) -> (SimState, SimConfig) {
    let mut cfg = cfg.clone().sanitized();
    cfg.start_noise = true;
    let track = Track::generate(seed, &cfg.track);
    (SimState::new(track, seed, cfg.clone()), cfg)
}

fn train_loop_inner(
    request: &TrainRequest,
    rx: &Receiver<ToTrainer>,
    tx: &SyncSender<ToUi>,
    ctx: &mut LoopCtx,
) -> Option<String> {
    let (mut env, _cfg) = build_env(request.track_seed, &request.sim_cfg);
    ctx.track_seed = request.track_seed;

    if ctx.agent.is_none() {
        ctx.agent = Some(match request.algo {
            rl::policy::AlgoKind::Dqn => {
                Box::new(DqnAgent::new(DqnConfig::default(), request.track_seed))
            }
            rl::policy::AlgoKind::Ppo => {
                Box::new(PpoAgent::new(PpoConfig::default(), request.track_seed))
            }
        });
    }
    let agent = ctx.agent.as_mut().unwrap();

    // 继续训练：恢复已保存的模型
    if let Some(resume) = request.resume.as_ref() {
        agent.restore(resume.record.clone(), resume.steps);
    }

    let mut episode = 0u64;
    let mut ep_reward = 0.0f32;
    let mut speed_sum = 0.0f32;
    let mut max_speed = 0.0f32;
    let mut ep_parts = sim::reward::RewardParts::ZERO;
    let mut trace: Vec<TracePoint> = Vec::new();
    let mut next_metrics = ctx.steps + METRICS_EVERY;
    let mut last_frame = Instant::now();
    let mut paused = false;
    let mut last_obs_rays: [f32; sim::observation::N_RAYS];

    loop {
        // 每 64 步查一次命令，避免每步 try_recv 的开销；暂停时也必须抢命令，
        // 否则步数不推进，暂停后就再也不会停下/换环境。
        if ctx.steps & 63 == 0 || paused {
            let mut switch: Option<(u64, SimConfig)> = None;
            loop {
                match rx.try_recv() {
                    Ok(ToTrainer::Stop) => return None,
                    Ok(ToTrainer::Pause(p)) => paused = p,
                    Ok(ToTrainer::SetEnv { seed, sim_cfg }) => switch = Some((seed, sim_cfg)),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                }
            }
            if let Some((seed, sim_cfg)) = switch {
                // 安全点：丢掉当前局，原地重建环境；策略权重保留，经验池清空
                let (new_env, _) = build_env(seed, &sim_cfg);
                env = new_env;
                ctx.track_seed = seed;
                env.reset();
                ep_reward = 0.0;
                speed_sum = 0.0;
                max_speed = 0.0;
                ep_parts = sim::reward::RewardParts::ZERO;
                trace.clear();
                agent.on_env_changed();
                let _ = tx.try_send(ToUi::EnvApplied { seed, sim_cfg });
            }
        }
        if paused {
            std::thread::sleep(Duration::from_millis(10));
            continue;
        }

        let obs = env.observation();
        last_obs_rays = std::array::from_fn(|i| obs.data[i]);
        let action = agent.act(&obs);
        let out = env.step(action);
        ep_reward += out.reward;
        ep_parts.add(&out.parts);
        speed_sum += env.car.speed;
        max_speed = max_speed.max(env.car.speed);
        ctx.steps += 1;
        if ctx.steps % TRACE_STRIDE == 0 {
            if trace.len() >= TRACE_MAX {
                // 撑满后隔点丢弃（保留偶数位），保持全程覆盖而不是只剩开头
                let keep: Vec<TracePoint> = trace.iter().step_by(2).copied().collect();
                trace = keep;
            }
            trace.push(TracePoint {
                x: env.car.pos.x,
                y: env.car.pos.y,
                speed: env.car.speed,
            });
        }
        agent.feedback(Feedback {
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
                max_speed,
                parts: ep_parts,
            };
            ctx.episodes += 1;
            if summary.has_lap() {
                ctx.laps += 1;
                ctx.best_lap = Some(match ctx.best_lap {
                    Some(b) if b <= summary.lap_time => b,
                    _ => summary.lap_time,
                });
            }
            ctx.recent_rewards.push(summary.reward);
            if ctx.recent_rewards.len() > 50 {
                ctx.recent_rewards.remove(0);
            }
            let _ = tx.try_send(ToUi::EpisodeTrace(EpisodeTrace {
                episode,
                points: std::mem::take(&mut trace),
                lap: summary.has_lap(),
            }));
            let _ = tx.try_send(ToUi::EpisodeDone(summary));
            agent.on_episode_end();
            ep_reward = 0.0;
            speed_sum = 0.0;
            max_speed = 0.0;
            ep_parts = sim::reward::RewardParts::ZERO;
            env.reset();
        }

        if ctx.steps >= next_metrics {
            let mut m = agent.metrics();
            m.step = ctx.steps;
            let _ = tx.try_send(ToUi::StepMetrics(m));
            next_metrics += METRICS_EVERY;
        }
        if last_frame.elapsed() >= FRAME_EVERY {
            let _ = tx.try_send(ToUi::TrainFrame(frame_of(
                &env,
                ctx.steps,
                episode,
                ep_reward,
                last_obs_rays,
            )));
            last_frame = Instant::now();
        }
    }
}

/// 从训练环境采样一帧镜像（只读，不改变仿真）。
fn frame_of(
    env: &SimState,
    step: u64,
    episode: u64,
    ep_reward: f32,
    rays: [f32; sim::observation::N_RAYS],
) -> TrainFrame {
    TrainFrame {
        step,
        episode,
        ep_step: env.step_count,
        ep_reward,
        x: env.car.pos.x,
        y: env.car.pos.y,
        heading: env.car.heading,
        speed: env.car.speed,
        progress: env.progress_frac(),
        result: env.result,
        rays,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::ToTrainer;
    use rl::policy::{AlgoKind, Policy};
    use std::sync::mpsc;
    use std::time::Duration;

    /// 训练线程协议端到端：消息可达 + 快照可加载出可用策略 + 训练中换环境不中断。
    /// 用 release 跑（`cargo test --release -p rl-racer-app`）。
    #[test]
    fn trainer_protocol_end_to_end() {
        for algo in [AlgoKind::Dqn, AlgoKind::Ppo] {
            let (tx_cmd, rx_cmd) = mpsc::channel::<ToTrainer>();
            let (tx_ui, rx_ui) = mpsc::sync_channel::<ToUi>(1024);
            let request = TrainRequest {
                algo,
                track_seed: 42,
                sim_cfg: sim::config::SimConfig::default(),
                resume: None,
            };
            let handle = spawn(request, rx_cmd, tx_ui);

            let deadline = Instant::now() + Duration::from_secs(120);
            let mut got_metrics = false;
            let mut got_episode = false;
            let mut got_trace = false;
            let mut got_frame = false;
            while Instant::now() < deadline {
                match rx_ui.recv_timeout(Duration::from_secs(5)) {
                    Ok(ToUi::StepMetrics(m)) => {
                        assert!(m.step > 0);
                        got_metrics = true;
                    }
                    Ok(ToUi::EpisodeDone(e)) => {
                        assert!(e.episode > 0);
                        // 奖励分解必须自洽：各分量之和等于该局总奖励
                        assert!((e.parts.total() - e.reward).abs() < 0.5, "分解不一致");
                        got_episode = true;
                    }
                    Ok(ToUi::EpisodeTrace(t)) => {
                        assert!(t.points.len() >= 2, "轨迹至少两个点");
                        got_trace = true;
                    }
                    Ok(ToUi::TrainFrame(f)) => {
                        assert!(f.x.is_finite() && f.y.is_finite() && f.heading.is_finite());
                        got_frame = true;
                    }
                    Ok(ToUi::EnvApplied { .. }) => {}
                    Ok(ToUi::CheckpointSaved(path)) => {
                        assert!(
                            std::path::Path::new(&path).exists(),
                            "存档文件应存在：{path}"
                        );
                    }
                    Ok(ToUi::Stopped(_)) => panic!("训练线程不应自行退出"),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if got_metrics && got_episode && got_trace && got_frame {
                    break;
                }
            }
            assert!(got_metrics, "{algo:?} 应产生 StepMetrics");
            assert!(got_episode, "{algo:?} 应产生 EpisodeDone");
            assert!(got_trace, "{algo:?} 应产生 EpisodeTrace");
            assert!(got_frame, "{algo:?} 应产生 TrainFrame");

            // 训练中换环境：换赛道 + 换车型，训练必须继续往下走
            let mut sport = sim::config::SimConfig::default();
            sport.car = sim::car::CarConfig::sport();
            tx_cmd
                .send(ToTrainer::SetEnv {
                    seed: 7,
                    sim_cfg: sport.clone(),
                })
                .unwrap();

            let mut env_applied = false;
            let mut steps_after = 0u64;
            let deadline = Instant::now() + Duration::from_secs(30);
            while Instant::now() < deadline {
                match rx_ui.recv_timeout(Duration::from_secs(3)) {
                    Ok(ToUi::EnvApplied { seed, sim_cfg }) => {
                        assert_eq!(seed, 7);
                        assert!((sim_cfg.car.max_speed - sport.car.max_speed).abs() < 1e-3);
                        env_applied = true;
                    }
                    Ok(ToUi::TrainFrame(f)) if env_applied => {
                        steps_after = f.step;
                        assert!(f.all_finite(), "换环境后镜像帧仍应有限");
                        if steps_after > 0 {
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            assert!(env_applied, "{algo:?} 换环境应回 EnvApplied");
            assert!(steps_after > 0, "{algo:?} 换环境后训练应继续推进");

            // 停训 → 存档 → 存档能被当作策略载入并给出合法动作
            let _ = tx_cmd.send(ToTrainer::Stop);
            let _ = handle.join();
            let saved = std::fs::read_dir("checkpoints")
                .map(|es| {
                    es.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "rec").unwrap_or(false))
                        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
                })
                .ok()
                .flatten()
                .expect("停训应保存存档");
            let record = burn::store::ModuleRecord::load(saved.to_string_lossy().to_string())
                .expect("存档应可读");
            let policy = Policy::load(rl::policy::PolicySnapshot {
                algo,
                hidden: 128,
                record,
            });
            let cfg = sim::config::SimConfig::default();
            let track = Track::generate(42, &cfg.track);
            let env = SimState::new(track, 42, cfg);
            let a = policy.act(&env.observation());
            assert!((-1.0..=1.0).contains(&a.steer));
            assert!((-1.0..=1.0).contains(&a.throttle));
        }
    }
}
