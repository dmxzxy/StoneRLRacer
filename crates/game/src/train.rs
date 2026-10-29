//! 内置训练：在游戏内直接训练 DQN（后台线程），
//! 每秒推送策略快照热替换观赛策略——边训练边看 AI 进步。

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use rl::dqn::{DqnAgent, DqnConfig};

use crate::state::Game;
use rl::metrics::EpisodeSummary;
use rl::{Agent, Feedback};
use sim::config::SimConfig;
use sim::episode::{EpisodeResult, SimState};
use sim::track::Track;

use crate::model;

/// trainer → 游戏主循环。
pub enum TrainMsg {
    Stats {
        steps: u64,
        episodes: u64,
        avg_reward: f32,
        sps: f32,
    },
    Snapshot(rl::policy::PolicySnapshot),
    Done(Option<String>),
}

/// 游戏 → trainer。
pub enum TrainCmd {
    Stop,
}

pub struct TrainHandle {
    pub tx: Sender<TrainCmd>,
    /// Mutex 使 Receiver 满足 Resource 的 Sync 要求（仅 poll_training 访问）。
    pub rx: std::sync::Mutex<Receiver<TrainMsg>>,
    pub thread: JoinHandle<()>,
}

/// 启动训练线程（当前赛道 + 车辆配置）。
pub fn start(cfg: SimConfig, seed: u64) -> TrainHandle {
    let (tx_cmd, rx_cmd) = std::sync::mpsc::channel::<TrainCmd>();
    let (tx_msg, rx_msg) = std::sync::mpsc::channel::<TrainMsg>();
    let thread = std::thread::Builder::new()
        .name("trainer".into())
        .spawn(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    train_loop(&cfg, seed, &rx_cmd, &tx_msg)
                }));
            let err = match result {
                Ok(Some(e)) => Some(e),
                Ok(None) => None,
                Err(_) => Some("训练线程 panic".into()),
            };
            let _ = tx_msg.send(TrainMsg::Done(err));
        })
        .expect("spawn trainer");
    TrainHandle {
        tx: tx_cmd,
        rx: std::sync::Mutex::new(rx_msg),
        thread,
    }
}

fn train_loop(
    cfg: &SimConfig,
    seed: u64,
    rx: &Receiver<TrainCmd>,
    tx: &Sender<TrainMsg>,
) -> Option<String> {
    let mut cfg = cfg.clone();
    cfg.start_noise = true;
    let track = Track::generate(seed, &cfg.track);
    let mut env = SimState::new(track, seed, cfg);
    let mut agent = DqnAgent::new(DqnConfig::default(), seed);

    let mut episode = 0u64;
    let mut ep_reward = 0.0f32;
    let mut recent: Vec<f32> = Vec::new();
    let mut steps = 0u64;
    let t0 = Instant::now();
    let mut next_stats = 5_000u64;
    let mut last_snap = Instant::now();
    let mut done = false;

    loop {
        // 每 64 步查一次命令
        if steps & 63 == 0 {
            loop {
                match rx.try_recv() {
                    Ok(TrainCmd::Stop) => {
                        done = true;
                        break;
                    }
                    Err(_) => break,
                }
            }
            if done {
                break;
            }
        }

        let obs = env.observation();
        let action = agent.act(&obs);
        let out = env.step(action);
        ep_reward += out.reward;
        steps += 1;
        agent.feedback(Feedback {
            next_obs: &out.obs,
            reward: out.reward,
            done: out.done,
            terminal: out.done && !matches!(out.result, EpisodeResult::Timeout),
        });
        if out.done {
            episode += 1;
            recent.push(ep_reward);
            if recent.len() > 50 {
                recent.remove(0);
            }
            ep_reward = 0.0;
            env.reset();
        }

        if steps >= next_stats {
            let avg = recent.iter().sum::<f32>() / recent.len().max(1) as f32;
            let sps = steps as f32 / t0.elapsed().as_secs_f32().max(0.001);
            let _ = tx.send(TrainMsg::Stats {
                steps,
                episodes: episode,
                avg_reward: avg,
                sps,
            });
            next_stats += 5_000;
        }
        if last_snap.elapsed() >= Duration::from_secs(1) {
            let _ = tx.send(TrainMsg::Snapshot(agent.snapshot()));
            last_snap = Instant::now();
        }
    }

    // 停止：保存模型（与桌面训练站格式一致）
    let snap = agent.snapshot();
    let dir = model::model_dirs().into_iter().next().unwrap_or_default();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("dqn-h{}-seed{seed}-s{steps}.rec", snap.hidden));
    match snap.record.save(&path) {
        Ok(()) => None,
        Err(e) => Some(format!("保存失败：{e}")),
    }
}

/// 游戏侧的训练状态。
#[derive(Default)]
pub struct TrainState {
    pub handle: Option<TrainHandle>,
    pub steps: u64,
    pub episodes: u64,
    pub avg_reward: f32,
    pub sps: f32,
    /// 奖励曲线历史（每次统计一点，上限 100）。
    pub history: Vec<f32>,
}

impl TrainState {
    pub fn stop(&mut self) {
        if let Some(h) = &self.handle {
            let _ = h.tx.send(TrainCmd::Stop);
        }
    }
}

/// 每帧轮询训练线程：更新统计、热替换策略快照、结束时收尾。
pub fn poll_training(mut game: ResMut<Game>) {
    // 先把消息全部 drain 出来（避免 game 的可变借用与 handle 冲突）
    let mut msgs = Vec::new();
    {
        let Some(handle) = &game.training.handle else {
            return;
        };
        let Ok(rx) = handle.rx.lock() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(m) => msgs.push(m),
                Err(_) => break,
            }
        }
    }
    for m in msgs {
        match m {
            TrainMsg::Stats { steps, episodes, avg_reward, sps } => {
                game.training.steps = steps;
                game.training.episodes = episodes;
                game.training.avg_reward = avg_reward;
                game.training.sps = sps;
                game.training.history.push(avg_reward);
                if game.training.history.len() > 100 {
                    game.training.history.remove(0);
                }
            }
            TrainMsg::Snapshot(snap) => {
                // 热替换观赛策略——边训练边看进步
                game.viewer.policy = Some(rl::policy::Policy::load(snap));
                game.viewer.policy_label = "训练中的最新策略".into();
            }
            TrainMsg::Done(err) => {
                if let Some(e) = err {
                    eprintln!("训练结束：{e}");
                }
                // 收线程、重扫模型、载入最新（步数最多）的
                if let Some(h) = game.training.handle.take() {
                    let _ = h.thread.join();
                }
                game.models = crate::model::scan_models();
                if !game.models.is_empty() {
                    game.model_idx = 0;
                    game.load_selected_model();
                }
                return;
            }
        }
    }
}
