//! rl-racer sim：纯计算赛车仿真（无渲染、无 burn、无 UI）。
//!
//! 设计目标：无头模式下每秒可跑数万步，训练线程与渲染线程共用同一套逻辑。

pub mod action;
pub mod car;
pub mod config;
pub mod episode;
pub mod math;
pub mod observation;
pub mod raycast;
pub mod reward;
pub mod rng;
pub mod track;

/// 仿真固定步长（秒）。训练与渲染共用，保证观赛与训练行为一致。
pub const DT: f32 = 0.05;
