//! RL Racer 桌面入口。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    rl_racer_app::run()
}
