//! 模型文件扫描与加载（与桌面训练站共享 checkpoints/ 目录格式）。

use rl::policy::{AlgoKind, Policy, PolicySnapshot};

/// 模型元信息（从文件名解析）。
pub struct ModelMeta {
    pub name: String,
    pub algo: AlgoKind,
    pub hidden: usize,
    pub steps: u64,
}

impl ModelMeta {
    pub fn algo_label(&self) -> &'static str {
        self.algo.label()
    }
}

/// 模型目录：桌面用工作目录；安卓用应用外部文件目录（adb push 免权限）。
pub fn model_dirs() -> Vec<std::path::PathBuf> {
    #[cfg(target_os = "android")]
    {
        // 内部存储：应用自身始终可读写（新版 Android 限制外部 Android/data 的原始路径访问）。
        // 模型通过 adb shell run-as 推送（见 README）。
        vec![
            std::path::PathBuf::from("/data/data/com.rex.rlracer/files/checkpoints"),
            std::path::PathBuf::from(
                "/storage/emulated/0/Android/data/com.rex.rlracer/files/checkpoints",
            ),
            std::path::PathBuf::from("checkpoints"),
        ]
    }
    #[cfg(not(target_os = "android"))]
    {
        vec![
            std::path::PathBuf::from("checkpoints"),
            std::path::PathBuf::from("../checkpoints"),
            std::path::PathBuf::from("../../checkpoints"),
        ]
    }
}

/// 文件名格式：{algo}-h{hidden}-seed{seed}-s{steps}.rec
fn parse_meta(name: &str) -> Option<ModelMeta> {
    let stem = name.strip_suffix(".rec")?;
    let mut parts = stem.split('-');
    let algo = match parts.next()? {
        "dqn" => AlgoKind::Dqn,
        "ppo" => AlgoKind::Ppo,
        _ => return None,
    };
    let hidden: usize = parts.next()?.strip_prefix('h')?.parse().ok()?;
    let _seed: u64 = parts.next()?.strip_prefix("seed")?.parse().ok()?;
    let steps: u64 = parts.next()?.strip_prefix('s')?.parse().ok()?;
    Some(ModelMeta {
        name: name.to_owned(),
        algo,
        hidden,
        steps,
    })
}

/// 扫描所有模型，按步数降序（最新在前）。
pub fn scan_models() -> Vec<ModelMeta> {
    let mut models = Vec::new();
    for dir in model_dirs() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.filter_map(|e| e.ok()) {
                if e.path().extension().map(|x| x == "rec").unwrap_or(false) {
                    if let Some(name) = e.file_name().to_str() {
                        if let Some(m) = parse_meta(name) {
                            models.push(m);
                        }
                    }
                }
            }
        }
        if !models.is_empty() {
            break;
        }
    }
    models.sort_by_key(|m| std::cmp::Reverse(m.steps));
    bevy::log::info!("[model] 扫描到 {} 个模型", models.len());
    models
}

/// 读取模型文件并构建推理策略。
pub fn load_model(meta: &ModelMeta) -> Option<Policy> {
    for dir in model_dirs() {
        let path = dir.join(&meta.name);
        match burn::store::ModuleRecord::load(&path) {
            Ok(record) => {
                let snap = PolicySnapshot {
                    algo: meta.algo,
                    hidden: meta.hidden,
                    record,
                };
                return Some(Policy::load(snap));
            }
            Err(e) => {
                bevy::log::info!("[model] 加载失败 {}: {e}", path.display());
            }
        }
    }
    None
}

pub fn fmt_steps(s: u64) -> String {
    if s >= 1000 {
        format!("{:.0}k", s as f64 / 1000.0)
    } else {
        s.to_string()
    }
}
