//! 赛道生成诊断：各难度、各 seed 的约束通过情况。
//! `cargo run --release -p sim --example diag_tracks`

use sim::track::{Track, TrackGenConfig, TrackShape};

fn stats(name: &str, cfg: &TrackGenConfig, seeds: u64) {
    println!("== {name} ==");
    for seed in 0..seeds {
        let t = Track::generate(seed, cfg);
        println!(
            "seed {seed:>3}: 长 {:>5.0}m  最小弯径 {:>5.1}m  防自交 {}  形状 {:?}",
            t.length,
            t.min_corner_radius(),
            t.passes_proximity(),
            cfg.shape,
        );
    }
}

fn main() {
    stats("简单", &TrackGenConfig::easy(), 6);
    stats("中等", &TrackGenConfig::medium(), 6);
    stats("困难(随机游走)", &TrackGenConfig::hard(), 10);

    // 真实赛道近似：打印防自交违例位置
    for t in sim::track::RealTrack::ALL {
        let track = Track::generate(0, &t.to_track_gen_config());
        println!(
            "== {} ==  长 {:.0}m  最小弯径 {:.1}m",
            t.label(),
            track.length,
            track.min_corner_radius()
        );
        let m = track.samples.len();
        let ds = track.length / m as f32;
        let guard = (track.width * 1.8 / ds).ceil() as usize + 2;
        let min_d = track.width * 1.15;
        let mut violations = 0;
        for i in 0..m {
            for j in (i + guard)..m {
                if j - i + guard > m {
                    break;
                }
                let d = (track.samples[j].pos - track.samples[i].pos).len();
                if d < min_d {
                    if violations < 8 {
                        println!(
                            "  违例: #{i} ({:.0},{:.0}) ↔ #{j} ({:.0},{:.0}) 距离 {:.1}m",
                            track.samples[i].pos.x,
                            track.samples[i].pos.y,
                            track.samples[j].pos.x,
                            track.samples[j].pos.y,
                            d
                        );
                    }
                    violations += 1;
                }
            }
        }
        println!("  共 {violations} 处违例");
    }

    // 随机游走的原始候选统计（不经 generate 的重试挑选）
    let cfg = TrackGenConfig::hard();
    let mut ok = 0;
    let total = 200;
    for i in 0..total {
        let t = Track::generate_raw_for_diag(42 + i, &cfg);
        if t.passes_proximity() {
            ok += 1;
        }
    }
    println!(
        "\n随机游走原始候选：{ok}/{total} 通过防自交（{:}%）",
        ok * 100 / total
    );
}
