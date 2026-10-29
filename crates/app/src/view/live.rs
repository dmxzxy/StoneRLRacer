//! 实时驾驶遥测：赛道画布下方一组滚动曲线。
//!
//! 曲线不是训练指标，而是"策略此刻在做什么"——速度、横向偏移、转向、油门。
//! 配合画布上的雷达射线，能直接看懂观测 → 动作 → 结果这一条链。

use egui::{Color32, RichText};
use egui_plot::{HLine, Line, Plot, PlotPoints};

use crate::viewer::Viewer;

const C_TEXT: Color32 = Color32::from_rgb(200, 204, 214);
const C_DIM: Color32 = Color32::from_rgb(130, 136, 148);
const C_SPEED: Color32 = Color32::from_rgb(120, 190, 150);
const C_LATERAL: Color32 = Color32::from_rgb(96, 140, 200);
const C_STEER: Color32 = Color32::from_rgb(226, 175, 74);
const C_THROTTLE: Color32 = Color32::from_rgb(206, 120, 96);

/// 一格的图。
struct Panel {
    title: &'static str,
    unit: &'static str,
    color: Color32,
    /// 固定的 y 范围（保持尺度稳定，方便肉眼比较）
    y_range: (f64, f64),
    zero_line: bool,
}

const PANELS: [Panel; 4] = [
    Panel {
        title: "车速",
        unit: "km/h",
        color: C_SPEED,
        y_range: (-10.0, 200.0),
        zero_line: false,
    },
    Panel {
        title: "横向偏移",
        unit: "±1 = 贴边",
        color: C_LATERAL,
        y_range: (-1.1, 1.1),
        zero_line: true,
    },
    Panel {
        title: "转向",
        unit: "−左 +右",
        color: C_STEER,
        y_range: (-1.1, 1.1),
        zero_line: true,
    },
    Panel {
        title: "油门",
        unit: "−刹 +油",
        color: C_THROTTLE,
        y_range: (-1.1, 1.1),
        zero_line: true,
    },
];

fn cell(ui: &mut egui::Ui, viewer: &Viewer, idx: usize) {
    let p = &PANELS[idx];
    let n = viewer.live.len();
    if n < 2 {
        return;
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new(p.title).small().strong().color(C_TEXT));
        ui.label(RichText::new(p.unit).small().color(C_DIM));
    });
    let pts: Vec<[f64; 2]> = viewer
        .live
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let y = match idx {
                0 => s.speed as f64,
                1 => s.lateral as f64,
                2 => s.steer as f64,
                _ => s.throttle as f64,
            };
            [i as f64, y]
        })
        .collect();
    Plot::new(format!("live_{idx}"))
        .height(78.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show_grid(false)
        .show_axes([false, true])
        .y_axis_min_width(30.0)
        .include_y(p.y_range.0)
        .include_y(p.y_range.1)
        .show(ui, |pu| {
            pu.line(
                Line::new(p.title.to_owned(), PlotPoints::from(pts))
                    .color(p.color)
                    .width(1.4),
            );
            if p.zero_line {
                pu.hline(
                    HLine::new("0", 0.0)
                        .color(Color32::from_rgb(66, 70, 80))
                        .width(1.0),
                );
            }
        });
}

/// 画整条遥测带（宽度自适应：宽屏一行四格，窄屏两行）。
pub fn draw(ui: &mut egui::Ui, viewer: &Viewer) {
    if viewer.live.len() < 8 {
        ui.label(
            RichText::new("遥测：策略开始跑动后，这里会滚动显示速度 / 偏移 / 转向 / 油门")
                .small()
                .color(C_DIM),
        );
        return;
    }
    let width = ui.available_width();
    ui.horizontal(|ui| {
        ui.label(RichText::new("实时遥测").small().strong().color(C_TEXT));
        // 数字走 monospace 11（和面板其它数值一致）
        ui.label(crate::app::dim_job(
            &format!("最近 {} 步", viewer.live.len()),
            C_DIM,
        ));
    });
    if width >= 560.0 {
        ui.columns(4, |cols| {
            for (i, col) in cols.iter_mut().enumerate() {
                cell(col, viewer, i);
            }
        });
    } else {
        ui.columns(2, |cols| {
            cell(&mut cols[0], viewer, 0);
            cell(&mut cols[1], viewer, 1);
        });
        ui.columns(2, |cols| {
            cell(&mut cols[0], viewer, 2);
            cell(&mut cols[1], viewer, 3);
        });
    }
}
