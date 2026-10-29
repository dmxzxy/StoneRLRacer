# UI 评审修单 #1（Opus 评审实拍截图后给出）

> 输入：`docs/shots/train-idle.png`、`docs/shots/demo-playing.png` + `docs/ui-spec.md` + 现有代码。
> 实现方按 P0 → P1 →（有余力）P2 顺序做。

## 先修根因

截图里整个应用跑在**浅色主题**下。`lib.rs` 的 `configure_style` 只调用了
`ctx.set_style_of(egui::Theme::Dark, style)`，但当前主题（ThemePreference）仍是跟随系统/浅色，
所以控件用的是 Light style：标题文字深色（在深底上显灰）、`TextEdit` 白底、页签浅蓝、
图表白底。所有"看起来颜色不对"的现象都出自这里。

**改法**：在 `configure_style` 末尾加 `ctx.set_theme(egui::Theme::Dark);`
（egui 0.37 `Context::set_theme`，见 context.rs:2594）。

## P0 功能性缺陷

1. **`lib.rs` `configure_style`**：末尾加 `ctx.set_theme(egui::Theme::Dark);`。
2. **`controls.rs` `transport`**：`with_layout(right_to_left(Center))` 会占满剩余高度
   （egui ui.rs:2591 文档），「回到起点」被放到剩余高度竖直中间，展示环境卡被挤出可视区。
   - 改：先 `allocate_exact_size(vec2(W, 28))` 拿到整行；左侧状态文字 11px 弱色，宽度上限 `W-96`，
     用 `Painter::layout` / `galley` 截断；已载入模型时用 `action_button_at` 把「回到起点」放在
     `Rect::from_min_size(pos2(row.right()-88, row.top()), vec2(88,28))`。
   - 没载入时这一行高 16。播放键前后间距 8。
3. **`header_row`（面板标题）**：第 1 条修完后颜色即为 `#DCDFE6`。不能加载粗体字体，
   所以把同一 galley 在 x+0.5 再画一次做伪粗体。
4. **`status_pill` 语义**：签名改成 `(text, color, icon)`，由 `header_row` 按模式选择：
   - 展示厅：有策略且在跑 → 实心点 +「播放中」`#48B270`；有策略且暂停 → 双竖条 +「已暂停」`#78AADC`；
     没有策略 → 空心点 +「未载入」`#8C919B`。
   - 训练工作台保持现状。
5. **`app.rs` `Panel::right`**：加 `.frame(Frame::side_top_panel(ui.style()).inner_margin(0))`
   与 `.size_range(300.0..=520.0)`。现在面板自带 (8,2) 内边距，内容宽只有 372/260，
   规范要求 388/276。
6. **页签栏统一**（`app.rs` 两个模式的中央区头部）：抽共用函数 `center_header`。
   - 行高 36，竖直居中。
   - 训练页签：`allocate_ui(vec2(172,28))` 包住 `controls::segmented`，每格 84。
   - 展示厅：左侧 3×12 `#78AADC` 竖条 + 13px 主色「展示厅」。
   - 状态：6px 圆点 + 12px 文字。训练中 `#48B270`，已暂停 `#DCB95F`，未训练 `#8C919B`。
   - 统计数字：LayoutJob，数值 monospace 12 主色，单位 11px 弱色。
   - `separator()` 换成 1px `#2A2D37` 细线，下方留 8。
   - 倍速：`slider_width = 96`，前面加 11px 弱色「倍速」。

## P1 观感差距

1. **`env_body` 垂直节奏**：现在 8/4/8/4/6 不均匀。统一：赛道→seed 8；seed→难度 12；
   难度→车型 12；标签→控件 4；车型→调校 8；调校→「当前」8。
2. **`model_row`**：没有统计时补第二行「隐藏层 {hidden} · 无统计」，11px 弱色，y+25。
   管理行静止时加 1px `#2A2D37` 描边，否则两个按钮像浮在空中。
3. **分段控件**：选中描边改 1.5px（Inside），填充仍 alpha 46；车型色条改高 3、宽 `cw-20`、
   底部上移 6；未选 alpha 90 → 160（现在「肌肉」的蓝色几乎看不见）。
4. **主按钮**：琥珀底黑字合格；图标与文字间距 6 → 8，三角改 10×12。
5. **大播放键暂停态**：加 hover 反馈——hover 填充 `#323745`、描边 2px `#58C682`；
   按下填充 `#262A33`。
6. **seed 输入框**：`.min_size(vec2(edit_w, 28))`，与「随机」按钮等高。
7. **`draw_hud`**：速度底板顶边与徽标对齐（`num_pos.y = top+20`），右边距 12（`right = rect.right()-22`）；
   状态行数值用 monospace 13 主色、标签 12px `HUD_DIM`（`Scene.status_line` 改成 `Vec<(&str, String)>`）。
8. **`draw_policy_panel`**：展示厅不该出现琥珀。最优格改 `#78AADC`、字色 `#0B1620`；
   数字 monospace 11；脚注 9.5 → 11；面板高 152 → 156。PPO 均值条同样改色。

## P2 精修

- 滚动条：`scroll.bar_width = 6`，`dormant_handle_opacity = 0.5`。
- 禁用按钮：填充 `#22252C`，1px `#2A2D37` 描边，文字 `#6B707B`（现在 `#5A5F6A` 对比度仅 2.4:1）。
- 画布：`rect_filled(rect, 8, GRASS)` + 1px `#343845` Inside 描边。
- `mirror_status` 与底部曲线标签的数字改 monospace 11。

## 约束

- 只用 egui 0.37 现有 API；不加依赖、不用图片、不加载字体；图标只用 `Painter` 几何，不用 emoji 字形。
- 不改功能语义；所有交互元素保留；点击区 ≥28px；300px 面板宽不溢出。
- 第 1 条会锁定深色、不再跟随系统主题。

## 验收标准

整窗 `#17191F` 深底，文字与控件里没有白色。展示厅自上而下依次是：
选模型卡 → 大播放键 → 状态行（右侧「回到起点」）→ 展示环境卡。
HUD 的徽标、Q 值面板、速度底板都离画布边缘 12px。所有数字等宽，单位弱色。
