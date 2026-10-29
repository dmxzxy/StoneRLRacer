# RL Racer 右侧面板与外壳视觉规范

> 来源：Claude（设计）+ 交给实现方执行。数值都是拍定的，不要改。

## A. 设计基调

- 定位：深色赛车遥测台，信息成块（卡片），强调色只留一个。
- 原则 1：**模式色**。训练用琥珀 `#D69134`，展示用冷蓝 `#78AADC`。选中态、滑杆填充、面板顶部 2px 色条、状态 pill 全部跟随模式色。
- 原则 2：绿色 `#48B270` 只表示"正在跑 / 可播放"，红色只表示危险，其余一律中性灰。
- 原则 3：一屏只有一个主操作。训练是琥珀实心大按钮，展示是绿色大播放键。

## B. Token

- 背景：面板 `#17191F`（沿用 lib.rs configure_style）。卡片 `#1E2027`。控件 `#262932`。悬停 `#2F333E`。
- 描边：普通 `#343845`，强 `#454A59`，宽度均为 1px。
- 文字：主 `#DCDFE6`，次 `#B4B9C4`，弱 `#8C919B`。
- 状态色：警告 `#DCB95F`，危险 `#CE5C54`，危险文字 `#E57F78`。
- 绿色：`#48B270`，悬停 `#58C682`，按下 `#3E9A60`，绿底上的文字 `#08180E`。
- 琥珀：悬停 `#E2A04A`，琥珀底上的文字 `#1A1205`，选中文字 `#E8B565`。
- 选中态底色：模式色 alpha 46（行选中用 28）。
- 圆角：控件 6，卡片 8，大按钮 12，pill 为高度的一半。
- 间距刻度：4 / 8 / 12 / 16。卡片内边距 12，卡片之间 8，`item_spacing` 沿用 (8,6)。
- 字号阶梯：
  - 面板标题 16，strong，主色。
  - 区块标题 12，strong，次色。
  - 正文 13，主色。
  - 说明 11，弱色。
  - 数值 13，`FontId::monospace`（egui 内置，不算自定义字体），主色。
  - 大数值 18，strong，monospace。
- 行高：控件 28，主按钮 36，模型行 44，统计行 22，播放键 64。

## C. 面板结构

- 整体：`controls::draw` 外层套 `Frame` 内边距 12。内容宽 388px（窗口 <900px 时 276px）。
- 面板顶部：2px 模式色条。
- 顺序（训练）：
  1. 标题行（"RL Racer" 加右侧状态 pill）。
  2. 模式切换。
  3. 说明（11px 弱色，允许折 2 行）。
  4. 训练卡。
  5. 训练环境卡。
  6. 存档模型卡。
  7. 奖励口径（CollapsingHeader）。
- 顺序（展示）：标题行、模式切换、说明、选择模型卡、大播放键（独立，不在卡内）、展示环境卡。
- 分隔方式：删掉所有 `ui.separator()`，改为卡片之间留 8px。
- 区块标题：3×12 的模式色竖条，右接 12px strong 标题。标题右侧用 1px `#2A2D37` 细线填满剩余宽度。

## D. 控件

### 分段选择器（赛道、难度、车型、模式）
- 每格独立，间距 4，高 28，圆角 6。
- 未选：填充 `#262932`，描边 `#343845`，文字次色。
- 悬停：填充 `#2F333E`，描边 `#454A59`。
- 选中：填充模式色 alpha 46，描边模式色，文字 `#E8B565`（展示模式用 `#78AADC`）。
- 等分宽度：`cell_w = floor((W − (cols−1)·4) / cols)`，余数补给最后一格。
- 列数：`W/n ≥ 52` 时单行，否则 `cols = ceil(n/2)`（6 条赛道在 388px 单行，276px 下 3×2；5 款车型在 276px 仍单行）。
- 格宽 <56 时字号 12，否则 13。
- 车型格底部加 2px 车色条，宽度为格宽减 16；选中不透明，未选 alpha 90。
- "难度""车型"标签改为 11px 弱色，放在分段上方，间隔 4。

### 滑杆
- 上一行：左 12px 标签，右 monospace 数值加弱色单位（"马力 (kW)" 拆成"马力"与"kW"）。
- 下一行：`Slider::show_value(true).trailing_fill(true)`；`spacing.slider_width = available − 64`；`slider_rail_height = 4`。
- 轨道用 `widgets.inactive.bg_fill`，填充用 `selection.bg_fill`（模式色），把手圆形。
- 滑杆行间距 10。

### 按钮（自绘，统一函数 `action_button`）
- 主（开始训练 / 继续）：高 36，填充琥珀，文字 `#1A1205`，14px strong，左侧 12px 三角。
- 危险（停止并保存）：高 32，填充危险色 alpha 30，描边 `#CE5C54`，文字 `#E57F78`，左侧 10px 方块。
- 次要（暂停 / 看板 / 续训 / 回到起点 / 随机）：高 28，填充 `#262932`，描边 `#343845`；暂停按钮左侧双竖条。
- 禁用：填充 `#1E2027`，无描边，文字 `#5A5F6A`。
- 删除二次确认态：实心 `#CE5C54`，白字。
- 训练中三个按钮同一行：宽 `(W−12)/3`，间距 6，276px 下每个 88px。

### 模型行
- 高 44，行间距 4，整行可点击。悬停 `#2F333E`；选中模式色 alpha 28 + 描边（模式色 alpha 120）。
- 左侧 3px 指示条，上下各内缩 6；选中模式色，播放中绿色。
- 第一行 13px 主色："DQN · 200k 步"，位于 y+8。第二行 11px 弱色："完圈 68% · 最佳 41.2s"，位于 y+25，x+14。
- 展示厅右侧：6px 圆点 + 11px "播放中"（绿）/"已暂停"（弱色）。
- 训练工作台右侧：续训、删除两个次要按钮，各宽 48、高 28，垂直居中。
- `ScrollArea::max_height`：展示厅 188，训练工作台 140。

### 大播放键
- 全宽，高 64，圆角 12。
- 播放态：填充 `#48B270`，顶部内缩 12px 画一条 1px 白色 alpha 40 高光线，文字 `#08180E`。
- 三角顶点：(cx−9, cy−11)、(cx−9, cy+11)、(cx+13, cy)。
- 暂停态：填充 `#2B2F39`，描边 1.5px `#48B270`，文字 `#E8EBF0`；双竖条各 7×22，圆角 2，中心 cx±7.5，颜色 `#6FD596`。
- 禁用：填充 `#232730`，图标与文字 `#5F6470`。
- 文字 22px strong；图标宽 22，图标与文字间距 12，整组居中：`x0 = cx − (22+12+tw)/2`。

### 数值统计
- 训练中用 2×2 网格，格宽 `(W−8)/2`，高 46。每格：11px 弱色标签，下面 18px strong monospace 数值 + 11px 弱色单位。
- 其余行用 22px 高的左右对齐行：标签在左，monospace 数值右对齐，行间 1px `#262932` 细线。

### 输入框
- `TextEdit::singleline` 配 monospace 13、`margin(8,6)`、`desired_width = W − 72`；右侧"随机"按钮宽 64、高 28。
- `selection.stroke` 改成模式色（现在 lib.rs 里是 BLACK），聚焦时描边可见。

### CollapsingHeader
- 标题 12px strong 次色，`spacing.indent = 12`。
- 展开内容放进 `#14161B` 底、圆角 6、内边距 8 的 Frame，默认折叠。

## E. 信息层级

- 常驻：标题行、模式切换、主操作、环境分段、模型列表。
- 折叠：调校（已折叠）、奖励口径（改成 CollapsingHeader）。
- "换赛道会重建环境"这句只在训练中显示。上次存档缩成 11px 一行。
- 空闲：状态 pill 灰色空心圆点 + "空闲"；琥珀主按钮占视觉中心；统计网格隐藏。
- 训练中：pill 绿色实心圆点 + "训练中"；显示统计网格；危险按钮可见；训练卡左侧 3px 琥珀条。
- 暂停：pill `#DCB95F` + 双竖条图标；统计数值 alpha 降到 150；"继续"升级为琥珀主按钮。

## F. 画布 HUD（track_view.rs draw_hud）

- 徽标 pill：锚点 `rect.left_top()+(12,12)`。字号 12，内边距 (9,5)，圆角取高度一半。最左画 6px 状态圆点，圆点后间隔 6 接文字。返回值 rect 不要丢弃。
- 状态行：`pill.left()+2`，`pill.bottom()+8`，13px，`HUD_DIM`。取代硬编码的 (14,40)。
- 结果徽标：`status.bottom()+6`，与徽标左对齐，12px strong，做成 pill，填充结果色 alpha 40、描边 alpha 140，取代裸文字。
- 速度：`FontId::monospace(34)`，`RIGHT_TOP` 对齐 `rect.right_top()+(-16,12)`；"km/h" 11px，锚在数字 bottom()+2；两者外包 `black_alpha(110)`、圆角 10 底板，外扩 (10,8)。

## G. 实现映射

- 卡片：`Frame::new().fill().stroke().corner_radius(CornerRadius::same(8)).inner_margin(Margin::same(12)).show()`。
- 自绘控件：`let (rect, resp) = ui.allocate_exact_size(vec2(w, h), Sense::click())` 后用 `ui.painter()`：
  `rect_filled`、`rect_stroke(.., StrokeKind::Inside)`、`Shape::convex_polygon`（三角）、`circle_filled`（圆点）。
- 悬停/按下：`resp.hovered()`、`resp.is_pointer_button_down_on()`；`on_hover_cursor(CursorIcon::PointingHand)`。
- 分段内部格子：`Rect::from_min_size` 手算后逐格 `ui.interact(rect, id, Sense::click())`。
- 模型行内按钮：`ui.put(btn_rect, Button)`。
- 文字宽度：`painter.layout_no_wrap`。
- 全局：在 lib.rs `configure_style` 设 `widgets.*.corner_radius`、`slider_rail_height`、`indent`、`inactive.bg_stroke`。

## H. 不要做

1. 每帧 `scan_models()` 读目录和 `.stats` 文件——缓存 1 秒。
2. 在 `horizontal` 里平铺 6 个 `selectable_label`（276px 会溢出），必须用分段算法。
3. 用 ▶ 等字形做图标，只允许 Painter 几何。
4. 给每个区块配不同颜色，或叠加渐变 Mesh、阴影。
5. 嵌套多层 `ScrollArea`，或在 Frame 内对 `available_width` 反复 `set_width`。

## 最小改动路径（3 个文件，约 80% 观感）

1. `crates/app/src/lib.rs`：改 `configure_style`，写入色板、圆角 6、描边、`slider_rail_height`、`selection.stroke`、`indent`。
2. `crates/app/src/view/controls.rs`：先写 `section_card`、`segmented`、`action_button`；用它们替换散落的 `selectable_label`、"▶ 开始训练" 和所有 separator；再重做模型行与 `transport_button`。
3. `crates/app/src/view/track_view.rs`：改 `draw_pill`（加圆点、返回 rect）与 `draw_hud`（锚点链式推算、结果 pill、速度底板）。

`app.rs` 的页签栏和状态行放第二期。

## 风险

- `Slider` 把手形状/半径的可调项未对照 egui 0.37 源码核实；不可调就接受默认大小。
- `ui.put` 与整行点击重叠时，后注册的控件优先响应点击。
