# RL Racer — Rust 强化学习赛车实验台

桌面端 2D 俯视赛车游戏，用于强化学习实验：小赛车通过 RL 学会完成程序生成的赛道，
游戏窗口内嵌实时训练看板。纯 Rust 实现（仿真 + burn 深度学习 + egui 界面），零 Python 依赖。

## 快速开始

```bash
cargo run --release -p rl-racer-game   # 游玩端（Bevy，看 AI 开车）
cargo run --release -p rl-racer-app    # 训练站（eframe，训练 + 看板）
cargo test --release --workspace       # 全部测试（注意：debug 编译产物 ~7GB，建议用 release）
```

两个应用共享 sim/rl 核心与 checkpoints/ 模型格式：

- **游玩端**（`crates/game`，Bevy 0.20）：选赛道/车型/模型 → 看 AI 驾驶；
  全景/跟随镜头、速度变焦、轨迹烟粒子、雷达射线；桌面 + 安卓同构
- **训练站**（`crates/app`，eframe）：DQN/PPO 训练、曲线看板、模型管理（观看/续训/删除）

安卓版：`dist/rl-racer.apk`（arm64）。构建在 Mac（10.10.110.29）上进行：

```bash
# 源码同步后，在 Mac 上：
cd ~/workspace/playground/rl-racer
ANDROID_HOME=~/Library/Android/sdk NDK_HOME=$ANDROID_HOME/ndk/25.1.8937393 \
  cargo apk build --release -p rl-racer-app --lib
adb install -r target/release/apk/rl_racer_app.apk
```

要点：eframe 需 `android-native-activity` 特性（与 accesskit 冲突，安卓端关闭）；
签名 keystore 在 `crates/app/rl-racer.keystore`（不入库）；manifest 嵌套为
`[package.metadata.android.application.activity]`；横屏锁定 + 触屏驾驶按钮。

应用内分成两个模式，互不影响：

**训练工作台**（训练就只是训练）

1. 选赛道（随机/真实）/ seed / 难度 / 车型，选算法（DQN / PPO），**开始训练**
2. 中央区两个页签：**训练镜像**（训练线程里那辆车本身）与 **训练看板**（曲线）
3. 训练中随时换赛道 / 换车型 / 换 seed：交给训练线程在安全点重建环境
   （当前局作废、策略权重保留、经验池清空），看板上留一条"换环境"竖线
4. 停止训练自动存档到 `checkpoints/`，模型列表里能看到完圈率与最佳圈时

**展示厅**（展示就只是展示）

面板从上到下是三段，只有一套播放入口：

1. **选择模型**：单选列表（算法、步数、完圈率、最佳圈时），选中的行高亮，正在跑的那条标「播放中」
2. **大播放键**：未选模型时灰掉；选中后变绿三角，按一下载入并开跑；播放中变「暂停」双竖条。
   旁边只有一个小「回到起点」。双击模型行可以直接开跑
3. **展示环境**：赛道 / seed / 难度 / 车型 / 调校，随便换，只影响展示环境

删除模型放在**训练工作台**的「存档模型」里（续训、删除），展示厅不再碰文件——
这样"看模型"和"管模型"不混在一起。

## 界面

右侧控制面板与外壳按一份成形规范做（`docs/ui-spec.md`），要点：

- 深色单强调色：训练用琥珀 `#D69134`，展示用冷蓝 `#78AADC`；绿色 `#48B270` 只表示"正在跑"
- 信息成块：每块一张卡片（圆角 8、内边距 12、1px 描边），块间 8px，不再用满屏分割线
- 分段选择器代替平铺文字按钮，等分宽度 + 窄屏回退（300px 面板不溢出）
- 数值一律等宽字体、单位弱化；图标全部用 `Painter` 画几何（不用 emoji 字形）
- 主题在 `configure_style` 里锁定为 Dark（否则系统浅色时整个应用会退化成浅色控件）

实拍：`docs/shots/train-3.png`（训练工作台）、`docs/shots/demo-3.png`（展示厅）、`docs/shots/board-3.png`（看板）。

## 展示厅：连续跑圈

展示厅里完圈不停车：仿真自带的 `SimConfig::continuous_laps` 只把圈计时/里程回绕，
车接着跑下一圈，圈数一路累加，`完圈` 徽标闪 2.5 秒后自动消失。
撞墙或超时仍然回起点（车已经废了/太慢）。

训练默认 `continuous_laps = false`——训练必须"一圈 = 一个 episode"，否则完圈奖励与时序建模都会乱。

## 画布镜头

两种模式，右下角浮层切换（也支持滚轮直接缩放）：

| 模式 | 操作 | 说明 |
|---|---|---|
| **自由** | 拖动平移、滚轮缩放、双击回正 | 以光标为锚点缩放，看一眼整条赛道或者贴近看弯道 |
| **跟随** | 车恒在画面中心，滚轮/滑杆调缩放 | 展示厅是平滑跟车（每帧收敛 22%）；训练镜像一秒几千步、车每帧能跳一百多米，所以落后超过半个视口就直接贴上去，变成"追车镜头" |

两种模式各自记住缩放档位（0.3–30 px/m），切来切去不会把对方手感改掉；换赛道/换环境后自动重新对准整条赛道。

演示用：`RL_RACER_CAM=follow` 启动即进跟随镜头。

## 训练可视化（训练站的重点）

看板的目的不是"曲线好看"，而是每一块都对应一个 RL 概念：

| 区域 | 看什么 | 对应的问题 |
|---|---|---|
| KPI 卡片 | 步数 / SPS / 局数 / 完圈率 / 最佳圈时 / 平均奖励 / ε / 经验池 | 现在到底训得怎样，一眼够用 |
| 局总奖励 | 每局原始值 + 50 局滑动平均 + 里程碑竖线（首次完圈、刷新圈时） | 学习曲线是不是真的在涨，拐点在哪 |
| 完圈率 | 近 50 局完圈比例 | 策略从"会开"到"开完"的那段台阶 |
| 奖励分解 | 进度 / 速度 / 走线 / 时间 / 撞墙 / 完圈 六项堆叠面积 | 策略在追哪一项奖励、被哪一项惩罚 |
| 最近 80 局 | 绿=完圈、红=撞墙、黄=超时 的结果条 | 稳定性：是连贯完圈还是碰运气 |
| loss / Q 均值 / 策略熵 | 误差、估值、动作不确定性 | 网络在学什么、有没有塌缩 |
| ε / SPS / 经验池 | 探索率衰减、训练速度、样本量 | 探索够不够、机器跑得动吗 |

画布上：

- **训练镜像**页画的是训练线程里的那辆车本身——车的位置/朝向/车速/9 条观测射线都来自
  训练环境那一步，不是另开一个环境重放。徽标「训练环境 · 实时镜像」标明这一点
- **轨迹叠加**：每局跑完留一条，旧淡新亮，完圈绿 / 撞墙红，最新一条按车速着色，
  能直接看到策略逐步收敛到走线
- **展示厅徽标**「展示厅 · 存档模型」：这里的车是载入的模型在独立环境里跑
- **左下策略判断**（展示厅）：DQN 画出 9 个动作的实时 Q 值表（行=油门 +1/0/−1，列=转向），
  高亮当前被选中的动作；PPO 画动作均值 μ、标准差 σ 与状态价值 V
- **底部实时遥测**（展示厅）：速度 / 横向偏移 / 转向 / 油门 四条滚动曲线，看清"观测 → 动作 → 结果"

存档旁边会写一个同名 `.stats`（纯文本 `key=value`），模型列表里直接显示完圈率与最佳圈时，
不用重新跑一遍才知道这个模型好不好。

演示/截图用环境变量（不影响正常启动）：

```bash
RL_RACER_DEMO=1 RL_RACER_ALGO=ppo RL_RACER_TAB=board ./target/release/rl-racer   # 开训 + 停在看板
RL_RACER_MODE=demo ./target/release/rl-racer                                      # 直接进展示厅
RL_RACER_MODEL=dqn-h128-seed7-s200000.rec ./target/release/rl-racer               # 直接播某个模型
```

## 已验证的学习效果（简单赛道，seed 42）

| 算法 | SPS | 首次完圈 | 稳定圈时 | 收敛奖励 |
|---|---|---|---|---|
| DQN (Double) | ~2 400 | 30k 步 | ~29.4s | ~453 |
| PPO | ~7 700 | 40k 步 | ~28.5s | ~454 |

无头基准：纯仿真 **98 000 SPS**（随机策略）。

无头训练脚本（不带 UI）：

```bash
cargo run --release -p rl --example train_dqn_headless -- 200000 42
cargo run --release -p rl --example train_ppo_headless -- 300000 42
cargo run --release -p sim --example headless -- 1000000 42
```

## 架构

```
crates/
├── sim/    纯仿真（零依赖）：赛道生成、车辆运动学、观测、奖励、episode 状态机
├── rl/     burn 网络：Double DQN + PPO + 统一 Agent trait + 推理策略
└── app/    eframe/egui：赛道画布 + 实时遥测 + 训练看板（egui_plot）、训练线程、观赛
```

关键设计：

- **训练/展示彻底分离**：训练环境只存在于训练线程里，UI 通过 `TrainFrame`
  （约 30 Hz、每帧几十字节）把它的车"投影"到画布上——训练镜像和训练看板看的是同一个环境；
  展示厅另有一个完全独立的推理环境，训练与展示之间没有任何共享可变状态。
  训练中改环境走 `ToTrainer::SetEnv`，由训练线程在安全点重建（`Agent::on_env_changed`
  清空 DQN replay buffer / PPO rollout，策略权重保留），UI 从不直接改训练环境
- **同 seed 完全可复现**：sim 自带 splitmix64 确定性 PRG
- **赛道生成**：随机模式两种形状——谐波星形（简单/中等，直道+弯角组合）与
  **闭合随机游走**（困难，发卡弯/S 弯/平行路段，非星形）；可变宽度（收窄/加宽段）；
  双向弯径约束（防太圆、防不可驾驶）+ 防自交校验（非相邻路段间距 > 赛道宽），
  不达标换 sub-seed 重试，控制多边形先做两轮拉普拉斯平滑保证可驾驶性
- **真实赛道近似**：蒙扎 / 摩纳哥 / 斯帕 / 银石 / 上海——手绘控制点保留
  标志性行结构（蒙扎长直道+帕拉波利卡、斯帕折返+Bus Stop、上海"上"字形细长椭圆等），
  非测绘数据；全部通过防自交校验（诊断：`cargo run --release -p sim --example diag_tracks`）
- **三种车辆预设**：均衡街车 / 极速跑车（快但推头）/ 卡丁车（慢但灵活）——
  换车即动力学迁移实验
- **观测** `[f32; 14]`（栈上零分配）：9 条雷达射线 + 速度 + 航向误差 + 横向偏移 + 上一步动作
- **奖励**：带符号弧长进度（防倒车/转圈刷分）+ 速度 + 走线 − 时间成本；撞墙 −5、完圈 +10
- **奖励分解**：sim 每步返回 `(progress, speed, center, time, crash, lap)` 六项分量，
  标量奖励是它们之和——所以看板能把"总奖励"拆回"为什么"
- **DQN**：Double DQN + Huber loss + 周期硬同步 target + ε-greedy 线性衰减
- **PPO**：双头 actor-critic，可学习 log_std，GAE(λ=0.95) + clip 0.2 + 熵正则，
  超时截断与真终局区分（截断做 value bootstrap）
- **模型管理**：停止训练自动保存；列表可「观看」（载入推理）/「续训」（热启动继续
  训练，保存时替换来源文件，一条血统一个模型）/「删除」（两步确认）

## 环境备忘（Windows / burn 0.22）

- burn **0.22** 是 2026-10 的大重构：无泛型 `Module`、`Device::flex()`、
  `GradientsParams::from_grads` + `ModuleOptimizer::step(lr: f64, ..)`、统一 `ModuleRecord`。
  网上旧教程（`Model<B>` + `Autodiff<B>`）全部不可用，只看官方 book 与 docs.rs
- 不用 burn 的 `cpu` feature（CubeCL 后端）：其 matmul 自动调优在本机触发
  `Shared memory budget exceeded`（L1d 32KB < kernel 需求 64KB）。用 `flex` 纯 Rust 后端
- 磁盘紧张时 `cargo clean`：debug 产物约 7GB，日常用 `--release`
