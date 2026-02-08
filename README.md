# AI 贪吃蛇训练项目（DQN）

这是一个可直接运行的强化学习项目，用 DQN 训练 AI 玩贪吃蛇。  
当前版本重点是“先跑通训练闭环”：环境、训练、评估、可视化演示、模型保存与加载都已具备。

## 1. 项目目标

- 提供一个清晰、可复现的 DQN 训练基线
- 训练出能稳定吃到食物的贪吃蛇策略
- 支持训练后快速评估和可视化演示

## 2. 核心能力

- 自定义贪吃蛇环境：`reset / step / render`
- DQN 智能体：
  - 经验回放（Replay Buffer）
  - 目标网络（Target Network）
  - epsilon-greedy 探索
- 训练日志输出（CSV）
- 模型 checkpoint 保存/加载
- 评估脚本（平均分、最高分、平均步数）
- pygame 可视化自动游玩

## 3. 项目结构

```text
ai_game/
  agents/
    dqn_agent.py
    replay_buffer.py
  env/
    snake_env.py
  train/
    train_dqn.py
    eval.py
  ui/
    play_demo.py
  configs/
    dqn.yaml
  outputs/                 # 训练后自动生成（模型与日志）
  requirements.txt
  start.bat                # 一键启动脚本（Windows）
  README.md
```

## 4. 状态、动作与奖励设计

### 4.1 状态（11 维）

- 前方是否危险
- 右方是否危险
- 左方是否危险
- 当前朝向（左/右/上/下，4 维 one-hot）
- 食物相对蛇头的位置（食物在左/右/上/下，4 维）

### 4.2 动作（相对动作）

- `0`：直行
- `1`：右转
- `2`：左转

### 4.3 奖励（默认）

- 吃到食物：`+10`
- 死亡（撞墙或撞自己）：`-10`
- 每步：`-0.01`
- 可选 shaping（靠近食物加分、远离扣分）

## 5. 环境要求

- Python 3.10+
- Windows / Linux / macOS（`start.bat` 仅用于 Windows）
- 建议使用虚拟环境

## 6. 安装依赖

```bash
pip install -r requirements.txt
```

## 7. 运行方式

### 7.1 命令行方式

1. 训练：

```bash
python train/train_dqn.py --config configs/dqn.yaml
```

2. 评估：

```bash
python train/eval.py --config configs/dqn.yaml --model outputs/checkpoints/latest.pt --episodes 100
```

3. 演示（打开 pygame 窗口）：

```bash
python ui/play_demo.py --config configs/dqn.yaml --model outputs/checkpoints/latest.pt --fps 12
```

### 7.2 一键启动（Windows）

双击 `start.bat`，或在终端运行：

```bash
start.bat
```

脚本提供菜单：

- `1` 初始化环境并安装依赖
- `2` 开始训练
- `3` 评估最新模型
- `4` 可视化演示
- `5` 全流程（初始化 -> 训练 -> 评估 -> 演示）

## 8. 训练输出说明

默认输出目录：`outputs/`

- `outputs/checkpoints/latest.pt`：最新模型
- `outputs/checkpoints/dqn_ep*.pt`：周期保存模型
- `outputs/logs/train_metrics.csv`：训练过程指标

`train_metrics.csv` 字段：

- `episode`
- `score`
- `total_reward`
- `avg_loss`
- `epsilon`
- `steps`

## 9. 配置文件说明

配置文件路径：`configs/dqn.yaml`

- `env`：地图大小、奖励参数、渲染参数等
- `agent`：DQN 网络和训练超参（`gamma/lr/batch_size/epsilon` 等）
- `train`：训练轮数、最大步数、日志间隔、checkpoint 间隔、输出目录

建议先仅调整以下参数：

- `train.episodes`
- `agent.lr`
- `agent.epsilon_decay`
- `env.reward_shaping`

## 10. 常见问题

1. 训练分数不涨？
- 先增加训练轮数（如 800 -> 3000）
- 适当调低学习率（如 `1e-3 -> 5e-4`）
- 调整 `epsilon_decay`，避免探索衰减太快

2. 演示窗口打不开？
- 确认已安装 `pygame`
- 远程无图形界面环境下可先只跑训练/评估

3. 评估报模型不存在？
- 先运行训练，确认有 `outputs/checkpoints/latest.pt`

## 11. 后续迭代建议

- 接入 TensorBoard
- 增加 Double DQN / Dueling DQN
- 支持并行采样与更快训练
- 扩展为 CNN 输入整张棋盘（PPO 或更强 DQN 变体）
