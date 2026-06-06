# RL Racing Playground

一个从头实现的强化学习探索项目。第一阶段目标是做一个可人工游玩的
2D 随机赛道赛车环境，后续在同一套环境接口上接入手写 DQN、PPO 等算法。

## 当前目标

- 用 `pygame` 实现一个轻量 2D 赛车游戏。
- 随机生成带起点、终点、边界和圆形障碍物的赛道。
- 支持人工键盘控制，便于直觉调试任务难度。
- 支持两种渲染模式：
  - `follow`: 摄像机跟随车辆，只显示局部视野。
  - `global`: 固定视角显示整张地图。
- 游戏状态、物理和奖励逻辑与渲染解耦，为 RL 训练保留接口。
- 默认 RL 观测是局部传感器 `sensor`，不直接暴露中心线偏移或未来中心线点。
- 支持 episode 轨迹保存、加载、离屏回放渲染和 best-record 比较接口。

## 运行

先安装项目：

```bash
python -m pip install -e .
```

人工试玩：

```bash
python -m rl_racing.play --view follow --seed 0
python -m rl_racing.play --view global --seed 0
```

人工试玩时，模拟速度和渲染速度是解耦的。比如渲染仍保持 60 FPS，但每秒推进 2 秒模拟时间：

```bash
python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 2.0
```

也可以临时调高车辆参数试手感：

```bash
python -m rl_racing.play --seed 0 --sim-speed 1.5 --max-speed 380 --acceleration 520
```

也可以使用安装后的脚本：

```bash
rl-racing-play --view follow --seed 0
```

窗口内操作：

- `W` / `Up`: 加速。
- `S` / `Down`: 刹车，速度降到 0 后继续按会倒车。
- `A` / `Left`: 左转。
- `D` / `Right`: 右转。
- 松开方向键后方向盘会逐步自动回正，不会瞬间归零。
- `V`: 切换 `follow` / `global` 视角。
- `R`: 用当前 seed 重置。
- `N`: 切到下一个 seed。
- `Esc`: 退出。

人工试玩、模型在线观察和轨迹回放都会在左下角显示方向键 overlay：

- 灰色表示当前没有按下或模型没有选择该方向。
- 红色表示当前帧正在加速、刹车或转向。
- 组合动作会同时点亮多个键，例如加速右转会同时点亮上键和右键。

训练代码不经过 pygame 主循环；直接调用 `env.step(action)`，所以不会被渲染 FPS 限速。

RL 接口示例：

```python
from rl_racing.env import RacingEnv
from rl_racing.episode import run_episode
from rl_racing.policies import RandomPolicy

env = RacingEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)
result = run_episode(env, RandomPolicy(seed=0), seed=0, trajectory_path="runs/random_seed_0")
```

Sensor DQN 训练入口：

```bash
rl-racing-train-dqn --device auto --total-steps 200000 --run-name dqn_sensor_seed0
```

在 3050 Ti 4GB 这类小显存机器上，建议先降低 batch：

```bash
rl-racing-train-dqn --device cuda --batch-size 64 --replay-size 100000 --total-steps 50000
```

训练输出默认写到 `runs/dqn_sensor/<run-name>/`，包含 `config.json`、
`metrics.csv`、`checkpoints/`、`trajectories/` 和 `best_records/`。

PPO 训练入口：

```bash
rl-racing-train-ppo --device auto --total-steps 2000000 --run-name ppo_sensor_seed0
```

训练后可以在线观察 DQN 或 PPO checkpoint。`watch_policy` 会自动识别 checkpoint 类型：

```bash
python -m rl_racing.watch_policy \
  runs/ppo_sensor/ppo_sensor_seed0/checkpoints/best_eval.pt \
  --device auto \
  --seed 10000 \
  --view follow
```

模型观察窗口内快捷键：

- `V`: 切换 `follow` / `global` 视角。
- `R`: 用当前 seed 重新运行模型。
- `N`: 切到下一个 seed 并重新运行模型。
- `Esc`: 退出。

也可以直接回放保存过的轨迹，例如训练 eval 成功轨迹：

```bash
python -m rl_racing.view_trajectory \
  runs/ppo_sensor/ppo_sensor_seed0/trajectories/step_50000_seed_10000 \
  --view follow \
  --loop
```

轨迹回放窗口内快捷键：

- `V`: 切换 `follow` / `global` 视角。
- `R`: 从轨迹开头重新播放。
- `Esc`: 退出。

运行测试：

```bash
python -m pytest
```

## 设计文档

- 游戏阶段计划见 [docs/game_plan.md](docs/game_plan.md)。
- 长期游戏设计基准见 [docs/game_design.md](docs/game_design.md)。
- 环境安装和验证见 [docs/setup.md](docs/setup.md)。
- 换机器继续开发的上下文见 [docs/handoff.md](docs/handoff.md)。

## Headless 服务器说明

开发环境在 headless server 上时，人工试玩建议使用 X11 forwarding：

```bash
ssh -X user@server
python -m rl_racing.play
```

如果只是跑自动测试或训练，可以使用 SDL dummy video driver：

```bash
SDL_VIDEODRIVER=dummy python -m pytest
```
