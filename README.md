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

控制：

- `W` / `Up`: 加速。
- `S` / `Down`: 刹车，速度降到 0 后继续按会倒车。
- `A` / `Left`: 左转。
- `D` / `Right`: 右转。
- 松开方向键后方向盘会逐步自动回正，不会瞬间归零。
- `R`: 用当前 seed 重置。
- `N`: 切到下一个 seed。
- `V`: 切换 `follow` / `global` 视角。
- `Esc`: 退出。

训练代码不经过 pygame 主循环；直接调用 `env.step(action)`，所以不会被渲染 FPS 限速。

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
