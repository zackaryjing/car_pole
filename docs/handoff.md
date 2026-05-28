# Project Handoff

这份文档用于在另一台本地机器上继续 agent coding。它汇总项目背景、当前实现状态、设计约束、运行命令、已知问题和下一步任务。

## 1. 总目标

这个项目的长期目标是从头实现一套强化学习学习平台：

- 第一阶段：自己用 Python + pygame 做一个 2D 赛车环境。
- 第二阶段：把这个环境做成稳定的 RL environment，支持图像观测和结构化观测。
- 第三阶段：手写经典 RL 算法，先做 DQN，再做 PPO。
- 第四阶段：比较不同观测方式、奖励设计、课程学习和随机化策略。

项目动机是深入理解 RL，而不是直接调用现成 RL 框架。后续神经网络可以用 PyTorch，RL 框架尽量自己写；如果某个部分不稳定或收益不大，再考虑引入成熟库。

这个方向和具身智能有关：用户希望通过一个可控的仿真环境，理解 RL 中的环境建模、状态表示、奖励设计、探索、训练稳定性和视觉输入问题。

## 2. 当前任务定义

当前游戏不是传统 cart-pole，而是用户自定义的 2D racing task：

- 赛道随机生成。
- 有起点和终点。
- 赛道上有随机圆形 pole 障碍。
- 小车从起点开到终点算成功。
- 出界或撞 pole 算失败。
- 支持人工试玩。
- 支持后续 RL 训练。

重要设计约束：

- 内部世界状态和渲染方式解耦。
- `follow` 视角和 `global` 视角只改变 camera，不改变环境状态。
- 训练时不依赖 pygame 窗口。
- 图像观测和结构化观测共享同一套环境状态。
- 模拟推进和人工渲染帧率解耦。

## 3. 当前代码状态

仓库地址：

```text
git@github.com:zackaryjing/car_pole.git
```

当前主分支：

```text
main
```

最近关键提交：

```text
62297f4 Decouple manual simulation speed from rendering
0ffc314 Add smooth steering dynamics
cde9ad3 Implement first playable racing environment
a75df9a Document racing environment design
e7ed220 Initialize RL racing project plan
```

当前已经实现第一版可玩环境。

## 4. 文件结构

核心代码：

```text
src/rl_racing/
  actions.py       # 离散 action 到 throttle/steer 控制输入
  config.py        # dataclass 配置，集中放环境/车辆/奖励/观测/渲染参数
  env.py           # RacingEnv reset/step/observe/render
  geometry.py      # 2D 几何、polyline 投影、坐标变换、ray-circle
  observations.py  # structured observation，后续 image observation 可继续扩展
  play.py          # pygame 人工试玩入口
  renderer.py      # pygame global/follow 渲染和离屏 RGB frame
  track.py         # 随机赛道生成、障碍生成、progress 查询
  vehicle.py       # 车辆状态和简单运动学
```

测试：

```text
tests/
  test_env.py          # env reset/step/done/image observation
  test_play_timing.py  # 人工试玩模拟速度和渲染速度解耦
  test_track.py        # 赛道 seed determinism/progress/on-track
  test_vehicle.py      # 速度积分、转向积分、自动回正
```

文档：

```text
docs/
  game_design.md  # 长期环境设计基准
  game_plan.md    # 第一阶段计划
  setup.md        # 当前服务器环境和安装记录
  handoff.md      # 本文档，新机器继续开发入口
```

## 5. 当前玩法和控制

人工试玩：

```bash
python -m rl_racing.play --view follow --seed 0
python -m rl_racing.play --view global --seed 0
```

也可以使用脚本入口：

```bash
rl-racing-play --view follow --seed 0
```

控制：

- `W` / `Up`: 加速。
- `S` / `Down`: 刹车，速度降到 0 后继续按会倒车。
- `A` / `Left`: 左转。
- `D` / `Right`: 右转。
- `R`: 用当前 seed 重置。
- `N`: 切换到下一个 seed。
- `V`: 切换 `follow` / `global` 视角。
- `Esc`: 退出。

车辆不是瞬时转向。`Control.steer` 是目标输入，车辆内部有 `VehicleState.steering`，会按 `steering_rate` 积分接近输入目标；松开方向键后按 `steering_return_rate` 自动回正。

## 6. 模拟速度和渲染速度

当前人工试玩中，模拟速度和渲染速度已解耦：

```bash
python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 2.0
```

含义：

- `--render-fps`: pygame 窗口刷新率。
- `--sim-speed`: 真实 1 秒推进多少秒模拟时间。
- `env.step(action)`: 训练时直接调用，不经过 pygame 主循环，不受渲染 FPS 限速。

如果只是觉得车本身慢，可以临时调车辆参数：

```bash
python -m rl_racing.play --seed 0 --sim-speed 1.5 --max-speed 380 --acceleration 520
```

这些 CLI 参数目前只影响人工试玩入口，不会修改默认环境配置。

## 7. Environment API

主类是 `RacingEnv`：

```python
from rl_racing.env import RacingEnv

env = RacingEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render(mode="rgb_array", view="follow")
```

返回语义接近 Gymnasium：

- `terminated=True`: 成功、出界、撞障碍等任务语义结束。
- `truncated=True`: 达到 `max_steps`。
- `info`: 调试信息，不应默认作为训练输入。

`info` 当前包含：

- `progress`
- `lap_distance`
- `off_track`
- `collision`
- `success`
- `steps`
- `seed`
- `done_reason`
- `reward`

## 8. Observation

当前默认观测是 structured observation。

shape 当前是：

```text
(67,)
```

组成：

- 速度，归一化。
- 角速度，归一化。
- steering 状态，范围 `[-1, 1]`。
- 与赛道切线方向的 heading error。
- progress。
- 多条 ray 到赛道边界的距离。
- 多条 ray 到障碍的距离。
- 未来若干中心线点在车辆局部坐标中的位置。

图像观测也已经能通过 `obs_type="image"` 生成离屏 RGB frame，主要测试已覆盖 dummy video driver 下不打开窗口也能生成图像。图像观测后续还需要做得更适合训练，比如车头朝上、局部视野裁剪、灰度/缩放、帧堆叠等。

## 9. 当前测试状态

当前测试命令：

```bash
python -m pytest
```

当前结果：

```text
14 passed, 1 warning
```

warning 来自 pygame 依赖里的 `pkg_resources` deprecation，不影响运行。

测试覆盖：

- 赛道 seed determinism。
- 起点在赛道内。
- 沿中心线 progress 单调增加。
- 车辆加速。
- 车辆转向。
- steering 逐步增加和自动回正。
- env reset/step shape。
- off-track termination。
- obstacle collision termination。
- finish success termination。
- dummy video driver 下 image observation。
- 人工 play loop 的 simulation accumulator。

## 10. 本地机器建议安装

推荐在本地机器新建环境，不一定沿用服务器 base 环境：

```bash
conda create -n rl-racing python=3.11 -y
conda activate rl-racing
pip install -e ".[dev]"
```

如果要装 PyTorch，根据本地机器驱动选择对应 CUDA wheel。游戏本身当前只需要：

- numpy
- pygame
- pytest

PyTorch 是后续 RL 训练需要，不是当前 pygame 游戏必须依赖。

在本地有显示器的机器上，不需要 X11 forwarding，直接运行：

```bash
python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 1.5
```

## 11. 为什么不继续用 X11 Forwarding 测试手感

当前 headless server 通过 SSH X11 forwarding 玩 pygame，体验受网络延迟和带宽影响很大。

这个游戏的手感判断需要本地低延迟输入和渲染，所以更适合：

1. 在本地机器 clone 仓库。
2. 本地安装依赖。
3. 本地直接运行 pygame 窗口。
4. 根据真实体验调车辆参数、camera、赛道宽度和障碍密度。

服务器更适合后续训练、批量实验、长时间跑 PyTorch，而不是人工调手感。

## 12. 已知问题和设计债

当前第一版能玩，但还不是最终训练环境。

已知问题：

- 车辆模型仍然很简化，没有轮胎侧滑、转弯半径约束和真实摩擦。
- 赛道生成只是中心线加宽，没有生成精确 polygon 边界。
- `follow` 视角需要本地实际试玩后再调 camera scale、车在屏幕位置和朝向。
- 图像观测只是基础 RGB frame，还没有为 CNN 训练优化。
- 奖励目前是简单 progress + time penalty + success/failure，后续需要实验。
- 默认车辆速度和加速度可能仍需要手感调参。
- reset 后碰到 terminated 会立即重置，人工试玩时可能看不到失败瞬间，后续可加 pause/death screen。
- 缺少 episode replay、trajectory 记录和 debug overlay 开关。
- 还没有实现 RL 训练代码。

## 13. 下一步建议

建议下一位 agent 在本地机器按这个顺序继续：

1. 本地运行游戏，先调手感。
   - 调 `VehicleConfig.max_forward_speed`
   - 调 `VehicleConfig.acceleration`
   - 调 `VehicleConfig.drag`
   - 调 `VehicleConfig.steering_rate`
   - 调 `VehicleConfig.steering_return_rate`
   - 调 `RenderConfig.follow_pixels_per_unit`

2. 改进人工试玩体验。
   - 增加暂停。
   - episode 失败/成功后停留 1 秒显示结果。
   - 增加 debug overlay 开关。
   - 增加当前 seed/参数导出。

3. 改进赛道生成。
   - 避免过窄或过急弯。
   - 更稳定地放置障碍，保证可通过。
   - 支持不同难度等级。

4. 完善 observation。
   - 结构化观测增加可选字段开关。
   - 图像观测固定车头朝上。
   - 添加灰度、resize、frame stack。
   - 添加 observation shape 测试。

5. 做 baseline policy 和评估工具。
   - random policy runner。
   - simple centerline-following heuristic。
   - episode 成功率统计。
   - 保存 replay 或 mp4。

6. 再进入 RL。
   - replay buffer。
   - MLP DQN for structured obs。
   - CNN DQN for image obs。
   - PPO rollout buffer。
   - TensorBoard 或 CSV logging。

## 14. Coding Rules For Next Agent

继续开发时请遵守：

- 环境状态推进、观测、渲染继续保持解耦。
- 不要让训练代码依赖 pygame window。
- 新增核心行为必须加测试。
- 调整默认参数时说明原因，并尽量在文档记录。
- 关键节点 commit + push。
- 优先保持简单可解释，不要过早引入复杂框架。
- PyTorch 可用于网络和 tensor，但 RL 算法优先手写。

## 15. Quick Start For Next Agent

```bash
git clone git@github.com:zackaryjing/car_pole.git
cd car_pole
conda create -n rl-racing python=3.11 -y
conda activate rl-racing
pip install -e ".[dev]"
python -m pytest
python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 1.5
```

如果游戏太慢：

```bash
python -m rl_racing.play --seed 0 --sim-speed 2.0 --max-speed 420 --acceleration 650
```

如果游戏太难：

- 降低 `--sim-speed`。
- 调低 `max_speed`。
- 临时在 `TrackConfig` 里增大 `width` 或减少 obstacle 数量。

