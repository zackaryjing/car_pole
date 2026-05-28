# 游戏阶段 Plan

## 1. 设计原则

这个游戏本质上要成为一个 RL environment，而不是只做一个 pygame 小游戏。
所以核心设计是：状态推进、碰撞判定、奖励计算、观测生成、渲染彼此分离。
同一份内部世界状态可以被人工玩家、图像观测智能体、结构化观测智能体共同使用。

## 2. 第一版游戏范围

第一版保持二维俯视角。

- 车辆：
  - 状态包含位置、朝向、线速度、角速度。
  - 动作为离散键盘/离散 RL action：左转、右转、加速、刹车/倒车、无操作，以及组合动作。
  - 先使用简单运动学模型，不引入复杂轮胎动力学。
- 赛道：
  - 随机生成一条从起点到终点的中心线。
  - 用固定宽度沿中心线生成可行驶区域。
  - 边界外视为失败或强惩罚。
  - 终点线通过后视为成功。
- 障碍物：
  - 在赛道可行驶区域内随机放置圆形 pole。
  - 与 pole 碰撞视为失败，或第一版先设置大惩罚并 reset。
- 回合：
  - 成功：通过终点。
  - 失败：出界、撞 pole、超出最大步数。

## 3. 模块划分

建议代码结构：

```text
src/rl_racing/
  __init__.py
  config.py          # 环境参数、颜色、尺寸、奖励权重
  geometry.py        # 向量、线段距离、碰撞、坐标变换
  track.py           # 随机赛道生成、边界、progress 计算
  vehicle.py         # 车辆状态和运动学积分
  env.py             # reset/step/observe/reward/done
  observations.py    # 图像观测和结构化观测
  renderer.py        # pygame 渲染，follow/global 两种 camera
  play.py            # 人工试玩入口
```

## 4. 内部状态表示

环境内部状态保持完整且与视角无关：

- `track.centerline`: 世界坐标下的中心线点列。
- `track.width`: 赛道宽度。
- `track.obstacles`: 圆形障碍物的世界坐标和半径。
- `vehicle.pose`: 车辆世界坐标位置和朝向。
- `vehicle.velocity`: 车体坐标或世界坐标速度。
- `progress`: 当前沿赛道中心线的前进进度。

渲染模式只改变 camera transform，不改变这些状态。

## 5. 观测接口

先预留两类观测。

### 5.1 图像观测

图像观测来自 follow camera 的局部视野，适合 CNN/DQN/PPO：

- shape 可设为 `(84, 84, 3)` 或 `(96, 96, 3)`。
- 车固定在图像下半部或中心，朝上。
- 渲染内容包括赛道边界、pole、终点方向。
- 可选择 RGB 或灰度。

### 5.2 结构化局部观测

模拟人类有限视野，不直接给全地图。

建议第一版使用 ray casting + 车辆状态：

- 车辆自身状态：
  - 归一化速度。
  - 归一化角速度。
  - 与赛道切线方向的夹角。
  - 当前 progress。
- 局部赛道感知：
  - 以车辆朝向为中心发出固定数量射线，比如 17 或 33 条。
  - 每条射线返回到赛道边界的归一化距离。
  - 每条射线返回到最近 pole 的归一化距离，没有命中则为 1。
  - 可额外返回该方向是否看到终点线。
- 前方路形：
  - 取未来若干 progress 采样点，比如 10 个。
  - 转换到车辆局部坐标系。
  - 只保留有限距离内的点，并归一化到 `[-1, 1]`。

这样结构化观测既不是全知视角，也不会被纯像素学习拖慢早期实验。

## 6. 奖励设计第一版

第一版奖励保持简单，方便调试：

- 每步奖励：沿赛道 progress 的增量。
- 小惩罚：时间惩罚，鼓励快到终点。
- 大惩罚：出界或碰撞。
- 大奖励：到达终点。
- 可选惩罚：速度方向与赛道方向夹角过大。

先不要把奖励做得太复杂。人工试玩确认任务可解之后，再逐步调参。

## 7. 渲染和运行方式

pygame 渲染入口支持参数：

```bash
python -m rl_racing.play --view follow
python -m rl_racing.play --view global
python -m rl_racing.play --seed 0
```

headless server 上人工试玩：

- 优先使用 `ssh -X` 或 `ssh -Y` 做 X11 forwarding。
- 本地需要 X server，比如 Linux 桌面、macOS XQuartz、Windows VcXsrv/MobaXterm。
- 如果只跑训练或测试，使用 `SDL_VIDEODRIVER=dummy`，不打开窗口。

## 8. 实现里程碑

### Milestone 1: 可玩游戏

- 完成随机赛道生成。
- 完成车辆运动学。
- 完成碰撞、终点、reset。
- 完成 global/follow 两种渲染。
- 完成键盘试玩入口。

### Milestone 2: 环境 API

- 实现 `reset(seed=None) -> observation, info`。
- 实现 `step(action) -> observation, reward, terminated, truncated, info`。
- 实现图像观测和结构化观测切换。
- 加入基础单元测试，覆盖赛道生成、碰撞、done 条件。

### Milestone 3: RL 准备

- 加入 replay buffer、rollout buffer 等基础组件。
- 先用随机策略和人工策略跑通 episode 记录。
- 保存训练日志和 episode replay。

### Milestone 4: 算法

- DQN：离散动作 + 图像/结构化观测。
- PPO：离散动作起步，后续可扩展连续动作。
- 对比不同观测方式的 sample efficiency。

