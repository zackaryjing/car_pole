# RL Racing Game Design

这份文档是游戏环境的长期设计基准。后续实现、重构、训练算法接入，都应优先保持这里定义的接口和语义稳定。

## 1. Product Goal

我们要做的不是一个复杂赛车游戏，而是一个适合学习强化学习的可控仿真环境。

核心目标：

- 人可以直接玩，快速判断任务是否合理。
- 智能体可以通过同一套 `reset/step/observe` 接口训练。
- 图像观测和结构化观测共享同一个内部世界状态。
- 赛道、障碍、初始条件支持随机化，但可通过 seed 复现。
- 第一版足够简单，后续可以自然扩展到更复杂动力学、连续动作、多任务和课程学习。

非目标：

- 第一版不追求真实车辆物理。
- 第一版不追求高保真视觉效果。
- 第一版不依赖现成 RL environment 框架。
- 第一版不把训练算法和 pygame 渲染绑在一起。

## 2. Environment Contract

环境主类建议命名为 `RacingEnv`，提供接近 Gymnasium 的接口，但不要强依赖 Gymnasium：

```python
obs, info = env.reset(seed=0, options=None)
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render(mode="rgb_array")
env.close()
```

语义：

- `terminated=True`: 任务自然结束，包括到达终点、撞障碍、出界。
- `truncated=True`: 非任务语义的截断，比如达到最大步数。
- `info`: 调试和评估信息，不作为默认训练输入。

`info` 至少包含：

- `progress`: 当前赛道进度，范围 `[0, 1]`。
- `lap_distance`: 已沿赛道中心线前进的距离。
- `off_track`: 是否出界。
- `collision`: 是否撞到 pole。
- `success`: 是否到达终点。
- `steps`: 当前 episode 步数。
- `seed`: 当前 episode seed。

## 3. World Coordinates

内部世界坐标使用右手 2D 平面：

- 单位：像素或仿真单位都可以，第一版用像素单位减少转换成本。
- `x` 向右为正。
- `y` 向下为正，与 pygame screen 坐标一致。
- 角度 `heading` 使用弧度，`0` 表示朝向 `+x`，逆时针为正。

车辆、赛道、障碍都存世界坐标。渲染器负责世界坐标到屏幕坐标的 camera transform。

## 4. Track Generation

第一版赛道使用中心线表示：

```python
Track(
    centerline: np.ndarray,  # shape: (N, 2)
    width: float,
    obstacles: list[CircleObstacle],
    start_pose: Pose2D,
    finish_segment: Segment2D,
)
```

### 4.1 Centerline

随机赛道生成流程：

1. 从起点开始生成若干控制点。
2. 每一步向前推进固定距离，并对方向加入有限随机扰动。
3. 限制转角变化，避免出现不可驾驶的锐角。
4. 对控制点做平滑插值或 Chaikin smoothing。
5. 计算累计弧长，得到 `s` 坐标。

默认参数建议：

- 控制点数：12 到 24。
- 每段长度：120 到 220。
- 最大转角扰动：20 到 35 度。
- 赛道宽度：120。
- 最小中心线自交距离：`2.5 * width`。

第一版可以只做开放赛道，从起点到终点。不要一开始做闭环，因为终点判定和 progress 更容易稳定。

### 4.2 Drivable Area

给定车辆位置 `p`：

1. 找到 `p` 到中心线 polyline 的最近点。
2. 得到横向距离 `lateral_distance`。
3. 如果 `lateral_distance <= width / 2 - car_radius`，车辆在赛道内。
4. 最近点所在弧长除以总弧长得到 `progress`。

这个判定近似但足够好。后续如果需要更精确的边界，可以把赛道边界 polygon 化。

### 4.3 Obstacles

pole 是圆形障碍：

```python
CircleObstacle(center: Vec2, radius: float)
```

生成约束：

- 必须在可行驶区域内。
- 距离起点和终点至少 `2 * width`。
- 距离中心线不要总是太近，保留可通过空间。
- 障碍之间保持最小间距。

默认参数建议：

- 半径：12 到 24。
- 数量：`track_length / 450` 左右。
- 与边界的最小距离：`radius + car_radius + 8`。

## 5. Vehicle Model

第一版用稳定、可控、可解释的简单运动学模型。

车辆状态：

```python
VehicleState(
    position: np.ndarray,      # shape: (2,)
    heading: float,            # radians
    speed: float,              # signed scalar
    steering: float,           # normalized steering wheel state, [-1, 1]
    angular_velocity: float,   # radians / second
)
```

离散时间步：

- `dt = 1 / 30` 秒。
- 训练和人工试玩使用同一套 physics step。
- 如果渲染帧率更高，不额外推进物理。

控制输入：

```python
Action(
    throttle: float,  # -1, 0, 1
    steer: float,     # desired steering input, -1, 0, 1
)
```

`steer` 不是瞬时车辆转向角。车辆内部维护 `VehicleState.steering`，按 `steering_rate` 逐步接近输入目标；没有转向输入时，按 `steering_return_rate` 自动回正。这样键盘和 RL 离散动作都不会让车辆方向瞬间跳变。

离散 action 映射：

| id | throttle | steer | meaning |
| --- | --- | --- | --- |
| 0 | 0 | 0 | coast |
| 1 | 1 | 0 | accelerate |
| 2 | -1 | 0 | brake/reverse |
| 3 | 0 | -1 | steer left |
| 4 | 0 | 1 | steer right |
| 5 | 1 | -1 | accelerate left |
| 6 | 1 | 1 | accelerate right |
| 7 | -1 | -1 | brake left |
| 8 | -1 | 1 | brake right |

动力学：

```text
speed += throttle * accel * dt
speed -= drag * speed * dt
speed = clip(speed, -max_reverse_speed, max_forward_speed)

steering = move_toward(steering, steer_target or 0, steering_rate_or_return_rate * dt)
turn_rate = steering * max_turn_rate * speed / max_forward_speed
heading += turn_rate * dt
position += [cos(heading), sin(heading)] * speed * dt
```

注意：

- 低速转向能力可以保留一部分，避免车完全卡死。
- 第一版不模拟侧滑，但可以把“朝向偏离赛道切线”作为奖励 shaping。
- 车辆碰撞体先用圆形 `car_radius`，后续再换矩形。

## 6. Termination

一个 episode 结束条件：

- 成功：`progress >= 1.0` 或穿过 finish segment。
- 失败：车辆圆心加半径越界。
- 失败：车辆圆与任意 pole 圆相交。
- 截断：步数达到 `max_steps`。

推荐默认：

- `max_steps = 2000`。
- 撞障碍和出界直接 `terminated=True`。
- 训练早期可以提供一个 `soft_fail=False/True` 配置，soft fail 只给大惩罚但不结束，便于探索对比。

## 7. Reward

第一版奖励必须简单，便于定位学习失败原因。

基础奖励：

```text
reward = progress_delta * progress_scale
reward += time_penalty
reward += success_bonus if success
reward += failure_penalty if collision or off_track
```

默认参数建议：

- `progress_scale = 100.0`
- `time_penalty = -0.01`
- `success_bonus = 100.0`
- `failure_penalty = -50.0`

可选 shaping：

- `heading_alignment = cos(vehicle_heading - track_tangent_heading)`
- 小权重奖励朝向和赛道切线一致。
- 对倒车或长时间无进展加惩罚。

原则：

- progress 是主奖励。
- shaping 不能大到让模型学会绕圈、蹭边或停在局部高奖励区域。
- 每次调整 reward 都要保留配置和实验记录。

## 8. Observation Design

观测生成独立于 physics 和 renderer。环境内部是全状态，但智能体只能拿到配置指定的观测。

### 8.1 Image Observation

图像观测来自局部 camera，不直接暴露全地图。

默认：

- `obs_type="image"`
- shape: `(96, 96, 3)`
- dtype: `uint8`
- camera: follow view
- 车辆绘制在图像中心偏下，朝向屏幕上方。

图像内容：

- 赛道区域。
- 边界。
- pole。
- 终点线或终点方向提示。
- 车辆自身。

实现要求：

- 图像观测不能依赖可见 pygame window。
- 可以用 pygame Surface 离屏渲染。
- 人工渲染窗口和训练图像观测共享 renderer 核心逻辑。

### 8.2 Structured Observation

结构化观测用于早期算法验证，避免一开始被视觉学习难度拖慢。

建议 shape：

```text
[
  ego_speed,
  ego_angular_velocity,
  heading_error_to_track,
  progress,
  ray_track_distances[ray_count],
  ray_obstacle_distances[ray_count],
  future_centerline_points[future_count * 2],
]
```

默认：

- `ray_count = 21`
- `ray_fov = 180 degrees`
- `ray_max_distance = 350`
- `future_count = 10`
- `future_step_distance = 80`

归一化：

- speed: 除以 `max_forward_speed`，clip 到 `[-1, 1]`。
- angular velocity: 除以 `max_turn_rate`。
- heading error: 除以 `pi`。
- progress: `[0, 1]`。
- ray distance: `distance / ray_max_distance`，无命中为 `1`。
- future point: 转到车辆局部坐标后除以 `ray_max_distance`，clip 到 `[-1, 1]`。

有限视野约束：

- 结构化观测不包含全局终点坐标。
- 未来中心线点只取有限距离内。
- 障碍只通过 ray 或局部邻域出现。

### 8.3 Privileged Observation

可以为调试和 teacher policy 保留 `obs_type="privileged"`：

- 全中心线。
- 全障碍列表。
- 车辆完整状态。
- 最近中心线投影。

这个观测不能作为默认 RL baseline，避免不小心训练出全知策略。

## 9. Rendering

渲染器只读世界状态，不推进环境。

### 9.1 Global View

用途：

- 调试赛道生成。
- 看完整 episode 行为。
- 制作 replay。

特点：

- 固定 camera，自动缩放到完整赛道 bounding box。
- 显示起点、终点、障碍、车辆轨迹。
- 可以叠加 progress 和 reward debug 文本。

### 9.2 Follow View

用途：

- 人工试玩。
- 图像观测。
- 模拟有限视野。

特点：

- camera 跟随车辆。
- 可选择车头始终朝上，或世界方向固定。
- 训练图像观测默认使用“车头朝上”，降低视觉学习难度。
- 人工试玩可以用“世界方向固定”，手感更直观。

### 9.3 Headless Support

运行模式：

- 有窗口人工玩：正常 pygame display。
- SSH X11 forwarding：依赖本地 X server 和 `ssh -X`/`ssh -Y`。
- 无窗口测试/训练：`SDL_VIDEODRIVER=dummy`。
- 图像观测：使用离屏 Surface，不要求 display。

## 10. Manual Play

人工试玩入口：

```bash
python -m rl_racing.play --view follow --seed 0
python -m rl_racing.play --view global --seed 0
```

默认按键：

- `W` 或 `Up`: 加速。
- `S` 或 `Down`: 刹车/倒车。
- `A` 或 `Left`: 左转。
- `D` 或 `Right`: 右转。
- `R`: 重置当前 seed。
- `N`: 新随机 seed。
- `V`: 切换 follow/global。
- `Esc`: 退出。

屏幕上可以显示：

- speed。
- progress。
- reward。
- step。
- seed。
- done reason。

## 11. Determinism

所有随机性必须来自环境持有的 RNG：

```python
rng = np.random.default_rng(seed)
```

要求：

- 相同 seed 下赛道、障碍、起始状态一致。
- `reset(seed=seed)` 必须重置 RNG。
- 测试里固定 seed，避免 flaky。
- 训练日志记录 seed 和配置 hash。

## 12. Testing Strategy

第一批测试不需要覆盖 pygame 视觉细节，先覆盖环境语义。

必须测试：

- 相同 seed 生成相同 track。
- 车辆在起点时 `off_track=False`。
- 手动放到赛道外时 `off_track=True`。
- 手动放到 pole 上时 `collision=True`。
- progress 沿中心线前进时单调增加。
- 到达终点后 `terminated=True` 且 `success=True`。
- `reset/step` 返回字段 shape 和 dtype 正确。
- dummy video driver 下 image observation 可生成。

## 13. File Layout

长期建议结构：

```text
src/rl_racing/
  __init__.py
  actions.py
  config.py
  env.py
  geometry.py
  observations.py
  play.py
  renderer.py
  track.py
  vehicle.py

tests/
  test_track.py
  test_vehicle.py
  test_env.py
  test_observations.py

docs/
  game_plan.md
  game_design.md
```

## 14. Implementation Order

推荐执行顺序：

1. `geometry.py`: 向量工具、线段投影、圆碰撞。
2. `track.py`: 固定 seed 的随机中心线和 progress 查询。
3. `vehicle.py`: 简单运动学 step。
4. `env.py`: reset/step/reward/done。
5. `renderer.py`: global view。
6. `renderer.py`: follow view。
7. `play.py`: 键盘人工试玩。
8. `observations.py`: structured observation。
9. `observations.py`: image observation。
10. tests: 补齐核心行为测试。

## 15. RL Extension Points

后续算法层不要直接碰 pygame，也不要直接读取 renderer 的状态。

训练代码只依赖：

- `env.reset`
- `env.step`
- `env.observation_space_spec`
- `env.action_space_spec`
- `env.render(mode="rgb_array")`

后续目录可以加：

```text
src/rl_racing/rl/
  replay_buffer.py
  rollout_buffer.py
  networks.py
  dqn.py
  ppo.py
  train.py
  eval.py
```

先做离散动作算法：

- DQN + structured observation。
- DQN + image observation。
- PPO + structured observation。
- PPO + image observation。

如果后续改连续控制，再增加 continuous action wrapper，不破坏已有离散接口。

## 16. Configuration

配置应使用 dataclass，不要散落 magic numbers。

建议：

```python
@dataclass(frozen=True)
class EnvConfig:
    dt: float = 1 / 30
    max_steps: int = 2000
    obs_type: str = "structured"
    view_mode: str = "follow"
```

配置分层：

- `TrackConfig`
- `VehicleConfig`
- `RewardConfig`
- `ObservationConfig`
- `RenderConfig`
- `EnvConfig`

每个 episode 的 `info` 记录使用的配置摘要，方便后续实验复现。

## 17. Quality Bar

每个关键节点提交前至少满足：

- `python -m pytest` 通过。
- `python -m rl_racing.play --seed 0` 能启动。
- `SDL_VIDEODRIVER=dummy python -m pytest` 能跑。
- 文档中命令和实际入口一致。
- 若改动环境语义，同步更新本设计文档。
