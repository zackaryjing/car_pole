"""Episode evaluation, trajectory persistence, and best-record helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_racing.env import RacingEnv
from rl_racing.policies import Policy
from rl_racing.track import CircleObstacle, Track
from rl_racing.vehicle import VehicleState


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    steps: int
    sim_time: float
    total_reward: float
    done_reason: str
    seed: int | None
    trajectory_path: str | None = None


@dataclass(frozen=True)
class RaceComparison:
    current: EpisodeResult
    best: EpisodeResult | None
    best_trajectory_path: str | None


@dataclass
class EpisodeTrajectory:
    metadata: dict[str, Any]
    actions: NDArray[np.int_]
    rewards: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    vehicle_states: NDArray[np.float64]
    progress: NDArray[np.float64]
    centerline: NDArray[np.float64]
    obstacles: NDArray[np.float64]

    def result(self, path: Path | None = None) -> EpisodeResult:
        final_info = self.metadata["final_info"]
        return EpisodeResult(
            success=bool(final_info["success"]),
            steps=int(final_info["steps"]),
            sim_time=float(final_info["steps"]) * float(self.metadata["dt"]),
            total_reward=float(np.sum(self.rewards)),
            done_reason=str(final_info["done_reason"]),
            seed=self.metadata.get("seed"),
            trajectory_path=str(path) if path is not None else None,
        )


def run_episode(
    env: RacingEnv,
    policy: Policy,
    seed: int | None = None,
    record: bool = True,
    trajectory_path: str | Path | None = None,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    policy.reset(seed, info)

    actions: list[int] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    infos: list[dict[str, Any]] = []
    vehicle_states: list[list[float]] = []
    progress: list[float] = []
    total_reward = 0.0

    if record:
        vehicle_states.append(_vehicle_row(env))
        progress.append(float(info["progress"]))

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(policy.act(obs, info))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if record:
            actions.append(action)
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            infos.append(_json_safe_info(info))
            vehicle_states.append(_vehicle_row(env))
            progress.append(float(info["progress"]))

    result = EpisodeResult(
        success=bool(info["success"]),
        steps=int(info["steps"]),
        sim_time=float(info["steps"]) * env.config.dt,
        total_reward=float(total_reward),
        done_reason=str(info["done_reason"]),
        seed=info.get("seed"),
        trajectory_path=str(trajectory_path) if trajectory_path is not None and record else None,
    )
    if record and trajectory_path is not None:
        assert env.track is not None
        trajectory = EpisodeTrajectory(
            metadata={
                "version": 1,
                "created_at": time(),
                "policy_name": policy.name,
                "seed": info.get("seed"),
                "dt": env.config.dt,
                "obs_type": env.config.observation.obs_type,
                "config": asdict(env.config),
                "final_info": _json_safe_info(info),
                "infos": infos,
            },
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float64),
            terminated=np.asarray(terminated_flags, dtype=np.bool_),
            truncated=np.asarray(truncated_flags, dtype=np.bool_),
            vehicle_states=np.asarray(vehicle_states, dtype=np.float64),
            progress=np.asarray(progress, dtype=np.float64),
            centerline=env.track.centerline.copy(),
            obstacles=_obstacle_array(env.track),
        )
        save_trajectory(trajectory, trajectory_path)
    return result


def compare_policy_to_best(
    env: RacingEnv,
    policy: Policy,
    records_dir: str | Path,
    seed: int,
    current_trajectory_path: str | Path | None = None,
) -> RaceComparison:
    current = run_episode(env, policy, seed=seed, record=current_trajectory_path is not None, trajectory_path=current_trajectory_path)
    best_path = best_record_path(records_dir, seed)
    if not best_path.exists():
        return RaceComparison(current=current, best=None, best_trajectory_path=None)

    record = json.loads(best_path.read_text(encoding="utf-8"))
    best = EpisodeResult(
        success=True,
        steps=int(record["steps"]),
        sim_time=float(record["sim_time"]),
        total_reward=float(record["total_reward"]),
        done_reason=str(record["done_reason"]),
        seed=record.get("seed"),
        trajectory_path=str(record["trajectory_path"]),
    )
    return RaceComparison(current=current, best=best, best_trajectory_path=str(record["trajectory_path"]))


def save_trajectory(trajectory: EpisodeTrajectory, path: str | Path) -> None:
    base = _base_path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base.with_suffix(".npz"),
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        vehicle_states=trajectory.vehicle_states,
        progress=trajectory.progress,
        centerline=trajectory.centerline,
        obstacles=trajectory.obstacles,
    )
    base.with_suffix(".json").write_text(json.dumps(trajectory.metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_trajectory(path: str | Path) -> EpisodeTrajectory:
    base = _base_path(path)
    metadata = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(base.with_suffix(".npz")) as data:
        return EpisodeTrajectory(
            metadata=metadata,
            actions=data["actions"].copy(),
            rewards=data["rewards"].copy(),
            terminated=data["terminated"].copy(),
            truncated=data["truncated"].copy(),
            vehicle_states=data["vehicle_states"].copy(),
            progress=data["progress"].copy(),
            centerline=data["centerline"].copy(),
            obstacles=data["obstacles"].copy(),
        )


def best_record_path(records_dir: str | Path, seed: int | None) -> Path:
    return Path(records_dir) / f"best_seed_{seed}.json"


def maybe_update_best_record(result: EpisodeResult, records_dir: str | Path) -> bool:
    if not result.success or result.trajectory_path is None:
        return False
    path = best_record_path(records_dir, result.seed)
    current: dict[str, Any] | None = None
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    if current is not None and int(current["steps"]) <= result.steps:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": result.seed,
                "steps": result.steps,
                "sim_time": result.sim_time,
                "total_reward": result.total_reward,
                "done_reason": result.done_reason,
                "trajectory_path": result.trajectory_path,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return True


def trajectory_track(trajectory: EpisodeTrajectory) -> Track:
    obstacles = [CircleObstacle(row[:2].astype(np.float64), float(row[2])) for row in trajectory.obstacles]
    centerline = trajectory.centerline.astype(np.float64)
    from rl_racing.geometry import Pose2D, angle_of, polyline_lengths

    cumulative = polyline_lengths(centerline)
    return Track(
        centerline=centerline,
        width=float(trajectory.metadata["config"]["track"]["width"]),
        obstacles=obstacles,
        start_pose=Pose2D(centerline[0].copy(), angle_of(centerline[1] - centerline[0])),
        finish_center=centerline[-1].copy(),
        finish_heading=angle_of(centerline[-1] - centerline[-2]),
        cumulative_lengths=cumulative,
    )


def trajectory_vehicle_state(trajectory: EpisodeTrajectory, index: int) -> VehicleState:
    row = trajectory.vehicle_states[index]
    return VehicleState(
        position=row[:2].astype(np.float64),
        heading=float(row[2]),
        speed=float(row[3]),
        steering=float(row[4]),
        angular_velocity=float(row[5]),
    )


def render_trajectory_frame(
    trajectory: EpisodeTrajectory,
    index: int,
    view: str = "follow",
    size: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    from rl_racing.config import EnvConfig, ObservationConfig
    from rl_racing.renderer import render_rgb_array

    cfg = EnvConfig(observation=ObservationConfig(obs_type="image"))
    track = trajectory_track(trajectory)
    vehicle = trajectory_vehicle_state(trajectory, index)
    return render_rgb_array(track, vehicle, cfg, view=view, size=size)


def render_race_frame(
    current: EpisodeTrajectory,
    best: EpisodeTrajectory,
    index: int,
    view: str = "global",
    size: tuple[int, int] = (1000, 720),
) -> NDArray[np.uint8]:
    import pygame

    from rl_racing.config import EnvConfig
    from rl_racing.renderer import draw_world

    current_index = min(max(index, 0), len(current.vehicle_states) - 1)
    best_index = min(max(index, 0), len(best.vehicle_states) - 1)
    surface = pygame.Surface(size)
    draw_world(
        surface,
        trajectory_track(current),
        trajectory_vehicle_state(current, current_index),
        EnvConfig(),
        view=view,
        ghost_vehicles=[trajectory_vehicle_state(best, best_index)],
    )
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8)


def _vehicle_row(env: RacingEnv) -> list[float]:
    assert env.vehicle is not None
    return [
        float(env.vehicle.position[0]),
        float(env.vehicle.position[1]),
        float(env.vehicle.heading),
        float(env.vehicle.speed),
        float(env.vehicle.steering),
        float(env.vehicle.angular_velocity),
    ]


def _obstacle_array(track: Track) -> NDArray[np.float64]:
    if not track.obstacles:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[obs.center[0], obs.center[1], obs.radius] for obs in track.obstacles], dtype=np.float64)


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.generic):
            value = value.item()
        safe[key] = value
    return safe


def _base_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix in (".json", ".npz"):
        return path.with_suffix("")
    return path
