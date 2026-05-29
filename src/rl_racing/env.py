"""RL-style racing environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_racing.actions import action_to_control
from rl_racing.config import EnvConfig
from rl_racing.geometry import normalize_angle
from rl_racing.observations import (
    sensor_observation,
    sensor_observation_size,
    structured_observation,
    structured_observation_size,
)
from rl_racing.track import Track, generate_track
from rl_racing.vehicle import VehicleState, step_vehicle


@dataclass(frozen=True)
class SpaceSpec:
    shape: tuple[int, ...]
    dtype: type
    n: int | None = None


class RacingEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng()
        self.seed: int | None = None
        self.track: Track | None = None
        self.vehicle: VehicleState | None = None
        self.steps = 0
        self.prev_progress = 0.0
        self.last_reward = 0.0
        self.last_reward_breakdown = _empty_reward_breakdown()
        self.done_reason = ""

    @property
    def action_space_spec(self) -> SpaceSpec:
        return SpaceSpec(shape=(), dtype=int, n=9)

    @property
    def observation_space_spec(self) -> SpaceSpec:
        if self.config.observation.obs_type == "sensor":
            return SpaceSpec(shape=(sensor_observation_size(self.config),), dtype=np.float32)
        if self.config.observation.obs_type in ("privileged", "structured"):
            return SpaceSpec(shape=(structured_observation_size(self.config),), dtype=np.float32)
        if self.config.observation.obs_type == "image":
            size = self.config.observation.image_size
            return SpaceSpec(shape=(size, size, 3), dtype=np.uint8)
        raise ValueError(f"unsupported obs_type: {self.config.observation.obs_type}")

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[NDArray, dict]:
        del options
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        elif self.seed is None:
            self.seed = int(self.rng.integers(0, 2**31 - 1))
            self.rng = np.random.default_rng(self.seed)

        self.track = generate_track(self.rng, self.config.track, self.config.vehicle)
        pose = self.track.start_pose
        self.vehicle = VehicleState(position=pose.position.copy(), heading=pose.heading)
        self.steps = 0
        self.prev_progress = 0.0
        self.last_reward = 0.0
        self.last_reward_breakdown = _empty_reward_breakdown()
        self.done_reason = ""
        return self.observe(), self._info(False, False, False)

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._step(action, observe=True)
        assert obs is not None
        return obs, reward, terminated, truncated, info

    def advance(self, action: int) -> tuple[float, bool, bool, dict]:
        """Advance simulation without building an observation.

        Manual pygame play uses this path so UI responsiveness is not coupled to
        training observation cost.
        """

        _, reward, terminated, truncated, info = self._step(action, observe=False)
        return reward, terminated, truncated, info

    def _step(self, action: int, observe: bool) -> tuple[NDArray | None, float, bool, bool, dict]:
        self._require_state()
        assert self.track is not None and self.vehicle is not None

        control = action_to_control(action)
        self.vehicle = step_vehicle(self.vehicle, control, self.config.vehicle, self.config.dt)
        self.steps += 1

        query = self.track.query(self.vehicle.position)
        progress = float(query["progress"])
        progress_delta = progress - self.prev_progress
        if progress_delta < -0.25:
            progress_delta = 0.0
        self.prev_progress = max(self.prev_progress, progress)

        off_track = not self.track.is_on_track(self.vehicle.position, self.config.vehicle.radius)
        collision = self.track.collides_obstacle(self.vehicle.position, self.config.vehicle.radius)
        success = progress >= 0.995
        truncated = self.steps >= self.config.max_steps
        terminated = off_track or collision or success

        heading_error = normalize_angle(self.vehicle.heading - float(query["tangent_heading"]))
        reward_progress = progress_delta * self.config.reward.progress_scale
        reward_time = self.config.reward.time_penalty
        reward_heading = float(np.cos(heading_error) * self.config.reward.heading_alignment_scale)
        reward_success = 0.0
        reward_failure = 0.0
        if success:
            reward_success = self.config.reward.success_bonus
            self.done_reason = "success"
        elif off_track:
            reward_failure = self.config.reward.failure_penalty
            self.done_reason = "off_track"
        elif collision:
            reward_failure = self.config.reward.failure_penalty
            self.done_reason = "collision"
        elif truncated:
            self.done_reason = "max_steps"
        else:
            self.done_reason = ""

        reward = reward_progress + reward_time + reward_heading + reward_success + reward_failure
        self.last_reward = float(reward)
        self.last_reward_breakdown = {
            "reward_progress": float(reward_progress),
            "reward_time": float(reward_time),
            "reward_heading": float(reward_heading),
            "reward_success": float(reward_success),
            "reward_failure": float(reward_failure),
            "reward_total": float(reward),
        }
        obs = self.observe() if observe else None
        return obs, float(reward), terminated, truncated, self._info(off_track, collision, success)

    def observe(self) -> NDArray:
        self._require_state()
        assert self.track is not None and self.vehicle is not None
        if self.config.observation.obs_type == "sensor":
            return sensor_observation(self.track, self.vehicle, self.config)
        if self.config.observation.obs_type in ("privileged", "structured"):
            return structured_observation(self.track, self.vehicle, self.config)
        if self.config.observation.obs_type != "image":
            raise ValueError(f"unsupported obs_type: {self.config.observation.obs_type}")
        from rl_racing.renderer import render_rgb_array

        return render_rgb_array(self.track, self.vehicle, self.config)

    def render(self, mode: str = "rgb_array", view: str = "follow") -> NDArray:
        self._require_state()
        assert self.track is not None and self.vehicle is not None
        if mode != "rgb_array":
            raise ValueError(f"unsupported render mode: {mode}")
        from rl_racing.renderer import render_rgb_array

        return render_rgb_array(self.track, self.vehicle, self.config, view=view)

    def close(self) -> None:
        pass

    def _info(self, off_track: bool, collision: bool, success: bool) -> dict:
        assert self.track is not None and self.vehicle is not None
        query = self.track.query(self.vehicle.position)
        return {
            "progress": float(query["progress"]),
            "lap_distance": float(query["arc_length"]),
            "off_track": bool(off_track),
            "collision": bool(collision),
            "success": bool(success),
            "steps": self.steps,
            "seed": self.seed,
            "done_reason": self.done_reason,
            "reward": self.last_reward,
            **self.last_reward_breakdown,
        }

    def _require_state(self) -> None:
        if self.track is None or self.vehicle is None:
            raise RuntimeError("call reset() before using the environment")


def _empty_reward_breakdown() -> dict[str, float]:
    return {
        "reward_progress": 0.0,
        "reward_time": 0.0,
        "reward_heading": 0.0,
        "reward_success": 0.0,
        "reward_failure": 0.0,
        "reward_total": 0.0,
    }
