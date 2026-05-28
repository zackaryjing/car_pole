"""RL-style racing environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_racing.actions import action_to_control
from rl_racing.config import EnvConfig
from rl_racing.geometry import normalize_angle
from rl_racing.observations import structured_observation, structured_observation_size
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
        self.done_reason = ""

    @property
    def action_space_spec(self) -> SpaceSpec:
        return SpaceSpec(shape=(), dtype=int, n=9)

    @property
    def observation_space_spec(self) -> SpaceSpec:
        if self.config.observation.obs_type == "structured":
            return SpaceSpec(shape=(structured_observation_size(self.config),), dtype=np.float32)
        size = self.config.observation.image_size
        return SpaceSpec(shape=(size, size, 3), dtype=np.uint8)

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
        self.done_reason = ""
        return self.observe(), self._info(False, False, False)

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:
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
        reward = progress_delta * self.config.reward.progress_scale + self.config.reward.time_penalty
        reward += np.cos(heading_error) * self.config.reward.heading_alignment_scale
        if success:
            reward += self.config.reward.success_bonus
            self.done_reason = "success"
        elif off_track:
            reward += self.config.reward.failure_penalty
            self.done_reason = "off_track"
        elif collision:
            reward += self.config.reward.failure_penalty
            self.done_reason = "collision"
        elif truncated:
            self.done_reason = "max_steps"
        else:
            self.done_reason = ""

        self.last_reward = float(reward)
        return self.observe(), float(reward), terminated, truncated, self._info(off_track, collision, success)

    def observe(self) -> NDArray:
        self._require_state()
        assert self.track is not None and self.vehicle is not None
        if self.config.observation.obs_type == "structured":
            return structured_observation(self.track, self.vehicle, self.config)
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
        }

    def _require_state(self) -> None:
        if self.track is None or self.vehicle is None:
            raise RuntimeError("call reset() before using the environment")

