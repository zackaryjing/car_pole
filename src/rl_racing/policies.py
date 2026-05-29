"""Small policy interfaces and baselines for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rl_racing.env import RacingEnv
from rl_racing.geometry import normalize_angle


class Policy(Protocol):
    name: str

    def reset(self, seed: int | None, info: dict) -> None:
        ...

    def act(self, obs: NDArray, info: dict) -> int:
        ...


@dataclass
class RandomPolicy:
    action_count: int = 9
    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None, info: dict) -> None:
        del info
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: NDArray, info: dict) -> int:
        del obs, info
        return int(self.rng.integers(0, self.action_count))


@dataclass
class ReplayPolicy:
    actions: NDArray[np.int_]
    name: str = "replay"

    def reset(self, seed: int | None, info: dict) -> None:
        del seed, info
        self.index = 0

    def act(self, obs: NDArray, info: dict) -> int:
        del obs, info
        if self.index >= len(self.actions):
            return 0
        action = int(self.actions[self.index])
        self.index += 1
        return action


@dataclass
class CenterlineHeuristicPolicy:
    """Privileged sanity-check policy, not a training baseline."""

    env: RacingEnv
    name: str = "centerline_heuristic"

    def reset(self, seed: int | None, info: dict) -> None:
        del seed, info

    def act(self, obs: NDArray, info: dict) -> int:
        del obs, info
        assert self.env.track is not None and self.env.vehicle is not None
        query = self.env.track.query(self.env.vehicle.position)
        heading_error = normalize_angle(float(query["tangent_heading"]) - self.env.vehicle.heading)
        if heading_error < -0.12:
            return 5
        if heading_error > 0.12:
            return 6
        return 1
