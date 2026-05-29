"""Numpy replay buffer for off-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayBatch:
    observations: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.float32]
    dones: NDArray[np.float32]


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: NDArray,
        action: int,
        reward: float,
        next_observation: NDArray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError("not enough transitions in replay buffer")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
