import numpy as np
import pytest

from rl_racing.rl.replay_buffer import ReplayBuffer


def test_replay_buffer_adds_and_samples_batch():
    buffer = ReplayBuffer(capacity=4, observation_shape=(3,), seed=0)
    for idx in range(4):
        obs = np.full((3,), idx, dtype=np.float32)
        buffer.add(obs, action=idx, reward=float(idx), next_observation=obs + 1, done=idx == 3)

    batch = buffer.sample(2)

    assert len(buffer) == 4
    assert batch.observations.shape == (2, 3)
    assert batch.next_observations.shape == (2, 3)
    assert batch.actions.shape == (2,)
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.float32


def test_replay_buffer_rejects_oversized_sample():
    buffer = ReplayBuffer(capacity=4, observation_shape=(3,), seed=0)

    with pytest.raises(ValueError):
        buffer.sample(1)


def test_replay_buffer_add_batch():
    buffer = ReplayBuffer(capacity=8, observation_shape=(2,), seed=0)
    observations = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    next_observations = observations + 1.0

    buffer.add_batch(
        observations,
        np.array([1, 2]),
        np.array([0.5, -0.5], dtype=np.float32),
        next_observations,
        np.array([False, True]),
    )

    assert len(buffer) == 2
    batch = buffer.sample(2)
    assert batch.observations.shape == (2, 2)
