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
