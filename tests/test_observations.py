import numpy as np

from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv


def test_sensor_observation_shape_and_no_privileged_fields():
    cfg = EnvConfig(observation=ObservationConfig(obs_type="sensor", ray_count=21))
    env = RacingEnv(cfg)
    obs, info = env.reset(seed=0)

    assert obs.shape == (3 + 21 * 3,)
    assert obs.dtype == np.float32
    assert env.observation_space_spec.shape == obs.shape
    assert info["progress"] == 0.0
    assert obs.shape != (5 + 21 * 2 + cfg.observation.future_count * 2,)


def test_privileged_observation_keeps_old_structured_shape():
    cfg = EnvConfig(observation=ObservationConfig(obs_type="privileged", ray_count=21, future_count=10))
    env = RacingEnv(cfg)
    obs, _ = env.reset(seed=0)

    assert obs.shape == (5 + 21 * 2 + 10 * 2,)
    assert obs.dtype == np.float32


def test_finish_ray_detects_visible_finish_segment():
    cfg = EnvConfig(observation=ObservationConfig(obs_type="sensor", ray_count=21))
    env = RacingEnv(cfg)
    env.reset(seed=0)
    assert env.track is not None and env.vehicle is not None
    point, heading = env.track.sample_at(env.track.length - 80.0)
    env.vehicle.position = point
    env.vehicle.heading = heading

    obs = env.observe()
    finish_distances = obs[3 + cfg.observation.ray_count * 2 :]

    assert finish_distances.min() < 1.0
