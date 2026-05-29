import numpy as np

from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv


def test_reset_returns_structured_observation_shape():
    env = RacingEnv()
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space_spec.shape
    assert obs.dtype == np.float32
    assert info["progress"] == 0.0


def test_step_moves_environment_forward():
    env = RacingEnv()
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(1)

    assert obs.shape == env.observation_space_spec.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["steps"] == 1
    assert info["reward"] == info["reward_total"]
    assert np.isclose(
        info["reward_total"],
        info["reward_progress"]
        + info["reward_time"]
        + info["reward_heading"]
        + info["reward_success"]
        + info["reward_failure"],
    )


def test_advance_skips_observation(monkeypatch):
    env = RacingEnv()
    env.reset(seed=0)

    def fail_observe():
        raise AssertionError("advance should not build observations")

    monkeypatch.setattr(env, "observe", fail_observe)
    reward, terminated, truncated, info = env.advance(1)

    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["steps"] == 1


def test_off_track_terminates():
    env = RacingEnv()
    env.reset(seed=0)
    assert env.vehicle is not None
    env.vehicle.position = np.array([-10_000.0, -10_000.0], dtype=np.float64)

    _, _, terminated, _, info = env.step(0)

    assert terminated
    assert info["off_track"]
    assert info["done_reason"] == "off_track"
    assert info["reward_failure"] < 0.0


def test_collision_terminates_when_obstacle_exists():
    env = RacingEnv()
    env.reset(seed=0)
    assert env.track is not None and env.vehicle is not None
    if not env.track.obstacles:
        return
    env.vehicle.position = env.track.obstacles[0].center.copy()

    _, _, terminated, _, info = env.step(0)

    assert terminated
    assert info["collision"]


def test_success_near_finish():
    env = RacingEnv()
    env.reset(seed=0)
    assert env.track is not None and env.vehicle is not None
    point, heading = env.track.sample_at(env.track.length * 0.997)
    env.vehicle.position = point
    env.vehicle.heading = heading

    _, _, terminated, _, info = env.step(0)

    assert terminated
    assert info["success"]
    assert info["done_reason"] == "success"
    assert info["reward_success"] > 0.0


def test_image_observation_with_dummy_video_driver(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    cfg = EnvConfig(observation=ObservationConfig(obs_type="image", image_size=64))
    env = RacingEnv(cfg)

    obs, _ = env.reset(seed=0)

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert obs.max() > obs.min()
