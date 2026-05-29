import json

from rl_racing.episode import (
    EpisodeResult,
    compare_policy_to_best,
    load_trajectory,
    maybe_update_best_record,
    render_race_frame,
    render_trajectory_frame,
    run_episode,
)
from rl_racing.config import EnvConfig
from rl_racing.env import RacingEnv
from rl_racing.policies import RandomPolicy, ReplayPolicy


def _small_env() -> RacingEnv:
    return RacingEnv(EnvConfig(max_steps=40))


def test_run_episode_saves_loadable_trajectory(tmp_path):
    env = _small_env()
    path = tmp_path / "episode"
    result = run_episode(env, ReplayPolicy([0] * env.config.max_steps), seed=0, record=True, trajectory_path=path)

    trajectory = load_trajectory(path)

    assert result.trajectory_path == str(path)
    assert trajectory.actions.shape == (result.steps,)
    assert trajectory.vehicle_states.shape[0] == result.steps + 1
    assert trajectory.metadata["final_info"]["steps"] == result.steps


def test_replay_policy_reproduces_recorded_final_state(tmp_path):
    env = _small_env()
    path = tmp_path / "episode"
    result = run_episode(env, ReplayPolicy([1] * env.config.max_steps), seed=0, record=True, trajectory_path=path)
    trajectory = load_trajectory(path)

    replay_env = _small_env()
    replay_result = run_episode(replay_env, ReplayPolicy(trajectory.actions), seed=0, record=False)

    assert replay_result.steps == result.steps
    assert replay_result.done_reason == result.done_reason
    assert replay_result.success == result.success


def test_best_record_only_accepts_success_and_fewer_steps(tmp_path):
    failed = EpisodeResult(False, 10, 1.0, -1.0, "collision", seed=0, trajectory_path=str(tmp_path / "failed"))
    assert maybe_update_best_record(failed, tmp_path) is False

    slow = EpisodeResult(True, 100, 3.3, 1.0, "success", seed=0, trajectory_path=str(tmp_path / "slow"))
    fast = EpisodeResult(True, 90, 3.0, 2.0, "success", seed=0, trajectory_path=str(tmp_path / "fast"))
    slower = EpisodeResult(True, 110, 3.7, 3.0, "success", seed=0, trajectory_path=str(tmp_path / "slower"))

    assert maybe_update_best_record(slow, tmp_path) is True
    assert maybe_update_best_record(slower, tmp_path) is False
    assert maybe_update_best_record(fast, tmp_path) is True

    record = json.loads((tmp_path / "best_seed_0.json").read_text(encoding="utf-8"))
    assert record["steps"] == 90
    assert record["trajectory_path"] == str(tmp_path / "fast")


def test_random_policy_runner_finishes_episode():
    env = _small_env()
    result = run_episode(env, RandomPolicy(seed=0), seed=0, record=False)

    assert result.steps <= env.config.max_steps
    assert result.done_reason


def test_compare_policy_to_best_returns_current_and_best(tmp_path):
    best = EpisodeResult(True, 30, 1.0, 10.0, "success", seed=0, trajectory_path=str(tmp_path / "best"))
    assert maybe_update_best_record(best, tmp_path)

    comparison = compare_policy_to_best(_small_env(), ReplayPolicy([0] * 40), records_dir=tmp_path, seed=0)

    assert comparison.current.steps <= 40
    assert comparison.best is not None
    assert comparison.best.steps == 30
    assert comparison.best_trajectory_path == str(tmp_path / "best")


def test_trajectory_can_render_headless_frame(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = _small_env()
    path = tmp_path / "episode"
    run_episode(env, ReplayPolicy([0] * env.config.max_steps), seed=0, record=True, trajectory_path=path)
    trajectory = load_trajectory(path)

    frame = render_trajectory_frame(trajectory, 0, view="follow", size=(64, 64))

    assert frame.shape == (64, 64, 3)
    assert frame.max() > frame.min()


def test_race_frame_renders_current_and_best_headlessly(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    current_path = tmp_path / "current"
    best_path = tmp_path / "best"
    run_episode(_small_env(), ReplayPolicy([0] * 40), seed=0, record=True, trajectory_path=current_path)
    run_episode(_small_env(), ReplayPolicy([1] * 40), seed=0, record=True, trajectory_path=best_path)

    frame = render_race_frame(load_trajectory(current_path), load_trajectory(best_path), index=5, size=(96, 72))

    assert frame.shape == (72, 96, 3)
    assert frame.max() > frame.min()
