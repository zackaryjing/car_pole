import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_racing.rl.ppo import PPOConfig, compute_gae, load_policy, train_ppo


def test_compute_gae_respects_episode_boundaries():
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [0.25]], dtype=np.float32)
    dones = np.array([[False], [True]])

    advantages, returns = compute_gae(rewards, values, dones, next_value=np.array([9.0]), gamma=1.0, gae_lambda=1.0)

    np.testing.assert_allclose(advantages[:, 0], np.array([2.5, 1.75]))
    np.testing.assert_allclose(returns[:, 0], np.array([3.0, 2.0]))


def test_ppo_vector_training_smoke_cpu(tmp_path):
    config = PPOConfig(
        total_steps=8,
        seed=0,
        env_max_steps=4,
        num_envs=2,
        rollout_steps=4,
        batch_size=4,
        update_epochs=1,
        hidden_dim=32,
        eval_interval=100,
        checkpoint_interval=100,
        device="cpu",
        run_name="smoke",
        progress=False,
    )

    result = train_ppo(config, output_root=tmp_path)

    assert result.final_step == 8
    assert (tmp_path / "smoke" / "config.json").exists()
    assert (tmp_path / "smoke" / "metrics.csv").exists()
    assert (tmp_path / "smoke" / "updates.csv").exists()
    assert (tmp_path / "smoke" / "eval_metrics.csv").exists()
    assert (tmp_path / "smoke" / "checkpoints" / "final.pt").exists()


def test_ppo_checkpoint_loads_policy(tmp_path):
    config = PPOConfig(
        total_steps=4,
        seed=0,
        env_max_steps=2,
        num_envs=2,
        rollout_steps=2,
        batch_size=2,
        update_epochs=1,
        hidden_dim=32,
        eval_interval=100,
        checkpoint_interval=100,
        device="cpu",
        run_name="load",
        progress=False,
    )
    train_ppo(config, output_root=tmp_path)

    policy = load_policy(tmp_path / "load" / "checkpoints" / "final.pt", device="cpu")

    assert policy.name == "ppo"


def test_ppo_eval_writes_metrics_and_best_checkpoint(tmp_path):
    config = PPOConfig(
        total_steps=2,
        seed=0,
        env_max_steps=1,
        num_envs=1,
        rollout_steps=2,
        batch_size=2,
        update_epochs=1,
        hidden_dim=16,
        eval_interval=2,
        eval_episodes=1,
        checkpoint_interval=100,
        device="cpu",
        run_name="eval",
        progress=False,
    )
    train_ppo(config, output_root=tmp_path)

    eval_metrics = (tmp_path / "eval" / "eval_metrics.csv").read_text(encoding="utf-8")

    assert "step,successes,episodes,success_rate" in eval_metrics
    assert (tmp_path / "eval" / "checkpoints" / "best_eval.pt").exists()
