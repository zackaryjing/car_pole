import pytest

torch = pytest.importorskip("torch")

from rl_racing.rl.dqn import DQNConfig, load_policy, train_dqn, train_dqn_v2


def test_dqn_training_smoke_cpu(tmp_path):
    config = DQNConfig(
        total_steps=12,
        seed=0,
        env_max_steps=8,
        replay_size=64,
        learning_starts=4,
        batch_size=4,
        train_frequency=1,
        target_update_interval=5,
        eval_interval=100,
        checkpoint_interval=100,
        hidden_dim=32,
        device="cpu",
        run_name="smoke",
        progress=False,
    )

    result = train_dqn(config, output_root=tmp_path)

    assert result.final_step == 12
    assert (tmp_path / "smoke" / "config.json").exists()
    assert (tmp_path / "smoke" / "metrics.csv").exists()
    assert (tmp_path / "smoke" / "checkpoints" / "final.pt").exists()


def test_dqn_checkpoint_loads_policy(tmp_path):
    config = DQNConfig(
        total_steps=4,
        seed=0,
        env_max_steps=4,
        replay_size=16,
        learning_starts=100,
        batch_size=4,
        eval_interval=100,
        checkpoint_interval=100,
        hidden_dim=32,
        device="cpu",
        run_name="load",
        progress=False,
    )
    train_dqn(config, output_root=tmp_path)

    policy = load_policy(tmp_path / "load" / "checkpoints" / "final.pt", device="cpu")

    assert policy.name == "dqn"


def test_dqn_v2_vector_training_smoke_cpu(tmp_path):
    config = DQNConfig(
        total_steps=12,
        seed=0,
        env_max_steps=8,
        replay_size=64,
        learning_starts=4,
        batch_size=4,
        train_frequency=1,
        gradient_steps=1,
        target_update_interval=6,
        eval_interval=100,
        checkpoint_interval=100,
        hidden_dim=32,
        device="cpu",
        run_name="v2",
        progress=False,
        num_envs=2,
    )

    result = train_dqn_v2(config, output_root=tmp_path)

    assert result.final_step >= 12
    assert (tmp_path / "v2" / "config.json").exists()
    assert (tmp_path / "v2" / "metrics.csv").exists()
    assert (tmp_path / "v2" / "checkpoints" / "final.pt").exists()
