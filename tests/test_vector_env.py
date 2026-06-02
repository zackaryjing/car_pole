import numpy as np

from rl_racing.rl.vector_env import SubprocVectorEnv, make_sensor_env_config


def test_subproc_vector_env_reset_and_step():
    env_cfg = make_sensor_env_config(max_steps=20)
    with SubprocVectorEnv(num_envs=2, env_config=env_cfg, seed=0) as envs:
        obs, infos = envs.reset()

        assert obs.shape == (2, 66)
        assert len(infos) == 2

        next_obs, rewards, dones, step_infos, current_obs = envs.step(np.array([1, 1], dtype=np.int64))

        assert next_obs.shape == (2, 66)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(step_infos) == 2
        assert step_infos[0]["terminated"] is False
        assert step_infos[0]["truncated"] is False
        assert current_obs.shape == (2, 66)


def test_subproc_vector_env_exposes_time_limit_truncation():
    env_cfg = make_sensor_env_config(max_steps=1)
    with SubprocVectorEnv(num_envs=1, env_config=env_cfg, seed=0) as envs:
        envs.reset()
        _, _, dones, infos, current_obs = envs.step(np.array([0], dtype=np.int64))

        assert dones.tolist() == [True]
        assert infos[0]["terminated"] is False
        assert infos[0]["truncated"] is True
        assert infos[0]["done_reason"] == "max_steps"
        assert current_obs.shape == (1, 66)
