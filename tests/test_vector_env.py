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
        assert current_obs.shape == (2, 66)

