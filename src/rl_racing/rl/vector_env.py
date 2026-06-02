"""Multiprocess vector environment for CPU-bound racing simulation."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv


class SubprocVectorEnv:
    """Synchronous subprocess vector env.

    Each worker owns one `RacingEnv`, so CPU-bound observation and geometry code
    can run in parallel without contending on the Python GIL.
    """

    def __init__(self, num_envs: int, env_config: EnvConfig, seed: int):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.closed = False
        ctx = mp.get_context("spawn")
        self.parents: list[Connection] = []
        self.processes: list[mp.Process] = []
        for index in range(self.num_envs):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child, env_config, seed + index))
            process.daemon = True
            process.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(process)

    def reset(self) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        for parent in self.parents:
            parent.send(("reset", None))
        results = [parent.recv() for parent in self.parents]
        observations, infos = zip(*results)
        return np.stack(observations).astype(np.float32), list(infos)

    def step(
        self, actions: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]], NDArray[np.float32]]:
        for parent, action in zip(self.parents, actions):
            parent.send(("step", int(action)))
        results = [parent.recv() for parent in self.parents]
        transition_obs, rewards, dones, infos, next_current_obs = zip(*results)
        return (
            np.stack(transition_obs).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
            np.stack(next_current_obs).astype(np.float32),
        )

    def close(self) -> None:
        if self.closed:
            return
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
        for parent in self.parents:
            parent.close()
        self.closed = True

    def __enter__(self) -> "SubprocVectorEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_sensor_env_config(max_steps: int) -> EnvConfig:
    return EnvConfig(max_steps=max_steps, observation=ObservationConfig(obs_type="sensor"))


def _worker(conn: Connection, env_config: EnvConfig, seed: int) -> None:
    env = RacingEnv(env_config)
    obs, info = env.reset(seed=seed)
    current_seed = seed
    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                obs, info = env.reset(seed=current_seed)
                conn.send((obs, info))
            elif command == "step":
                next_obs, reward, terminated, truncated, info = env.step(int(payload))
                done = bool(terminated or truncated)
                info = dict(info)
                info["terminated"] = bool(terminated)
                info["truncated"] = bool(truncated)
                transition_obs = next_obs
                if done:
                    current_seed += 10_000
                    reset_obs, _ = env.reset(seed=current_seed)
                    next_current_obs = reset_obs
                else:
                    next_current_obs = next_obs
                conn.send((transition_obs, float(reward), done, info, next_current_obs))
            elif command == "close":
                conn.close()
                return
            else:
                raise ValueError(f"unknown command: {command}")
    except EOFError:
        return
