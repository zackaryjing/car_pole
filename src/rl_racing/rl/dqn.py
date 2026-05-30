"""Minimal DQN training loop for sensor observations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import strftime
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv
from rl_racing.episode import maybe_update_best_record, run_episode
from rl_racing.policies import Policy
from rl_racing.rl.networks import MLPQNetwork
from rl_racing.rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 200_000
    seed: int = 0
    env_max_steps: int = 2_000
    replay_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 1e-4
    train_frequency: int = 1
    target_update_interval: int = 1_000
    hidden_dim: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000
    grad_clip_norm: float = 10.0
    eval_interval: int = 10_000
    eval_episodes: int = 5
    checkpoint_interval: int = 50_000
    device: str = "auto"
    run_name: str | None = None
    progress: bool = True


@dataclass(frozen=True)
class DQNTrainResult:
    run_dir: str
    final_step: int
    best_successes: int


class DQNPolicy:
    def __init__(self, network: MLPQNetwork, device: torch.device, action_count: int, epsilon: float = 0.0):
        self.name = "dqn"
        self.network = network
        self.device = device
        self.action_count = action_count
        self.epsilon = epsilon
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None, info: dict) -> None:
        del info
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> int:
        del info
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_count))
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.network(tensor)
            return int(torch.argmax(q_values, dim=1).item())


def train_dqn(config: DQNConfig, output_root: str | Path = "runs/dqn_sensor") -> DQNTrainResult:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    run_dir = _make_run_dir(output_root, config)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "best_records").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")

    env_cfg = EnvConfig(max_steps=config.env_max_steps, observation=ObservationConfig(obs_type="sensor"))
    env = RacingEnv(env_cfg)
    obs, info = env.reset(seed=config.seed)
    obs_shape = tuple(obs.shape)
    action_count = env.action_space_spec.n
    assert action_count is not None

    replay = ReplayBuffer(config.replay_size, obs_shape, seed=config.seed)
    online = MLPQNetwork(int(np.prod(obs_shape)), action_count, config.hidden_dim).to(device)
    target = MLPQNetwork(int(np.prod(obs_shape)), action_count, config.hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "episode", "epsilon", "episode_reward", "episode_steps", "success", "done_reason", "loss"],
        )
        writer.writeheader()

    episode = 0
    episode_reward = 0.0
    episode_steps = 0
    last_loss: float | None = None
    best_successes = 0

    progress_bar = tqdm(
        range(1, config.total_steps + 1),
        total=config.total_steps,
        desc="DQN",
        unit="step",
        dynamic_ncols=True,
        disable=not config.progress,
    )
    for step in progress_bar:
        epsilon = _epsilon_at_step(config, step)
        action = _epsilon_greedy_action(online, obs, action_count, epsilon, device)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)

        episode_reward += reward
        episode_steps += 1
        obs = next_obs
        info = next_info

        if step >= config.learning_starts and len(replay) >= config.batch_size and step % config.train_frequency == 0:
            last_loss = _optimize_step(online, target, optimizer, replay, config, device)

        if step % config.target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        if done:
            _append_metrics(
                metrics_path,
                {
                    "step": step,
                    "episode": episode,
                    "epsilon": epsilon,
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                    "success": bool(info["success"]),
                    "done_reason": info["done_reason"],
                    "loss": "" if last_loss is None else last_loss,
                },
            )
            episode += 1
            obs, info = env.reset(seed=config.seed + episode)
            episode_reward = 0.0
            episode_steps = 0

        if step % config.eval_interval == 0:
            successes = evaluate_and_record(online, config, run_dir, step, action_count, device)
            best_successes = max(best_successes, successes)
            if config.progress:
                progress_bar.write(f"eval step={step} successes={successes}/{config.eval_episodes}")

        if step % config.checkpoint_interval == 0:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", online, target, optimizer, config, step)

        if config.progress and (step == 1 or step % 100 == 0 or done):
            progress_bar.set_postfix(
                {
                    "episode": episode,
                    "epsilon": f"{epsilon:.3f}",
                    "loss": "" if last_loss is None else f"{last_loss:.4f}",
                    "best_eval": best_successes,
                }
            )

    save_checkpoint(run_dir / "checkpoints" / "final.pt", online, target, optimizer, config, config.total_steps)
    return DQNTrainResult(run_dir=str(run_dir), final_step=config.total_steps, best_successes=best_successes)


def evaluate_and_record(
    network: MLPQNetwork,
    config: DQNConfig,
    run_dir: Path,
    step: int,
    action_count: int,
    device: torch.device,
) -> int:
    successes = 0
    policy = DQNPolicy(network, device, action_count, epsilon=0.0)
    for offset in range(config.eval_episodes):
        seed = config.seed + 10_000 + offset
        env = RacingEnv(EnvConfig(max_steps=config.env_max_steps, observation=ObservationConfig(obs_type="sensor")))
        trajectory_path = run_dir / "trajectories" / f"step_{step}_seed_{seed}"
        result = run_episode(env, policy, seed=seed, record=True, trajectory_path=trajectory_path)
        if result.success:
            successes += 1
        maybe_update_best_record(result, run_dir / "best_records")
    return successes


def save_checkpoint(
    path: str | Path,
    online: MLPQNetwork,
    target: MLPQNetwork,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    step: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_policy(checkpoint_path: str | Path, device: str = "auto") -> DQNPolicy:
    resolved_device = _select_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    cfg = DQNConfig(**checkpoint["config"])
    env = RacingEnv(EnvConfig(max_steps=cfg.env_max_steps, observation=ObservationConfig(obs_type="sensor")))
    action_count = env.action_space_spec.n
    assert action_count is not None
    obs_dim = int(np.prod(env.observation_space_spec.shape))
    network = MLPQNetwork(obs_dim, action_count, cfg.hidden_dim).to(resolved_device)
    network.load_state_dict(checkpoint["online"])
    network.eval()
    return DQNPolicy(network, resolved_device, action_count, epsilon=0.0)


def _optimize_step(
    online: MLPQNetwork,
    target: MLPQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: DQNConfig,
    device: torch.device,
) -> float:
    batch = replay.sample(config.batch_size)
    observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = online(observations).gather(1, actions)
    with torch.no_grad():
        next_actions = online(next_observations).argmax(dim=1, keepdim=True)
        next_q = target(next_observations).gather(1, next_actions)
        targets = rewards + (1.0 - dones) * config.gamma * next_q

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), config.grad_clip_norm)
    optimizer.step()
    return float(loss.item())


def _epsilon_greedy_action(
    network: MLPQNetwork,
    obs: np.ndarray,
    action_count: int,
    epsilon: float,
    device: torch.device,
) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(0, action_count))
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(torch.argmax(network(tensor), dim=1).item())


def _epsilon_at_step(config: DQNConfig, step: int) -> float:
    fraction = min(max(step / max(config.epsilon_decay_steps, 1), 0.0), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_run_dir(output_root: str | Path, config: DQNConfig) -> Path:
    name = config.run_name or strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / name


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "episode", "epsilon", "episode_reward", "episode_steps", "success", "done_reason", "loss"],
        )
        writer.writerow(row)
