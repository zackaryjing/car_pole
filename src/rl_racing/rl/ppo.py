"""Sensor-observation PPO training with subprocess rollout workers."""

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
from torch.distributions import Categorical
from tqdm.auto import tqdm

from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv
from rl_racing.episode import maybe_update_best_record, run_episode
from rl_racing.rl.networks import MLPActorCritic
from rl_racing.rl.vector_env import SubprocVectorEnv, make_sensor_env_config


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 1_000_000
    seed: int = 0
    env_max_steps: int = 2_000
    num_envs: int = 64
    rollout_steps: int = 256
    batch_size: int = 4_096
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    eval_interval: int = 50_000
    eval_episodes: int = 20
    checkpoint_interval: int = 100_000
    device: str = "auto"
    run_name: str | None = None
    progress: bool = True


@dataclass(frozen=True)
class PPOTrainResult:
    run_dir: str
    final_step: int
    best_successes: int


class PPOPolicy:
    def __init__(self, network: MLPActorCritic, device: torch.device):
        self.name = "ppo"
        self.network = network
        self.device = device

    def reset(self, seed: int | None, info: dict) -> None:
        del seed, info

    def act(self, obs: np.ndarray, info: dict) -> int:
        del info
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.network(tensor)
            return int(torch.argmax(logits, dim=1).item())


def train_ppo(config: PPOConfig, output_root: str | Path = "runs/ppo_sensor") -> PPOTrainResult:
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    rng = np.random.default_rng(config.seed)
    run_dir = _make_run_dir(output_root, config)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "best_records").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")

    env_cfg = make_sensor_env_config(config.env_max_steps)
    probe_env = RacingEnv(env_cfg)
    probe_obs, _ = probe_env.reset(seed=config.seed)
    obs_shape = tuple(probe_obs.shape)
    obs_dim = int(np.prod(obs_shape))
    action_count = probe_env.action_space_spec.n
    assert action_count is not None

    network = MLPActorCritic(obs_dim, action_count, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=1e-5)
    _write_csv_header(
        run_dir / "metrics.csv",
        ["step", "episode", "episode_reward", "episode_steps", "success", "done_reason"],
    )
    _write_csv_header(
        run_dir / "updates.csv",
        ["step", "update", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"],
    )
    _write_csv_header(
        run_dir / "eval_metrics.csv",
        ["step", "successes", "episodes", "success_rate", "mean_reward", "mean_progress", "mean_steps"],
    )

    global_step = 0
    update = 0
    episode = 0
    best_successes = -1
    episode_rewards = np.zeros(config.num_envs, dtype=np.float64)
    episode_steps = np.zeros(config.num_envs, dtype=np.int64)
    progress_bar = tqdm(
        total=config.total_steps,
        desc=f"PPO[{config.num_envs} envs]",
        unit="step",
        dynamic_ncols=True,
        disable=not config.progress,
    )

    with SubprocVectorEnv(config.num_envs, env_cfg, config.seed) as vec_env:
        current_obs, _ = vec_env.reset()
        while global_step < config.total_steps:
            rollout = _collect_rollout(
                vec_env,
                network,
                current_obs,
                episode_rewards,
                episode_steps,
                config,
                device,
                run_dir / "metrics.csv",
                global_step,
                episode,
            )
            current_obs = rollout["current_obs"]
            previous_step = global_step
            global_step += int(rollout["collected_steps"])
            episode = int(rollout["episode"])
            update += 1

            update_metrics = _optimize(network, optimizer, rollout, config, device, rng)
            _append_csv(
                run_dir / "updates.csv",
                {
                    "step": global_step,
                    "update": update,
                    **update_metrics,
                },
            )

            if global_step // config.eval_interval != previous_step // config.eval_interval:
                eval_step = (global_step // config.eval_interval) * config.eval_interval
                successes = evaluate_and_record(network, config, run_dir, eval_step, device)
                if successes > best_successes:
                    best_successes = successes
                    save_checkpoint(run_dir / "checkpoints" / "best_eval.pt", network, optimizer, config, global_step)
                if config.progress:
                    progress_bar.write(f"eval step={eval_step} successes={successes}/{config.eval_episodes}")

            if global_step // config.checkpoint_interval != previous_step // config.checkpoint_interval:
                checkpoint_step = (global_step // config.checkpoint_interval) * config.checkpoint_interval
                save_checkpoint(run_dir / "checkpoints" / f"step_{checkpoint_step}.pt", network, optimizer, config, global_step)

            if config.progress:
                progress_bar.update(min(global_step, config.total_steps) - min(previous_step, config.total_steps))
                progress_bar.set_postfix(
                    {
                        "update": update,
                        "episodes": episode,
                        "policy": f"{update_metrics['policy_loss']:.4f}",
                        "value": f"{update_metrics['value_loss']:.4f}",
                        "best_eval": max(best_successes, 0),
                    }
                )

    progress_bar.close()
    save_checkpoint(run_dir / "checkpoints" / "final.pt", network, optimizer, config, global_step)
    return PPOTrainResult(run_dir=str(run_dir), final_step=global_step, best_successes=max(best_successes, 0))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = next_value
        else:
            next_values = values[step + 1]
        next_non_terminal = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def evaluate_and_record(
    network: MLPActorCritic,
    config: PPOConfig,
    run_dir: Path,
    step: int,
    device: torch.device,
) -> int:
    policy = PPOPolicy(network, device)
    results = []
    for offset in range(config.eval_episodes):
        seed = config.seed + 10_000 + offset
        env = RacingEnv(EnvConfig(max_steps=config.env_max_steps, observation=ObservationConfig(obs_type="sensor")))
        trajectory_path = run_dir / "trajectories" / f"step_{step}_seed_{seed}"
        result = run_episode(env, policy, seed=seed, record=True, trajectory_path=trajectory_path)
        maybe_update_best_record(result, run_dir / "best_records")
        results.append(result)

    successes = sum(result.success for result in results)
    progresses = [_trajectory_final_progress(run_dir / "trajectories" / f"step_{step}_seed_{result.seed}") for result in results]
    _append_csv(
        run_dir / "eval_metrics.csv",
        {
            "step": step,
            "successes": successes,
            "episodes": len(results),
            "success_rate": successes / len(results),
            "mean_reward": float(np.mean([result.total_reward for result in results])),
            "mean_progress": float(np.mean(progresses)),
            "mean_steps": float(np.mean([result.steps for result in results])),
        },
    )
    return successes


def save_checkpoint(
    path: str | Path,
    network: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    step: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "ppo",
            "step": step,
            "config": asdict(config),
            "actor_critic": network.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_policy(checkpoint_path: str | Path, device: str = "auto") -> PPOPolicy:
    resolved_device = _select_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = PPOConfig(**checkpoint["config"])
    env = RacingEnv(make_sensor_env_config(config.env_max_steps))
    action_count = env.action_space_spec.n
    assert action_count is not None
    obs_dim = int(np.prod(env.observation_space_spec.shape))
    network = MLPActorCritic(obs_dim, action_count, config.hidden_dim).to(resolved_device)
    network.load_state_dict(checkpoint["actor_critic"])
    network.eval()
    return PPOPolicy(network, resolved_device)


def _collect_rollout(
    vec_env: SubprocVectorEnv,
    network: MLPActorCritic,
    current_obs: np.ndarray,
    episode_rewards: np.ndarray,
    episode_steps: np.ndarray,
    config: PPOConfig,
    device: torch.device,
    metrics_path: Path,
    global_step: int,
    episode: int,
) -> dict[str, Any]:
    observations = np.zeros((config.rollout_steps, config.num_envs, current_obs.shape[1]), dtype=np.float32)
    actions = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
    log_probs = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
    rewards = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
    dones = np.zeros((config.rollout_steps, config.num_envs), dtype=np.bool_)
    values = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)

    for step in range(config.rollout_steps):
        observations[step] = current_obs
        with torch.no_grad():
            tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device)
            logits, value = network(tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        action_batch = action.cpu().numpy().astype(np.int64)
        transition_obs, reward_batch, done_batch, infos, next_current_obs = vec_env.step(action_batch)
        episode_rewards += reward_batch

        truncated = np.asarray([bool(info["truncated"]) for info in infos], dtype=np.bool_)
        if np.any(truncated):
            with torch.no_grad():
                terminal_tensor = torch.as_tensor(transition_obs[truncated], dtype=torch.float32, device=device)
                _, terminal_values = network(terminal_tensor)
            reward_batch = reward_batch.copy()
            reward_batch[truncated] += config.gamma * terminal_values.cpu().numpy()

        actions[step] = action_batch
        log_probs[step] = log_prob.cpu().numpy()
        rewards[step] = reward_batch
        dones[step] = done_batch
        values[step] = value.cpu().numpy()
        episode_steps += 1
        current_obs = next_current_obs

        for env_idx, done in enumerate(done_batch):
            if not done:
                continue
            info = infos[env_idx]
            _append_csv(
                metrics_path,
                {
                    "step": global_step + (step + 1) * config.num_envs,
                    "episode": episode,
                    "episode_reward": float(episode_rewards[env_idx]),
                    "episode_steps": int(episode_steps[env_idx]),
                    "success": bool(info["success"]),
                    "done_reason": info["done_reason"],
                },
            )
            episode += 1
            episode_rewards[env_idx] = 0.0
            episode_steps[env_idx] = 0

    with torch.no_grad():
        tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device)
        _, next_value = network(tensor)
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        next_value.cpu().numpy(),
        config.gamma,
        config.gae_lambda,
    )
    return {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "advantages": advantages,
        "returns": returns,
        "current_obs": current_obs,
        "collected_steps": config.rollout_steps * config.num_envs,
        "episode": episode,
    }


def _optimize(
    network: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    observations = rollout["observations"].reshape((-1, rollout["observations"].shape[-1]))
    actions = rollout["actions"].reshape(-1)
    old_log_probs = rollout["log_probs"].reshape(-1)
    advantages = rollout["advantages"].reshape(-1)
    returns = rollout["returns"].reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics: list[tuple[float, float, float, float, float]] = []
    indices = np.arange(len(observations))
    for _ in range(config.update_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            obs_tensor = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(actions[batch_indices], dtype=torch.int64, device=device)
            old_log_prob_tensor = torch.as_tensor(old_log_probs[batch_indices], dtype=torch.float32, device=device)
            advantage_tensor = torch.as_tensor(advantages[batch_indices], dtype=torch.float32, device=device)
            return_tensor = torch.as_tensor(returns[batch_indices], dtype=torch.float32, device=device)

            logits, values = network(obs_tensor)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(action_tensor)
            log_ratio = new_log_probs - old_log_prob_tensor
            ratio = log_ratio.exp()
            policy_loss = -torch.min(
                advantage_tensor * ratio,
                advantage_tensor * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
            ).mean()
            value_loss = nn.functional.mse_loss(values, return_tensor)
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()
            metrics.append(
                (
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy.item()),
                    float(approx_kl.item()),
                    float(clip_fraction.item()),
                )
            )
    means = np.mean(np.asarray(metrics, dtype=np.float64), axis=0)
    return {
        "policy_loss": float(means[0]),
        "value_loss": float(means[1]),
        "entropy": float(means[2]),
        "approx_kl": float(means[3]),
        "clip_fraction": float(means[4]),
    }


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_run_dir(output_root: str | Path, config: PPOConfig) -> Path:
    return Path(output_root) / (config.run_name or strftime("%Y%m%d_%H%M%S"))


def _write_csv_header(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=list(row)).writerow(row)


def _trajectory_final_progress(path: Path) -> float:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return float(metadata["final_info"]["progress"])
