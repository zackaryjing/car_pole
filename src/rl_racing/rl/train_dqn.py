"""Command line entry point for sensor-observation DQN training."""

from __future__ import annotations

import argparse

from rl_racing.rl.dqn import DQNConfig, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/dqn_sensor")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--total-steps", type=int, default=DQNConfig.total_steps)
    parser.add_argument("--seed", type=int, default=DQNConfig.seed)
    parser.add_argument("--env-max-steps", type=int, default=DQNConfig.env_max_steps)
    parser.add_argument("--replay-size", type=int, default=DQNConfig.replay_size)
    parser.add_argument("--learning-starts", type=int, default=DQNConfig.learning_starts)
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    parser.add_argument("--learning-rate", type=float, default=DQNConfig.learning_rate)
    parser.add_argument("--train-frequency", type=int, default=DQNConfig.train_frequency)
    parser.add_argument("--target-update-interval", type=int, default=DQNConfig.target_update_interval)
    parser.add_argument("--hidden-dim", type=int, default=DQNConfig.hidden_dim)
    parser.add_argument("--epsilon-start", type=float, default=DQNConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=DQNConfig.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=DQNConfig.epsilon_decay_steps)
    parser.add_argument("--eval-interval", type=int, default=DQNConfig.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    parser.add_argument("--checkpoint-interval", type=int, default=DQNConfig.checkpoint_interval)
    parser.add_argument("--device", default=DQNConfig.device)
    args = parser.parse_args()

    config = DQNConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        env_max_steps=args.env_max_steps,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        train_frequency=args.train_frequency,
        target_update_interval=args.target_update_interval,
        hidden_dim=args.hidden_dim,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        run_name=args.run_name,
    )
    result = train_dqn(config, output_root=args.output_root)
    print(f"run_dir={result.run_dir}")
    print(f"final_step={result.final_step}")
    print(f"best_successes={result.best_successes}")


if __name__ == "__main__":
    main()
