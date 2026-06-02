"""Command line entry point for sensor-observation PPO training."""

from __future__ import annotations

import argparse

from rl_racing.rl.ppo import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/ppo_sensor")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--total-steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--env-max-steps", type=int, default=PPOConfig.env_max_steps)
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--clip-coef", type=float, default=PPOConfig.clip_coef)
    parser.add_argument("--value-loss-coef", type=float, default=PPOConfig.value_loss_coef)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--hidden-dim", type=int, default=PPOConfig.hidden_dim)
    parser.add_argument("--eval-interval", type=int, default=PPOConfig.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    parser.add_argument("--checkpoint-interval", type=int, default=PPOConfig.checkpoint_interval)
    parser.add_argument("--device", default=PPOConfig.device)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = PPOConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        env_max_steps=args.env_max_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        run_name=args.run_name,
        progress=not args.no_progress,
    )
    result = train_ppo(config, output_root=args.output_root)
    print(f"run_dir={result.run_dir}")
    print(f"final_step={result.final_step}")
    print(f"best_successes={result.best_successes}")


if __name__ == "__main__":
    main()
