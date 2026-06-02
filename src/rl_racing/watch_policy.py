"""Watch a trained DQN policy in a pygame window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl_racing.actions import action_to_control
from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv
from rl_racing.renderer import draw_control_overlay, draw_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--view", choices=["follow", "global"], default="follow")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--sim-speed", type=float, default=1.0)
    parser.add_argument("--pause-on-done", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    policy = _load_policy(args.checkpoint, device=args.device)
    env = RacingEnv(EnvConfig(observation=ObservationConfig(obs_type="sensor")))
    obs, info = env.reset(seed=args.seed)
    policy.reset(args.seed, info)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"RL Racing Policy: {args.checkpoint}")
    clock = pygame.time.Clock()

    running = True
    accumulator = 0.0
    done_pause = 0.0
    reward = 0.0
    action = 0
    view = args.view
    seed = args.seed

    while running:
        frame_seconds = clock.tick(args.render_fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    view = "global" if view == "follow" else "follow"
                elif event.key == pygame.K_r:
                    obs, info = env.reset(seed=seed)
                    policy.reset(seed, info)
                    done_pause = 0.0
                    accumulator = 0.0
                elif event.key == pygame.K_n:
                    seed += 1
                    obs, info = env.reset(seed=seed)
                    policy.reset(seed, info)
                    done_pause = 0.0
                    accumulator = 0.0

        if done_pause > 0.0:
            done_pause = max(0.0, done_pause - frame_seconds)
            if done_pause == 0.0:
                obs, info = env.reset(seed=seed)
                policy.reset(seed, info)
                accumulator = 0.0
        else:
            accumulator += frame_seconds * max(args.sim_speed, 0.0)
            while accumulator >= env.config.dt:
                action = int(policy.act(obs, info))
                obs, reward, terminated, truncated, info = env.step(action)
                accumulator -= env.config.dt
                if terminated or truncated:
                    done_pause = max(args.pause_on_done, 0.0)
                    break

        assert env.track is not None and env.vehicle is not None
        control = action_to_control(action)
        debug = [
            f"checkpoint {args.checkpoint}",
            f"seed {seed} view {view}",
            f"sim_speed {args.sim_speed:.2f}x render_fps {args.render_fps}",
            f"action {action}: {_control_label(control.throttle, control.steer)}",
            f"speed {env.vehicle.speed:6.1f}",
            f"steer {env.vehicle.steering:6.2f}",
            f"progress {info['progress']:.3f}",
            f"reward {reward:7.3f}",
            f"steps {info['steps']}",
            f"done {info['done_reason']} success {info['success']}",
        ]
        draw_world(screen, env.track, env.vehicle, env.config, view=view, show_debug=True, debug_lines=debug)
        draw_control_overlay(screen, control.throttle, control.steer, corner="bottom_left")
        pygame.display.flip()

    pygame.quit()


def _load_policy(checkpoint: Path, device: str):
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("algorithm") == "ppo" or "actor_critic" in payload:
        from rl_racing.rl.ppo import load_policy
    else:
        from rl_racing.rl.dqn import load_policy
    return load_policy(checkpoint, device=device)


def _control_label(throttle: float, steer: float) -> str:
    longitudinal = "COAST"
    if throttle > 0.0:
        longitudinal = "THROTTLE"
    elif throttle < 0.0:
        longitudinal = "BRAKE"

    lateral = "STRAIGHT"
    if steer < 0.0:
        lateral = "LEFT"
    elif steer > 0.0:
        lateral = "RIGHT"
    return f"{longitudinal} + {lateral}"


if __name__ == "__main__":
    main()
