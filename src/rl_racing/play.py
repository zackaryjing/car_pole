"""Manual pygame entry point."""

from __future__ import annotations

import argparse

from rl_racing.actions import Control
from rl_racing.config import EnvConfig, ObservationConfig
from rl_racing.env import RacingEnv
from rl_racing.renderer import draw_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["follow", "global"], default="follow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import pygame

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("RL Racing Playground")
    clock = pygame.time.Clock()

    env = RacingEnv(EnvConfig(observation=ObservationConfig(obs_type="structured")))
    env.reset(seed=args.seed)
    view = args.view
    seed = args.seed
    running = True

    while running:
        throttle = 0.0
        steer = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset(seed=seed)
                elif event.key == pygame.K_n:
                    seed += 1
                    env.reset(seed=seed)
                elif event.key == pygame.K_v:
                    view = "global" if view == "follow" else "follow"

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            throttle -= 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer += 1.0

        _, reward, terminated, truncated, info = env.step(Control(throttle=throttle, steer=steer))
        if terminated or truncated:
            env.reset(seed=seed)

        assert env.track is not None and env.vehicle is not None
        debug = [
            f"seed {seed} view {view}",
            f"speed {env.vehicle.speed:6.1f}",
            f"steer {env.vehicle.steering:6.2f}",
            f"progress {info['progress']:.3f}",
            f"reward {reward:7.3f}",
            f"steps {info['steps']}",
            f"done {info['done_reason']}",
        ]
        draw_world(screen, env.track, env.vehicle, env.config, view=view, show_debug=True, debug_lines=debug)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
