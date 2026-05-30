"""Manual pygame entry point."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, replace

from rl_racing.actions import Control
from rl_racing.config import EnvConfig, ObservationConfig, VehicleConfig
from rl_racing.env import RacingEnv
from rl_racing.renderer import draw_control_overlay, draw_world


def consume_simulation_steps(
    accumulator: float, frame_seconds: float, sim_speed: float, sim_dt: float, max_steps: int = 20
) -> tuple[int, float]:
    accumulator += frame_seconds * sim_speed
    steps = min(int(accumulator / sim_dt), max_steps)
    accumulator -= steps * sim_dt
    return steps, accumulator


@dataclass(frozen=True)
class ManualFrameResult:
    sim_steps: int
    accumulator: float
    reward: float
    info: dict
    reset: bool


def advance_manual_frame(
    env: RacingEnv,
    control: Control,
    accumulator: float,
    frame_seconds: float,
    sim_speed: float,
    seed: int,
    reward: float,
    info: dict,
) -> ManualFrameResult:
    sim_steps, accumulator = consume_simulation_steps(
        accumulator=accumulator,
        frame_seconds=frame_seconds,
        sim_speed=max(sim_speed, 0.0),
        sim_dt=env.config.dt,
    )
    reset = False
    for _ in range(sim_steps):
        reward, terminated, truncated, info = env.advance(control)
        if terminated or truncated:
            env.reset(seed=seed)
            accumulator = 0.0
            reset = True
            break
    return ManualFrameResult(
        sim_steps=sim_steps,
        accumulator=accumulator,
        reward=reward,
        info=info,
        reset=reset,
    )


def control_from_pressed_keys(is_pressed: Callable[[int], bool]) -> Control:
    import pygame

    throttle = 0.0
    steer = 0.0
    if is_pressed(pygame.K_w) or is_pressed(pygame.K_UP):
        throttle += 1.0
    if is_pressed(pygame.K_s) or is_pressed(pygame.K_DOWN):
        throttle -= 1.0
    if is_pressed(pygame.K_a) or is_pressed(pygame.K_LEFT):
        steer -= 1.0
    if is_pressed(pygame.K_d) or is_pressed(pygame.K_RIGHT):
        steer += 1.0
    return Control(throttle=throttle, steer=steer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["follow", "global"], default="follow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--sim-speed", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--acceleration", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("RL Racing Playground")
    clock = pygame.time.Clock()

    vehicle_cfg = VehicleConfig()
    if args.max_speed is not None:
        vehicle_cfg = replace(vehicle_cfg, max_forward_speed=args.max_speed)
    if args.acceleration is not None:
        vehicle_cfg = replace(vehicle_cfg, acceleration=args.acceleration)
    env = RacingEnv(EnvConfig(vehicle=vehicle_cfg, observation=ObservationConfig(obs_type="structured")))
    env.reset(seed=args.seed)
    view = args.view
    seed = args.seed
    running = True
    accumulator = 0.0
    reward = 0.0
    info = {
        "progress": 0.0,
        "steps": 0,
        "done_reason": "",
    }
    frame_seconds = 0.0

    while running:
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
        control = control_from_pressed_keys(lambda key: bool(keys[key]))

        frame_result = advance_manual_frame(
            env=env,
            control=control,
            accumulator=accumulator,
            frame_seconds=frame_seconds,
            sim_speed=args.sim_speed,
            seed=seed,
            reward=reward,
            info=info,
        )
        sim_steps = frame_result.sim_steps
        accumulator = frame_result.accumulator
        reward = frame_result.reward
        info = frame_result.info

        assert env.track is not None and env.vehicle is not None
        debug = [
            f"seed {seed} view {view}",
            f"sim_speed {args.sim_speed:.2f}x render_fps {args.render_fps}",
            f"sim_steps/frame {sim_steps}",
            f"speed {env.vehicle.speed:6.1f}",
            f"steer {env.vehicle.steering:6.2f}",
            f"progress {info['progress']:.3f}",
            f"reward {reward:7.3f}",
            f"steps {info['steps']}",
            f"done {info['done_reason']}",
        ]
        draw_world(screen, env.track, env.vehicle, env.config, view=view, show_debug=True, debug_lines=debug)
        draw_control_overlay(screen, control.throttle, control.steer, corner="bottom_left")
        pygame.display.flip()
        frame_seconds = clock.tick(args.render_fps) / 1000.0

    pygame.quit()


if __name__ == "__main__":
    main()
