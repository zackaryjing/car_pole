"""Pygame trajectory replay viewer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl_racing.actions import action_to_control
from rl_racing.episode import load_trajectory, trajectory_track, trajectory_vehicle_state
from rl_racing.renderer import draw_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, help="Trajectory base path, .json, or .npz")
    parser.add_argument("--view", choices=["follow", "global"], default="follow")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    from rl_racing.config import EnvConfig

    trajectory = load_trajectory(args.trajectory)
    track = trajectory_track(trajectory)
    cfg = EnvConfig()
    frame_count = len(trajectory.vehicle_states)
    if frame_count == 0:
        raise ValueError("trajectory has no vehicle states")

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"RL Racing Trajectory: {args.trajectory}")
    clock = pygame.time.Clock()

    running = True
    frame = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    args.view = "global" if args.view == "follow" else "follow"
                elif event.key == pygame.K_r:
                    frame = 0.0

        index = min(int(frame), frame_count - 1)
        vehicle = trajectory_vehicle_state(trajectory, index)
        final_info = trajectory.metadata["final_info"]
        action = int(trajectory.actions[min(index, len(trajectory.actions) - 1)]) if len(trajectory.actions) else 0
        control = action_to_control(action)
        debug = [
            f"trajectory {args.trajectory}",
            f"view {args.view} speed {args.speed:.2f}x",
            f"frame {index}/{frame_count - 1}",
            f"seed {trajectory.metadata.get('seed')}",
            f"action {action}: {_control_label(control.throttle, control.steer)}",
            f"input  {_control_bars(control.throttle, control.steer)}",
            f"progress {trajectory.progress[index]:.3f}",
            f"result {final_info['done_reason']} success {final_info['success']}",
        ]
        draw_world(screen, track, vehicle, cfg, view=args.view, show_debug=True, debug_lines=debug)
        pygame.display.flip()

        dt = clock.tick(args.fps) / 1000.0
        frame += args.speed * dt / float(trajectory.metadata["dt"])
        if frame >= frame_count:
            if args.loop:
                frame = 0.0
            else:
                frame = float(frame_count - 1)

    pygame.quit()


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


def _control_bars(throttle: float, steer: float) -> str:
    gas = "####" if throttle > 0.0 else "...."
    brake = "####" if throttle < 0.0 else "...."
    left = "####" if steer < 0.0 else "...."
    right = "####" if steer > 0.0 else "...."
    return f"THR[{gas}] BRK[{brake}] L[{left}] R[{right}]"


if __name__ == "__main__":
    main()
