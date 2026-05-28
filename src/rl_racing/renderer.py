"""Pygame rendering for global and follow camera views."""

from __future__ import annotations

from math import cos, sin
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from rl_racing.config import EnvConfig
from rl_racing.geometry import rotate, world_to_local
from rl_racing.track import Track, finish_segment
from rl_racing.vehicle import VehicleState


ScreenPointFn = Callable[[NDArray[np.float64]], tuple[int, int]]


def render_rgb_array(
    track: Track,
    vehicle: VehicleState,
    cfg: EnvConfig,
    view: str = "follow",
    size: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    import pygame

    if size is None:
        image_size = cfg.observation.image_size
        size = (image_size, image_size)
    surface = pygame.Surface(size)
    draw_world(surface, track, vehicle, cfg, view=view)
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8)


def draw_world(
    surface,
    track: Track,
    vehicle: VehicleState,
    cfg: EnvConfig,
    view: str = "follow",
    show_debug: bool = False,
    debug_lines: list[str] | None = None,
) -> None:
    import pygame

    surface.fill(cfg.render.grass_color)
    to_screen = _make_camera(surface.get_size(), track, vehicle, cfg, view)

    center_points = [to_screen(p) for p in track.centerline]
    if len(center_points) >= 2:
        pygame.draw.lines(surface, cfg.render.track_color, False, center_points, int(track.width))
        pygame.draw.lines(surface, cfg.render.boundary_color, False, center_points, int(track.width + 5))
        pygame.draw.lines(surface, cfg.render.track_color, False, center_points, int(track.width))
        pygame.draw.lines(surface, cfg.render.centerline_color, False, center_points, 2)

    for obstacle in track.obstacles:
        center = to_screen(obstacle.center)
        pygame.draw.circle(surface, cfg.render.obstacle_color, center, max(3, int(obstacle.radius)))

    start = to_screen(track.start_pose.position)
    pygame.draw.circle(surface, (90, 210, 110), start, 9)
    f0, f1 = finish_segment(track)
    pygame.draw.line(surface, cfg.render.finish_color, to_screen(f0), to_screen(f1), 5)

    _draw_car(surface, vehicle, cfg, to_screen)

    if show_debug:
        _draw_debug(surface, debug_lines or [])


def _make_camera(
    size: tuple[int, int], track: Track, vehicle: VehicleState, cfg: EnvConfig, view: str
) -> ScreenPointFn:
    width, height = size
    if view == "global":
        mins = track.centerline.min(axis=0) - track.width
        maxs = track.centerline.max(axis=0) + track.width
        extent = np.maximum(maxs - mins, 1.0)
        scale = min((width - 40) / extent[0], (height - 40) / extent[1])
        offset = np.array([20.0, 20.0], dtype=np.float64) - mins * scale

        def global_camera(point: NDArray[np.float64]) -> tuple[int, int]:
            p = point * scale + offset
            return int(p[0]), int(p[1])

        return global_camera

    if view != "follow":
        raise ValueError(f"unknown view: {view}")

    scale = cfg.render.follow_pixels_per_unit
    center = np.array([width * 0.5, height * 0.62], dtype=np.float64)

    def follow_camera(point: NDArray[np.float64]) -> tuple[int, int]:
        local = world_to_local(point, vehicle.position, vehicle.heading)
        screen = center + np.array([local[1], -local[0]], dtype=np.float64) * scale
        return int(screen[0]), int(screen[1])

    return follow_camera


def _draw_car(surface, vehicle: VehicleState, cfg: EnvConfig, to_screen: ScreenPointFn) -> None:
    import pygame

    nose = vehicle.position + np.array([cos(vehicle.heading), sin(vehicle.heading)], dtype=np.float64) * 22.0
    rear_left = vehicle.position + rotate(np.array([-14.0, -10.0], dtype=np.float64), vehicle.heading)
    rear_right = vehicle.position + rotate(np.array([-14.0, 10.0], dtype=np.float64), vehicle.heading)
    pygame.draw.polygon(surface, cfg.render.car_color, [to_screen(nose), to_screen(rear_left), to_screen(rear_right)])
    pygame.draw.circle(surface, (30, 30, 30), to_screen(vehicle.position), int(cfg.vehicle.radius), 1)


def _draw_debug(surface, lines: list[str]) -> None:
    import pygame

    font = pygame.font.Font(None, 22)
    y = 8
    for line in lines:
        text = font.render(line, True, (245, 245, 235))
        surface.blit(text, (8, y))
        y += 22

