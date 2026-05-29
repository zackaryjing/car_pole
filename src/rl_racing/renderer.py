"""Pygame rendering for global and follow camera views."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from rl_racing.config import EnvConfig
from rl_racing.geometry import rotate, world_to_local
from rl_racing.track import Track, finish_segment
from rl_racing.vehicle import VehicleState


ScreenPointFn = Callable[[NDArray[np.float64]], tuple[int, int]]


@dataclass(frozen=True)
class Camera:
    to_screen: ScreenPointFn
    scale: float


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
    ghost_vehicles: list[VehicleState] | None = None,
) -> None:
    import pygame

    surface.fill(cfg.render.grass_color)
    camera = _make_camera(surface.get_size(), track, vehicle, cfg, view)
    to_screen = camera.to_screen

    _draw_track(surface, track, cfg, camera)

    for obstacle in track.obstacles:
        center = to_screen(obstacle.center)
        pygame.draw.circle(surface, cfg.render.obstacle_color, center, max(3, _scaled_width(obstacle.radius, camera.scale)))

    start = to_screen(track.start_pose.position)
    pygame.draw.circle(surface, (90, 210, 110), start, max(5, _scaled_width(9.0, camera.scale)))
    f0, f1 = finish_segment(track)
    pygame.draw.line(surface, cfg.render.finish_color, to_screen(f0), to_screen(f1), max(3, _scaled_width(5.0, camera.scale)))

    for ghost in ghost_vehicles or []:
        _draw_car(surface, ghost, cfg, camera, color=(80, 170, 255))

    _draw_car(surface, vehicle, cfg, camera)

    if show_debug:
        _draw_debug(surface, debug_lines or [])


def _make_camera(
    size: tuple[int, int], track: Track, vehicle: VehicleState, cfg: EnvConfig, view: str
) -> Camera:
    width, height = size
    if view == "global":
        mins = track.centerline.min(axis=0) - track.width
        maxs = track.centerline.max(axis=0) + track.width
        extent = np.maximum(maxs - mins, 1.0)
        scale = min((width - 40) / extent[0], (height - 40) / extent[1])
        offset = np.array([20.0, 20.0], dtype=np.float64) - mins * scale

        def global_camera(point: NDArray[np.float64]) -> tuple[int, int]:
            p = point * scale + offset
            return _screen_point(p)

        return Camera(global_camera, float(scale))

    if view != "follow":
        raise ValueError(f"unknown view: {view}")

    scale = cfg.render.follow_pixels_per_unit
    center = np.array([width * 0.5, height * 0.62], dtype=np.float64)

    def follow_camera(point: NDArray[np.float64]) -> tuple[int, int]:
        local = world_to_local(point, vehicle.position, vehicle.heading)
        screen = center + np.array([local[1], -local[0]], dtype=np.float64) * scale
        return _screen_point(screen)

    return Camera(follow_camera, float(scale))


def _draw_track(surface, track: Track, cfg: EnvConfig, camera: Camera) -> None:
    import pygame

    _draw_ribbon_layer(surface, track.centerline, (track.width + 5.0) * 0.5, cfg.render.boundary_color, camera)
    _draw_ribbon_layer(surface, track.centerline, track.width * 0.5, cfg.render.track_color, camera)

    center_points = [camera.to_screen(p) for p in track.centerline]
    if len(center_points) >= 2:
        pygame.draw.lines(surface, cfg.render.centerline_color, False, center_points, max(1, _scaled_width(2.0, camera.scale)))


def _draw_ribbon_layer(
    surface, centerline: NDArray[np.float64], half_width: float, color: tuple[int, int, int], camera: Camera
) -> None:
    import pygame

    polygon = _ribbon_polygon(centerline, half_width, camera.to_screen)
    if len(polygon) >= 3:
        pygame.draw.polygon(surface, color, polygon)

    radius = _scaled_width(half_width, camera.scale)
    pygame.draw.circle(surface, color, camera.to_screen(centerline[0]), radius)
    pygame.draw.circle(surface, color, camera.to_screen(centerline[-1]), radius)


def _ribbon_polygon(
    centerline: NDArray[np.float64], half_width: float, to_screen: ScreenPointFn
) -> list[tuple[int, int]]:
    if len(centerline) < 2:
        return []

    left: list[NDArray[np.float64]] = []
    right: list[NDArray[np.float64]] = []
    for idx, point in enumerate(centerline):
        if idx == 0:
            tangent = centerline[1] - centerline[0]
        elif idx == len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[idx + 1] - centerline[idx - 1]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-9:
            normal = np.array([0.0, 1.0], dtype=np.float64)
        else:
            tangent = tangent / norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        left.append(point + normal * half_width)
        right.append(point - normal * half_width)

    return [to_screen(point) for point in left] + [to_screen(point) for point in reversed(right)]


def _draw_car(
    surface, vehicle: VehicleState, cfg: EnvConfig, camera: Camera, color: tuple[int, int, int] | None = None
) -> None:
    import pygame

    to_screen = camera.to_screen
    car_color = color or cfg.render.car_color
    nose = vehicle.position + np.array([cos(vehicle.heading), sin(vehicle.heading)], dtype=np.float64) * 22.0
    rear_left = vehicle.position + rotate(np.array([-14.0, -10.0], dtype=np.float64), vehicle.heading)
    rear_right = vehicle.position + rotate(np.array([-14.0, 10.0], dtype=np.float64), vehicle.heading)
    pygame.draw.polygon(surface, car_color, [to_screen(nose), to_screen(rear_left), to_screen(rear_right)])
    pygame.draw.circle(surface, (30, 30, 30), to_screen(vehicle.position), max(2, _scaled_width(cfg.vehicle.radius, camera.scale)), 1)


def _screen_point(point: NDArray[np.float64]) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _scaled_width(world_width: float, scale: float) -> int:
    return max(1, int(round(world_width * scale)))


def _draw_debug(surface, lines: list[str]) -> None:
    import pygame

    font = pygame.font.Font(None, 22)
    y = 8
    for line in lines:
        text = font.render(line, True, (245, 245, 235))
        surface.blit(text, (8, y))
        y += 22
