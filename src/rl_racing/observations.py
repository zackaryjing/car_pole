"""Observation builders for the racing environment."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from rl_racing.config import EnvConfig
from rl_racing.geometry import normalize_angle, ray_circle_intersection, rotate, world_to_local
from rl_racing.track import Track
from rl_racing.vehicle import VehicleState


def structured_observation(track: Track, vehicle: VehicleState, cfg: EnvConfig) -> NDArray[np.float32]:
    obs_cfg = cfg.observation
    veh_cfg = cfg.vehicle
    query = track.query(vehicle.position)
    heading_error = normalize_angle(vehicle.heading - float(query["tangent_heading"]))

    values: list[float] = [
        float(np.clip(vehicle.speed / veh_cfg.max_forward_speed, -1.0, 1.0)),
        float(np.clip(vehicle.angular_velocity / veh_cfg.max_turn_rate, -1.0, 1.0)),
        float(heading_error / np.pi),
        float(query["progress"]),
    ]

    angles = np.linspace(
        -radians(obs_cfg.ray_fov_degrees) * 0.5,
        radians(obs_cfg.ray_fov_degrees) * 0.5,
        obs_cfg.ray_count,
    )
    track_distances: list[float] = []
    obstacle_distances: list[float] = []
    for angle in angles:
        direction = rotate(np.array([1.0, 0.0], dtype=np.float64), vehicle.heading + float(angle))
        track_distances.append(_ray_track_distance(track, vehicle.position, direction, cfg))
        obstacle_distances.append(_ray_obstacle_distance(track, vehicle.position, direction, cfg))

    values.extend(track_distances)
    values.extend(obstacle_distances)

    current_s = float(query["arc_length"])
    for i in range(obs_cfg.future_count):
        s = current_s + (i + 1) * obs_cfg.future_step_distance
        point, _ = track.sample_at(s)
        local = world_to_local(point, vehicle.position, vehicle.heading) / obs_cfg.ray_max_distance
        values.extend(np.clip(local, -1.0, 1.0).tolist())

    return np.asarray(values, dtype=np.float32)


def structured_observation_size(cfg: EnvConfig) -> int:
    obs_cfg = cfg.observation
    return 4 + obs_cfg.ray_count * 2 + obs_cfg.future_count * 2


def _ray_track_distance(track: Track, origin: NDArray[np.float64], direction: NDArray[np.float64], cfg: EnvConfig) -> float:
    max_distance = cfg.observation.ray_max_distance
    for distance in np.linspace(0.0, max_distance, 64):
        point = origin + direction * float(distance)
        if not track.is_on_track(point, cfg.vehicle.radius):
            return float(distance / max_distance)
    return 1.0


def _ray_obstacle_distance(track: Track, origin: NDArray[np.float64], direction: NDArray[np.float64], cfg: EnvConfig) -> float:
    max_distance = cfg.observation.ray_max_distance
    best = max_distance
    for obstacle in track.obstacles:
        hit = ray_circle_intersection(origin, direction, obstacle.center, obstacle.radius + cfg.vehicle.radius)
        if hit is not None and hit <= max_distance:
            best = min(best, hit)
    return float(best / max_distance)

