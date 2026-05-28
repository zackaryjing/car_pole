"""Track generation and spatial queries."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

from rl_racing.config import TrackConfig, VehicleConfig
from rl_racing.geometry import (
    Pose2D,
    angle_of,
    heading_vector,
    polyline_lengths,
    project_point_to_polyline,
    sample_polyline_at,
    vec2,
)


@dataclass(frozen=True)
class CircleObstacle:
    center: NDArray[np.float64]
    radius: float


@dataclass
class Track:
    centerline: NDArray[np.float64]
    width: float
    obstacles: list[CircleObstacle]
    start_pose: Pose2D
    finish_center: NDArray[np.float64]
    finish_heading: float
    cumulative_lengths: NDArray[np.float64]

    @property
    def length(self) -> float:
        return float(self.cumulative_lengths[-1])

    def query(self, position: NDArray[np.float64]) -> dict[str, float | NDArray[np.float64]]:
        projection = project_point_to_polyline(position, self.centerline, self.cumulative_lengths)
        progress = projection.arc_length / max(self.length, 1e-6)
        _, tangent_heading = sample_polyline_at(self.centerline, self.cumulative_lengths, projection.arc_length)
        return {
            "nearest": projection.point,
            "lateral_distance": projection.distance,
            "arc_length": projection.arc_length,
            "progress": progress,
            "tangent_heading": tangent_heading,
            "segment_index": float(projection.segment_index),
        }

    def is_on_track(self, position: NDArray[np.float64], car_radius: float) -> bool:
        query = self.query(position)
        return float(query["lateral_distance"]) <= self.width * 0.5 - car_radius

    def collides_obstacle(self, position: NDArray[np.float64], car_radius: float) -> bool:
        for obstacle in self.obstacles:
            if float(np.linalg.norm(position - obstacle.center)) <= car_radius + obstacle.radius:
                return True
        return False

    def sample_at(self, arc_length: float) -> tuple[NDArray[np.float64], float]:
        return sample_polyline_at(self.centerline, self.cumulative_lengths, arc_length)


def generate_track(rng: np.random.Generator, cfg: TrackConfig, vehicle_cfg: VehicleConfig) -> Track:
    points = _generate_centerline(rng, cfg)
    points = _smooth_centerline(points, cfg.smoothing_passes)
    cumulative = polyline_lengths(points)
    start_heading = angle_of(points[1] - points[0])
    start_pose = Pose2D(position=points[0].copy(), heading=start_heading)
    finish_center = points[-1].copy()
    finish_heading = angle_of(points[-1] - points[-2])
    obstacles = _generate_obstacles(rng, points, cumulative, cfg, vehicle_cfg)
    return Track(
        centerline=points,
        width=cfg.width,
        obstacles=obstacles,
        start_pose=start_pose,
        finish_center=finish_center,
        finish_heading=finish_heading,
        cumulative_lengths=cumulative,
    )


def _generate_centerline(rng: np.random.Generator, cfg: TrackConfig) -> NDArray[np.float64]:
    points = [vec2(0.0, 0.0)]
    heading = 0.0
    max_turn = np.deg2rad(cfg.max_turn_degrees)
    for _ in range(cfg.control_points - 1):
        heading += float(rng.uniform(-max_turn, max_turn))
        length = float(rng.uniform(cfg.segment_length_min, cfg.segment_length_max))
        points.append(points[-1] + heading_vector(heading) * length)

    arr = np.vstack(points)
    arr -= arr.min(axis=0)
    arr += np.array([cfg.width * 2.0, cfg.width * 2.0], dtype=np.float64)
    return arr


def _smooth_centerline(points: NDArray[np.float64], passes: int) -> NDArray[np.float64]:
    smoothed = points.copy()
    for _ in range(passes):
        new_points = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p = smoothed[i]
            q = smoothed[i + 1]
            new_points.append(0.75 * p + 0.25 * q)
            new_points.append(0.25 * p + 0.75 * q)
        new_points.append(smoothed[-1])
        smoothed = np.vstack(new_points)
    return smoothed.astype(np.float64)


def _generate_obstacles(
    rng: np.random.Generator,
    centerline: NDArray[np.float64],
    cumulative: NDArray[np.float64],
    cfg: TrackConfig,
    vehicle_cfg: VehicleConfig,
) -> list[CircleObstacle]:
    length = float(cumulative[-1])
    count = max(1, int(length / cfg.obstacle_spacing))
    obstacles: list[CircleObstacle] = []
    attempts = count * 30
    min_s = cfg.start_clearance
    max_s = max(min_s, length - cfg.finish_clearance)
    lateral_limit = cfg.width * 0.5 - cfg.obstacle_margin - vehicle_cfg.radius
    if lateral_limit <= cfg.obstacle_radius_max:
        return obstacles

    for _ in range(attempts):
        if len(obstacles) >= count:
            break
        s = float(rng.uniform(min_s, max_s))
        center, heading = sample_polyline_at(centerline, cumulative, s)
        normal = np.array([-sin(heading), cos(heading)], dtype=np.float64)
        radius = float(rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max))
        lateral = float(rng.uniform(-lateral_limit + radius, lateral_limit - radius))
        candidate = center + normal * lateral
        if all(np.linalg.norm(candidate - obs.center) >= radius + obs.radius + cfg.obstacle_margin for obs in obstacles):
            obstacles.append(CircleObstacle(candidate, radius))
    return obstacles


def finish_segment(track: Track) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal = np.array([-sin(track.finish_heading), cos(track.finish_heading)], dtype=np.float64)
    half = normal * track.width * 0.5
    return track.finish_center - half, track.finish_center + half

