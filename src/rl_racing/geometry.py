"""Small 2D geometry helpers used by the environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np
from numpy.typing import NDArray

Vec2 = NDArray[np.float64]


@dataclass(frozen=True)
class Pose2D:
    position: Vec2
    heading: float


@dataclass(frozen=True)
class SegmentProjection:
    point: Vec2
    distance: float
    segment_index: int
    segment_t: float
    arc_length: float


def vec2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float64)


def normalize_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def heading_vector(heading: float) -> Vec2:
    return vec2(cos(heading), sin(heading))


def angle_of(vector: Vec2) -> float:
    return atan2(float(vector[1]), float(vector[0]))


def rotate(vector: Vec2, angle: float) -> Vec2:
    c = cos(angle)
    s = sin(angle)
    return vec2(c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])


def world_to_local(point: Vec2, origin: Vec2, heading: float) -> Vec2:
    return rotate(point - origin, -heading)


def local_to_world(point: Vec2, origin: Vec2, heading: float) -> Vec2:
    return origin + rotate(point, heading)


def polyline_lengths(points: NDArray[np.float64]) -> NDArray[np.float64]:
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def project_point_to_polyline(
    point: Vec2, points: NDArray[np.float64], cumulative_lengths: NDArray[np.float64] | None = None
) -> SegmentProjection:
    if len(points) < 2:
        raise ValueError("polyline needs at least two points")
    if cumulative_lengths is None:
        cumulative_lengths = polyline_lengths(points)

    best_distance = float("inf")
    best_point = points[0].copy()
    best_index = 0
    best_t = 0.0
    best_arc = 0.0

    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            t = 0.0
        else:
            t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
        projected = start + segment * t
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_point = projected
            best_index = idx
            best_t = t
            segment_length = cumulative_lengths[idx + 1] - cumulative_lengths[idx]
            best_arc = float(cumulative_lengths[idx] + segment_length * t)

    return SegmentProjection(best_point, best_distance, best_index, best_t, best_arc)


def sample_polyline_at(
    points: NDArray[np.float64], cumulative_lengths: NDArray[np.float64], arc_length: float
) -> tuple[Vec2, float]:
    total = float(cumulative_lengths[-1])
    s = float(np.clip(arc_length, 0.0, total))
    idx = int(np.searchsorted(cumulative_lengths, s, side="right") - 1)
    idx = min(max(idx, 0), len(points) - 2)
    seg_len = cumulative_lengths[idx + 1] - cumulative_lengths[idx]
    t = 0.0 if seg_len <= 1e-12 else float((s - cumulative_lengths[idx]) / seg_len)
    point = points[idx] + (points[idx + 1] - points[idx]) * t
    heading = angle_of(points[idx + 1] - points[idx])
    return point, heading


def ray_circle_intersection(origin: Vec2, direction: Vec2, center: Vec2, radius: float) -> float | None:
    oc = origin - center
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]
    hits = [r for r in roots if r >= 0.0]
    return min(hits) if hits else None

