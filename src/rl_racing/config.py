"""Configuration objects for the racing environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackConfig:
    control_points: int = 18
    segment_length_min: float = 130.0
    segment_length_max: float = 210.0
    max_turn_degrees: float = 28.0
    smoothing_passes: int = 3
    width: float = 130.0
    obstacle_radius_min: float = 12.0
    obstacle_radius_max: float = 22.0
    obstacle_spacing: float = 440.0
    obstacle_margin: float = 34.0
    start_clearance: float = 260.0
    finish_clearance: float = 220.0


@dataclass(frozen=True)
class VehicleConfig:
    radius: float = 14.0
    max_forward_speed: float = 260.0
    max_reverse_speed: float = 90.0
    acceleration: float = 340.0
    brake_acceleration: float = 430.0
    drag: float = 1.4
    steering_rate: float = 2.8
    steering_return_rate: float = 1.9
    max_turn_rate: float = 3.1
    low_speed_turn_factor: float = 0.35


@dataclass(frozen=True)
class RewardConfig:
    progress_scale: float = 100.0
    time_penalty: float = -0.01
    success_bonus: float = 100.0
    failure_penalty: float = -50.0
    heading_alignment_scale: float = 0.02


@dataclass(frozen=True)
class ObservationConfig:
    obs_type: str = "sensor"
    ray_count: int = 21
    ray_fov_degrees: float = 220.0
    ray_max_distance: float = 350.0
    future_count: int = 10
    future_step_distance: float = 80.0
    image_size: int = 96


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1000
    height: int = 720
    follow_pixels_per_unit: float = 1.15
    background_color: tuple[int, int, int] = (31, 36, 34)
    grass_color: tuple[int, int, int] = (57, 92, 60)
    track_color: tuple[int, int, int] = (70, 74, 76)
    boundary_color: tuple[int, int, int] = (230, 230, 220)
    centerline_color: tuple[int, int, int] = (120, 126, 130)
    car_color: tuple[int, int, int] = (235, 86, 72)
    obstacle_color: tuple[int, int, int] = (52, 120, 210)
    finish_color: tuple[int, int, int] = (245, 210, 78)


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 1.0 / 30.0
    max_steps: int = 2000
    track: TrackConfig = TrackConfig()
    vehicle: VehicleConfig = VehicleConfig()
    reward: RewardConfig = RewardConfig()
    observation: ObservationConfig = ObservationConfig()
    render: RenderConfig = RenderConfig()
