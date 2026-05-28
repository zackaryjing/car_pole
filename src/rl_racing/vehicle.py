"""Vehicle state and simple kinematic integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np
from numpy.typing import NDArray

from rl_racing.actions import Control
from rl_racing.config import VehicleConfig
from rl_racing.geometry import normalize_angle


@dataclass
class VehicleState:
    position: NDArray[np.float64]
    heading: float
    speed: float = 0.0
    angular_velocity: float = 0.0


def step_vehicle(state: VehicleState, control: Control, cfg: VehicleConfig, dt: float) -> VehicleState:
    accel = cfg.acceleration if control.throttle >= 0.0 else cfg.brake_acceleration
    speed = state.speed + control.throttle * accel * dt
    speed -= cfg.drag * speed * dt
    speed = float(np.clip(speed, -cfg.max_reverse_speed, cfg.max_forward_speed))

    speed_ratio = abs(speed) / cfg.max_forward_speed
    turn_authority = cfg.low_speed_turn_factor + (1.0 - cfg.low_speed_turn_factor) * speed_ratio
    direction = 1.0 if speed >= 0.0 else -1.0
    angular_velocity = control.steer * cfg.max_turn_rate * turn_authority * direction
    heading = normalize_angle(state.heading + angular_velocity * dt)
    position = state.position + np.array([cos(heading), sin(heading)], dtype=np.float64) * speed * dt
    return VehicleState(position=position, heading=heading, speed=speed, angular_velocity=angular_velocity)

