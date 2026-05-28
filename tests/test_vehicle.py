import numpy as np

from rl_racing.actions import Control
from rl_racing.config import VehicleConfig
from rl_racing.vehicle import VehicleState, step_vehicle


def test_vehicle_accelerates_forward():
    cfg = VehicleConfig()
    state = VehicleState(position=np.array([0.0, 0.0]), heading=0.0)

    for _ in range(10):
        state = step_vehicle(state, Control(throttle=1.0, steer=0.0), cfg, 1.0 / 30.0)

    assert state.speed > 0.0
    assert state.position[0] > 0.0


def test_vehicle_turn_changes_heading():
    cfg = VehicleConfig()
    state = VehicleState(position=np.array([0.0, 0.0]), heading=0.0, speed=100.0)

    next_state = step_vehicle(state, Control(throttle=0.0, steer=1.0), cfg, 1.0 / 30.0)

    assert next_state.heading > state.heading
    assert next_state.angular_velocity > 0.0

