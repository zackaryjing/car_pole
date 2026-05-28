"""Discrete action mapping shared by humans and RL agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Control:
    throttle: float
    steer: float


DISCRETE_ACTIONS: tuple[Control, ...] = (
    Control(0.0, 0.0),
    Control(1.0, 0.0),
    Control(-1.0, 0.0),
    Control(0.0, -1.0),
    Control(0.0, 1.0),
    Control(1.0, -1.0),
    Control(1.0, 1.0),
    Control(-1.0, -1.0),
    Control(-1.0, 1.0),
)


def action_to_control(action: int | Control) -> Control:
    if isinstance(action, Control):
        return action
    return DISCRETE_ACTIONS[int(action)]

