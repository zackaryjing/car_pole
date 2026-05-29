from rl_racing.actions import Control
from rl_racing.env import RacingEnv
from rl_racing.play import advance_manual_frame, consume_simulation_steps, control_from_pressed_keys


def test_consume_simulation_steps_accumulates_fractional_frames():
    steps, acc = consume_simulation_steps(0.0, frame_seconds=1.0 / 60.0, sim_speed=1.0, sim_dt=1.0 / 30.0)

    assert steps == 0
    assert acc > 0.0

    steps, acc = consume_simulation_steps(acc, frame_seconds=1.0 / 60.0, sim_speed=1.0, sim_dt=1.0 / 30.0)

    assert steps == 1
    assert abs(acc) < 1e-9


def test_sim_speed_scales_simulation_steps():
    steps, acc = consume_simulation_steps(0.0, frame_seconds=1.0 / 60.0, sim_speed=4.0, sim_dt=1.0 / 30.0)

    assert steps == 2
    assert abs(acc) < 1e-9


def test_advance_manual_frame_supports_headless_input_simulation():
    env = RacingEnv()
    _, info = env.reset(seed=0)

    first = advance_manual_frame(
        env=env,
        control=Control(throttle=1.0, steer=0.0),
        accumulator=0.0,
        frame_seconds=1.0 / 60.0,
        sim_speed=1.0,
        seed=0,
        reward=0.0,
        info=info,
    )
    assert first.sim_steps == 0
    assert first.info["steps"] == 0
    assert not first.reset

    second = advance_manual_frame(
        env=env,
        control=Control(throttle=1.0, steer=0.0),
        accumulator=first.accumulator,
        frame_seconds=1.0 / 60.0,
        sim_speed=1.0,
        seed=0,
        reward=first.reward,
        info=first.info,
    )
    assert second.sim_steps == 1
    assert second.info["steps"] == 1
    assert not second.reset


def test_control_from_pressed_keys_maps_keyboard_state():
    import pygame

    pressed = {pygame.K_w, pygame.K_a}
    control = control_from_pressed_keys(lambda key: key in pressed)

    assert control == Control(throttle=1.0, steer=-1.0)
