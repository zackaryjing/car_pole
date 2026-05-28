from rl_racing.play import consume_simulation_steps


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
