import numpy as np

from rl_racing.config import TrackConfig, VehicleConfig
from rl_racing.track import generate_track


def test_track_generation_is_seeded():
    cfg = TrackConfig()
    vehicle_cfg = VehicleConfig()
    t1 = generate_track(np.random.default_rng(123), cfg, vehicle_cfg)
    t2 = generate_track(np.random.default_rng(123), cfg, vehicle_cfg)

    np.testing.assert_allclose(t1.centerline, t2.centerline)
    assert len(t1.obstacles) == len(t2.obstacles)
    for a, b in zip(t1.obstacles, t2.obstacles):
        np.testing.assert_allclose(a.center, b.center)
        assert a.radius == b.radius


def test_start_pose_is_on_track():
    track = generate_track(np.random.default_rng(0), TrackConfig(), VehicleConfig())
    assert track.is_on_track(track.start_pose.position, VehicleConfig().radius)


def test_progress_increases_along_centerline():
    track = generate_track(np.random.default_rng(0), TrackConfig(), VehicleConfig())
    p0, _ = track.sample_at(track.length * 0.2)
    p1, _ = track.sample_at(track.length * 0.7)

    assert float(track.query(p1)["progress"]) > float(track.query(p0)["progress"])

