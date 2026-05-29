import math

import numpy as np

from rl_racing.env import RacingEnv
from rl_racing.renderer import draw_world


def test_follow_renderer_has_no_enclosed_grass_holes_when_rotated(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    try:
        env = RacingEnv()
        env.reset(seed=0)
        assert env.track is not None and env.vehicle is not None
        point, heading = env.track.sample_at(700.0)
        env.vehicle.position = point.copy()
        env.vehicle.heading = heading + math.pi / 3.0

        surface = pygame.Surface((360, 280))
        draw_world(surface, env.track, env.vehicle, env.config, view="follow")

        arr = pygame.surfarray.array3d(surface)
        grass = np.array(env.config.render.grass_color, dtype=np.uint8)
        grass_mask = np.all(arr == grass, axis=2)
        non_grass = ~grass_mask

        enclosed_holes = 0
        for y in range(2, arr.shape[1] - 2):
            for x in range(2, arr.shape[0] - 2):
                if grass_mask[x, y] and non_grass[x - 2 : x + 3, y - 2 : y + 3].sum() >= 20:
                    enclosed_holes += 1

        assert enclosed_holes == 0
    finally:
        pygame.quit()
