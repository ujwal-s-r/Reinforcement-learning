import numpy as np

from drone_rescue.environment.grid_world import GridWorldEnv


def test_reaches_goal_and_awards_success():
    obstacles = np.zeros((3, 3), dtype=bool)
    env = GridWorldEnv(obstacles, (0, 0), (0, 2))
    env.reset()
    result = env.step(3)
    result = env.step(3)
    assert result.terminated
    assert result.reward == 99.0
    assert result.info["path_length"] == 2


def test_collision_keeps_drone_in_place():
    obstacles = np.array([[False, True], [False, False]])
    env = GridWorldEnv(obstacles, (0, 0), (1, 1))
    result = env.step(3)
    assert result.info["collision"]
    assert result.reward == -5.0
    assert tuple(result.observation["drone"]) == (0, 0)

