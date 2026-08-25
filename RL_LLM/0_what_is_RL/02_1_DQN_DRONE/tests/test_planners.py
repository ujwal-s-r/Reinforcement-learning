import numpy as np

from drone_rescue.planners.astar import astar
from drone_rescue.planners.dijkstra import dijkstra
from drone_rescue.planners.world_model import GridWorldModel


def test_astar_and_dijkstra_find_optimal_path():
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[1, 1:3] = True
    model = GridWorldModel(obstacles)
    start, goal = (0, 0), (3, 3)
    astar_path, astar_cost, _ = astar(model, start, goal)
    dijkstra_path, dijkstra_cost, _ = dijkstra(model, start, goal)
    assert len(astar_path) - 1 == 6
    assert len(dijkstra_path) - 1 == 6
    assert astar_cost == dijkstra_cost == 6.0

