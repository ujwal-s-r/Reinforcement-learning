from __future__ import annotations

import heapq
from time import perf_counter

from .world_model import GridWorldModel, reconstruct_path


def manhattan_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def astar(model: GridWorldModel, start: tuple[int, int], goal: tuple[int, int]):
    started = perf_counter()
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(manhattan_distance(start, goal), 0.0, start)]
    while queue:
        _, distance, cell = heapq.heappop(queue)
        if distance > distances.get(cell, float("inf")):
            continue
        if cell == goal:
            return reconstruct_path(previous, goal), distance, perf_counter() - started
        for neighbor in model.neighbors(cell):
            candidate = distance + model.cost(cell, neighbor)
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = cell
                heapq.heappush(queue, (candidate + manhattan_distance(neighbor, goal), candidate, neighbor))
    raise ValueError("no path exists")


class AStarPlanner:
    def plan(self, model: GridWorldModel, start: tuple[int, int], goal: tuple[int, int]):
        return astar(model, start, goal)

