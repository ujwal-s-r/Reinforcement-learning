from __future__ import annotations

import heapq
from time import perf_counter

from .world_model import GridWorldModel, reconstruct_path


def dijkstra(model: GridWorldModel, start: tuple[int, int], goal: tuple[int, int]):
    started = perf_counter()
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance > distances.get(cell, float("inf")):
            continue
        if cell == goal:
            elapsed = perf_counter() - started
            return reconstruct_path(previous, goal), distance, elapsed
        for neighbor in model.neighbors(cell):
            candidate = distance + model.cost(cell, neighbor)
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = cell
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError("no path exists")


class DijkstraPlanner:
    def plan(self, model: GridWorldModel, start: tuple[int, int], goal: tuple[int, int]):
        return dijkstra(model, start, goal)

