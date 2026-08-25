from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class GridWorldModel:
    obstacles: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.obstacles.shape

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        height, width = self.shape
        result = []
        for delta_row, delta_column in NEIGHBORS:
            candidate = (cell[0] + delta_row, cell[1] + delta_column)
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and not self.obstacles[candidate]
            ):
                result.append(candidate)
        return result

    def cost(self, current: tuple[int, int], next_cell: tuple[int, int]) -> float:
        return 1.0


def reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], end: tuple[int, int]) -> list[tuple[int, int]]:
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return list(reversed(path))


def pop_min(queue: list, priority: dict) -> tuple[int, int]:
    _, cell = heapq.heappop(queue)
    while cell not in priority:
        _, cell = heapq.heappop(queue)
    del priority[cell]
    return cell

