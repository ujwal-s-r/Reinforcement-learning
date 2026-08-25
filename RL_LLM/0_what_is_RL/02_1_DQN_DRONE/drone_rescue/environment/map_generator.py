from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratedMap:
    obstacles: np.ndarray
    start: tuple[int, int]
    rescue: tuple[int, int]
    solvable: bool


def generate_static_map(
    size: int,
    *,
    obstacle_probability: float,
    seed: int,
) -> GeneratedMap:
    rng = np.random.default_rng(seed)
    while True:
        cells = rng.random((size, size)) < obstacle_probability
        start, rescue = _distinct_free_cells(cells, rng)
        if start is None:
            continue
        reachable = _reachable_cells(cells, start)
        if rescue in reachable:
            return GeneratedMap(cells, start, rescue, True)


def _distinct_free_cells(
    grid: np.ndarray,
    rng: np.Generator,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    free = np.argwhere(~grid)
    if len(free) < 2:
        return None, None
    first, second = rng.choice(len(free), size=2, replace=False)
    return tuple(map(int, free[first])), tuple(map(int, free[second]))


def _reachable_cells(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = grid.shape
    frontier = {start}
    seen = {start}
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while frontier:
        row, column = frontier.pop()
        for delta_row, delta_column in deltas:
            neighbor = (row + delta_row, column + delta_column)
            if (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
                and not grid[neighbor]
                and neighbor not in seen
            ):
                seen.add(neighbor)
                frontier.add(neighbor)
    return seen

