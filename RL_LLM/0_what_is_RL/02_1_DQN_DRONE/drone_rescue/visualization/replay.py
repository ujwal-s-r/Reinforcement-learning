from __future__ import annotations

import numpy as np


def render_grid(obstacles: np.ndarray, start, rescue, drone=None) -> str:
    symbols = np.full(obstacles.shape, ".", dtype=object)
    symbols[obstacles] = "X"
    symbols[start] = "S"
    symbols[rescue] = "R"
    if drone is not None and tuple(drone) not in {tuple(start), tuple(rescue)}:
        symbols[tuple(drone)] = "D"
    border = "+" + "-" * (obstacles.shape[1] * 2 + 1) + "+"
    rows = ["| " + " ".join(row) + " |" for row in symbols]
    return "\n".join([border, *rows, border])


def render_path(obstacles: np.ndarray, path, start, rescue) -> str:
    symbols = np.full(obstacles.shape, ".", dtype=object)
    symbols[obstacles] = "X"
    for cell in path:
        if tuple(cell) != tuple(start):
            symbols[tuple(cell)] = "o"
    symbols[start] = "S"
    symbols[rescue] = "R"
    border = "+" + "-" * (obstacles.shape[1] * 2 + 1) + "+"
    rows = ["| " + " ".join(row) + " |" for row in symbols]
    return "\n".join([border, *rows, border])
