from __future__ import annotations

import numpy as np


def encode_observation(observation: dict, size: int | None = None) -> np.ndarray:
    obstacles = observation["obstacles"]
    height, width = obstacles.shape
    encoded = np.zeros((3, height, width), dtype=np.float32)
    encoded[0] = obstacles.astype(np.float32)
    drone_row, drone_column = map(int, observation["drone"])
    rescue_row, rescue_column = map(int, observation["rescue"])
    encoded[1, drone_row, drone_column] = 1.0
    encoded[2, rescue_row, rescue_column] = 1.0
    if size is not None:
        padded = np.zeros((3, size, size), dtype=np.float32)
        padded[:, :height, :width] = encoded
        return padded
    return encoded
