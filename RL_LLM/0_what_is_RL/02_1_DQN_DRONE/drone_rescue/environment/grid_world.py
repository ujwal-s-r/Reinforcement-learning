from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTIONS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
ACTION_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class GridWorldEnv:
    def __init__(
        self,
        obstacles: np.ndarray,
        start: tuple[int, int],
        rescue: tuple[int, int],
        max_steps: int | None = None,
    ) -> None:
        self.obstacles = np.asarray(obstacles, dtype=bool)
        self.height, self.width = self.obstacles.shape
        self.start = self._valid_cell(start)
        self.rescue = self._valid_cell(rescue)
        if self.obstacles[self.rescue]:
            raise ValueError("rescue cell cannot be an obstacle")
        self.max_steps = max_steps or 4 * self.height * self.width
        self._drone = self.start
        self._steps = 0

    def reset(self) -> StepResult:
        self._drone = self.start
        self._steps = 0
        return StepResult(self.observation(), 0.0, False, False, self.info())

    def step(self, action: int | str) -> StepResult:
        action_index = self._action_index(action)
        target = self._drone + ACTION_DELTAS[action_index]
        collision = bool(
            target[0] < 0
            or target[0] >= self.height
            or target[1] < 0
            or target[1] >= self.width
            or self.obstacles[tuple(target)]
        )
        reward = -5.0 if collision else -1.0
        self._drone = tuple(target) if not collision else self._drone
        self._steps += 1
        success = self._drone == self.rescue
        timeout = self._steps >= self.max_steps and not success
        if success:
            reward += 100.0
        elif timeout:
            reward -= 20.0
        return StepResult(
            self.observation(),
            reward,
            success,
            timeout,
            self.info(collision=collision),
        )

    def observation(self) -> dict[str, object]:
        return {
            "obstacles": self.obstacles.copy(),
            "drone": np.array(self._drone, dtype=np.int16),
            "rescue": np.array(self.rescue, dtype=np.int16),
            "shape": (self.height, self.width),
        }

    def info(self, *, collision: bool = False) -> dict[str, object]:
        return {
            "collision": collision,
            "success": self._drone == self.rescue,
            "timeout": self._steps >= self.max_steps and self._drone != self.rescue,
            "path_length": self._steps,
        }

    def _action_index(self, action: int | str) -> int:
        if isinstance(action, str):
            try:
                return ACTIONS[action.upper()]
            except KeyError as error:
                raise ValueError(f"unknown action: {action}") from error
        if action not in range(len(ACTION_DELTAS)):
            raise ValueError(f"invalid action index: {action}")
        return int(action)

    def _valid_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, column = map(int, cell)
        if not (0 <= row < self.height and 0 <= column < self.width):
            raise ValueError(f"cell outside grid: {cell}")
        if self.obstacles[row, column]:
            raise ValueError(f"cell is an obstacle: {cell}")
        return row, column

