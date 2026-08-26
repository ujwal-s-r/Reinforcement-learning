from __future__ import annotations


class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int) -> None:
        self.start = start
        self.end = end
        self.duration = max(1, duration)
        self._step = 0

    @property
    def value(self) -> float:
        fraction = min(self._step / self.duration, 1.0)
        return self.start + (self.end - self.start) * fraction

    def step(self) -> None:
        self._step += 1
