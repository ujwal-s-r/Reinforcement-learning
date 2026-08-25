from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..environment.grid_world import GridWorldEnv
from ..planners.world_model import GridWorldModel


@dataclass(frozen=True)
class EvaluationResult:
    planner: str
    success_rate: float
    average_path_length: float
    optimal_path_length: float
    average_cost: float
    planning_latency_ms: float
    path_efficiency: float
    episodes: int


def evaluate_planner(
    name: str,
    plan_function,
    maps: list,
) -> EvaluationResult:
    successes = []
    path_lengths = []
    costs = []
    latencies = []
    optimal_lengths = []
    for generated_map in maps:
        model = GridWorldModel(generated_map.obstacles)
        started = perf_counter()
        path, cost, _ = plan_function(model, generated_map.start, generated_map.rescue)
        latencies.append((perf_counter() - started) * 1000.0)
        env = GridWorldEnv(generated_map.obstacles, generated_map.start, generated_map.rescue)
        success, steps, collisions = execute_path(env, path)
        successes.append(success)
        path_lengths.append(steps)
        costs.append(cost if success else np.nan)
        optimal_lengths.append(len(path) - 1)
    return EvaluationResult(
        planner=name,
        success_rate=float(np.mean(successes)),
        average_path_length=float(np.mean(path_lengths)),
        optimal_path_length=float(np.mean(optimal_lengths)),
        average_cost=float(np.nanmean(costs)),
        planning_latency_ms=float(np.mean(latencies)),
        path_efficiency=float(np.divide(optimal_lengths, path_lengths).mean()),
        episodes=len(maps),
    )


def execute_path(env: GridWorldEnv, path):
    state = env.reset()
    collisions = 0
    for current, next_cell in zip(path[:-1], path[1:]):
        result = env.step(action_to_index(current, next_cell))
        collisions += int(result.info["collision"])
        state = result
        if result.terminated or result.truncated:
            break
    return bool(state.info["success"]), int(state.info["path_length"]), collisions


def action_to_index(current, next_cell) -> int:
    from ..environment.grid_world import ACTIONS
    from ..planners.world_model import NEIGHBORS

    if False:
        return int(current)
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    return [index for index, candidate in enumerate(NEIGHBORS) if candidate == delta][0]
