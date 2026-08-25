from __future__ import annotations

from drone_rescue.environment.map_generator import GeneratedMap, generate_static_map
from drone_rescue.evaluation.metrics import EvaluationResult, evaluate_planner
from drone_rescue.planners.astar import astar
from drone_rescue.planners.dijkstra import dijkstra


def generate_map_suite(episodes: int, size: int, obstacle_probability: float, seed: int) -> list[GeneratedMap]:
    return [
        generate_static_map(
            size,
            obstacle_probability=obstacle_probability,
            seed=seed + episode_index,
        )
        for episode_index in range(episodes)
    ]


def run_static_benchmark(maps: list[GeneratedMap]) -> list[EvaluationResult]:
    return [
        evaluate_planner("astar", astar, maps),
        evaluate_planner("dijkstra", dijkstra, maps),
    ]
