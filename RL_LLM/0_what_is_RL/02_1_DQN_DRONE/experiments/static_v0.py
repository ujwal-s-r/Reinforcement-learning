from __future__ import annotations

import argparse
import json
from pathlib import Path

from drone_rescue.environment.map_generator import generate_static_map
from drone_rescue.planners.astar import astar
from drone_rescue.planners.world_model import GridWorldModel
from drone_rescue.evaluation.benchmark import generate_map_suite, run_static_benchmark
from drone_rescue.visualization.replay import render_grid, render_path
from drone_rescue.planners.astar import astar
from drone_rescue.planners.dijkstra import dijkstra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--obstacles", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    maps = generate_map_suite(args.episodes, args.size, args.obstacles, args.seed)
    results = run_static_benchmark(maps)

    output_dir = Path("results/static_v0")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [result.__dict__ | {"planner": result.planner} for result in results]
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def replay_example(episodes: int = 10, size: int = 8, obstacles: float = 0.2, seed: int = 42):
    maps = generate_static_map(
        size,
        obstacle_probability=obstacles,
        seed=seed,
    )
    model = GridWorldModel(maps.obstacles)
    path, cost, _ = astar(model, maps.start, maps.rescue)
    print(render_grid(maps.obstacles, maps.start, maps.rescue))
    print()
    print(render_path(maps.obstacles, path, maps.start, maps.rescue))
    return {"map": maps, "path": path, "cost": cost}


if __name__ == "__main__":
    main()
