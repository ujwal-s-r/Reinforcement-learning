from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_rescue.training.train_dqn import TrainingConfig, train_dqn


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Drone Rescue (v2 fixed)")
    parser.add_argument("--episodes",      type=int,   default=3000)
    parser.add_argument("--size",          type=int,   default=10)
    parser.add_argument("--obstacles",     type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--lr",            type=float, default=2.5e-4)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--eval-interval", type=int,   default=50)
    parser.add_argument("--eval-episodes", type=int,   default=20)
    parser.add_argument("--device",        type=str,   default="auto")
    return parser.parse_args()


def plot_and_save(history, out_dir):
    eval_rows = [r for r in history if "success_rate" in r]
    if not eval_rows:
        print("No eval records - skipping plot.")
        return
    episodes   = [r["episode"]        for r in eval_rows]
    success    = [r["success_rate"]   for r in eval_rows]
    avg_reward = [r["average_reward"] for r in eval_rows]
    col_rate   = [r.get("collision_rate", float("nan")) for r in eval_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("DQN Drone Rescue - Training Curves (v2 fixed)", fontsize=13)
    axes[0].plot(episodes, success,    marker="o", markersize=3)
    axes[0].set(xlabel="Episode", ylabel="Success Rate", ylim=(-0.05, 1.05), title="Greedy Success Rate")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(episodes, avg_reward, marker="o", markersize=3, color="tab:orange")
    axes[1].set(xlabel="Episode", ylabel="Avg Episode Reward", title="Greedy Average Reward")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(episodes, col_rate,   marker="o", markersize=3, color="tab:red")
    axes[2].set(xlabel="Episode", ylabel="Collision Rate", ylim=(-0.05, 1.05), title="Greedy Collision Rate")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "training_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> Plot saved to {p}")


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    print("\n[Smoke test - 10 episodes]")
    smoke_cfg = TrainingConfig(episodes=10, evaluation_interval=5, evaluation_episodes=3, device=device)
    train_dqn(smoke_cfg)
    print("Smoke test passed.\n")

    config = TrainingConfig(
        episodes=args.episodes,
        size=args.size,
        obstacle_probability=args.obstacles,
        seed=args.seed,
        learning_rate=args.lr,
        shaping_scale=args.shaping_scale,
        batch_size=args.batch_size,
        evaluation_interval=args.eval_interval,
        evaluation_episodes=args.eval_episodes,
        device=device,
    )

    print(f"[Full training - {config.episodes} episodes]")
    agent, history = train_dqn(config)

    out_dir = Path("results") / "dqn_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {metrics_path}")
    model_path = out_dir / "model.pt"
    torch.save(agent.online_network.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    plot_and_save(history, out_dir)


if __name__ == "__main__":
    main()
