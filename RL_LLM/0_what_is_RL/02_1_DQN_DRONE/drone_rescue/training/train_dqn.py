from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from ..agents.dqn.agent import DQNAgent
from ..environment.grid_world import GridWorldEnv
from ..environment.map_generator import generate_static_map
from ..environment.observation import encode_observation


@dataclass
class TrainingConfig:
    episodes: int = 500
    size: int = 10
    obstacle_probability: float = 0.2
    seed: int = 42
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    target_update_frequency: int = 500
    updates_per_step: float = 1.0
    evaluation_interval: int = 25
    evaluation_episodes: int = 10
    device: str = "cpu"


def train_dqn(config: TrainingConfig):
    rng = np.random.default_rng(config.seed)
    first_map = generate_static_map(
        config.size,
        obstacle_probability=config.obstacle_probability,
        seed=config.seed,
    )
    observation_shape = (3, config.size, config.size)
    agent = DQNAgent(
        observation_shape,
        actions=4,
        device=config.device,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_frequency=config.target_update_frequency,
    )
    history = []

    for episode in range(config.episodes):
        generated_map = generate_static_map(
            config.size,
            obstacle_probability=config.obstacle_probability,
            seed=int(rng.integers(1_000_000)),
        )
        environment = GridWorldEnv(generated_map.obstacles, generated_map.start, generated_map.rescue)
        state = encode_observation(environment.reset().observation)
        episode_reward = 0.0
        losses = []
        done = False
        while not done:
            action = agent.select_action(state)
            result = environment.step(action)
            next_state = encode_observation(result.observation)
            agent.remember(state, action, result.reward, next_state, result.terminated or result.truncated)
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            state = next_state
            episode_reward += result.reward
            done = result.terminated or result.truncated
        agent.exploration.step()

        record = {
            "episode": episode + 1,
            "reward": episode_reward,
            "epsilon": agent.exploration.value,
            "loss": float(np.mean(losses)) if losses else None,
            "success": bool(result.info["success"]),
        }
        history.append(record)
        if (episode + 1) % config.evaluation_interval == 0:
            metrics = evaluate_dqn(agent, config)
            history[-1].update(metrics)
            print(
                f"episode {episode + 1:04d} | reward {metrics['average_reward']:8.1f} | "
                f"success {metrics['success_rate']:.2f} | epsilon {agent.exploration.value:.3f}"
            )
    return agent, history


def evaluate_dqn(agent: DQNAgent, config: TrainingConfig) -> dict[str, float]:
    rewards = []
    successes = []
    path_lengths = []
    collisions = []
    for offset in range(config.evaluation_episodes):
        generated_map = generate_static_map(
            config.size,
            obstacle_probability=config.obstacle_probability,
            seed=900_000 + offset,
        )
        environment = GridWorldEnv(generated_map.obstacles, generated_map.start, generated_map.rescue)
        observation = encode_observation(environment.reset().observation)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(observation, greedy=True)
            result = environment.step(action)
            observation = encode_observation(result.observation)
            total_reward += result.reward
            collisions.append(bool(result.info["collision"]))
            done = result.terminated or result.truncated
        rewards.append(total_reward)
        successes.append(bool(result.info["success"]))
        path_lengths.append(int(result.info["path_length"]))
    return {
        "average_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "average_path_length": float(np.mean(path_lengths)),
        "collision_rate": float(np.mean(collisions)),
    }
