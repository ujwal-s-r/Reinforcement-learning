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
    episodes: int = 2000
    size: int = 10
    obstacle_probability: float = 0.2
    seed: int = 42
    learning_rate: float = 2.5e-4
    shaping_scale: float = 1.0
    max_steps: int | None = 120
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_frequency: int = 250
    updates_per_step: float = 1.0
    evaluation_interval: int = 40
    evaluation_episodes: int = 15
    exploration_start: float = 1.0
    exploration_end: float = 0.08
    exploration_duration: int = 50_000
    curriculum: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_dqn(config: TrainingConfig):
    print(f"Training on device: {config.device} (CUDA available: {torch.cuda.is_available()})")
    rng = np.random.default_rng(config.seed)
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
        exploration_start=config.exploration_start,
        exploration_end=config.exploration_end,
        exploration_duration=config.exploration_duration,
    )
    history = []

    for episode in range(config.episodes):
        # Curriculum: start with smaller grid & fewer obstacles, progress to full size
        if config.curriculum:
            if episode < int(0.25 * config.episodes):
                cur_size = max(5, config.size - 4)
                cur_obs_prob = max(0.08, config.obstacle_probability * 0.5)
            elif episode < int(0.50 * config.episodes):
                cur_size = max(6, config.size - 2)
                cur_obs_prob = max(0.12, config.obstacle_probability * 0.75)
            else:
                cur_size = config.size
                cur_obs_prob = config.obstacle_probability
        else:
            cur_size = config.size
            cur_obs_prob = config.obstacle_probability

        generated_map = generate_static_map(
            cur_size,
            obstacle_probability=cur_obs_prob,
            seed=int(rng.integers(1_000_000)),
        )
        environment = GridWorldEnv(
            generated_map.obstacles,
            generated_map.start,
            generated_map.rescue,
            max_steps=config.max_steps,
        )
        state = encode_observation(environment.reset().observation, size=config.size)
        previous_distance = float(np.linalg.norm(np.array(environment.start) - environment.rescue))
        episode_reward = 0.0
        losses = []
        done = False
        while not done:
            action = agent.select_action(state)
            result = environment.step(action)
            next_state = encode_observation(result.observation, size=config.size)
            current_distance = float(np.linalg.norm(np.array(environment._drone) - environment.rescue))
            
            # True potential-based shaping: gamma * Phi(s') - Phi(s) = previous_distance - gamma * current_distance
            shaped_reward = (
                result.reward + config.shaping_scale * (previous_distance - config.gamma * current_distance)
            )
            transition_done = result.terminated or result.truncated
            agent.remember(state, action, shaped_reward, next_state, transition_done)
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            state = next_state
            episode_reward += result.reward
            previous_distance = current_distance
            done = transition_done
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
        environment = GridWorldEnv(
            generated_map.obstacles,
            generated_map.start,
            generated_map.rescue,
            max_steps=config.max_steps,
        )
        observation = encode_observation(environment.reset().observation, size=config.size)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(observation, greedy=True)
            result = environment.step(action)
            observation = encode_observation(result.observation, size=config.size)
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
