import numpy as np

from drone_rescue.agents.dqn.agent import DQNAgent


def test_dqn_agent_can_learn_from_one_transition():
    agent = DQNAgent((3, 4, 4), actions=4, batch_size=1, buffer_size=2)
    observation = np.random.default_rng(0).random((3, 4, 4), dtype=np.float32)
    next_observation = np.random.default_rng(1).random((3, 4, 4), dtype=np.float32)
    agent.remember(observation, 0, 1.0, next_observation, True)
    assert isinstance(agent.train_step(), float)
