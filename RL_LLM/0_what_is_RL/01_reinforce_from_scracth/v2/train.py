import torch

from env import TinyEnv
from policy import Policy


GAMMA = 1.0
LEARNING_RATE = 0.01
EPISODES = 1000


env = TinyEnv()

policy = Policy()

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=LEARNING_RATE,
)


for episode in range(EPISODES):

    state = env.reset()

    state_tensor = torch.tensor(
        [state],
        dtype=torch.float32,
    )

    action, log_prob = policy.get_action(
        state_tensor
    )

    next_state, reward, done = env.step(
        action.item()
    )

    # One-step episode, so:
    return_ = reward

    loss = -return_ * log_prob

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (episode + 1) % 100 == 0:

        with torch.no_grad():

            state_tensor = torch.tensor(
                [0.0],
                dtype=torch.float32,
            )

            logits = policy(state_tensor)

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            print(
                f"Episode {episode + 1:4d} | "
                f"P(action 0) = {probabilities[0].item():.3f} | "
                f"P(action 1) = {probabilities[1].item():.3f}"
            )