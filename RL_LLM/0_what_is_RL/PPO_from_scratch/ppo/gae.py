import torch


def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantages = torch.zeros_like(rewards)

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_val * next_non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma * gae_lambda * next_non_terminal * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns