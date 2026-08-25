import torch

from env import TinyEnv
from policy import Actor, Critic


GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES = 1000


env = TinyEnv()

actor = Actor()
critic = Critic()

actor_optimizer = torch.optim.Adam(
    actor.parameters(),
    lr=LEARNING_RATE,
)

critic_optimizer = torch.optim.Adam(
    critic.parameters(),
    lr=LEARNING_RATE,
)


for episode in range(EPISODES):

    state = env.reset()

    state_tensor = torch.tensor(
        [state],
        dtype=torch.float32,
    )

    action, log_prob = actor.get_action(
        state_tensor
    )

    next_state, reward, done = env.step(
        action.item()
    )

    next_state_tensor = torch.tensor(
        [next_state],
        dtype=torch.float32,
    )

    value = critic(state_tensor)

    with torch.no_grad():
        next_value = critic(next_state_tensor)

        if done:
            target = torch.tensor(
                reward,
                dtype=torch.float32,
            )
        else:
            target = (
                reward
                + GAMMA * next_value
            )

    td_error = target - value

    # Actor
    actor_loss = (
        -log_prob * td_error.detach()
    )

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Critic
    critic_loss = td_error.pow(2)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if (episode + 1) % 100 == 0:

        with torch.no_grad():

            logits = actor(state_tensor)

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            print(
                f"Episode {episode + 1:4d} | "
                f"P(0)={probs[0].item():.3f} | "
                f"P(1)={probs[1].item():.3f} | "
                f"V={value.item():.3f} | "
                f"TD={td_error.item():.3f}"
            )