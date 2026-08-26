import torch


class RolloutCollector:
    def __init__(self, env, actor, critic, device="cpu"):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = device

    @torch.no_grad()
    def collect(self, buffer, num_steps):
        state, _ = self.env.reset()

        episode_reward = 0.0
        episode_rewards = []

        for _ in range(num_steps):
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            )

            action, log_prob = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)

            next_state, reward, terminated, truncated, _ = (
                self.env.step(action.item())
            )

            done = terminated or truncated

            buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            )

            episode_reward += reward
            state = next_state

            if done:
                episode_rewards.append(episode_reward)

                state, _ = self.env.reset()
                episode_reward = 0.0

        return state, episode_rewards
    
    @torch.no_grad()
    def get_value(self, state):
        if state is None:
            return 0.0
        
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )
        
        if state_tensor.dim() == 0:  # Handle scalar tensors
            state_tensor = state_tensor.unsqueeze(0)

        value = self.critic(state_tensor)
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() > 0 else 0.0
        return 0.0