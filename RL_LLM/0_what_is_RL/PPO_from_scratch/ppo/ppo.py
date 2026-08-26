import torch
import torch.nn.functional as F


class PPO:
    def __init__(
        self,
        actor,
        critic,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
    ):
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.clip_eps = clip_eps

    def update_batch(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
        returns,
    ):
        # -------------------------
        # 1. Current actor
        # -------------------------

        new_log_probs = self.actor.get_log_prob(
            states,
            actions,
        )

        # -------------------------
        # 2. PPO ratio
        # -------------------------

        ratios = torch.exp(
            new_log_probs - old_log_probs
        )
        clip_mask = (
        (ratios < 1.0 - self.clip_eps)
        | (ratios > 1.0 + self.clip_eps)
        )

        clip_fraction = clip_mask.float().mean().item()

        # -------------------------
        # 3. PPO clipped objective
        # -------------------------

        unclipped = ratios * advantages

        clipped_ratios = torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps,
        )

        clipped = clipped_ratios * advantages

        actor_loss = -torch.min(
            unclipped,
            clipped,
        ).mean()

        # -------------------------
        # 4. Update actor
        # -------------------------

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # -------------------------
        # 5. Current critic
        # -------------------------

        values = self.critic(states)

        # -------------------------
        # 6. Critic loss
        # -------------------------

        critic_loss = F.mse_loss(
            values,
            returns,
        )

        # -------------------------
        # 7. Update critic
        # -------------------------

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "ratio_mean": ratios.mean().item(),
            "ratio_min": ratios.min().item(),
            "ratio_max": ratios.max().item(),
            "clip_fraction": clip_fraction,
        }
        
    def update(
    self,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    batch_size=256,
    epochs=4,
        ):
            dataset_size = len(states)

            metrics = {
                "actor_loss": [],
                "critic_loss": [],
                "ratio_mean": [],
                "ratio_min": [],
                "ratio_max": [],
                "clip_fraction": [],
            }

            for epoch in range(epochs):
            
                # Shuffle the rollout indices.
                indices = torch.randperm(dataset_size)

                for start in range(0, dataset_size, batch_size):
                
                    batch_indices = indices[start:start + batch_size]

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    batch_metrics = self.update_batch(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_advantages,
                        batch_returns,
                    )

                    for key in metrics:
                        metrics[key].append(batch_metrics[key])

            return {
                key: sum(values) / len(values)
                for key, values in metrics.items()
            }