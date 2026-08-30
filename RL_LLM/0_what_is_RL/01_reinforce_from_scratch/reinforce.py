'''File 4: reinforce.py (Advantage Calculus & Training Step)This file contains the core 
tensor operations for policy updates. 
We support both mode configurations:
Raw REINFORCE: Uses raw returns $G_t$ directly ($A_t = G_t$).
REINFORCE with Batch Baseline: Subtracts the batch average return ($A_t = G_t - \bar{G}$), 
centering advantages around zero to reduce gradient variance without introducing bias.  '''
import torch
import torch.nn.functional as F
from model import TinyLM
from environment import SequenceMatchingEnv
from rollout_engine import collect_rollout_batch

def compute_advantages(rewards: torch.Tensor, use_baseline: bool = True) -> torch.Tensor:
    """
    Computes scalar advantages for each trajectory in the batch.
    
    Inputs:
        rewards: Shape [B] (scalar reward per trajectory from verifier)
        use_baseline: bool (whether to subtract batch mean return)
        
    Outputs:
        advantages: Shape [B]
    """
    if not use_baseline:
        # Raw REINFORCE: Advantage is directly the return G
        return rewards.clone()
    
    # Baseline Subtraction: b = mean(G) across batch
    baseline = rewards.mean()
    advantages = rewards - baseline
    return advantages

def compute_policy_gradient_loss(
    model: TinyLM,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor
) -> torch.Tensor:
    """
    Computes the standard Policy Gradient surrogate loss:
    L(theta) = - mean_batch ( sum_t ( log pi_theta(a_t | s_t) * A ) )
    
    Inputs:
        model: TinyLM policy network
        trajectories: Shape [B, T_total] (full sequence: prompt + actions)
        actions:      Shape [B, T_actions] (generated token IDs)
        advantages:   Shape [B] (scalar advantage per sequence)
    """
    B, T_actions = actions.shape
    
    # 1. Forward pass over the full context to get logits: Shape [B, T_total, Vocab]
    # We pass all tokens up to the second-to-last token to predict next tokens
    logits = model(trajectories[:, :-1])
    
    # 2. Extract logits corresponding strictly to the action generation steps
    # Action positions are the last T_actions positions of the sequence
    action_logits = logits[:, -T_actions:, :]  # Shape: [B, T_actions, Vocab]
    
    # 3. Compute log-softmax over vocabulary dimension: Shape [B, T_actions, Vocab]
    log_probs = F.log_softmax(action_logits, dim=-1)
    
    # 4. Gather the log-probabilities of the exact actions that were sampled
    # actions.unsqueeze(-1) shape: [B, T_actions, 1]
    # selected_log_probs shape:   [B, T_actions]
    selected_log_probs = torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    
    # 5. Sum log-probabilities across the sequence length for each trajectory: Shape [B]
    trajectory_log_probs = selected_log_probs.sum(dim=1)
    
    # 6. Weight by the trajectory's scalar advantage: Shape [B]
    # Negative sign because PyTorch optimizers perform gradient DESCENT (minimizing loss)
    policy_loss = -(trajectory_log_probs * advantages).mean()
    
    return policy_loss