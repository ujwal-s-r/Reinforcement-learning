'''File 3: rollout_engine.py (Live Autoregressive Rollout Sampling)
This executes the active exploration loop. It samples tokens step-by-step,
records the exact log-probabilities of the actions chosen, and evaluates rewards.'''
import torch
import torch.nn.functional as F
from model import TinyLM
from environment import SequenceMatchingEnv

def collect_rollout_batch(model: TinyLM, env: SequenceMatchingEnv, batch_size: int = 16, temperature: float = 1.0):
    """
    Runs the live environment rollout loop.
    Generates tokens autoregressively, records log_probs of sampled actions.
    """
    model.eval()
    
    # 1. Reset environment to get initial prompt: [batch_size, 1] containing <START> token
    current_tokens = env.reset(batch_size=batch_size)
    
    # Storage lists for the trajectory
    action_log_probs = []  # List of tensors of shape [batch_size] for each step
    actions_taken = []     # List of generated token IDs
    
    # 2. Autoregressive Generation Loop
    with torch.no_grad():
        for step in range(env.seq_len):
            # Forward pass: shape [batch_size, current_len, vocab_size]
            logits = model(current_tokens)
            
            # Extract logits at the VERY LAST position: shape [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :] / temperature
            
            # Convert logits to probability distribution: shape [batch_size, vocab_size]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Categorical distribution for sampling actions
            dist = torch.distributions.Categorical(probs)
            
            # Sample an action (token ID) for each sequence in batch: shape [batch_size]
            action = dist.sample()
            
            # Record log_prob of the chosen action: log(P(action | context))
            log_prob = dist.log_prob(action)
            
            # Store values
            action_log_probs.append(log_prob)
            actions_taken.append(action)
            
            # Append sampled token to current sequence for next step input: shape [batch_size, current_len + 1]
            current_tokens = torch.cat([current_tokens, action.unsqueeze(1)], dim=1)
            
    # Stack lists into dense tensors
    # actions:   [batch_size, seq_len] (The generated completion)
    # log_probs: [batch_size, seq_len] (Log-prob for each generated action)
    actions = torch.stack(actions_taken, dim=1)
    log_probs = torch.stack(action_log_probs, dim=1)
    
    # 3. Environment Verifier Evaluation
    rewards = env.compute_reward(actions)  # Shape: [batch_size]
    
    return {
        "trajectories": current_tokens,  # Full input (prompt + actions)
        "actions": actions,              # Only the generated tokens
        "log_probs": log_probs,          # Log probs of generated tokens
        "rewards": rewards               # Scalar rewards per trajectory
    }

if __name__ == "__main__":
    # Quick sanity check run
    torch.manual_seed(42)
    env = SequenceMatchingEnv(vocab_size=16, seq_len=4)
    model = TinyLM(vocab_size=16)
    
    rollout = collect_rollout_batch(model, env, batch_size=4)
    
    print("Target Sequence to find:", env.target_sequence.tolist())
    print("\n--- Rollout Batch (4 Samples) ---")
    print("Generated Actions shape:", rollout["actions"].shape)
    print("Generated Action IDs:\n", rollout["actions"])
    print("Log Probs shape:        ", rollout["log_probs"].shape)
    print("Rewards per sample:     ", rollout["rewards"].tolist())