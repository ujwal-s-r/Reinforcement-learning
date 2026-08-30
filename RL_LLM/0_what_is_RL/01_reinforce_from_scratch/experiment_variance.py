'''File 5: experiment_variance.py (Full Training & Empirical Comparison)
This script trains two identical TinyLM instances from the same random seed:
Run A: Raw REINFORCE (No Baseline)
Run B: REINFORCE with Baseline ($A = G - \bar{G}$)It records gradient norms and step 
rewards to demonstrate the variance reduction theorem directly.'''
import torch
import copy
from model import TinyLM
from environment import SequenceMatchingEnv
from rollout_engine import collect_rollout_batch
from reinforce import compute_advantages, compute_policy_gradient_loss

def train_agent(use_baseline: bool, num_epochs: int = 150, batch_size: int = 32, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    env = SequenceMatchingEnv(vocab_size=8, seq_len=4)
    model = TinyLM(vocab_size=8, embed_dim=32, n_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    reward_history = []
    grad_variance_history = []
    
    print(f"=== Starting Training: use_baseline={use_baseline} ===")
    
    for epoch in range(num_epochs):
        # 1. Rollout phase (Inference mode)
        rollout = collect_rollout_batch(model, env, batch_size=batch_size, temperature=1.0)
        
        # 2. Advantage calculation
        advantages = compute_advantages(rollout["rewards"], use_baseline=use_baseline)
        
        # 3. Policy loss calculation (Requires grad)
        model.train()
        optimizer.zero_grad()
        loss = compute_policy_gradient_loss(
            model=model,
            trajectories=rollout["trajectories"],
            actions=rollout["actions"],
            advantages=advantages
        )
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Measure gradient magnitude across all parameters
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
        # Telemetry
        mean_reward = rollout["rewards"].mean().item()
        reward_history.append(mean_reward)
        grad_variance_history.append(total_grad_norm)
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:03d} | Mean Reward: {mean_reward:.4f} | Grad Norm: {total_grad_norm:.4f}")
            
    return {
        "rewards": reward_history,
        "grad_norms": grad_variance_history,
        "final_model": model
    }

if __name__ == "__main__":
    # Run comparative experiment
    results_raw = train_agent(use_baseline=False)
    results_baseline = train_agent(use_baseline=True)
    
    # Print variance statistics
    grad_norm_var_raw = torch.tensor(results_raw["grad_norms"]).var().item()
    grad_norm_var_base = torch.tensor(results_baseline["grad_norms"]).var().item()
    
    print("\n--- Empirical Results ---")
    print(f"Gradient Norm Variance (Raw REINFORCE):  {grad_norm_var_raw:.6f}")
    print(f"Gradient Norm Variance (With Baseline):  {grad_norm_var_base:.6f}")
    print(f"Variance Reduction Factor:               {grad_norm_var_raw / (grad_norm_var_base + 1e-8):.2f}x")