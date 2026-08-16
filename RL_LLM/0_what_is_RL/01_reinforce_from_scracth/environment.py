'''File 1: environment.py (The Sequence Task & Reward Verifier)
This simulates our environment: given a prompt token (e.g., <START>), 
the model must learn to generate a target sequence of tokens (e.g., [3, 7, 2, <EOS>]).
If it matches, reward is $+1.0$; otherwise, $0.0$.'''
import torch

class SequenceMatchingEnv:
    """
    A sequence generation environment with dense matching rewards.
    """
    def __init__(self, vocab_size: int = 8, seq_len: int = 4):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.start_token = 1
        self.eos_token = 2
        
        # Ground truth target sequence: e.g. [3, 7, 5, 2]
        self.target_sequence = torch.tensor([3, 7, 5, self.eos_token], dtype=torch.long)

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        return torch.full((batch_size, 1), self.start_token, dtype=torch.long)

    def compute_reward(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        """
        Dense Reward Signal:
        Gives fractional credit for every token matched in the correct position.
        e.g., 2 correct tokens = 0.5, all 4 correct = 1.0.
        """
        batch_size = generated_tokens.shape[0]
        target_expanded = self.target_sequence.unsqueeze(0).expand(batch_size, -1)
        
        # Count token matches per sequence: shape [batch_size]
        matching_tokens = (generated_tokens == target_expanded).float().sum(dim=1)
        
        # Scale to [0.0, 1.0] range
        rewards = matching_tokens / self.seq_len
        return rewards