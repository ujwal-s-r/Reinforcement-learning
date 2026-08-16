'''File 2: model.py (Minimal Autoregressive Policy Network)
A minimal Causal Self-Attention Transformer block acting as our 
Language Model policy $\pi_\theta$.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):
    """
    A minimal Causal Transformer Decoder policy.
    Maps a sequence of token IDs -> probability distribution over vocabulary at each step.
    """
    def __init__(self, vocab_size: int = 16, embed_dim: int = 64, n_heads: int = 2, max_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        # Single Transformer Decoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Final language model projection head: embed_dim -> vocab_size
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parallel Forward Pass (Training / Scoring):
        Input shape:  [batch_size, seq_len]
        Output shape: [batch_size, seq_len, vocab_size] (Logits for all positions)
        """
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        
        # Embed tokens and add positional encodings
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        # Construct Causal Attention Mask (Prevents attending to future tokens)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        
        # Forward through Transformer
        hidden = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Project hidden states to vocabulary logits
        logits = self.lm_head(hidden)
        return logits