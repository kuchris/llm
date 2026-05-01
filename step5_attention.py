import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from step3_tiny_llama_parts import ModelConfig
from step4_rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x shape: batch, seq_len, kv_heads, head_dim.
    if n_rep == 1:
        return x

    batch, seq_len, kv_heads, head_dim = x.shape

    # Repeat each key/value head so it can match the number of query heads.
    x = x[:, :, :, None, :]
    x = x.expand(batch, seq_len, kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        # Split the model vector across multiple attention heads.
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
        self.n_rep = args.n_heads // args.n_kv_heads

        # Q has one head per attention head.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        # K and V can use fewer heads. This is grouped-query attention.
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)

        # Project the mixed attention output back to model dimension.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        # Build query, key, value vectors from the current token vectors.
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Add position information to Q and K only.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Repeat K/V heads so Q heads can compare against them.
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Move heads before seq_len for matrix multiplication.
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compare each query position with all key positions.
        scores = xq @ xk.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Causal mask blocks looking at future tokens.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores into attention weights.
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Mix value vectors using the attention weights.
        output = weights @ xv

        # Move back to batch, seq_len, n_heads * head_dim.
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(output)


def main() -> None:
    args = ModelConfig()
    attention = Attention(args)

    # Fake token vectors: batch size 1, sequence length 8, model dimension 128.
    x = torch.randn(1, 8, args.dim)

    freqs_cos, freqs_sin = precompute_freqs_cis(attention.head_dim, args.max_seq_len)
    y = attention(x, freqs_cos[: x.shape[1]], freqs_sin[: x.shape[1]])

    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"shape unchanged: {x.shape == y.shape}")
    print(f"attention params: {sum(p.numel() for p in attention.parameters())}")


if __name__ == "__main__":
    main()
