#uv run python step3_tiny_llama_parts.py

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    # Model width: each token becomes a vector with this many numbers.
    dim: int = 256

    # Keep the model shallow for CPU learning.
    n_layers: int = 4

    # Number of attention heads. dim must divide evenly by n_heads.
    n_heads: int = 8

    # Number of key/value heads for grouped-query attention later.
    n_kv_heads: int = 4

    # Character tokenizer vocab is tiny, but leave room for later data.
    vocab_size: int = 128

    # Inner size used by the MLP. None means calculate it from dim.
    hidden_dim: int | None = None

    # Round the MLP hidden size to a clean multiple.
    multiple_of: int = 32

    # Maximum number of tokens the model can look at at once.
    max_seq_len: int = 128

    # Small value that prevents division by zero in normalization.
    norm_eps: float = 1e-5

    # Dropout is useful during training. We can set it to 0 for first tests.
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        # eps keeps the denominator from becoming zero.
        self.eps = eps

        # One learnable scale value for each feature in the token vector.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean square over the last dimension: token vector -> one scale value.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt means 1 / sqrt(...). This normalizes the vector size.
        normalized = x * torch.rsqrt(mean_square + self.eps)

        # Apply the learnable scale.
        return normalized * self.weight


def main() -> None:
    args = ModelConfig()
    norm = RMSNorm(args.dim, args.norm_eps)

    # Batch size 1, sequence length 8, token vector size args.dim.
    x = torch.randn(1, 8, args.dim)
    y = norm(x)

    print(f"model dim: {args.dim}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"shape unchanged: {x.shape == y.shape}")
    print(f"learnable params: {sum(p.numel() for p in norm.parameters())}")


if __name__ == "__main__":
    main()
