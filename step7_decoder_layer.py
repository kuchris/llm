import torch
import torch.nn as nn

from step3_tiny_llama_parts import ModelConfig, RMSNorm
from step4_rope import precompute_freqs_cis
from step5_attention import Attention
from step6_mlp import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        # Attention is the group discussion between tokens.
        self.attention = Attention(args)

        # FeedForward is each token updating its own vector.
        self.feed_forward = FeedForward(args)

        # Normalize before attention and before MLP.
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        # First residual connection:
        # old token vectors + attention result.
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        # Second residual connection:
        # token vectors after discussion + MLP result.
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


def main() -> None:
    args = ModelConfig()
    layer = DecoderLayer(args)

    # Fake token vectors: batch size 1, sequence length 8, model dimension 128.
    x = torch.randn(1, 8, args.dim)

    head_dim = args.dim // args.n_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, args.max_seq_len)
    y = layer(x, freqs_cos[: x.shape[1]], freqs_sin[: x.shape[1]])

    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"shape unchanged: {x.shape == y.shape}")
    print(f"decoder layer params: {sum(p.numel() for p in layer.parameters())}")


if __name__ == "__main__":
    main()
