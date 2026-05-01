import torch
import torch.nn as nn
import torch.nn.functional as F

from step3_tiny_llama_parts import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        # LLaMA uses an MLP hidden size around 2/3 of 4 * dim.
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            hidden_dim = int(4 * args.dim * 2 / 3)

            # Round up so the hidden size is hardware-friendly.
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # w1 creates the main transformed values.
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)

        # w3 creates a gate that controls which values pass through.
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

        # w2 projects back to the original model dimension.
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(w1(x)) is the activated main path.
        main = F.silu(self.w1(x))

        # w3(x) is the gate. Multiplying means the gate can boost or suppress values.
        gated = main * self.w3(x)

        # Return to the same shape as the input.
        return self.w2(gated)


def main() -> None:
    args = ModelConfig()
    mlp = FeedForward(args)

    # Fake token vectors: batch size 1, sequence length 8, model dimension 128.
    x = torch.randn(1, 8, args.dim)
    y = mlp(x)

    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"shape unchanged: {x.shape == y.shape}")
    print(f"mlp params: {sum(p.numel() for p in mlp.parameters())}")


if __name__ == "__main__":
    main()
