import torch
import torch.nn as nn

from step3_tiny_llama_parts import ModelConfig, RMSNorm
from step4_rope import precompute_freqs_cis
from step7_decoder_layer import DecoderLayer


class TinyTransformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(DecoderLayer(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        if seq_len > self.args.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.args.max_seq_len}")

        x = self.tok_embeddings(tokens)
        x = self.dropout(x)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)

        x = self.norm(x)
        return self.output(x)


def main() -> None:
    args = ModelConfig()
    model = TinyTransformer(args)

    tokens = torch.randint(0, args.vocab_size, (1, 8))
    logits = model(tokens)

    print(f"token shape: {tuple(tokens.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"model params: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
