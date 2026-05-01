from pathlib import Path

import torch
import torch.nn as nn

from step1_tokenizer import CharTokenizer
from step3_tiny_llama_parts import ModelConfig, RMSNorm
from step4_rope import precompute_freqs_cis
from step7_decoder_layer import DecoderLayer


class TinyTransformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args

        # Token embedding table: token id -> vector.
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # Repeat the decoder layer n_layers times.
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

        # Final cleanup before predicting next token.
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # Output layer: vector -> score for each token in the vocabulary.
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Reuse the input embedding table as the output token table.
        # This reduces params and is common in small language models.
        self.output.weight = self.tok_embeddings.weight

        head_dim = args.dim // args.n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, args.max_seq_len)

        # Buffers move with the model but are not trainable parameters.
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        if seq_len > self.args.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.args.max_seq_len}")

        # Convert token ids into token vectors.
        h = self.tok_embeddings(tokens)

        # Use only the RoPE positions needed for this sequence length.
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        # Run through each Transformer decoder layer.
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # Convert final vectors into vocabulary scores.
        h = self.norm(h)
        logits = self.output(h)
        return logits


def main() -> None:
    data_path = Path("data/tiny_text.txt")
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    args = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = TinyTransformer(args)

    # Fake one-batch prompt from the first 8 dataset tokens.
    token_ids = tokenizer.encode(text[:8])
    tokens = torch.tensor([token_ids], dtype=torch.long)
    logits = model(tokens)

    print(f"input token shape: {tuple(tokens.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"expected logits shape: {(1, 8, tokenizer.vocab_size)}")
    print(f"shape ok: {logits.shape == (1, 8, tokenizer.vocab_size)}")
    print(f"total params: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
