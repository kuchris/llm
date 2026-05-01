#uv run python step4_rope.py

import torch

from step3_tiny_llama_parts import ModelConfig


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    # RoPE works on pairs of numbers, so we create dim / 2 frequencies.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Positions are 0, 1, 2, ... up to the maximum sequence length.
    positions = torch.arange(seq_len)

    # Each position gets a rotation angle for each frequency.
    angles = torch.outer(positions, freqs)

    # Store cosine and sine separately so we can rotate real tensors.
    return torch.cos(angles), torch.sin(angles)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Split the last dimension into pairs: even values and odd values.
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    # A 2D rotation changes [a, b] into [-b, a].
    rotated = torch.stack((-x_odd, x_even), dim=-1)

    # Flatten the pairs back into the original last dimension.
    return rotated.flatten(start_dim=-2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq/xk shape: batch, seq_len, heads, head_dim.
    # freqs shape: seq_len, head_dim / 2.

    # Repeat each cos/sin value so it matches both numbers in each pair.
    cos = freqs_cos.repeat_interleave(2, dim=-1)
    sin = freqs_sin.repeat_interleave(2, dim=-1)

    # Add empty batch/head dimensions so broadcasting lines up.
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Rotate query and key using the same position information.
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


def main() -> None:
    args = ModelConfig()
    head_dim = args.dim // args.n_heads

    # Fake query/key tensors like the attention layer will create later.
    xq = torch.randn(1, 8, args.n_heads, head_dim)
    xk = torch.randn(1, 8, args.n_kv_heads, head_dim)

    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, args.max_seq_len)
    xq_out, xk_out = apply_rotary_emb(
        xq,
        xk,
        freqs_cos[: xq.shape[1]],
        freqs_sin[: xq.shape[1]],
    )

    print(f"head dim: {head_dim}")
    print(f"query input shape : {tuple(xq.shape)}")
    print(f"query output shape: {tuple(xq_out.shape)}")
    print(f"key input shape   : {tuple(xk.shape)}")
    print(f"key output shape  : {tuple(xk_out.shape)}")
    print(f"query shape unchanged: {xq.shape == xq_out.shape}")
    print(f"key shape unchanged: {xk.shape == xk_out.shape}")
    print(f"query values changed: {not torch.equal(xq, xq_out)}")


if __name__ == "__main__":
    main()
