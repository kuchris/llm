import argparse

import torch

from hf_tokenizer import HFTokenizer
from step13_bpe_tokenizer import BPETokenizer
from step9_train_tiny_transformer import PRESETS, build_model_config
from step8_tiny_transformer import TinyTransformer


def fmt_bytes(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen_shape_scratch_pretrain")
    args = parser.parse_args()

    defaults = PRESETS[args.preset]
    tokenizer_type = defaults["tokenizer"]
    if tokenizer_type == "hf":
        tokenizer = HFTokenizer(defaults["tokenizer_name"])
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer.from_file(defaults["tokenizer_file"])
    else:
        raise ValueError(f"unsupported tokenizer for estimator: {tokenizer_type}")

    model_args = build_model_config(defaults, tokenizer.vocab_size, defaults["block_size"])
    with torch.device("meta"):
        model = TinyTransformer(model_args)

    params = sum(parameter.numel() for parameter in model.parameters())
    fp32_weights = params * 4
    bf16_weights = params * 2
    adamw_training = params * 16

    print(f"preset: {args.preset}")
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"config: {model_args}")
    print(f"parameters: {params:,} ({params / 1_000_000:.1f}M)")
    print(f"weights fp32 only: {fmt_bytes(fp32_weights)}")
    print(f"weights bf16 only: {fmt_bytes(bf16_weights)}")
    print(f"rough AdamW fp32 training memory before activations: {fmt_bytes(adamw_training)}")
    print("note: activations, CUDA kernels, and dataloader tensors add more memory.")


if __name__ == "__main__":
    main()
