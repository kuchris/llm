import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL_PATH = "models/qwen3-0.6b"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_params(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def count_safetensor_params(model_path: Path) -> tuple[int, Counter[str], list[tuple[str, tuple[int, ...], str, int]]]:
    total = 0
    by_top_level: Counter[str] = Counter()
    tensor_rows: list[tuple[str, tuple[int, ...], str, int]] = []

    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                params = tensor.numel()
                total += params
                by_top_level[key.split(".", 1)[0]] += params
                tensor_rows.append((key, tuple(tensor.shape), str(tensor.dtype), params))

    return total, by_top_level, tensor_rows


def print_config_summary(config) -> None:
    print("== Config ==")
    fields = [
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "hidden_act",
        "tie_word_embeddings",
        "torch_dtype",
    ]
    for field in fields:
        print(f"{field}: {getattr(config, field, None)}")
    print()


def print_tokenizer_summary(model_path: Path, limit: int) -> None:
    print("== Tokenizer ==")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"class: {tokenizer.__class__.__name__}")
    print(f"vocab size: {fmt_int(len(tokenizer))}")
    print(f"bos token: {tokenizer.bos_token!r} ({tokenizer.bos_token_id})")
    print(f"eos token: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"pad token: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")

    special_tokens = tokenizer.special_tokens_map
    print("special tokens:")
    for name, value in special_tokens.items():
        print(f"  {name}: {value}")

    vocab = tokenizer.get_vocab()
    first_tokens = sorted(vocab.items(), key=lambda item: item[1])[:limit]
    print(f"first {limit} vocab tokens:")
    for token, token_id in first_tokens:
        print(f"  {token_id:6d}: {token!r}")

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        compact = " ".join(chat_template.split())
        print(f"chat template preview: {compact[:500]}")
    print()


def print_weight_summary(model_path: Path, limit: int) -> None:
    print("== Weights ==")
    total, by_top_level, tensor_rows = count_safetensor_params(model_path)
    print(f"total parameters from safetensors: {fmt_int(total)} ({fmt_params(total)})")
    print("parameters by top-level module:")
    for name, value in by_top_level.most_common():
        print(f"  {name}: {fmt_int(value)} ({fmt_params(value)})")
    print()

    print(f"first {limit} tensors:")
    for name, shape, dtype, params in tensor_rows[:limit]:
        print(f"  {name}: shape={shape}, dtype={dtype}, params={fmt_params(params)}")
    print()


def print_layer_summary(model_path: Path) -> None:
    print("== Decoder Layer Pattern ==")
    _, _, tensor_rows = count_safetensor_params(model_path)
    layer_counts: Counter[str] = Counter()

    for name, _, _, params in tensor_rows:
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            layer_counts[".".join(parts[3:])] += params

    for name, value in layer_counts.most_common():
        print(f"  {name}: {fmt_int(value)} ({fmt_params(value)})")
    print()


def print_file_summary(model_path: Path) -> None:
    print("== Files ==")
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            print(f"{path.name}: {fmt_int(path.stat().st_size)} bytes")
        else:
            print(f"{path.name}/")
    print()


def print_raw_config(model_path: Path) -> None:
    config_path = model_path / "config.json"
    print("== Raw config.json ==")
    print(json.dumps(json.loads(config_path.read_text(encoding="utf-8")), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--raw-config", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model directory not found: {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    print_file_summary(model_path)
    print_config_summary(config)
    print_tokenizer_summary(model_path, args.limit)
    print_weight_summary(model_path, args.limit)
    print_layer_summary(model_path)

    if args.raw_config:
        print_raw_config(model_path)


if __name__ == "__main__":
    main()
