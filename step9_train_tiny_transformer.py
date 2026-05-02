#uv run python step9_train_tiny_transformer.py
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe


import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hf_tokenizer import HFTokenizer
from step1_tokenizer import CharTokenizer
from step13_bpe_tokenizer import BPETokenizer
from step3_tiny_llama_parts import ModelConfig
from step8_tiny_transformer import TinyTransformer


# Edit this when you want to switch the default training target.
DEFAULT_PRESET = "free_wikitext103_pretrain_bpe"
RESPONSE_MARKER = "### Response:\n"


def read_text_limited(path: Path, max_chars: int) -> str:
    if max_chars <= 0:
        return path.read_text(encoding="utf-8")

    chunks = []
    remaining = max_chars
    with path.open("r", encoding="utf-8") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    return "".join(chunks)


class MemmapTokenDataset:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.manifest["dtype"])
        self.shards = []
        for shard in self.manifest["shards"]:
            tokens = int(shard["tokens"])
            if tokens <= 0:
                continue
            path = self.root / shard["path"]
            data = np.memmap(path, dtype=self.dtype, mode="r", shape=(tokens,))
            self.shards.append({"path": path, "tokens": tokens, "data": data})
        if not self.shards:
            raise ValueError(f"no usable shards found in {manifest_path}")

    @property
    def vocab_size(self) -> int:
        return int(self.manifest["vocab_size"])

    def __len__(self) -> int:
        return sum(shard["tokens"] for shard in self.shards)


# Presets keep datasets and checkpoints separated.
PRESETS = {
    "tiny_text": {
        "data": "data/tiny_text.txt",
        "checkpoint": "checkpoints/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 32,
        "batch_size": 8,
        "max_iters": 120,
        "learning_rate": 1e-3,
        "tokenizer": "char",
        "tokenizer_file": "",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "tiny_shakespeare": {
        "data": "data/tiny_shakespeare.txt",
        "checkpoint": "checkpoints/tiny_shakespeare/tiny_transformer.pt",
        "max_chars": 500000,
        "block_size": 64,
        "batch_size": 8,
        "max_iters": 2000,
        "learning_rate": 1e-3,
        "tokenizer": "char",
        "tokenizer_file": "",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "wikitext2": {
        "data": "data/wikitext2.txt",
        "checkpoint": "checkpoints/wikitext2/tiny_transformer.pt",
        "max_chars": 500000,
        "block_size": 64,
        "batch_size": 8,
        "max_iters": 2000,
        "learning_rate": 1e-3,
        "tokenizer": "char",
        "tokenizer_file": "",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "wikitext2_bpe": {
        "data": "data/wikitext2.txt",
        "checkpoint": "checkpoints/wikitext2_bpe/tiny_transformer.pt",
        "max_chars": 500000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/wikitext2_bpe_2000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "wikitext2_bpe_4layer": {
        "data": "data/wikitext2.txt",
        "checkpoint": "checkpoints/wikitext2_bpe_4layer/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/wikitext2_bpe_2000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "wikitext2_clean_bpe_4layer": {
        "data": "data/wikitext2_clean.txt",
        "checkpoint": "checkpoints/wikitext2_clean_bpe_4layer/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 10000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/wikitext2_clean_bpe_4000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "alpaca_bpe_4layer": {
        "data": "data/alpaca_sft.txt",
        "checkpoint": "checkpoints/alpaca_bpe_4layer/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/alpaca_bpe_4000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "assistant_pretrain_bpe_4layer": {
        "data": "data/wikitext103_clean.txt",
        "checkpoint": "checkpoints/assistant_pretrain_bpe_4layer/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/assistant_bpe_4000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "assistant_sft_bpe_4layer": {
        "data": "data/assistant_sft.txt",
        "checkpoint": "checkpoints/assistant_sft_bpe_4layer/tiny_transformer.pt",
        "max_chars": 2000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 5e-4,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/assistant_bpe_4000.json",
        "tokenizer_name": "",
        "task": "sft",
        "init_checkpoint": "checkpoints/assistant_pretrain_bpe_4layer/tiny_transformer.pt",
    },
    "assistant_eos_pretrain_bpe": {
        "data": "data/wikitext103_clean.txt",
        "checkpoint": "checkpoints/assistant_eos_pretrain_bpe/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/assistant_eos_bpe_4000.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "assistant_eos_sft_bpe": {
        "data": "data/assistant_eos_sft.txt",
        "checkpoint": "checkpoints/assistant_eos_sft_bpe/tiny_transformer.pt",
        "max_chars": 2000000,
        "block_size": 128,
        "batch_size": 8,
        "max_iters": 5000,
        "learning_rate": 5e-4,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/assistant_eos_bpe_4000.json",
        "tokenizer_name": "",
        "task": "sft",
        "init_checkpoint": "checkpoints/assistant_eos_pretrain_bpe/tiny_transformer.pt",
    },
    "free_wikitext103_pretrain_bpe": {
        "data": "data/wikitext103_clean.txt",
        "checkpoint": "checkpoints/free_wikitext103_pretrain_bpe/tiny_transformer.pt",
        "max_chars": 5000000,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 20000,
        "learning_rate": 1e-3,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/free_bpe_6144.json",
        "tokenizer_name": "",
        "task": "lm",
        "init_checkpoint": "",
    },
    "free_dolly_sft_bpe": {
        "data": "data/dolly_train_eos_sft.txt",
        "checkpoint": "checkpoints/free_dolly_sft_bpe/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 8000,
        "learning_rate": 5e-4,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/free_bpe_6144.json",
        "tokenizer_name": "",
        "task": "sft",
        "init_checkpoint": "checkpoints/free_wikitext103_pretrain_bpe/tiny_transformer.pt",
    },
    "free_bea_grammar_sft_bpe": {
        "data": "data/bea_grammar_train_eos_sft.txt",
        "checkpoint": "checkpoints/free_bea_grammar_sft_bpe/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 4000,
        "learning_rate": 3e-4,
        "tokenizer": "bpe",
        "tokenizer_file": "tokenizers/free_bpe_6144.json",
        "tokenizer_name": "",
        "task": "sft",
        "init_checkpoint": "checkpoints/free_dolly_sft_bpe/tiny_transformer.pt",
    },
    "qwen_tokenizer_tiny_pretrain": {
        "data": "data/fineweb_edu_qwen_uint32/manifest.json",
        "checkpoint": "checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 50000,
        "learning_rate": 3e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "lm_memmap",
        "init_checkpoint": "",
        "dim": 768,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.0,
    },
    "qwen_tokenizer_tiny_pretrain_sharded": {
        "data": "data/fineweb_edu_qwen_uint32/manifest.json",
        "checkpoint": "checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 50000,
        "learning_rate": 3e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "lm_memmap",
        "init_checkpoint": "",
        "dim": 768,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.0,
    },
    "qwen_tokenizer_tiny_sft": {
        "data": "data/alpaca_cleaned_train_eos_sft.txt",
        "checkpoint": "checkpoints/qwen_tokenizer_tiny_alpaca_sft/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 13000,
        "learning_rate": 1e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "sft",
        "init_checkpoint": "checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt",
        "dim": 768,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.1,
    },
    "qwen_tokenizer_overfit": {
        "data": "data/qwen_overfit_paragraph.txt",
        "checkpoint": "checkpoints/qwen_tokenizer_overfit/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 32,
        "batch_size": 8,
        "max_iters": 200,
        "learning_rate": 1e-3,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "lm",
        "init_checkpoint": "",
        "dim": 256,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "hidden_dim": 512,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.0,
    },
    "english_bpe_tiny_pretrain": {
        "data": "data/fineweb_edu_sample_10bt.txt",
        "checkpoint": "checkpoints/english_bpe_tiny_pretrain/tiny_transformer.pt",
        "max_chars": 200000000,
        "block_size": 512,
        "batch_size": 4,
        "max_iters": 50000,
        "learning_rate": 3e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "tokenizers/english_bpe_8192",
        "task": "lm",
        "init_checkpoint": "",
        "dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 3,
        "hidden_dim": 1024,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.1,
    },
    "english_bpe_tiny_sft": {
        "data": "data/alpaca_cleaned_train_eos_sft.txt",
        "checkpoint": "checkpoints/english_bpe_tiny_alpaca_sft/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 512,
        "batch_size": 4,
        "max_iters": 10000,
        "learning_rate": 1e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "tokenizers/english_bpe_8192",
        "task": "sft",
        "init_checkpoint": "checkpoints/english_bpe_tiny_pretrain/tiny_transformer.pt",
        "dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 3,
        "hidden_dim": 1024,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.1,
    },
    "bpe32k_tiny_pretrain": {
        "data": "data/fineweb_edu_bpe32k_uint16/manifest.json",
        "checkpoint": "checkpoints/bpe32k_tiny_pretrain/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 50000,
        "learning_rate": 3e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "tokenizers/english_bpe_32768",
        "task": "lm_memmap",
        "init_checkpoint": "",
        "dim": 768,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.0,
    },
    "bpe32k_tiny_sft": {
        "data": "data/alpaca_cleaned_train_eos_sft.txt",
        "checkpoint": "checkpoints/bpe32k_tiny_alpaca_sft/tiny_transformer.pt",
        "max_chars": 0,
        "block_size": 256,
        "batch_size": 4,
        "max_iters": 13000,
        "learning_rate": 1e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "tokenizers/english_bpe_32768",
        "task": "sft",
        "init_checkpoint": "checkpoints/bpe32k_tiny_pretrain/tiny_transformer.pt",
        "dim": 768,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.1,
    },
    "qwen_shape_scratch_pretrain": {
        "data": "data/wikitext103_clean.txt",
        "checkpoint": "checkpoints/qwen_shape_scratch_pretrain/tiny_transformer.pt",
        "max_chars": 50000,
        "block_size": 64,
        "batch_size": 1,
        "max_iters": 100,
        "learning_rate": 1e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "lm",
        "init_checkpoint": "",
        "dim": 1024,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 128,
        "hidden_dim": 3072,
        "multiple_of": 64,
        "norm_eps": 1e-6,
        "dropout": 0.0,
    },
}


def build_model_config(defaults: dict, vocab_size: int, block_size: int) -> ModelConfig:
    base = ModelConfig()
    return ModelConfig(
        dim=defaults.get("dim", base.dim),
        n_layers=defaults.get("n_layers", base.n_layers),
        n_heads=defaults.get("n_heads", base.n_heads),
        n_kv_heads=defaults.get("n_kv_heads", base.n_kv_heads),
        head_dim=defaults.get("head_dim", base.head_dim),
        vocab_size=vocab_size,
        hidden_dim=defaults.get("hidden_dim", base.hidden_dim),
        multiple_of=defaults.get("multiple_of", base.multiple_of),
        max_seq_len=block_size,
        norm_eps=defaults.get("norm_eps", base.norm_eps),
        dropout=defaults.get("dropout", base.dropout),
    )


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Pick random start positions inside the token stream.
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    # x is the current text window.
    x = torch.stack([data[start : start + block_size] for start in starts])

    # y is the same text window shifted one token to the left.
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x, y


def get_batch_memmap(
    data: MemmapTokenDataset,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    usable_shards = [shard for shard in data.shards if shard["tokens"] > block_size + 1]
    if not usable_shards:
        raise ValueError("no memmap shard is long enough for the chosen block size")

    shard_indexes = torch.randint(0, len(usable_shards), (batch_size,))
    for shard_index in shard_indexes.tolist():
        shard = usable_shards[shard_index]
        start = int(torch.randint(0, shard["tokens"] - block_size - 1, (1,)).item())
        window = np.asarray(shard["data"][start : start + block_size + 1], dtype=np.int64)
        x = torch.from_numpy(window[:-1].copy()).long()
        y = torch.from_numpy(window[1:].copy()).long()
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def get_batch_sft(
    data: torch.Tensor,
    loss_mask: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Same as get_batch, but also returns which target tokens should count for loss.
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    y_mask = torch.stack([loss_mask[start + 1 : start + block_size + 1] for start in starts])
    return x, y, y_mask


def build_sft_token_stream(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = []
    loss_mask = []

    for block in text.split("### Instruction:\n"):
        block = block.strip()
        if not block or RESPONSE_MARKER not in block:
            continue

        example = "### Instruction:\n" + block + "\n\n"
        prompt, response = example.split(RESPONSE_MARKER, 1)
        prompt += RESPONSE_MARKER

        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)

        token_ids.extend(prompt_ids)
        loss_mask.extend([0] * len(prompt_ids))
        token_ids.extend(response_ids)
        loss_mask.extend([1] * len(response_ids))

    if not token_ids:
        raise ValueError("no SFT examples found; expected text containing '### Response:'")

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(loss_mask, dtype=torch.float32)


def build_checkpoint(
    *,
    model: TinyTransformer,
    model_args: ModelConfig,
    optimizer: torch.optim.Optimizer,
    tokenizer_type: str,
    tokenizer_file: str,
    tokenizer,
    task: str,
    preset: str,
    data_path: Path,
    step: int,
    max_iters: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    first_loss: float | None,
    last_loss: float | None,
) -> dict:
    checkpoint = {
        "model": model.state_dict(),
        "config": model_args,
        "optimizer": optimizer.state_dict(),
        "tokenizer_type": tokenizer_type,
        "task": task,
        "preset": preset,
        "data_path": str(data_path),
        "step": step,
        "max_iters": max_iters,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }
    if tokenizer_type == "char":
        checkpoint["vocab"] = tokenizer.stoi
    elif tokenizer_type == "bpe":
        checkpoint["tokenizer_json"] = Path(tokenizer_file).read_text(encoding="utf-8")
    elif tokenizer_type == "hf":
        checkpoint["tokenizer_name"] = tokenizer.name_or_path
    return checkpoint


def save_checkpoint(checkpoint_path: Path, checkpoint: dict, label: str) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"{label}: {checkpoint_path}")


def write_loss_log(log_path: Path, step: int, loss: float, learning_rate: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("step,loss,learning_rate\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{step},{loss:.8f},{learning_rate:.10g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--tokenizer", choices=["char", "bpe", "hf"], default=None)
    parser.add_argument("--tokenizer-file", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--task", choices=["lm", "sft", "lm_memmap"], default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    cli_args = parser.parse_args()
    defaults = PRESETS[cli_args.preset]

    torch.manual_seed(1)
    if cli_args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cli_args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch install has no CUDA support. Reinstall CUDA-enabled torch or use --device cpu.")
    print(f"device: {device}")
    if device == "cuda":
        # Skip per-shape kernel benchmarking — shapes are fixed so autotuning buys nothing
        # and causes a multi-minute hang on the first step with unusual sizes (e.g. 151k vocab).
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"amp: {amp_dtype}, grad_scaler: {scaler.is_enabled()}")

    # Load and tokenize the dataset.
    data_path = Path(cli_args.data or defaults["data"])
    max_chars = cli_args.max_chars if cli_args.max_chars is not None else defaults["max_chars"]

    tokenizer_type = cli_args.tokenizer or defaults["tokenizer"]
    tokenizer_file = cli_args.tokenizer_file if cli_args.tokenizer_file is not None else defaults["tokenizer_file"]
    tokenizer_name = cli_args.tokenizer_name if cli_args.tokenizer_name is not None else defaults["tokenizer_name"]

    task = cli_args.task or defaults["task"]
    text = "" if task == "lm_memmap" else read_text_limited(data_path, max_chars)

    if tokenizer_type == "char":
        tokenizer = CharTokenizer(text)
    elif tokenizer_type == "bpe":
        if not tokenizer_file:
            raise ValueError("--tokenizer-file is required when --tokenizer bpe")
        tokenizer = BPETokenizer.from_file(tokenizer_file)
    elif tokenizer_type == "hf":
        if not tokenizer_name:
            raise ValueError("--tokenizer-name is required when --tokenizer hf")
        tokenizer = HFTokenizer(tokenizer_name)
    else:
        raise ValueError(f"unsupported tokenizer type: {tokenizer_type}")
    if task == "sft":
        token_ids, loss_mask = build_sft_token_stream(tokenizer, text)
    elif task == "lm_memmap":
        token_ids = MemmapTokenDataset(data_path)
        if token_ids.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"memmap vocab size {token_ids.vocab_size} does not match "
                f"tokenizer vocab size {tokenizer.vocab_size}"
            )
        loss_mask = None
    else:
        token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        loss_mask = None

    # Small CPU-friendly training settings.
    block_size = cli_args.block_size or defaults["block_size"]
    batch_size = cli_args.batch_size or defaults["batch_size"]
    max_iters = cli_args.max_iters or defaults["max_iters"]
    learning_rate = cli_args.learning_rate or defaults["learning_rate"]
    checkpoint_path = Path(cli_args.checkpoint or defaults["checkpoint"])
    resume_path = checkpoint_path
    log_path = Path(cli_args.log_file) if cli_args.log_file else checkpoint_path.with_name("loss_log.csv")
    if not cli_args.resume and log_path.exists():
        log_path.unlink()

    if len(token_ids) <= block_size + 1:
        raise ValueError("dataset is too short for the chosen block size")

    # Build the Transformer using the selected tokenizer vocabulary size.
    model_args = build_model_config(defaults, tokenizer.vocab_size, block_size)
    model = TinyTransformer(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Warmup for 500 steps then cosine decay to 10% of peak LR.
    warmup_steps = min(500, max_iters // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_iters - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    first_loss = None
    last_loss = None
    start_step = 0
    resumed = False

    if cli_args.resume and resume_path.exists():
        resume_data = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_data["model"])
        if "optimizer" in resume_data:
            optimizer.load_state_dict(resume_data["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        first_loss = resume_data.get("first_loss")
        last_loss = resume_data.get("last_loss")
        start_step = int(resume_data.get("step", -1)) + 1
        resumed = True
        print(f"resumed checkpoint: {resume_path}")
        print(f"resume step: {start_step}")
    else:
        init_checkpoint = cli_args.init_checkpoint if cli_args.init_checkpoint is not None else defaults["init_checkpoint"]
        if init_checkpoint:
            init_path = Path(init_checkpoint)
            if init_path.exists():
                init_data = torch.load(init_path, map_location="cpu", weights_only=False)
                model.load_state_dict(init_data["model"])
                print(f"loaded init checkpoint: {init_path}")
            else:
                print(f"init checkpoint not found, training from scratch: {init_path}")

    if start_step >= max_iters:
        print(f"checkpoint already reached step {start_step - 1}, which is >= max_iters {max_iters}")
        return

    # LambdaLR with last_epoch >= 0 requires initial_lr in param_groups (normally set
    # on first scheduler.step()). Set it explicitly so resume works correctly.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

    if device == "cuda" and sys.platform != "win32":
        model = torch.compile(model)
        print("torch.compile enabled (first step will compile — expect ~1 min delay)")
    elif device == "cuda":
        print("torch.compile skipped (Triton not supported on Windows)")

    model.train()
    use_tqdm = sys.stderr.isatty()
    progress_steps = range(start_step, max_iters + 1)
    progress = tqdm(progress_steps, desc="training", unit="step") if use_tqdm else progress_steps
    last_completed_step = start_step - 1

    try:
        for step in progress:
            if task == "sft":
                x, y, y_mask = get_batch_sft(token_ids, loss_mask, batch_size, block_size)
            elif task == "lm_memmap":
                x, y = get_batch_memmap(token_ids, batch_size, block_size)
                y_mask = None
            else:
                x, y = get_batch(token_ids, batch_size, block_size)
                y_mask = None
            x = x.to(device)
            y = y.to(device)
            if y_mask is not None:
                y_mask = y_mask.to(device)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                # logits shape: batch, block_size, vocab_size.
                logits = model(x)

                # Cross entropy compares predicted next-token scores with target token ids.
                if task == "sft":
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, tokenizer.vocab_size),
                        y.reshape(-1),
                        reduction="none",
                    )
                    y_mask = y_mask.reshape(-1)
                    loss = (loss_per_token * y_mask).sum() / y_mask.sum().clamp_min(1.0)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            last_completed_step = step

            if first_loss is None:
                first_loss = loss.item()

            if cli_args.log_every > 0 and step % cli_args.log_every == 0:
                last_loss = loss.item()
                current_lr = optimizer.param_groups[0]["lr"]
                write_loss_log(log_path, step, last_loss, current_lr)
                if use_tqdm:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
                    tqdm.write(f"step {step:3d} | loss {last_loss:.4f} | lr {current_lr:.3g}")
                else:
                    print(f"step {step:5d} / {max_iters:5d} | loss {last_loss:.4f} | lr {current_lr:.3g}")

            if cli_args.save_every > 0 and step > start_step and step % cli_args.save_every == 0:
                checkpoint = build_checkpoint(
                    model=model,
                    model_args=model_args,
                    optimizer=optimizer,
                    tokenizer_type=tokenizer_type,
                    tokenizer_file=tokenizer_file,
                    tokenizer=tokenizer,
                    task=task,
                    preset=cli_args.preset,
                    data_path=data_path,
                    step=step,
                    max_iters=max_iters,
                    block_size=block_size,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    first_loss=first_loss,
                    last_loss=last_loss,
                )
                save_checkpoint(checkpoint_path, checkpoint, "autosaved checkpoint")
    except KeyboardInterrupt:
        checkpoint = build_checkpoint(
            model=model,
            model_args=model_args,
            optimizer=optimizer,
            tokenizer_type=tokenizer_type,
            tokenizer_file=tokenizer_file,
            tokenizer=tokenizer,
            task=task,
            preset=cli_args.preset,
            data_path=data_path,
            step=last_completed_step,
            max_iters=max_iters,
            block_size=block_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            first_loss=first_loss,
            last_loss=last_loss,
        )
        save_checkpoint(checkpoint_path, checkpoint, "interrupted checkpoint saved")
        print("training interrupted")
        return

    checkpoint = build_checkpoint(
        model=model,
        model_args=model_args,
        optimizer=optimizer,
        tokenizer_type=tokenizer_type,
        tokenizer_file=tokenizer_file,
        tokenizer=tokenizer,
        task=task,
        preset=cli_args.preset,
        data_path=data_path,
        step=last_completed_step,
        max_iters=max_iters,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        first_loss=first_loss,
        last_loss=last_loss,
    )
    save_checkpoint(checkpoint_path, checkpoint, "saved checkpoint")

    print()
    print(f"first loss: {first_loss:.4f}")
    print(f"last printed loss: {last_loss:.4f}")
    print(f"loss went down: {last_loss < first_loss}")
    if resumed:
        print(f"resumed from step: {start_step}")


if __name__ == "__main__":
    main()
