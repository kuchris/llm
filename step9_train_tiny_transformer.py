#uv run python step9_train_tiny_transformer.py
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe


import argparse
import os
import sys
from pathlib import Path

# This tiny CPU project does not use torch.compile.
# Disabling Dynamo avoids slow lazy-import startup during optimizer creation.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

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
        "data": "data/wikitext103_clean.txt",
        "checkpoint": "checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 2,
        "max_iters": 2000,
        "learning_rate": 1e-3,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
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
    "qwen_tokenizer_tiny_sft": {
        "data": "data/dolly_train_eos_sft.txt",
        "checkpoint": "checkpoints/qwen_tokenizer_tiny_sft/tiny_transformer.pt",
        "max_chars": 1000000,
        "block_size": 128,
        "batch_size": 2,
        "max_iters": 1000,
        "learning_rate": 5e-4,
        "tokenizer": "hf",
        "tokenizer_file": "",
        "tokenizer_name": "models/qwen3-0.6b",
        "task": "sft",
        "init_checkpoint": "checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt",
        "dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 3,
        "hidden_dim": 1024,
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
    parser.add_argument("--task", choices=["lm", "sft"], default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=200)
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

    # Load and tokenize the tiny dataset.
    data_path = Path(cli_args.data or defaults["data"])
    text = data_path.read_text(encoding="utf-8")
    max_chars = cli_args.max_chars if cli_args.max_chars is not None else defaults["max_chars"]
    if max_chars > 0:
        text = text[:max_chars]

    tokenizer_type = cli_args.tokenizer or defaults["tokenizer"]
    tokenizer_file = cli_args.tokenizer_file if cli_args.tokenizer_file is not None else defaults["tokenizer_file"]
    tokenizer_name = cli_args.tokenizer_name if cli_args.tokenizer_name is not None else defaults["tokenizer_name"]
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

    task = cli_args.task or defaults["task"]
    if task == "sft":
        token_ids, loss_mask = build_sft_token_stream(tokenizer, text)
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

    if len(token_ids) <= block_size + 1:
        raise ValueError("dataset is too short for the chosen block size")

    # Build the Transformer using the selected tokenizer vocabulary size.
    model_args = build_model_config(defaults, tokenizer.vocab_size, block_size)
    model = TinyTransformer(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    first_loss = None
    last_loss = None
    start_step = 0
    resumed = False

    if cli_args.resume and resume_path.exists():
        resume_data = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_data["model"])
        if "optimizer" in resume_data:
            optimizer.load_state_dict(resume_data["optimizer"])
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

    if start_step > max_iters:
        print(f"checkpoint already reached step {start_step - 1}, which is >= max_iters {max_iters}")
        return

    model.train()
    use_tqdm = sys.stderr.isatty()
    progress_steps = range(start_step, max_iters + 1)
    progress = tqdm(progress_steps, desc="training", unit="step") if use_tqdm else progress_steps
    last_completed_step = start_step - 1

    try:
        for step in progress:
            if task == "sft":
                x, y, y_mask = get_batch_sft(token_ids, loss_mask, batch_size, block_size)
            else:
                x, y = get_batch(token_ids, batch_size, block_size)
                y_mask = None
            x = x.to(device)
            y = y.to(device)
            if y_mask is not None:
                y_mask = y_mask.to(device)

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
            loss.backward()
            optimizer.step()
            last_completed_step = step

            if first_loss is None:
                first_loss = loss.item()

            if step % 20 == 0:
                last_loss = loss.item()
                if use_tqdm:
                    progress.set_postfix(loss=f"{last_loss:.4f}")
                    tqdm.write(f"step {step:3d} | loss {last_loss:.4f}")
                else:
                    print(f"step {step:5d} / {max_iters:5d} | loss {last_loss:.4f}")

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
