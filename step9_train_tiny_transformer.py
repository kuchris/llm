#uv run python step9_train_tiny_transformer.py
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe
#uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe


import argparse
import os
from pathlib import Path

# This tiny CPU project does not use torch.compile.
# Disabling Dynamo avoids slow lazy-import startup during optimizer creation.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from step1_tokenizer import CharTokenizer
from step13_bpe_tokenizer import BPETokenizer
from step3_tiny_llama_parts import ModelConfig
from step8_tiny_transformer import TinyTransformer


# Edit this when you want to switch the default training target.
DEFAULT_PRESET = "assistant_eos_sft_bpe"
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
        "data": "data/wikitext2_clean.txt",
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
        "data": "data/wikitext2_clean.txt",
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
}


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
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--tokenizer-file", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--task", choices=["lm", "sft"], default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
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

    if len(token_ids) <= block_size + 1:
        raise ValueError("dataset is too short for the chosen block size")

    # Build the tiny Transformer using the real tokenizer vocabulary size.
    model_args = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=block_size)
    model = TinyTransformer(model_args).to(device)
    init_checkpoint = cli_args.init_checkpoint if cli_args.init_checkpoint is not None else defaults["init_checkpoint"]
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if init_path.exists():
            init_data = torch.load(init_path, map_location="cpu", weights_only=False)
            model.load_state_dict(init_data["model"])
            print(f"loaded init checkpoint: {init_path}")
        else:
            print(f"init checkpoint not found, training from scratch: {init_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    first_loss = None
    last_loss = None

    model.train()
    progress = tqdm(range(max_iters + 1), desc="training", unit="step")
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

        if step == 0:
            first_loss = loss.item()

        if step % 20 == 0:
            last_loss = loss.item()
            progress.set_postfix(loss=f"{last_loss:.4f}")
            tqdm.write(f"step {step:3d} | loss {last_loss:.4f}")

    # Save the trained weights so the next step can generate text from them.
    checkpoint_path = Path(cli_args.checkpoint or defaults["checkpoint"])
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": model_args,
        "tokenizer_type": tokenizer_type,
        "task": task,
    }
    if tokenizer_type == "char":
        checkpoint["vocab"] = tokenizer.stoi
    elif tokenizer_type == "bpe":
        checkpoint["tokenizer_json"] = Path(tokenizer_file).read_text(encoding="utf-8")

    torch.save(checkpoint, checkpoint_path)

    print()
    print(f"first loss: {first_loss:.4f}")
    print(f"last printed loss: {last_loss:.4f}")
    print(f"loss went down: {last_loss < first_loss}")
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
