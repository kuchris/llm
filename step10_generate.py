#uv run python step10_generate.py


import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from step1_tokenizer import CharTokenizer
from step13_bpe_tokenizer import BPETokenizer
from step8_tiny_transformer import TinyTransformer


# Edit this when you want to switch the default generation target.
DEFAULT_PRESET = "assistant_eos_sft_bpe"

# Presets keep dataset results separated.
PRESETS = {
    "tiny_shakespeare": {
        "checkpoint": "checkpoints/tiny_shakespeare/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "{prompt}",
        "stop": [],
        "max_new_tokens": 300,
        "temperature": 0.6,
        "out": "",
    },
    "wikitext2": {
        "checkpoint": "checkpoints/wikitext2/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "{prompt}",
        "stop": [],
        "max_new_tokens": 300,
        "temperature": 0.6,
        "out": "",
    },
    "wikitext2_bpe": {
        "checkpoint": "checkpoints/wikitext2_bpe/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "{prompt}",
        "stop": [],
        "max_new_tokens": 120,
        "temperature": 0.5,
        "out": "",
    },
    "wikitext2_bpe_4layer": {
        "checkpoint": "checkpoints/wikitext2_bpe_4layer/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "{prompt}",
        "stop": [],
        "max_new_tokens": 120,
        "temperature": 0.5,
        "out": "",
    },
    "wikitext2_clean_bpe_4layer": {
        "checkpoint": "checkpoints/wikitext2_clean_bpe_4layer/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "{prompt}",
        "stop": [],
        "max_new_tokens": 120,
        "temperature": 0.4,
        "out": "",
    },
    "alpaca_bpe_4layer": {
        "checkpoint": "checkpoints/alpaca_bpe_4layer/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "### Instruction:\n{prompt}\n\n### Response:\n",
        "stop": ["\n\n### Instruction:", "\n\n\n"],
        "max_new_tokens": 160,
        "temperature": 0.6,
        "out": "",
    },
    "assistant_sft_bpe_4layer": {
        "checkpoint": "checkpoints/assistant_sft_bpe_4layer/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "### Instruction:\n{prompt}\n\n### Response:\n",
        "stop": ["\n\n### Instruction:", "\n\n\n"],
        "max_new_tokens": 160,
        "temperature": 0.5,
        "out": "",
    },
    "assistant_eos_sft_bpe": {
        "checkpoint": "checkpoints/assistant_eos_sft_bpe/tiny_transformer.pt",
        "prompt": None,
        "prompt_template": "### Instruction:\n{prompt}\n\n### Response:\n",
        "stop": ["<eos>", "\n\n### Instruction:"],
        "max_new_tokens": 160,
        "temperature": 0.5,
        "out": "",
    },
}


def build_tokenizer_from_vocab(vocab: dict[str, int]) -> CharTokenizer:
    # Make an empty tokenizer object, then fill it with the saved vocab.
    fallback_char = " " if " " in vocab else next(iter(vocab))
    tokenizer = CharTokenizer("", unk_token=fallback_char)
    tokenizer.stoi = vocab
    tokenizer.itos = {i: ch for ch, i in vocab.items()}
    tokenizer.unk_token = fallback_char
    return tokenizer


def build_tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer | BPETokenizer:
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        return BPETokenizer.from_json(checkpoint["tokenizer_json"])

    return build_tokenizer_from_vocab(checkpoint["vocab"])


@torch.no_grad()
def generate(
    model: TinyTransformer,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> str:
    model.eval()

    # Start from the user's prompt.
    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Keep only the latest tokens that fit in the model context window.
        context = tokens[:, -model.args.max_seq_len :]

        # Get next-token scores for every position.
        logits = model(context)

        # We only need the scores from the last position.
        next_token_logits = logits[:, -1, :] / temperature

        # Convert scores to probabilities and sample one next token.
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the sampled token and continue.
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


def main() -> None:
    defaults = PRESETS[DEFAULT_PRESET]

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch install has no CUDA support. Reinstall CUDA-enabled torch or use --device cpu.")
    print(f"device: {device}")

    defaults = PRESETS[args.preset]
    checkpoint_path = Path(args.checkpoint or defaults["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    tokenizer = build_tokenizer_from_checkpoint(checkpoint)
    model = TinyTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])

    prompt = args.prompt if args.prompt is not None else defaults["prompt"]
    if prompt is None:
        prompt = input("Prompt: ")
        if not prompt:
            prompt = "The "

    prompt = defaults["prompt_template"].format(prompt=prompt)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens or defaults["max_new_tokens"],
        temperature=args.temperature or defaults["temperature"],
        device=device,
    )
    stop_texts = defaults["stop"]
    stop_indexes = [text.index(stop_text) for stop_text in stop_texts if stop_text in text]
    if stop_indexes:
        text = text[: min(stop_indexes)]
    print(text)

    output_path_text = args.out if args.out is not None else defaults["out"]
    if output_path_text:
        output_path = Path(output_path_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print()
        print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
