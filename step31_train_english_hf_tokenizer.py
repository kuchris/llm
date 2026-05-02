import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_DATA_PATHS = [
    "data/fineweb_edu_sample_10bt.txt",
    "data/alpaca_cleaned_train_eos_sft.txt",
]
DEFAULT_OUTPUT_DIR = "tokenizers/english_bpe_8192"
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_MAX_CHARS = 100_000_000
SPECIAL_TOKENS = ["<unk>", "<eos>", "<|im_start|>", "<|im_end|>"]


def iter_text(paths: list[Path], max_chars: int) -> Iterable[str]:
    remaining = max_chars if max_chars > 0 else None
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if remaining is not None:
                    if remaining <= 0:
                        return
                    if len(line) > remaining:
                        line = line[:remaining]
                    remaining -= len(line)
                if line:
                    yield line


def create_tokenizer_config(save_dir: Path) -> None:
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    special_tokens_map = {
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<eos>", "<|im_start|>"],
    }

    (save_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (save_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_tokenizer(data_paths: list[Path], save_dir: Path, vocab_size: int, max_chars: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(iter_text(data_paths, max_chars), trainer=trainer)
    tokenizer.save(str(save_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(save_dir / "tokenizer.json"),
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        additional_special_tokens=["<eos>", "<|im_start|>"],
    )
    fast_tokenizer.save_pretrained(save_dir)
    create_tokenizer_config(save_dir)


def eval_tokenizer(save_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    sample = "### Instruction:\nHow are you?\n\n### Response:\nI'm fine, thank you.<|im_end|>"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)

    print(f"saved tokenizer: {save_dir}")
    print(f"vocab size: {len(tokenizer)}")
    print(f"special tokens: {tokenizer.all_special_tokens}")
    print(f"sample ids: {ids[:40]}")
    print(f"round trip ok: {decoded == sample}")
    print(f"decoded: {decoded!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    data_paths = [Path(path) for path in (args.data or DEFAULT_DATA_PATHS)]
    missing_paths = [path for path in data_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing tokenizer training data: {missing_paths}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir = Path(args.out)
    train_tokenizer(data_paths, save_dir, args.vocab_size, args.max_chars)
    eval_tokenizer(save_dir)


if __name__ == "__main__":
    main()
