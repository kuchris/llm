#uv run python step26_compare_tokenizers.py


import argparse
from pathlib import Path

from transformers import AutoTokenizer

from step13_bpe_tokenizer import BPETokenizer


DEFAULT_BPE_TOKENIZER = "tokenizers/free_bpe_6144.json"
DEFAULT_QWEN_TOKENIZER = "models/qwen3-0.6b"

DEFAULT_SAMPLES = [
    "She go to school yesterday.",
    "Where is Hong Kong?",
    "Correct the grammar, spelling, and punctuation mistakes in the following text.",
    "The house was painted green and purple.",
    "Europium is a chemical element.",
    "香港はどこですか?",
    "def fibonacci(n): return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
]


def preview_ids(ids: list[int], limit: int) -> str:
    shown = ids[:limit]
    suffix = " ..." if len(ids) > limit else ""
    return f"{shown}{suffix}"


def print_tokenization(
    *,
    label: str,
    text: str,
    ids: list[int],
    decoded: str,
    limit: int,
) -> None:
    print(f"{label} tokens: {len(ids)}")
    print(f"{label} ids: {preview_ids(ids, limit)}")
    print(f"{label} round trip: {decoded == text}")
    if decoded != text:
        print(f"{label} decoded: {ascii(decoded)}")


def load_samples(args: argparse.Namespace) -> list[str]:
    samples = list(DEFAULT_SAMPLES)
    if args.text:
        samples.append(args.text)
    if args.file:
        file_text = Path(args.file).read_text(encoding="utf-8")
        samples.append(file_text[: args.max_file_chars])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-tokenizer", default=DEFAULT_BPE_TOKENIZER)
    parser.add_argument("--qwen-tokenizer", default=DEFAULT_QWEN_TOKENIZER)
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--max-file-chars", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    bpe = BPETokenizer.from_file(args.bpe_tokenizer)
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer)

    print("== Tokenizers ==")
    print(f"repo BPE: {args.bpe_tokenizer} vocab={bpe.vocab_size:,}")
    print(f"qwen: {args.qwen_tokenizer} vocab={len(qwen):,}")
    print()

    for index, text in enumerate(load_samples(args), start=1):
        bpe_ids = bpe.encode(text)
        qwen_ids = qwen.encode(text, add_special_tokens=False)

        print(f"== Sample {index} ==")
        print(f"text: {ascii(text)}")
        print_tokenization(
            label="repo BPE",
            text=text,
            ids=bpe_ids,
            decoded=bpe.decode(bpe_ids),
            limit=args.limit,
        )
        print_tokenization(
            label="qwen",
            text=text,
            ids=qwen_ids,
            decoded=qwen.decode(qwen_ids),
            limit=args.limit,
        )
        if qwen_ids:
            ratio = len(bpe_ids) / len(qwen_ids)
            print(f"repo/qwen token count ratio: {ratio:.2f}x")
        print()


if __name__ == "__main__":
    main()
