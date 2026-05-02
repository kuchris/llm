import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT_PATH = "data/fineweb_edu_sample_10bt.txt"
DEFAULT_MAX_CHARS = 200_000_000
DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, args.config, split="train", streaming=True)

    written_chars = 0
    written_docs = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset:
            text = clean_text(row.get("text", ""))
            if not text:
                continue

            remaining = args.max_chars - written_chars if args.max_chars > 0 else None
            if remaining is not None and remaining <= 0:
                break

            if remaining is not None and len(text) > remaining:
                text = text[:remaining].rstrip()

            handle.write(text)
            handle.write("\n\n")
            written_chars += len(text) + 2
            written_docs += 1

            if written_docs % 1000 == 0:
                print(f"docs: {written_docs:,} | chars: {written_chars:,}")

            if args.max_chars > 0 and written_chars >= args.max_chars:
                break

    print(f"dataset: {args.dataset}/{args.config}")
    print(f"docs: {written_docs:,}")
    print(f"chars: {written_chars:,}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
