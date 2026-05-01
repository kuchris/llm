import argparse
from pathlib import Path


DEFAULT_WIKITEXT_PATH = "data/wikitext103_clean.txt"
DEFAULT_DOLLY_PATH = "data/dolly_train_eos_sft.txt"
DEFAULT_BEA_PATH = "data/bea_grammar_train_eos_sft.txt"
DEFAULT_OUTPUT_PATH = "data/free_tokenizer_train.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikitext", default=DEFAULT_WIKITEXT_PATH)
    parser.add_argument("--dolly", default=DEFAULT_DOLLY_PATH)
    parser.add_argument("--bea", default=DEFAULT_BEA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    parts = [
        Path(args.wikitext).read_text(encoding="utf-8"),
        Path(args.dolly).read_text(encoding="utf-8"),
        Path(args.bea).read_text(encoding="utf-8"),
    ]
    text = "\n\n".join(parts)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"characters: {len(text)}")
    print(f"preview: {ascii(text[:240])}")


if __name__ == "__main__":
    main()
