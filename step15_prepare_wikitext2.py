import argparse
import re
from pathlib import Path


# Edit these defaults when you do not want to type command options.
DEFAULT_INPUT_PATH = "data/wikitext103.txt"
DEFAULT_OUTPUT_PATH = "data/wikitext103_clean.txt"


def clean_wikitext(text: str) -> str:
    # WikiText uses tokenization artifacts like "40 @-@ minute".
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")

    # Normalize headings like "= = = Career = =" into "### Career".
    text = re.sub(
        r"(?m)^\s*(=+)\s*(.*?)\s*\1\s*$",
        lambda match: "#" * min(len(match.group(1)), 6) + " " + match.group(2).strip(),
        text,
    )

    # Remove extra spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Keep paragraphs, but avoid huge blank gaps.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)

    text = input_path.read_text(encoding="utf-8")
    cleaned = clean_wikitext(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"input chars: {len(text)}")
    print(f"output chars: {len(cleaned)}")
    print(f"removed @-@ count: {text.count(' @-@ ')}")
    print(f"preview: {ascii(cleaned[:160])}")


if __name__ == "__main__":
    main()
