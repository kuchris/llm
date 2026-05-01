from pathlib import Path

from datasets import load_dataset


def main() -> None:
    # WikiText-2 raw keeps case, punctuation, and numbers.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    output_path = Path("data/wikitext2.txt")
    output_path.parent.mkdir(exist_ok=True)

    parts = []
    for split in ("train", "validation", "test"):
        lines = [row["text"] for row in dataset[split] if row["text"].strip()]
        parts.append("\n".join(lines))

    text = "\n\n".join(parts)
    output_path.write_text(text, encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"characters: {len(text)}")
    preview = ascii(text[:120])
    print(f"preview: {preview}")


if __name__ == "__main__":
    main()
