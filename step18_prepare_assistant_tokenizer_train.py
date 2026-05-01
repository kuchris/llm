from pathlib import Path


# Edit these defaults when you do not want to type command options.
WIKITEXT_PATH = "data/wikitext2_clean.txt"
ASSISTANT_SFT_PATH = "data/assistant_sft.txt"
OUTPUT_PATH = "data/assistant_tokenizer_train.txt"


def main() -> None:
    wikitext = Path(WIKITEXT_PATH).read_text(encoding="utf-8")
    assistant_sft = Path(ASSISTANT_SFT_PATH).read_text(encoding="utf-8")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(wikitext + "\n\n" + assistant_sft, encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"characters: {output_path.read_text(encoding='utf-8').__len__()}")


if __name__ == "__main__":
    main()
