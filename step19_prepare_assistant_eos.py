from pathlib import Path

from step16_prepare_alpaca import format_example
from step17_prepare_assistant_sft import IDENTITY_EXAMPLES, IDENTITY_REPEAT


# Edit these defaults when you do not want to type command options.
ALPACA_PATH = "data/alpaca_sft.txt"
WIKITEXT_PATH = "data/wikitext2_clean.txt"
SFT_OUTPUT_PATH = "data/assistant_eos_sft.txt"
TOKENIZER_TRAIN_OUTPUT_PATH = "data/assistant_eos_tokenizer_train.txt"
EOS_TOKEN = "<eos>"


def add_eos_to_sft_text(text: str) -> str:
    examples = []
    for block in text.split("### Instruction:\n"):
        block = block.strip()
        if not block or "### Response:\n" not in block:
            continue

        example = "### Instruction:\n" + block
        if not example.endswith(EOS_TOKEN):
            example += EOS_TOKEN
        examples.append(example)

    return "\n\n".join(examples) + "\n"


def main() -> None:
    identity_blocks = []
    for _ in range(IDENTITY_REPEAT):
        for instruction, input_text, output in IDENTITY_EXAMPLES:
            identity_blocks.append(format_example(instruction, input_text, output + EOS_TOKEN))

    alpaca_text = Path(ALPACA_PATH).read_text(encoding="utf-8")
    sft_text = "\n".join(identity_blocks) + "\n" + add_eos_to_sft_text(alpaca_text)

    sft_output_path = Path(SFT_OUTPUT_PATH)
    sft_output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_output_path.write_text(sft_text, encoding="utf-8")

    wikitext = Path(WIKITEXT_PATH).read_text(encoding="utf-8")
    tokenizer_text = wikitext + "\n\n" + sft_text
    tokenizer_output_path = Path(TOKENIZER_TRAIN_OUTPUT_PATH)
    tokenizer_output_path.write_text(tokenizer_text, encoding="utf-8")

    print(f"saved SFT: {sft_output_path}")
    print(f"saved tokenizer train: {tokenizer_output_path}")
    print(f"SFT characters: {len(sft_text)}")
    print(f"tokenizer train characters: {len(tokenizer_text)}")
    print(f"preview: {ascii(sft_text[:260])}")


if __name__ == "__main__":
    main()
