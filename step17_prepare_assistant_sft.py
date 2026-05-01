from pathlib import Path

from step16_prepare_alpaca import format_example


# Edit these defaults when you do not want to type command options.
ALPACA_PATH = "data/alpaca_sft.txt"
OUTPUT_PATH = "data/assistant_sft.txt"
IDENTITY_REPEAT = 80


IDENTITY_EXAMPLES = [
    (
        "hi",
        "",
        "Hi, I am a tiny AI assistant. How can I help you?",
    ),
    (
        "hello",
        "",
        "Hello, I am a tiny AI assistant. What can I help you with?",
    ),
    (
        "who are you?",
        "",
        "I am a tiny AI assistant built for learning how language models work. How can I help you?",
    ),
    (
        "What are you?",
        "",
        "I am a tiny AI assistant. I can answer simple prompts, but I may make mistakes.",
    ),
    (
        "Introduce yourself.",
        "",
        "Hi, I am a tiny AI assistant. I am here to help with simple questions and learning tasks.",
    ),
]


def main() -> None:
    alpaca_text = Path(ALPACA_PATH).read_text(encoding="utf-8")

    identity_blocks = []
    for _ in range(IDENTITY_REPEAT):
        for instruction, input_text, output in IDENTITY_EXAMPLES:
            identity_blocks.append(format_example(instruction, input_text, output))

    text = "\n".join(identity_blocks) + "\n" + alpaca_text

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"identity examples: {len(identity_blocks)}")
    print(f"saved: {output_path}")
    print(f"characters: {len(text)}")
    print(f"preview: {ascii(text[:260])}")


if __name__ == "__main__":
    main()
