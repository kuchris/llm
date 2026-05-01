import argparse
from pathlib import Path

from datasets import load_dataset


# Edit these defaults when you do not want to type command options.
DEFAULT_OUTPUT_PATH = "data/alpaca_sft.txt"
DEFAULT_MAX_EXAMPLES = 0


def format_example(instruction: str, input_text: str, output: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    output = output.strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    examples = []
    for row in dataset:
        examples.append(format_example(row["instruction"], row["input"], row["output"]))

    text = "\n".join(examples)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"examples: {len(examples)}")
    print(f"saved: {output_path}")
    print(f"characters: {len(text)}")
    print(f"preview: {ascii(text[:240])}")


if __name__ == "__main__":
    main()
