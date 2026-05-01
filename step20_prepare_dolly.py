import argparse
from pathlib import Path

from datasets import load_dataset

from step16_prepare_alpaca import format_example


DEFAULT_TRAIN_OUTPUT_PATH = "data/dolly_train_eos_sft.txt"
DEFAULT_EVAL_OUTPUT_PATH = "data/dolly_eval_eos_sft.txt"
DEFAULT_EVAL_SIZE = 1000
EOS_TOKEN = "<eos>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--eval-out", default=DEFAULT_EVAL_OUTPUT_PATH)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.shuffle(seed=42)

    eval_size = max(0, min(args.eval_size, len(dataset)))
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, len(dataset)))

    if args.max_train_examples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_examples, len(train_dataset))))
    if args.max_eval_examples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_examples, len(eval_dataset))))

    def build_text(split_dataset) -> str:
        examples = []
        for row in split_dataset:
            examples.append(format_example(row["instruction"], row["context"], row["response"] + EOS_TOKEN))
        return "\n".join(examples)

    train_text = build_text(train_dataset)
    eval_text = build_text(eval_dataset)

    train_output_path = Path(args.train_out)
    eval_output_path = Path(args.eval_out)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_output_path.write_text(train_text, encoding="utf-8")
    eval_output_path.write_text(eval_text, encoding="utf-8")

    print(f"train examples: {len(train_dataset)}")
    print(f"eval examples: {len(eval_dataset)}")
    print(f"saved train: {train_output_path}")
    print(f"saved eval: {eval_output_path}")
    print(f"train preview: {ascii(train_text[:240])}")


if __name__ == "__main__":
    main()
