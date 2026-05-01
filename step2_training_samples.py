#uv run python step2_training_samples.py


from pathlib import Path

from step1_tokenizer import CharTokenizer


def make_samples(token_ids: list[int], block_size: int) -> list[tuple[list[int], list[int]]]:
    samples = []

    # Each sample uses block_size characters as input.
    # The target is the same window shifted left by one character.
    for start in range(0, len(token_ids) - block_size):
        x = token_ids[start : start + block_size]
        y = token_ids[start + 1 : start + block_size + 1]
        samples.append((x, y))

    return samples


def main() -> None:
    # Load the same tiny dataset from Step 1.
    data_path = Path("data/tiny_text.txt")
    text = data_path.read_text(encoding="utf-8")

    # Build the character tokenizer and encode the whole dataset.
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    # Keep this small so the first examples are easy to read.
    block_size = 8
    samples = make_samples(token_ids, block_size)

    print(f"total token ids: {len(token_ids)}")
    print(f"block size: {block_size}")
    print(f"training samples: {len(samples)}")
    print()

    # Show the first few input/target pairs as text and as token ids.
    for index, (x, y) in enumerate(samples[:5], start=1):
        print(f"sample {index}")
        print(f"input text : {tokenizer.decode(x)!r}")
        print(f"target text: {tokenizer.decode(y)!r}")
        print(f"input ids  : {x}")
        print(f"target ids : {y}")
        print()

    # This confirms the first target starts at the second input character.
    first_x, first_y = samples[0]
    shifted_ok = first_x[1:] == first_y[:-1]
    print(f"shifted target ok: {shifted_ok}")


if __name__ == "__main__":
    main()
