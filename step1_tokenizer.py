#uv run python step1_tokenizer.py

from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str, unk_token: str | None = None):
        self.unk_token = unk_token

        # Collect every different character from the training text.
        chars = sorted(set(text))
        if unk_token is not None and unk_token not in chars:
            chars.append(unk_token)

        # stoi means "string to integer": character -> token id.
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # itos means "integer to string": token id -> character.
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        # The vocabulary size is how many different tokens we know.
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        # Turn each character into its token id.
        if self.unk_token is None:
            return [self.stoi[ch] for ch in text]

        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        # Turn each token id back into a character, then join them.
        return "".join(self.itos[i] for i in ids)


def main() -> None:
    # Load the tiny dataset from disk.
    data_path = Path("data/tiny_text.txt")
    text = data_path.read_text(encoding="utf-8")

    # Build the tokenizer from the characters found in the dataset.
    tokenizer = CharTokenizer(text)

    # Try a small text that only uses characters from the dataset.
    sample = "hello llm. this is a tiny language model lesson."

    # Encode text -> token ids, then decode token ids -> text.
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    # Print a small verification report.
    print(f"dataset chars: {len(text)}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"sample text: {sample!r}")
    print(f"encoded ids: {ids}")
    print(f"decoded text: {decoded!r}")
    print(f"round trip ok: {decoded == sample}")


if __name__ == "__main__":
    main()
