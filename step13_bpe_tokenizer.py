import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# Edit these defaults when you do not want to type command options.
DEFAULT_DATA_PATH = "data/free_tokenizer_train.txt"
DEFAULT_OUTPUT_PATH = "tokenizers/free_bpe_6144.json"
DEFAULT_VOCAB_SIZE = 6144
DEFAULT_MAX_CHARS = 10000000


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_json(cls, text: str) -> "BPETokenizer":
        return cls(Tokenizer.from_str(text))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def train_bpe_tokenizer(
    data_path: Path,
    output_path: Path,
    vocab_size: int,
    max_chars: int,
    special_tokens: list[str],
) -> None:
    text = data_path.read_text(encoding="utf-8")
    if max_chars > 0:
        text = text[:max_chars]

    temp_path = output_path.with_suffix(".train.txt")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(text, encoding="utf-8")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(temp_path)], trainer)
    tokenizer.save(str(output_path))
    temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--special-token", action="append", default=[])
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.out)
    special_tokens = list(dict.fromkeys(args.special_token or ["<unk>"]))
    train_bpe_tokenizer(data_path, output_path, args.vocab_size, args.max_chars, special_tokens)

    tokenizer = BPETokenizer.from_file(output_path)
    sample = "Europium is a chemical element."
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print(f"saved tokenizer: {output_path}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"sample: {sample!r}")
    print(f"ids: {ids}")
    print(f"decoded: {decoded!r}")
    print(f"round trip ok: {decoded == sample}")


if __name__ == "__main__":
    main()
