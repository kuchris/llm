from pathlib import Path

from transformers import AutoTokenizer


class HFTokenizer:
    def __init__(self, name_or_path: str | Path):
        self.name_or_path = str(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)
