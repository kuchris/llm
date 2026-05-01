from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    # Tiny Shakespeare raw text, referenced by the Hugging Face dataset card.
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = Path("data/tiny_shakespeare.txt")
    output_path.parent.mkdir(exist_ok=True)

    print(f"downloading: {url}")
    urlretrieve(url, output_path)

    text = output_path.read_text(encoding="utf-8")
    print(f"saved: {output_path}")
    print(f"characters: {len(text)}")
    print(f"preview: {text[:80]!r}")


if __name__ == "__main__":
    main()
