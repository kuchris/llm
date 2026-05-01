import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path

from step16_prepare_alpaca import format_example


ARCHIVE_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
DEFAULT_ARCHIVE_PATH = "data/cache/wi_locness_v2.1.bea19.tar.gz"
DEFAULT_TRAIN_OUTPUT_PATH = "data/bea_grammar_train_eos_sft.txt"
DEFAULT_EVAL_OUTPUT_PATH = "data/bea_grammar_eval_eos_sft.txt"
EOS_TOKEN = "<eos>"
INSTRUCTION = "Correct the grammar, spelling, and punctuation mistakes in the following text."


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_edits(text: str, edits: list[list[int | str | None]]) -> str:
    pieces = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda item: item[0]):
        pieces.append(text[cursor:start])
        if replacement:
            pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def extract_first_annotator_edits(row: dict) -> list[list[int | str | None]]:
    raw_edits = row.get("edits", [])
    if not raw_edits:
        return []

    first = raw_edits[0]
    if len(first) < 2:
        return []

    return first[1]


def iter_archive_rows(archive_path: Path, wanted_paths: set[str]):
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in wanted_paths:
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue

            for line in file_obj:
                yield member.name, json.loads(line.decode("utf-8"))


def build_examples(rows, max_examples: int) -> list[str]:
    examples = []
    for _, row in rows:
        source_text = normalize_text(row["text"])
        corrected_text = normalize_text(apply_edits(row["text"], extract_first_annotator_edits(row)))
        if not source_text or not corrected_text:
            continue

        examples.append(format_example(INSTRUCTION, source_text, corrected_text + EOS_TOKEN))
        if max_examples > 0 and len(examples) >= max_examples:
            break

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--eval-out", default=DEFAULT_EVAL_OUTPUT_PATH)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"downloading: {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    train_paths = {
        "wi+locness/json/A.train.json",
        "wi+locness/json/B.train.json",
        "wi+locness/json/C.train.json",
    }
    eval_paths = {
        "wi+locness/json/A.dev.json",
        "wi+locness/json/B.dev.json",
        "wi+locness/json/C.dev.json",
        "wi+locness/json/N.dev.json",
    }

    train_examples = build_examples(iter_archive_rows(archive_path, train_paths), args.max_train_examples)
    eval_examples = build_examples(iter_archive_rows(archive_path, eval_paths), args.max_eval_examples)

    train_output_path = Path(args.train_out)
    eval_output_path = Path(args.eval_out)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_output_path.write_text("\n".join(train_examples), encoding="utf-8")
    eval_output_path.write_text("\n".join(eval_examples), encoding="utf-8")

    print(f"train examples: {len(train_examples)}")
    print(f"eval examples: {len(eval_examples)}")
    print(f"saved train: {train_output_path}")
    print(f"saved eval: {eval_output_path}")
    if train_examples:
        print(f"train preview: {ascii(train_examples[0][:240])}")


if __name__ == "__main__":
    main()
