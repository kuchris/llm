import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = "models/qwen3-0.6b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
    )

    print(f"model id: {args.model_id}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
