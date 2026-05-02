import argparse
import json
from pathlib import Path

import numpy as np

from hf_tokenizer import HFTokenizer


DEFAULT_DATA_PATH = "data/fineweb_edu_sample_10bt.txt"
DEFAULT_OUTPUT_DIR = "data/fineweb_edu_qwen_uint32"
DEFAULT_TOKENIZER_NAME = "models/qwen3-0.6b"
DEFAULT_CHUNK_CHARS = 2_000_000
DEFAULT_SHARD_TOKENS = 100_000_000
DEFAULT_MAX_CHARS = 0
DTYPE = "uint32"


def finalize_existing_shards(output_dir: Path, data_path: Path, tokenizer: HFTokenizer, shard_tokens: int) -> None:
    shards = []
    total_tokens = 0
    for path in sorted(output_dir.glob("shard_*.bin")):
        tokens = path.stat().st_size // np.dtype(np.uint32).itemsize
        if tokens <= 0:
            continue
        shards.append({"path": path.name, "tokens": tokens})
        total_tokens += tokens

    if not shards:
        raise ValueError(f"no shard_*.bin files found in {output_dir}")

    manifest = {
        "source": str(data_path),
        "tokenizer_type": "hf",
        "tokenizer_name": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "dtype": DTYPE,
        "shard_tokens": shard_tokens,
        "total_tokens": total_tokens,
        "total_chars": None,
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")
    print(f"total tokens: {total_tokens:,}")
    print(f"shards: {len(shards):,}")


def open_shard(output_dir: Path, shard_index: int):
    path = output_dir / f"shard_{shard_index:05d}.bin"
    return path, path.open("wb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()

    if args.chunk_chars <= 0:
        raise ValueError("--chunk-chars must be > 0")
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be > 0")

    data_path = Path(args.data)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HFTokenizer(args.tokenizer_name)
    if args.finalize_existing:
        finalize_existing_shards(output_dir, data_path, tokenizer, args.shard_tokens)
        return

    manifest = {
        "source": str(data_path),
        "tokenizer_type": "hf",
        "tokenizer_name": args.tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "dtype": DTYPE,
        "shard_tokens": args.shard_tokens,
        "total_tokens": 0,
        "total_chars": 0,
        "shards": [],
    }

    shard_index = 0
    shard_token_count = 0
    shard_path, shard_handle = open_shard(output_dir, shard_index)

    remaining_chars = args.max_chars if args.max_chars > 0 else None
    try:
        with data_path.open("r", encoding="utf-8") as text_handle:
            while True:
                read_size = args.chunk_chars if remaining_chars is None else min(args.chunk_chars, remaining_chars)
                if read_size <= 0:
                    break

                text = text_handle.read(read_size)
                if not text:
                    break
                if remaining_chars is not None:
                    remaining_chars -= len(text)
                manifest["total_chars"] += len(text)

                ids = tokenizer.encode(text)
                offset = 0
                while offset < len(ids):
                    available = args.shard_tokens - shard_token_count
                    end = min(offset + available, len(ids))
                    np.asarray(ids[offset:end], dtype=np.uint32).tofile(shard_handle)
                    written = end - offset
                    shard_token_count += written
                    manifest["total_tokens"] += written
                    offset = end

                    if shard_token_count >= args.shard_tokens:
                        shard_handle.close()
                        manifest["shards"].append(
                            {
                                "path": shard_path.name,
                                "tokens": shard_token_count,
                            }
                        )
                        print(f"saved shard {shard_index:05d}: {shard_token_count:,} tokens")
                        shard_index += 1
                        shard_token_count = 0
                        shard_path, shard_handle = open_shard(output_dir, shard_index)

                if manifest["total_chars"] and manifest["total_chars"] % 100_000_000 < args.chunk_chars:
                    print(
                        f"chars: {manifest['total_chars']:,} | "
                        f"tokens: {manifest['total_tokens']:,} | "
                        f"current shard: {shard_index:05d}"
                    )
    finally:
        shard_handle.close()

    if shard_token_count > 0:
        manifest["shards"].append(
            {
                "path": shard_path.name,
                "tokens": shard_token_count,
            }
        )
        print(f"saved shard {shard_index:05d}: {shard_token_count:,} tokens")
    elif shard_path.exists():
        shard_path.unlink()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")
    print(f"total chars: {manifest['total_chars']:,}")
    print(f"total tokens: {manifest['total_tokens']:,}")
    print(f"shards: {len(manifest['shards']):,}")


if __name__ == "__main__":
    main()
