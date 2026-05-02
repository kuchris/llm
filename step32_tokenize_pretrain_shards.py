import argparse
import json
import os
import queue
import re
import threading
from pathlib import Path

import numpy as np

from hf_tokenizer import HFTokenizer


DEFAULT_DATA_PATH = "data/fineweb_edu_sample_10bt.txt"
DEFAULT_OUTPUT_DIR = "data/fineweb_edu_qwen_uint32"
DEFAULT_TOKENIZER_NAME = "models/qwen3-0.6b"
DEFAULT_CHUNK_CHARS = 2_000_000
DEFAULT_SHARD_TOKENS = 100_000_000
DEFAULT_MAX_CHARS = 0
DEFAULT_BATCH_LINES = 4096
DEFAULT_NUM_THREADS = 0


def finalize_existing_shards(
    output_dir: Path,
    data_path: Path,
    tokenizer: HFTokenizer,
    shard_tokens: int,
    dtype_name: str,
) -> None:
    dtype = np.dtype(dtype_name)
    shards = []
    total_tokens = 0
    for path in sorted(output_dir.glob("shard_*.bin")):
        tokens = path.stat().st_size // dtype.itemsize
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
        "dtype": dtype_name,
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


def append_shard(output_dir: Path, shard_name: str):
    path = output_dir / shard_name
    return path, path.open("ab")


def shard_index_from_name(shard_name: str) -> int:
    match = re.fullmatch(r"shard_(\d+)\.bin", shard_name)
    if not match:
        raise ValueError(f"invalid shard name in manifest: {shard_name}")
    return int(match.group(1))


def write_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def iter_encoded_batches(tokenizer: HFTokenizer, text: str, batch_lines: int):
    if batch_lines <= 0:
        yield tokenizer.encode(text)
        return

    lines = text.splitlines(keepends=True)
    for start in range(0, len(lines), batch_lines):
        batch = lines[start : start + batch_lines]
        encoded = tokenizer.tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        ids = []
        for item in encoded:
            ids.extend(item)
        yield ids


def validate_resume_manifest(manifest: dict, data_path: Path, tokenizer: HFTokenizer, args: argparse.Namespace) -> None:
    if Path(manifest["source"]) != data_path:
        raise ValueError(f"resume manifest source is {manifest['source']}, expected {data_path}")
    if manifest["tokenizer_name"] != args.tokenizer_name:
        raise ValueError(
            f"resume manifest tokenizer is {manifest['tokenizer_name']}, "
            f"expected {args.tokenizer_name}"
        )
    if int(manifest["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError(
            f"resume manifest vocab size is {manifest['vocab_size']}, "
            f"expected {tokenizer.vocab_size}"
        )
    if manifest["dtype"] != args.dtype:
        raise ValueError(f"resume manifest dtype is {manifest['dtype']}, expected {args.dtype}")
    if int(manifest["shard_tokens"]) != args.shard_tokens:
        raise ValueError(
            f"resume manifest shard_tokens is {manifest['shard_tokens']}, "
            f"expected {args.shard_tokens}"
        )


def _chunk_reader(
    text_handle,
    chunk_chars: int,
    max_chars: "int | None",
    out_queue: "queue.Queue",
) -> None:
    remaining = max_chars
    try:
        while True:
            size = chunk_chars if remaining is None else min(chunk_chars, remaining)
            if size <= 0:
                break
            text = text_handle.read(size)
            if not text:
                break
            pos = text_handle.tell()
            if remaining is not None:
                remaining -= len(text)
            out_queue.put((text, pos))
    except Exception as exc:
        out_queue.put(exc)
    finally:
        out_queue.put(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint32")
    parser.add_argument("--batch-lines", type=int, default=DEFAULT_BATCH_LINES)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()

    if args.chunk_chars <= 0:
        raise ValueError("--chunk-chars must be > 0")
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be > 0")
    if args.batch_lines < 0:
        raise ValueError("--batch-lines must be >= 0")

    if args.num_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)
    if args.batch_lines > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    data_path = Path(args.data)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = HFTokenizer(args.tokenizer_name)
    dtype = np.dtype(args.dtype)
    if tokenizer.vocab_size > np.iinfo(dtype).max + 1:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size} does not fit in {args.dtype}; "
            "use --dtype uint32"
        )
    if args.finalize_existing:
        finalize_existing_shards(output_dir, data_path, tokenizer, args.shard_tokens, args.dtype)
        return

    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(manifest, data_path, tokenizer, args)
        if manifest.get("complete"):
            print(f"manifest is already complete: {manifest_path}")
            return
        current_shard = manifest.get("current_shard") or {}
        shard_name = current_shard.get("path", f"shard_{len(manifest['shards']):05d}.bin")
        shard_token_count = int(current_shard.get("tokens", 0))
        shard_index = shard_index_from_name(shard_name)
        shard_path = output_dir / shard_name
        actual_tokens = shard_path.stat().st_size // dtype.itemsize
        if actual_tokens > shard_token_count:
            expected_bytes = shard_token_count * dtype.itemsize
            with shard_path.open("ab") as handle:
                handle.truncate(expected_bytes)
            print(
                f"truncated partial shard {shard_path}: "
                f"{actual_tokens:,} -> {shard_token_count:,} tokens"
            )
        elif actual_tokens < shard_token_count:
            raise ValueError(
                f"{shard_path} has {actual_tokens:,} tokens, "
                f"but manifest expected {shard_token_count:,}; "
                "restart this shard directory or restore the missing shard data"
            )
        shard_path, shard_handle = append_shard(output_dir, shard_name)
        source_position = int(manifest.get("source_position", 0))
        print(f"resuming from char position: {source_position:,}")
        print(f"resuming shard {shard_index:05d}: {shard_token_count:,} tokens")
    else:
        manifest = {
            "source": str(data_path),
            "tokenizer_type": "hf",
            "tokenizer_name": args.tokenizer_name,
            "vocab_size": tokenizer.vocab_size,
            "dtype": args.dtype,
            "shard_tokens": args.shard_tokens,
            "total_tokens": 0,
            "total_chars": 0,
            "source_position": 0,
            "current_shard": {"path": "shard_00000.bin", "tokens": 0},
            "complete": False,
            "shards": [],
        }
        shard_index = 0
        shard_token_count = 0
        shard_path, shard_handle = open_shard(output_dir, shard_index)
        source_position = 0

    remaining_chars = None
    if args.max_chars > 0:
        remaining_chars = max(0, args.max_chars - int(manifest["total_chars"]))
    try:
        with data_path.open("r", encoding="utf-8") as text_handle:
            if source_position:
                text_handle.seek(source_position)

            chunk_queue: queue.Queue = queue.Queue(maxsize=2)
            reader = threading.Thread(
                target=_chunk_reader,
                args=(text_handle, args.chunk_chars, remaining_chars, chunk_queue),
                daemon=True,
            )
            reader.start()

            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                text, source_pos = item

                manifest["total_chars"] += len(text)
                manifest["source_position"] = source_pos

                for ids in iter_encoded_batches(tokenizer, text, args.batch_lines):
                    arr = np.asarray(ids, dtype=dtype)
                    offset = 0
                    while offset < len(arr):
                        available = args.shard_tokens - shard_token_count
                        end = min(offset + available, len(arr))
                        arr[offset:end].tofile(shard_handle)
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
                            manifest["current_shard"] = {"path": shard_path.name, "tokens": 0}

                        manifest["current_shard"] = {"path": shard_path.name, "tokens": shard_token_count}

                shard_handle.flush()
                write_manifest(output_dir, manifest)
                if manifest["total_chars"] and manifest["total_chars"] % 100_000_000 < args.chunk_chars:
                    print(
                        f"chars: {manifest['total_chars']:,} | "
                        f"tokens: {manifest['total_tokens']:,} | "
                        f"current shard: {shard_index:05d}"
                    )

            reader.join()
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

    manifest.pop("current_shard", None)
    manifest["complete"] = True
    write_manifest(output_dir, manifest)
    print(f"saved manifest: {manifest_path}")
    print(f"total chars: {manifest['total_chars']:,}")
    print(f"total tokens: {manifest['total_tokens']:,}")
    print(f"shards: {len(manifest['shards']):,}")


if __name__ == "__main__":
    main()
