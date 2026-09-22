"""Re-encode Baguette's raw corpus with the target checkpoint's tokenizer.

Do not reuse an existing train.bin based only on matching vocabulary size: the
token IDs can represent entirely different text after tokenizer retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["fineweb", "wiki", "chat"])
    args = parser.parse_args()

    source = args.baguette_source.resolve()
    sys.path.insert(0, str(source))
    from data import encode_pretrain, load_tokenizer

    raw_paths = [source / "data" / "raw" / f"{name}.jsonl" for name in args.sources]
    missing = [str(path) for path in raw_paths if not path.is_file()]
    if missing:
        parser.error("missing raw sources: " + ", ".join(missing))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")

    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = torch.load(args.target_checkpoint, map_location="cpu", weights_only=True)
    vocab_size = checkpoint["model_cfg"]["vocab_size"]
    if tokenizer.get_vocab_size() != vocab_size:
        parser.error("tokenizer vocabulary size differs from target checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_copy = args.output_dir / "tokenizer.json"
    shutil.copyfile(args.tokenizer, tokenizer_copy)
    stats = encode_pretrain(tokenizer, raw_paths, args.output_dir)
    manifest = {
        "tokenizer_sha256": hashlib.sha256(tokenizer_copy.read_bytes()).hexdigest(),
        "target_checkpoint": str(args.target_checkpoint.resolve()),
        "raw_sources": [str(path.resolve()) for path in raw_paths],
        "pretrain": stats,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
