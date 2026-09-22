"""Run the phase-1 baseline with an external Baguette checkpoint and tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Allow `python scripts/baseline.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.baseline import generate  # noqa: E402
from specdec.cached_generation import generate_cached  # noqa: E402
from specdec.loader import load_baguette  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true", help="use the attention KV cache")
    args = parser.parse_args()

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_baguette(args.baguette_source, args.target_checkpoint, device)
    if tokenizer.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint model_cfg")

    prompt_ids = tokenizer.encode(args.prompt).ids
    generate_fn = generate_cached if args.cache else generate
    result = generate_fn(
        model, prompt_ids, max_new_tokens=args.max_new_tokens,
        greedy=args.greedy, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, eos_id=model.cfg.eos_id,
    )
    print(json.dumps({
        "prompt": args.prompt,
        "generated_text": tokenizer.decode(result.token_ids, skip_special_tokens=False),
        "generated_tokens": len(result.token_ids),
        "total_seconds": result.total_seconds,
        "time_to_first_token": result.time_to_first_token,
        "tokens_per_second": result.tokens_per_second,
        "device": device_name,
        "seed": args.seed,
        "cache": args.cache,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
