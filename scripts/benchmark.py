"""Compare the simple target baseline with exact speculative decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.baseline import generate  # noqa: E402
from specdec.cached_generation import generate_cached, generate_speculative_cached  # noqa: E402
from specdec.loader import assert_tokenizer_compatible, load_baguette  # noqa: E402
from specdec.speculative import generate_speculative  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="target tokenizer JSON")
    parser.add_argument("--draft-tokenizer", type=Path, required=True, help="tokenizer used to train the draft")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true", help="use attention KV caches for both models")
    parser.add_argument("--output", type=Path, help="write measured results as JSON")
    args = parser.parse_args()
    if any(k < 1 for k in args.draft_lengths):
        parser.error("draft lengths must be positive")

    assert_tokenizer_compatible(args.tokenizer, args.draft_tokenizer)
    if args.device == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        name = args.device
    device = torch.device(name)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    target = load_baguette(args.baguette_source, args.target_checkpoint, device)
    draft = load_baguette(args.baguette_source, args.draft_checkpoint, device)
    vocab_size = tokenizer.get_vocab_size()
    if target.cfg.vocab_size != vocab_size or draft.cfg.vocab_size != vocab_size:
        raise ValueError("checkpoint vocab size differs from tokenizer")
    ids = tokenizer.encode(args.prompt).ids
    if not ids:
        raise ValueError("prompt produced no tokens")

    settings = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p, seed=args.seed,
                    eos_id=target.cfg.eos_id)
    baseline_fn = generate_cached if args.cache else generate
    speculative_fn = generate_speculative_cached if args.cache else generate_speculative
    baseline = baseline_fn(target, ids, **settings)
    results = {
        "prompt": args.prompt,
        "device": name,
        "seed": args.seed,
        "cache": args.cache,
        "target_checkpoint": str(args.target_checkpoint),
        "draft_checkpoint": str(args.draft_checkpoint),
        "baseline": {
            "generated_tokens": len(baseline.token_ids),
            "tokens_per_second": baseline.tokens_per_second,
            "time_to_first_token": baseline.time_to_first_token,
            "total_seconds": baseline.total_seconds,
        },
        "speculative": [],
    }
    for K in args.draft_lengths:
        result = speculative_fn(target, draft, ids, draft_length=K, **settings)
        results["speculative"].append({
            "draft_length": K,
            "generated_tokens": len(result.token_ids),
            "tokens_per_second": result.tokens_per_second,
            "speedup": result.tokens_per_second / baseline.tokens_per_second,
            "time_to_first_token": result.time_to_first_token,
            "total_seconds": result.total_seconds,
            "acceptance_rate": result.acceptance_rate,
            "accepted_per_block": result.accepted_per_block,
            "target_forward_passes": result.target_forward_passes,
            "draft_forward_passes": result.draft_forward_passes,
            "target_seconds": result.target_seconds,
            "draft_seconds": result.draft_seconds,
        })
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
