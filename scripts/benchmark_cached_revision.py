"""Compare cached kernels with a trusted local Git revision, in alternating order.

This loads cache.py and cached_generation.py from Git into isolated modules.
Sampling and model loading use the current shared helpers, so comparisons
are intended for revisions that share those helpers and result types.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specdec import cached_generation as current
from specdec.loader import assert_tokenizer_compatible, load_baguette


def reference_engine(revision: str):
    def read(path: str) -> str:
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)

    cache = types.ModuleType("_reference_cache")
    sys.modules[cache.__name__] = cache
    exec(compile(read("specdec/cache.py"), f"{revision}/cache.py", "exec"), cache.__dict__)
    engine = types.ModuleType("_reference_cached_generation")
    sys.modules[engine.__name__] = engine
    source = read("specdec/cached_generation.py").replace(
        "from specdec.cache import", "from _reference_cache import"
    )
    exec(compile(source, f"{revision}/cached_generation.py", "exec"), engine.__dict__)
    return engine


def summarize(rows: list[dict], k: int) -> dict:
    trials = [r for r in rows if r["draft_length"] == k]
    return {
        "draft_length": k,
        "trials": len(trials),
        "old_baseline_tokens_per_second": median(
            r["old_baseline"]["tokens_per_second"] for r in trials
        ),
        "new_baseline_tokens_per_second": median(
            r["new_baseline"]["tokens_per_second"] for r in trials
        ),
        "old_spec_tokens_per_second": median(r["old_spec"]["tokens_per_second"] for r in trials),
        "new_spec_tokens_per_second": median(r["new_spec"]["tokens_per_second"] for r in trials),
        "paired_spec_improvement": median(
            r["new_spec"]["tokens_per_second"] / r["old_spec"]["tokens_per_second"] for r in trials
        ),
        "paired_baseline_improvement": median(
            r["new_baseline"]["tokens_per_second"] / r["old_baseline"]["tokens_per_second"]
            for r in trials
        ),
        "old_speedup_vs_baseline": median(
            r["old_spec"]["tokens_per_second"] / r["old_baseline"]["tokens_per_second"]
            for r in trials
        ),
        "new_speedup_vs_baseline": median(
            r["new_spec"]["tokens_per_second"] / r["new_baseline"]["tokens_per_second"]
            for r in trials
        ),
        "paired_draft_forward_reduction": median(
            1 - r["new_spec"]["draft_forward_passes"] / r["old_spec"]["draft_forward_passes"]
            for r in trials
        ),
        "matching_spec_outputs": sum(r["spec_tokens_match"] for r in trials),
        "matching_baseline_outputs": sum(r["baseline_tokens_match"] for r in trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="c683099")
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--draft-tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or any(k < 1 for k in args.draft_lengths):
        parser.error("token counts and draft lengths must be positive")

    assert_tokenizer_compatible(args.tokenizer, args.draft_tokenizer)
    old = reference_engine(args.reference)
    device = torch.device(args.device)
    target = load_baguette(args.baguette_source, args.target_checkpoint, device)
    draft = load_baguette(args.baguette_source, args.draft_checkpoint, device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if (
        target.cfg.vocab_size != tokenizer.get_vocab_size()
        or draft.cfg.vocab_size != tokenizer.get_vocab_size()
    ):
        raise ValueError("checkpoint vocabulary differs from tokenizer")
    prompts = [
        "Bonjour, je suis",
        "Explique en français pourquoi le ciel est bleu.",
        "La programmation en Python permet de",
    ]
    rows = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        for k in args.draft_lengths:
            settings = dict(
                max_new_tokens=args.max_new_tokens,
                greedy=args.greedy,
                temperature=0.5,
                top_k=20,
                top_p=0.95,
                eos_id=target.cfg.eos_id,
            )
            runners = {
                "old_baseline": lambda options: old.generate_cached(target, ids, **options),
                "new_baseline": lambda options: current.generate_cached(target, ids, **options),
                "old_spec": lambda options: old.generate_speculative_cached(
                    target, draft, ids, draft_length=k, **options
                ),
                "new_spec": lambda options: current.generate_speculative_cached(
                    target, draft, ids, draft_length=k, **options
                ),
            }
            for run in runners.values():
                run({**settings, "seed": 0})
            for seed in args.seeds:
                # Rotate and reverse the order to spread warm-up/thermal effects.
                names = list(runners)
                offset = len(rows) % len(names)
                order = names[offset:] + names[:offset]
                if len(rows) % 2:
                    order.reverse()
                result = {name: runners[name]({**settings, "seed": seed}) for name in order}
                row = dict(
                    prompt=prompt, prompt_tokens=len(ids), draft_length=k, seed=seed, order=order
                )
                row.update({name: asdict(value) for name, value in result.items()})
                row["baseline_tokens_match"] = (
                    result["old_baseline"].token_ids == result["new_baseline"].token_ids
                )
                row["spec_tokens_match"] = (
                    result["old_spec"].token_ids == result["new_spec"].token_ids
                )
                row["greedy_matches_target"] = (
                    result["new_baseline"].token_ids == result["new_spec"].token_ids
                )
                rows.append(row)
                ratio = result["new_spec"].tokens_per_second / result["old_spec"].tokens_per_second
                print(f"{len(rows)}: K={k}, seed={seed}, new/old={ratio:.3f}x", flush=True)

    kernels = b"".join(
        (ROOT / "specdec" / name).read_bytes() for name in ("cache.py", "cached_generation.py")
    )
    output = {
        "reference": subprocess.check_output(
            ["git", "rev-parse", args.reference], cwd=ROOT, text=True
        ).strip(),
        "kernel_sha256": hashlib.sha256(kernels).hexdigest(),
        "device": args.device,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "target_checkpoint": args.target_checkpoint.name,
        "draft_checkpoint": args.draft_checkpoint.name,
        "target_parameters": sum(p.numel() for p in target.parameters()),
        "draft_parameters": sum(p.numel() for p in draft.parameters()),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
        "warmup_runs_per_prompt_and_k": 1,
        "temperature": 0.5,
        "top_k": 20,
        "top_p": 0.95,
        "summary": [summarize(rows, k) for k in args.draft_lengths],
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)
    if args.greedy and not all(
        r["greedy_matches_target"] and r["spec_tokens_match"] and r["baseline_tokens_match"]
        for r in rows
    ):
        raise RuntimeError("greedy outputs differ; inspect saved trials")


if __name__ == "__main__":
    main()
