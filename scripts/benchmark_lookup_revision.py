"""Compare a local historical lookup revision against the current engine and target."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from benchmark_lookup import WORKLOADS, kernel_digest

from specdec.cached_generation import generate_cached
from specdec.loader import load_baguette
from specdec.lookup import generate_lookup


def historical_lookup(ref):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:specdec/lookup.py"], cwd=ROOT)
    module = types.ModuleType("_historical_lookup")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{commit}:specdec/lookup.py", "exec"), module.__dict__)
    return module.generate_lookup, commit, hashlib.sha256(source).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--previous-ref", default="6f8cc74")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous, commit, previous_sha = historical_lookup(args.previous_ref)
    digest = kernel_digest()
    target = load_baguette(args.baguette_source, args.target_checkpoint, torch.device(args.device))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    workloads = [(label, category, prompt, 64) for label, category, prompt in WORKLOADS]
    workloads += [
        (label + "-long", category, prompt, 128) for label, category, prompt in WORKLOADS[:3]
    ]
    trials = []
    for label, category, prompt, budget in workloads:
        ids = tokenizer.encode(prompt).ids
        for greedy in (False, True):
            settings = dict(
                max_new_tokens=budget,
                greedy=greedy,
                temperature=0.5,
                top_k=20,
                top_p=0.95,
                eos_id=target.cfg.eos_id,
            )
            runners = {
                "baseline": lambda opts: generate_cached(target, ids, **opts),
                "previous": lambda opts: previous(target, ids, draft_length=8, **opts),
                "adaptive": lambda opts: generate_lookup(target, ids, draft_length=8, **opts),
            }
            for runner in runners.values():
                runner({**settings, "seed": 0})
            for repetition, seed in enumerate(args.seeds):
                names = list(runners)
                offset = repetition % len(names)
                order = names[offset:] + names[:offset]
                if greedy:
                    order.reverse()
                values = {name: runners[name]({**settings, "seed": seed}) for name in order}
                row = dict(
                    label=label,
                    category=category,
                    prompt=prompt,
                    budget=budget,
                    greedy=greedy,
                    seed=seed,
                    order=order,
                    same_tokens=all(
                        v.token_ids == values["baseline"].token_ids for v in values.values()
                    ),
                    speedup=values["adaptive"].tokens_per_second
                    / values["baseline"].tokens_per_second,
                    revision_speedup=values["adaptive"].tokens_per_second
                    / values["previous"].tokens_per_second,
                )
                row.update({name: asdict(result) for name, result in values.items()})
                trials.append(row)
                print(
                    f"{len(trials)} {label} {'greedy' if greedy else 'sampling'} seed={seed}: "
                    f"target={row['speedup']:.3f}x previous={row['revision_speedup']:.3f}x "
                    f"passes={values['previous'].target_forward_passes}->{values['adaptive'].target_forward_passes} "
                    f"matches={row['same_tokens']}",
                    flush=True,
                )
    summary = []
    for label, category, _, budget in workloads:
        for greedy in (False, True):
            rows = [r for r in trials if r["label"] == label and r["greedy"] == greedy]
            summary.append(
                dict(
                    label=label,
                    category=category,
                    greedy=greedy,
                    budget=budget,
                    median_speedup=median(r["speedup"] for r in rows),
                    median_revision_speedup=median(r["revision_speedup"] for r in rows),
                    previous_passes=[r["previous"]["target_forward_passes"] for r in rows],
                    adaptive_passes=[r["adaptive"]["target_forward_passes"] for r in rows],
                    all_tokens_match=all(r["same_tokens"] for r in rows),
                )
            )
    if digest != kernel_digest():
        raise RuntimeError("engine changed during benchmark")
    report = dict(
        device=args.device,
        torch=str(torch.__version__),
        kernel_sha256=digest,
        previous_commit=commit,
        previous_lookup_sha256=previous_sha,
        target_checkpoint=args.target_checkpoint.name,
        block_length=8,
        temperature=0.5,
        top_k=20,
        top_p=0.95,
        full_warmups_per_method=1,
        summary=summary,
        trials=trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all(r["same_tokens"] for r in trials):
        raise RuntimeError("token mismatch; inspect saved trials")


if __name__ == "__main__":
    main()
