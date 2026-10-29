"""Paired benchmark of target, prompt lookup, and optional trained draft on MPS/CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specdec.cached_generation import generate_cached, generate_speculative_cached
from specdec.loader import assert_tokenizer_compatible, load_baguette
from specdec.lookup import generate_lookup

WORKLOADS = [
    ("greeting", "libre", "Bonjour, je suis"),
    ("rain", "libre", "Explique simplement pourquoi la pluie tombe."),
    ("python", "libre", "Comment organiser un petit projet Python ?"),
    (
        "copy-demo",
        "copie",
        "Recopie exactement ce texte :\nLe vélo permet de se déplacer en ville sans bruit. Il demande peu de place et facilite les trajets quotidiens.\nTexte recopié :\nLe vélo permet",
    ),
    (
        "copy-library",
        "copie",
        "Recopie exactement ce texte :\nLa bibliothèque ouvre à neuf heures. Les lecteurs peuvent emprunter trois livres pour deux semaines. Le retour des livres se fait à l'accueil.\nTexte recopié :\nLa bibliothèque ouvre",
    ),
    ("repeated-code", "repetitif", "def addition(a, b):\n    return a + b\n\n" * 6),
    (
        "repeated-csv",
        "repetitif",
        "Complète ce tableau en gardant le même format :\n"
        + "ville,pays\nParis,France\nRome,Italie\nMadrid,Espagne\n" * 4
        + "ville,pays\nParis,",
    ),
]


def kernel_digest() -> str:
    content = b"".join(
        (ROOT / "specdec" / f).read_bytes()
        for f in ("lookup.py", "cache.py", "cached_generation.py", "sampling.py")
    )
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--include-draft", action="store_true")
    parser.add_argument("--draft-checkpoint", type=Path)
    parser.add_argument("--draft-tokenizer", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.include_draft and (args.draft_checkpoint is None or args.draft_tokenizer is None):
        parser.error("--include-draft requires both draft paths")
    digest = kernel_digest()
    device = torch.device(args.device)
    target = load_baguette(args.baguette_source, args.target_checkpoint, device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != target.cfg.vocab_size:
        raise ValueError("target vocabulary differs from tokenizer")
    draft = None
    if args.include_draft:
        assert_tokenizer_compatible(args.tokenizer, args.draft_tokenizer)
        draft = load_baguette(args.baguette_source, args.draft_checkpoint, device)
        if draft.cfg.vocab_size != target.cfg.vocab_size:
            raise ValueError("draft vocabulary differs from target")

    rows = []
    for label, category, prompt in WORKLOADS:
        ids = tokenizer.encode(prompt).ids
        for greedy in (False, True):
            settings = dict(
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                temperature=0.5,
                top_k=20,
                top_p=0.95,
                eos_id=target.cfg.eos_id,
            )
            runners = {
                "baseline": lambda options: generate_cached(target, ids, **options),
                "auto": lambda options: generate_lookup(target, ids, draft_length=8, **options),
            }
            if draft is not None:
                runners["draft"] = lambda options: generate_speculative_cached(
                    target, draft, ids, draft_length=2, **options
                )
            for run in runners.values():
                run({**settings, "seed": 0})
            for repetition, seed in enumerate(args.seeds):
                names = list(runners)
                offset = len(rows) % len(names)
                order = names[offset:] + names[:offset]
                if len(rows) % 2:
                    order.reverse()
                values = {name: runners[name]({**settings, "seed": seed}) for name in order}
                row = dict(
                    label=label,
                    category=category,
                    prompt=prompt,
                    greedy=greedy,
                    seed=seed,
                    repetition=repetition,
                    order=order,
                    prompt_tokens=len(ids),
                    same_tokens=values["auto"].token_ids == values["baseline"].token_ids,
                    auto_speedup=values["auto"].tokens_per_second
                    / values["baseline"].tokens_per_second,
                )
                row.update({name: asdict(result) for name, result in values.items()})
                if "draft" in values:
                    row["draft_speedup"] = (
                        values["draft"].tokens_per_second / values["baseline"].tokens_per_second
                    )
                rows.append(row)
                print(
                    f"{len(rows)} {label} {'greedy' if greedy else 'sampling'} seed={seed}: "
                    f"auto={row['auto_speedup']:.3f}x, matches={row['same_tokens']}",
                    flush=True,
                )

    summary = []
    for label, category, _ in WORKLOADS:
        for greedy in (False, True):
            trials = [r for r in rows if r["label"] == label and r["greedy"] == greedy]
            item = dict(
                label=label,
                category=category,
                greedy=greedy,
                trials=len(trials),
                baseline_tps=median(r["baseline"]["tokens_per_second"] for r in trials),
                auto_tps=median(r["auto"]["tokens_per_second"] for r in trials),
                median_speedup=median(r["auto_speedup"] for r in trials),
                min_speedup=min(r["auto_speedup"] for r in trials),
                max_speedup=max(r["auto_speedup"] for r in trials),
                all_tokens_match=all(r["same_tokens"] for r in trials),
            )
            if draft is not None:
                item["draft_median_speedup"] = median(r["draft_speedup"] for r in trials)
            summary.append(item)
    if digest != kernel_digest():
        raise RuntimeError("engine changed during benchmark; discard these measurements")
    report = dict(
        device=args.device,
        torch=str(torch.__version__),
        kernel_sha256=digest,
        target_checkpoint=args.target_checkpoint.name,
        draft_checkpoint=args.draft_checkpoint.name if draft is not None else None,
        target_parameters=sum(p.numel() for p in target.parameters()),
        max_new_tokens=args.max_new_tokens,
        auto_block_length=8,
        draft_block_length=2,
        temperature=0.5,
        top_k=20,
        top_p=0.95,
        full_warmups_per_method=1,
        summary=summary,
        trials=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if any(r["greedy"] and not r["same_tokens"] for r in rows):
        raise RuntimeError("greedy outputs differ; inspect saved trials")


if __name__ == "__main__":
    main()
