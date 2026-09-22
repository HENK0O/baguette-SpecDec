"""Run a JSON-configured parameter sweep and save raw CSV/JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.experiments import run_trials  # noqa: E402
from specdec.loader import assert_tokenizer_compatible, load_baguette  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = ("drafts", "prompts", "generation_lengths", "draft_lengths",
                "temperatures", "top_ps", "top_ks", "seeds")
    missing = [key for key in required if key not in config]
    if missing:
        parser.error("missing config keys: " + ", ".join(missing))
    if args.device == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        name = args.device
    device = torch.device(name)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    target = load_baguette(args.baguette_source, args.target_checkpoint, device)
    if tokenizer.get_vocab_size() != target.cfg.vocab_size:
        raise ValueError("target tokenizer vocabulary size differs from checkpoint")
    encoded = [(prompt, tokenizer.encode(prompt).ids) for prompt in config["prompts"]]
    if any(not ids for _, ids in encoded):
        raise ValueError("every prompt must produce at least one token")

    rows: list[dict] = []
    for draft_spec in config["drafts"]:
        checkpoint = Path(draft_spec["checkpoint"])
        draft_tokenizer = Path(draft_spec["tokenizer"])
        assert_tokenizer_compatible(args.tokenizer, draft_tokenizer)
        draft = load_baguette(args.baguette_source, checkpoint, device)
        if draft.cfg.vocab_size != target.cfg.vocab_size:
            raise ValueError(f"draft {checkpoint} vocabulary size differs from target")
        rows.extend(run_trials(
            target, draft, encoded,
            draft_label=draft_spec.get("label", checkpoint.stem),
            generation_lengths=config["generation_lengths"],
            draft_lengths=config["draft_lengths"],
            temperatures=config["temperatures"],
            top_ps=config["top_ps"],
            top_ks=config["top_ks"],
            seeds=config["seeds"], cache=args.cache,
        ))
        del draft
    if not rows:
        raise ValueError("experiment produced no rows; check the config grid")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"Saved {len(rows)} measured trials to {args.output_csv}")


if __name__ == "__main__":
    main()
