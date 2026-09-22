"""Distill a small Baguette draft toward the target's next-token distribution.

The objective is mean tokenwise KL(target || draft), with optional hard-label
cross entropy. This trains the draft to agree with the target, the quantity
that controls speculative acceptance, using only external model/data paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.loader import assert_tokenizer_compatible, load_baguette  # noqa: E402


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def export_draft(model, config, path: Path, step: int) -> None:
    weights = {name: tensor.detach().cpu().half() if tensor.is_floating_point()
               else tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"model": weights, "model_cfg": config.to_dict(), "step": step,
                "stage": "distilled-draft"}, path)


def distillation_loss(teacher_logits, draft_logits, labels, hard_weight: float):
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
    kl = (teacher_probs * (teacher_log_probs - draft_log_probs)).sum(-1).mean()
    hard = F.cross_entropy(draft_logits.float().reshape(-1, draft_logits.shape[-1]),
                           labels.reshape(-1))
    return (1 - hard_weight) * kl + hard_weight * hard, kl.detach(), hard.detach()


@torch.inference_mode()
def evaluate(teacher, draft, corpus, batches: int, batch_size: int, seed: int,
             device: torch.device, hard_weight: float) -> dict:
    draft.eval()
    totals = {"val_loss": 0.0, "val_kl": 0.0, "val_hard_ce": 0.0}
    for index in range(batches):
        x, y, _ = corpus.get_batch(1_000_000 + index, batch_size, seed, str(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
            teacher_logits, _, _ = teacher(x)
            draft_logits, _, _ = draft(x)
        loss, kl, hard = distillation_loss(teacher_logits, draft_logits, y, hard_weight)
        totals["val_loss"] += float(loss)
        totals["val_kl"] += float(kl)
        totals["val_hard_ce"] += float(hard)
    draft.train()
    return {name: value / batches for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--draft-initial-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hard-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 1:
        parser.error("steps, batch size and sequence length must be positive")
    if not 0 <= args.hard_weight <= 1:
        parser.error("hard-weight must be in [0, 1]")
    if args.save_every < 1 or args.log_every < 1 or args.eval_batches < 1:
        parser.error("save-every, log-every and eval-batches must be positive")
    assert_tokenizer_compatible(args.tokenizer, args.data_dir / "tokenizer.json")

    source = args.baguette_source.resolve()
    sys.path.insert(0, str(source))
    from data import BinCorpus, load_tokenizer
    from model import ModelConfig, PRESETS, build_model

    tokenizer = load_tokenizer(args.tokenizer)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    teacher = load_baguette(source, args.target_checkpoint, device)
    teacher.requires_grad_(False)
    if tokenizer.get_vocab_size() != teacher.cfg.vocab_size:
        raise ValueError("target vocabulary does not match tokenizer")
    if args.draft_initial_checkpoint:
        payload = torch.load(args.draft_initial_checkpoint, map_location="cpu", weights_only=True)
        config = ModelConfig.from_dict(payload["model_cfg"])
        draft = build_model(config)
        draft.load_state_dict(payload["model"])
    else:
        config = ModelConfig(**PRESETS["nano"])
        config.vocab_size = tokenizer.get_vocab_size()
        config.eos_id = tokenizer.token_to_id("<|endoftext|>")
        config.bos_id = config.eos_id
        draft = build_model(config)
    if config.vocab_size != teacher.cfg.vocab_size or args.seq_len > config.max_seq_len:
        raise ValueError("draft vocab/context is incompatible with target or sequence length")
    draft = draft.to(device).train()
    corpus = BinCorpus(args.data_dir / "train.bin", args.seq_len)
    validation = BinCorpus(args.data_dir / "val.bin", args.seq_len)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "distill_metrics.jsonl"
    use_autocast = device.type != "cpu"

    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(1, args.steps + 1):
            start = time.perf_counter()
            x, y, _ = corpus.get_batch(step, args.batch_size, args.seed, str(device))
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                 enabled=use_autocast):
                teacher_logits, _, _ = teacher(x)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
                draft_logits, _, _ = draft(x)
            loss, kl, hard = distillation_loss(teacher_logits, draft_logits, y,
                                               args.hard_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            elapsed = time.perf_counter() - start
            record = None
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()),
                          "kl": float(kl), "hard_ce": float(hard),
                          "tokens": step * args.batch_size * args.seq_len,
                          "tokens_per_second": args.batch_size * args.seq_len / elapsed}
            if step % args.save_every == 0 or step == args.steps:
                validation_metrics = evaluate(teacher, draft, validation, args.eval_batches,
                                              args.batch_size, args.seed, device,
                                              args.hard_weight)
                if record is None:
                    record = {"step": step}
                record.update(validation_metrics)
                export_draft(draft, config, args.out_dir / f"draft-step{step}.pt", step)
            if record is not None:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
