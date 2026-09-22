"""Deliberately simple no-cache target generation for a correctness baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from specdec.sampling import sample_token


@dataclass
class GenerationResult:
    token_ids: list[int]
    total_seconds: float
    time_to_first_token: float
    tokens_per_second: float


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous device work before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    eos_id: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerationResult:
    """Recompute the full prefix each step; caching belongs to a later phase.

    Baguette's forward returns logits with shape [batch, sequence, vocabulary].
    We use batch=1 and draw one token from logits[0, -1, :].
    """
    if not prompt_ids or max_new_tokens < 1:
        raise ValueError("prompt must be nonempty and max_new_tokens must be >= 1")
    max_seq_len = model.cfg.max_seq_len
    if len(prompt_ids) + max_new_tokens > max_seq_len:
        raise ValueError(f"prompt + output exceeds model context ({max_seq_len} tokens)")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device.type).manual_seed(seed)
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output: list[int] = []
    model.eval()
    synchronize(device)
    start = clock()
    first_token_at = 0.0

    for _ in range(max_new_tokens):
        logits, _, _ = model(sequence)
        token = sample_token(
            logits[0, -1, :], greedy=greedy, temperature=temperature,
            top_k=top_k, top_p=top_p, generator=generator,
        )
        synchronize(device)
        if not output:
            first_token_at = clock() - start
        output.append(token)
        if eos_id is not None and token == eos_id:
            break
        sequence = torch.cat((sequence, torch.tensor([[token]], device=device)), dim=1)

    synchronize(device)
    elapsed = clock() - start
    return GenerationResult(output, elapsed, first_token_at, len(output) / elapsed)
