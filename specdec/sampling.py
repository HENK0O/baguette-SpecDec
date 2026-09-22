"""Sampling shared by baseline and, later, speculative decoding.

The order matters: divide logits by temperature, restrict to top-k and top-p,
then normalize. Both draft and target must use the same policy in SpecDec.
"""

from __future__ import annotations

import torch


def probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Return a normalized distribution for one vocabulary vector [vocab]."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive for stochastic sampling")
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("top_k must be >= 0 and top_p must be in (0, 1]")

    scores = logits.float() / temperature
    if top_k:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1:
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(0, sorted_ids, sorted_scores)
    return torch.softmax(scores, dim=-1)


def sample_token(
    logits: torch.Tensor,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    if greedy:
        if logits.ndim != 1:
            raise ValueError("logits must have shape [vocab]")
        return int(torch.argmax(logits).item())
    probs = probabilities(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probs, 1, generator=generator).item())
