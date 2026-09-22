"""Exact stochastic speculative sampling with batched target verification.

The draft proposes x_1,...,x_K from q_i. A single target pass gives p_i for
each proposal and p_(K+1) for the bonus token. For x_i, accept with probability
min(1, p_i(x_i)/q_i(x_i)). If rejected, draw from (p_i-q_i)_+ normalized.
The accepted prefix plus that correction has exactly the target distribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from specdec.baseline import synchronize
from specdec.sampling import probabilities, seeded_generator


def acceptance_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> float:
    """p and q are normalized vectors with shape [vocab]."""
    proposed_probability = float(q[token].item())
    if proposed_probability <= 0:
        raise ValueError("draft proposed a zero-probability token")
    return min(1.0, float(p[token].item()) / proposed_probability)


def residual_probabilities(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The correction distribution r(x) ∝ max(0, p(x)-q(x))."""
    residual = torch.clamp(p - q, min=0)
    total = residual.sum()
    if total <= 0:
        raise ValueError("residual is empty: rejection is impossible for p == q")
    return residual / total


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    total_seconds: float
    time_to_first_token: float
    tokens_per_second: float
    proposed: int
    accepted: int
    blocks: int
    target_forward_passes: int
    draft_forward_passes: int
    target_seconds: float
    draft_seconds: float

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def accepted_per_block(self) -> float:
        return self.accepted / self.blocks if self.blocks else 0.0


@torch.inference_mode()
def generate_speculative(
    target: torch.nn.Module,
    draft: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    draft_length: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    eos_id: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SpeculativeResult:
    """Batch=1 reference implementation; both models recompute the prefix.

    `target_logits[0, T-1+i, :]` predicts proposal i for prefix length T.
    `target_logits[0, T+K-1, :]` predicts the bonus after all K proposals.
    """
    if not prompt_ids or max_new_tokens < 1 or draft_length < 1:
        raise ValueError("prompt, max_new_tokens and draft_length must be positive")
    context_limit = min(target.cfg.max_seq_len, draft.cfg.max_seq_len)
    if len(prompt_ids) + max_new_tokens > context_limit:
        raise ValueError(f"prompt + output exceeds shared context ({context_limit} tokens)")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft vocab sizes differ")
    target_device = next(target.parameters()).device
    draft_device = next(draft.parameters()).device
    if target_device != draft_device:
        raise ValueError("target and draft must be on the same device")

    target.eval()
    draft.eval()
    generator = seeded_generator(target_device, seed)
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    generated: list[int] = []
    proposed = accepted = blocks = target_passes = draft_passes = 0
    target_seconds = draft_seconds = 0.0
    first_token_at = 0.0
    synchronize(target_device)
    start = clock()

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        K = min(draft_length, remaining)
        prefix_length = sequence.shape[1]
        draft_sequence = sequence
        proposals: list[int] = []
        q_vectors: list[torch.Tensor] = []

        for _ in range(K):
            synchronize(target_device)
            t0 = clock()
            draft_logits, _, _ = draft(draft_sequence)
            q = probabilities(draft_logits[0, -1, :], temperature, top_k, top_p)
            token = int(torch.multinomial(q, 1, generator=generator).item())
            synchronize(target_device)
            draft_seconds += clock() - t0
            draft_passes += 1
            proposals.append(token)
            q_vectors.append(q)
            draft_sequence = torch.cat(
                (draft_sequence, torch.tensor([[token]], device=target_device)), dim=1
            )
            if eos_id is not None and token == eos_id:
                break

        K = len(proposals)
        proposed += K
        blocks += 1
        synchronize(target_device)
        t0 = clock()
        target_logits, _, _ = target(draft_sequence)
        p_vectors = [
            probabilities(target_logits[0, prefix_length - 1 + i, :], temperature, top_k, top_p)
            for i in range(K + 1)
        ]
        synchronize(target_device)
        target_seconds += clock() - t0
        target_passes += 1

        output_block: list[int] = []
        rejected = False
        for i, token in enumerate(proposals):
            p, q = p_vectors[i], q_vectors[i]
            if float(torch.rand((), generator=generator, device=target_device).item()) < acceptance_probability(p, q, token):
                accepted += 1
                output_block.append(token)
                if eos_id is not None and token == eos_id:
                    break
            else:
                correction = int(torch.multinomial(residual_probabilities(p, q), 1, generator=generator).item())
                output_block.append(correction)
                rejected = True
                break

        # When all proposals survive, draw one extra token from p_(K+1).
        if not rejected and len(output_block) == K and len(generated) + K < max_new_tokens:
            if eos_id is None or output_block[-1] != eos_id:
                bonus = int(torch.multinomial(p_vectors[K], 1, generator=generator).item())
                output_block.append(bonus)

        if not generated:
            synchronize(target_device)
            first_token_at = clock() - start
        for token in output_block:
            generated.append(token)
            sequence = torch.cat((sequence, torch.tensor([[token]], device=target_device)), dim=1)
            if eos_id is not None and token == eos_id:
                break
        if eos_id is not None and generated[-1] == eos_id:
            break

    synchronize(target_device)
    elapsed = clock() - start
    return SpeculativeResult(
        generated, elapsed, first_token_at, len(generated) / elapsed,
        proposed, accepted, blocks, target_passes, draft_passes,
        target_seconds, draft_seconds,
    )
