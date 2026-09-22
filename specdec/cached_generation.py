"""Cached generation for attention-only Baguette target and draft models."""

from __future__ import annotations

import time

import torch

from specdec.baseline import GenerationResult, synchronize
from specdec.cache import allocate, forward_all
from specdec.sampling import probabilities, sample_token, seeded_generator
from specdec.speculative import (
    SpeculativeResult,
    acceptance_probability,
    residual_probabilities,
)


@torch.inference_mode()
def generate_cached(
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
) -> GenerationResult:
    if not prompt_ids or max_new_tokens < 1:
        raise ValueError("prompt must be nonempty and max_new_tokens must be >= 1")
    if len(prompt_ids) + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("prompt + output exceeds model context")
    model.eval()
    device = next(model.parameters()).device
    generator = seeded_generator(device, seed)
    caches = allocate(model, model.cfg.max_seq_len)
    synchronize(device)
    start = time.perf_counter()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = forward_all(model, prompt, caches, 0)[0, -1, :]
    generated: list[int] = []
    first_token_at = 0.0

    for step in range(max_new_tokens):
        token = sample_token(logits, greedy=greedy, temperature=temperature,
                             top_k=top_k, top_p=top_p, generator=generator)
        synchronize(device)
        if not generated:
            first_token_at = time.perf_counter() - start
        generated.append(token)
        if (eos_id is not None and token == eos_id) or step + 1 == max_new_tokens:
            break
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        logits = forward_all(model, token_tensor, caches, len(prompt_ids) + step)[0, -1, :]

    synchronize(device)
    elapsed = time.perf_counter() - start
    return GenerationResult(generated, elapsed, first_token_at, len(generated) / elapsed)


@torch.inference_mode()
def generate_speculative_cached(
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
) -> SpeculativeResult:
    if not prompt_ids or max_new_tokens < 1 or draft_length < 1:
        raise ValueError("prompt, max_new_tokens and draft_length must be positive")
    context_limit = min(target.cfg.max_seq_len, draft.cfg.max_seq_len)
    if len(prompt_ids) + max_new_tokens > context_limit:
        raise ValueError("prompt + output exceeds shared context")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft vocab sizes differ")
    device = next(target.parameters()).device
    if device != next(draft.parameters()).device:
        raise ValueError("target and draft must be on the same device")
    target.eval()
    draft.eval()
    generator = seeded_generator(device, seed)
    target_caches = allocate(target, context_limit)
    draft_caches = allocate(draft, context_limit)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    proposed = accepted = blocks = 0
    target_passes = draft_passes = 0
    target_seconds = draft_seconds = 0.0
    first_token_at = 0.0
    synchronize(device)
    start = time.perf_counter()

    t0 = time.perf_counter()
    target_next = forward_all(target, prompt, target_caches, 0)[0, -1, :]
    synchronize(device)
    target_seconds += time.perf_counter() - t0
    target_passes += 1
    t0 = time.perf_counter()
    draft_next = forward_all(draft, prompt, draft_caches, 0)[0, -1, :]
    synchronize(device)
    draft_seconds += time.perf_counter() - t0
    draft_passes += 1

    while len(generated) < max_new_tokens:
        position = len(prompt_ids) + len(generated)
        K = min(draft_length, max_new_tokens - len(generated))
        proposals: list[int] = []
        q_vectors: list[torch.Tensor] = []
        for i in range(K):
            q = probabilities(draft_next, temperature, top_k, top_p)
            token = int(torch.multinomial(q, 1, generator=generator).item())
            proposals.append(token)
            q_vectors.append(q)
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            draft_next = forward_all(draft, token_tensor, draft_caches, position + i)[0, -1, :]
            synchronize(device)
            draft_seconds += time.perf_counter() - t0
            draft_passes += 1
            if eos_id is not None and token == eos_id:
                break

        K = len(proposals)
        proposed += K
        blocks += 1
        proposal_tensor = torch.tensor([proposals], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        verification_logits = forward_all(target, proposal_tensor, target_caches, position)
        p_vectors = [probabilities(target_next, temperature, top_k, top_p)]
        p_vectors += [probabilities(verification_logits[0, i, :], temperature, top_k, top_p)
                      for i in range(K)]
        synchronize(device)
        target_seconds += time.perf_counter() - t0
        target_passes += 1

        output_block: list[int] = []
        accepted_this_block = 0
        rejected = False
        for i, token in enumerate(proposals):
            p, q = p_vectors[i], q_vectors[i]
            if float(torch.rand((), generator=generator, device=device).item()) < acceptance_probability(p, q, token):
                accepted += 1
                accepted_this_block += 1
                output_block.append(token)
                if eos_id is not None and token == eos_id:
                    break
            else:
                correction = int(torch.multinomial(residual_probabilities(p, q), 1, generator=generator).item())
                output_block.append(correction)
                rejected = True
                break

        if not rejected and len(output_block) == K and len(generated) + K < max_new_tokens:
            if eos_id is None or output_block[-1] != eos_id:
                output_block.append(int(torch.multinomial(p_vectors[K], 1, generator=generator).item()))

        if not generated:
            synchronize(device)
            first_token_at = time.perf_counter() - start
        stopped = False
        for token in output_block:
            generated.append(token)
            if eos_id is not None and token == eos_id:
                stopped = True
                break
        if stopped or len(generated) == max_new_tokens:
            break

        # The one newly sampled correction/bonus has not yet entered either
        # cache. Reuse accepted proposal states, then overwrite stale suffixes.
        last_token = generated[-1]
        last_tensor = torch.tensor([[last_token]], dtype=torch.long, device=device)
        next_position = position + accepted_this_block
        t0 = time.perf_counter()
        target_next = forward_all(target, last_tensor, target_caches, next_position)[0, -1, :]
        synchronize(device)
        target_seconds += time.perf_counter() - t0
        target_passes += 1
        t0 = time.perf_counter()
        draft_next = forward_all(draft, last_tensor, draft_caches, next_position)[0, -1, :]
        synchronize(device)
        draft_seconds += time.perf_counter() - t0
        draft_passes += 1

    synchronize(device)
    elapsed = time.perf_counter() - start
    return SpeculativeResult(
        generated, elapsed, first_token_at, len(generated) / elapsed,
        proposed, accepted, blocks, target_passes, draft_passes,
        target_seconds, draft_seconds,
    )
