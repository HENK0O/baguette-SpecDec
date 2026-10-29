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
    caches = allocate(model, len(prompt_ids) + max_new_tokens)
    synchronize(device)
    start = time.perf_counter()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = forward_all(model, prompt, caches, 0, last_only=True)[0, -1, :]
    generated: list[int] = []
    first_token_at = 0.0

    for step in range(max_new_tokens):
        token = sample_token(
            logits,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
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
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    eos_id: int | None = None,
) -> SpeculativeResult:
    """Exact decoding with deferred draft catch-up and one target pass per block.

    Stage times cover model forwards; sampling and acceptance are included in
    total_seconds. Cache capacity is limited to this request's token budget.
    """
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
    cache_length = len(prompt_ids) + max_new_tokens
    target_caches = allocate(target, cache_length)
    draft_caches = allocate(draft, cache_length)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    proposed = accepted = blocks = 0
    target_passes = draft_passes = 0
    target_seconds = draft_seconds = 0.0
    first_token_at = 0.0
    pending_target_token: int | None = None
    synchronize(device)
    start = time.perf_counter()

    t0 = time.perf_counter()
    target_next = forward_all(target, prompt, target_caches, 0, last_only=True)[0, -1, :]
    synchronize(device)
    target_seconds += time.perf_counter() - t0
    target_passes += 1
    t0 = time.perf_counter()
    draft_next = forward_all(draft, prompt, draft_caches, 0, last_only=True)[0, -1, :]
    synchronize(device)
    draft_seconds += time.perf_counter() - t0
    draft_passes += 1

    while len(generated) < max_new_tokens:
        position = len(prompt_ids) + len(generated)
        K = min(draft_length, max_new_tokens - len(generated))
        proposals: list[int] = []
        q_vectors: list[torch.Tensor] = []
        for i in range(K):
            if greedy:
                token = int(torch.argmax(draft_next).item())
            else:
                q = probabilities(draft_next, temperature, top_k, top_p)
                token = int(torch.multinomial(q, 1, generator=generator).item())
                q_vectors.append(q)
            proposals.append(token)
            # No further proposal needs the logits after the final token.
            # If accepted, it will enter the cache with the bonus token below.
            if i + 1 == K or (eos_id is not None and token == eos_id):
                break
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            draft_next = forward_all(draft, token_tensor, draft_caches, position + i)[0, -1, :]
            synchronize(device)
            draft_seconds += time.perf_counter() - t0
            draft_passes += 1

        K = len(proposals)
        proposed += K
        blocks += 1
        if pending_target_token is None:
            verification_ids = proposals
            verification_pos = position
        else:
            # Verify the token emitted at the previous block boundary together
            # with the new proposals. This needs only one target pass per block.
            verification_ids = [pending_target_token, *proposals]
            verification_pos = position - 1
        proposal_tensor = torch.tensor([verification_ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        verification_logits = forward_all(target, proposal_tensor, target_caches, verification_pos)
        if pending_target_token is None:
            p_vectors = [target_next]
            p_vectors += [verification_logits[0, i, :] for i in range(K)]
        else:
            p_vectors = [verification_logits[0, i, :] for i in range(K + 1)]
        synchronize(device)
        target_seconds += time.perf_counter() - t0
        target_passes += 1

        output_block: list[int] = []
        accepted_this_block = 0
        rejected = False
        for i, token in enumerate(proposals):
            # Later distributions are unused as soon as a proposal is rejected.
            p = p_vectors[i] if greedy else probabilities(p_vectors[i], temperature, top_k, top_p)
            if greedy:
                correction = int(torch.argmax(p).item())
                accept = token == correction
            else:
                accept = float(
                    torch.rand((), generator=generator, device=device).item()
                ) < acceptance_probability(p, q_vectors[i], token)
            if accept:
                accepted += 1
                accepted_this_block += 1
                output_block.append(token)
                if eos_id is not None and token == eos_id:
                    break
            else:
                if not greedy:
                    correction = int(
                        torch.multinomial(
                            residual_probabilities(p, q_vectors[i]), 1, generator=generator
                        ).item()
                    )
                output_block.append(correction)
                rejected = True
                break

        if not rejected and len(output_block) == K and len(generated) + K < max_new_tokens:
            if eos_id is None or output_block[-1] != eos_id:
                bonus = sample_token(
                    p_vectors[K],
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )
                output_block.append(bonus)

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

        # On full acceptance, catch up the unprocessed final proposal and the
        # bonus in a single pass. On rejection, overwrite the rejected position
        # with its correction. The target defers that boundary token as before.
        last_token = generated[-1]
        pending_target_token = last_token
        if rejected:
            catchup_ids = [last_token]
            next_position = position + accepted_this_block
        else:
            catchup_ids = [proposals[-1], last_token]
            next_position = position + K - 1
        catchup_tensor = torch.tensor([catchup_ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        draft_next = forward_all(
            draft, catchup_tensor, draft_caches, next_position, last_only=True
        )[0, -1, :]
        synchronize(device)
        draft_seconds += time.perf_counter() - t0
        draft_passes += 1

    synchronize(device)
    elapsed = time.perf_counter() - start
    return SpeculativeResult(
        generated,
        elapsed,
        first_token_at,
        len(generated) / elapsed,
        proposed,
        accepted,
        blocks,
        target_passes,
        draft_passes,
        target_seconds,
        draft_seconds,
    )
