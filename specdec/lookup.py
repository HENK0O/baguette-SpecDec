"""Exact prompt-lookup decoding with no assistant model.

Candidates are deterministic copies of prior text. Sampling a target token and
checking whether it equals a candidate is equivalent to accepting a point-mass
draft with probability p(candidate), then using the conditional residual on a
mismatch. Each committed token therefore follows the target distribution.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass

import torch

from specdec.baseline import synchronize
from specdec.cache import allocate, forward_all
from specdec.sampling import sample_token, seeded_generator
from specdec.speculative import SpeculativeResult


class PromptLookup:
    """Index the most recent continuation of each suffix, including new output."""

    def __init__(self, tokens: list[int], min_match: int = 2, max_match: int = 4):
        if not 1 <= min_match <= max_match:
            raise ValueError("matching lengths must satisfy 1 <= min <= max")
        self.min_match = min_match
        self.max_match = max_match
        self.tokens: list[int] = []
        self.positions: dict[tuple[int, ...], list[int]] = {}
        for token in tokens:
            self.append(token)

    def append(self, token: int) -> None:
        next_position = len(self.tokens)
        self.tokens.append(token)
        for n in range(self.min_match, min(self.max_match, next_position) + 1):
            key = tuple(self.tokens[next_position - n : next_position])
            self.positions.setdefault(key, []).append(next_position)

    def propose(self, count: int) -> list[int]:
        for n in range(min(self.max_match, len(self.tokens)), self.min_match - 1, -1):
            positions = self.positions.get(tuple(self.tokens[-n:]))
            if positions:
                # Prefer a recent occurrence with a whole block available.
                # The latest occurrence alone often has only one token after it.
                index = bisect_right(positions, len(self.tokens) - count)
                position = positions[max(0, index - 1)]
                return self.tokens[position : position + count]
        return []


@dataclass
class LookupResult(SpeculativeResult):
    verification_blocks: int
    fallback_tokens: int
    cooldown_tokens: int
    adaptive_blocks: int = 0
    discovered_reuse: bool = False


@torch.inference_mode()
def generate_lookup(
    target: torch.nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    draft_length: int = 8,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    eos_id: int | None = None,
) -> LookupResult:
    """Verify copied blocks with a guard adapted to recent complete matches.

    Start with two ordinary target tokens. After two full accepted blocks,
    keep only the first-token guard, saving a model call per useful block.
    Every proposed token is still checked against a target draw.
    """
    if not prompt_ids or max_new_tokens < 1 or draft_length < 1:
        raise ValueError("prompt, max_new_tokens and draft_length must be positive")
    if len(prompt_ids) + max_new_tokens > target.cfg.max_seq_len:
        raise ValueError("prompt + output exceeds model context")
    target.eval()
    device = next(target.parameters()).device
    generator = seeded_generator(device, seed)
    caches = allocate(target, len(prompt_ids) + max_new_tokens)
    target_passes = 0
    target_seconds = 0.0
    generated: list[int] = []
    proposed = accepted = attempts = verified = fallback = cooldown_tokens = 0
    cooldown = poor_blocks = successful_blocks = adaptive_blocks = 0
    discovered_reuse = False
    first_token_at = 0.0
    synchronize(device)
    start = time.perf_counter()
    lookup = PromptLookup(prompt_ids)
    # Short prompts start on target alone. Probe again after 32 generated
    # tokens, then every 16 tokens, so new repetitions can become useful.
    lookup_enabled = len(prompt_ids) >= 32 or len(lookup.propose(3)) >= 3

    def forward(tokens: list[int], position: int, last_only: bool) -> torch.Tensor:
        nonlocal target_passes, target_seconds
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        logits = forward_all(target, ids, caches, position, last_only=last_only)
        synchronize(device)
        target_seconds += time.perf_counter() - t0
        target_passes += 1
        return logits

    def sample(logits: torch.Tensor) -> int:
        return sample_token(
            logits,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

    def emit(token: int) -> bool:
        nonlocal first_token_at
        if not generated:
            synchronize(device)
            first_token_at = time.perf_counter() - start
        generated.append(token)
        if lookup_enabled:
            lookup.append(token)
        return len(generated) == max_new_tokens or (eos_id is not None and token == eos_id)

    next_logits = forward(prompt_ids, 0, True)[0, -1]
    while len(generated) < max_new_tokens:
        position = len(prompt_ids) + len(generated)
        remaining = max_new_tokens - len(generated)
        if not lookup_enabled and len(generated) >= 32 and len(generated) % 16 == 0:
            probe = PromptLookup(prompt_ids + generated)
            if len(probe.propose(3)) >= 3:
                lookup = probe
                lookup_enabled = discovered_reuse = True
        candidates = (
            [] if cooldown or not lookup_enabled else lookup.propose(min(draft_length, remaining))
        )
        if eos_id is not None and eos_id in candidates:
            candidates = candidates[: candidates.index(eos_id) + 1]
        token = sample(next_logits)
        can_verify = len(candidates) >= 2
        if can_verify:
            attempts += 1
            proposed += len(candidates)
        if not can_verify or token != candidates[0]:
            successful_blocks = 0
            fallback += 1
            if cooldown:
                cooldown_tokens += 1
                cooldown -= 1
            if emit(token):
                break
            next_logits = forward([token], position, True)[0, -1]
            continue

        accepted += 1
        accepted_in_block = 1
        if emit(token):
            fallback += 1
            break
        # A reliable run needs only the first-token guard. A mismatch restores
        # the two-token guard immediately; neither mode skips target sampling.
        guard = 1 if successful_blocks >= 2 and len(candidates) >= 3 else 2
        if guard == 2:
            next_logits = forward([token], position, True)[0, -1]
            token = sample(next_logits)
            second_matches = token == candidates[1]
            if second_matches:
                accepted += 1
                accepted_in_block += 1
            if emit(token):
                fallback += 2
                break
            if not second_matches or len(candidates) == 2:
                successful_blocks = 0
                fallback += 2
                next_logits = forward([token], position + 1, True)[0, -1]
                continue
        else:
            adaptive_blocks += 1
        verified += 1
        # The final candidate need not enter the cache at the output boundary.
        end = -1 if len(candidates) == remaining else len(candidates)
        verification_ids = candidates[guard - 1 : end]
        logits = forward(verification_ids, position + guard - 1, False)[0]
        rejected = stopped = False
        for i, candidate in enumerate(candidates[guard:], start=guard):
            token = sample(logits[i - guard])
            matches = token == candidate
            if matches:
                accepted += 1
                accepted_in_block += 1
            stopped = emit(token)
            if stopped or not matches:
                rejected = not matches
                break
        if stopped:
            break
        if rejected:
            successful_blocks = 0
            next_logits = forward([token], position + accepted_in_block, True)[0, -1]
            if accepted_in_block < max(3, len(candidates) // 2):
                poor_blocks += 1
                cooldown = min(64, 8 << min(poor_blocks - 1, 3))
            else:
                poor_blocks = 0
        else:
            successful_blocks += 1
            poor_blocks = 0
            next_logits = logits[-1]

    synchronize(device)
    elapsed = time.perf_counter() - start
    return LookupResult(
        token_ids=generated,
        total_seconds=elapsed,
        time_to_first_token=first_token_at,
        tokens_per_second=len(generated) / elapsed,
        proposed=proposed,
        accepted=accepted,
        blocks=attempts,
        target_forward_passes=target_passes,
        draft_forward_passes=0,
        target_seconds=target_seconds,
        draft_seconds=0.0,
        verification_blocks=verified,
        fallback_tokens=fallback,
        cooldown_tokens=cooldown_tokens,
        adaptive_blocks=adaptive_blocks,
        discovered_reuse=discovered_reuse,
    )
