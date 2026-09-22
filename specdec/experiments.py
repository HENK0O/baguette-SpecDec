"""Run a small, explicit parameter grid and return one row per measured trial."""

from __future__ import annotations

from itertools import product

import torch

from specdec.baseline import generate
from specdec.cached_generation import generate_cached, generate_speculative_cached
from specdec.speculative import generate_speculative


def run_trials(
    target: torch.nn.Module,
    draft: torch.nn.Module,
    encoded_prompts: list[tuple[str, list[int]]],
    *,
    draft_label: str,
    generation_lengths: list[int],
    draft_lengths: list[int],
    temperatures: list[float],
    top_ps: list[float],
    top_ks: list[int],
    seeds: list[int],
    cache: bool,
) -> list[dict]:
    """Measure baseline once and each K once for every other combination."""
    if not all((encoded_prompts, generation_lengths, draft_lengths,
                temperatures, top_ps, top_ks, seeds)):
        raise ValueError("all experiment grid lists must be nonempty")
    if any(n < 1 for n in generation_lengths + draft_lengths):
        raise ValueError("generation and draft lengths must be positive")
    baseline_fn = generate_cached if cache else generate
    spec_fn = generate_speculative_cached if cache else generate_speculative
    target_params = sum(p.numel() for p in target.parameters())
    draft_params = sum(p.numel() for p in draft.parameters())
    rows: list[dict] = []
    for (prompt, ids), n, temperature, top_p, top_k, seed in product(
        encoded_prompts, generation_lengths, temperatures, top_ps, top_ks, seeds
    ):
        settings = dict(max_new_tokens=n, temperature=temperature, top_p=top_p,
                        top_k=top_k, seed=seed, eos_id=target.cfg.eos_id)
        baseline = baseline_fn(target, ids, **settings)
        for K in draft_lengths:
            result = spec_fn(target, draft, ids, draft_length=K, **settings)
            rows.append({
                "draft_label": draft_label,
                "target_parameters": target_params,
                "draft_parameters": draft_params,
                "cache": cache,
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "max_new_tokens": n,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "draft_length": K,
                "baseline_generated_tokens": len(baseline.token_ids),
                "baseline_tokens_per_second": baseline.tokens_per_second,
                "baseline_time_to_first_token": baseline.time_to_first_token,
                "spec_generated_tokens": len(result.token_ids),
                "spec_tokens_per_second": result.tokens_per_second,
                "spec_time_to_first_token": result.time_to_first_token,
                "speedup": result.tokens_per_second / baseline.tokens_per_second,
                "acceptance_rate": result.acceptance_rate,
                "accepted_per_block": result.accepted_per_block,
                "blocks": result.blocks,
                "target_forward_passes": result.target_forward_passes,
                "draft_forward_passes": result.draft_forward_passes,
                "target_seconds": result.target_seconds,
                "draft_seconds": result.draft_seconds,
            })
    return rows
