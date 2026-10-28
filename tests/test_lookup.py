import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specdec.baseline import generate
from specdec.lookup import PromptLookup, generate_lookup


class TransitionModel(torch.nn.Module):
    def __init__(self, soft=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=128, vocab_size=3)
        self.soft = soft

    def forward(self, ids):
        next_ids = ((ids.cumsum(-1) + 1) % 3) if self.soft else ((ids + 1) % 3)
        logits = torch.full((*ids.shape, 3), -1.5 if self.soft else -1000.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        return logits, None, {}


def fake_forward(model, ids, cache, pos, *, last_only=False):
    # Model the cache's absolute positions, including overwrites after rejection.
    assert pos <= len(cache["tokens"])
    assert pos + ids.shape[1] <= cache["capacity"]
    cache["tokens"][pos:] = ids[0].tolist()
    logits, _, _ = model(torch.tensor([cache["tokens"]]))
    return logits[:, -1:] if last_only else logits[:, -ids.shape[1]:]


class LookupTests(unittest.TestCase):
    def run_lookup(self, model, prompt, **settings):
        with patch("specdec.lookup.allocate", side_effect=lambda m, n: {"tokens": [], "capacity": n}), \
             patch("specdec.lookup.forward_all", side_effect=fake_forward):
            return generate_lookup(model, prompt, **settings)

    def test_index_has_real_continuations_and_prefers_complete_blocks(self):
        index = PromptLookup([1, 2, 3, 4, 1, 2])
        self.assertEqual(index.propose(3), [3, 4, 1])
        repeated = PromptLookup([0, 1, 2] * 10)
        self.assertEqual(repeated.propose(8), [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(PromptLookup([0, 1, 2]).propose(8), [])
        index.append(3)
        self.assertEqual(index.propose(2), [4, 1])

    def test_reuse_saves_target_passes(self):
        model = TransitionModel()
        prompt = [0, 1, 2] * 8
        result = self.run_lookup(model, prompt, max_new_tokens=24,
                                 draft_length=8, greedy=True)
        expected = generate(model, prompt, max_new_tokens=24, greedy=True)
        self.assertEqual(result.token_ids, expected.token_ids)
        self.assertLessEqual(result.target_forward_passes, 8)
        self.assertGreater(result.verification_blocks, 0)
        self.assertEqual(result.draft_forward_passes, 0)

    def test_short_unstructured_prompt_stays_on_target(self):
        model = TransitionModel()
        result = self.run_lookup(model, [1, 0, 2], max_new_tokens=24, greedy=True)
        self.assertEqual(result.verification_blocks, 0)
        self.assertEqual(result.proposed, 0)
        self.assertEqual(result.draft_forward_passes, 0)
        self.assertEqual(result.target_forward_passes, 24)

    def test_sampling_matches_baseline_through_acceptance_and_rejection(self):
        model = TransitionModel(soft=True)
        verified = 0
        for prompt in ([0, 1, 2] * 5, [1, 0, 2]):
            for k in (1, 2, 8):
                for seed in range(8):
                    with self.subTest(prompt=prompt, k=k, seed=seed):
                        settings = dict(max_new_tokens=16, seed=seed, temperature=0.8,
                                        top_k=3, top_p=0.95)
                        result = self.run_lookup(model, prompt, draft_length=k, **settings)
                        expected = generate(model, prompt, **settings)
                        self.assertEqual(result.token_ids, expected.token_ids)
                        self.assertLessEqual(result.target_forward_passes, len(result.token_ids))
                        verified += result.verification_blocks
        self.assertGreater(verified, 0)

    def test_unhelpful_blocks_trigger_cooldown(self):
        model = TransitionModel()
        with patch.object(PromptLookup, "propose", return_value=[0, 1, 0, 0, 0, 0, 0, 0]):
            result = self.run_lookup(model, [2], max_new_tokens=24, greedy=True)
        self.assertEqual(result.token_ids, generate(model, [2], max_new_tokens=24, greedy=True).token_ids)
        self.assertGreater(result.cooldown_tokens, 0)

    def test_eos_and_exact_budget(self):
        model = TransitionModel()
        for n in (1, 2, 3, 16):
            for eos in (None, 0, 1):
                settings = dict(max_new_tokens=n, greedy=True, eos_id=eos)
                expected = generate(model, [0, 1, 2] * 8, **settings)
                result = self.run_lookup(model, [0, 1, 2] * 8, **settings)
                self.assertEqual(result.token_ids, expected.token_ids)

    def test_context_and_empty_prompt(self):
        for prompt, n in (([], 1), ([0] * 128, 1), ([0], 0)):
            with self.assertRaises(ValueError):
                generate_lookup(TransitionModel(), prompt, max_new_tokens=n)


if __name__ == "__main__":
    unittest.main()
