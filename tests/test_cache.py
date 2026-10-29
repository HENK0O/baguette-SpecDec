import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from specdec.baseline import generate
from specdec.cache import allocate, forward_all
from specdec.cached_generation import generate_cached, generate_speculative_cached
from specdec.lookup import generate_lookup
from specdec.speculative import generate_speculative


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get("BAGUETTE_SOURCE", "../LLM")).resolve()
        if not (source / "model.py").exists():
            raise unittest.SkipTest("set BAGUETTE_SOURCE to run Baguette integration tests")
        sys.path.insert(0, str(source))
        from model import ModelConfig, build_model

        torch.manual_seed(3)
        cls.model = build_model(
            ModelConfig(
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_kv_head=1,
                d_model=32,
                head_dim=16,
                d_ff=64,
                max_seq_len=16,
                hybrid=False,
            )
        ).eval()

    def test_prefill_and_block_match_full_forward(self):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        full, _, _ = self.model(tokens)
        caches = allocate(self.model, 16)
        prompt = forward_all(self.model, tokens[:, :3], caches, 0)
        block = forward_all(self.model, tokens[:, 3:], caches, 3)
        torch.testing.assert_close(prompt, full[:, :3], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(block, full[:, 3:], rtol=1e-4, atol=1e-4)

    def test_rejection_rewind_matches_full_forward(self):
        caches = allocate(self.model, 16)
        forward_all(self.model, torch.tensor([[1, 2]]), caches, 0)
        forward_all(self.model, torch.tensor([[3, 4, 5]]), caches, 2)
        # Accept 3, reject 4, overwrite with correction 7, then continue.
        corrected = forward_all(self.model, torch.tensor([[7]]), caches, 3)
        next_logits = forward_all(self.model, torch.tensor([[8]]), caches, 4)
        full, _, _ = self.model(torch.tensor([[1, 2, 3, 7, 8]]))
        torch.testing.assert_close(corrected, full[:, 3:4], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(next_logits, full[:, 4:5], rtol=1e-4, atol=1e-4)

    def test_last_only_preserves_logits_and_cache(self):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        full, _, _ = self.model(tokens)
        caches = allocate(self.model, 6)
        prompt = forward_all(self.model, tokens[:, :3], caches, 0, last_only=True)
        block = forward_all(self.model, tokens[:, 3:5], caches, 3, last_only=True)
        final = forward_all(self.model, tokens[:, 5:], caches, 5, last_only=True)
        self.assertEqual(prompt.shape, (1, 1, 32))
        torch.testing.assert_close(prompt, full[:, 2:3], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(block, full[:, 4:5], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(final, full[:, 5:6], rtol=1e-4, atol=1e-4)
        with self.assertRaisesRegex(ValueError, "allocated cache"):
            forward_all(self.model, tokens[:, :1], caches, 6)

    def test_sampling_survives_deferred_catchup(self):
        draft = deepcopy(self.model)
        draft.lm_head.weight = torch.nn.Parameter(-self.model.lm_head.weight.detach().clone() * 10)
        for k in (1, 2, 4):
            for seed in range(8):
                with self.subTest(k=k, seed=seed):
                    options = dict(
                        max_new_tokens=10,
                        draft_length=k,
                        seed=seed,
                        temperature=0.8,
                        top_k=16,
                        top_p=0.95,
                    )
                    reference = generate_speculative(self.model, draft, [1, 2], **options)
                    cached = generate_speculative_cached(self.model, draft, [1, 2], **options)
                    self.assertEqual(cached.token_ids, reference.token_ids)
                    self.assertEqual(cached.accepted, reference.accepted)
                    self.assertEqual(cached.proposed, reference.proposed)
                    # One prefill + K-1 steps per block + one catch-up between
                    # blocks. The old extra forward per block must not return.
                    self.assertEqual(cached.draft_forward_passes, cached.proposed)

    def test_eos_and_budget_at_cache_boundary(self):
        prompt = [1, 2] * 5
        baseline = generate_cached(self.model, prompt, max_new_tokens=6, greedy=True)
        for eos in (None, baseline.token_ids[0], baseline.token_ids[2]):
            for k in (1, 2, 8):
                with self.subTest(eos=eos, k=k):
                    expected = generate_cached(
                        self.model, prompt, max_new_tokens=6, greedy=True, eos_id=eos
                    )
                    result = generate_speculative_cached(
                        self.model,
                        self.model,
                        prompt,
                        max_new_tokens=6,
                        draft_length=k,
                        greedy=True,
                        eos_id=eos,
                    )
                    self.assertEqual(result.token_ids, expected.token_ids)
                    self.assertEqual(result.draft_forward_passes, result.proposed)

    def test_rejected_draft_eos_does_not_stop_target(self):
        expected = generate_cached(self.model, [1, 2], max_new_tokens=6, greedy=True)
        eos = (expected.token_ids[0] + 1) % self.model.cfg.vocab_size
        draft = deepcopy(self.model)
        draft.lm_head = torch.nn.Linear(32, 32, bias=True)
        with torch.no_grad():
            draft.lm_head.weight.zero_()
            draft.lm_head.bias.fill_(-1000)
            draft.lm_head.bias[eos] = 0
        expected = generate_cached(self.model, [1, 2], max_new_tokens=6, greedy=True, eos_id=eos)
        for greedy in (False, True):
            options = dict(max_new_tokens=6, draft_length=4, greedy=greedy, top_k=1, eos_id=eos)
            actual = generate_speculative_cached(self.model, draft, [1, 2], **options)
            reference = generate_speculative(self.model, draft, [1, 2], **options)
            self.assertEqual(actual.token_ids, expected.token_ids)
            self.assertEqual(actual.token_ids, reference.token_ids)
            self.assertEqual(actual.draft_forward_passes, actual.proposed)

    def test_cached_generation_matches_reference(self):
        prompt = [1, 2]
        reference = generate(self.model, prompt, max_new_tokens=5, seed=7)
        cached = generate_cached(self.model, prompt, max_new_tokens=5, seed=7)
        self.assertEqual(cached.token_ids, reference.token_ids)
        reference_spec = generate_speculative(
            self.model, self.model, prompt, max_new_tokens=5, draft_length=2, seed=7
        )
        cached_spec = generate_speculative_cached(
            self.model, self.model, prompt, max_new_tokens=5, draft_length=2, seed=7
        )
        self.assertEqual(cached_spec.token_ids, reference_spec.token_ids)
        self.assertEqual(cached_spec.target_forward_passes, cached_spec.blocks + 1)

    def test_lookup_matches_cached_target(self):
        for greedy in (False, True):
            for k in (1, 2, 8):
                settings = dict(
                    max_new_tokens=6, greedy=greedy, seed=3, temperature=0.8, top_k=16, top_p=0.95
                )
                expected = generate_cached(self.model, [1, 2] * 5, **settings)
                actual = generate_lookup(self.model, [1, 2] * 5, draft_length=k, **settings)
                self.assertEqual(actual.token_ids, expected.token_ids)
                self.assertEqual(actual.draft_forward_passes, 0)

    def test_lookup_adaptive_guard_with_real_cache(self):
        model = deepcopy(self.model)
        model.lm_head = torch.nn.Linear(32, 32, bias=True)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.fill_(-1000)
            model.lm_head.bias[0] = 0
        for greedy in (False, True):
            settings = dict(max_new_tokens=9, greedy=greedy, seed=7)
            expected = generate_cached(model, [0] * 7, **settings)
            actual = generate_lookup(model, [0] * 7, draft_length=3, **settings)
            self.assertEqual(actual.token_ids, expected.token_ids)
            self.assertEqual(actual.adaptive_blocks, 1)
            self.assertEqual(actual.target_forward_passes, 6)

    def test_cached_rejection_matches_reference(self):
        from model import ModelConfig, build_model

        torch.manual_seed(19)
        draft = build_model(
            ModelConfig(
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_kv_head=1,
                d_model=32,
                head_dim=16,
                d_ff=64,
                max_seq_len=16,
                hybrid=False,
            )
        ).eval()
        draft.lm_head.weight = torch.nn.Parameter(-self.model.lm_head.weight.detach().clone())
        options = dict(max_new_tokens=6, draft_length=2, top_k=1, seed=4)
        reference = generate_speculative(self.model, draft, [1, 2], **options)
        cached = generate_speculative_cached(self.model, draft, [1, 2], **options)
        self.assertEqual(cached.token_ids, reference.token_ids)
        self.assertLess(cached.accepted, cached.proposed)
        self.assertEqual(cached.target_forward_passes, cached.blocks + 1)

    def test_cached_greedy_matches_target(self):
        from model import ModelConfig, build_model

        accepted = generate_speculative_cached(
            self.model,
            self.model,
            [1, 2],
            max_new_tokens=6,
            draft_length=2,
            greedy=True,
        )
        baseline = generate_cached(self.model, [1, 2], max_new_tokens=6, greedy=True)
        self.assertEqual(accepted.token_ids, baseline.token_ids)
        self.assertEqual(accepted.accepted, accepted.proposed)

        torch.manual_seed(19)
        draft = build_model(
            ModelConfig(
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_kv_head=1,
                d_model=32,
                head_dim=16,
                d_ff=64,
                max_seq_len=16,
                hybrid=False,
            )
        ).eval()
        draft.lm_head.weight = torch.nn.Parameter(-self.model.lm_head.weight.detach().clone())
        options = dict(max_new_tokens=6, draft_length=2, greedy=True)
        reference = generate_speculative(self.model, draft, [1, 2], **options)
        cached = generate_speculative_cached(self.model, draft, [1, 2], **options)
        self.assertEqual(cached.token_ids, baseline.token_ids)
        self.assertEqual(reference.token_ids, baseline.token_ids)
        self.assertLess(cached.accepted, cached.proposed)
        self.assertEqual(cached.target_forward_passes, cached.blocks + 1)


if __name__ == "__main__":
    unittest.main()
