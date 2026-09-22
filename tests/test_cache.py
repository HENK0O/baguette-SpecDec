import os
import sys
import unittest
from pathlib import Path

import torch

from specdec.cache import allocate, forward_all
from specdec.baseline import generate
from specdec.cached_generation import generate_cached, generate_speculative_cached
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
        cls.model = build_model(ModelConfig(
            vocab_size=32, n_layer=2, n_head=2, n_kv_head=1,
            d_model=32, head_dim=16, d_ff=64, max_seq_len=16,
            hybrid=False,
        )).eval()

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

    def test_cached_generation_matches_reference(self):
        prompt = [1, 2]
        reference = generate(self.model, prompt, max_new_tokens=5, seed=7)
        cached = generate_cached(self.model, prompt, max_new_tokens=5, seed=7)
        self.assertEqual(cached.token_ids, reference.token_ids)
        reference_spec = generate_speculative(self.model, self.model, prompt,
                                              max_new_tokens=5, draft_length=2, seed=7)
        cached_spec = generate_speculative_cached(self.model, self.model, prompt,
                                                 max_new_tokens=5, draft_length=2, seed=7)
        self.assertEqual(cached_spec.token_ids, reference_spec.token_ids)
        self.assertEqual(cached_spec.target_forward_passes, cached_spec.blocks + 1)

    def test_cached_rejection_matches_reference(self):
        from model import ModelConfig, build_model

        torch.manual_seed(19)
        draft = build_model(ModelConfig(
            vocab_size=32, n_layer=2, n_head=2, n_kv_head=1,
            d_model=32, head_dim=16, d_ff=64, max_seq_len=16,
            hybrid=False,
        )).eval()
        draft.lm_head.weight = torch.nn.Parameter(-self.model.lm_head.weight.detach().clone())
        options = dict(max_new_tokens=6, draft_length=2, top_k=1, seed=4)
        reference = generate_speculative(self.model, draft, [1, 2], **options)
        cached = generate_speculative_cached(self.model, draft, [1, 2], **options)
        self.assertEqual(cached.token_ids, reference.token_ids)
        self.assertLess(cached.accepted, cached.proposed)
        self.assertEqual(cached.target_forward_passes, cached.blocks + 1)


if __name__ == "__main__":
    unittest.main()
