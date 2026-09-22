import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from specdec.loader import assert_tokenizer_compatible
from specdec.speculative import (
    acceptance_probability,
    generate_speculative,
    residual_probabilities,
)


class StaticModel(torch.nn.Module):
    def __init__(self, probs):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=32, vocab_size=len(probs))
        self.logits = torch.tensor(probs).log()

    def forward(self, ids):
        return self.logits.expand(1, ids.shape[1], -1), None, {}


class TransitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=32, vocab_size=3)

    def forward(self, ids):
        logits = torch.full((1, ids.shape[1], 3), -1000.0)
        logits.scatter_(2, ((ids + 1) % 3).unsqueeze(-1), 0.0)
        return logits, None, {}


class SpeculativeTests(unittest.TestCase):
    def test_residual_restores_target_distribution(self):
        p = torch.tensor([0.15, 0.35, 0.50])
        q = torch.tensor([0.65, 0.25, 0.10])
        r = residual_probabilities(p, q)
        accepted_mass = torch.tensor([q[i] * acceptance_probability(p, q, i) for i in range(3)])
        rejected_mass = 1 - accepted_mass.sum()
        actual = accepted_mass + rejected_mass * r
        torch.testing.assert_close(actual, p)
        self.assertAlmostEqual(r.sum().item(), 1.0, places=6)

    def test_empirical_one_token_distribution(self):
        p = [0.15, 0.35, 0.50]
        target, draft = StaticModel(p), StaticModel([0.65, 0.25, 0.10])
        counts = [0, 0, 0]
        for seed in range(1200):
            result = generate_speculative(target, draft, [0], max_new_tokens=1,
                                          draft_length=1, seed=seed)
            counts[result.token_ids[0]] += 1
        frequencies = [count / 1200 for count in counts]
        for observed, expected in zip(frequencies, p):
            self.assertLess(abs(observed - expected), 0.04)

    def test_batched_verification_and_bonus_alignment(self):
        model = TransitionModel()
        result = generate_speculative(model, model, [0], max_new_tokens=5,
                                      draft_length=2, seed=12)
        self.assertEqual(result.token_ids, [1, 2, 0, 1, 2])
        self.assertEqual(result.target_forward_passes, 2)
        self.assertEqual(result.accepted, 4)
        self.assertEqual(result.proposed, 4)

    def test_vocab_and_context_checks(self):
        with self.assertRaisesRegex(ValueError, "vocab sizes"):
            generate_speculative(StaticModel([0.5, 0.5]), StaticModel([1.0]),
                                 [0], max_new_tokens=1, draft_length=1)
        with self.assertRaisesRegex(ValueError, "shared context"):
            generate_speculative(StaticModel([0.5, 0.5]), StaticModel([0.5, 0.5]),
                                 [0] * 32, max_new_tokens=1, draft_length=1)

    def test_tokenizer_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            a.write_text("one")
            b.write_text("one")
            assert_tokenizer_compatible(a, b)
            b.write_text("two")
            with self.assertRaisesRegex(ValueError, "differ"):
                assert_tokenizer_compatible(a, b)


if __name__ == "__main__":
    unittest.main()
