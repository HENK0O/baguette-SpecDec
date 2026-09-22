import unittest
from types import SimpleNamespace

import torch

from specdec.baseline import generate
from specdec.sampling import probabilities, sample_token


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=16)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        logits = torch.zeros(batch, length, 3)
        logits[..., 1] = 2.0
        return logits, None, {}


class SamplingTests(unittest.TestCase):
    def test_probability_normalization_and_filters(self):
        logits = torch.tensor([4.0, 2.0, 1.0, -1.0])
        p = probabilities(logits, top_k=2)
        self.assertAlmostEqual(p.sum().item(), 1.0, places=6)
        self.assertEqual(p[2:].tolist(), [0.0, 0.0])
        p = probabilities(logits, top_p=0.5)
        self.assertEqual(p.tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_temperature_and_seed(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        cold = probabilities(logits, temperature=0.5)
        hot = probabilities(logits, temperature=2.0)
        self.assertGreater(cold[-1].item(), hot[-1].item())
        first = torch.Generator().manual_seed(42)
        second = torch.Generator().manual_seed(42)
        draws_1 = [sample_token(logits, generator=first) for _ in range(20)]
        draws_2 = [sample_token(logits, generator=second) for _ in range(20)]
        self.assertEqual(draws_1, draws_2)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            probabilities(torch.ones(3), temperature=0)
        with self.assertRaises(ValueError):
            probabilities(torch.ones(3), top_p=0)


class BaselineTests(unittest.TestCase):
    def test_greedy_generation_and_metrics(self):
        result = generate(ToyModel(), [0, 2], max_new_tokens=3, greedy=True)
        self.assertEqual(result.token_ids, [1, 1, 1])
        self.assertGreater(result.total_seconds, 0)
        self.assertGreater(result.time_to_first_token, 0)
        self.assertGreater(result.tokens_per_second, 0)

    def test_fixed_seed(self):
        model = ToyModel()
        a = generate(model, [0], max_new_tokens=8, seed=7)
        b = generate(model, [0], max_new_tokens=8, seed=7)
        self.assertEqual(a.token_ids, b.token_ids)

    def test_context_guard(self):
        with self.assertRaisesRegex(ValueError, "exceeds model context"):
            generate(ToyModel(), [0] * 15, max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
