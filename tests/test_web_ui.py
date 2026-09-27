import unittest
from types import SimpleNamespace

import torch

from scripts.web_ui import ComparisonService


class StaticModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=16, vocab_size=len(probabilities), eos_id=None)
        self.layers = [SimpleNamespace(kind="other")]
        self.logits = torch.tensor(probabilities).log()

    def forward(self, ids):
        return self.logits.expand(1, ids.shape[1], -1), None, {}


class TinyTokenizer:
    def encode(self, prompt):
        return SimpleNamespace(ids=[0] if prompt else [])

    def decode(self, ids, *, skip_special_tokens=False):
        return " ".join(map(str, ids))


class WebUITests(unittest.TestCase):
    def setUp(self):
        target = StaticModel([0.1, 0.8, 0.1])
        draft = StaticModel([0.8, 0.1, 0.1])
        self.service = ComparisonService(target, draft, TinyTokenizer(), torch.device("cpu"))

    def test_greedy_comparison_corrects_draft(self):
        result = self.service.compare({
            "prompt": "Bonjour", "mode": "greedy", "max_new_tokens": 3,
            "draft_length": 2, "cache": False,
        })
        self.assertEqual(result["baseline"]["text"], result["speculative"]["text"])
        self.assertEqual(result["speculative"]["acceptance_rate"], 0)
        self.assertEqual(result["baseline"]["generated_tokens"], 3)
        self.assertGreater(result["speculative"]["speedup"], 0)

    def test_rejects_invalid_input(self):
        for payload in (
            {"prompt": ""},
            {"prompt": "Bonjour", "max_new_tokens": 257},
            {"prompt": "Bonjour", "temperature": float("nan")},
            {"prompt": "Bonjour", "cache": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.compare(payload)


if __name__ == "__main__":
    unittest.main()
