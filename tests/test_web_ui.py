import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts.web_ui import ComparisonService
from specdec.baseline import GenerationResult
from specdec.lookup import LookupResult


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
            "draft_length": 2, "cache": False, "strategy": "draft",
        })
        self.assertEqual(result["baseline"]["text"], result["speculative"]["text"])
        self.assertEqual(result["speculative"]["acceptance_rate"], 0)
        self.assertEqual(result["baseline"]["generated_tokens"], 3)
        self.assertGreater(result["speculative"]["speedup"], 0)

    def test_auto_does_not_load_draft_or_claim_fallback_as_speculation(self):
        target = StaticModel([0.1, 0.8, 0.1])
        target.layers = [SimpleNamespace(kind="attn")]
        loader = Mock(side_effect=AssertionError("auto must not load a draft"))
        service = ComparisonService(target, None, TinyTokenizer(), torch.device("cpu"),
                                    draft_loader=loader)
        baseline = GenerationResult([1, 1, 1], 0.1, 0.01, 30)
        lookup = LookupResult([1, 1, 1], 0.1, 0.01, 30, 0, 0, 0, 3, 0, 0.1, 0,
                              verification_blocks=0, fallback_tokens=3, cooldown_tokens=0)
        with patch("scripts.web_ui.generate_cached", return_value=baseline), \
             patch("scripts.web_ui.generate_lookup", return_value=lookup):
            result = service.compare({"prompt": "Bonjour", "max_new_tokens": 3})
        loader.assert_not_called()
        self.assertEqual(result["execution"], "target")
        self.assertTrue(result["same_tokens"])
        self.assertEqual(result["speculative"]["draft_forward_passes"], 0)

    def test_draft_is_loaded_only_when_selected(self):
        loader = Mock(return_value=self.service.draft)
        service = ComparisonService(self.service.target, None, TinyTokenizer(),
                                    torch.device("cpu"), draft_loader=loader)
        self.assertTrue(service.info()["draft_available"])
        self.assertIsNone(service.info()["draft_parameters"])
        service.compare({"prompt": "Bonjour", "strategy": "draft", "cache": False,
                         "max_new_tokens": 3, "mode": "greedy"})
        loader.assert_called_once()

    def test_rejects_invalid_input(self):
        for payload in (
            {"prompt": ""},
            {"prompt": "Bonjour", "max_new_tokens": 257},
            {"prompt": "Bonjour", "temperature": float("nan")},
            {"prompt": "Bonjour", "cache": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.compare({"strategy": "draft", **payload})


if __name__ == "__main__":
    unittest.main()
