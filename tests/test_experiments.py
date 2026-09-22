import unittest
from types import SimpleNamespace

import torch

from specdec.experiments import run_trials


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_len=16, vocab_size=2, eos_id=None)

    def forward(self, ids):
        return torch.zeros(1, ids.shape[1], 2), None, {}


class ExperimentTests(unittest.TestCase):
    def test_grid_produces_raw_rows(self):
        model = TinyModel()
        rows = run_trials(
            model, model, [("A", [0]), ("B", [1])], draft_label="tiny",
            generation_lengths=[2], draft_lengths=[1, 2],
            temperatures=[0.5], top_ps=[1.0], top_ks=[0], seeds=[1, 2],
            cache=False,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["draft_length"] for row in rows}, {1, 2})
        self.assertTrue(all(row["baseline_tokens_per_second"] > 0 for row in rows))
        self.assertTrue(all(row["target_forward_passes"] > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
