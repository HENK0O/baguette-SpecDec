import unittest

import torch

from scripts.distill_draft import distillation_loss


class DistillationTests(unittest.TestCase):
    def test_matching_logits_have_zero_kl(self):
        logits = torch.tensor([[[1.0, 2.0, -1.0]]])
        labels = torch.tensor([[1]])
        loss, kl, hard = distillation_loss(logits, logits, labels, 0.0)
        self.assertAlmostEqual(kl.item(), 0.0, places=6)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertGreater(hard.item(), 0)

    def test_kl_penalizes_wrong_draft(self):
        teacher = torch.tensor([[[8.0, -8.0]]])
        good = teacher.clone()
        bad = -teacher
        labels = torch.tensor([[0]])
        good_loss, _, _ = distillation_loss(teacher, good, labels, 0.1)
        bad_loss, _, _ = distillation_loss(teacher, bad, labels, 0.1)
        self.assertLess(good_loss.item(), bad_loss.item())
