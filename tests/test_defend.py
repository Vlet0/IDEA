"""Unit tests for the defender-side analysis package (defend/)."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from defend.detection import (activation_clustering_audit, feature_space_report,
                              linear_probe_separability)
from defend.model_repair import fine_pruning_curve, fine_tuning_curve
from models.backbone import build_backbone


class DetectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.m = 128
        n = 500
        z = torch.randn(n, self.m)
        self.mask = torch.zeros(n, dtype=torch.bool)
        self.mask[:50] = True
        z[:50, :3] += 4.0  # strong poison signature in a 3-D subspace
        self.z = z

    def test_separable_poison_is_detected(self):
        ac = activation_clustering_audit(self.z, self.mask)
        self.assertGreater(ac["auroc"], 0.9)
        self.assertGreater(ac["recall"], 50.0)

    def test_linear_probe_detects_and_is_calibrated(self):
        sep = linear_probe_separability(torch.randn(200, self.m), torch.randn(200, self.m) + 3.0)
        self.assertGreater(sep["auroc"], 0.95)
        null = linear_probe_separability(torch.randn(200, self.m), torch.randn(200, self.m))
        self.assertLess(abs(null["auroc"] - 0.5), 0.1)  # unseparable features ≈ chance

    def test_report_has_every_block(self):
        rep = feature_space_report(self.z, self.mask, torch.randn(100, self.m),
                                   torch.randn(100, self.m) + 2.0)
        for key in ("spectral_auroc_c1", "actclust_auroc", "linprobe_auroc"):
            self.assertIn(key, rep)


class ModelRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = build_backbone("hpeli", m=128, num_joints=17, subcarrier_num=114).eval()
        X = torch.rand(48, 3, 114, 10)
        Y = torch.randn(48, 17, 3) * 0.1
        self.loader = DataLoader(TensorDataset(X, Y), batch_size=16, shuffle=True)
        self.X = X

    def _measure(self, model):
        with torch.no_grad():
            p, _ = model(self.X[:8])
        return {"binary_fall_asr": float(p.abs().mean()) * 100, "clean_mpjpe_cm": float(p.std())}

    def test_fine_pruning_prunes_expected_channel_counts_and_restores(self):
        rows = fine_pruning_curve(self.net, self.loader, self._measure, torch.device("cpu"),
                                  fractions=(0.0, 0.25, 0.5), finetune_epochs=0)
        self.assertEqual([r["n_pruned"] for r in rows], [0, 32, 64])
        # Hook removed → the model forwards normally afterwards.
        with torch.no_grad():
            pose, _ = self.net(self.X[:4])
        self.assertTrue(torch.isfinite(pose).all())

    def test_fine_tuning_does_not_mutate_the_original_model(self):
        before = [p.detach().clone() for p in self.net.parameters()]
        rows = fine_tuning_curve(self.net, self.loader, self._measure, torch.device("cpu"),
                                 epochs=2, lr=1e-3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.net.parameters())))
        self.assertEqual(rows[0]["epoch"], 0)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
